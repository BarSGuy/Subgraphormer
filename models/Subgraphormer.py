import torch
import torch.nn as nn
from models import layers
import logging
import torch_scatter as pys
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class Subgraphormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # preprocess
        self.max_dis = cfg.data.preprocess.max_dis
        self.num_eigen_vectors = cfg.model.PE.num_eigen_vectors

        # atom encoder
        self.atom_dim = cfg.model.atom_encoder.in_dim
        self.use_linear_atom_encoder = cfg.model.atom_encoder.linear

        # edge encoder
        self.use_linear_edge_encoder = cfg.model.edge_encoder.linear
        self.use_edge_attr = cfg.model.edge_encoder.use_edge_attr
        self.edge_attr_dim = cfg.model.edge_encoder.in_dim

        # model
        self.model_name = cfg.model.model_name
        self.num_layers = cfg.model.num_layer
        self.aggs = cfg.model.aggs
        self.H = cfg.model.H
        self.final_dim = cfg.model.final_dim
        self.dropout = cfg.model.dropout
        self.use_residual = cfg.model.residual
        self.use_linear = cfg.model.layer_encoder.linear
        self.attention_type = cfg.model.attention_type

        # pooling
        self.use_sum_pooling = cfg.model.sum_pooling

        # general
        self.dataset = cfg.data.name

        logging.info("Checking dimentions.")
        self.post_concat_dim_embed, self.each_agg_dim_embed = Subgraphormer.compute_final_embedding_dimension(
            dim_embed=cfg.model.dim_embed, num_aggs=len(self.aggs), H=self.H)

        logging.info("Initializing Atom encoder + NM + PE layer.")
        self.atom_encoder, self.PE_layer = self.get_preprocess_layer(
            dim=self.post_concat_dim_embed, max_dis=self.max_dis, use_linear=self.use_linear_atom_encoder, atom_dim=self.atom_dim, num_eigen_vectors=self.num_eigen_vectors)

        logging.info(f"Initializing all {self.num_layers} layers")
        # MPNN (local/global)
        self.MPNNs = nn.ModuleList()
        self.EDGE_ENCODER = nn.ModuleList()
        # POINT (point)
        self.EPSs = nn.ParameterList()
        self.POINT_ENCODERs = nn.ModuleList()
        # LAYER AGG (point || local || global)
        self.CAT_ENCODERs = nn.ModuleList()
        self.BNORM_RELUs = nn.ModuleList()
        # DROPOUT
        self.DROP_OUTs = nn.ModuleList()

        for layer_idx in range(self.num_layers):
            logging.info(f"Initializing layer number {layer_idx}.")
            # MPNN (local/global)
            MPNN_i = {}
            EDGE_ENCODER_i = {}

            # POINT (point)
            EPS_i = nn.ParameterDict({})
            POINT_ENCODER_i = {}
            for agg in self.aggs:
                logging.info(f"Initializing aggregation {agg}.")
                if self.is_point_agg(agg):  # GIN
                    eps_i = self.init_eps()
                    EPS_i[agg] = eps_i
                    point_encoder_i = self.init_point_encoder(
                        use_linear=self.use_linear, in_dim=self.post_concat_dim_embed, out_dim=self.each_agg_dim_embed)
                    POINT_ENCODER_i[agg] = point_encoder_i
                else:  # MPNN (local/global)
                    if self.use_edge_encoder(agg=agg):
                        edge_out_dim = self.get_edge_encoder_out_dim(
                            layer_idx=layer_idx)
                        edge_encoder_i = self.init_edge_encoder(
                            use_linear=self.use_linear_edge_encoder, in_dim=self.edge_attr_dim, out_dim=edge_out_dim)
                        EDGE_ENCODER_i[agg] = edge_encoder_i
                    else:
                        edge_out_dim = None
                        EDGE_ENCODER_i[agg] = None
                    mpnn_i = self.init_mpnn(in_dim=self.post_concat_dim_embed, out_dim=self.each_agg_dim_embed,
                                            H=self.H, edge_in_dim=edge_out_dim, type=self.attention_type)
                    MPNN_i[agg] = mpnn_i
            # MPNN (local/global)
            self.MPNNs.append(nn.ModuleDict(MPNN_i))
            self.EDGE_ENCODER.append(nn.ModuleDict(EDGE_ENCODER_i))
            # POINT (point)
            self.EPSs.append(EPS_i)
            self.POINT_ENCODERs.append(nn.ModuleDict(POINT_ENCODER_i))
            # LAYER AGG (point || local || global)
            cat_encoder_i = self.get_cat_encoder(
                use_linear=self.use_linear, in_dim=self.post_concat_dim_embed, out_dim=self.post_concat_dim_embed)
            self.CAT_ENCODERs.append(cat_encoder_i)
            self.BNORM_RELUs.append(layers.NormReLU(self.post_concat_dim_embed))
            if self.dropout > 0:
                self.DROP_OUTs.append(nn.Dropout(p=self.dropout))

        logging.info(f"Initializing pooling")
        self.POOLING = layers.Pooling(
            self.post_concat_dim_embed, self.final_dim)

    def forward(self, batch):
        # ATOM ENCODER + NM
        batch = self.atom_encoder(batch)
        # PE
        if self.PE_layer != None:
            batch = self.PE_layer(batch)
        # LAYERS
        for layer_idx in range(self.num_layers):
            # AGGS
            all_aggs = []
            for agg in self.aggs:
                # GIN
                if self.is_point_agg(agg=agg):
                    agg_element = self.gin_encoder(
                        batch=batch, eps=self.EPSs[layer_idx][agg], agg=agg, encoder=self.POINT_ENCODERs[layer_idx][agg])
                # MPNN
                else:
                    agg_element = self.agg_encoder(
                        batch=batch, agg=agg, edge_encoder=self.EDGE_ENCODER[layer_idx][agg], mpnn=self.MPNNs[layer_idx][agg])
                all_aggs.append(agg_element)
            # CAT ALL AGGS
            all_aggs_cat = torch.cat(all_aggs, dim=1)
            # BN_RELU + MLP
            batch_x = self.BNORM_RELUs[layer_idx](
                self.CAT_ENCODERs[layer_idx](all_aggs_cat))
            # DROPOUT
            if self.dropout > 0:
                batch_x = self.DROP_OUTs[layer_idx](batch_x)
            # RESIDUAL
            if self.use_residual:
                batch.x = batch_x + batch.x
            else:
                batch.x = batch_x
        # POOL
        pool_value = self.pooling_forward(
            batch=batch, use_sum_pooling=self.use_sum_pooling)
        return pool_value

    # ============================= forward - helpers ============================= #
    def pooling_forward(self, batch, use_sum_pooling):
        if use_sum_pooling:
            batch.x = self.aggregate(graph=batch, agg="uG", encode=None)
            global_pool = self.POOLING(batch)
            return global_pool
        else:
            subgraph_rep = self.aggregate(
                graph=batch, agg="uG", encode=None, pool_efficiently=True)
            global_pool_efficient = self.POOLING(
                batch=batch, subgraph_rep=subgraph_rep, efficient=True)
            return global_pool_efficient

    def get_edge_attr(self, agg, batch, edge_encoder):
        if not self.use_edge_encoder(agg=agg):
            edge_attr = None
            return edge_attr
        if "L" in agg:
            edge_attr = batch.get(f"attrs_{agg}", None)
            if edge_attr != None:
                edge_attr = edge_encoder(
                    message=-1, attrs=edge_attr, dont_use_message=True)
        else:  # global
            edge_attr = None
        return edge_attr

    def aggregate(self, graph, agg, encode=None, pool_efficiently=False):
        if pool_efficiently:
            dst, src = graph[f"index_{agg}_pool"]
        else:
            dst, src = graph[f"index_{agg}"]

        message = torch.index_select(graph.x, dim=0, index=src)
        if encode is not None:
            message = encode(message, graph[f"attrs_{agg}"])

        return pys.scatter(message, dim=0, index=dst, dim_size=len(graph.x))

    def gin_encoder(self, batch, eps, agg, encoder):
        self_elem = batch.x * (1.0 + eps)
        agg_elem = self.aggregate(graph=batch, agg=agg, encode=None)
        agg_final_element = self_elem + agg_elem
        agg_final_element = encoder(agg_final_element)
        return agg_final_element

    def mpnn_encoder(self, mpnn, x, edge_index, edge_attr):
        return mpnn(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def agg_encoder(self, batch, agg, edge_encoder, mpnn):
        encoded_edge_atr = self.get_edge_attr(
            agg=agg, batch=batch, edge_encoder=edge_encoder)
        agg_final_element = self.mpnn_encoder(
            mpnn=mpnn, x=batch.x, edge_index=batch[f"index_{agg}"], edge_attr=encoded_edge_atr)
        return agg_final_element

    # ============================= init - helpers ============================= #
    def init_mpnn(self, in_dim, out_dim, edge_in_dim, H, type):
        mpnn = layers.Attention_block(
            d=in_dim, H=H, d_output=out_dim, edge_dim=edge_in_dim, type=type)
        return mpnn

    def init_edge_encoder(self, use_linear, in_dim, out_dim):
        edge_encoder = layers.Bond(
            dim=out_dim, linear=use_linear, linear_in_dim=in_dim)
        return edge_encoder

    def init_eps(self):
        eps = nn.Parameter(torch.zeros(1))
        return eps

    def init_point_encoder(self, use_linear, in_dim, out_dim):
        if use_linear:
            layer_encoder = layers.LINEAR(in_dim, out_dim)
        else:
            layer_encoder = layers.MLP(in_dim, out_dim)
        return layer_encoder

    def is_point_agg(self, agg):
        return not (("L" in agg) or ("G" in agg))

    def use_edge_encoder(self, agg):
        if "L" in agg and self.use_edge_attr:
            return True
        else:
            return False

    def get_edge_encoder_out_dim(self, layer_idx):
        edge_out_dim = self.post_concat_dim_embed
        if layer_idx == 0 and self.dataset == "alchemy":
            edge_out_dim = 6
        return edge_out_dim

    def get_cat_encoder(self, use_linear, in_dim, out_dim):
        if use_linear:
            cat_encoder = layers.LINEAR(in_dim, out_dim)
        else:
            cat_encoder = layers.MLP(in_dim, out_dim)
        return cat_encoder

    # ============================= general - helpers ============================= #

    def get_preprocess_layer(self, dim, num_eigen_vectors, max_dis, use_linear, atom_dim):
        use_PE = False
        nm_dim = dim
        if "_PE" in self.model_name:
            logging.info("Using PE\n")
            use_PE = True
            nm_dim = dim - num_eigen_vectors
            assert nm_dim >= 0, "nm_dim shoould alway be greater than 0! decrease num_eigen_vectors in the PE layer"
        else:
            logging.info(
                "Not using PE -- ignoring the parameter: num_eigen_vectors\n")
        self.atom_encoder = layers.Atom(
            dim=nm_dim, max_dis=max_dis, encode=True, use_linear=use_linear, atom_dim=atom_dim)
        if use_PE:
            self.PE_layer = layers.PE_layer(num_eigen_vectors=num_eigen_vectors)
        else:
            self.PE_layer = None

        return self.atom_encoder, self.PE_layer

    @staticmethod
    def compute_final_embedding_dimension(dim_embed, num_aggs, H=1):
        each_head_dim = (dim_embed // num_aggs) // H
        each_agg_dim_embed = each_head_dim * H
        post_concat_dim_embed = each_agg_dim_embed * num_aggs
        if dim_embed != post_concat_dim_embed:
            logging.info(
                "Modified the embedding dim to fit the concatenation and the heads!\n")

            logging.info(
                f"Original embedding final dimension of layers concatenated: {dim_embed}")
            logging.info(
                f"Modified embedding final dimension of layers concatenated: {post_concat_dim_embed}")
            logging.info(f"Each agg embedding size: {each_agg_dim_embed}")
            logging.info(f"Number of aggs: {num_aggs}")

            logging.info(f"Each head embedding size: {each_head_dim}")
            logging.info(f"Number of heads: {H}")
            # logging.info(
            #     f"Asserting that \n   1)  {H=} * {each_head_dim=} = {each_agg_dim_embed=}  \nand\n   2)  {each_agg_dim_embed=} * {num_aggs=} = {post_concat_dim_embed=}"
            # )

            assert H * each_head_dim == each_agg_dim_embed, (
                f"Dimension mismatch: Expected {H * each_head_dim} but got {each_agg_dim_embed}"
            )

            assert each_agg_dim_embed * num_aggs == post_concat_dim_embed, (
                f"Dimension mismatch: Expected {each_agg_dim_embed * num_aggs} but got {post_concat_dim_embed}"
            )
        return post_concat_dim_embed, each_agg_dim_embed


def get_model_params(model, dim_embed, AtomEncoder=AtomEncoder, BondEncoder=BondEncoder):

    try:
        total_params = sum(param.numel() for param in model.parameters())
        unused_atom_embed_params = sum(
            sum(param.numel() for param in m.parameters()) - 30 * dim_embed
            for m in model.modules() if isinstance(m, AtomEncoder)
        )

        unused_bond_embed_params = sum(
            sum(param.numel() for param in m.parameters()) - 5 * dim_embed
            for m in model.modules() if isinstance(m, BondEncoder)
        )

        unused_params = unused_atom_embed_params + unused_bond_embed_params
        return total_params, unused_params, total_params - unused_params

    except Exception as e:
        print(
            f"An error occurred when caculating the model size: {e}!\n Skipping this part.")
        print(f"\n")
        return -1, -1, -1
