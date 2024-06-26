import torch
import torch.nn as nn
import torch_geometric.nn as gnn 
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import logging
import torch.nn.functional as F


# --------------------------------- Atom/Bond EMBEDDING -------------------------------- #
class Atom(nn.Module):
    """Atom encoder

    Args:
        dim (int): embedding dimension
        dis (int): maximum encoding distance
        encode (bool): whether to use encoder

    """

    def __init__(self, dim: int, max_dis: int, encode: bool = True, 
                 use_linear: bool = False,
                 atom_dim: int = 6):
        super().__init__()
        self.max_dis = max_dis
        self.encode = encode
        if use_linear:
            logging.info("Using an MLP to encode atoms -- uses atom_dim variable.")
        else:
            logging.info("Using a look up table to encode atoms -- not uses atom_dim variable.")
        self.use_linear = use_linear
        self.embed_v = encode and AtomEncoder(dim)

        if use_linear: # linear layer
            self.embed_v = encode and nn.Sequential(nn.Linear(atom_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        else: # look up table
            self.embed_v = encode and AtomEncoder(dim)


        # Warning: To account for infinite distance, which we encode with the value max_dis + 2
        self.embed_d = nn.Embedding(max_dis + 2, dim)

    def forward(self, batch):
        if self.encode:
            if not self.use_linear:
                if batch.x.dtype == torch.float32:
                    batch.x = batch.x.int()
            x = self.embed_v(batch.x)
        else:
            x = 0
        # Warning: Assumes the value of 1001 accounts for 2 nodes which are unreachable from each other!
        d = Atom.custom_clamp(tensor=batch.d, min_val=None, max_val=self.max_dis)
        d = self.embed_d(d)
        # batch.x = torch.cat([x, d], dim=-1)
        batch.x = x + d

        del batch.d

        return batch
    @staticmethod
    def custom_clamp(tensor, min_val, max_val):
        # First, clamp all values between min_val and max_val
        clamped_tensor = torch.clamp(tensor, min_val, max_val)
        
        # Create a mask for values that are greater than 1000 in the original tensor
        mask = tensor > 1000
        
        # Update the elements where the condition in 'mask' is True
        clamped_tensor[mask] = max_val + 1
        
        return clamped_tensor


class Bond(nn.Module):
    """Bond encoder

    Args:
        dim (int): embedding dimension

    """

    def __init__(self, dim: int, 
                linear: bool = False,
                linear_in_dim: int = 4):
        super().__init__()
        self.linear = linear
        if self.linear:
            logging.info("Using an MLP to encode bonds -- uses atom_dim variable.")
            self.bond_encoder = nn.Sequential(nn.Linear(linear_in_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        else:
            logging.info("Using a look up table to encode bonds -- not uses edge_attr_dim variable.")

        self.embed = BondEncoder(dim)

    def forward(self, message, attrs, dont_use_message=False, peptides_flag = False, SSWL = False):
        if attrs is None:
            return F.relu(message)
        if self.linear and SSWL:
            attr_of_each_edge = attrs
            if peptides_flag:
                return self.bond_encoder(attr_of_each_edge.float())
            else:
                attr_of_each_edge = attrs
                return F.relu(message + self.bond_encoder(attr_of_each_edge))
        
        if self.linear:
            attr_of_each_edge = attrs
            if peptides_flag:
                return self.bond_encoder(attr_of_each_edge.float())
            else:
                return self.bond_encoder(attr_of_each_edge)

        if dont_use_message:
            attr_of_each_edge = attrs
            if attr_of_each_edge.shape[-1] == 1:
                return F.relu(self.embed(attr_of_each_edge)) # Graphs x Embs
            else: # alchemy 
                return F.relu(self.embed(attr_of_each_edge.to(torch.int)[:, None]).mean(dim=1)) # Graphs x Embs

        else:
            attr_of_each_edge = attrs
            return F.relu(message + self.embed(attr_of_each_edge))


# --------------------------------- PE layer -------------------------------- #
        
class PE_layer(nn.Module):
    def __init__(self, num_eigen_vectors):
        super().__init__()
        self.num_eigen_vectors = num_eigen_vectors


    def forward(self, batch):
        PE_vector = self.sign_fliper(batch.subgraph_PE)
        PE_vector = PE_vector[:, :self.num_eigen_vectors]
        x = torch.cat([batch.x, PE_vector], dim=-1)
        batch.x = x
        # Cleanup attributes
        if hasattr(batch, 'subgraph_PE'):
            del batch.subgraph_PE
        if hasattr(batch, 'subgraph_SE'):
            del batch.subgraph_SE
        del x
        # torch.cuda.empty_cache()
        return batch
    
    def sign_fliper(self, tensor):
        N, K = tensor.shape
        sign_tensor = (torch.randint(0, 2, (1, K), device=tensor.device) * 2 - 1)

        flipped_tensor = tensor * sign_tensor
        return flipped_tensor

# --------------------------------- Attention block -------------------------------- #

class Attention_block(torch.nn.Module):
    def __init__(self, d, H=1, d_output=64, edge_dim=64, type='Gat'):
        super(Attention_block, self).__init__()
        self.H = H
        self.d_output = d_output
        self.edge_dim = edge_dim
        self.d = d
        assert self.d_output == (
            (self.d_output // self.H) * self.H), f"Invalid self.d_output value. Expected: {(self.d_output // self.H) * self.H}, but got: {self.d_output}."
        if type == 'GatV2':
            self.attn_layer = gnn.GATv2Conv(
                in_channels=self.d, out_channels=self.d_output // self.H, heads=self.H, edge_dim=self.edge_dim)
        elif type == 'Transformer_conv':
            self.attn_layer = gnn.TransformerConv(
                in_channels=self.d, out_channels=self.d_output // self.H, heads=self.H, edge_dim=self.edge_dim)
        elif type == 'Gat':
            self.attn_layer = gnn.GATConv(
                in_channels=self.d, out_channels=self.d_output // self.H, heads=self.H, edge_dim=self.edge_dim)
        else:
            raise ValueError(f"{type} is not a valid transformer model.")

    def forward(self, x, edge_index, edge_attr):
        x_attn, _ = self.attn_layer(
            x=x, edge_index=edge_index, edge_attr=edge_attr, return_attention_weights=False)

        return x_attn



# --------------------------------- General Layers -------------------------------- #

class NormReLU(nn.Sequential):

    def __init__(self, dim: int):
        super().__init__()

        self.add_module("bn", nn.BatchNorm1d(dim))
        self.add_module("ac", nn.ReLU())


class MLP(nn.Sequential):

    def __init__(self, idim: int, odim: int, hdim: int = None, norm: bool = True):
        super().__init__()
        hdim = hdim or idim

        self.add_module("input", nn.Linear(idim, hdim))
        self.add_module("input_nr", NormReLU(hdim) if norm else nn.ReLU())
        self.add_module("output", nn.Linear(hdim, odim))

class LINEAR(nn.Sequential):

    def __init__(self, idim: int, odim: int):
        super().__init__()

        self.add_module("input", nn.Linear(idim, odim))


class Pooling(nn.Module):
    """Final pooling

    Args:
        idim (int): input dimension
        odim (int): output dimension

    """

    def __init__(self, idim: int, odim: int):
        super().__init__()

        self.predict = MLP(idim, odim, hdim=idim*2, norm=False)

    def forward(self, batch, subgraph_rep = None, efficient=False):
        if efficient: # Warning: this is mean pool!!!
            return self.predict(gnn.global_mean_pool(subgraph_rep, batch.batch))
        else:
            # Warning: this is sum pool !!!
            return self.predict(gnn.global_mean_pool(batch.x, batch.batch))


