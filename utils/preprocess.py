import torch
import numpy as np
import torch_geometric as pyg
import torch_geometric.data as data


def subgraph_construction(cfg):

    uv = "uv" in cfg.model.aggs
    vu = "vu" in cfg.model.aggs
    uu = "uu" in cfg.model.aggs
    vv = "vv" in cfg.model.aggs
    uG = "uG" in cfg.model.aggs
    vG = "vG" in cfg.model.aggs
    GG = "GG" in cfg.model.aggs
    uL = "uL" in cfg.model.aggs
    vL = "vL" in cfg.model.aggs

    dropout_p = 1 - cfg.data.sampling.keep_subgraph_prob

    def call(graph):

        subgraph_node_indices = torch.arange(graph.num_nodes**2) \
            .view((graph.num_nodes, graph.num_nodes))

        sample = False
        if dropout_p > 0:
            sample = True
            n = graph.num_nodes
            sample_size = round((1 - dropout_p) * n)
            if sample_size == 0:
                sample_size = 1

            sampled_subgraphs_idx = sample_indices(n=n, k=sample_size)
            unsampled_subgraphs_idx = exclude_indices_efficient(
                sampled_subgraphs_idx, N=n)

            sampled_nodes_idx = from_subgraph_indices_to_nodes_indices(
                indices=sampled_subgraphs_idx, n=n)
            unsampled_nodes_idx = exclude_indices_efficient(
                sampled_nodes_idx, N=n**2)

        num_subgraphs = graph.num_nodes
        adj_of_origianl_graph = pyg.utils.to_dense_adj(
            graph.edge_index, max_num_nodes=graph.num_nodes).squeeze(0)

        apsp = get_all_pairs_shortest_paths(adj=adj_of_origianl_graph)
        edge_index_of_original_graph = graph.edge_index
        edge_attr_original_graph = graph.edge_attr

        subgraphs_x = get_subgraph_node_features_multi(
            original_graph_features=graph.x, subgraph_node_indices=subgraph_node_indices)

        # Warning: Assumes the value of 1001 accounts for 2 nodes which are unreachable from each other!
        # Replaces all 'inf' values with 1001
        apsp[torch.isinf(apsp)] = 1001.0
        apsp = apsp.to(int).flatten(end_dim=1)

        if sample:
            num_subgraphs = len(sampled_subgraphs_idx)
            subgraphs_x = select_rows(
                matrix=subgraphs_x, indices=sampled_nodes_idx)
            apsp = select_rows(matrix=apsp.reshape(-1, 1),
                               indices=sampled_nodes_idx)
            apsp = apsp.flatten(end_dim=1)
            sampled_nodes_idx = torch.tensor(sampled_nodes_idx).reshape(-1, n)
            mapping = generate_index_mapping(sampled_nodes_idx)

        data_dict = {
            "num_subgraphs_bch": num_subgraphs,
            # original_graph_properties
            "orig_edge_index": edge_index_of_original_graph,
            "y": graph.y,
            # node features
            "x": subgraphs_x,
            # node marking
            "d": apsp
        }

        # edge indexes for each agg
        # edge_index = [[     dst      ],
        #              [      src     ]]

        if uv:
            edge_index = get_edge_index_uv(subgraph_node_indices)
            data_dict["index_uv"] = edge_index

        if vu:
            edge_index = get_edge_index_vu(subgraph_node_indices)
            data_dict["index_vu"] = edge_index

        if uu:
            edge_index = get_edge_index_uu(subgraph_node_indices)
            data_dict["index_uu"] = edge_index

        if vv:

            edge_index = get_edge_index_vv(subgraph_node_indices)
            if dropout_p > 0:
                filtered_edge_index, mask_removed = remove_edges(
                    edge_index, unsampled_nodes_idx)
                filtered_edge_index = apply_index_mapping(
                    filtered_edge_index, mapping)
                data_dict["index_vv"] = filtered_edge_index
            else:
                data_dict["index_vv"] = edge_index

        if cfg.model.sum_pooling:
            if dropout_p > 0:
                filtered_edge_index = get_edge_index_uG(sampled_nodes_idx)
                filtered_edge_index = apply_index_mapping(
                    filtered_edge_index, mapping)
                data_dict["index_uG"] = filtered_edge_index
            else:
                edge_index = get_edge_index_uG(subgraph_node_indices)
                data_dict["index_uG"] = edge_index
        if not cfg.model.sum_pooling:
            edge_index = get_edge_index_uG_efficient_pooling(num_subgraphs)
            data_dict["index_uG_pool"] = edge_index

        if vG:
            edge_index = get_edge_index_vG(subgraph_node_indices)
            data_dict["index_vG"] = edge_index

        if GG:
            edge_index = get_edge_index_GG(subgraph_node_indices)
            data_dict["index_GG"] = edge_index

        if uL:  # inside the same subgraph
            if dropout_p > 0:
                filtered_edge_index = edge_index_of_original_graph
                filtered_edge_index = filtered_edge_index.type(torch.int64)

                filtered_edge_index = get_Lu_edge_index_sampling(
                    filtered_edge_index, sampled_subgraphs_idx, n)
                filtered_edge_index = apply_index_mapping(
                    filtered_edge_index, mapping)

                filtered_edge_attr = edge_attr_original_graph
                if filtered_edge_attr is not None:
                    filtered_edge_attr = edge_attr_original_graph

                    filtered_edge_attr = get_Lu_edge_attr_sampling(
                        filtered_edge_attr, sampled_subgraphs_idx)
                data_dict.update({"index_uL": filtered_edge_index,
                                  "attrs_uL": filtered_edge_attr})
                filtered_edge_index_uL = filtered_edge_index
                filtered_edge_attr_uL = filtered_edge_attr

            else:
                edge_index = get_edge_index_uL(
                    subgraph_node_indices, edge_index_of_original_graph)
                if edge_attr_original_graph == None:
                    data_dict.update({"index_uL": edge_index,
                                      "attrs_uL": None})
                else:
                    edge_attr = get_edge_attr_uL(
                        subgraph_node_indices, edge_attr_original_graph)
                    data_dict.update({"index_uL": edge_index,
                                      "attrs_uL": edge_attr})
        if vL:
            if dropout_p > 0:
                filtered_edge_index, mask_removed = remove_edges(
                    edge_index_of_original_graph, unsampled_subgraphs_idx)
                filtered_edge_index = get_Lv_edge_index_sampling(
                    filtered_edge_index, n)
                filtered_edge_index = apply_index_mapping(
                    filtered_edge_index, mapping)

                filtered_edge_attr = edge_attr_original_graph
                if filtered_edge_attr is not None:
                    filtered_edge_attr = edge_attr_original_graph[~mask_removed]
                    filtered_edge_attr = get_Lv_edge_attr_sampling(
                        filtered_edge_attr, n)
                # Warning: if filtered_edge_index.numel() == 0 it uses uL again!
                if filtered_edge_index.numel() == 0:
                    data_dict.update({"index_vL": filtered_edge_index_uL,
                                      "attrs_vL": filtered_edge_attr_uL})

                else:
                    data_dict.update({"index_vL": filtered_edge_index,
                                      "attrs_vL": filtered_edge_attr})
            else:
                edge_index = get_edge_index_vL(
                    subgraph_node_indices, edge_index_of_original_graph)
                if edge_attr_original_graph == None:
                    data_dict.update({"index_vL": edge_index,
                                      "attrs_vL": None})
                else:
                    edge_attr = get_edge_attr_vL(
                        subgraph_node_indices, edge_attr_original_graph)
                    data_dict.update({"index_vL": edge_index,
                                      "attrs_vL": edge_attr})

        # PE
        # Warning: k has to be <= 16!
        num_eigen_vectors = 16

        if dropout_p > 0:
            indices = sampled_nodes_idx.reshape(-1)
            indices = indices.clone().detach()

            lap_cart = get_laplacian_pe_for_kron_graph(data=graph, norm_type='none',
                                                       pos_enc_dim=num_eigen_vectors)

            data_dict.update({"subgraph_PE": lap_cart[indices]})
        else:
            data_dict.update({"subgraph_PE": get_laplacian_pe_for_kron_graph(data=graph, norm_type='none',
                                                                             pos_enc_dim=num_eigen_vectors)})

        return data.Data(**data_dict)

    return call


# ---------------------- Pre-process - helpers ----------------------------- #

def sample_indices(n, k):
    """
    Samples k different indices from the range 0 to n-1, without repetition.

    Parameters:
    n (int): The upper limit of the range (exclusive).
    k (int): The number of indices to sample.

    Returns:
    list: A list of k sampled indices.
    """
    if k > n:
        raise ValueError("k cannot be greater than n.")
    indices = np.random.choice(n, k, replace=False).tolist()
    indices.sort()
    assert len(
        indices) == k, f"The number of indices should be {k}, but got {len(indices)}."
    return torch.tensor(indices)


def exclude_indices_efficient(sampled_indices, N):
    """
    Function to find unsampled indices in a more efficient way without using loops.

    :param sampled_indices: torch.Tensor containing sampled indices.
    :param N: The range limit (0 to N).
    :return: torch.Tensor containing the unsampled indices.
    """
    all_indices = torch.arange(N)
    mask = torch.ones(N, dtype=torch.bool)
    mask[sampled_indices] = False
    return all_indices[mask]


def from_subgraph_indices_to_nodes_indices(indices, n):
    """
    Creates a concatenated list based on the given indices and value n, without using loops.

    Parameters:
    indices (list): A list of integer indices.
    n (int): The value used to generate range of numbers for each index.

    Returns:
    list: A concatenated list based on the provided indices and n.
    """
    indices_array = np.array(
        indices)[:, None]  # Convert indices to a 2D column array
    # Broadcasting to create the ranges
    ranges = indices_array * n + np.arange(n)
    return np.sort(ranges.ravel())  # Flatten and sort the array


def get_all_pairs_shortest_paths(adj):
    spd = torch.where(~torch.eye(len(adj), dtype=bool) & (adj == 0),
                      torch.full_like(adj, float("inf")), adj)
    # Floyd-Warshall

    for k in range(len(spd)):
        dist_from_source_to_k = spd[:, [k]]
        dist_from_k_to_target = spd[[k], :]
        dist_from_source_to_target_via_k = dist_from_source_to_k + dist_from_k_to_target
        spd = torch.minimum(spd, dist_from_source_to_target_via_k)
    return spd


def get_subgraph_node_features_multi(original_graph_features, subgraph_node_indices):
    N = len(subgraph_node_indices)
    expanded_tensor = torch.cat([original_graph_features] * N, dim=0)
    return expanded_tensor


def select_rows(matrix, indices):
    """
    Selects rows from a matrix based on a list of indices.

    Parameters:
    matrix (numpy.ndarray): A 2D numpy array (matrix) of dimensions N x D.
    indices (list): A list of row indices to be selected from the matrix.

    Returns:
    numpy.ndarray: A new matrix composed of the rows corresponding to the provided indices.
    """
    indices.sort()
    return matrix[indices, :]


def generate_index_mapping(node_indices):
    """
    Generate a mapping from original node indices to new sequential node indices
    using functional programming (map and zip).

    :param node_indices: A 2D torch tensor representing node indices.
    :return: A mapping (dictionary) from original node indices to new sequential indices.
    """
    # Flatten the tensor and find unique indices
    unique_indices = torch.unique(node_indices.flatten()).tolist()

    # Use map and zip to create the mapping
    index_map = dict(zip(unique_indices, map(int, range(len(unique_indices)))))

    return index_map


def get_edge_index_uv(subgraph_node_indices):
    # uv <- uv
    src_nodes = subgraph_node_indices
    target_nodes = subgraph_node_indices
    index_uv = torch.stack((target_nodes, src_nodes)).flatten(start_dim=1)
    # i = 5  # or any other node index
    # neighbors = index_uv[1][index_uv[0] == i]
    # print(f'Agg uv: node is {i} and the neigbours are {neighbors}')
    return index_uv


def get_edge_index_vu(subgraph_node_indices):
    # uv <- vu
    src_nodes = subgraph_node_indices.T
    target_nodes = subgraph_node_indices
    index_vu = torch.stack((target_nodes, src_nodes)).flatten(start_dim=1)
    return index_vu


def get_edge_index_uu(subgraph_node_indices):
    # uv <- uu
    target_nodes = subgraph_node_indices
    # broadcasts the diagonal left and right
    _, src_nodes = torch.stack(torch.broadcast_tensors(
        target_nodes, torch.diag(subgraph_node_indices)[:, None]))
    index_uu = torch.stack((target_nodes, src_nodes)).flatten(start_dim=1)
    return index_uu


def get_edge_index_vv(subgraph_node_indices):
    # uv <- vv
    target_nodes = subgraph_node_indices
    # broadcasts the diagonal up and down
    _, src_nodes = torch.stack(torch.broadcast_tensors(
        target_nodes, torch.diag(subgraph_node_indices)[None, :]))
    index_vv = torch.stack((target_nodes, src_nodes)).flatten(start_dim=1)
    return index_vv


def get_edge_index_uG(subgraph_node_indices):
    # uv <- uG
    # make each node ([:, :, None]) match all nodes of its subgraph ([:, None, :])
    target_nodes, src_nodes = torch.broadcast_tensors(
        subgraph_node_indices[:, :, None], subgraph_node_indices[:, None, :])
    index_uG = torch.stack((target_nodes, src_nodes)).flatten(start_dim=1)
    return index_uG


def get_edge_index_uG_efficient_pooling(num_subgraphs):
    src_nodes = torch.arange(num_subgraphs**2)
    target_nodes = torch.repeat_interleave(
        torch.arange(num_subgraphs), num_subgraphs)
    index_uG_pool = torch.stack((target_nodes, src_nodes)).flatten(start_dim=1)
    return index_uG_pool


def get_edge_index_vG(subgraph_node_indices):
    # uv <- vG
    target_nodes, src_nodes = torch.broadcast_tensors(
        subgraph_node_indices[None, :, :], subgraph_node_indices[:, None, :])
    index_vG = torch.stack((target_nodes, src_nodes)).flatten(start_dim=1)

    return index_vG


def get_edge_index_GG(subgraph_node_indices):
    N = len(subgraph_node_indices.flatten(start_dim=0))

    # Generate all possible pairs of nodes as edges (fully connected graph) using torch
    row_indices, col_indices = torch.triu_indices(N, N, offset=1)

    # Convert the indices to a torch tensor
    index_GG = torch.stack([row_indices, col_indices], dim=0)
    return index_GG


def get_edge_index_uL(subgraph_node_indices, edge_index_of_original_graph):
    """
    Compute the edge index for subgraphs based on original graph's edge index and subgraph node indices.

    Args:
    - subgraph_node_indices (torch.Tensor): Indices of the nodes in the subgraph.
    - edge_index_of_original_graph (torch.Tensor): Edge index tensor of the original graph.

    Returns:
    - torch.Tensor: Computed edge index for subgraphs.
    """

    # Adjust the subgraph's node indices to the shape of the edge index tensor
    adjusted_node_indices = subgraph_node_indices[None, None, :, 0]

    # Combine the adjusted node indices and the edge indices of the original graph
    # 2 X num_edges X num_nodes_per_subgraph
    combined_indices = adjusted_node_indices + \
        edge_index_of_original_graph[:, :, None]

    # Flatten the tensor to get the final edge index
    index_uL = combined_indices.flatten(start_dim=1)

    return index_uL



def get_edge_index_vL(subgraph_node_indices, edge_index_of_original_graph):
    """
    Compute the edge index for subgraphs based on original graph's edge index and subgraph node indices.

    Args:
    - subgraph_node_indices (torch.Tensor): Indices of the nodes in the subgraph.
    - edge_index_of_original_graph (torch.Tensor): Edge index tensor of the original graph.

    Returns:
    - torch.Tensor: Computed edge index for subgraphs.
    """

    # Number of nodes in the subgraph
    num_nodes_per_subgraph = len(subgraph_node_indices)

    # Adjust the edge indices based on the subgraph's node indices
    adjusted_edge_index = edge_index_of_original_graph[:,
                                                       :, None] * num_nodes_per_subgraph

    # Add the subgraph node indices
    # 2 X num_edges X num_nodes_per_subgraph
    indexed_subgraph = subgraph_node_indices[None,
                                             None, 0, :] + adjusted_edge_index

    # Flatten the tensor to get the final edge index
    index_vL = indexed_subgraph.flatten(start_dim=1)

    return index_vL



def get_edge_attr_uL(subgraph_node_indices, edge_attr_of_original_graph):
    # num_features X num_edges X num_nodes_per_subgraph
    edge_attr_of_original_graph_2d = prepare_edge_attributes_for_subgraph_processing(
        edge_attr_of_original_graph=edge_attr_of_original_graph, subgraph_node_indices=subgraph_node_indices)
    return edge_attr_of_original_graph_2d.T



def get_edge_attr_vL(subgraph_node_indices, edge_attr_of_original_graph):
    return get_edge_attr_uL(subgraph_node_indices, edge_attr_of_original_graph)



def prepare_edge_attributes_for_subgraph_processing(edge_attr_of_original_graph, subgraph_node_indices):
    """
    Prepares the edge attributes of the original graph for subgraph processing.

    Args:
    - edge_attr_of_original_graph: Tensor representing edge attributes of the original graph.
    - subgraph_node_indices: List of node indices in the subgraphs.

    Returns:
    - Tensor processed for subgraph operations.
    """

    # Ensure the feature dimension is 2D
    # num_edges x num_features
    edge_attr_2d = ensure_feature_dim_2d(edge_attr_of_original_graph)
    # Add a new dimension at the beginning
    # 1 x num_edges x num_features
    expanded_dim = edge_attr_2d[None, :, :]

    # Expand the tensor based on the length of subgraph_node_indices
    # num_subgraphs x num_edges x num_features
    expanded_tensor = expanded_dim.expand(len(subgraph_node_indices), -1, -1)

    # Permute the tensor dimensions
    permuted_tensor = expanded_tensor.permute(2, 1, 0)

    # Flatten the tensor starting from the second dimension
    # num_features X num_edges X num_nodes_per_subgraph
    # should flatten dimenstion with: num_edges x num_subgraphs
    flattened_tensor = permuted_tensor.flatten(start_dim=1)


    return flattened_tensor


def ensure_feature_dim_2d(tensor):
    if len(tensor.shape) == 1:
        return tensor.unsqueeze(1)  # or tensor.view(-1, 1)
    return tensor


def remove_edges(edge_list, node_indices):
    """
    Remove any edge that includes a node present in node_indices using NumPy.

    :param edge_list: 2D list or array representing the edges (2 x E).
    :param node_indices: List of node indices to remove edges from.
    :return: The modified edge list with specified edges removed.
    """
    # Convert edge_list and node_indices to numpy arrays
    edge_array = np.array(edge_list)
    node_indices_array = np.array(node_indices)

    # Identify columns (edges) to remove
    mask_removed = np.isin(edge_array, node_indices_array).any(axis=0)
    # mask_removed = True -> remove edge!

    # Remove the identified edges
    filtered_edges = np.delete(edge_array, np.where(mask_removed), axis=1)
    filtered_edges = torch.from_numpy(filtered_edges)

    return filtered_edges, mask_removed


def apply_index_mapping(tensor, index_map):
    """
    Apply the index mapping to a new tensor.

    :param tensor: A torch tensor with old node indices.
    :param index_map: The mapping from old node indices to new sequential indices.
    :return: A new tensor with updated indices according to the mapping.
    """
    # Apply the mapping
    mapped_tensor = tensor.apply_(lambda x: index_map[x])

    return mapped_tensor


def get_Lu_edge_index_sampling(filtered_edge_index, subgraph_sampled_indices, num_nodes):
    """
    Create a new edge index by concatenating the original edge_index with itself,
    each time offset by elements of subgraph_sampled_indices, in the specified order.

    :param filtered_edge_index: A 2D torch tensor representing the edges of a graph.
    :param subgraph_sampled_indices: A 1D torch tensor of indices to be added to edge_index.
    :param num_nodes: number of nodes
    :return: A new edge index tensor.
    """
    # Get the number of edges and number of indices
    num_edges = filtered_edge_index.size(1)
    num_indices = subgraph_sampled_indices.size(0)

    # Expand and repeat the edge_index and subgraph_sampled_indices
    expanded_edge_index = filtered_edge_index.repeat(1, num_indices)
    expanded_indices = subgraph_sampled_indices.repeat_interleave(
        num_edges).view(1, -1).repeat(2, 1)

    # Add the expanded indices to the expanded edge index
    final_edge_index = expanded_edge_index + (expanded_indices * num_nodes)

    # Reshape to get the final concatenated edge index
    final_edge_index = final_edge_index.view(2, -1)

    return final_edge_index


def get_Lu_edge_attr_sampling(filtered_edge_attr, subgraph_sampled_indices):
    """
        Concatenate the matrix A with itself K times row-wise.

        :param filtered_edge_attr: A torch tensor of shape (N, D).
        :param subgraph_sampled_indices: The number of times to repeat A row-wise.
        :return: A new torch tensor of shape (N*subgraph_sampled_indices, D).
    """
    if filtered_edge_attr is None:
        return filtered_edge_attr
    if len(filtered_edge_attr.shape) == 1:
        filtered_edge_attr = filtered_edge_attr.reshape(-1, 1)
    K = len(subgraph_sampled_indices)
    # Repeat A K times along the row dimension
    filtered_edge_attr = filtered_edge_attr.repeat(K, 1)

    return filtered_edge_attr


def get_Lv_edge_index_sampling(filtered_edge_index, num_nodes):
    """
    Create a new edge index by expanding each edge in the edge_index.
    For each edge (a, b), create 'num_nodes' edges:
    (a*num_nodes, b*num_nodes), (a*num_nodes+1, b*num_nodes+1), ..., 
    (a*num_nodes+num_nodes-1, b*num_nodes+num_nodes-1).

    :param filtered_edge_index: A 2D torch tensor representing the edges of a graph.
    :param num_nodes: Number of nodes to expand each edge into.
    :return: A new edge index tensor.
    """
    # Create offsets
    offsets = torch.arange(0, num_nodes).to(filtered_edge_index.device)

    # Expand edge_index and add offsets
    a_expanded = (filtered_edge_index[0].unsqueeze(1).repeat(
        1, num_nodes) * num_nodes + offsets).view(-1)
    b_expanded = (filtered_edge_index[1].unsqueeze(1).repeat(
        1, num_nodes) * num_nodes + offsets).view(-1)

    # Combine into new edge index
    final_edge_index = torch.stack([a_expanded, b_expanded], dim=0)

    return final_edge_index


def get_Lv_edge_attr_sampling(filtered_edge_attr, num_nodes):
    """
    Repeat each row of the matrix filtered_edge_attr K times, where K is the length of subgraph_sampled_indices.

    :param filtered_edge_attr: A torch tensor of shape (N, D).
    :param num_nodes: A list or tensor to determine the number of repetitions.
    :return: A new torch tensor of shape (N*K, D).
    """
    if filtered_edge_attr is None:
        return filtered_edge_attr

    K = num_nodes
    if len(filtered_edge_attr.shape) == 1:
        filtered_edge_attr = filtered_edge_attr.reshape(-1, 1)
    N, D = filtered_edge_attr.shape

    # Repeat each row of filtered_edge_attr K times
    repeated_filtered_edge_attr = filtered_edge_attr.unsqueeze(
        1).repeat(1, K, 1).view(N*K, D)

    return repeated_filtered_edge_attr


def get_laplacian_pe_for_kron_graph(data, norm_type='none', pos_enc_dim=8):
    L = get_laplacian(data, norm_type)
    EigVal, EigVec = np.linalg.eig(L)
    EigVec = sign_fliper(EigVec)
    idx = EigVal.argsort()  # increasing order

    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    EigVec_top_k = EigVec[:, 1:pos_enc_dim+1]
    EigVal_top_k = EigVal[1:pos_enc_dim+1]
    if EigVec_top_k[:, :int((pos_enc_dim+1)**0.5)].size == 0:
        EigVec_top_k = np.zeros((data.num_nodes, pos_enc_dim))

    EigVec_top_k_cartesian = np.kron(EigVec_top_k[:, :int(
        (pos_enc_dim+1)**0.5)], EigVec_top_k[:, :int((pos_enc_dim+1)**0.5)])

    EigVal_top_k_cartesian = EigVal_top_k[:int((pos_enc_dim+1)**0.5)]
    EigVal_top_k_cartesian = EigVal_top_k_cartesian.reshape(
        -1, 1) + EigVal_top_k_cartesian
    EigVal_top_k_cartesian = EigVal_top_k_cartesian.reshape(-1)

    idx = EigVal_top_k_cartesian.argsort()
    EigVal_top_k_cartesian, EigVec_top_k_cartesian = EigVal_top_k_cartesian[idx][:pos_enc_dim], np.real(
        EigVec_top_k_cartesian[:, idx])[:, :pos_enc_dim]
    PE = torch.from_numpy(EigVec_top_k_cartesian).float()
    PE = pad_pe(pe_matrix=PE, D=pos_enc_dim)

    del L
    del EigVal, EigVec
    del EigVec_top_k, EigVal_top_k
    del EigVal_top_k_cartesian, EigVec_top_k_cartesian

    return PE


def get_laplacian(data, norm_type='none'):
    """
    This function calculates the Laplacian of a given graph data.

    Parameters:
    data (torch_geometric.data.Data): Input graph data.
    norm_type (str, optional): Type of normalization. 
        It can be 'none' (stands for L = D - A), 'symmetric', or 'random_walk'. Default is 'none'.

    Returns:
    torch.Tensor: The Laplacian of the graph data.
    """

    edge_index = data.edge_index
    num_nodes = data.num_nodes

    # Convert edge_index to adjacency matrix
    adj = pyg.utils.to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze()

    # Degree matrix
    deg = pyg.utils.degree(edge_index[0], num_nodes, dtype=torch.float)

    if norm_type == 'none':
        # No normalization Laplacian: L = D - A
        L = torch.diag(deg) - adj
    elif norm_type == 'symmetric':
        # Symmetric normalization Laplacian: L = I - D^-0.5 * A * D^-0.5
        D_sqrt_inv = deg.pow(-0.5)
        D_sqrt_inv[deg == 0] = 0
        L = torch.eye(num_nodes) - D_sqrt_inv.view(-1, 1) * \
            adj * D_sqrt_inv.view(1, -1)
    elif norm_type == 'random_walk':
        # Random walk normalization Laplacian: L = I - D^-1 * A
        D_inv = deg.reciprocal()
        D_inv[deg == 0] = 0
        L = torch.eye(num_nodes) - D_inv.view(-1, 1) * adj
    else:
        raise ValueError(
            'norm_type must be either "none", "symmetric" or "random_walk"')

    return L


def sign_fliper(tensor):
    N, K = tensor.shape
    sign_tensor = (np.random.randint(0, 2, (1, N)) * 2 - 1)
    flipped_tensor = tensor * sign_tensor
    return flipped_tensor


def pad_pe(pe_matrix, D):
    # Get shape of the matrix
    N, K = pe_matrix.shape

    # Assert K <= D
    assert K <= D, "K should be less than or equal to D"

    # If K is already equal to D, do nothing and return the matrix
    if K == D:
        return pe_matrix

    # Create an array of zeros to pad each row
    padding = torch.zeros((N, D-K))

    # Concatenate the original matrix with the padding
    padded_matrix = torch.hstack((pe_matrix, padding))

    return padded_matrix
