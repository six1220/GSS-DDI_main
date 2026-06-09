# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# fixme have changed
import torch


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen):
    x = x + 1
    xlen1, xlen2, xlen3 = x.size()
    if xlen1 < padlen:
        new_x = x.new_zeros([padlen, padlen, xlen3], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_4d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def collator(batch):
    max_d_node=256
    multi_hop_max_dist=20
    spatial_pos_max=20
    # 收集所有满足条件的样本数据
    filtered_samples = []
    
    # 2D collator - 过滤并收集数据
    for sample in batch:
        d_node, d_in_degree, d_out_degree, p_node, p_in_degree, p_out_degree, \
        label, d1, d2, mask_1, mask_2, d1_position, d1_z, d1_batch, d2_position, d2_z, d2_batch, \
        (adj_1, node_fts_1, edge_fts_1), (adj_2, node_fts_2, edge_fts_2), d1_expr, d2_expr, d1_morgan_fp, d2_morgan_fp, \
        tanimoto_similarity, mask_1_gnn, mask_2_gnn = sample
        
        if d_node.size(0) <= max_d_node and p_node.size(0) <= max_d_node:
            filtered_samples.append({
                'd1_node': d_node,
                'd1_in_degree': d_in_degree,
                'd1_out_degree': d_out_degree,
                'd2_node': p_node,
                'd2_in_degree': p_in_degree,
                'd2_out_degree': p_out_degree,
                'label': label,
                'd1': d1,
                'd2': d2,
                'mask_1': mask_1,
                'mask_2': mask_2,
                'd1_position': d1_position,
                'd1_z': d1_z,
                'd2_position': d2_position,
                'd2_z': d2_z,
                'adj_1': adj_1,
                'node_fts_1': node_fts_1,
                'edge_fts_1': edge_fts_1,
                'adj_2': adj_2,
                'node_fts_2': node_fts_2,
                'edge_fts_2': edge_fts_2,
                'd1_expr': d1_expr,
                'd2_expr': d2_expr,
                'd1_morgan_fp': d1_morgan_fp,
                'd2_morgan_fp': d2_morgan_fp,
                'tanimoto_similarity': tanimoto_similarity,
                'mask_1_gnn': mask_1_gnn,
                'mask_2_gnn': mask_2_gnn
            })
    
    n_samples = len(filtered_samples)
    if n_samples == 0:
        # 如果没有样本通过过滤，返回一个空批次
        # 这样训练循环会跳过这个批次，而不是失败
        return None
    
    # 2D collator - 处理 node 和 degree
    drug1_node = torch.cat([pad_2d_unsqueeze(s['d1_node'], max_d_node) for s in filtered_samples])
    drug2_node = torch.cat([pad_2d_unsqueeze(s['d2_node'], max_d_node) for s in filtered_samples])
    drug1_in_degree = torch.cat([pad_1d_unsqueeze(s['d1_in_degree'], max_d_node) for s in filtered_samples])
    drug2_in_degree = torch.cat([pad_1d_unsqueeze(s['d2_in_degree'], max_d_node) for s in filtered_samples])
    drug1_out_degree = torch.cat([pad_1d_unsqueeze(s['d1_out_degree'], max_d_node) for s in filtered_samples])
    drug2_out_degree = torch.cat([pad_1d_unsqueeze(s['d2_out_degree'], max_d_node) for s in filtered_samples])

    # 1D collator
    n_targets = 1
    n_emb = filtered_samples[0]['d1'].shape[0]
    n_mask = filtered_samples[0]['mask_1'].shape[0]

    target_tensor = torch.zeros(n_samples, n_targets)
    d1_emb_tensor = torch.zeros(n_samples, n_emb)
    d2_emb_tensor = torch.zeros(n_samples, n_emb)
    mask_1_tensor = torch.zeros(n_samples, n_mask)
    mask_2_tensor = torch.zeros(n_samples, n_mask)

    for i, s in enumerate(filtered_samples):
        target_tensor[i] = torch.tensor(s['label'])
        d1_emb_tensor[i] = torch.IntTensor(s['d1'])
        d2_emb_tensor[i] = torch.IntTensor(s['d2'])
        mask_1_tensor[i] = torch.tensor(s['mask_1'])
        mask_2_tensor[i] = torch.tensor(s['mask_2'])

    # 3D collator
    d1_batch_positions, d1_batch_z, d1_batch, d2_batch_positions, d2_batch_z, d2_batch = [], [], [], [], [], []
    for i, s in enumerate(filtered_samples):
        d1_batch_positions.append(s['d1_position'])
        d1_batch_z.append(s['d1_z'])
        d1_batch.append([i for _ in range(len(s['d1_z']))])

        d2_batch_positions.append(s['d2_position'])
        d2_batch_z.append(s['d2_z'])
        d2_batch.append([i for _ in range(len(s['d2_z']))])

    d1_batch_positions = torch.cat(d1_batch_positions, dim=0).detach().clone()
    d1_batch_z = torch.cat(d1_batch_z, dim=0).detach().clone()
    d1_batch = torch.tensor([_ for line in d1_batch for _ in line])

    d2_batch_positions = torch.cat(d2_batch_positions, dim=0).detach().clone()
    d2_batch_z = torch.cat(d2_batch_z, dim=0).detach().clone()
    d2_batch = torch.tensor([_ for line in d2_batch for _ in line])

    # GNN collator
    n_nodes_largest_graph_1 = max(s['adj_1'].shape[0] for s in filtered_samples)
    n_nodes_largest_graph_2 = max(s['adj_2'].shape[0] for s in filtered_samples)

    n_node_fts_1 = filtered_samples[0]['node_fts_1'].shape[1]
    n_edge_fts_1 = filtered_samples[0]['edge_fts_1'].shape[2]
    n_node_fts_2 = filtered_samples[0]['node_fts_2'].shape[1]
    n_edge_fts_2 = filtered_samples[0]['edge_fts_2'].shape[2]

    adjacency_tensor_1 = torch.zeros(n_samples, n_nodes_largest_graph_1, n_nodes_largest_graph_1)
    node_tensor_1 = torch.zeros(n_samples, n_nodes_largest_graph_1, n_node_fts_1)
    edge_tensor_1 = torch.zeros(n_samples, n_nodes_largest_graph_1, n_nodes_largest_graph_1, n_edge_fts_1)

    adjacency_tensor_2 = torch.zeros(n_samples, n_nodes_largest_graph_2, n_nodes_largest_graph_2)
    node_tensor_2 = torch.zeros(n_samples, n_nodes_largest_graph_2, n_node_fts_2)
    edge_tensor_2 = torch.zeros(n_samples, n_nodes_largest_graph_2, n_nodes_largest_graph_2, n_edge_fts_2)

    for i, s in enumerate(filtered_samples):
        n_nodes_1 = s['adj_1'].shape[0]
        n_nodes_2 = s['adj_2'].shape[0]
        adjacency_tensor_1[i, :n_nodes_1, :n_nodes_1] = torch.Tensor(s['adj_1'])
        node_tensor_1[i, :n_nodes_1, :] = torch.Tensor(s['node_fts_1'])
        edge_tensor_1[i, :n_nodes_1, :n_nodes_1, :] = torch.Tensor(s['edge_fts_1'])

        adjacency_tensor_2[i, :n_nodes_2, :n_nodes_2] = torch.Tensor(s['adj_2'])
        node_tensor_2[i, :n_nodes_2, :] = torch.Tensor(s['node_fts_2'])
        edge_tensor_2[i, :n_nodes_2, :n_nodes_2, :] = torch.Tensor(s['edge_fts_2'])
    
    # 基因表达特征 collator
    d1_expr_tensor = torch.zeros(n_samples, 978)
    d2_expr_tensor = torch.zeros(n_samples, 978)
    for i, s in enumerate(filtered_samples):
        d1_expr_tensor[i] = s['d1_expr']
        d2_expr_tensor[i] = s['d2_expr']
    
    # Morgan 指纹 collator
    n_morgan_bits = filtered_samples[0]['d1_morgan_fp'].shape[0]
    d1_morgan_tensor = torch.zeros(n_samples, n_morgan_bits)
    d2_morgan_tensor = torch.zeros(n_samples, n_morgan_bits)
    for i, s in enumerate(filtered_samples):
        d1_morgan_tensor[i] = s['d1_morgan_fp']
        d2_morgan_tensor[i] = s['d2_morgan_fp']
    
    # GNN 掩码 collator
    d1_mask_gnn_tensor = torch.zeros(n_samples, n_nodes_largest_graph_1)
    d2_mask_gnn_tensor = torch.zeros(n_samples, n_nodes_largest_graph_2)
    for i, s in enumerate(filtered_samples):
        n_nodes_1 = s['adj_1'].shape[0]
        n_nodes_2 = s['adj_2'].shape[0]
        d1_mask_gnn_tensor[i, :n_nodes_1] = torch.tensor(s['mask_1_gnn'], dtype=torch.float32)
        d2_mask_gnn_tensor[i, :n_nodes_2] = torch.tensor(s['mask_2_gnn'], dtype=torch.float32)
    
    # Tanimoto 相似性 collator
    tanimoto_tensor = torch.zeros(n_samples, 1)
    for i, s in enumerate(filtered_samples):
        tanimoto_tensor[i] = s['tanimoto_similarity']
    
    return drug1_node, drug1_in_degree, drug1_out_degree, drug2_node, drug2_in_degree, drug2_out_degree, \
           target_tensor, d1_emb_tensor, d2_emb_tensor, mask_1_tensor, mask_2_tensor, \
           d1_batch_positions, d1_batch_z, d1_batch, d2_batch_positions, d2_batch_z, d2_batch, \
           (adjacency_tensor_1, node_tensor_1, edge_tensor_1), (adjacency_tensor_2, node_tensor_2, edge_tensor_2), \
           d1_expr_tensor, d2_expr_tensor, d1_morgan_tensor, d2_morgan_tensor, tanimoto_tensor, \
           d1_mask_gnn_tensor, d2_mask_gnn_tensor
