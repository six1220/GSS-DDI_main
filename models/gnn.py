import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
import copy


class GNN(nn.Module):

    # def __init__(self, **config):
    def __init__(self, args):
        super(GNN, self).__init__()
        self.args = args
        self.device = args.device
        self.node_features = args.GNN_node_features
        self.edge_features = args.GNN_edge_features
        self.message_passes = args.GNN_message_passes
        self.message_size = args.GNN_message_size
        self.msg_depth = args.GNN_message_depth
        self.msg_hidden_dim = args.GNN_message_hidden_dim
        self.att_depth = args.GNN_attention_depth
        self.att_hidden_dim = args.GNN_attention_hidden_dim
        self.gather_width = args.GNN_gather_width
        self.gather_att_depth = args.GNN_gather_attention_depth
        self.gather_att_hidden_dim = args.GNN_gather_attention_hidden_dim
        self.gather_emb_depth = args.GNN_gather_embedding_depth
        self.gather_emb_hidden_dim = args.GNN_gather_embedding_hidden_dim
        self.out_depth = args.GNN_out_depth
        self.out_hidden_dim = args.GNN_out_hidden_dim
        self.out_layer_shrinkage = args.GNN_out_layer_shrinkage

        self.out_fts = args.GNN_out_features

        self.msg_nns_1 = nn.ModuleList()
        self.att_nns_1 = nn.ModuleList()
        self.msg_nns_2 = nn.ModuleList()
        self.att_nns_2 = nn.ModuleList()
        self.dropout = args.GNN_dropout_rate
        for _ in range(self.edge_features):
            self.msg_nns_1.append(
                FeedForwardNetwork(self.node_features, [self.msg_hidden_dim] * self.msg_depth, self.message_size,
                                   dropout_p=self.dropout, bias=False))  # 一堆linear层
            self.att_nns_1.append(
                FeedForwardNetwork(self.node_features, [self.att_hidden_dim] * self.att_depth, self.message_size,
                                   dropout_p=self.dropout, bias=False))
            self.msg_nns_2.append(
                FeedForwardNetwork(self.node_features, [self.msg_hidden_dim] * self.msg_depth, self.message_size,
                                   dropout_p=self.dropout, bias=False))
            self.att_nns_2.append(
                FeedForwardNetwork(self.node_features, [self.att_hidden_dim] * self.att_depth, self.message_size,
                                   dropout_p=self.dropout, bias=False))

        self.gru_1 = nn.GRUCell(input_size=self.message_size, hidden_size=self.node_features, bias=False)
        self.gru_2 = nn.GRUCell(input_size=self.message_size, hidden_size=self.node_features, bias=False)

        self.gather_1 = GraphGather(
            self.node_features, self.gather_width,
            self.gather_att_depth, self.gather_att_hidden_dim, self.dropout,
            self.gather_emb_depth, self.gather_emb_hidden_dim, self.dropout)
        self.gather_2 = GraphGather(
            self.node_features, self.gather_width,
            self.gather_att_depth, self.gather_att_hidden_dim, self.dropout,
            self.gather_emb_depth, self.gather_emb_hidden_dim, self.dropout)

        out_layer_sizes = [round(self.out_hidden_dim * (self.out_layer_shrinkage ** (i / (self.out_depth - 1 + 1e-9))))
                           for i in range(self.out_depth)]
        # example: depth 5, dim 50, shrinkage 0.5 => out_layer_sizes [50, 42, 35, 30, 25]
        self.out_nn = FeedForwardNetwork(self.gather_width * 2, out_layer_sizes, self.out_fts, dropout_p=self.dropout)

    def aggregate_message_1(self, nodes, node_neighbours, edges, node_neighbour_mask):
        energy_mask = (node_neighbour_mask == 0).float() * 1e6
        embeddings_masked_per_edge = [
            edges[:, :, i].unsqueeze(-1) * self.msg_nns_1[i](node_neighbours) for i in range(self.edge_features)
        ]  # 对于每个边的类型，均与其邻居相乘。如果边属于该类型，则该类型embed为邻居特征的变换，否则为0；边共四个类型，则得出4个张量，每个张量383(节点数)*4(最大邻居数)*25(线性变换后维度)；对不同类型边使用不同的网络参数加以区分
        embedding = sum(embeddings_masked_per_edge)  # 把不同边类型emb加一起 得出节点邻居emb
        energies_masked_per_edge = [
            edges[:, :, i].unsqueeze(-1) * self.att_nns_1[i](node_neighbours) for i in range(self.edge_features)
        ]  # 同上，此时将其视作注意力权重？
        energies = sum(energies_masked_per_edge) - energy_mask.unsqueeze(-1)  # 将不存在的边权重设置为无穷负数
        attention = torch.softmax(energies, dim=1)  #
        return torch.sum(attention * embedding, dim=1)  # 加和成为节点emb，聚合完成

    def aggregate_message_2(self, nodes, node_neighbours, edges, node_neighbour_mask):

        energy_mask = ((node_neighbour_mask == 0).float() * 1e6)
        embeddings_masked_per_edge = [
            edges[:, :, i].unsqueeze(-1) * self.msg_nns_2[i](node_neighbours) for i in range(self.edge_features)
        ]
        embedding = sum(embeddings_masked_per_edge)
        energies_masked_per_edge = [
            edges[:, :, i].unsqueeze(-1) * self.att_nns_2[i](node_neighbours) for i in range(self.edge_features)
        ]
        energies = (sum(energies_masked_per_edge) - energy_mask.unsqueeze(-1))
        attention = torch.softmax(energies, dim=1)
        return torch.sum(attention * embedding, dim=1)

    # inputs are "batches" of shape (maximum number of nodes in batch, number of features)
    def update_1(self, nodes, messages):
        return self.gru_1(messages)

    def readout_1(self, hidden_nodes, input_nodes, node_mask):
        graph_embeddings, nodes_embeddings = self.gather_1(hidden_nodes, input_nodes, node_mask)
        return graph_embeddings, nodes_embeddings

    def update_2(self, nodes, messages):
        return self.gru_2(messages)

    def readout_2(self, hidden_nodes, input_nodes, node_mask):
        graph_embeddings = self.gather_2(hidden_nodes, input_nodes, node_mask)
        return graph_embeddings

    def readout(self, input_nodes, node_mask):
        graph_embeddings = []
        for i in range(input_nodes.shape[0]):
            emb = input_nodes[i][0]
            for j in range(input_nodes.shape[1] - 1):
                emb = torch.cat([emb, input_nodes[i][j + 1]], dim=0)

            emb = emb.detach().numpy()
            graph_embeddings.append(emb)
        graph_embeddings = np.array(graph_embeddings)
        graph_embeddings = torch.from_numpy(graph_embeddings)

        return graph_embeddings

    def final_layer(self, connected_vector):
        return self.out_nn(connected_vector)

    def forward(self, adj_1, nd_1, ed_1, adj_2, nd_2, ed_2):

        edge_batch_batch_indices_1, edge_batch_node_indices_1, edge_batch_neighbour_indices_1 = adj_1.nonzero().unbind(
            -1)  # 拆成稀疏矩阵
        node_batch_batch_indices_1, node_batch_node_indices_1 = adj_1.sum(-1).nonzero().unbind(-1)
        node_batch_adj_1 = adj_1[node_batch_batch_indices_1, node_batch_node_indices_1, :]
        node_batch_size_1 = node_batch_batch_indices_1.shape[0]

        node_degrees_1 = node_batch_adj_1.sum(-1).long()  # 获得节点度

        max_node_degree_1 = node_degrees_1.max()
        node_batch_node_neighbours_1 = torch.zeros(node_batch_size_1, max_node_degree_1, self.node_features).to(self.device)  # 用于存储节点邻居特征
        node_batch_edges_1 = torch.zeros(node_batch_size_1, max_node_degree_1, self.edge_features).to(self.device)  # 用于存储节点边

        node_batch_neighbour_neighbour_indices_1 = torch.cat([torch.arange(i) for i in node_degrees_1]).to(self.device)  # 指示掩码列在哪里不掩
        edge_batch_node_batch_indices_1 = torch.cat(
            [i * torch.ones(degree) for i, degree in enumerate(node_degrees_1)]
        ).long().to(self.device)  # 指示上面的指示在第几行 反正用来做掩码
        node_batch_node_neighbour_mask_1 = torch.zeros(node_batch_size_1, max_node_degree_1).to(self.device)

        edge_batch_batch_indices_2, edge_batch_node_indices_2, edge_batch_neighbour_indices_2 = adj_2.nonzero().unbind(
            -1)
        node_batch_batch_indices_2, node_batch_node_indices_2 = adj_2.sum(-1).nonzero().unbind(-1)
        node_batch_adj_2 = adj_2[node_batch_batch_indices_2, node_batch_node_indices_2, :]
        node_batch_size_2 = node_batch_batch_indices_2.shape[0]
        node_degrees_2 = node_batch_adj_2.sum(-1).long()
        max_node_degree_2 = node_degrees_2.max()
        node_batch_node_neighbours_2 = torch.zeros(node_batch_size_2, max_node_degree_2, self.node_features).to(self.device)
        node_batch_edges_2 = torch.zeros(node_batch_size_2, max_node_degree_2, self.edge_features).to(self.device)
        node_batch_neighbour_neighbour_indices_2 = torch.cat([torch.arange(i) for i in node_degrees_2]).to(self.device)
        edge_batch_node_batch_indices_2 = torch.cat(
            [i * torch.ones(degree) for i, degree in enumerate(node_degrees_2)]
        ).long().to(self.device)
        node_batch_node_neighbour_mask_2 = torch.zeros(node_batch_size_2, max_node_degree_2).to(self.device)

        #
        node_batch_node_neighbour_mask_1[edge_batch_node_batch_indices_1, node_batch_neighbour_neighbour_indices_1] = 1
        node_batch_edges_1[edge_batch_node_batch_indices_1, node_batch_neighbour_neighbour_indices_1, :] = \
            ed_1[edge_batch_batch_indices_1, edge_batch_node_indices_1, edge_batch_neighbour_indices_1, :]  # 填充边特征矩阵
        hidden_nodes_1 = nd_1.clone()

        node_batch_node_neighbour_mask_2[edge_batch_node_batch_indices_2, node_batch_neighbour_neighbour_indices_2] = 1
        node_batch_edges_2[edge_batch_node_batch_indices_2, node_batch_neighbour_neighbour_indices_2, :] = \
            ed_2[edge_batch_batch_indices_2, edge_batch_node_indices_2, edge_batch_neighbour_indices_2, :]
        hidden_nodes_2 = nd_2.clone()

        for i in range(self.message_passes):
            node_batch_nodes_1 = hidden_nodes_1[node_batch_batch_indices_1, node_batch_node_indices_1,
                                 :]  # 将16个图所有节点特征展开
            node_batch_node_neighbours_1[edge_batch_node_batch_indices_1, node_batch_neighbour_neighbour_indices_1, :] = \
                hidden_nodes_1[edge_batch_batch_indices_1, edge_batch_neighbour_indices_1, :]
            messages_1 = self.aggregate_message_1(
                node_batch_nodes_1, node_batch_node_neighbours_1.clone(), node_batch_edges_1,
                node_batch_node_neighbour_mask_1
            )  # 公式3 得到m 表示节点emb
            hidden_nodes_1[node_batch_batch_indices_1, node_batch_node_indices_1, :] = self.update_1(
                node_batch_nodes_1, messages_1)  # 得到经gru更新后的emb 公式4

            node_batch_nodes_2 = hidden_nodes_2[node_batch_batch_indices_2, node_batch_node_indices_2, :]
            node_batch_node_neighbours_2[edge_batch_node_batch_indices_2, node_batch_neighbour_neighbour_indices_2, :] = \
                hidden_nodes_2[edge_batch_batch_indices_2, edge_batch_neighbour_indices_2, :]
            messages_2 = self.aggregate_message_2(
                node_batch_nodes_2, node_batch_node_neighbours_2.clone(), node_batch_edges_2,
                node_batch_node_neighbour_mask_2
            )
            hidden_nodes_2[node_batch_batch_indices_2, node_batch_node_indices_2, :] = self.update_2(node_batch_nodes_2,
                                                                                                     messages_2)

        node_mask_1 = (adj_1.sum(-1) != 0)  # 找到每张图存在的节点
        graph_embedding1, node_embedding1 = self.readout_1(hidden_nodes_1, nd_1, node_mask_1)  # 公式5 & 公式6 得到2D graph的输出
        node_mask_2 = (adj_2.sum(-1) != 0)
        graph_embedding2, node_embedding2 = self.readout_2(hidden_nodes_2, nd_2, node_mask_2)

        return graph_embedding1, node_embedding1, graph_embedding2, node_embedding2


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_features, hidden_layer_sizes, out_features, activation='SELU', bias=False, dropout_p=0.0):
        super(FeedForwardNetwork, self).__init__()

        if activation == 'SELU':
            Activation = nn.SELU
            Dropout = nn.AlphaDropout
            init_constant = 1.0
        elif activation == 'ReLU':
            Activation = nn.ReLU
            Dropout = nn.Dropout
            init_constant = 2.0
        layer_sizes = [in_features] + hidden_layer_sizes + [out_features]
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(Dropout(dropout_p))
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias))
            layers.append(Activation())

        layers.append(Dropout(dropout_p))
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1], bias))

        self.seq = nn.Sequential(*layers)
        count = 0
        for i in range(1, len(layers), 3):
            nn.init.normal_(layers[i].weight, std=math.sqrt(init_constant / layers[i].weight.size(1)))
            count += 1

    def forward(self, input):
        return self.seq(input)

    # I'm probably *supposed to* override extra_repr but then self.seq (unreadable) will be printed too
    def __repr__(self):
        ffnn = type(self).__name__
        in_features = self.seq[1].in_features
        hidden_layer_sizes = [linear.out_features for linear in self.seq[1:-1:3]]
        out_features = self.seq[-1].out_features
        if len(self.seq) > 2:
            activation = str(self.seq[2])
        else:
            activation = 'None'
        bias = self.seq[1].bias is not None
        dropout_p = self.seq[0].p
        return '{}(in_features={}, hidden_layer_sizes={}, out_features={}, activation={}, bias={}, dropout_p={})'.format(
            ffnn, in_features, hidden_layer_sizes, out_features, activation, bias, dropout_p
        )


class GraphGather(nn.Module):
    def __init__(self, node_features, out_features,
                 att_depth=2, att_hidden_dim=100, att_dropout_p=0.0,
                 emb_depth=2, emb_hidden_dim=100, emb_dropout_p=0.0):
        super(GraphGather, self).__init__()

        # denoted i and j in GGNN, MPNN and PotentialNet papers
        self.att_nn = FeedForwardNetwork(
            node_features * 2, [att_hidden_dim] * att_depth, out_features, dropout_p=att_dropout_p, bias=False)
        self.emb_nn = FeedForwardNetwork(
            node_features, [emb_hidden_dim] * emb_depth, out_features, dropout_p=emb_dropout_p, bias=False)

    def forward(self, hidden_nodes, input_nodes, node_mask):
        cat = torch.cat([hidden_nodes, input_nodes], dim=2)

        energy_mask = (node_mask == 0).float() * 1e6
        energies = self.att_nn(cat) - energy_mask.unsqueeze(-1)
        attention = torch.sigmoid(energies)
        embedding = self.emb_nn(hidden_nodes)
        return torch.sum(attention * embedding, dim=1), attention * embedding  # 跟聚合节点信息时一样，都是有个注意力参数
