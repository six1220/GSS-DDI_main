from __future__ import print_function
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from models.AMDE import AMDE
from models.GNN import GNN
from models.SchNet import SchNet
from models.cross_att import Cross_MultiAttention
from models.my_molormer import Molormer
from models.gene_expr_encoder import GeneExpressionEncoder
from SRR_Modules import SubstructureExtractor, BNOptimizer, MorganFingerprintCalculator, SubstructureImportanceScorer, DMPNNSubstructureScorer

torch.manual_seed(1)
np.random.seed(1)


class BilinearDecoder(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0):
        super(BilinearDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.relation = Parameter(torch.FloatTensor(input_dim, input_dim))
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.relation.data)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs_row = inputs
        inputs_col = inputs.transpose(0, 1)
        inputs_row = self.dropout(inputs_row)
        inputs_col = self.dropout(inputs_col)
        intermediate_product = torch.mm(inputs_row, self.relation)
        rec = torch.mm(intermediate_product, inputs_col)
        rec = nn.ReLU(True)(rec)
        n = rec.size(0)
        # print(n)
        rec = nn.BatchNorm1d(n).cuda()(rec)
        outputs = nn.Linear(n, 1).cuda()(rec)
        print('outputs: ', outputs)
        return outputs


class MultiLevelDDI(nn.Module):
    def __init__(self, args, batch_size=None):
        super(MultiLevelDDI, self).__init__()
        self.args = args
        self.batch_size = batch_size if batch_size else args.batch_size
        self.hidden_dim = args.conv_hidden_dim
        self.input_dropout = nn.Dropout(args.input_dropout_rate)
        self.icnn = nn.Conv1d(self.hidden_dim, 16, 3, 3)
        self.ilin = nn.Linear(1360, 336)
        # self.decoder = BilinearDecoder(539)
        self.decoder_input_dim = 0

        if args.use_SCH:
            self.decoder_input_dim += args.SCH_out_channels
            self.sch1 = SchNet(energy_and_force=False, cutoff=args.SCH_cutoff, num_layers=args.SCH_num_layers,
                               hidden_channels=args.SCH_hidden_channels, num_filters=args.SCH_num_filters,
                               num_gaussians=args.SCH_num_gaussians, out_channels=args.SCH_out_channels)
            self.sch2 = SchNet(energy_and_force=False, cutoff=args.SCH_cutoff, num_layers=args.SCH_num_layers,
                               hidden_channels=args.SCH_hidden_channels, num_filters=args.SCH_num_filters,
                               num_gaussians=args.SCH_num_gaussians, out_channels=args.SCH_out_channels)
        if args.use_AMDE:
            self.decoder_input_dim += args.AMDE_out_dim
            self.amde = AMDE(args)
        if args.use_Mol:
            self.decoder_input_dim += 336
            self.molor = Molormer(args)  #
        if args.use_GNN:
            self.decoder_input_dim += args.GNN_gather_width
            self.gnn = GNN(args)
            # 使用GNN的gather_width作为隐藏维度，确保与节点嵌入维度匹配
            self.gnn_hidden_dim = args.GNN_gather_width
            print(f"Initializing SubstructureImportanceScorer with hidden_dim={self.gnn_hidden_dim}")
            # 子结构重要性得分模块
            self.substructure_scorer = SubstructureImportanceScorer(self.gnn_hidden_dim)
            # DMPNN 子结构得分模块
            self.dmpnn_scorer = DMPNNSubstructureScorer(
                edge_dim=6,  # 假设边特征维度为6
                n_feats=self.gnn_hidden_dim,
                n_iter=3
            )

        # 基因表达特征模块
        if hasattr(args, 'use_gene_expr') and args.use_gene_expr:
            self.gene_expr_encoder = GeneExpressionEncoder(
                input_dim=978,
                hidden_dim=args.gene_expr_hidden_dim,
                output_dim=args.gene_expr_out_dim,
                dropout=getattr(args, 'gene_expr_dropout', 0.1)
            )
        else:
            self.gene_expr_encoder = None

        # Morgan 指纹通道
        self.use_Morgan = getattr(args, 'use_Morgan', True)
        if self.use_Morgan:
            self.decoder_input_dim += 2048  # Morgan 指纹维度

        # Tanimoto 相似性通道
        self.use_Tanimoto = getattr(args, 'use_Tanimoto', True)
        if self.use_Tanimoto:
            self.decoder_input_dim += 1  # Tanimoto 相似性维度

        # 计算模块数量，包括 Morgan 指纹和 Tanimoto 相似性通道
        # 基因表达数据已经在 AMDE 模块中拼接，不作为单独的通道
        self.mod_num = int(args.use_SCH) + int(args.use_Mol) + int(args.use_AMDE) + int(args.use_GNN)
        if self.use_Morgan:
            self.mod_num += 1  # Morgan 指纹通道
        if self.use_Tanimoto:
            self.mod_num += 1  # Tanimoto 相似性通道

        if args.use_cross_attention:
            assert args.use_SCH or args.use_Mol or args.use_GNN

            # 对于 no3D 训练的模型，使用与训练时一致的维度
            if args.aug_type == 'subgraph' and not args.use_SCH:
                # 训练时使用的是256维度，与Mol_hidden_dim保持一致
                q_dim = args.Mol_hidden_dim if args.use_Mol else 256
                kv_dim = args.Mol_hidden_dim if args.use_Mol else 256
                self.inner_cross = Cross_MultiAttention(q_dim, kv_dim, args.cross_out_dim, args.cross_inner_num_head)
            elif self.mod_num == 1 or (self.mod_num == 2 and args.use_AMDE):
                qkv_dim = args.SCH_out_channels if args.use_SCH else args.Mol_hidden_dim if args.use_Mol else args.GNN_gather_width if args.use_GNN else 256
                self.inner_cross = Cross_MultiAttention(qkv_dim, qkv_dim, args.cross_out_dim, args.cross_inner_num_head)
            else:
                # 动态设置q_dim和kv_dim，确保与实际输入特征维度匹配
                if args.use_SCH:
                    # sch在前
                    q_dim = args.SCH_out_channels
                    kv_dim = args.GNN_gather_width if args.use_GNN else 256
                else:
                    # 当use_SCH=False时，使用Mol或GNN的特征维度
                    q_dim = args.Mol_hidden_dim if args.use_Mol else args.GNN_gather_width if args.use_GNN else 256
                    kv_dim = args.Mol_hidden_dim if args.use_Mol else args.GNN_gather_width if args.use_GNN else 256

                # mol在前
                # kv_dim = args.SCH_out_channels
                # q_dim = args.GNN_gather_width if args.use_GNN else 256
                self.inner_cross = Cross_MultiAttention(q_dim, kv_dim, args.cross_out_dim, args.cross_inner_num_head)

            self.outer_cross = Cross_MultiAttention(args.cross_out_dim, args.cross_out_dim, args.cross_out_dim, args.cross_outer_num_head)
            # 在 cross attention 模式下，decoder_input_dim 包含 cross attention 输出、AMDE（如果有）、Morgan 指纹和 Tanimoto 相似性
            # 基因表达数据已经在 AMDE 模块中拼接
            self.decoder_input_dim = args.cross_out_dim + (args.AMDE_out_dim if args.use_AMDE else 0)
            # 添加 Morgan 指纹和 Tanimoto 相似性通道
            if self.use_Morgan:
                self.decoder_input_dim += 2048  # Morgan 指纹维度
            if self.use_Tanimoto:
                self.decoder_input_dim += 1  # Tanimoto 相似性维度
            self.mod_num = 2 if args.use_AMDE else 1

        # self.decoder = nn.Sequential(
        #     nn.Linear(self.decoder_input_dim, 128),
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(128),
        #     nn.Linear(128, 32),
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(32),
        #     nn.Linear(32, 1)
        # )
        self.decoder = nn.Sequential(
                # nn.Linear(self.flatten_dim, 512),
                nn.Linear(self.decoder_input_dim, 512),
                # nn.Linear(539, 256),
                nn.ReLU(True),
                nn.BatchNorm1d(512),
                nn.Linear(512, 256),
                nn.ReLU(True),
                nn.BatchNorm1d(256),
                nn.Linear(256, args.num_class)
            )
        # 权重归一化
        self.w = nn.Parameter(F.softmax(torch.ones(self.mod_num), dim=0))
        
        # 添加 pass_smiles 属性并设置为 0
        self.pass_smiles = 0


    def padding_sch_result(self, batch, batch_atoms):
        """
        The schNet's result is [all_batch_atoms, atom_dim], this method change it to [batch_size, largest_num_atom, atom_dim].
        If atom len less than largest number of atom in mol, padding it by torch.zeros()
        :param adj: each adjacency matrix in batch, like [batch_size, atom_num, atom_num]. This method will abandon the atom that degree is zero
        :param batch_atoms: schNet result, like [all_batch_atoms, atom_dim]
        :return: changed result, like [batch_size, most_atom_in_mol, atom_dim]
        """
        # 使用实际的批次大小
        actual_batch_size = len(torch.unique(batch))
        each_node_num = torch.bincount(batch)
        all_node_num = each_node_num.sum(-1)
        assert all_node_num == batch_atoms.shape[0]
        output = torch.zeros((actual_batch_size, self.args.max_mol_len, batch_atoms.shape[-1])).to(self.args.device)
        cnt = 0
        for i in range(actual_batch_size):
            # 确保索引不越界
            if i < len(each_node_num):
                output[i, :each_node_num[i], :] = batch_atoms[cnt: cnt + each_node_num[i], :]
                cnt += each_node_num[i]
        return output

    def padding_GNN_result(self, batch_atoms):
        """
        :param batch_atoms: GNN's or molormer's result, like [batch_size, batch_largest_num_atom, atom_dim]
        :return: changed result, like [batch_size, most_atom_in_mol, atom_dim]
        """
        # 使用实际的批次大小
        actual_batch_size = batch_atoms.shape[0]
        input_atom_num = batch_atoms.shape[1]
        output = torch.zeros((actual_batch_size, self.args.max_mol_len, batch_atoms.shape[-1])).to(self.args.device)
        for i in range(actual_batch_size):
            output[i, :input_atom_num, :] = batch_atoms[i, :, :]
        return output

    def padding_adj(self, batch_adjs):
        """
        :param batch_adjs: each adjacency matrix in batch, like [batch_size, atom_num, atom_num].
        :return: pad result, like [batch_size, most_atom_in_mol, most_atom_in_mol]
        """
        # 使用实际的批次大小
        actual_batch_size = batch_adjs.shape[0]
        input_atom_num = batch_adjs.shape[1]
        output = torch.zeros((actual_batch_size, self.args.max_mol_len, self.args.max_mol_len)).to(self.args.device)
        for i in range(actual_batch_size):
            output[i, :input_atom_num, :batch_adjs.shape[-1]] = batch_adjs[i, :, :]
        return output

    def padding_mask(self, batch1, batch2):
        """
         :param batch_adjs2: each adjacency matrix in batch, like [batch_size, atom_num, atom_num].
         :param batch_adjs1:
         :return: pad mask, like [batch_size, most_atom_in_mol, most_atom_in_mol].
         """
        # 使用实际的批次大小
        actual_batch_size = len(torch.unique(batch1))
        each_node_num_1 = torch.bincount(batch1)
        each_node_num_2 = torch.bincount(batch2)
        output = torch.zeros((actual_batch_size, self.args.max_mol_len, self.args.max_mol_len)).to(self.args.device)
        for i in range(actual_batch_size):
            # 确保索引不越界
            if i < len(each_node_num_1) and i < len(each_node_num_2):
                output[i, :each_node_num_1[i], :each_node_num_2[i]] = torch.ones([each_node_num_1[i], each_node_num_2[i]])
        return output

    def padding_col(self, batch):
        # 使用实际的批次大小
        actual_batch_size = len(torch.unique(batch))
        each_node_num = torch.bincount(batch)
        output = torch.zeros((actual_batch_size, self.args.max_mol_len, self.args.max_mol_len)).to(self.args.device)
        cnt = 0
        for i in range(actual_batch_size):
            # 确保索引不越界
            if i < len(each_node_num):
                output[i, :, :each_node_num[i]] = torch.ones([self.args.max_mol_len, each_node_num[i]])
                cnt += each_node_num[i]
        return output

    def forward(self, d1_node, d1_in_degree, d1_out_degree, d2_node, d2_in_degree, d2_out_degree,
                d1, d2, mask_1, mask_2, d1_positions, d1_z, d1_batch, d2_positions, d2_z, d2_batch,
                adj_1, nd_1, ed_1, adj_2, nd_2, ed_2, d1_expr=None, d2_expr=None, d1_morgan=None, d2_morgan=None, tanimoto_similarity=None,
                mask_1_gnn=None, mask_2_gnn=None
                ):

        # 确保 use_Morgan 和 use_Tanimoto 属性存在
        if not hasattr(self, 'use_Morgan'):
            self.use_Morgan = getattr(self.args, 'use_Morgan', True)
        if not hasattr(self, 'use_Tanimoto'):
            self.use_Tanimoto = getattr(self.args, 'use_Tanimoto', True)

        drug1_n_graph = d1_node.size()[0]
        d1_mol_features, d1_atom_features = [], []
        d2_mol_features, d2_atom_features = [], []
        cnt_mod = 0

        # 基因表达特征处理
        d1_expr_feat, d2_expr_feat = None, None
        if self.gene_expr_encoder is not None:
            if d1_expr is not None and d2_expr is not None:
                # 确保基因表达特征在正确的设备上
                if d1_expr.device != self.args.device:
                    d1_expr = d1_expr.to(self.args.device)
                if d2_expr.device != self.args.device:
                    d2_expr = d2_expr.to(self.args.device)
                
                d1_expr_feat = self.gene_expr_encoder(d1_expr)  # (batch_size, gene_expr_out_dim)
                d2_expr_feat = self.gene_expr_encoder(d2_expr)  # (batch_size, gene_expr_out_dim)
            else:
                # 如果没有提供基因表达特征，使用零向量
                d1_expr_feat = torch.zeros((drug1_n_graph, self.args.gene_expr_out_dim)).to(self.args.device)
                d2_expr_feat = torch.zeros((drug1_n_graph, self.args.gene_expr_out_dim)).to(self.args.device)

        # sequence-based channel
        if self.args.use_AMDE:
            # 将基因表达数据传递给 AMDE 模块
            d1_seq_fts_layer1, d2_seq_fts_layer1 = self.amde(d1, d2, mask_1, mask_2, d1_expr_feat, d2_expr_feat)

            d1_mol_features.append(self.w[cnt_mod] * d1_seq_fts_layer1)
            d2_mol_features.append(self.w[cnt_mod] * d2_seq_fts_layer1)
            cnt_mod += 1

        # 3D channel
        if self.args.use_SCH:
            (d1_3d_ft, d1_atom_ft), (d2_3d_ft, d2_atom_ft) = self.sch1((d1_z, d1_positions, d1_batch)), self.sch2((d2_z, d2_positions, d2_batch))
            if not self.args.use_cross_attention:
                d1_mol_features.append(self.w[cnt_mod] * d1_3d_ft)
                d2_mol_features.append(self.w[cnt_mod] * d2_3d_ft)
                cnt_mod += 1
            d1_atom_ft = self.padding_sch_result(d1_batch, d1_atom_ft)
            d2_atom_ft = self.padding_sch_result(d2_batch, d2_atom_ft)
            d1_atom_features.append(d1_atom_ft)
            d2_atom_features.append(d2_atom_ft)

        # semantic info-based channel
        if self.args.use_Mol:
            molor_d1_feature, molor_d2_feature = self.molor(d1_node, d1_in_degree, d1_out_degree,
                                                            d2_node, d2_in_degree, d2_out_degree)
            if not self.args.use_cross_attention:
                molor_d1_feature = self.icnn(molor_d1_feature.permute(0, 2, 1))  # [b, 16, 21])
                d1_feature = self.ilin(molor_d1_feature.view(drug1_n_graph, -1))  # [b, 336]
                molor_d2_feature = self.icnn(molor_d2_feature.permute(0, 2, 1))
                d2_feature = self.ilin(molor_d2_feature.view(drug1_n_graph, -1))
                d1_mol_features.append(self.w[cnt_mod] * d1_feature)
                d2_mol_features.append(self.w[cnt_mod] * d2_feature)
                cnt_mod += 1
            else:
                d1_atom_features.append(molor_d1_feature)
                d2_atom_features.append(molor_d2_feature)

        # 初始化子结构重要性得分变量
        substructure_importance_1 = None
        substructure_importance_2 = None
        dmpnn_importance_1 = None
        dmpnn_importance_2 = None

        if self.args.use_GNN:
            graph_embedding1, node_embedding1, graph_embedding2, node_embedding2 = self.gnn(adj_1, nd_1, ed_1, adj_2, nd_2, ed_2)
            
            # 使用 SRR 模块计算子结构重要性得分
            if hasattr(self, 'substructure_scorer'):
                # 计算子结构重要性得分
                batch_size1 = node_embedding1.size(0)
                max_nodes1 = node_embedding1.size(1)
                batch_size2 = node_embedding2.size(0)
                max_nodes2 = node_embedding2.size(1)
                
                # 为每个节点创建批次索引
                batch_indices1 = torch.arange(batch_size1).unsqueeze(1).repeat(1, max_nodes1).flatten().to(self.args.device)
                batch_indices2 = torch.arange(batch_size2).unsqueeze(1).repeat(1, max_nodes2).flatten().to(self.args.device)
                
                # 计算子结构重要性得分
                gx1, scores1 = self.substructure_scorer(node_embedding1.flatten(0, 1), None, batch_indices1)
                gx2, scores2 = self.substructure_scorer(node_embedding2.flatten(0, 1), None, batch_indices2)
                
                # 将得分重新调整为批次格式
                scores1 = scores1.view(batch_size1, max_nodes1)
                scores2 = scores2.view(batch_size2, max_nodes2)
                
                # 使用 GNN 掩码特征
                if mask_1_gnn is not None:
                    # 确保 mask_1_gnn 在正确的设备上
                    if mask_1_gnn.device != self.args.device:
                        mask_1_gnn = mask_1_gnn.to(self.args.device)
                    if mask_1_gnn.dim() == 2:
                        # 确保掩码形状与得分形状匹配
                        if mask_1_gnn.size(1) < max_nodes1:
                            # 填充掩码
                            padding = torch.ones(batch_size1, max_nodes1 - mask_1_gnn.size(1)).to(self.args.device)
                            mask_1_gnn = torch.cat([mask_1_gnn, padding], dim=1)
                        elif mask_1_gnn.size(1) > max_nodes1:
                            # 截断掩码
                            mask_1_gnn = mask_1_gnn[:, :max_nodes1]
                        # 应用掩码
                        scores1 = scores1 * mask_1_gnn
                
                if mask_2_gnn is not None:
                    # 确保 mask_2_gnn 在正确的设备上
                    if mask_2_gnn.device != self.args.device:
                        mask_2_gnn = mask_2_gnn.to(self.args.device)
                    if mask_2_gnn.dim() == 2:
                        # 确保掩码形状与得分形状匹配
                        if mask_2_gnn.size(1) < max_nodes2:
                            # 填充掩码
                            padding = torch.ones(batch_size2, max_nodes2 - mask_2_gnn.size(1)).to(self.args.device)
                            mask_2_gnn = torch.cat([mask_2_gnn, padding], dim=1)
                        elif mask_2_gnn.size(1) > max_nodes2:
                            # 截断掩码
                            mask_2_gnn = mask_2_gnn[:, :max_nodes2]
                        # 应用掩码
                        scores2 = scores2 * mask_2_gnn
                
                # 保存子结构重要性得分
                substructure_importance_1 = scores1
                substructure_importance_2 = scores2
                
                # 使用得分加权节点嵌入
                node_embedding1 = node_embedding1 * scores1.unsqueeze(-1)
                node_embedding2 = node_embedding2 * scores2.unsqueeze(-1)
            
            # 使用 DMPNN 子结构得分模块
            if hasattr(self, 'dmpnn_scorer'):
                # 计算 DMPNN 子结构重要性得分
                try:
                    # DMPNN 计算暂时禁用，因为需要边索引而不是邻接矩阵
                    # 后续需要修改数据处理代码，将邻接矩阵转换为边索引
                    pass
                except Exception as e:
                    # 如果 DMPNN 计算失败，跳过
                    print(f"DMPNN 计算失败: {e}")
                    dmpnn_importance_1 = None
                    dmpnn_importance_2 = None
            
            if not self.args.use_cross_attention:
                d1_mol_features.append(self.w[cnt_mod] * graph_embedding1)
                d2_mol_features.append(self.w[cnt_mod] * graph_embedding2)
                cnt_mod += 1
            node_embedding1 = self.padding_GNN_result(node_embedding1)
            node_embedding2 = self.padding_GNN_result(node_embedding2)
            d1_atom_features.append(node_embedding1)
            d2_atom_features.append(node_embedding2)

        # 保存当前 cnt_mod 值，用于 cross attention 模式下的权重索引
        original_cnt_mod = cnt_mod

        # Morgan 指纹通道
        if self.use_Morgan and d1_morgan is not None and d2_morgan is not None:
            # 确保 Morgan 指纹在正确的设备上
            if d1_morgan.device != self.args.device:
                d1_morgan = d1_morgan.to(self.args.device)
            if d2_morgan.device != self.args.device:
                d2_morgan = d2_morgan.to(self.args.device)
            
            # 确保 Morgan 指纹的维度正确
            if d1_morgan.dim() == 1:
                d1_morgan = d1_morgan.unsqueeze(0)
            if d2_morgan.dim() == 1:
                d2_morgan = d2_morgan.unsqueeze(0)
            
            # 将 Morgan 指纹添加到分子特征中
            if not self.args.use_cross_attention:
                d1_mol_features.append(self.w[cnt_mod] * d1_morgan)
                d2_mol_features.append(self.w[cnt_mod] * d2_morgan)
                cnt_mod += 1
            else:
                # 在 cross attention 模式下，直接添加特征，不使用权重
                d1_mol_features.append(d1_morgan)
                d2_mol_features.append(d2_morgan)
        
        # Tanimoto 相似性通道
        if self.use_Tanimoto and tanimoto_similarity is not None:
            # 确保 Tanimoto 相似性在正确的设备上
            if tanimoto_similarity.device != self.args.device:
                tanimoto_similarity = tanimoto_similarity.to(self.args.device)
            
            # 确保 Tanimoto 相似性的维度正确
            if tanimoto_similarity.dim() == 0:
                tanimoto_similarity = tanimoto_similarity.unsqueeze(0).unsqueeze(1)
            elif tanimoto_similarity.dim() == 1:
                tanimoto_similarity = tanimoto_similarity.unsqueeze(1)
            
            # 将 Tanimoto 相似性添加到分子特征中
            # 由于相似性是成对的，我们将其同时添加到两个药物的特征中
            if not self.args.use_cross_attention:
                d1_mol_features.append(self.w[cnt_mod] * tanimoto_similarity)
                d2_mol_features.append(self.w[cnt_mod] * tanimoto_similarity)
                cnt_mod += 1
            else:
                # 在 cross attention 模式下，直接添加特征，不使用权重
                d1_mol_features.append(tanimoto_similarity)
                d2_mol_features.append(tanimoto_similarity)

        if self.args.use_cross_attention:
            # 在 cross attention 模式下，确保 cnt_mod 的值正确
            # - 如果有 AMDE：cnt_mod 应该是 1（AMDE 使用 0）
            # - 如果没有 AMDE：cnt_mod 应该是 0
            # 重置 cnt_mod 以确保正确
            if self.args.use_AMDE:
                cnt_mod = 1  # AMDE 使用 0，cross attention 将使用 1
            else:
                cnt_mod = 0  # cross attention 将使用 0
            
            # 使用实际的批次大小
            actual_batch_size = drug1_n_graph
            
            if self.args.use_Mol:
                # mol当q
                # pad_adjs_1 = self.padding_col(d1_batch)
                # pad_adjs_2 = self.padding_col(d2_batch)
                # pad_mask_1 = None
                # pad_mask_2 = None

                # sch当q  第一个掩码只掩列，第二个行列都掩
                pad_adjs_1 = torch.ones([actual_batch_size, self.args.max_mol_len, self.args.max_mol_len]).to(self.args.device)
                pad_adjs_2 = torch.ones([actual_batch_size, self.args.max_mol_len, self.args.max_mol_len]).to(self.args.device)
                pad_mask_1 = self.padding_mask(d1_batch, d2_batch)
                pad_mask_2 = self.padding_mask(d2_batch, d1_batch)
            else:
                pad_adjs_1 = self.padding_adj(adj_1)
                pad_adjs_2 = self.padding_adj(adj_2)
                pad_mask_1 = self.padding_mask(d1_batch, d2_batch)
                pad_mask_2 = self.padding_mask(d2_batch, d1_batch)
            
            # 确保 d1_atom_features 和 d2_atom_features 不为空
            if len(d1_atom_features) == 0 or len(d2_atom_features) == 0:
                # 如果没有原子特征，使用默认值
                d1_atom_ft = torch.zeros((actual_batch_size, self.args.max_mol_len, self.args.cross_out_dim)).to(self.args.device)
                d2_atom_ft = torch.zeros((actual_batch_size, self.args.max_mol_len, self.args.cross_out_dim)).to(self.args.device)
                att_w_in_1 = None
                att_w_in_2 = None
            else:
                # 确保有足够的特征进行交叉注意力计算
                if len(d1_atom_features) >= 2 and len(d2_atom_features) >= 2:
                    # 检查两个特征的维度是否相同
                    if d1_atom_features[0].shape[-1] == d1_atom_features[-1].shape[-1]:
                        # 如果维度相同，直接使用
                        d1_atom_ft, att_w_in_1 = self.inner_cross(d1_atom_features[0], d1_atom_features[-1], pad_adjs_1)
                        d2_atom_ft, att_w_in_2 = self.inner_cross(d2_atom_features[0], d2_atom_features[-1], pad_adjs_2)
                    else:
                        # 如果维度不同，使用第一个特征作为输入
                        d1_atom_ft, att_w_in_1 = self.inner_cross(d1_atom_features[0], d1_atom_features[0], pad_adjs_1)
                        d2_atom_ft, att_w_in_2 = self.inner_cross(d2_atom_features[0], d2_atom_features[0], pad_adjs_2)
                else:
                    # 如果只有一个特征，使用自身作为输入
                    d1_atom_ft, att_w_in_1 = self.inner_cross(d1_atom_features[0], d1_atom_features[0], pad_adjs_1)
                    d2_atom_ft, att_w_in_2 = self.inner_cross(d2_atom_features[0], d2_atom_features[0], pad_adjs_2)

            # fixme 存在标注错误，此处得出的为d2和d1 cross atom ft
            d1_cross_atom_ft, att_w_out_1 = self.outer_cross(d1_atom_ft, d2_atom_ft, pad_mask_1)
            d2_cross_atom_ft, att_w_out_2 = self.outer_cross(d2_atom_ft, d1_atom_ft, pad_mask_2)

            d1_cross_atom_ft += d1_atom_ft
            d2_cross_atom_ft += d2_atom_ft
            # d1_cross_atom_ft, att_w_out_1 = self.outer_cross(d2_atom_ft, d1_atom_ft, pad_mask_2)
            # d2_cross_atom_ft, att_w_out_2 = self.outer_cross(d1_atom_ft, d2_atom_ft, pad_mask_1)
            #
            # d1_cross_atom_ft += d1_atom_ft
            # d2_cross_atom_ft += d2_atom_ft

            d1_mol_features.append(self.w[cnt_mod] * torch.sum(d1_cross_atom_ft, dim=1))
            d2_mol_features.append(self.w[cnt_mod] * torch.sum(d2_cross_atom_ft, dim=1))

        d1_feature = torch.cat(d1_mol_features, dim=1)
        d2_feature = torch.cat(d2_mol_features, dim=1)  # [b, 539]
        final_fts_sum = d1_feature + d2_feature
        score = self.decoder(final_fts_sum)

        # 可解释性输出
        explanations = {
            'attention_weights': {
                'inner_cross_1': att_w_in_1 if 'att_w_in_1' in locals() else None,
                'inner_cross_2': att_w_in_2 if 'att_w_in_2' in locals() else None,
                'outer_cross_1': att_w_out_1 if 'att_w_out_1' in locals() else None,
                'outer_cross_2': att_w_out_2 if 'att_w_out_2' in locals() else None
            },
            'substructure_importance': {
                'd1_importance': substructure_importance_1 if 'substructure_importance_1' in locals() else None,
                'd2_importance': substructure_importance_2 if 'substructure_importance_2' in locals() else None,
                'd1_dmpnn_importance': dmpnn_importance_1 if 'dmpnn_importance_1' in locals() else None,
                'd2_dmpnn_importance': dmpnn_importance_2 if 'dmpnn_importance_2' in locals() else None
            },
            'features': {
                'd1_feature': d1_feature,
                'd2_feature': d2_feature,
                'final_fts_sum': final_fts_sum
            }
        }

        return score, explanations

