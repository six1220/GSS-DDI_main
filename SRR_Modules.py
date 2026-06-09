import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric.utils import degree

class SubstructureExtractor:
    """子结构提取与精修模块"""
    
    @staticmethod
    def atom_features(atom, atom_symbols, explicit_H=True, use_chirality=False):
        """提取原子特征"""
        results = []
        # 原子符号
        results.extend([atom.GetSymbol() == symbol for symbol in atom_symbols])
        results.append(atom.GetSymbol() not in atom_symbols)  # Unknown
        
        # 原子度数
        degree_values = list(range(11))
        results.extend([atom.GetDegree() == d for d in degree_values])
        
        # 隐含价态
        implicit_valence = [0, 1, 2, 3, 4, 5, 6]
        results.extend([atom.GetImplicitValence() == v for v in implicit_valence])
        
        # 形式电荷和自由基电子
        results.extend([atom.GetFormalCharge(), atom.GetNumRadicalElectrons()])
        
        # 杂化类型
        hybridization = [
            Chem.rdchem.HybridizationType.SP, 
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, 
            Chem.rdchem.HybridizationType.SP3D, 
            Chem.rdchem.HybridizationType.SP3D2
        ]
        results.extend([atom.GetHybridization() == h for h in hybridization])
        
        # 芳香性
        results.append(atom.GetIsAromatic())
        
        # 显式氢原子
        if explicit_H:
            total_hs = [0, 1, 2, 3, 4]
            results.extend([atom.GetTotalNumHs() == h for h in total_hs])
        
        # 手性
        if use_chirality:
            try:
                results.extend([atom.GetProp('_CIPCode') == 'R', atom.GetProp('_CIPCode') == 'S'])
            except:
                results.extend([False, False])
            results.append(atom.HasProp('_ChiralityPossible'))
        
        return torch.tensor(results, dtype=torch.float32)
    
    @staticmethod
    def edge_features(bond):
        """提取键特征"""
        bond_type = bond.GetBondType()
        return torch.tensor([
            bond_type == Chem.rdchem.BondType.SINGLE,
            bond_type == Chem.rdchem.BondType.DOUBLE,
            bond_type == Chem.rdchem.BondType.TRIPLE,
            bond_type == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing()
        ], dtype=torch.float32)
    
    @staticmethod
    def extract_substructures(mol, atom_symbols):
        """从分子中提取子结构特征"""
        # 提取键特征
        bonds = mol.GetBonds()
        if bonds:
            edge_list = []
            edge_feats = []
            for bond in bonds:
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                edge_list.append([begin_idx, end_idx])
                edge_list.append([end_idx, begin_idx])
                feat = SubstructureExtractor.edge_features(bond)
                edge_feats.append(feat)
                edge_feats.append(feat)
            edge_list = torch.tensor(edge_list, dtype=torch.long).T
            edge_feats = torch.stack(edge_feats)
        else:
            edge_list = torch.empty((2, 0), dtype=torch.long)
            edge_feats = torch.empty((0, 6), dtype=torch.float32)
        
        # 提取原子特征
        atoms = mol.GetAtoms()
        atom_feats = []
        for atom in atoms:
            feat = SubstructureExtractor.atom_features(atom, atom_symbols)
            atom_feats.append(feat)
        atom_feats = torch.stack(atom_feats)
        
        # 构建线图边索引
        line_graph_edge_index = torch.empty((2, 0), dtype=torch.long)
        if edge_list.nelement() != 0:
            conn = (edge_list[1].unsqueeze(1) == edge_list[0].unsqueeze(0)) & \
                   (edge_list[0].unsqueeze(1) != edge_list[1].unsqueeze(0))
            line_graph_edge_index = conn.nonzero(as_tuple=False).T
        
        return atom_feats, edge_list, edge_feats, line_graph_edge_index

class BNOptimizer(nn.Module):
    """BN 层优化模块"""
    
    def __init__(self, n_feats):
        super().__init__()
        self.snd_n_feats = 6 * n_feats
        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Linear(n_feats, self.snd_n_feats),
        )
        self.lin2 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin3 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin4 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats)
        )
        self.lin5 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, n_feats)
        )
    
    def forward(self, x):
        """前向传播"""
        x = self.lin1(x)
        x = (self.lin3(self.lin2(x)) + x) / 2
        x = (self.lin4(x) + x) / 2
        x = self.lin5(x)
        return x

class MorganFingerprintCalculator:
    """Morgan 指纹和 Tanimoto 系数计算模块"""
    
    @staticmethod
    def calculate_morgan_fingerprint(mol, radius=2, n_bits=2048):
        """计算 Morgan 指纹"""
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return fingerprint
    
    @staticmethod
    def fingerprint_to_tensor(fingerprint):
        """将指纹转换为 PyTorch 张量"""
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fingerprint, arr)
        return torch.tensor(arr, dtype=torch.float32)
    
    @staticmethod
    def calculate_tanimoto_similarity(fingerprint1, fingerprint2):
        """计算 Tanimoto 系数"""
        return DataStructs.FingerprintSimilarity(fingerprint1, fingerprint2)
    
    @staticmethod
    def batch_tanimoto_similarity(target_mol, reference_mols, radius=2, n_bits=2048):
        """批量计算 Tanimoto 系数"""
        target_fp = MorganFingerprintCalculator.calculate_morgan_fingerprint(target_mol, radius, n_bits)
        similarities = []
        for mol in reference_mols:
            ref_fp = MorganFingerprintCalculator.calculate_morgan_fingerprint(mol, radius, n_bits)
            sim = MorganFingerprintCalculator.calculate_tanimoto_similarity(target_fp, ref_fp)
            similarities.append(sim)
        return torch.tensor(similarities, dtype=torch.float32)

class SubstructureImportanceScorer(nn.Module):
    """子结构重要性得分模块"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, edge_index, batch):
        """计算子结构重要性得分"""
        x_conv = self.conv(x)
        scores = softmax(x_conv, batch, dim=0)
        gx = global_add_pool(x * scores, batch)
        return gx, scores

class DMPNNSubstructureScorer(nn.Module):
    """基于 DMPNN 的子结构重要性得分模块"""
    
    def __init__(self, edge_dim, n_feats, n_iter):
        super().__init__()
        self.n_iter = n_iter
        
        self.lin_u = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_v = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_edge = nn.Linear(edge_dim, n_feats, bias=False)
        
        self.att = SubstructureImportanceScorer(n_feats)
        self.a = nn.Parameter(torch.zeros(1, n_feats, n_iter))
        self.lin_gout = nn.Linear(n_feats, n_feats)
        self.a_bias = nn.Parameter(torch.zeros(1, 1, n_iter))
        
        # 初始化参数
        nn.init.xavier_uniform_(self.a)
        
        self.lin_block = BNOptimizer(n_feats)
    
    def forward(self, atom_feats, edge_index, edge_attr, line_graph_edge_index, edge_index_batch):
        """前向传播"""
        edge_u = self.lin_u(atom_feats)
        edge_v = self.lin_v(atom_feats)
        edge_uv = self.lin_edge(edge_attr)
        edge_attr = (edge_u[edge_index[0]] + edge_v[edge_index[1]] + edge_uv) / 3
        out = edge_attr
        
        out_list = []
        gout_list = []
        for n in range(self.n_iter):
            if line_graph_edge_index.nelement() != 0:
                out = scatter(out[line_graph_edge_index[0]], line_graph_edge_index[1], 
                             dim_size=edge_attr.size(0), dim=0, reduce='add')
            out = edge_attr + out
            gout, scores = self.att(out, line_graph_edge_index, edge_index_batch)
            out_list.append(out)
            gout_list.append(F.tanh((self.lin_gout(gout))))
        
        gout_all = torch.stack(gout_list, dim=-1)
        out_all = torch.stack(out_list, dim=-1)
        scores = (gout_all * self.a).sum(1, keepdim=True) + self.a_bias
        scores = torch.softmax(scores, dim=-1)
        scores = scores.repeat_interleave(degree(edge_index_batch, dtype=edge_index_batch.dtype), dim=0)
        out = (out_all * scores).sum(-1)
        x = atom_feats + scatter(out, edge_index[1], dim_size=atom_feats.size(0), dim=0, reduce='add')
        x = self.lin_block(x)
        
        return x, scores