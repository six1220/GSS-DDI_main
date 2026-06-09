from re import M
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
# from rdkit.Chem.Draw import IPythonConsle
from rdkit import rdBase
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import numpy as np
import torch
import os
import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})

# 用于跟踪已报告的错误文件，避免重复警告
_reported_bad_files = set()

# allowable multiple choice node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'possible_is_conjugated_list': [False, True],
}
def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    # print(allowable_features)
    # assert False
    atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
            ]
    return atom_feature

def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list']
        ]))

def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
                safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
                allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
                allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
            ]
    return bond_feature

def get_bond_feature_dims():
    return list(map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_bond_stereo_list'],
        allowable_features['possible_is_conjugated_list']
        ]))

def atom_feature_vector_to_dict(atom_feature):
    [atomic_num_idx,
    chirality_idx,
    degree_idx,
    formal_charge_idx,
    num_h_idx,
    number_radical_e_idx,
    hybridization_idx,
    is_aromatic_idx,
    is_in_ring_idx] = atom_feature

    feature_dict = {
        'atomic_num': allowable_features['possible_atomic_num_list'][atomic_num_idx],
        'chirality': allowable_features['possible_chirality_list'][chirality_idx],
        'degree': allowable_features['possible_degree_list'][degree_idx],
        'formal_charge': allowable_features['possible_formal_charge_list'][formal_charge_idx],
        'num_h': allowable_features['possible_numH_list'][num_h_idx],
        'num_rad_e': allowable_features['possible_number_radical_e_list'][number_radical_e_idx],
        'hybridization': allowable_features['possible_hybridization_list'][hybridization_idx],
        'is_aromatic': allowable_features['possible_is_aromatic_list'][is_aromatic_idx],
        'is_in_ring': allowable_features['possible_is_in_ring_list'][is_in_ring_idx]
    }

    return feature_dict

def bond_feature_vector_to_dict(bond_feature):
    [bond_type_idx,
    bond_stereo_idx,
    is_conjugated_idx] = bond_feature

    feature_dict = {
        'bond_type': allowable_features['possible_bond_type_list'][bond_type_idx],
        'bond_stereo': allowable_features['possible_bond_stereo_list'][bond_stereo_idx],
        'is_conjugated': allowable_features['possible_is_conjugated_list'][is_conjugated_idx]
    }

    return feature_dict

def sdf2graph(id, args):
    # print(id)
    # 构建 SDF 文件路径，考虑 dataset_name
    if hasattr(args, 'dataset_name') and args.dataset_name:
        drug = os.path.join(args.dataset_path, args.dataset_name, 'sdf', id + '.sdf')
    else:
        drug = os.path.join(args.dataset_path, 'sdf', id + '.sdf')
    
    # 尝试读取 SDF 文件，如果失败则返回默认空图
    try:
        mol = Chem.MolFromMolFile(drug)  # 获得药物信息
        if mol is None:
            # 只在第一次遇到该文件错误时报告
            if drug not in _reported_bad_files:
                print(f"Warning: Failed to parse SDF file {drug}, using empty graph.")
                _reported_bad_files.add(drug)
            # 返回默认空图（单个节点，无边）
            num_bond_features = 3
            x = np.array([[0] * 9], dtype=np.int64)  # 单个节点，9个特征维度
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype=np.int64)
            x = torch.tensor(x, dtype=torch.long)
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            return x, edge_attr, edge_index
    except (OSError, IOError, Exception) as e:
        # 只在第一次遇到该文件错误时报告
        if drug not in _reported_bad_files:
            print(f"Warning: Error reading SDF file {drug}: {str(e)}, using empty graph.")
            _reported_bad_files.add(drug)
        # 返回默认空图（单个节点，无边）
        num_bond_features = 3
        x = np.array([[0] * 9], dtype=np.int64)  # 单个节点，9个特征维度
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)
        x = torch.tensor(x, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        return x, edge_attr, edge_index
    
    # smi = Chem.MolToSmiles(mol,isomericSmiles=True)
    # smi='NC(CCC(O)=O)C=C'
    # mol1 = Chem.MolFromSmiles(smi)
    # mols=[]
    # for _ in range(10):
    #     smi1 = Chem.MolToSmiles(mol1, doRandom=True)
    #     print(smi1)
    #     # smi1 = Chem.MolToSmiles(mol1, isomericSmiles=True)
    # assert False
    
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))  # 又用另外一种方式获取了原子信息维度为9
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []  # 又用另外一种方式获取了键信息维度为3
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    # graph = dict()
    # graph['edge_index'] = edge_index
    # graph['edge_feat'] = edge_attr                  # dim = 3
    # graph['node_feat'] = x
    # graph['num_nodes'] = len(x)
    # return graph

    x = torch.tensor(x, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return x, edge_attr, edge_index  # x: 分子中的另一种原子特征向量  edge_attr: 另一种边特征  edge_index: 分子边的起始、终止点

def mol_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

# def get_smiles(drug_id):
#     smiles=[]
#     for id in drug_id:
#         drug = 'dataset/sdf/' + id + '.sdf'
#         mol = Chem.MolFromMolFile(drug)
#         smi = Chem.MolToSmiles(mol,isomericSmiles=True)
#         smiles.append(smi)
#     return smiles




