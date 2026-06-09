"""
基因表达数据加载器
用于从 LINCS 数据中加载药物的基因表达特征
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path


class GeneExpressionLoader:
    def __init__(self, expr_path, use_cell_specific=True, preferred_cell_line=None):
        """
        加载基因表达数据
        
        Args:
            expr_path: 基因表达数据路径
            use_cell_specific: 是否使用细胞系特定的表达（True）还是所有细胞系合并（False）
            preferred_cell_line: 优先使用的细胞系（如 'A375'），如果为 None 则平均所有细胞系
        """
        self.use_cell_specific = use_cell_specific
        self.preferred_cell_line = preferred_cell_line
        
        print(f"加载基因表达数据: {expr_path}")
        print(f"  - 使用细胞系特定数据: {use_cell_specific}")
        print(f"  - 优先细胞系: {preferred_cell_line}")
        
        if use_cell_specific:
            # 使用按细胞系分别保存的版本
            self.df = pd.read_csv(expr_path)
            print(f"  - 数据形状: {self.df.shape}")
            print(f"  - 列: {self.df.columns[:5].tolist()}...")
            
            # 构建映射
            self._build_mapping_cell_specific()
        else:
            # 使用所有细胞系合并的版本
            self.df = pd.read_csv(expr_path, index_col=0)
            print(f"  - 数据形状: {self.df.shape}")
            self._build_mapping_merged()
        
        print(f"  - 唯一 drugbank_id 数: {len(self.mapping)}")
    
    def _build_mapping_cell_specific(self):
        """构建细胞系特定数据的映射 drugbank_id → 978维向量"""
        gene_cols = [c for c in self.df.columns 
                    if c not in ['drugbank_id', 'cell_id']]
        
        self.mapping = {}
        
        if self.preferred_cell_line:
            # 优先使用指定细胞系
            preferred_data = self.df[self.df['cell_id'] == self.preferred_cell_line]
            for _, row in preferred_data.iterrows():
                drugbank_id = row['drugbank_id']
                if drugbank_id not in self.mapping:
                    self.mapping[drugbank_id] = row[gene_cols].values.astype(np.float32)
            
            # 对于没有指定细胞系数据的药物，使用其他细胞系的平均值
            remaining_drugs = set(self.df['drugbank_id'].unique()) - set(self.mapping.keys())
            if remaining_drugs:
                remaining_data = self.df[self.df['drugbank_id'].isin(remaining_drugs)]
                for drugbank_id in remaining_drugs:
                    drug_data = remaining_data[remaining_data['drugbank_id'] == drugbank_id]
                    if len(drug_data) > 0:
                        self.mapping[drugbank_id] = drug_data[gene_cols].mean().values.astype(np.float32)
        else:
            # 平均所有细胞系
            for drugbank_id in self.df['drugbank_id'].unique():
                drug_data = self.df[self.df['drugbank_id'] == drugbank_id]
                self.mapping[drugbank_id] = drug_data[gene_cols].mean().values.astype(np.float32)
    
    def _build_mapping_merged(self):
        """构建合并数据的映射"""
        self.mapping = {}
        for drugbank_id, row in self.df.iterrows():
            self.mapping[drugbank_id] = row.values.astype(np.float32)
    
    def get_expression(self, drugbank_id):
        """
        获取药物的基因表达向量
        
        Args:
            drugbank_id: DrugBank ID (如 'DB00175')
        
        Returns:
            torch.Tensor: 978维基因表达向量，如果不存在则返回零向量
        """
        if drugbank_id in self.mapping:
            expr_vec = self.mapping[drugbank_id]
            return torch.tensor(expr_vec, dtype=torch.float32)
        else:
            # 如果药物不在数据中，返回零向量
            # 可以根据需要改为其他处理方式（如插值、使用相似药物等）
            return torch.zeros(978, dtype=torch.float32)
    
    def has_drug(self, drugbank_id):
        """检查药物是否在表达数据中"""
        return drugbank_id in self.mapping
    
    def get_coverage_stats(self, drug_ids):
        """
        获取数据覆盖统计
        
        Args:
            drug_ids: 药物ID列表
        
        Returns:
            dict: 统计信息
        """
        total = len(drug_ids)
        covered = sum(1 for did in drug_ids if self.has_drug(did))
        return {
            'total': total,
            'covered': covered,
            'missing': total - covered,
            'coverage_rate': covered / total if total > 0 else 0.0
        }

