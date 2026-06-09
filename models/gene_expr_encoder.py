"""
基因表达特征编码器
将978维基因表达向量编码为固定维度的特征向量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneExpressionEncoder(nn.Module):
    """基因表达特征编码器"""
    def __init__(self, input_dim=978, hidden_dim=256, output_dim=128, dropout=0.1):
        """
        Args:
            input_dim: 输入维度（基因数量，默认978）
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            dropout: Dropout率
        """
        super(GeneExpressionEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)#978维向量 → hidden_dim维向量 978->256
        self.fc2 = nn.Linear(hidden_dim, output_dim)#hidden_dim维向量 → output_dim维向量 256->128
        self.dropout = nn.Dropout(dropout)#防止过拟合
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)#批量归一化    
        self.bn2 = nn.BatchNorm1d(output_dim)
    
    def forward(self, x):
        """
        Args:
            x: 输入基因表达向量 (batch_size, 978) 或 (978,)
        
        Returns:
            编码后的特征向量 (batch_size, output_dim) 或 (output_dim,)
        """
        # 处理单个样本的情况
        if x.dim() == 1:
            x = x.unsqueeze(0)#如果输入是单个样本，则添加batch维度 (978,) → (1, 978)
            single_sample = True
        else:
            single_sample = False
        #978维向量 → hidden_dim维向量 978->256
        x = self.fc1(x)
        x = self.bn1(x)#批量归一化
        x = self.relu(x)#激活函数
        x = self.dropout(x)#防止过拟合
        #hidden_dim维向量 → output_dim维量 256->128
        x = self.fc2(x)
        x = self.bn2(x)#批量归一化
        x = self.relu(x)#激活函数
        
        if single_sample:#如果是单个样本，则去掉batch维度
            x = x.squeeze(0)
        
        return x

