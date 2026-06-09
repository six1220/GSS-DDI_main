# GSS-DDI: Gene-Substructure-Similarity based Drug-Drug Interaction Prediction

GSS-DDI is a multi-modal deep learning model for predicting Drug-Drug Interactions (DDI). It integrates multiple information channels including sequence-based, 1D/2D structural, semantic, graph-based features, as well as gene expression data and molecular similarity measures.

## Model Architecture

The GSS-DDI model (`MultiLevelDDI`) consists of the following key components:

### 1. Model Components

| Component | Description | Module File |
|-----------|-------------|-------------|
| **AMDE** | Attention-based Molecular Drug Encoder using Transformer architecture for SMILES sequence encoding | `models/AMDE.py` |
| **GNN** | Graph Neural Network for molecular graph representation learning | `models/gnn.py` |
| **SchNet** | 3D molecular structure encoder (optional) | External dependency |
| **Molormer** | Semantic information encoder (optional) | External dependency |
| **Cross-Attention** | Multi-modal feature fusion mechanism | `models/model.py` |
| **GeneExpressionEncoder** | Gene expression data encoder | External dependency |

### 2. Feature Channels

The model supports multiple feature channels that can be combined:

- **Sequence-based Channel (AMDE)**: Transformer-based SMILES encoding
- **Semantic Channel (Molormer)**: Molecular semantic information
- **Graph Channel (GNN)**: Graph neural network with substructure importance scoring
- **Morgan Fingerprint**: 2048-bit molecular fingerprint
- **Tanimoto Similarity**: Chemical similarity between drug pairs
- **GEO Gene Expression**: Gene expression profile integration

### 3. Cross-Attention Mechanism

The model implements a two-level cross-attention mechanism:
- **Inner Cross-Attention**: Fuses features within each drug
- **Outer Cross-Attention**: Models interactions between drug pairs

## File Structure

```
GSS-DDI/
├── models/
│   ├── model.py        # Main MultiLevelDDI model class
│   ├── AMDE.py         # Transformer-based sequence encoder
│   └── gnn.py          # Graph neural network module
└── README.md           # This file
```

## Key Classes

### MultiLevelDDI (model.py)

The main model class that integrates all components:

- **BilinearDecoder**: Bilinear decoder for final prediction
- **MultiLevelDDI**: Core model with multi-channel feature extraction and fusion
- Supports weighted feature combination with learnable weights

### AMDE (AMDE.py)

Attention-based Molecular Drug Encoder:

- **Embeddings**: Combined word and positional embeddings
- **SelfAttention**: Multi-head self-attention mechanism
- **Encoder**: Transformer encoder layer with residual connections
- **Encoder_MultipleLayers**: Stacked encoder layers

### GNN (gnn.py)

Graph Neural Network module:

- **GNN**: Main graph neural network with message passing
- **FeedForwardNetwork**: Flexible feed-forward network
- **GraphGather**: Attention-based graph readout mechanism

## Usage

The model is designed to be used with PyTorch. Example usage:

```python
from models.model import MultiLevelDDI
import argparse

# Configure arguments
args = argparse.ArgumentParser()
args.use_AMDE = True
args.use_GNN = True
args.use_Mol = False
args.use_cross_attention = True
# ... other configurations

# Initialize model
model = MultiLevelDDI(args)

# Forward pass
score, explanations = model(
    d1_node, d1_in_degree, d1_out_degree,
    d2_node, d2_in_degree, d2_out_degree,
    d1, d2, mask_1, mask_2,
    d1_positions, d1_z, d1_batch,
    d2_positions, d2_z, d2_batch,
    adj_1, nd_1, ed_1, adj_2, nd_2, ed_2,
    d1_expr=d1_expr, d2_expr=d2_expr,
    d1_morgan=d1_morgan, d2_morgan=d2_morgan,
    tanimoto_similarity=tanimoto_similarity
)
```

## Datasets

Model performance was evaluated on **AdverseDDI** (12,228 positive/negative DDI pairs) and **Twosides** (645 nodes, 63,473 edges in a maximum weakly connected component). Gene-expression signatures from L1000 **GSE70138** and **GSE92742** (GEO database) were integrated, providing level-5 moderated expression profiles for 978 landmark genes. After preprocessing, 892 unique molecules with matched gene-expression signatures were obtained.

## Dependencies

- PyTorch (>= 1.8.0)
- NumPy
- RDKit (for molecular processing)

## Model Outputs

The model returns two outputs:
1. **score**: Predicted DDI score (classification or regression)
2. **explanations**: Dictionary containing attention weights and substructure importance scores for model interpretability
