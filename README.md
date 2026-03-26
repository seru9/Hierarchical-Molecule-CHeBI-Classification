# Hierarchical Molecule (CHeBI)
## src3
### Embeddings preparation
**`vectorize_data.py`**
We convert molecules from SMILES strings into graphs, incorporating a custom feature extraction step where `get_atom_features()` encodes the attributes of each atom. Using RDKit's `Chem.MolFromSmiles`, the script parses strings (such as `CCO` for ethanol) loaded from Parquet files. To ensure efficient data handling, the resulting graph objects are not stored as JSON or text; instead, they are saved directly as binary `.pt` files using `torch.save`, which allows for seamless loading into PyTorch.
### Model 
Theoretically structures such Graph Neural Network, Graph Isomorphism Network etc. should work ideally with molecule data, because of its graph like structure. Based on the pre-print there are better solutions that I will discuss later. As Graph Neural Networks only learn node embeddings so that Graph Isomorphism Nettwork learn graph embeddings! The **ChEBIGIN** model uses a Graph Isomorphism Network architecture with four `GINEConv` layers to embed 80-dimensional node features and 6-dimensional edge attributes into a 128-dimensional hidden space. It employs a dual global pooling strategy (mean and max) to aggregate molecular data for multi-label classification across 500 ChEBI classes. 
The training pipeline in `train.py` utilizes `BCEWithLogitsLoss` and the Adam optimizer, featuring a dynamic evaluation loop that tunes the classification threshold (0.05–0.95) to maximize the **Macro-F1 score**.
