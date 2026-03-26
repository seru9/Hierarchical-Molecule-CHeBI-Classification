import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit import RDLogger

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Keep RDKit errors visible, but suppress frequent parser warnings from edge-case SMILES.
RDLogger.DisableLog("rdApp.warning")

# Basic atomic numbers we might encounter or want to one-hot encode.
# For simplicity, we'll just use a slightly broad list, or raw values.
def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

def get_atom_features(atom):
    # 1. Atomic Number (Common organic elements + Unknown) -> 44 wymiary
    permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    
    # 2. Degree (0-10) -> 12 wymiarów
    degree_enc = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'MoreThan10'])
    
    # 3. Formal Charge -> 8 wymiarów
    formal_charge_enc = one_hot_encoding(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3, 'Extreme'])
    
    # 4. Hybridization -> 7 wymiarów
    hybridization_enc = one_hot_encoding(str(atom.GetHybridization()), ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED'])
    
    # 5. Is Aromatic -> 1 wymiar
    is_aromatic = [int(atom.GetIsAromatic())]
    
    # 6. Implicit Valence -> 8 wymiarów
    imp_valence_enc = one_hot_encoding(atom.GetValence(Chem.ValenceType.IMPLICIT), [0, 1, 2, 3, 4, 5, 6, 'MoreThan6'])

    # 7. Is In Ring -> 1 wymiar (DODANE)
    is_in_ring = [int(atom.IsInRing())]
    
    # RAZEM: 44 + 12 + 8 + 7 + 1 + 8 + 1 = 81 wymiarów
    atom_features = atom_type_enc + degree_enc + formal_charge_enc + hybridization_enc + is_aromatic + imp_valence_enc + is_in_ring
    return atom_features

def get_bond_features(bond):
    # 1. Bond Type -> 4 wymiary
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    
    # 2. Is Conjugated -> 1 wymiar
    is_conjugated = [int(bond.GetIsConjugated())]
    
    # 3. Is In Ring -> 1 wymiar
    is_in_ring = [int(bond.IsInRing())]
    
    # RAZEM: 4 + 1 + 1 = 6 wymiarów
    bond_features = bond_type_enc + is_conjugated + is_in_ring
    return bond_features

def smiles_to_graph(smiles, y_labels=None, mol_id=None, device=torch.device("cpu")):
    """
    Converts a single SMILES string to a PyTorch Geometric Data object.
    Returns None if RDKit cannot parse the SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Nodes
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(get_atom_features(atom))
    x = torch.tensor(node_features, dtype=torch.float, device=device)
    
    # Edges
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        edge_feature = get_bond_features(bond)
        
        # Undirected graph: add both directions
        edge_indices += [[i, j], [j, i]]
        edge_features += [edge_feature, edge_feature]
        
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long, device=device).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float, device=device)
    else:
        # For molecules with only 1 atom (no bonds)
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, sum(len(x) for x in [[1]*4, [1], [1]])), dtype=torch.float, device=device)
        
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    if y_labels is not None:
        data.y = torch.tensor([y_labels], dtype=torch.float, device=device)
        
    if mol_id is not None:
        data.mol_id = mol_id
        
    return data

def process_parquet(parquet_path, output_path, is_test=False, device=torch.device("cpu")):
    print(f"Reading {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    data_list = []
    skipped = 0
    total = len(df)

    # Assume target columns are those starting with 'class_'
    class_cols = [c for c in df.columns if c.startswith('class_')]

    for idx, row in df.iterrows():
        smiles = row['SMILES']
        mol_id = row['mol_id']

        y = None
        if not is_test and len(class_cols) > 0:
            y = row[class_cols].values.astype(np.float32).tolist()

        data = smiles_to_graph(smiles, y_labels=y, mol_id=mol_id, device=device)

        if data is not None:
            # Save CPU tensors for maximum portability during later loading.
            data_list.append(data.cpu())
        else:
            skipped += 1
            print(f"Warning: RDKit could not parse SMILES '{smiles}' (mol_id: {mol_id})")

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{total} molecules")

    print(f"Finished processing. Total graphs: {len(data_list)}. Skipped: {skipped}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving graph dataset to {output_path}...")
    torch.save(data_list, output_path)
    print("Saved successfully.")

def select_device(device_arg):
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda requested, but CUDA is not available.")
        return torch.device("cuda")

    if device_arg == "cpu":
        return torch.device("cpu")

    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Vectorize ChEBI parquet datasets into PyG graphs.")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device used for tensor creation during vectorization.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = select_device(args.device)
    print(f"Using device for vectorization: {device}")

    test_parquet = os.path.join(PROJECT_ROOT, "chebi_dataset_test_empty.parquet")
    train_parquet = os.path.join(PROJECT_ROOT, "chebi_dataset_train.parquet")
    output_test_graphs = os.path.join(PROJECT_ROOT, "processed_data", "test_empty_graphs.pt")
    output_graphs = os.path.join(PROJECT_ROOT, "processed_data", "train_graphs.pt")

    # Przetworzenie pliku testowego (bez etykiet) - wymagane przez predict.py
    process_parquet(
        test_parquet,
        output_test_graphs,
        is_test=True,
        device=device
    )

    # Przetworzenie pełnego pliku treningowego
    process_parquet(
        train_parquet,
        output_graphs,
        is_test=False,
        device=device
    )
    
    # Provide a simple check
    try:
        # These are locally generated dataset files, not model weights.
        test_graphs = torch.load(output_test_graphs, weights_only=False)
        train_graphs = torch.load(output_graphs, weights_only=False)
        print("Loaded test dataset successfully! It contains", len(test_graphs), "graphs.")
        print("Loaded train dataset successfully! It contains", len(train_graphs), "graphs.")

        if len(train_graphs) > 0:
            g = train_graphs[0]
            print("First graph format:", g)
            print("x shape:", g.x.shape)
            print("edge_index shape:", g.edge_index.shape)
            print("edge_attr shape:", g.edge_attr.shape)
            if hasattr(g, 'y') and g.y is not None:
                print("y shape:", g.y.shape)
    except Exception as e:
        print("Failed to load or test the output dataset:", e)