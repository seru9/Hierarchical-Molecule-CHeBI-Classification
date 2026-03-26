import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from model import ChEBIGIN
from dataset import ChEBIDataset
import copy
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        # Omit 'y' from inputs. Model signature: x, edge_index, edge_attr, batch
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
        # ChEBI dataset targets are [batch_size, 500]
        # BCEWithLogitsLoss expects raw logits, which our model produces.
        target = batch.y.float()
        loss = criterion(out, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device, thresholds=None):
    model.eval()
    total_loss = 0
    all_probs = []
    all_targets = []
    
    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)
        
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        target = batch.y.float()
        
        loss = criterion(out, target)
        total_loss += loss.item() * batch.num_graphs

        probs = torch.sigmoid(out)
        all_probs.append(probs.detach().cpu())
        all_targets.append(target.detach().cpu())

    val_loss = total_loss / len(loader.dataset)
    if len(all_probs) == 0:
        return val_loss, 0.0, 0.5

    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    target_bin = (all_targets >= 0.5).float()

    if thresholds is None:
        thresholds = torch.arange(0.1, 0.91, 0.05)

    best_threshold = 0.5
    best_macro_f1 = 0.0
    eps = 1e-12

    for threshold in thresholds:
        preds = (all_probs >= threshold).float()

        tp = ((preds == 1) & (target_bin == 1)).sum(dim=0)
        fp = ((preds == 1) & (target_bin == 0)).sum(dim=0)
        fn = ((preds == 0) & (target_bin == 1)).sum(dim=0)

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1_per_class = 2 * precision * recall / (precision + recall + eps)
        macro_f1 = f1_per_class.mean().item()

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = float(threshold)

    return val_loss, best_macro_f1, best_threshold

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Dataset
    print("Loading data...")
    try:
        # Ładowanie całego wygenerowanego zbioru treningowego
        full_dataset = ChEBIDataset(root="processed_data", file_name="train_graphs.pt")
    except Exception as e:
        print("Please run vectorize_data.py first to generate processed_data/train_graphs.pt")
        return

    # Split dataset into Train and Val (e.g. 80/20)
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    sample_graph = full_dataset[0]
    node_dim = sample_graph.x.shape[1]
    edge_dim = sample_graph.edge_attr.shape[1] if sample_graph.edge_attr.ndim == 2 else 6
    
    # 2. Initialize Model
    model = ChEBIGIN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=256, 
        num_classes=500, 
        num_layers=4
    ).to(device)

    # 3. Training Setup
    print("Computing class imbalance weights (pos_weight) from train split...")
    num_classes = sample_graph.y.view(-1).shape[0]
    pos_counts = torch.zeros(num_classes)

    for data in tqdm(train_dataset, desc="Counting positives"):
        pos_counts += data.y.view(-1).float().cpu()

    total_samples = len(train_dataset)
    neg_counts = total_samples - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-6)
    pos_weight = torch.clamp(pos_weight, max=50.0).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    epochs = 100 # Just a short run for verification
    best_loss = float('inf')
    best_f1 = 0.0
    best_threshold = 0.5
    best_model_weights = None
    
    print("\nStarting Training...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_macro_f1, val_best_threshold = evaluate(model, val_loader, criterion, device)
        
        print(
            f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Macro-F1: {val_macro_f1:.4f} "
            f"(best threshold={val_best_threshold:.2f})"
        )
        
        if (val_macro_f1 > best_f1) or (
            abs(val_macro_f1 - best_f1) < 1e-12 and val_loss < best_loss
        ):
            best_loss = val_loss
            best_f1 = val_macro_f1
            best_threshold = val_best_threshold
            best_model_weights = copy.deepcopy(model.state_dict())
            
    print("\nTraining complete!")
    print(f"Best Val Loss: {best_loss:.4f}")
    print(f"Best Val Macro-F1: {best_f1:.4f}")
    print(f"Best threshold: {best_threshold:.2f}")
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    torch.save(best_model_weights, "models/best_chebi_gin.pth")
    with open("models/best_threshold.txt", "w", encoding="utf-8") as f:
        f.write(f"{best_threshold:.6f}\n")
    print("Saved best model to models/best_chebi_gin.pth")
    print("Saved best threshold to models/best_threshold.txt")

if __name__ == "__main__":
    main()
