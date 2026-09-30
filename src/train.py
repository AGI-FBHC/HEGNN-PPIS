"""
Train and validate the single-stage dual-branch HEGNN-PPIS model.

Architecture:
- VN-EGNN backbone (shared)
- Full hypergraph branch (local_full) - baseline hypergraph
- Selective hypergraph branch (local_selective) - S2 hotspot surface patch
- Residual fusion: h = h_VN + alpha_full * h_full + alpha_selective * h_selective
- Checkpoints selected using Train335 validation AUPRC
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics as sk_metrics

from dataloader import ProDatasetDual, graph_collate_dual, init
from model import HEGNNPPIS_Dual

init()


def generate_dataframe(dataset):
    IDs, sequences, labels = [], [], []
    for ID in dataset:
        IDs.append(ID)
        item = dataset[ID]
        sequences.append(item[0])
        labels.append(item[1])
    return pd.DataFrame({"ID": IDs, "sequence": sequences, "label": labels})


def evaluate_model(model, validation_loader, device):
    """Evaluate on the supplied validation loader."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in validation_loader:
            (sequence_name, labels, node_features, virtual_node_features, pos, virtual_pos,
             edge_index, A2V_edge_index, V2A_edge_index, hypergraph_full, hypergraph_selective) = data

            node_features = node_features.float().to(device)
            virtual_node_features = virtual_node_features.float().to(device)
            edge_index = edge_index.long().to(device)
            A2V_edge_index = A2V_edge_index.long().to(device)
            V2A_edge_index = V2A_edge_index.long().to(device)
            y_true = labels.to(device).squeeze().long()
            pos = pos.float().to(device)
            virtual_pos = virtual_pos.float().to(device)
            hypergraph_full = hypergraph_full[0]
            hypergraph_selective = hypergraph_selective[0]

            y_pred = model(node_features, pos, virtual_node_features, virtual_pos,
                          edge_index, A2V_edge_index, V2A_edge_index,
                          hypergraph_full, hypergraph_selective)

            probs = F.softmax(y_pred, dim=1)[:, 1].cpu().numpy()
            labels_np = y_true.cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels_np)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    binary_preds = (all_preds >= 0.5).astype(int)
    precision, recall, _ = sk_metrics.precision_recall_curve(all_labels, all_preds)
    auprc = sk_metrics.auc(recall, precision)
    auroc = sk_metrics.roc_auc_score(all_labels, all_preds)
    mcc = sk_metrics.matthews_corrcoef(all_labels, binary_preds)

    return {'AUPRC': auprc, 'AUC': auroc, 'MCC': mcc}


def train_and_save_checkpoints(seed, dataset, validation_dataset,
                               hypergraph_dir_full, hypergraph_dir_selective,
                               psepos_train, psepos_valid, epochs=30,
                               alpha_full=0.05, alpha_selective=0.10,
                               checkpoint_dir=None):
    """Train, save checkpoints, and return per-epoch validation metrics."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model = HEGNNPPIS_Dual(
        in_dim=67, in_edge_dim=1, hidden_dim=67, layers=4,
        alpha_full=alpha_full, alpha_selective=alpha_selective,
        use_virtual_nodes=True, use_hyperedges=True
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_df = generate_dataframe(dataset)
    valid_df = generate_dataframe(validation_dataset)

    train_loader = DataLoader(
        dataset=ProDatasetDual(train_df, psepos_path=psepos_train, hypernodes=3,
                               hypergraph_dir_full=hypergraph_dir_full,
                               hypergraph_dir_selective=hypergraph_dir_selective),
        batch_size=1, shuffle=True, num_workers=0, collate_fn=graph_collate_dual, pin_memory=False)
    valid_loader = DataLoader(
        dataset=ProDatasetDual(valid_df, psepos_path=psepos_valid, hypernodes=3,
                               hypergraph_dir_full=hypergraph_dir_full,
                               hypergraph_dir_selective=hypergraph_dir_selective),
        batch_size=1, shuffle=False, num_workers=0, collate_fn=graph_collate_dual, pin_memory=False)

    epoch_records = []

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n = 0

        for data in train_loader:
            model.optimizer.zero_grad()
            (sequence_name, labels, node_features, virtual_node_features, pos, virtual_pos,
             edge_index, A2V_edge_index, V2A_edge_index, hypergraph_full, hypergraph_selective) = data

            node_features = node_features.float().to(device)
            virtual_node_features = virtual_node_features.float().to(device)
            edge_index = edge_index.long().to(device)
            A2V_edge_index = A2V_edge_index.long().to(device)
            V2A_edge_index = V2A_edge_index.long().to(device)
            y_true = labels.to(device).squeeze().long()
            pos = pos.float().to(device)
            virtual_pos = virtual_pos.float().to(device)
            hypergraph_full = hypergraph_full[0]
            hypergraph_selective = hypergraph_selective[0]

            y_pred = model(node_features, pos, virtual_node_features, virtual_pos,
                          edge_index, A2V_edge_index, V2A_edge_index,
                          hypergraph_full, hypergraph_selective)

            loss = model.criterion(y_pred, y_true)
            loss.backward()
            model.optimizer.step()
            train_loss += loss.item()
            n += 1

        # Evaluate current weights on the Train335 validation partition.
        valid_metrics = evaluate_model(model, valid_loader, device)

        # Save checkpoint
        ckpt_path = None
        if checkpoint_dir:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)

        epoch_records.append({
            'epoch': epoch + 1,
            'path': ckpt_path,
            'val_AUPRC': valid_metrics['AUPRC'],
            'val_AUC': valid_metrics['AUC'],
            'val_MCC': valid_metrics['MCC'],
        })

        model.scheduler.step(valid_metrics['AUPRC'])

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Val AUPRC: {valid_metrics['AUPRC']:.4f} | Loss: {train_loss/n:.4f}")

    best_auprc = max(r['val_AUPRC'] for r in epoch_records)
    print(f"    Training complete. Best validation single-epoch AUPRC: {best_auprc:.4f}")
    return epoch_records




if __name__ == '__main__':
    from train_validation_only import main
    main()
