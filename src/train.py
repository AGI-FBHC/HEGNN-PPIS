"""
Dual-branch selective hypergraph experiment for HEGNN-PPIS.

Architecture:
- VN-EGNN backbone (shared)
- Full hypergraph branch (local_full) - baseline hypergraph
- Selective hypergraph branch (local_selective) - S2 hotspot surface patch
- Residual fusion: h = h_VN + alpha_full * h_full + alpha_selective * h_selective
- SWA-best5 selected by validation AUPRC for final Test_60 evaluation

Expected: AUPRC > 0.680 (baseline: 0.6680 +- 0.0090)
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics as sk_metrics

from dataloader import ProDatasetDual, graph_collate_dual, init
from model import HEGNNPPIS_Dual

init()
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.dirname(SOURCE_DIR)


def generate_dataframe(dataset):
    IDs, sequences, labels = [], [], []
    for ID in dataset:
        IDs.append(ID)
        item = dataset[ID]
        sequences.append(item[0])
        labels.append(item[1])
    return pd.DataFrame({"ID": IDs, "sequence": sequences, "label": labels})


def split_train_validation(dataset, val_fraction=0.15, split_seed=2026):
    """Protein-level deterministic Train_335 split."""
    if not 0 < val_fraction < 1:
        raise ValueError("--val_fraction must be between 0 and 1")

    ids = sorted(dataset.keys())
    rng = np.random.default_rng(split_seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)

    val_size = max(1, int(round(len(shuffled) * val_fraction)))
    val_ids = set(shuffled[:val_size])
    train_split = {pid: dataset[pid] for pid in ids if pid not in val_ids}
    valid_split = {pid: dataset[pid] for pid in ids if pid in val_ids}
    return train_split, valid_split


def sample_std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def evaluate_model(model, test_loader, device):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
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


def train_and_save_checkpoints(seed, dataset, test_dataset,
                               hypergraph_dir_full, hypergraph_dir_selective,
                               psepos_train, psepos_test, epochs=30,
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
    valid_df = generate_dataframe(test_dataset)

    train_loader = DataLoader(
        dataset=ProDatasetDual(train_df, psepos_path=psepos_train, hypernodes=3,
                               hypergraph_dir_full=hypergraph_dir_full,
                               hypergraph_dir_selective=hypergraph_dir_selective),
        batch_size=1, shuffle=True, num_workers=0, collate_fn=graph_collate_dual, pin_memory=False)
    valid_loader = DataLoader(
        dataset=ProDatasetDual(valid_df, psepos_path=psepos_test, hypernodes=3,
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

        # Evaluate current weights on validation only; Test_60 is held out until final evaluation.
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


def load_and_evaluate_checkpoint(checkpoint_info, model, test_loader, device):
    """Load one selected checkpoint and evaluate it once on Test_60."""
    state = torch.load(checkpoint_info['path'], map_location=device)
    model.load_state_dict(state)
    metrics = evaluate_model(model, test_loader, device)
    print(
        f"  [Best checkpoint] Epoch {checkpoint_info['epoch']} "
        f"| Val AUPRC: {checkpoint_info['val_AUPRC']:.4f} "
        f"| Test AUPRC: {metrics['AUPRC']:.4f}"
    )
    return metrics


def evaluate_swa_best5(epoch_records, model, test_loader, device):
    """Apply SWA-best5: average top-5 checkpoints by validation AUPRC."""
    selected = sorted(epoch_records, key=lambda x: x['val_AUPRC'])[-5:]

    if len(selected) == 0:
        raise ValueError("No checkpoints selected for SWA-best5")

    avg_state = None
    for info in selected:
        state = torch.load(info['path'], map_location=device)
        if avg_state is None:
            avg_state = {k: v.clone() for k, v in state.items()}
        else:
            for k in avg_state:
                avg_state[k] += state[k]

    for k in avg_state:
        avg_state[k] /= len(selected)

    model.load_state_dict(avg_state)
    metrics = evaluate_model(model, test_loader, device)

    selected_epochs = [r['epoch'] for r in selected]
    selected_val = [r['val_AUPRC'] for r in selected]
    print(
        f"  [SWA-best5] Averaged epochs {selected_epochs} "
        f"| Mean Val AUPRC: {np.mean(selected_val):.4f} "
        f"| Test AUPRC: {metrics['AUPRC']:.4f}"
    )
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=os.path.join(SOURCE_DIR, "Dataset", "Train_335.pkl"))
    parser.add_argument("--test_dataset", type=str, default=os.path.join(SOURCE_DIR, "Dataset", "Test_60.pkl"))
    parser.add_argument("--psepos_train", type=str, default=os.path.join(SOURCE_DIR, "Feature", "psepos", "Train335_psepos_SC.pkl"))
    parser.add_argument("--psepos_test", type=str, default=os.path.join(SOURCE_DIR, "Feature", "psepos", "Test60_psepos_SC.pkl"))
    parser.add_argument("--hypergraph_dir_full", type=str, default=os.path.join(SOURCE_DIR, "Graph", "SC", "hypergraph"))
    parser.add_argument("--hypergraph_dir_selective", type=str, default=os.path.join(SOURCE_DIR, "Graph", "SC", "hypergraph_surface", "hotspot_surface_r10"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[2020, 2021, 2022])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val_fraction", type=float, default=0.15,
                        help="Fraction of Train_335 held out for validation checkpoint selection")
    parser.add_argument("--split_seed", type=int, default=2026,
                        help="Seed for the fixed protein-level train/validation split")
    parser.add_argument("--alpha_full", type=float, default=0.05)
    parser.add_argument("--alpha_selective", type=float, default=0.10)
    parser.add_argument("--output_dir", type=str, default=os.path.join(REPOSITORY_ROOT, "output", "train"))
    parser.add_argument("--cleanup", action="store_true", help="Delete non-best5 checkpoints after evaluation")
    args = parser.parse_args()

    with open(args.dataset, 'rb') as f:
        train_dataset = pickle.load(f)
    if '2j3rA' in train_dataset:
        train_dataset.pop('2j3rA')
    train_split, valid_split = split_train_validation(
        train_dataset, val_fraction=args.val_fraction, split_seed=args.split_seed)

    with open(args.test_dataset, 'rb') as f:
        test_dataset = pickle.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    swa_results = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"=" * 80)
    print("HEGNN-PPIS Dual-branch Selective Hypergraph Experiment")
    print(f"  alpha_full={args.alpha_full}, alpha_selective={args.alpha_selective}")
    print(f"  full_dir={args.hypergraph_dir_full}")
    print(f"  selective_dir={args.hypergraph_dir_selective}")
    print(
        f"  split={len(train_split)} train proteins / {len(valid_split)} validation proteins "
        f"(val_fraction={args.val_fraction}, split_seed={args.split_seed})"
    )
    print(f"=" * 80)

    for seed in args.seeds:
        print(f"\n{'=' * 80}")
        print(f"Training with seed {seed}...")
        print(f"{'=' * 80}")

        ckpt_dir = os.path.join(args.output_dir, f"checkpoints_seed{seed}")
        epoch_records = train_and_save_checkpoints(
            seed, train_split, valid_split,
            args.hypergraph_dir_full, args.hypergraph_dir_selective,
            args.psepos_train, args.psepos_train,
            args.epochs, alpha_full=args.alpha_full, alpha_selective=args.alpha_selective,
            checkpoint_dir=ckpt_dir)

        model = HEGNNPPIS_Dual(
            in_dim=67, in_edge_dim=1, hidden_dim=67, layers=4,
            alpha_full=args.alpha_full, alpha_selective=args.alpha_selective,
            use_virtual_nodes=True, use_hyperedges=True
        ).to(device)

        test_df = generate_dataframe(test_dataset)
        test_loader = DataLoader(
            dataset=ProDatasetDual(test_df, psepos_path=args.psepos_test, hypernodes=3,
                                   hypergraph_dir_full=args.hypergraph_dir_full,
                                   hypergraph_dir_selective=args.hypergraph_dir_selective),
            batch_size=1, shuffle=False, num_workers=0, collate_fn=graph_collate_dual, pin_memory=False)

        print(f"\n  Evaluating SWA-best5 for seed {seed}...")
        best_checkpoint = max(epoch_records, key=lambda x: x['val_AUPRC'])
        best_metrics = load_and_evaluate_checkpoint(best_checkpoint, model, test_loader, device)
        all_results.append({
            'seed': seed,
            'selected_epoch': best_checkpoint['epoch'],
            'selection_val_AUPRC': best_checkpoint['val_AUPRC'],
            **best_metrics,
        })

        swa_metrics = evaluate_swa_best5(epoch_records, model, test_loader, device)
        swa_results.append({'seed': seed, **swa_metrics})

        # Optional cleanup
        if args.cleanup:
            best5_epochs = {r['epoch'] for r in sorted(epoch_records, key=lambda x: x['val_AUPRC'])[-5:]}
            best5_epochs.add(best_checkpoint['epoch'])
            for r in epoch_records:
                if r['epoch'] not in best5_epochs and r['path'] and os.path.exists(r['path']):
                    os.remove(r['path'])
            print(f"  Cleaned up: kept only best5 checkpoints for seed {seed}")

    # Final summary
    print(f"\n{'=' * 80}")
    print("FINAL RESULTS")
    print(f"{'=' * 80}")

    best_singles = [r['AUPRC'] for r in all_results]
    swa_auprcs = [r['AUPRC'] for r in swa_results]
    swa_aucs = [r['AUC'] for r in swa_results]
    swa_mccs = [r['MCC'] for r in swa_results]

    print(f"\n{'Metric':<25} {'Mean':>10} {'Std':>10} {'vs Baseline':>12}")
    print("-" * 60)
    print(f"{'HEGNN-PPIS AUPRC':<25} {np.mean(best_singles):>10.4f} {sample_std(best_singles):>10.4f} {'--':>12}")
    print(f"{'SWA-best5 AUPRC':<25} {np.mean(swa_auprcs):>10.4f} {sample_std(swa_auprcs):>10.4f} {'--':>12}")
    print(f"{'SWA-best5 AUC':<25} {np.mean(swa_aucs):>10.4f} {sample_std(swa_aucs):>10.4f} {'--':>12}")
    print(f"{'SWA-best5 MCC':<25} {np.mean(swa_mccs):>10.4f} {sample_std(swa_mccs):>10.4f} {'--':>12}")

    summary = {
        'config': vars(args),
        'split': {
            'train_size': len(train_split),
            'validation_size': len(valid_split),
            'validation_ids': sorted(valid_split.keys()),
        },
        'best_single': {
            'AUPRC_mean': float(np.mean(best_singles)),
            'AUPRC_std': sample_std(best_singles),
            'results': all_results,
        },
        'swa_best5': {
            'AUPRC_mean': float(np.mean(swa_auprcs)),
            'AUPRC_std': sample_std(swa_auprcs),
            'AUC_mean': float(np.mean(swa_aucs)),
            'MCC_mean': float(np.mean(swa_mccs)),
            'results': swa_results,
        }
    }

    with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
