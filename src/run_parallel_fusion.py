"""Parallel stable/adaptive fusion with Train335 validation selection."""
import argparse
import copy
import json
import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics as sk_metrics
from torch.utils.data import DataLoader

from dataloader import ProDatasetDual, graph_collate_dual, init
from homology_split import split_train_validation_homology
from model import HEGNNPPIS_Dual
from train import generate_dataframe

init()
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SOURCE_DIR)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_model(device):
    return HEGNNPPIS_Dual(in_dim=67, in_edge_dim=1, hidden_dim=67, layers=4,
                          alpha_full=0.05, alpha_selective=0.10,
                          use_virtual_nodes=True, use_hyperedges=True).to(device)


def make_loader(dataset, psepos, full_dir, selective_dir, shuffle):
    return DataLoader(ProDatasetDual(generate_dataframe(dataset), psepos_path=psepos,
                      hypernodes=3, hypergraph_dir_full=full_dir,
                      hypergraph_dir_selective=selective_dir), batch_size=1,
                      shuffle=shuffle, num_workers=0, collate_fn=graph_collate_dual)


def forward(model, data, device):
    (_, labels, node_features, virtual_node_features, pos, virtual_pos,
     edge_index, A2V_edge_index, V2A_edge_index, hg_full, hg_selective) = data
    logits = model(node_features.float().to(device), pos.float().to(device),
                   virtual_node_features.float().to(device), virtual_pos.float().to(device),
                   edge_index.long().to(device), A2V_edge_index.long().to(device),
                   V2A_edge_index.long().to(device), hg_full[0], hg_selective[0])
    return logits, labels.to(device).squeeze().long()


def train_epoch(model, loader, device):
    model.train(); losses = []
    for data in loader:
        model.optimizer.zero_grad()
        logits, labels = forward(model, data, device)
        loss = model.criterion(logits, labels)
        loss.backward(); model.optimizer.step(); losses.append(loss.item())
    return float(np.mean(losses))


def predictions(model, loader, device):
    model.eval(); ps, ys = [], []
    with torch.no_grad():
        for data in loader:
            logits, labels = forward(model, data, device)
            ps.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
            ys.extend(labels.cpu().numpy())
    return np.asarray(ps), np.asarray(ys)


def metric_dict(y, p):
    precision, recall, _ = sk_metrics.precision_recall_curve(y, p)
    binary = (p >= 0.5).astype(int)
    return {
        'AUPRC': float(sk_metrics.auc(recall, precision)),
        'AUROC': float(sk_metrics.roc_auc_score(y, p)),
        'MCC': float(sk_metrics.matthews_corrcoef(y, binary)),
        'Acc': float(sk_metrics.accuracy_score(y, binary)),
        'Precision': float(sk_metrics.precision_score(y, binary, zero_division=0)),
        'Recall': float(sk_metrics.recall_score(y, binary, zero_division=0)),
        'F1': float(sk_metrics.f1_score(y, binary, zero_division=0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=os.path.join(SOURCE_DIR, 'Dataset', 'Train_335.pkl'))
    ap.add_argument('--psepos_train', default=os.path.join(SOURCE_DIR, 'Feature', 'psepos', 'Train335_psepos_SC.pkl'))
    ap.add_argument('--hypergraph_dir_full', default=os.path.join(SOURCE_DIR, 'Graph', 'SC', 'hypergraph'))
    ap.add_argument('--hypergraph_dir_selective', default=os.path.join(SOURCE_DIR, 'Graph', 'SC', 'hypergraph_surface', 'hotspot_surface_r10'))
    ap.add_argument('--homology_manifest', default=os.path.join(ROOT, 'protocol', 'homology_i30c80.json'))
    ap.add_argument('--seed', type=int, default=2020)
    ap.add_argument('--shared_epochs', type=int, default=15)
    ap.add_argument('--branch_epochs', type=int, default=15)
    ap.add_argument('--adaptive_lr', type=float, default=3e-4)
    ap.add_argument('--output_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.dataset, 'rb') as f: dataset = pickle.load(f)
    dataset.pop('2j3rA', None)
    train_set, valid_set, split = split_train_validation_homology(dataset, args.homology_manifest,
                                                                     val_fraction=.15, split_seed=2026)
    train_loader = make_loader(train_set, args.psepos_train, args.hypergraph_dir_full, args.hypergraph_dir_selective, True)
    valid_loader = make_loader(valid_set, args.psepos_train, args.hypergraph_dir_full, args.hypergraph_dir_selective, False)
    print(f'Train335 homology split {len(train_set)} train / {len(valid_set)} validation')

    shared = make_model(device)
    for epoch in range(1, args.shared_epochs + 1):
        loss = train_epoch(shared, train_loader, device)
        if epoch == 1 or epoch % 5 == 0: print(f'shared {epoch}/{args.shared_epochs}: loss={loss:.4f}')
    shared_state = copy.deepcopy(shared.state_dict())

    stable, adaptive = make_model(device), make_model(device)
    stable.load_state_dict(shared_state); adaptive.load_state_dict(shared_state)
    for group in adaptive.optimizer.param_groups: group['lr'] = args.adaptive_lr

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]  # alpha is stable-branch weight
    best = None
    for epoch in range(1, args.branch_epochs + 1):
        stable_loss = train_epoch(stable, train_loader, device)
        adaptive_loss = train_epoch(adaptive, train_loader, device)
        p_stable, y = predictions(stable, valid_loader, device)
        p_adaptive, y2 = predictions(adaptive, valid_loader, device)
        assert np.array_equal(y, y2)
        for alpha in alphas:
            m = metric_dict(y, alpha * p_stable + (1-alpha) * p_adaptive)
            if best is None or m['AUPRC'] > best['val_metrics']['AUPRC']:
                best = {'branch_epoch': epoch, 'alpha_stable': alpha, 'val_metrics': m,
                        'stable_state': copy.deepcopy(stable.state_dict()),
                        'adaptive_state': copy.deepcopy(adaptive.state_dict())}
        if epoch == 1 or epoch % 5 == 0:
            print(f'parallel {epoch}/{args.branch_epochs}: stable_loss={stable_loss:.4f}, adaptive_loss={adaptive_loss:.4f}, best_val_AUPRC={best["val_metrics"]["AUPRC"]:.4f}')


    stable.load_state_dict(best.pop('stable_state'))
    adaptive.load_state_dict(best.pop('adaptive_state'))
    torch.save(stable.state_dict(), os.path.join(args.output_dir, 'stable.pt'))
    torch.save(adaptive.state_dict(), os.path.join(args.output_dir, 'adaptive.pt'))
    result = {'protocol': 'Train335 homology-isolated validation parallel fusion',
              'config': vars(args),
              'split': split, 'selection': best}
    with open(os.path.join(args.output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

