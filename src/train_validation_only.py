"""Train on Train335 and select checkpoints using homology-isolated validation."""

import argparse
import json
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataloader import ProDatasetDual, graph_collate_dual, init
from homology_split import split_train_validation_homology
from model import HEGNNPPIS_Dual
from train import evaluate_model, generate_dataframe, train_and_save_checkpoints

SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=SOURCE_DIR / "Dataset/Train_335.pkl")
    parser.add_argument("--psepos", type=Path, default=SOURCE_DIR / "Feature/psepos/Train335_psepos_SC.pkl")
    parser.add_argument("--full-hypergraphs", "--hypergraph_dir_full", type=Path, default=SOURCE_DIR / "Graph/SC/hypergraph")
    parser.add_argument("--selective-hypergraphs", "--hypergraph_dir_selective", type=Path, default=SOURCE_DIR / "Graph/SC/hypergraph_surface/hotspot_surface_r10")
    parser.add_argument("--homology-manifest", "--homology_manifest", type=Path, default=ROOT / "protocol/homology_i30c80.json")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--split-seed", "--split_seed", type=int, default=2026)
    parser.add_argument("--val-fraction", "--val_fraction", type=float, default=0.15)
    parser.add_argument("--alpha-full", "--alpha_full", type=float, default=0.05)
    parser.add_argument("--alpha-selective", "--alpha_selective", type=float, default=0.10)
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=ROOT / "output/train")
    return parser.parse_args()


def average_checkpoints(records):
    """Average up to five epochs selected by validation AUPRC."""
    selected = sorted(records, key=lambda item: item["val_AUPRC"])[-5:]
    states = [torch.load(item["path"], map_location="cpu", weights_only=True) for item in selected]
    averaged = {key: states[0][key].clone() for key in states[0]}
    for state in states[1:]:
        for key in averaged:
            averaged[key] += state[key]
    for key in averaged:
        averaged[key] /= len(states)
    return averaged, selected


def main():
    args = parse_args()
    init()
    with args.dataset.open("rb") as handle:
        dataset = pickle.load(handle)
    dataset.pop("2j3rA", None)
    train_set, validation_set, split = split_train_validation_homology(
        dataset, args.homology_manifest, args.val_fraction, args.split_seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = train_and_save_checkpoints(
        args.seed, train_set, validation_set,
        str(args.full_hypergraphs), str(args.selective_hypergraphs),
        str(args.psepos), str(args.psepos), args.epochs,
        alpha_full=args.alpha_full, alpha_selective=args.alpha_selective,
        checkpoint_dir=str(args.output_dir / "checkpoints"),
    )
    averaged, selected = average_checkpoints(records)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HEGNNPPIS_Dual(
        in_dim=67, in_edge_dim=1, hidden_dim=67, layers=4,
        alpha_full=args.alpha_full, alpha_selective=args.alpha_selective,
        use_virtual_nodes=True, use_hyperedges=True,
    ).to(device)
    model.load_state_dict(averaged, strict=True)
    torch.save(averaged, args.output_dir / "swa_best5.pt")
    validation_loader = DataLoader(
        ProDatasetDual(
            generate_dataframe(validation_set), psepos_path=str(args.psepos),
            hypernodes=3, hypergraph_dir_full=str(args.full_hypergraphs),
            hypergraph_dir_selective=str(args.selective_hypergraphs),
        ),
        batch_size=1, shuffle=False, num_workers=0, collate_fn=graph_collate_dual,
    )
    report = {
        "protocol": "Train335 homology-component validation",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "split": split,
        "best_epoch": max(records, key=lambda item: item["val_AUPRC"]),
        "swa_best5_epochs": [item["epoch"] for item in selected],
        "swa_best5_validation_metrics": evaluate_model(model, validation_loader, device),
    }
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
