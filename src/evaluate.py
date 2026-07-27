"""Reproduce the frozen three-seed Test60 evaluation."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit

from dataloader import init
from inference import align_prediction, make_loader, predict_loader, split_train_validation
from metrics import compute_metrics, probability_column
from model_ablation import HEGNNPPISDualAblation


EPS = 1e-6
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def build_model(checkpoint_path, device):
    model = HEGNNPPISDualAblation(
        in_dim=67,
        in_edge_dim=1,
        hidden_dim=67,
        layers=4,
        alpha_full=0.05,
        alpha_selective=0.10,
        use_virtual_nodes=True,
        use_hyperedges=True,
        ablation="full",
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=REPOSITORY_ROOT / "src/Dataset/Train_335.pkl",
    )
    parser.add_argument(
        "--test-dataset",
        type=Path,
        default=REPOSITORY_ROOT / "src/Dataset/Test_60.pkl",
    )
    parser.add_argument(
        "--train-psepos",
        type=Path,
        default=REPOSITORY_ROOT / "src/Feature/psepos/Train335_psepos_SC.pkl",
    )
    parser.add_argument(
        "--test-psepos",
        type=Path,
        default=REPOSITORY_ROOT / "src/Feature/psepos/Test60_psepos_SC.pkl",
    )
    parser.add_argument(
        "--full-hypergraphs",
        type=Path,
        default=REPOSITORY_ROOT / "src/Graph/SC/hypergraph",
    )
    parser.add_argument(
        "--selective-hypergraphs",
        type=Path,
        default=REPOSITORY_ROOT
        / "src/Graph/SC/hypergraph_surface/hotspot_surface_r10",
    )
    parser.add_argument(
        "--validation-anchor",
        type=Path,
        default=REPOSITORY_ROOT
        / "results/test60/validation_anchor_predictions.csv",
    )
    parser.add_argument(
        "--test-anchor",
        type=Path,
        default=REPOSITORY_ROOT / "results/test60/test_anchor_predictions.csv",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs=3,
        default=[
            REPOSITORY_ROOT / "checkpoints/seed_2181_epoch3.pt",
            REPOSITORY_ROOT / "checkpoints/seed_2182_epoch3.pt",
            REPOSITORY_ROOT / "checkpoints/seed_2183_epoch3.pt",
        ],
    )
    parser.add_argument("--blend-weight", type=float, default=0.25)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "output/test60",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    init()

    with args.train_dataset.resolve().open("rb") as handle:
        train_data = pickle.load(handle)
    train_data.pop("2j3rA", None)
    _, validation_data = split_train_validation(train_data, 0.15, 2026)

    with args.test_dataset.resolve().open("rb") as handle:
        test_data = pickle.load(handle)

    validation_anchor = pd.read_csv(args.validation_anchor.resolve())
    test_anchor = pd.read_csv(args.test_anchor.resolve())
    validation_loader = make_loader(
        validation_data,
        args.train_psepos.resolve(),
        args.full_hypergraphs.resolve(),
        args.selective_hypergraphs.resolve(),
    )
    test_loader = make_loader(
        test_data,
        args.test_psepos.resolve(),
        args.full_hypergraphs.resolve(),
        args.selective_hypergraphs.resolve(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [build_model(path.resolve(), device) for path in args.checkpoints]
    validation_members = []
    test_members = []
    for model in models:
        validation_members.append(
            align_prediction(
                predict_loader(model, validation_loader, device), validation_anchor
            )
        )
        test_members.append(
            align_prediction(predict_loader(model, test_loader, device), test_anchor)
        )

    validation_prior = np.mean(validation_members, axis=0)
    test_prior = np.mean(test_members, axis=0)
    base_validation = np.clip(
        probability_column(validation_anchor), EPS, 1.0 - EPS
    )
    base_test = np.clip(probability_column(test_anchor), EPS, 1.0 - EPS)
    base_validation_logit = logit(base_validation)
    validation_prior_logit = logit(
        np.clip(validation_prior, EPS, 1.0 - EPS)
    )
    test_prior_logit = logit(np.clip(test_prior, EPS, 1.0 - EPS))
    aligned_test_logit = (
        (test_prior_logit - validation_prior_logit.mean())
        / max(float(validation_prior_logit.std()), EPS)
        * max(float(base_validation_logit.std()), EPS)
        + base_validation_logit.mean()
    )
    final_probability = expit(
        (1.0 - args.blend_weight) * logit(base_test)
        + args.blend_weight * aligned_test_logit
    )

    labels = test_anchor["label"].to_numpy(int)
    predictions = test_anchor["selected_prediction"].to_numpy(int)
    metrics = compute_metrics(labels, final_probability, predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "protein_id": test_anchor["protein_id"],
            "label": labels,
            "selected_probability": final_probability,
            "selected_prediction": predictions,
        }
    ).to_csv(args.output_dir / "predictions.csv", index=False)
    pd.DataFrame([{"method": "three_seed_checkpoint_ensemble", **metrics}]).to_csv(
        args.output_dir / "metrics.csv", index=False
    )
    (args.output_dir / "run.json").write_text(
        json.dumps(
            {
                "device": str(device),
                "blend_weight": args.blend_weight,
                "checkpoints": [str(path.resolve()) for path in args.checkpoints],
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
