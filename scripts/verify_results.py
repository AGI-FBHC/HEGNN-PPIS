"""Verify the published Test60 predictions and metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics import compute_metrics, probability_column  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=ROOT / "results/test60",
        help="Directory containing predictions.csv and, optionally, metrics.csv",
    )
    return parser.parse_args()


def verify_published_checksums(reference_dir):
    manifest_path = reference_dir / "checksums.sha256"
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative_path = line.split(maxsplit=1)
        artifact_path = (ROOT / relative_path.strip()).resolve()
        observed = sha256(artifact_path)
        if observed != expected:
            raise SystemExit(f"SHA-256 mismatch: {artifact_path}")


def main():
    args = parse_args()
    result_dir = args.result_dir.resolve()
    reference_dir = ROOT / "results/test60"
    verify_published_checksums(reference_dir)
    predictions = pd.read_csv(result_dir / "predictions.csv")
    expected = pd.read_csv(reference_dir / "metrics.csv").iloc[0]
    actual = compute_metrics(
        predictions["label"].to_numpy(int),
        probability_column(predictions),
        predictions["selected_prediction"].to_numpy(int),
    )
    for name, value in actual.items():
        expected_value = float(expected[name])
        if not math.isclose(float(value), expected_value, rel_tol=0.0, abs_tol=1e-12):
            raise SystemExit(f"{name}: {value} != {expected_value}")

    local_metrics_path = result_dir / "metrics.csv"
    if local_metrics_path.is_file():
        local_metrics = pd.read_csv(local_metrics_path).iloc[0]
        for name, value in actual.items():
            if not math.isclose(
                float(value), float(local_metrics[name]), rel_tol=0.0, abs_tol=1e-12
            ):
                raise SystemExit(
                    f"Local metrics mismatch for {name}: "
                    f"{value} != {local_metrics[name]}"
                )

    experiment = json.loads((reference_dir / "experiment.json").read_text(encoding="utf-8"))
    for checkpoint in experiment["checkpoints"]:
        checkpoint_path = (reference_dir / checkpoint["path"]).resolve()
        observed = sha256(checkpoint_path)
        if observed != checkpoint["sha256"]:
            raise SystemExit(f"Checkpoint SHA-256 mismatch: {checkpoint_path}")

    print(
        "Result verification passed: "
        f"ACC={actual['ACC']:.9f}, "
        f"Precision={actual['Precision']:.9f}, "
        f"AUROC={actual['AUROC']:.9f}, "
        f"AUPRC={actual['AUPRC']:.9f}"
    )


if __name__ == "__main__":
    main()
