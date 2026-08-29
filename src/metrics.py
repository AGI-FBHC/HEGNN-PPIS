"""Metric helpers for residue-level interface prediction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score


EPS = 1e-12


def probability_column(frame):
    for name in ("selected_probability", "probability", "score"):
        if name in frame:
            return frame[name].to_numpy(float)
    raise ValueError(f"No probability column found. Available columns: {list(frame.columns)}")


def compute_metrics(labels, probabilities, predictions):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = np.asarray(predictions, dtype=int)
    if not (len(labels) == len(probabilities) == len(predictions)):
        raise ValueError("Labels, probabilities, and predictions must have equal length")

    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return {
        "ACC": (tp + tn) / len(labels),
        "Precision": precision,
        "Recall": recall,
        "F1": 2.0 * precision * recall / max(precision + recall, EPS),
        "MCC": float(matthews_corrcoef(labels, predictions)),
        "AUROC": float(roc_auc_score(labels, probabilities)),
        "AUPRC": float(average_precision_score(labels, probabilities)),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }
