"""Label-blind homology-component splitting for Train335 validation."""

import json
from pathlib import Path

import numpy as np


def load_manifest(path):
    manifest = json.loads(Path(path).resolve().read_text(encoding="ascii"))
    if float(manifest.get("identity_threshold", 0.0)) != 0.30:
        raise ValueError("Expected identity threshold 0.30")
    if float(manifest.get("coverage_threshold", 0.0)) != 0.80:
        raise ValueError("Expected shorter-sequence coverage 0.80")
    if manifest.get("labels_used_for_homology_or_assignment") is not False:
        raise ValueError("Homology manifest is not label-blind")
    if manifest.get("heldout_labels_accessed") is not False:
        raise ValueError("Homology manifest accessed held-out labels")
    if manifest.get("heldout_sequences_accessed") is not False:
        raise ValueError("Homology manifest accessed held-out sequences")
    return manifest


def split_train_validation_homology(dataset, manifest_path, val_fraction=0.15, split_seed=2026):
    """Split whole homology components; no component crosses train/validation."""
    if not 0 < val_fraction < 1:
        raise ValueError("--val_fraction must be between 0 and 1")
    manifest = load_manifest(manifest_path)
    dataset_ids = {str(value) for value in dataset}
    manifest_ids = {str(value) for value in manifest.get("train_ids", [])}
    if dataset_ids != manifest_ids:
        missing = sorted(manifest_ids - dataset_ids)
        extra = sorted(dataset_ids - manifest_ids)
        raise ValueError(f"Dataset IDs do not match frozen homology manifest; missing={missing[:3]}, extra={extra[:3]}")
    components = [sorted(map(str, component)) for component in manifest["clusters"]]
    if sorted(value for component in components for value in component) != sorted(dataset_ids):
        raise ValueError("Homology components do not partition the training dataset")
    rng = np.random.default_rng(split_seed)
    order = rng.permutation(len(components)).tolist()
    target = max(1, int(round(len(dataset_ids) * val_fraction)))
    validation_ids = []
    for index in order:
        validation_ids.extend(components[index])
        if len(validation_ids) >= target:
            break
    validation_set = set(validation_ids)
    train_split = {pid: dataset[pid] for pid in sorted(dataset) if str(pid) not in validation_set}
    valid_split = {pid: dataset[pid] for pid in sorted(dataset) if str(pid) in validation_set}
    if set(train_split) & set(valid_split):
        raise AssertionError("Homology split overlap")
    return train_split, valid_split, {
        "protocol": "label-blind whole-homology-component train/validation split",
        "manifest": str(Path(manifest_path).resolve()),
        "identity_threshold": manifest["identity_threshold"],
        "coverage_threshold": manifest["coverage_threshold"],
        "split_seed": split_seed,
        "validation_target_proteins": target,
        "train_proteins": len(train_split),
        "validation_proteins": len(valid_split),
        "validation_components": sum(1 for component in components if set(component) <= validation_set),
        "labels_used_for_assignment": False,
    }
