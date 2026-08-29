"""Data loading and inference helpers shared by evaluation scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataloader import ProDatasetDual, graph_collate_dual
from metrics import probability_column


def generate_dataframe(dataset):
    rows = [
        {"ID": protein_id, "sequence": item[0], "label": item[1]}
        for protein_id, item in dataset.items()
    ]
    return pd.DataFrame(rows)


def split_train_validation(dataset, val_fraction=0.15, split_seed=2026):
    """Create the deterministic protein-level validation split."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    ids = sorted(dataset)
    rng = np.random.default_rng(split_seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    validation_size = max(1, int(round(len(shuffled) * val_fraction)))
    validation_ids = set(shuffled[:validation_size])
    train_split = {key: dataset[key] for key in ids if key not in validation_ids}
    validation_split = {key: dataset[key] for key in ids if key in validation_ids}
    return train_split, validation_split


def make_loader(dataset, psepos_path, full_dir, selective_dir, shuffle=False):
    return DataLoader(
        dataset=ProDatasetDual(
            generate_dataframe(dataset),
            psepos_path=str(psepos_path),
            hypernodes=3,
            hypergraph_dir_full=str(full_dir),
            hypergraph_dir_selective=str(selective_dir),
        ),
        batch_size=1,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=graph_collate_dual,
        pin_memory=False,
    )


def batch_to_device(data, device):
    (
        sequence_name,
        labels,
        node_features,
        virtual_node_features,
        pos,
        virtual_pos,
        edge_index,
        a2v_edge_index,
        v2a_edge_index,
        hypergraph_full,
        hypergraph_selective,
    ) = data
    return (
        sequence_name,
        labels.to(device).squeeze().long(),
        node_features.float().to(device),
        virtual_node_features.float().to(device),
        pos.float().to(device),
        virtual_pos.float().to(device),
        edge_index.long().to(device),
        a2v_edge_index.long().to(device),
        v2a_edge_index.long().to(device),
        hypergraph_full[0],
        hypergraph_selective[0],
    )


def forward_batch(model, data, device):
    (
        _,
        labels,
        node_features,
        virtual_node_features,
        pos,
        virtual_pos,
        edge_index,
        a2v_edge_index,
        v2a_edge_index,
        hypergraph_full,
        hypergraph_selective,
    ) = batch_to_device(data, device)
    logits = model(
        node_features,
        pos,
        virtual_node_features,
        virtual_pos,
        edge_index,
        a2v_edge_index,
        v2a_edge_index,
        hypergraph_full,
        hypergraph_selective,
    )
    return logits, labels


def predict_loader(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for data in loader:
            protein_id = str(data[0][0])
            logits, labels = forward_batch(model, data, device)
            probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            label_values = labels.detach().cpu().numpy().astype(int)
            rows.append(
                pd.DataFrame(
                    {
                        "protein_id": [protein_id] * len(label_values),
                        "label": label_values,
                        "selected_probability": probabilities,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def align_prediction(frame, anchor):
    key_columns = ["protein_id", "label"]
    if frame[key_columns].reset_index(drop=True).equals(
        anchor[key_columns].reset_index(drop=True)
    ):
        return probability_column(frame)

    aligned = frame.copy()
    aligned["residue_order"] = aligned.groupby("protein_id", sort=False).cumcount()
    anchor_key = anchor[key_columns].copy()
    anchor_key["residue_order"] = anchor_key.groupby("protein_id", sort=False).cumcount()
    merged = anchor_key.merge(
        aligned[
            ["protein_id", "label", "residue_order", "selected_probability"]
        ],
        on=["protein_id", "label", "residue_order"],
        how="left",
        validate="one_to_one",
    )
    if merged["selected_probability"].isna().any():
        raise ValueError("Model predictions do not align with the anchor rows")
    return merged["selected_probability"].to_numpy(float)
