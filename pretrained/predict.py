"""Generate residue probabilities from the pretrained weights."""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "src"
sys.path.insert(0, str(SOURCE))

from dataloader import ProDataset, ProDatasetDual, graph_collate, graph_collate_dual  # noqa: E402
from model import HEGNNPPIS, HEGNNPPIS_Dual  # noqa: E402


BENCHMARKS = {
    "btest": ("BTest_31-6.pkl", "Test60_psepos_SC.pkl", "HEGNN_PPIS_BTest_31-6_Final.pkl"),
    "test315": ("Test_315-28.pkl", "Test315-28_psepos_SC.pkl", "HEGNN_PPIS_Test315-28_Final.pkl"),
    "ubtest": ("UBtest_31-6.pkl", "UBtest31-6_psepos_SC.pkl", "HEGNN_PPIS_UBtest_31-6_Final.pkl"),
    "test60_ensemble": ("Test_60.pkl", "Test60_psepos_SC.pkl", "HEGNN_PPIS_Ensemble_Final.pt"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=BENCHMARKS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit-proteins", type=int, default=0, help="Use a small subset for a smoke check")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def make_models(benchmark, weight_path, device):
    if benchmark == "test60_ensemble":
        bundle = torch.load(weight_path, map_location="cpu", weights_only=True)
        if bundle.get("format") != "HEGNN-PPIS-checkpoint-ensemble-v1":
            raise ValueError("Unexpected ensemble format")
        models = []
        for state in bundle["state_dicts"]:
            model = HEGNNPPIS_Dual(
                in_dim=67, in_edge_dim=1, hidden_dim=67, layers=4,
                alpha_full=0.05, alpha_selective=0.10,
            ).to(device)
            model.load_state_dict(state, strict=True)
            model.eval()
            models.append(model)
        return models
    model = HEGNNPPIS(in_dim=67, in_edge_dim=1, hidden_dim=67, layers=4).to(device)
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return [model]


def predict_batch(model, batch, device, dual):
    x = batch[2].float().to(device)
    virtual_x = batch[3].float().to(device)
    pos = batch[4].float().to(device)
    virtual_pos = batch[5].float().to(device)
    edge = batch[6].long().to(device)
    a2v = batch[7].long().to(device)
    v2a = batch[8].long().to(device)
    if dual:
        logits = model(x, pos, virtual_x, virtual_pos, edge, a2v, v2a, batch[9][0], batch[10][0])
    else:
        logits = model(x, pos, virtual_x, virtual_pos, edge, a2v, v2a, batch[9][0])
    return torch.softmax(logits, dim=1)[:, 1]


def main():
    args = parse_args()
    if args.limit_proteins < 0:
        raise ValueError("--limit-proteins must be nonnegative")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    dataset_name, psepos_name, weights_name = BENCHMARKS[args.benchmark]
    with (SOURCE / "Dataset" / dataset_name).open("rb") as handle:
        dataset = pickle.load(handle)
    if args.limit_proteins:
        dataset = dict(list(dataset.items())[:args.limit_proteins])
    frame = pd.DataFrame(
        [{"ID": key, "sequence": value[0], "label": [0] * len(value[0])} for key, value in dataset.items()]
    )
    common = {
        "psepos_path": str(SOURCE / "Feature/psepos" / psepos_name),
        "hypernodes": 3,
        "random_virtual_rotations": False,
    }
    dual = args.benchmark == "test60_ensemble"
    if dual:
        samples = ProDatasetDual(
            frame,
            hypergraph_dir_full=str(SOURCE / "Graph/SC/hypergraph"),
            hypergraph_dir_selective=str(SOURCE / "Graph/SC/hypergraph_surface/hotspot_surface_r10"),
            **common,
        )
        collate = graph_collate_dual
    else:
        samples = ProDataset(frame, hypergraph_dir=str(SOURCE / "Graph/SC/hypergraph"), **common)
        collate = graph_collate
    if len(samples) != len(frame):
        raise ValueError("Some proteins lack input coordinates")
    loader = DataLoader(samples, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    models = make_models(args.benchmark, HERE / weights_name, device)
    output = args.output or ROOT / "output/pretrained" / f"{args.benchmark}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("protein_id", "residue_index", "probability", "prediction_at_0_5"))
        with torch.no_grad():
            for batch in loader:
                probabilities = torch.stack(
                    [predict_batch(model, batch, device, dual) for model in models]
                ).mean(dim=0).cpu().tolist()
                protein_id = str(batch[0][0])
                for index, probability in enumerate(probabilities):
                    writer.writerow((protein_id, index, probability, int(probability >= 0.5)))
                    total += 1
    print(f"Wrote {total} residue predictions to {output}")


if __name__ == "__main__":
    main()
