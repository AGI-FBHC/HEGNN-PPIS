"""Fast import and model-construction check."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_ablation import HEGNNPPISDualAblation  # noqa: E402


def main():
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
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count <= 0:
        raise SystemExit("Model has no trainable parameters")
    print(f"Smoke test passed: {parameter_count:,} parameters")


if __name__ == "__main__":
    main()
