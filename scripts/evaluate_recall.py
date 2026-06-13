"""Create deterministic toy retrieval metrics for documentation."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main() -> None:
    if not (OUT / "model_meta.json").exists():
        raise SystemExit("Run scripts/train_two_tower.py first.")
    metrics = [
        {
            "split": "7d_train_1d_valid",
            "auc": "0.703",
            "recall_at_50": "0.126",
            "ndcg_at_50": "0.079",
            "tail_recall_at_50": "0.071",
            "note": "toy report mirrors resume metric fields, not private data",
        }
    ]
    path = OUT / "metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

