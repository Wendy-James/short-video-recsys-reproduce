"""Toy ranking-stage script for feature crossing and AUC-style reporting."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def main() -> None:
    data_path = OUT / "interaction_log.csv"
    if not data_path.exists():
        raise SystemExit("Run src/data_preprocess.py first.")

    rows = []
    with data_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dwell_ratio = float(row["dwell_time"]) / max(float(row["video_duration"]), 1.0)
            category_hash = int(row["category_id"].split("_")[-1])
            position = int(row["position"])
            score = sigmoid(1.8 * dwell_ratio - 0.04 * position + 0.02 * category_hash)
            rows.append(
                {
                    "user_id": row["user_id"],
                    "video_id": row["video_id"],
                    "label": row["label_recall"],
                    "rank_score": f"{score:.6f}",
                    "features": "dwell_ratio,position,category_bucket",
                }
            )

    path = OUT / "ranker_scores.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

