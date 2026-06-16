"""Build negative-sampling diagnostics for the pseudo recommendation log.

The training script consumes exposure labels directly. This module makes the
negative-sampling assumptions inspectable:

- random negatives: exposed but unclicked/low-dwell items
- popular negatives: head items from the training window
- same-category negatives: harder negatives with category overlap
- in-batch negatives: other items in the same mini-batch
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SEED = 2026


def _load_rows() -> list[dict[str, str]]:
    path = OUT / "interaction_log.csv"
    if not path.exists():
        raise SystemExit("Run src/data_preprocess.py first.")
    return list(csv.DictReader(path.open(encoding="utf-8")))


def main() -> None:
    random.seed(SEED)
    rows = [row for row in _load_rows() if int(row["day"]) <= 7]
    positives = [row for row in rows if int(row["label_recall"]) == 1]
    weak_negatives = [row for row in rows if int(row["label_recall"]) == 0]
    item_pop = Counter(row["video_id"] for row in rows)
    category_to_items: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        category_to_items[row["category_id"]].append(row["video_id"])

    samples = []
    for pos in positives[:80]:
        random_neg = random.choice(weak_negatives)["video_id"] if weak_negatives else ""
        popular_neg = item_pop.most_common(1)[0][0] if item_pop else ""
        same_category_pool = [item for item in category_to_items[pos["category_id"]] if item != pos["video_id"]]
        same_category_neg = random.choice(same_category_pool) if same_category_pool else random_neg
        samples.append(
            {
                "user_id": pos["user_id"],
                "positive_video_id": pos["video_id"],
                "category_id": pos["category_id"],
                "random_negative": random_neg,
                "popular_negative": popular_neg,
                "same_category_negative": same_category_neg,
                "in_batch_negative_note": "other items in the same mini-batch can be reused as negatives",
            }
        )

    path = OUT / "negative_samples.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)

    summary = {
        "num_train_rows": len(rows),
        "num_positive_rows": len(positives),
        "num_weak_negative_rows": len(weak_negatives),
        "strategies": ["random_negative", "popular_negative", "same_category_negative", "in_batch_negative"],
        "positive_rate": round(len(positives) / max(len(rows), 1), 4),
    }
    (OUT / "negative_sampling_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
