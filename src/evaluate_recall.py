"""Evaluate trained Two-Tower embeddings with Recall@K and NDCG@K."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
K = 50


def _load_rows() -> list[dict[str, str]]:
    return list(csv.DictReader((OUT / "interaction_log.csv").open(encoding="utf-8")))


def _recall_ndcg(ranked: list[int], positives: set[int], k: int) -> tuple[float, float]:
    if not positives:
        return 0.0, 0.0
    topk = ranked[:k]
    hits = [1 if item in positives else 0 for item in topk]
    recall = sum(hits) / len(positives)
    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
    ideal_hits = [1] * min(len(positives), k)
    idcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(ideal_hits))
    return recall, dcg / (idcg + 1e-8)


def _auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += float(p > n) + 0.5 * float(p == n)
    return wins / (len(pos) * len(neg))


def main() -> None:
    required = [OUT / "user_embeddings.npy", OUT / "item_index_flatip.npy", OUT / "user_id_map.json", OUT / "item_id_map.json"]
    if not all(path.exists() for path in required):
        raise SystemExit("Run src/train_twotower.py and src/build_faiss_index.py first.")

    user_emb = np.load(OUT / "user_embeddings.npy")
    item_emb = np.load(OUT / "item_index_flatip.npy")
    user2idx = json.loads((OUT / "user_id_map.json").read_text(encoding="utf-8"))
    item2idx = json.loads((OUT / "item_id_map.json").read_text(encoding="utf-8"))
    rows = _load_rows()
    item_counts: dict[int, int] = defaultdict(int)
    positives_by_user: dict[int, set[int]] = defaultdict(set)
    valid_pairs: list[tuple[int, int, int]] = []
    for row in rows:
        item_idx = item2idx[row["video_id"]]
        if int(row["day"]) <= 7:
            item_counts[item_idx] += 1
        else:
            user_idx = user2idx[row["user_id"]]
            label = int(row["label_recall"])
            valid_pairs.append((user_idx, item_idx, label))
            if label == 1:
                positives_by_user[user_idx].add(item_idx)

    scores = user_emb @ item_emb.T
    recalls, ndcgs, tail_recalls = [], [], []
    count_values = np.array(list(item_counts.values()) or [0])
    # Resume口径：长尾视频定义为曝光量后80%内容，即去掉最热门约20%。
    tail_cutoff = float(np.quantile(count_values, 0.8))
    for user_idx, positives in positives_by_user.items():
        ranked = list(np.argsort(-scores[user_idx]))
        recall, ndcg = _recall_ndcg(ranked, positives, K)
        recalls.append(recall)
        ndcgs.append(ndcg)
        tail_pos = {item for item in positives if item_counts.get(item, 0) <= tail_cutoff}
        if tail_pos:
            tail_recall, _ = _recall_ndcg(ranked, tail_pos, K)
            tail_recalls.append(tail_recall)

    pair_scores = [float(scores[u, i]) for u, i, _ in valid_pairs]
    pair_labels = [y for _, _, y in valid_pairs]
    metrics = {
        "split": "7d_train_1d_valid",
        "auc": f"{_auc(pair_scores, pair_labels):.3f}",
        "recall_at_50": f"{(np.mean(recalls) if recalls else 0.0):.3f}",
        "ndcg_at_50": f"{(np.mean(ndcgs) if ndcgs else 0.0):.3f}",
        "tail_recall_at_50": f"{(np.mean(tail_recalls) if tail_recalls else 0.0):.3f}",
        "note": "computed from trained two-tower embeddings on pseudo validation rows",
    }
    path = OUT / "metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
