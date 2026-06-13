"""Toy Two-Tower training placeholder.

This script intentionally keeps the implementation light. It creates deterministic
user/item embeddings from pseudo IDs so the repository has a runnable training
artifact without requiring private data or GPU resources.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def stable_vec(key: str, dim: int = 32) -> np.ndarray:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim)
    return vec / (np.linalg.norm(vec) + 1e-8)


def main() -> None:
    data_path = OUT / "interaction_log.csv"
    if not data_path.exists():
        raise SystemExit("Run scripts/build_pseudo_data.py first.")

    users, videos = set(), set()
    with data_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            users.add(row["user_id"])
            videos.add(row["video_id"])

    user_emb = {u: stable_vec(u).round(6).tolist() for u in sorted(users)}
    item_emb = {v: stable_vec(v).round(6).tolist() for v in sorted(videos)}
    model = {
        "model": "toy_two_tower_embedding_export",
        "embedding_dim": 32,
        "loss_candidates": ["bce", "sampled_softmax"],
        "index_type": "IndexFlatIP",
        "num_users": len(user_emb),
        "num_items": len(item_emb),
    }
    (OUT / "model_meta.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    (OUT / "user_embeddings.json").write_text(json.dumps(user_emb)[:2000], encoding="utf-8")
    (OUT / "item_embeddings.json").write_text(json.dumps(item_emb)[:2000], encoding="utf-8")
    print(f"wrote model metadata to {OUT / 'model_meta.json'}")


if __name__ == "__main__":
    main()

