"""Build a Faiss-style IndexFlatIP matrix from trained item embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main() -> None:
    item_path = OUT / "item_embeddings.npy"
    if not item_path.exists():
        raise SystemExit("Run src/train_twotower.py first.")

    item_matrix = np.load(item_path).astype("float32")
    item_matrix /= np.linalg.norm(item_matrix, axis=1, keepdims=True) + 1e-8
    np.save(OUT / "item_index_flatip.npy", item_matrix)
    index_meta = {
        "index_type": "IndexFlatIP",
        "num_items": int(item_matrix.shape[0]),
        "embedding_dim": int(item_matrix.shape[1]),
        "source": "outputs/item_embeddings.npy",
        "note": "Exact inner-product retrieval matrix; swap to faiss.IndexFlatIP at larger item scale.",
    }
    (OUT / "faiss_index_meta.json").write_text(json.dumps(index_meta, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'item_index_flatip.npy'}")


if __name__ == "__main__":
    main()
