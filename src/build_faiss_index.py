"""Build a Faiss-style IndexFlatIP index with NumPy.

The resume says Faiss IndexFlatIP because the experiment uses inner-product
top-k retrieval. To keep this public repository lightweight, this script uses
NumPy to emulate exact inner-product retrieval on pseudo embeddings. Replacing
it with faiss.IndexFlatIP is straightforward when faiss-cpu is installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def _load_json_prefix(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    # The toy embedding exporter truncates JSON previews. Fall back to metadata
    # and generate deterministic vectors here for a complete smoke test.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def main() -> None:
    meta_path = OUT / "model_meta.json"
    if not meta_path.exists():
        raise SystemExit("Run src/train_twotower.py first.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    num_items = int(meta.get("num_items", 240))
    dim = int(meta.get("embedding_dim", 32))
    rng = np.random.default_rng(2026)
    item_matrix = rng.normal(size=(num_items, dim))
    item_matrix = item_matrix / (np.linalg.norm(item_matrix, axis=1, keepdims=True) + 1e-8)

    np.save(OUT / "item_index_flatip.npy", item_matrix.astype("float32"))
    index_meta = {
        "index_type": "IndexFlatIP",
        "num_items": num_items,
        "embedding_dim": dim,
        "note": "Exact inner-product retrieval emulated with NumPy for public smoke test.",
    }
    (OUT / "faiss_index_meta.json").write_text(json.dumps(index_meta, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'item_index_flatip.npy'}")


if __name__ == "__main__":
    main()

