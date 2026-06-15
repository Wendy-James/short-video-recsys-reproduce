"""Train a lightweight Two-Tower recall model on the pseudo interaction log.

The public dataset is intentionally small, but the training path is real:
user/video ids are mapped to embedding tables, exposure labels are trained with
binary cross entropy, and the learned user/item vectors are exported for exact
inner-product retrieval.

If PyTorch is installed, the script uses a tiny PyTorch module. Otherwise it
falls back to the same objective implemented with NumPy SGD so `make all` stays
CPU-friendly in minimal environments.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

try:  # Optional: the repo should still run in lightweight review environments.
    import torch
    from torch import nn
except Exception:  # pragma: no cover - exercised only when torch is available.
    torch = None
    nn = None


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
EMBED_DIM = 32
EPOCHS = 18
LR = 0.08
SEED = 2026


def _load_rows() -> list[dict[str, str]]:
    data_path = OUT / "interaction_log.csv"
    if not data_path.exists():
        raise SystemExit("Run src/data_preprocess.py first.")
    return list(csv.DictReader(data_path.open(encoding="utf-8")))


def _mappings(rows: list[dict[str, str]]) -> tuple[dict[str, int], dict[str, int]]:
    users = {row["user_id"] for row in rows}
    items = {row["video_id"] for row in rows}
    return {v: i for i, v in enumerate(sorted(users))}, {v: i for i, v in enumerate(sorted(items))}


def _arrays(rows: list[dict[str, str]], user2idx: dict[str, int], item2idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = [row for row in rows if int(row["day"]) <= 7]
    user_idx = np.array([user2idx[row["user_id"]] for row in train], dtype=np.int64)
    item_idx = np.array([item2idx[row["video_id"]] for row in train], dtype=np.int64)
    labels = np.array([float(row["label_recall"]) for row in train], dtype=np.float32)
    return user_idx, item_idx, labels


def _train_numpy(user_idx: np.ndarray, item_idx: np.ndarray, labels: np.ndarray, num_users: int, num_items: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    rng = np.random.default_rng(SEED)
    user_emb = rng.normal(0, 0.08, size=(num_users, EMBED_DIM)).astype("float32")
    item_emb = rng.normal(0, 0.08, size=(num_items, EMBED_DIM)).astype("float32")
    losses: list[float] = []

    for _ in range(EPOCHS):
        order = rng.permutation(len(labels))
        epoch_losses = []
        for idx in order:
            u = user_idx[idx]
            v = item_idx[idx]
            y = labels[idx]
            score = float(np.dot(user_emb[u], item_emb[v]))
            pred = 1.0 / (1.0 + np.exp(-score))
            grad = pred - y
            u_old = user_emb[u].copy()
            item_emb[v] -= LR * grad * u_old
            user_emb[u] -= LR * grad * item_emb[v]
            epoch_losses.append(-(y * np.log(pred + 1e-8) + (1 - y) * np.log(1 - pred + 1e-8)))
        losses.append(float(np.mean(epoch_losses)))

    user_emb /= np.linalg.norm(user_emb, axis=1, keepdims=True) + 1e-8
    item_emb /= np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-8
    return user_emb, item_emb, losses


def _train_torch(user_idx: np.ndarray, item_idx: np.ndarray, labels: np.ndarray, num_users: int, num_items: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    assert torch is not None and nn is not None
    torch.manual_seed(SEED)

    class TwoTower(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.user = nn.Embedding(num_users, EMBED_DIM)
            self.item = nn.Embedding(num_items, EMBED_DIM)

        def forward(self, users: "torch.Tensor", items: "torch.Tensor") -> "torch.Tensor":
            u = nn.functional.normalize(self.user(users), dim=1)
            v = nn.functional.normalize(self.item(items), dim=1)
            return (u * v).sum(dim=1)

    model = TwoTower()
    optim = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = nn.BCEWithLogitsLoss()
    users = torch.tensor(user_idx, dtype=torch.long)
    items = torch.tensor(item_idx, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float32)
    losses: list[float] = []
    for _ in range(EPOCHS):
        optim.zero_grad()
        logits = model(users, items)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
        losses.append(float(loss.detach().cpu()))

    user_emb = nn.functional.normalize(model.user.weight.detach(), dim=1).cpu().numpy()
    item_emb = nn.functional.normalize(model.item.weight.detach(), dim=1).cpu().numpy()
    return user_emb.astype("float32"), item_emb.astype("float32"), losses


def main() -> None:
    rows = _load_rows()
    user2idx, item2idx = _mappings(rows)
    user_idx, item_idx, labels = _arrays(rows, user2idx, item2idx)
    if torch is not None:
        user_emb, item_emb, losses = _train_torch(user_idx, item_idx, labels, len(user2idx), len(item2idx))
        backend = "pytorch"
    else:
        user_emb, item_emb, losses = _train_numpy(user_idx, item_idx, labels, len(user2idx), len(item2idx))
        backend = "numpy_sgd"

    np.save(OUT / "user_embeddings.npy", user_emb)
    np.save(OUT / "item_embeddings.npy", item_emb)
    (OUT / "user_id_map.json").write_text(json.dumps(user2idx, indent=2), encoding="utf-8")
    (OUT / "item_id_map.json").write_text(json.dumps(item2idx, indent=2), encoding="utf-8")
    meta = {
        "model": "two_tower_recall",
        "backend": backend,
        "embedding_dim": EMBED_DIM,
        "loss": "binary_cross_entropy",
        "index_type": "IndexFlatIP",
        "num_users": len(user2idx),
        "num_items": len(item2idx),
        "train_rows": int(len(labels)),
        "positive_rate": round(float(labels.mean()), 4),
        "final_loss": round(losses[-1], 6),
    }
    (OUT / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (OUT / "training_curve.json").write_text(json.dumps({"loss": [round(x, 6) for x in losses]}, indent=2), encoding="utf-8")
    print(f"trained {meta['model']} with {backend}; final_loss={meta['final_loss']}")
    print(f"wrote model metadata to {OUT / 'model_meta.json'}")


if __name__ == "__main__":
    main()
