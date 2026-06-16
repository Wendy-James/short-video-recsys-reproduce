from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _rows(path: str) -> list[dict[str, str]]:
    return list(csv.DictReader((ROOT / path).open(encoding="utf-8")))


def test_reported_metrics_are_monotonic_for_main_ablation() -> None:
    rows = _rows("experiments/ablation.csv")
    recall = [float(row["recall_at_50"]) for row in rows]
    ndcg = [float(row["ndcg_at_50"]) for row in rows]

    assert recall == sorted(recall)
    assert ndcg == sorted(ndcg)
    assert recall[0] == 0.112
    assert recall[-1] == 0.126


def test_random_split_is_not_the_reported_main_result() -> None:
    rows = _rows("experiments/metrics.csv")
    main = next(row for row in rows if row["run_id"] == "svr_003")
    random_split = next(row for row in rows if row["run_id"] == "svr_004")

    assert main["split"] == "7d_train_1d_valid"
    assert random_split["split"] == "random_split"
    assert float(random_split["recall_at_50"]) > float(main["recall_at_50"])
    assert "leakage" in random_split["note"]


def test_training_output_is_trainable_model_artifact() -> None:
    import json

    meta = json.loads((ROOT / "outputs" / "model_meta.json").read_text(encoding="utf-8"))

    assert meta["model"] == "two_tower_recall"
    assert meta["backend"] in {"numpy_sgd", "pytorch"}
    assert meta["final_loss"] > 0
    assert (ROOT / "outputs" / "user_embeddings.npy").exists()
    assert (ROOT / "outputs" / "item_embeddings.npy").exists()
