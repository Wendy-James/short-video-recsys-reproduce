from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mixed_negative_sampling_is_documented_in_metrics() -> None:
    rows = list(csv.DictReader((ROOT / "experiments" / "metrics.csv").open(encoding="utf-8")))
    main = next(row for row in rows if row["run_id"] == "svr_003")

    assert main["negative_sampling"] == "mixed_weak_negative"
    assert main["loss"] == "sampled_softmax"


def test_weak_negative_noise_badcase_is_kept() -> None:
    rows = list(csv.DictReader((ROOT / "badcases" / "badcase_samples.csv").open(encoding="utf-8")))
    error_types = {row["error_type"] for row in rows}

    assert len(rows) >= 10
    assert "weak_negative_noise" in error_types
    assert "head_leakage" in error_types


def test_negative_sampling_script_outputs_all_strategies() -> None:
    import json

    from src import data_preprocess, negative_sampling

    data_preprocess.main()
    negative_sampling.main()
    summary = json.loads((ROOT / "outputs" / "negative_sampling_summary.json").read_text(encoding="utf-8"))
    samples = list(csv.DictReader((ROOT / "outputs" / "negative_samples.csv").open(encoding="utf-8")))

    assert samples
    assert set(summary["strategies"]) == {
        "random_negative",
        "popular_negative",
        "same_category_negative",
        "in_batch_negative",
    }
    assert {"random_negative", "popular_negative", "same_category_negative"}.issubset(samples[0])
