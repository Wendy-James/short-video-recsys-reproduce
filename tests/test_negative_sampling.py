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

    assert "weak_negative_noise" in error_types
    assert "head_leakage" in error_types
