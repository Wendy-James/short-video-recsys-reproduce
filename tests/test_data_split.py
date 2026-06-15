from __future__ import annotations

import csv
from pathlib import Path

from src import data_preprocess


ROOT = Path(__file__).resolve().parents[1]


def test_interaction_log_has_time_split_boundary() -> None:
    data_preprocess.main()
    path = ROOT / "outputs" / "interaction_log.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))

    assert rows
    days = {int(row["day"]) for row in rows}
    assert days.issubset(set(range(1, 9)))
    assert any(int(row["day"]) <= 7 for row in rows)
    assert any(int(row["day"]) == 8 for row in rows)


def test_interaction_log_has_positive_and_weak_negative_samples() -> None:
    path = ROOT / "outputs" / "interaction_log.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    labels = {int(row["label_recall"]) for row in rows}

    assert labels == {0, 1}
    assert any(int(row["is_click"]) == 0 and int(row["label_recall"]) == 0 for row in rows)
