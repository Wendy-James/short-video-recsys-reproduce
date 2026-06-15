"""Build a small pseudo interaction log for offline recommendation experiments."""

from __future__ import annotations

import csv
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main() -> None:
    random.seed(42)
    OUT.mkdir(exist_ok=True)
    path = OUT / "interaction_log.csv"
    fields = [
        "user_id",
        "video_id",
        "author_id",
        "category_id",
        "day",
        "is_click",
        "is_finish",
        "is_like",
        "is_favorite",
        "dwell_time",
        "video_duration",
        "position",
        "label_recall",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(600):
            user = f"u_{random.randint(1, 80):04d}"
            video = f"v_{random.randint(1, 240):05d}"
            duration = random.choice([8, 12, 15, 30, 45, 60])
            dwell = round(random.random() * duration, 2)
            click = int(random.random() < 0.18)
            finish = int(click and dwell / duration > 0.85)
            like = int(click and random.random() < 0.12)
            favorite = int(click and random.random() < 0.05)
            label = int(click or finish or like or favorite)
            writer.writerow(
                {
                    "user_id": user,
                    "video_id": video,
                    "author_id": f"a_{random.randint(1, 50):03d}",
                    "category_id": f"c_{random.randint(1, 12):02d}",
                    "day": random.randint(1, 8),
                    "is_click": click,
                    "is_finish": finish,
                    "is_like": like,
                    "is_favorite": favorite,
                    "dwell_time": dwell,
                    "video_duration": duration,
                    "position": random.randint(1, 20),
                    "label_recall": label,
                }
            )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
