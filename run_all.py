"""Run the full short-video recommendation reproduction pipeline."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMMANDS = [
    ["python3", "src/data_preprocess.py"],
    ["python3", "src/negative_sampling.py"],
    ["python3", "src/train_twotower.py"],
    ["python3", "src/build_faiss_index.py"],
    ["python3", "src/evaluate_recall.py"],
    ["python3", "src/train_ranker.py"],
    ["python3", "-m", "pytest", "-q"],
]


def main() -> None:
    for command in COMMANDS:
        print("$ " + " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)
    experiments_dir = ROOT / "experiments"
    experiments_dir.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "outputs" / "metrics.csv", experiments_dir / "generated_metrics.csv")
    print("wrote experiments/generated_metrics.csv")


if __name__ == "__main__":
    main()
