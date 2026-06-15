#!/usr/bin/env bash
set -euo pipefail

python3 src/data_preprocess.py
python3 src/train_twotower.py
python3 src/build_faiss_index.py
python3 src/evaluate_recall.py
python3 src/train_ranker.py
python3 -m pytest -q
