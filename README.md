# short-video-recsys-reproduce

Short-video recommendation reproduction project for algorithm internship interviews. The project focuses on a credible offline recommendation workflow: sample construction, time-based split, Two-Tower retrieval, Faiss-style top-k evaluation, feature ablation, leakage checks, and badcase analysis.

This repository does not contain private or platform data. It uses a pseudo schema and reproducible toy scripts to document the experiment protocol behind the resume project.

## Project Positioning

- Scenario: short-video feed recommendation / content recommendation / recall and ranking.
- Core problem: random split and head-item leakage can make offline metrics look better than real future prediction.
- Main model: Two-Tower retrieval baseline with user/item embeddings.
- Main metrics: Recall@50, NDCG@50, AUC, tail Recall@50.
- Main evidence: `experiments.csv`, `badcases.csv`, `data_schema.md`, runnable scripts.

## Data Protocol

The resume project uses a public short-video recommendation style schema inspired by KuaiRec / KuaiRand / Tenrec:

- Positive feedback: click, finish, like, favorite.
- Weak negative feedback: exposed but not clicked, or very low dwell time.
- Split: 7-day train / 1-day validation time split.
- Candidate items: about 30k videos in the resume-scale experiment.
- This repo provides pseudo records only; no real user data is included.

## Repository Structure

```text
.
├── README.md
├── data_schema.md
├── experiments.csv
├── badcases.csv
├── requirements.txt
├── scripts/
│   ├── build_pseudo_data.py
│   ├── train_two_tower.py
│   └── evaluate_recall.py
└── assets/
    └── results_summary.md
```

## Quick Start

```bash
python scripts/build_pseudo_data.py
python scripts/train_two_tower.py
python scripts/evaluate_recall.py
```

The scripts create a tiny pseudo dataset and deterministic toy metrics under `outputs/`. They are not intended to reproduce the resume numbers; they document the experiment workflow and field definitions.

## Interview Talking Points

1. Why time split: online recommendation uses past behavior to predict future behavior; random split can leak future popularity and user intent.
2. Why Recall@50: retrieval cares whether the positive item enters the candidate set; AUC alone can hide head-item bias.
3. Why IndexFlatIP: for about 30k videos, exact inner-product search is simple, stable, and avoids approximate-index noise.
4. Why weak negatives: exposure without click is noisy, so negative sampling and multi-metric evaluation are necessary.
5. Why tail metrics: head videos can dominate global metrics, so tail Recall@50 checks whether the model only learns popularity.

