# Short Video Recommendation Reproduction

**Two-Tower Recall · Faiss TopK Retrieval · Time Split · Recall@50/NDCG@50 · Leakage Check**

This repository is the evidence-chain project for my resume item **"短视频推荐系统训练复现：召回排序、特征交叉与泄漏检查"**. It is not a CTR-only toy demo and not an online production system. It documents a reproducible offline recommendation workflow with pseudo/anonymized data:

- exposure-level short-video recommendation samples
- 7-day train / 1-day validation time split
- DSSM / Two-Tower recall baseline
- sampled softmax / BCE training protocols
- Faiss-style `IndexFlatIP` top-k retrieval
- Recall@50, NDCG@50, AUC, tail Recall@50
- leakage checks, ablation records, and badcase analysis

No company-internal data is included. The pseudo schema follows public short-video recommendation dataset fields such as KuaiRec / KuaiRand / Tenrec.

## Why This Repo Exists

The resume project claims a recommendation retrieval/ranking pipeline, so this repo keeps the corresponding evidence chain:

| Resume Claim | Repository Evidence |
|---|---|
| 20w exposure samples, user/item/time fields | `data_schema.md`, `src/data_preprocess.py`, pseudo `outputs/interaction_log.csv` |
| 7-day train / 1-day validation | `data_schema.md`, `experiments/metrics.csv` |
| DSSM / Two-Tower | `src/train_twotower.py`, `outputs/model_meta.json` |
| Faiss IndexFlatIP top-k retrieval | `src/build_faiss_index.py`, `src/evaluate_recall.py` |
| Recall@50 / NDCG@50 / tail Recall@50 | `experiments/metrics.csv`, `assets/results_summary.md` |
| Feature ablation and leakage check | `experiments/ablation.csv`, `notebooks/data_distribution.ipynb` |
| Badcase review | `badcases/badcase_samples.csv` |

## Repository Structure

```text
.
├── README.md
├── data_schema.md
├── requirements.txt
├── src/
│   ├── data_preprocess.py
│   ├── train_twotower.py
│   ├── build_faiss_index.py
│   ├── evaluate_recall.py
│   └── train_ranker.py
├── experiments/
│   ├── metrics.csv
│   └── ablation.csv
├── badcases/
│   └── badcase_samples.csv
├── notebooks/
│   └── data_distribution.ipynb
├── assets/
│   └── results_summary.md
└── outputs/
```

## Quick Start

```bash
python3 src/data_preprocess.py
python3 src/train_twotower.py
python3 src/build_faiss_index.py
python3 src/evaluate_recall.py
python3 src/train_ranker.py
```

The scripts are CPU-friendly and run on pseudo data. They are meant to demonstrate a credible experiment workflow, not to expose private platform data.

## Offline Metrics Used in the Resume

| Setup | Split | Recall@50 | NDCG@50 | Tail Recall@50 | Note |
|---|---|---:|---:|---:|---|
| Two-Tower + mean pooling baseline | 7d train / 1d valid | 0.112 | 0.071 | 0.058 | ID/category/history baseline |
| + sequence + time decay + mixed negatives | 7d train / 1d valid | 0.126 | 0.079 | 0.071 | Main resume result |
| Same setup with random split | random split | 0.139 | 0.086 | 0.074 | Higher but not reported due to leakage risk |

## Key Interview Answers

### Where does the dataset come from?

This is an offline reproduction based on public short-video recommendation dataset field conventions. It is **not company-internal data**. Fields include user, item/video, exposure, click, finish, like, favorite, dwell time, author, category, and timestamp.

### Why time split?

Recommendation systems use past behavior to predict future feedback. Random split can leak future popularity and near-duplicate user behavior, making AUC look better. The resume only reports the 7-day train / 1-day validation split.

### What is inside the user tower?

User ID embedding, recent behavior sequence, category preference, author interaction, time-decay features, and history statistics.

### What is inside the item tower?

Video ID, author ID, category ID, publish time bucket, duration bucket, and content-side features. These help cold-start videos where pure ID embeddings are weak.

### Why Recall@50 instead of only AUC?

Two-Tower is a retrieval module. Its job is to place relevant candidates into top-k. AUC measures pairwise ranking quality, but it can hide head-item bias. Therefore Recall@50, NDCG@50, and tail Recall@50 are reported together.

### Why IndexFlatIP?

The offline candidate size is around 30k videos, so exact inner-product search is sufficient and avoids approximate-index noise. IVF/HNSW would be considered when the item scale grows to millions.

## What This Repo Does Not Claim

- It is not an online ByteDance/TikTok system.
- It does not contain private user data.
- It does not claim online A/B lift.
- It is a reproducible evidence-chain repo for recommendation retrieval/ranking interview discussion.

