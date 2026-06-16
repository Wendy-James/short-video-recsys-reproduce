# Short Video Recommendation Evidence Pack

## Interview positioning

This repository supports the resume project "short-video recommendation offline reproduction". It is positioned as an offline recall evaluation loop, not an online production recommendation system.

## Problem

Short-video recall needs to retrieve a candidate set before ranking. The project focuses on exposure-level sample construction, time split, Two-Tower recall, Faiss-style TopK retrieval, metric evaluation, ablation, leakage check, and badcase analysis.

## What I can explain

- Why one exposure is treated as one sample.
- Why click/finish/like/favorite are stronger positive feedback than raw click alone.
- Why exposure-not-click is only a weak negative.
- Why 7-day train / 1-day validation is safer than random split.
- Why Recall@50 must be paired with NDCG@50, tail Recall@50, popular-item ratio, and badcase review.

## Main files

- `data_schema.md`: public-field schema and label design.
- `run_all.py`: one-command pipeline for pseudo data, training, retrieval, and evaluation.
- `src/train_twotower.py`: lightweight trainable Two-Tower implementation.
- `src/negative_sampling.py`: random, popular, same-category, and in-batch negative sampling notes.
- `src/evaluate_recall.py`: Recall@50, NDCG@50, AUC, and tail Recall@50 evaluation.
- `experiments/metrics.csv`: baseline metrics.
- `experiments/ablation.csv`: ablation records.
- `badcases/badcase_samples.csv`: failure examples.
