# Roadmap

This repository is maintained as a public, runnable reproduction of a short-video recommendation recall workflow. The goal is steady, inspectable iteration rather than large one-off rewrites.

## Near-term

- Add a small sequence-tower comparison for mean pooling, GRU-style aggregation, and attention pooling.
- Add a clearer negative-sampling report covering random negatives, in-batch negatives, and weak exposure negatives.
- Expand leakage checks for future item popularity, author-repeat exposure, and duplicated user history.

## Evaluation

- Keep Recall@50, NDCG@50, AUC, and tail Recall@50 as the core metric set.
- Add per-category recall buckets to make head/tail trade-offs easier to inspect.
- Keep random split only as a leakage-risk contrast, not as the main reported result.

## Engineering

- Keep `make all` CPU-friendly for quick review.
- Keep tests for data split, metric calculation, and negative-sampling assumptions.
- Prefer small PR-sized changes with updated metrics or badcase notes.
