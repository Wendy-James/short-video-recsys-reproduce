# Experiment Log

## Goal

Build a defensible offline recommendation retrieval reproduction for algorithm-intern interviews. The project should answer:

- What data fields are used?
- How is train/validation split done?
- What does the Two-Tower model learn?
- Why does Recall@50 improve?
- How do we know the improvement is not leakage or head-item bias?

## Experiment Summary

| Run | Change | AUC | Recall@50 | NDCG@50 | Tail Recall@50 | Decision |
|---|---|---:|---:|---:|---:|---|
| `svr_001` | ID/category/history baseline | 0.681 | 0.112 | 0.071 | 0.058 | Keep as baseline |
| `svr_002` | Add recent behavior sequence | 0.694 | 0.119 | 0.075 | 0.064 | Keep, improves short-term interest |
| `svr_003` | Add time decay and mixed negatives | 0.703 | 0.126 | 0.079 | 0.071 | Main resume result |
| `svr_004` | Same setup with random split | 0.724 | 0.139 | 0.086 | 0.074 | Do not report as main result due to leakage risk |

## Ablation Interpretation

The important interview point is that the improvement is distributed across feature and sampling decisions:

- Recent sequence improves immediate interest matching.
- Time decay matters because short-video feedback is highly time-sensitive.
- User-category cross features improve candidate coverage in frequent preference buckets.
- Mixed weak negatives make the training task less trivial than random negatives.
- Tail Recall@50 is tracked so the model does not only improve by ranking head videos higher.

## Badcase Review

| Case | Root cause | Fix / next action |
|---|---|---|
| Head leakage | Random split makes trending author/video appear in train and valid | Report time split only |
| Author repeat | Same creator dominates top-k | Add author-repeat feature and diversity check |
| Cold start | New videos have weak ID embeddings | Use category, author, duration, and content-side buckets |
| Weak-negative noise | Exposure-not-click can be position or context noise | Control sampling ratio and review false negatives |

## Safe Resume Wording

Use:

> Reproduced an offline short-video recommendation recall/ranking workflow with pseudo/public-field exposure logs, 7-day train / 1-day validation split, Two-Tower recall, Faiss-style top-k retrieval, and Recall@50/NDCG@50/tail Recall@50 evaluation.

Avoid:

> Built an online recommendation system, improved ByteDance metrics, or trained on private platform traffic.
