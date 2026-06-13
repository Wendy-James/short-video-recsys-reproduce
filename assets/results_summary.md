# Results Summary

Resume-scale experiment record:

| model | split | Recall@50 | NDCG@50 | tail Recall@50 |
|---|---|---:|---:|---:|
| Two-Tower baseline | 7d train / 1d valid | 0.112 | 0.071 | 0.058 |
| Two-Tower + sequence + time decay | 7d train / 1d valid | 0.126 | 0.079 | 0.071 |

Random split produced higher AUC in the experiment log, but it is not used as the final claim because it may leak future popularity and user intent.

