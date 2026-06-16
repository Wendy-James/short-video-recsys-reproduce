# Interview Debug Notes

## What actually runs

`make all` runs the public pipeline end to end:

1. `src/data_preprocess.py` writes `outputs/interaction_log.csv`.
2. `src/negative_sampling.py` writes `outputs/negative_samples.csv` and `outputs/negative_sampling_summary.json`.
3. `src/train_twotower.py` trains a lightweight Two-Tower model with BCE. It uses PyTorch when available and a NumPy SGD fallback otherwise.
4. `src/build_faiss_index.py` builds an exact inner-product retrieval matrix from trained item embeddings.
5. `src/evaluate_recall.py` computes Recall@50, NDCG@50, AUC, and tail Recall@50 from validation rows.
6. `src/train_ranker.py` writes a lightweight ranking-score file.
7. `pytest` checks data split, metrics, negative sampling, and trainable model outputs.

## Why public metrics may differ from resume metrics

The resume reports a fixed offline experiment table from the prepared project package. The public repository is a sanitized, CPU-friendly reproduction on pseudo data. It proves the field design, training/evaluation path, and metric implementation; it does not claim to reproduce private platform data or online A/B results.

## How to answer if metrics look unstable

Use this answer:

> The public repo is intentionally small, so absolute metrics can move with the pseudo sample. I use it to show that the pipeline really runs: time split, negative sampling, Two-Tower training, item embedding export, IndexFlatIP-style retrieval, Recall/NDCG/tail-recall calculation, and badcase recording. For interview discussion, I focus on the evaluation protocol and ablation logic rather than pretending the pseudo-data metric is a production number.

## Failure examples to mention

- Head leakage: random split makes popular videos look better than they should.
- Author repeat: repeated author exposure can inflate similarity.
- Cold start: new/low-exposure items have sparse feedback.
- Weak negative noise: exposure without click is not always true dislike.
