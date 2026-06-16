# Run Commands

## Full smoke-test pipeline

```bash
make all
```

or:

```bash
python3 run_all.py
```

The pipeline generates pseudo exposure logs, preprocesses train/validation split, trains a lightweight Two-Tower model, builds a Faiss-style index, evaluates recall metrics, and writes output CSV files.

## Tests

```bash
pytest -q
```

## Important outputs

- `outputs/interaction_log.csv`
- `outputs/user_embeddings.npy`
- `outputs/item_embeddings.npy`
- `outputs/metrics.csv`
- `experiments/generated_metrics.csv`
- `outputs/negative_samples.csv`
