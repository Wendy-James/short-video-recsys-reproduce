# Development Log

## Public Evidence Boundary

This repository is a public evidence repo organized in **June 2026** for resume and interview review. It is not meant to pretend that the public commit history is the full history of the original learning and experimentation process.

The public version contains:

- pseudo/anonymized short-video recommendation fields
- runnable preprocessing, training, indexing, evaluation, and ranking scripts
- experiment CSV files for the resume-side metrics
- badcase records and interview notes
- pytest smoke tests for data split, metrics, and negative sampling assumptions

The public version does not contain:

- company-internal data
- private user behavior logs
- online service code
- A/B-test ownership claims

## Why The Repo Was Organized This Way

The resume mentions an offline short-video recommendation reproduction. To make the claim checkable, the repo exposes the same evidence an interviewer may ask for:

1. schema and sample-generation logic
2. 7-day train / 1-day validation boundary
3. Two-Tower training entry point
4. Faiss-style top-k retrieval step
5. metrics and ablation CSV
6. badcase analysis
7. tests and one-command execution

## Reproducible Commands

```bash
make all
```

or:

```bash
./run.sh
```

## Interview Wording

Safe wording:

> I organized a public evidence version in June 2026. It uses pseudo/public-field data to reproduce the same offline recommendation workflow and make the schema, metrics, ablations, and badcases inspectable.

Avoid:

> This public repo is a production service, or the commit history represents the full original timeline.
