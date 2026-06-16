# Boundary Statement

This public repository uses pseudo/anonymized data and does not contain company internal data, private logs, user identifiers, or production code.

The resume project is described as an offline recommendation recall reproduction. It does not claim online A/B lift, production ownership, or deployment in any company recommendation system.

The evidence I can discuss in an interview:

- data schema and label definition;
- 7-day train / 1-day validation time split;
- Two-Tower recall structure;
- negative sampling choices and false-negative risk;
- Recall@50, NDCG@50, AUC, and tail Recall@50;
- ablation logic and failure cases;
- what would be needed before online use: realtime features, incremental index update, latency monitoring, A/B guardrails, alerting, and rollback.

Short version:

This is an offline recall evaluation loop, not an online owner claim.
