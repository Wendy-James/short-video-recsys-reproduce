# Metric Definition

## Recall@50

For each validation user, retrieve top 50 candidate videos from item embeddings. Recall@50 checks whether the validation positive item appears in the candidate set.

Formula:

`Recall@K = hit_positive_items_in_topK / total_validation_positive_items`

Use case: recall-stage coverage. It answers whether the candidate generator can bring likely positives into the ranking pool.

## NDCG@50

NDCG@50 adds position awareness. A hit ranked at position 3 is better than a hit ranked at position 45.

Use case: avoids only optimizing candidate width while ignoring rank quality.

## AUC

AUC compares positive and negative scores. It is useful for pairwise score separation but is not enough for recall-stage evaluation because topK candidate coverage matters more than global pair ranking.

## Tail Recall@50

Tail Recall@50 evaluates recall quality on long-tail videos. It prevents a fake improvement where the model only retrieves more popular videos.

## Guardrail metrics

When discussing online extension, Recall@50 must be paired with finish rate, interaction rate, dwell time, negative feedback, popular-item ratio, latency, and empty-recall rate.
