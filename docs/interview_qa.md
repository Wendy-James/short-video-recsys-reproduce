# Interview Q&A

This document explains the public reproducible workflow. The safe positioning is: offline recommendation reproduction with pseudo/public-field schema, not a company production system.

## 1. Where does the dataset come from?

I did not use company-internal data. The schema follows public short-video recommendation datasets such as KuaiRec, KuaiRand, and Tenrec. The public repository uses pseudo/anonymized exposure logs with fields such as:

- `user_id`, `video_id`, `author_id`, `category_id`
- exposure timestamp, device/session bucket
- click, finish, like, favorite, dwell time
- duration bucket, publish-time bucket, tail/head item bucket

The resume-scale wording should be "about 200k exposure-style samples in an offline reproduction protocol", not "real platform traffic".

## 2. How is the label defined?

For recall training, strong positive samples are clicked or high-engagement exposures, such as click plus finish/like/favorite. Weak negatives are exposed but not clicked or very low dwell-time impressions. I treat exposure-not-click as noisy, so the repository records mixed negative sampling and badcases around false negatives.

## 3. Why use 7-day train / 1-day validation instead of random split?

Recommendation predicts future feedback from past behavior. Random split can leak future popularity and near-duplicate behavior, especially for head authors and trending videos. In the experiment table, random split gets higher metrics, but it is not used as the resume result because it overestimates offline quality.

## 4. What is the Two-Tower structure?

The user tower uses user ID embedding, recent behavior sequence, category preference, author interaction, time-decay statistics, and history counts. The item tower uses video ID, author ID, category ID, duration bucket, publish-time bucket, and content-side buckets. The two towers output normalized embeddings and rank items by inner product.

## 5. What loss did you use?

The baseline records BCE-style training for sampled user-item pairs. The stronger run records sampled-softmax-style training with mixed weak negatives. In an interview, the distinction is:

- BCE treats sampled pairs independently.
- Sampled softmax compares one positive item against sampled negatives for the same user/context.
- In-batch negatives are efficient but may introduce false negatives when users have overlapping interests.

## 6. Why Recall@50, NDCG@50, AUC, and tail Recall@50 together?

Two-Tower is a recall module, so Recall@50 checks whether relevant videos enter the candidate set. NDCG@50 checks whether positives are ranked earlier inside the retrieved list. AUC is still useful for pairwise discrimination, but it can hide head-item bias. Tail Recall@50 is used to verify that the lift is not only pushing popular videos.

## 7. Why Faiss IndexFlatIP?

The offline item scale is around tens of thousands in this reproduction, so exact inner-product search is enough and easier to debug. IVF or HNSW would be a latency-memory tradeoff when the candidate pool grows to millions.

## 8. Where did Recall@50 improve from 0.112 to 0.126?

The ablation is:

| Step | Recall@50 | Explanation |
|---|---:|---|
| ID/category baseline | 0.112 | Simple user/item/category/history baseline |
| Add recent sequence | 0.119 | Captures short-term interest |
| Add time decay | 0.123 | Short-video interest changes quickly |
| Add user-category cross | 0.125 | Improves category-level candidate coverage |
| Add mixed negatives | 0.126 | Harder weak negatives improve difficult buckets |

The answer should be "incremental feature and sampling improvements", not "the model magically got better".

## 9. How do you avoid leakage?

- Use time split instead of random split for the reported result.
- Compare random split only as a leakage-risk reference.
- Keep head/tail bucket metrics.
- Review head author repetition and trending video badcases.
- Avoid features that use future interactions or validation-day aggregate statistics.

## 10. What is the real deliverable?

The deliverable is a reproducible offline workflow: schema, preprocessing script, Two-Tower training script, Faiss-style retrieval script, metrics CSV, ablation CSV, badcase table, and interview notes. I do not claim online deployment or A/B-test ownership.
