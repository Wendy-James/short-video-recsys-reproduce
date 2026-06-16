# Interview Whiteboard

```text
User behavior log
  |
  v
Sample construction
exposure / click / finish / like / favorite / weak negatives
  |
  v
Time split
7d train / 1d validation
  |
  v
User Tower                         Item Tower
user_id                            video_id
history sequence                   author_id
category preference                category_id
author interaction                 content tags / duration bucket
time decay                         publish-time bucket
  |                                 |
  v                                 v
user embedding                      item embedding
          \                         /
           dot product / cosine
                    |
                    v
Faiss TopK Recall
                    |
                    v
Recall@50 / NDCG@50 / tail Recall@50 / badcases
```

## How to narrate it

Start from the business object: candidate video recall before ranking. Then explain sample granularity, label definition, time split, model structure, retrieval, metrics, and badcase. Do not start from model names.
