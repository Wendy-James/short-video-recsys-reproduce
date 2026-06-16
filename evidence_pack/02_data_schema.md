# Data Schema

This public repository uses pseudo/anonymized data generated with public short-video recommendation field conventions. It does not contain company data.

| Field | Meaning | Example use |
|---|---|---|
| `user_id` | anonymized user id | user tower id embedding |
| `video_id` | anonymized video item id | item tower id embedding |
| `author_id` | anonymized creator id | author preference and item feature |
| `category_id` | coarse content category | user-category preference, same-category negative |
| `event_time` | exposure timestamp | time split and leakage check |
| `is_click` | click label | positive feedback component |
| `is_finish` | finish/completion label | stronger positive feedback |
| `is_like` | like label | strong interaction |
| `is_favorite` | favorite label | strong interaction |
| `dwell_time` | watch duration proxy | weak negative / engagement signal |
| `duration_bucket` | video duration bucket | item feature |
| `is_tail_item` | tail item flag | tail Recall@50 |

## Label design

Positive samples: click, finish, like, or favorite.

Weak negatives: exposure-not-click or low dwell time. They are not treated as absolute dislikes because position, timing, cover image, and competing videos may affect behavior.

## Split design

The default split is 7 days for training and 1 day for validation. Random split is avoided as the primary result because it can leak future popularity and repeated user-item exposure patterns.
