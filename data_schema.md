# Data Schema

Pseudo schema for a short-video recommendation reproduction project. All fields are anonymized and illustrative.

## interaction_log.csv

| field | type | description |
|---|---:|---|
| user_id | string | anonymized user id |
| video_id | string | anonymized video id |
| author_id | string | anonymized creator id |
| category_id | string | video category |
| event_time | timestamp | exposure time |
| is_click | int | whether user clicked |
| is_finish | int | whether user finished watching |
| is_like | int | whether user liked |
| is_favorite | int | whether user favorited |
| dwell_time | float | watch duration in seconds |
| video_duration | float | video duration in seconds |
| position | int | exposure position in feed |

## label definition

- `label_recall = 1`: click, finish, like, or favorite.
- `label_recall = 0`: exposed but not clicked, or dwell time lower than a small threshold.
- Exposure-not-click is treated as weak negative, not absolute dislike.

## time split

- Train: day 1 to day 7.
- Validation: day 8.
- Features for a sample must only use behaviors before `event_time`.

## tail video definition

Videos are sorted by exposure count. The bottom 80 percent by exposure count are treated as tail videos for tail Recall@50 analysis.

