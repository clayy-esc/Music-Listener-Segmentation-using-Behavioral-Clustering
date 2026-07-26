# Data Dictionary — Music Streaming Habits 2026

Source: [Kaggle — Music Streaming Habits 2026](https://www.kaggle.com/datasets/uditjain13/music-streaming-habits-2026)
Shape: 4,000 rows × 15 columns, no missing values.

| Column | Description |
|---|---|
| `listener_id` | Unique row identifier (1–4000). Dropped before modeling. |
| `age` | Listener's age, 13–57 (avg 28). Demographic — used for profiling only, not clustering. |
| `country` | Listener's country (10 values: Australia, Germany, Japan, India, Nigeria, Canada, USA, UK, Brazil, France). Demographic — used for profiling only, not clustering. |
| `platform` | Primary streaming platform used (Spotify, Apple Music, Amazon Music, Tidal, SoundCloud, YouTube Music). Demographic — used for profiling only, not clustering. |
| `subscription` | Subscription tier (Free, Family, Premium, Student). Demographic — used for profiling only, not clustering. |
| `top_genre` | Listener's most-played genre (12 values, e.g. Pop, Hip-Hop, R&B, EDM, Lo-fi). Taste feature — context only. |
| `top_artist` | Listener's most-played artist (14 unique artists). Too high-cardinality to encode usefully; dropped entirely. |
| `daily_listening_minutes` | Minutes of music listened per day, 5–697 (avg 139). Behavior feature — used for clustering. Right-skewed. |
| `songs_per_day` | Number of individual songs played per day, 1–198 (avg 39). Behavior feature — dropped after feature engineering (see `average_song_length` below). Right-skewed and highly correlated (0.98) with `daily_listening_minutes`. |
| `playlists_count` | Number of playlists the listener maintains, 1–22 (avg 9). Behavior feature — used for clustering. Approximately bell-shaped with a slight right skew. |
| `skip_rate_pct` | Percentage of songs skipped before finishing, 0.0–70.9 (avg 28.2%). Behavior feature — used for clustering. Approximately symmetric, bell-shaped. |
| `discover_weekly_user` | Whether they use algorithmic discovery mixes (True/False, ~57/43 split). Behavior feature — used for clustering. |
| `uses_offline_mode` | Whether they download music for offline listening (True/False, ~43/57 split). Behavior feature — used for clustering. |
| `podcasts_too` | Whether they also listen to podcasts on the platform (True/False, ~47/53 split). Behavior feature — used for clustering. |
| `top_mood` | Self-reported usual listening mood (8 values: Happy, Sad, Energetic, Chill, Workout, Sleep, Focus, Party). Taste feature — context only. |

## Engineered feature (not in the original dataset)

| Column | Description |
|---|---|
| `average_song_length` | `daily_listening_minutes / songs_per_day`. Derived to resolve the 0.98 correlation between the two source columns. Heavily right-skewed (skew ≈ 8.5), with instability at low `songs_per_day` values. |
| `average_song_length_log` | `log1p(average_song_length)`, capped at the 99th percentile. Used in place of the raw ratio for clustering — skew reduced from ≈8.5 to ≈3.7 after the transform. |

## Feature roles summary

- **Clustering inputs**: `daily_listening_minutes`, `playlists_count`, `skip_rate_pct`, `average_song_length_log`, `discover_weekly_user`, `uses_offline_mode`, `podcasts_too`
- **Profiling only (excluded from clustering to avoid demographic bias)**: `age`, `country`, `platform`, `subscription`, `top_genre`, `top_mood`
- **Dropped entirely**: `listener_id`, `top_artist`, `songs_per_day` (superseded by `average_song_length`), raw `average_song_length` (superseded by its log-transformed, capped version)
