# Data Dictionary — Music Streaming Habits 2026

Source: [Kaggle — Music Streaming Habits 2026](https://www.kaggle.com/datasets/uditjain13/music-streaming-habits-2026)

| Column | Description |
|---|---|
| `listener_id` | Unique row identifier |
| `age` | Listener's age |
| `country` | Listener's country (Australia, Germany, Japan, India, Nigeria, Canada, USA, UK, Brazil, France) |
| `platform` | Primary streaming platform used (Spotify, Apple Music, Amazon Music, Tidal, SoundCloud, YouTube Music) |
| `subscription` | Subscription tier (Free, Student, Premium, Family) |
| `top_genre` | Listener's most-played genre (e.g. Pop, Hip-Hop, R&B, EDM, Lo-fi) |
| `top_artist` | Listener's most-played artist |
| `daily_listening_minutes` | Minutes of music listened per day |
| `songs_per_day` | Number of individual songs played per day |
| `playlists_count` | Number of playlists the listener maintains |
| `skip_rate_pct` | Percentage of songs skipped before finishing |
| `discover_weekly_user` | Whether they use algorithmic discovery mixes (True/False) |
| `uses_offline_mode` | Whether they download music for offline listening (True/False) |
| `podcasts_too` | Whether they also listen to podcasts on the platform (True/False) |
| `top_mood` | Self-reported usual listening mood (Happy, Sad, Energetic, Chill, Workout, Sleep, Focus, Party) |

## Engineered feature

| Column | Description |
|---|---|
| `average_song_length` | Derived from `daily_listening_minutes` divided by `songs_per_day` |
| `average_song_length_log` | Log-transformed and capped version of `average_song_length`, used in modeling |
