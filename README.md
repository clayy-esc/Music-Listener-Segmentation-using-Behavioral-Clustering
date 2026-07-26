# Music Listener Segmentation: Behavioral Clustering

Unsupervised listener segmentation on the [Music Streaming Habits 2026](https://www.kaggle.com/datasets/uditjain13/music-streaming-habits-2026) Kaggle dataset — grouping 4,000 listeners into distinct, interpretable behavioral personas using K-Means, with every modeling choice validated by evidence rather than assumption.

## Objective

This project builds and validates an unsupervised KMeans model that segments 4,000 listeners from the *Music Streaming Habits 2026* dataset into distinct, interpretable behavioral personas — clustering on listening habits alone (volume, skip rate, playlist activity, track length) while deliberately excluding demographics to avoid bias, with every modeling choice backed by evidence rather than assumption, and results reported honestly, including the model's limitations. The deliverable is a clean, reproducible analysis ending in six labeled, profiled personas and a saved, production-ready pipeline for applying the model to new data.

## Dataset

- **Source**: [Music Streaming Habits 2026](https://www.kaggle.com/datasets/uditjain13/music-streaming-habits-2026) (Kaggle)
- **Size**: 4,000 rows × 15 columns, no missing values
- **Contents**: per-listener behavior (daily listening minutes, songs/day, playlists, skip rate, discovery/offline/podcast usage) alongside demographic and taste attributes (age, country, platform, subscription, top genre, top mood)

See [`docs/data_dictionary.md`](docs/data_dictionary.md) for the full column-by-column description.

## Results at a glance

| | |
|---|---|
| Final model | K-Means, k = 6 |
| Silhouette score | ≈ 0.15–0.17 |
| Clusters | 6 behavioral personas, sizes ranging from 107 to 1,087 listeners |
| Bias check | Passed — clusters are behavior-driven, not demographic-driven |

**A note on the score:** 0.15–0.17 indicates soft, overlapping cluster boundaries, not sharply separated groups. This is treated as an honest finding about the data — listener behavior in this dataset is continuous rather than naturally clustered — not a modeling shortfall. Full reasoning is in the Evaluation Summary section of the notebook.

### The six personas

| Persona | Behavioral signature |
|---|---|
| **The Loyalists** | High daily listening volume, low skip rate |
| **The Restless Browsers** | High daily listening volume, high skip rate |
| **The Curators** | Heaviest listeners, most actively maintained playlists |
| **The Deep Listeners** | Moderate volume, longer average track length |
| **The Quick Spinners** | Moderate-low volume, shorter average track length |
| **The Long-Form Listeners** | Lowest volume, unusually long tracks, high podcast usage |

## Repository structure

```
music-listener-segmentation/
├── README.md
├── requirements.txt
├── notebooks/
│   └── music-listener-segmentation-behavioral-clustering.ipynb
├── data/
│   └── music_streaming_habits_2026.csv
├── docs/
│   └── data_dictionary.md
├── models/
│   ├── full_pipeline.pkl        # feature engineering + preprocessing, as one fitted sklearn Pipeline
│   ├── kmeans_final.pkl         # trained K-Means model (k=6)
│   └── cluster_profile.csv      # persona names, descriptions, and stats per cluster
└── LICENSE
```

## Notebook structure

The notebook is organized as a linear, reproducible pipeline:

1. **Setup & Imports**
2. **Load data & sanity checks** — shape, dtypes, and null checks
3. **Feature Selection** — behavior features (used for clustering) vs. demographic/taste features (held out for profiling only)
4. **Exploratory Data Analysis (EDA)** — distributions, outliers, correlations, and a demographic pre-check
5. **Feature Engineering** — resolves a 0.98 correlation between `daily_listening_minutes` and `songs_per_day` by deriving `average_song_length`, log-transforming it, and capping extreme values at the 99th percentile
6. **Preprocessing** — `RobustScaler` for skewed features, `StandardScaler` for near-normal ones, combined via `ColumnTransformer`
7. **Model Selection** — K-Means evaluated across k = 2–8 using silhouette score; k = 2 excluded as a degenerate, outlier-driven split
8. **Fit Final Model** — K-Means with k = 6, validated for stability across multiple random seeds
9. **Bias/Validity Check** — confirms clusters are not re-encoding age, country, platform, or subscription
10. **Cluster Profiling & Interpretation** — persona naming and write-ups based on each cluster's actual stats
11. **Evaluation Summary** — final silhouette score and an honest discussion of what it means
12. **Production Pipeline** — feature engineering + preprocessing consolidated into a single reusable `sklearn.Pipeline`
13. **Save Artifacts** — exports the fitted pipeline, model, and persona table
14. **Model Testing** — reloads the saved artifacts and predicts a persona for a new, unseen listener

## Setup

```bash
git clone https://github.com/<your-username>/music-listener-segmentation.git
cd music-listener-segmentation

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

### Run the full analysis
Open `notebooks/music-listener-segmentation-behavioral-clustering.ipynb` in Jupyter or upload it to Kaggle with the dataset attached, and run top to bottom.

### Use the trained model on new data
```python
import joblib
import pandas as pd

full_pipeline = joblib.load('models/full_pipeline.pkl')
kmeans_final = joblib.load('models/kmeans_final.pkl')
persona_table = pd.read_csv('models/cluster_profile.csv', index_col=0)

new_listener = pd.DataFrame([{
    'daily_listening_minutes': 145,
    'songs_per_day': 38,
    'playlists_count': 10,
    'skip_rate_pct': 22.5,
    'discover_weekly_user': True,
    'uses_offline_mode': False,
    'podcasts_too': True,
}])

X_new = full_pipeline.transform(new_listener)
predicted_cluster = kmeans_final.predict(X_new)[0]

result = persona_table.loc[predicted_cluster]
print(f"Persona: {result['persona_name']}")
print(f"Description: {result['description']}")
```

`full_pipeline` bundles both the custom feature-engineering step (the `average_song_length` ratio → log transform → percentile cap) and the fitted scalers into one object, so new data is transformed identically to the training data with a single `.transform()` call — no manual step needs to be reproduced by hand.

## Methodology highlights

A few decisions worth calling out, since they shaped the final result:

- **Demographics were deliberately excluded from clustering.** `age`, `country`, `platform`, and `subscription` were used only to *describe* clusters after the fact, not to define them — preventing the model from segmenting people by identity rather than behavior. Verified after fitting: age averages and platform/subscription proportions were nearly identical across all six clusters.
- **A misleadingly high silhouette score at k=2 was investigated, not accepted at face value.** It turned out to be a degenerate 138-vs-3,862 split caused by unstable ratio values (division by a very small `songs_per_day`), not a real behavioral distinction. Capping the derived feature at its 99th percentile resolved this and revealed a genuine, more moderate elbow at k=6.
- **Candidate engineered features were evaluated on their tradeoffs, not added by default.** `songs_per_playlist`, `skip_intensity`, and a proposed `loyalty_score` were considered but ultimately excluded — each risked reintroducing the same ratio instability already fixed, or duplicating signal already captured by existing features.
- **Results are reported honestly.** The final silhouette score (~0.17) reflects genuine, soft overlap between listener behaviors rather than sharply distinct groups, and the write-up says so directly rather than overstating cluster separation.

## Requirements

```
pandas
numpy
scikit-learn
seaborn
matplotlib
joblib
```

See [`requirements.txt`](requirements.txt) for pinned versions.

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details. The dataset itself is subject to its own license on Kaggle; refer to the [dataset page](https://www.kaggle.com/datasets/uditjain13/music-streaming-habits-2026) for terms.

## Acknowledgments

Dataset by [uditjain13](https://www.kaggle.com/uditjain13) on Kaggle.
