"""
decision_trees.py
Uses leave-one-year-out: for each target year, trains on all prior years only.
Identifies the top 15 matchup statistics by Gini importance.
Data: matchupstats_original.csv 
Filter: round-of-64 games only. Upsets already identified via is_upset column.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from pathlib import Path

DATA_PATH = Path(__file__).parent / "matchupstats_original_26.csv"
N_TREES = 100_000       # match paper; reduce to 10_000 for quick testing
RANDOM_STATE = 42
ROUND_OF_64 = 64
TOP_N = 15

# load and filter data

matchupstats_original_26 = pd.read_csv(DATA_PATH)

# Keep only round-of-64 games
matchupstats_original_26 = matchupstats_original_26[matchupstats_original_26["round"] == ROUND_OF_64].copy()

# is_upset is already a boolean
matchupstats_original_26["upset_int"] = matchupstats_original_26["is_upset"].astype(int)

# Identify the 115 matchup feature columns
matchup_cols = [c for c in matchupstats_original_26.columns if c.startswith("matchup_")]
print(f"Features found: {len(matchup_cols)}")
print(f"Total games after filter: {len(matchupstats_original_26)}")
print(f"Upsets: {matchupstats_original_26['upset_int'].sum()}  |  Non-upsets: {(matchupstats_original_26['upset_int'] == 0).sum()}")
print()

# leave-one-year-out feature importance

years = sorted(matchupstats_original_26["season"].unique())
# Need at least one prior year to train, so skip the earliest year
target_years = years[1:]

results = {}

for target_year in target_years:
    train = matchupstats_original_26[matchupstats_original_26["season"] < target_year]

    if train["upset_int"].sum() == 0:
        print(f"{target_year}: no upsets in training data, skipping.")
        continue

    X_train = train[matchup_cols].values
    y_train = train["upset_int"].values

    n_features = X_train.shape[1]
    max_feat = max(1, int(np.sqrt(n_features)))  # ~10-11 for 115 features

    clf = ExtraTreesClassifier(
        n_estimators=N_TREES,
        max_features=max_feat,
        min_samples_split=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,          # use all CPU cores
    )
    clf.fit(X_train, y_train)

    importances = pd.Series(clf.feature_importances_, index=matchup_cols)
    top15 = importances.nlargest(TOP_N)
    results[target_year] = top15

    print(f"Year {target_year} | trained on {len(train)} games ({y_train.sum()} upsets)")
    print(top15.to_string())
    print()

# summary: which features appear most often in the top 15 across all years

all_top = pd.concat(results.values())
frequency = (
    all_top
    .reset_index()
    .rename(columns={"index": "feature", 0: "importance"})
    .groupby("feature")
    .agg(
        times_in_top15=("importance", "count"),
        mean_importance=("importance", "mean"),
    )
    .sort_values(["times_in_top15", "mean_importance"], ascending=False)
)

print("=" * 60)
print("Feature frequency across all leave-one-year-out runs")
print("=" * 60)
print(frequency.to_string())

# Save results
out_path = Path(__file__).parent / "top15_by_year_26.csv"
rows = []
for year, series in results.items():
    for rank, (feat, imp) in enumerate(series.items(), start=1):
        rows.append({"target_year": year, "rank": rank, "feature": feat, "importance": imp})
pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"\nPer-year results saved to {out_path}")