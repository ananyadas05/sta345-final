import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

DATA_PATH = Path(__file__).parent / "matchupstats_original_26.csv"
TOP15_PATH = Path(__file__).parent / "top15_by_year_26.csv"
OUT_PATH = Path(__file__).parent / "upset_predictions_by_year_26.csv"

ROUND_OF_64 = 64
TOP_N = 15

# Probability threshold for classifying a game as an upset
THRESHOLD = 0.30

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)
df = df[df["round"] == ROUND_OF_64].copy()
df["upset_int"] = df["is_upset"].astype(int)

top15 = pd.read_csv(TOP15_PATH)

years = sorted(df["season"].unique())

results = []
year_summary = []

print(f"Total Round of 64 games: {len(df)}")
print(f"Total upsets: {df['upset_int'].sum()}")
print(f"Years in data: {years}")

# ------------------------------------------------------------
# Loop through years
# Train on all PRIOR years, predict on target year
# ------------------------------------------------------------

for target_year in years:
    train = df[df["season"] < target_year].copy()
    test = df[df["season"] == target_year].copy()

    # Get top features for this target year
    year_features = top15[top15["target_year"] == target_year]["feature"].tolist()
    year_features = year_features[:TOP_N]

    # Keep only features that exist in dataframe
    year_features = [f for f in year_features if f in df.columns]

    if len(year_features) == 0:
        print(f"Skipping {target_year}: no valid features found.")
        continue

    if len(train) == 0:
        print(f"Skipping {target_year}: no prior-year training data.")
        continue

    if train["upset_int"].nunique() < 2:
        print(f"Skipping {target_year}: training data has only one class.")
        continue

    X_train = train[year_features]
    y_train = train["upset_int"]

    X_test = test[year_features]
    y_test = test["upset_int"]

    # Pipeline: median imputation + logistic regression
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("logit", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    # Predicted probabilities
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)

    # Store game-level results
    test_out = test.copy()
    test_out["predicted_upset_prob"] = probs
    test_out["predicted_upset"] = preds
    test_out["threshold_used"] = THRESHOLD
    test_out["num_features_used"] = len(year_features)

    keep_cols = [
        "season",
        "wteam_school",
        "wteam_seed",
        "lteam_school",
        "lteam_seed",
        "is_upset",
        "upset_int",
        "predicted_upset_prob",
        "predicted_upset",
        "threshold_used",
        "num_features_used"
    ]

    existing_keep_cols = [c for c in keep_cols if c in test_out.columns]
    results.append(test_out[existing_keep_cols])

    # Metrics for that year
    acc = accuracy_score(y_test, preds)

    # ROC AUC only works if both classes appear in test year
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, probs)
    else:
        auc = np.nan

    brier = brier_score_loss(y_test, probs)

    year_summary.append({
        "season": target_year,
        "n_games": len(test),
        "actual_upsets": int(y_test.sum()),
        "predicted_upsets": int(preds.sum()),
        "accuracy": acc,
        "roc_auc": auc,
        "brier_score": brier,
        "features_used": ", ".join(year_features)
    })

    print(f"\nYear: {target_year}")
    print(f"  Training games: {len(train)}")
    print(f"  Test games: {len(test)}")
    print(f"  Actual upsets: {int(y_test.sum())}")
    print(f"  Predicted upsets: {int(preds.sum())}")
    print(f"  Accuracy: {acc:.3f}")
    if not np.isnan(auc):
        print(f"  ROC AUC: {auc:.3f}")
    else:
        print("  ROC AUC: NA (only one class in test year)")
    print(f"  Brier score: {brier:.3f}")
    print("  Top predicted upset games:")

    preview = test_out.sort_values("predicted_upset_prob", ascending=False).head(5)
    preview_cols = [
        c for c in [
            "wteam_school",
            "wteam_seed",
            "lteam_school",
            "lteam_seed",
            "predicted_upset_prob",
            "is_upset"
        ] if c in preview.columns
    ]
    print(preview[preview_cols].to_string(index=False))

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

if len(results) > 0:
    final_results = pd.concat(results, ignore_index=True)
    final_results.to_csv(OUT_PATH, index=False)
    print(f"\nSaved game-level predictions to: {OUT_PATH.name}")
else:
    print("\nNo game-level predictions were generated.")

if len(year_summary) > 0:
    summary_df = pd.DataFrame(year_summary)
    summary_out = Path(__file__).parent / "upset_model_year_summary_26.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"Saved year summary to: {summary_out.name}")
else:
    print("No year summaries were generated.")