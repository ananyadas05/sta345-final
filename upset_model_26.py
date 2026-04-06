import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

DATA_PATH = Path(__file__).parent / "matchupstats_original_26.csv"
TOP15_PATH = Path(__file__).parent / "top15_by_year_26.csv"
OUT_PATH = Path(__file__).parent / "upset_predictions_by_year_26.csv"
SUMMARY_OUT_PATH = Path(__file__).parent / "upset_model_year_summary_26.csv"

ROUND_OF_64 = 64
TOP_N = 15
THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

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
print(f"Total upsets: {int(df['upset_int'].sum())}")
print(f"Years in data: {years}")

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def get_year_features(target_year, top15_df, full_df, top_n=15):
    year_features = top15_df[top15_df["target_year"] == target_year]["feature"].tolist()
    year_features = year_features[:top_n]
    year_features = [f for f in year_features if f in full_df.columns]
    return year_features


def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("forest", ExtraTreesClassifier(
            n_estimators=1000,
            random_state=42,
            min_samples_leaf=2,
            max_features="sqrt"
        ))
    ])


def top_k_hits(probabilities, actuals, k):
    temp = pd.DataFrame({
        "prob": probabilities,
        "actual": actuals
    }).sort_values("prob", ascending=False)

    top_k = temp.head(k)
    return int(top_k["actual"].sum())


def choose_threshold_from_prior_years(train_df, candidate_features, thresholds):
    """
    Choose threshold using only prior years.
    For each validation year inside train_df:
      - train on even earlier years
      - validate on that year
    Pick threshold with best average upset recall.
    Tie-break using better average accuracy.
    """
    train_years = sorted(train_df["season"].unique())

    threshold_records = []

    for threshold in thresholds:
        recalls = []
        accuracies = []

        for val_year in train_years:
            inner_train = train_df[train_df["season"] < val_year].copy()
            inner_val = train_df[train_df["season"] == val_year].copy()

            if len(inner_train) == 0 or len(inner_val) == 0:
                continue

            if inner_train["upset_int"].nunique() < 2:
                continue

            X_inner_train = inner_train[candidate_features]
            y_inner_train = inner_train["upset_int"]

            X_inner_val = inner_val[candidate_features]
            y_inner_val = inner_val["upset_int"]

            model = build_model()
            model.fit(X_inner_train, y_inner_train)

            probs_val = model.predict_proba(X_inner_val)[:, 1]
            preds_val = (probs_val >= threshold).astype(int)

            actual_upsets_val = int(y_inner_val.sum())
            correct_upset_picks_val = int(((preds_val == 1) & (y_inner_val.values == 1)).sum())

            # recall for upsets: out of the true upsets, how many did we predict?
            if actual_upsets_val > 0:
                recall_val = correct_upset_picks_val / actual_upsets_val
                recalls.append(recall_val)

            acc_val = accuracy_score(y_inner_val, preds_val)
            accuracies.append(acc_val)

        avg_recall = np.mean(recalls) if len(recalls) > 0 else -1
        avg_accuracy = np.mean(accuracies) if len(accuracies) > 0 else -1

        threshold_records.append({
            "threshold": threshold,
            "avg_recall": avg_recall,
            "avg_accuracy": avg_accuracy
        })

    threshold_df = pd.DataFrame(threshold_records)

    # First maximize average recall of actual upsets.
    # Break ties with average accuracy.
    threshold_df = threshold_df.sort_values(
        by=["avg_recall", "avg_accuracy", "threshold"],
        ascending=[False, False, True]
    )

    return float(threshold_df.iloc[0]["threshold"])


# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------

for target_year in years:
    train = df[df["season"] < target_year].copy()
    test = df[df["season"] == target_year].copy()

    year_features = get_year_features(target_year, top15, df, top_n=TOP_N)

    if len(year_features) == 0:
        print(f"Skipping {target_year}: no valid features found.")
        continue

    if len(train) == 0:
        print(f"Skipping {target_year}: no prior-year training data.")
        continue

    if train["upset_int"].nunique() < 2:
        print(f"Skipping {target_year}: training data has only one class.")
        continue

    # ----------------------------
    # Choose threshold with NO leakage
    # Only uses prior years inside the training set
    # ----------------------------
    best_threshold = choose_threshold_from_prior_years(
        train_df=train,
        candidate_features=year_features,
        thresholds=THRESHOLDS
    )

    # ----------------------------
    # Fit final model on all prior years
    # ----------------------------
    X_train = train[year_features]
    y_train = train["upset_int"]

    X_test = test[year_features]
    y_test = test["upset_int"]

    model = build_model()
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= best_threshold).astype(int)

    # ----------------------------
    # Game-level results
    # ----------------------------
    test_out = test.copy()
    test_out["predicted_upset_prob"] = probs
    test_out["predicted_upset"] = preds
    test_out["threshold_used"] = best_threshold
    test_out["num_features_used"] = len(year_features)

    # Did we correctly flag the actual upset games?
    test_out["correct_upset_pick"] = (
        (test_out["predicted_upset"] == 1) & (test_out["upset_int"] == 1)
    ).astype(int)

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
        "correct_upset_pick",
        "threshold_used",
        "num_features_used"
    ]

    existing_keep_cols = [c for c in keep_cols if c in test_out.columns]
    results.append(test_out[existing_keep_cols])

    # ----------------------------
    # Metrics
    # ----------------------------
    actual_upsets = int(y_test.sum())
    predicted_upsets = int(preds.sum())

    # This is the one you care about most:
    # how many of the real upset games did we actually pick?
    correct_upset_picks = int(((preds == 1) & (y_test.values == 1)).sum())

    # Of all real upsets, how many did we catch?
    upset_recall = correct_upset_picks / actual_upsets if actual_upsets > 0 else np.nan

    # Of all games we predicted as upsets, how many were right?
    upset_precision = correct_upset_picks / predicted_upsets if predicted_upsets > 0 else np.nan

    acc = accuracy_score(y_test, preds)

    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, probs)
    else:
        auc = np.nan

    brier = brier_score_loss(y_test, probs)

    hits_top_3 = top_k_hits(probs, y_test.values, 3)
    hits_top_5 = top_k_hits(probs, y_test.values, 5)

    hit_rate_top_3 = hits_top_3 / actual_upsets if actual_upsets > 0 else np.nan
    hit_rate_top_5 = hits_top_5 / actual_upsets if actual_upsets > 0 else np.nan

    year_summary.append({
        "season": target_year,
        "n_games": len(test),
        "actual_upsets": actual_upsets,
        "predicted_upsets": predicted_upsets,
        "correct_upset_picks": correct_upset_picks,
        "upset_recall": upset_recall,
        "upset_precision": upset_precision,
        "accuracy": acc,
        "roc_auc": auc,
        "brier_score": brier,
        "threshold_used": best_threshold,
        "hits_top_3": hits_top_3,
        "hit_rate_top_3": hit_rate_top_3,
        "hits_top_5": hits_top_5,
        "hit_rate_top_5": hit_rate_top_5,
        "features_used": ", ".join(year_features)
    })

    print(f"\nYear: {target_year}")
    print(f"  Training games: {len(train)}")
    print(f"  Test games: {len(test)}")
    print(f"  Actual upsets: {actual_upsets}")
    print(f"  Predicted upsets: {predicted_upsets}")
    print(f"  Correct upset picks: {correct_upset_picks}")
    print(f"  Upset recall: {upset_recall if not np.isnan(upset_recall) else 'NA'}")
    print(f"  Upset precision: {upset_precision if not np.isnan(upset_precision) else 'NA'}")
    print(f"  Threshold used: {best_threshold}")
    print(f"  Accuracy: {acc:.3f}")

    if not np.isnan(auc):
        print(f"  ROC AUC: {auc:.3f}")
    else:
        print("  ROC AUC: NA (only one class in test year)")

    print(f"  Brier score: {brier:.3f}")
    print(f"  Hits in top 3: {hits_top_3}")
    print(f"  Hit rate top 3: {hit_rate_top_3 if not np.isnan(hit_rate_top_3) else 'NA'}")
    print(f"  Hits in top 5: {hits_top_5}")
    print(f"  Hit rate top 5: {hit_rate_top_5 if not np.isnan(hit_rate_top_5) else 'NA'}")
    print("  Top predicted upset games:")

    preview = test_out.sort_values("predicted_upset_prob", ascending=False).head(5)
    preview_cols = [
        c for c in [
            "wteam_school",
            "wteam_seed",
            "lteam_school",
            "lteam_seed",
            "predicted_upset_prob",
            "predicted_upset",
            "is_upset",
            "correct_upset_pick"
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
    summary_df.to_csv(SUMMARY_OUT_PATH, index=False)
    print(f"Saved year summary to: {SUMMARY_OUT_PATH.name}")
else:
    print("No year summaries were generated.")