"""
Combined pipeline for NCAA upset prediction, based on

Step 1: Use a random forest (Extra-Trees) to find the most predictive statistics
Step 2: Use BOSS to find current-year games that look most like historical upsets
Step 3: Use historical performance to filter which stat combinations to trust

All the key numbers the researchers chose are at the top. Try changing them
and see how the final accuracy changes.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from itertools import combinations
from pathlib import Path
from collections import Counter

# PARAMETERS - change these to experiment

# How many of the top statistics to keep after the extra-trees step.
# The paper used 15. 
TOP_N_FEATURES = 15

# How many statistics to use in each BOSS combination.
# The paper used 4. C(TOP_N_FEATURES, COMBO_SIZE) gives the number of combinations.
# With TOP_N=15 and COMBO_SIZE=4, that is 1365 combinations.
COMBO_SIZE = 4

# How many games BOSS selects per combination as candidates.
# The paper used 3. These 3 get voted on across all combinations to pick the final 2.
BOSS_GROUP_SIZE = 3

# How many final upset predictions to make per year.
# The paper used 2.
FINAL_SELECTIONS = 2

# Range of tau values to test during tuning.
# Tau controls how close to the best-performing combination a combo needs to be
# to get included in the final vote. Higher tau = more combinations included.
# The paper tested 1 through 20.
TAU_MIN = 1
TAU_MAX = 20

# Which seeds are eligible to be upsets (i.e., the underdog seeds).
# The paper used {13, 14, 15}. We added 16 to capture upsets like UMBC 2018.
UPSET_SEEDS = {13, 14, 15, 16}

# Minimum number of prior years needed before we start making predictions.
# Needs to be at least 1, but more prior years means a more stable treatment group.
MIN_PRIOR_YEARS = 3

# Number of trees in the Extra-Trees forest.
# More trees = more stable importance estimates but slower to run.
# The paper used 100,000. Use 1,000 for quick testing.
N_TREES = 100_000

# Random seed for reproducibility.
RANDOM_STATE = 42

# file paths:

DATA_PATH = Path(__file__).parent / "matchupstats_original.csv"
OUT_PATH = Path(__file__).parent / "boss_selections.csv"

# load and filter data:

data = pd.read_csv(DATA_PATH)
data = data[data["round"] == 64].copy()
data["upset_int"] = data["is_upset"].astype(int)

matchup_cols = [c for c in data.columns if c.startswith("matchup_")]
print(f"Total features available: {len(matchup_cols)}")
print(f"Total round-of-64 games: {len(data)}")
print(f"Upsets: {data['upset_int'].sum()} | Non-upsets: {(data['upset_int'] == 0).sum()}")
print()

all_years = sorted(data["season"].unique())

# Step 1: Extra-Trees feature selection
# For each target year, train only on prior years to avoid using future data.
# Ranks all 115 statistics by how useful they are for predicting upsets.

top_features_by_year = {}

for target_year in all_years:
    train = data[data["season"] < target_year]

    if train["upset_int"].sum() == 0:
        continue

    X_train = train[matchup_cols].values
    y_train = train["upset_int"].values

    clf = ExtraTreesClassifier(
        n_estimators=N_TREES,
        max_features=max(1, int(np.sqrt(len(matchup_cols)))),
        min_samples_split=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    importances = pd.Series(clf.feature_importances_, index=matchup_cols)
    top_features_by_year[target_year] = importances.nlargest(TOP_N_FEATURES).index.tolist()

    print(f"Year {target_year} | top {TOP_N_FEATURES} features identified from {len(train)} training games")

print()

# Balance measure functions
# These measure how similar two groups of games look across a set of statistics.
# Lower M(G) means the control group looks more like the treatment group.

def ks_statistic(a, b):
    # KS statistic: max vertical distance between the two empirical distributions
    combined = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(np.sort(a), combined, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), combined, side="right") / len(b)
    return np.max(np.abs(cdf_a - cdf_b))

def relative_difference(a, b):
    # Relative difference in means: catches horizontal spread differences the KS misses
    mean_b = np.mean(b)
    if mean_b == 0:
        return 0.0
    return abs(np.mean(a) - mean_b) / abs(mean_b)

def balance_measure(treatment, control, features):
    # M(G): sum across features of max(KS, relative difference)
    total = 0.0
    for feat in features:
        t = treatment[feat].values.astype(float)
        g = control[feat].values.astype(float)
        total += max(ks_statistic(t, g), relative_difference(t, g))
    return total

# Step 2 and 3: BOSS + tau tuning
# For each year, find current games that look most like historical upsets,
# then use historical performance to decide which stat combinations to trust.

boss_results = {}
combo_history = {}
final_selections = []

for target_year in all_years:
    prior_years = [y for y in all_years if y < target_year]

    if len(prior_years) < MIN_PRIOR_YEARS:
        print(f"{target_year}: skipping - not enough prior years.")
        continue

    # Treatment group: all historical upsets from prior years
    treatment = data[
        (data["season"].isin(prior_years)) &
        (data["upset_int"] == 1)
    ]

    if len(treatment) == 0:
        print(f"{target_year}: skipping - no historical upsets to learn from.")
        continue

    # Control pool: all eligible games in the target year (the candidates)
    control_pool = data[
        (data["season"] == target_year) &
        (data["lteam_seed"].isin(UPSET_SEEDS) | data["wteam_seed"].isin(UPSET_SEEDS))
    ]

    if len(control_pool) < BOSS_GROUP_SIZE:
        print(f"{target_year}: skipping - control pool too small.")
        continue

    if target_year not in top_features_by_year:
        print(f"{target_year}: skipping - no features available.")
        continue

    top_features = top_features_by_year[target_year]
    n_combos = len(list(combinations(top_features, COMBO_SIZE)))
    print(f"{target_year} | {len(treatment)} historical upsets | {len(control_pool)} candidate games | {n_combos} feature combinations")

    # For each combination of COMBO_SIZE features, find the BOSS_GROUP_SIZE
    # games in the control pool that best match the treatment group
    year_boss = {}
    pool_indices = list(control_pool.index)

    for combo in combinations(top_features, COMBO_SIZE):
        best_score = np.inf
        best_group = None
        for group in combinations(pool_indices, BOSS_GROUP_SIZE):
            g_df = control_pool.loc[list(group)]
            score = balance_measure(treatment, g_df, list(combo))
            if score < best_score:
                best_score = score
                best_group = group
        year_boss[combo] = list(best_group)

    boss_results[target_year] = year_boss

    # Track how many of each combo's selections were actual upsets this year
    actual_upsets = set(control_pool[control_pool["upset_int"] == 1].index)
    for combo, selected in year_boss.items():
        n_correct = len(set(selected) & actual_upsets)
        if combo not in combo_history:
            combo_history[combo] = {}
        combo_history[combo][target_year] = n_correct

    # Tau tuning: pick the tau value that would have worked best on prior years,
    # then apply it to make selections for the current year
    processed_years = sorted(boss_results.keys())
    if len(processed_years) < 2:
        print(f"  Skipping selection - need at least 2 processed years for tau tuning.")
        continue

    eval_years = processed_years[:-1]
    select_year = processed_years[-1]

    def cumulative_correct(combo, up_to_years):
        return sum(combo_history.get(combo, {}).get(y, 0) for y in up_to_years)

    best_tau = None
    best_tau_correct = -1

    for tau in range(TAU_MIN, TAU_MAX + 1):
        n_star = max((cumulative_correct(c, eval_years) for c in year_boss), default=0)
        P = [c for c in year_boss if cumulative_correct(c, eval_years) >= n_star - tau]

        if not P:
            continue

        val_year = eval_years[-1]
        if val_year not in boss_results:
            continue

        counts = Counter()
        for combo in P:
            if combo in boss_results[val_year]:
                for idx in boss_results[val_year][combo]:
                    counts[idx] += 1

        val_upsets = set(data[(data["season"] == val_year) & (data["upset_int"] == 1)].index)
        top_picks = [idx for idx, _ in counts.most_common(FINAL_SELECTIONS)]
        n_correct_val = len(set(top_picks) & val_upsets)

        if n_correct_val > best_tau_correct:
            best_tau_correct = n_correct_val
            best_tau = tau

    if best_tau is None:
        best_tau = TAU_MIN

    # Apply the chosen tau to make final selections for select_year
    all_prior_eval = [y for y in processed_years if y < select_year]
    n_star_final = max((cumulative_correct(c, all_prior_eval) for c in year_boss), default=0)
    P_final = [c for c in year_boss if cumulative_correct(c, all_prior_eval) >= n_star_final - best_tau]

    counts_final = Counter()
    for combo in P_final:
        if combo in boss_results[select_year]:
            for idx in boss_results[select_year][combo]:
                counts_final[idx] += 1

    top2_final = [idx for idx, _ in counts_final.most_common(FINAL_SELECTIONS)]

    for rank, idx in enumerate(top2_final, start=1):
        row = data.loc[idx]
        actual_upset = bool(row["upset_int"])
        final_selections.append({
            "target_year": select_year,
            "selection": rank,
            "wteam_school": row["wteam_school"],
            "lteam_school": row["lteam_school"],
            "wteam_seed": row["wteam_seed"],
            "lteam_seed": row["lteam_seed"],
            "is_upset": actual_upset,
            "tau_used": best_tau,
            "selection_count": counts_final[idx],
            "combos_in_P": len(P_final),
        })
        print(f"  Pick {rank}: {row['lteam_school']} (#{row['lteam_seed']}) vs "
              f"{row['wteam_school']} (#{row['wteam_seed']}) | "
              f"actual upset: {actual_upset} | tau: {best_tau}")

# Save and summarize

out_df = pd.DataFrame(final_selections)
out_df.to_csv(OUT_PATH, index=False)
print(f"\nSelections saved to {OUT_PATH}")

if len(out_df) > 0:
    correct = out_df["is_upset"].sum()
    total = len(out_df)
    print(f"\nOverall: {correct}/{total} correct ({100 * correct / total:.1f}%)")
    print(f"\nParameters used:")
    print(f"  TOP_N_FEATURES  = {TOP_N_FEATURES}")
    print(f"  COMBO_SIZE      = {COMBO_SIZE}")
    print(f"  BOSS_GROUP_SIZE = {BOSS_GROUP_SIZE}")
    print(f"  FINAL_SELECTIONS= {FINAL_SELECTIONS}")
    print(f"  UPSET_SEEDS     = {sorted(UPSET_SEEDS)}")
    print(f"  TAU range       = {TAU_MIN} to {TAU_MAX}")
    print(f"  MIN_PRIOR_YEARS = {MIN_PRIOR_YEARS}")
    print(f"  N_TREES         = {N_TREES}")