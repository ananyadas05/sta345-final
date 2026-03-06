"""
boss.py

Implements the BOSS (Balance Optimization Subset Selection) step from:
  Dutta, Jacobson, Sauppe (2017) - "Identifying NCAA tournament upsets using
  Balance Optimization Subset Selection"

Requires top15_by_year.csv produced by decision_trees.py.

Pipeline per target year Y:
  - Treatment group: all is_upset == True games from years < Y
  - Control pool:    all round-of-64 games in year Y where lteam_seed or wteam_seed is 13-16
  - Top-15 features: from year Y's leave-one-year-out Extra-Trees run
  - For each of the 1365 combinations of 4 from the top-15:
      enumerate all size-3 subsets of the control pool
      select the subset G that minimizes balance measure M(G)
  - Track each combination's historical performance
  - Use tau tuning to select high-performing combinations
  - Final output: 2 games most frequently selected across high-performing combos
"""

import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH     = Path(__file__).parent / "matchupstats_original.csv"
TOP15_PATH    = Path(__file__).parent / "top15_by_year.csv"
OUT_PATH      = Path(__file__).parent / "boss_selections.csv"

ROUND_OF_64   = 64
UPSET_SEEDS   = {13, 14, 15, 16}
CONTROL_SIZE  = 3    # BOSS selects 3 per combination
FINAL_SELECT  = 2    # final selections per year
TAU_VALUES    = range(1, 21)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

matchupstats_original = pd.read_csv(DATA_PATH)
matchupstats_original = matchupstats_original[
    matchupstats_original["round"] == ROUND_OF_64
].copy()
matchupstats_original["upset_int"] = matchupstats_original["is_upset"].astype(int)

top15_by_year = pd.read_csv(TOP15_PATH)

# ---------------------------------------------------------------------------
# Balance measure helpers
# ---------------------------------------------------------------------------

def ks_statistic(a, b):
    """
    Kolmogorov-Smirnov statistic between two 1D arrays.
    Measures the maximum vertical distance between their empirical CDFs.
    """
    combined = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(np.sort(a), combined, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), combined, side="right") / len(b)
    return np.max(np.abs(cdf_a - cdf_b))


def relative_difference(a, b):
    """
    R(f1, f2) = |mean(f1) - mean(f2)| / |mean(f2)|
    Measures horizontal spread difference between two distributions.
    Returns 0 if mean(b) is 0 to avoid division by zero.
    """
    mean_b = np.mean(b)
    if mean_b == 0:
        return 0.0
    return abs(np.mean(a) - mean_b) / abs(mean_b)


def balance_measure(treatment, control, features):
    """
    M(G) = sum over features of max(KS(T_i, G_i), R(T_i, G_i))
    treatment, control: DataFrames with feature columns
    """
    total = 0.0
    for feat in features:
        t = treatment[feat].values.astype(float)
        g = control[feat].values.astype(float)
        total += max(ks_statistic(t, g), relative_difference(t, g))
    return total


# ---------------------------------------------------------------------------
# BOSS: enumerate all size-3 subsets of control pool, pick best M(G)
# ---------------------------------------------------------------------------

def run_boss(treatment_df, control_pool_df, features):
    """
    Enumerate all C(|control_pool|, 3) subsets, return the indices of the
    control pool rows with minimum balance measure.
    Returns a list of index labels (from control_pool_df.index).
    """
    pool_indices = list(control_pool_df.index)
    best_score = np.inf
    best_group = None

    for group in combinations(pool_indices, CONTROL_SIZE):
        g_df = control_pool_df.loc[list(group)]
        score = balance_measure(treatment_df, g_df, features)
        if score < best_score:
            best_score = score
            best_group = group

    return list(best_group), best_score


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

# We need at least a few upset years before we can form a treatment group.
# Paper starts at 2001 (upsets from 1998-2000 form the first treatment group).
all_years = sorted(matchupstats_original["season"].unique())

# boss_results[year][combo] = list of 3 selected game indices
boss_results = {}

# historical_performance[combo] = number of correct selections across all years
# We track this incrementally as a dict: combo -> cumulative correct count per year
# combo_history[combo][year] = n correct selections that year
combo_history = {}  # filled as we go

final_selections = []  # rows written to output CSV

for target_year in all_years:
    prior_years = [y for y in all_years if y < target_year]
    if len(prior_years) < 3:
        # Need at least 3 prior years to have a meaningful treatment group
        print(f"{target_year}: skipping — not enough prior years.")
        continue

    # Treatment group: historical upsets from all prior years
    treatment = matchupstats_original[
        (matchupstats_original["season"].isin(prior_years)) &
        (matchupstats_original["upset_int"] == 1)
    ]

    if len(treatment) == 0:
        print(f"{target_year}: skipping — no historical upsets.")
        continue

    # Control pool: all seed 13-16 games in target year
    control_pool = matchupstats_original[
        (matchupstats_original["season"] == target_year) &
        (
            matchupstats_original["lteam_seed"].isin(UPSET_SEEDS) |
            matchupstats_original["wteam_seed"].isin(UPSET_SEEDS)
        )
    ]

    if len(control_pool) < CONTROL_SIZE:
        print(f"{target_year}: skipping — control pool too small ({len(control_pool)} games).")
        continue

    # Top-15 features for this year (from leave-one-year-out Extra-Trees)
    year_top15 = top15_by_year[top15_by_year["target_year"] == target_year]
    if len(year_top15) == 0:
        print(f"{target_year}: skipping — no top-15 features found.")
        continue
    top15_features = year_top15.sort_values("rank")["feature"].tolist()[:15]

    print(f"\n{target_year} | treatment={len(treatment)} upsets | "
          f"control pool={len(control_pool)} games | "
          f"top-15 features={len(top15_features)}")

    # Run BOSS for each of the C(15,4) = 1365 feature combinations
    year_boss = {}
    for combo in combinations(top15_features, 4):
        selected_indices, score = run_boss(treatment, control_pool, list(combo))
        year_boss[combo] = selected_indices

    boss_results[target_year] = year_boss

    # -----------------------------------------------------------------------
    # Track historical performance for each combo
    # correct = how many of the 3 selected games were actual upsets
    # -----------------------------------------------------------------------
    actual_upsets = set(control_pool[control_pool["upset_int"] == 1].index)
    for combo, selected in year_boss.items():
        n_correct = len(set(selected) & actual_upsets)
        if combo not in combo_history:
            combo_history[combo] = {}
        combo_history[combo][target_year] = n_correct

    # -----------------------------------------------------------------------
    # Tau tuning and final selection
    # Paper: evaluate tau on Y-1, apply to Y
    # We evaluate tau on all years up to Y-1 using years already processed
    # -----------------------------------------------------------------------
    processed_years = sorted(boss_results.keys())
    if len(processed_years) < 2:
        print(f"  Skipping final selection — need at least 2 processed years for tau tuning.")
        continue

    eval_years  = processed_years[:-1]   # all but current year for tau tuning
    select_year = processed_years[-1]    # current year = make selections for

    def cumulative_correct(combo, up_to_years):
        return sum(combo_history.get(combo, {}).get(y, 0) for y in up_to_years)

    best_tau = None
    best_tau_correct = -1

    for tau in TAU_VALUES:
        # Find N* = max cumulative correct across all combos up through eval_years
        n_star = max(
            (cumulative_correct(c, eval_years) for c in year_boss.keys()),
            default=0
        )
        threshold = n_star - tau

        # High-performing combos
        P = [c for c in year_boss.keys()
             if cumulative_correct(c, eval_years) >= threshold]

        if not P:
            continue

        # Count how often each game was selected across high-performing combos
        # for the validation year (last of eval_years)
        val_year = eval_years[-1]
        if val_year not in boss_results:
            continue

        counts = Counter()
        for combo in P:
            if combo in boss_results[val_year]:
                for idx in boss_results[val_year][combo]:
                    counts[idx] += 1

        # How many actual upsets were in the top-2 selections
        val_upsets = set(
            matchupstats_original[
                (matchupstats_original["season"] == val_year) &
                (matchupstats_original["upset_int"] == 1)
            ].index
        )
        top2 = [idx for idx, _ in counts.most_common(FINAL_SELECT)]
        n_correct_val = len(set(top2) & val_upsets)

        if n_correct_val > best_tau_correct:
            best_tau_correct = n_correct_val
            best_tau = tau

    if best_tau is None:
        best_tau = 1  # fallback

    # Apply best tau to select games for select_year
    all_prior_eval = [y for y in processed_years if y < select_year]
    n_star_final = max(
        (cumulative_correct(c, all_prior_eval) for c in year_boss.keys()),
        default=0
    )
    P_final = [c for c in year_boss.keys()
               if cumulative_correct(c, all_prior_eval) >= n_star_final - best_tau]

    counts_final = Counter()
    for combo in P_final:
        if combo in boss_results[select_year]:
            for idx in boss_results[select_year][combo]:
                counts_final[idx] += 1

    top2_final = [idx for idx, _ in counts_final.most_common(FINAL_SELECT)]

    # Resolve selections back to game info
    for rank, idx in enumerate(top2_final, start=1):
        row = matchupstats_original.loc[idx]
        actual_upset = bool(row["upset_int"])
        final_selections.append({
            "target_year":  select_year,
            "selection":    rank,
            "wteam_school": row["wteam_school"],
            "lteam_school": row["lteam_school"],
            "wteam_seed":   row["wteam_seed"],
            "lteam_seed":   row["lteam_seed"],
            "is_upset":     actual_upset,
            "tau_used":     best_tau,
            "selection_count": counts_final[idx],
            "combos_in_P":  len(P_final),
        })
        print(f"  Selection {rank}: {row['lteam_school']} (#{row['lteam_seed']}) "
              f"vs {row['wteam_school']} (#{row['wteam_seed']}) "
              f"| actual upset: {actual_upset} | count: {counts_final[idx]}")

# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------

out_df = pd.DataFrame(final_selections)
out_df.to_csv(OUT_PATH, index=False)
print(f"\nSelections saved to {OUT_PATH}")

# Summary
if len(out_df) > 0:
    correct = out_df["is_upset"].sum()
    total   = len(out_df)
    print(f"\nOverall: {correct}/{total} correct selections "
          f"({100 * correct / total:.1f}%)")