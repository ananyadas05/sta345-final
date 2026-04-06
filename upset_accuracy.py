import pandas as pd

df = pd.read_csv("upset_predictions_by_year_26.csv")

years = sorted(df["season"].unique())

top_k = 5

results = []

for year in years:
    year_df = df[df["season"] == year].copy()

    # sort by predicted probability
    year_df = year_df.sort_values("predicted_upset_prob", ascending=False)

    top_games = year_df.head(top_k)

    actual_upsets = year_df["is_upset"].sum()
    hits = top_games["is_upset"].sum()

    hit_rate = hits / actual_upsets if actual_upsets > 0 else None

    results.append({
        "season": year,
        "actual_upsets": int(actual_upsets),
        "hits_in_top_k": int(hits),
        "hit_rate": hit_rate
    })

results_df = pd.DataFrame(results)
print(results_df)