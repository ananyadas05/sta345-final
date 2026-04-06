import pandas as pd
from pathlib import Path

matchupstats_original_26 = pd.read_csv(Path("matchupstats_original_26.csv"))
r64 = matchupstats_original_26[matchupstats_original_26["round"] == 64]

# Actual upsets involving seeds 13-16
upsets = r64[r64["is_upset"] == True]
print(upsets[["season", "wteam_school", "wteam_seed", "lteam_school", "lteam_seed"]].to_string())