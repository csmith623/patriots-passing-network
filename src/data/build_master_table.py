from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "data" / "raw" / "pbp_2006_2007_2008_2016_raw.csv"
OUT_FILE = ROOT / "data" / "processed" / "patriots_master_table.csv"

SEASONS = [2006, 2007, 2008, 2016]
TEAM = "NE"

cols_needed = [
    "game_id",
    "game_date",
    "season",
    "season_type",
    "week",
    "posteam",
    "posteam_type",
    "home_team",
    "away_team",
    "pass",
    "complete_pass",
    "pass_touchdown",
    "yards_gained",
    "air_yards",
    "yards_after_catch",
    "first_down",
    "passer_player_name",
    "receiver_player_name",
    "receiver_player_id",
]

df = pd.read_csv(RAW_FILE, low_memory=False)

keep = [c for c in cols_needed if c in df.columns]
df = df[keep].copy()

df = df[
    (df["season"].isin(SEASONS)) &
    (df["season_type"] == "REG") &
    (df["posteam"] == TEAM) &
    (df["complete_pass"] == 1) &
    (df["passer_player_name"].notna()) &
    (df["receiver_player_name"].notna())
].copy()

df["opponent"] = df.apply(
    lambda r: r["away_team"] if r["home_team"] == TEAM else r["home_team"],
    axis=1
)

df["home_away"] = df["posteam_type"].map({"home": "home", "away": "away"}).fillna("unknown")

group_cols = [
    "season",
    "week",
    "game_id",
    "game_date",
    "opponent",
    "home_away",
    "passer_player_name",
    "receiver_player_name",
]

master = (
    df.groupby(group_cols, dropna=False)
      .agg(
          receptions=("complete_pass", "sum"),
          receiving_yards=("yards_gained", "sum"),
          pass_touchdowns=("pass_touchdown", "sum"),
          first_downs=("first_down", "sum"),
          total_air_yards=("air_yards", "sum"),
          total_yac=("yards_after_catch", "sum"),
      )
      .reset_index()
)

master = master.rename(columns={
    "passer_player_name": "passer",
    "receiver_player_name": "receiver"
})

master = master.sort_values(["season", "week", "game_id", "passer", "receiver"]).reset_index(drop=True)

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
master.to_csv(OUT_FILE, index=False)

print(f"Saved master table to: {OUT_FILE}")
print(f"Rows: {len(master)}")
print(master.head(10).to_string(index=False))