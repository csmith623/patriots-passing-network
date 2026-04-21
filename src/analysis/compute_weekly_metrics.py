from pathlib import Path
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
IN_FILE = ROOT / "data" / "processed" / "patriots_master_table.csv"
OUT_DIR = ROOT / "outputs" / "tables"

df = pd.read_csv(IN_FILE)
OUT_DIR.mkdir(parents=True, exist_ok=True)

required = {
    "season",
    "week",
    "game_id",
    "passer_player_name",
    "receiver_player_name",
    "receptions",
    "rec_yards",
    "rec_tds",
    "first_downs",
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

weekly_rows = []
cumulative_rows = []

for season in sorted(df["season"].unique()):
    season_df = df[df["season"] == season].copy()
    weeks = sorted(season_df["week"].unique())

    for week in weeks:
        week_df = season_df[season_df["week"] == week].copy()
        cum_df = season_df[season_df["week"] <= week].copy()

        for subdf, bucket_name in [(week_df, "weekly"), (cum_df, "cumulative")]:
            edge_df = (
                subdf.groupby(["passer_player_name", "receiver_player_name"], as_index=False)
                .agg(
                    weight=("receptions", "sum"),
                    rec_yards=("rec_yards", "sum"),
                    rec_tds=("rec_tds", "sum"),
                    first_downs=("first_downs", "sum"),
                    games=("game_id", "nunique"),
                )
            )

            G = nx.DiGraph()
            for _, row in edge_df.iterrows():
                G.add_edge(
                    row["passer_player_name"],
                    row["receiver_player_name"],
                    weight=float(row["weight"]),
                    rec_yards=float(row["rec_yards"]),
                    rec_tds=float(row["rec_tds"]),
                    first_downs=float(row["first_downs"]),
                    games=int(row["games"]),
                )

            n = G.number_of_nodes()
            m = G.number_of_edges()
            density = nx.density(G) if n > 1 else 0.0
            weak_components = nx.number_weakly_connected_components(G) if n > 0 else 0

            in_strength = dict(G.in_degree(weight="weight"))
            out_strength = dict(G.out_degree(weight="weight"))

            max_in_strength = max(in_strength.values()) if in_strength else 0
            max_out_strength = max(out_strength.values()) if out_strength else 0

            total_weight = sum(d["weight"] for _, _, d in G.edges(data=True))
            avg_edge_weight = total_weight / m if m > 0 else 0.0

            row = {
                "season": int(season),
                "week": int(week),
                "nodes": n,
                "edges": m,
                "density": density,
                "weak_components": weak_components,
                "total_receptions": total_weight,
                "avg_edge_weight": avg_edge_weight,
                "max_in_strength": max_in_strength,
                "max_out_strength": max_out_strength,
            }

            if bucket_name == "weekly":
                weekly_rows.append(row)
            else:
                cumulative_rows.append(row)

weekly_df = pd.DataFrame(weekly_rows).sort_values(["season", "week"]).reset_index(drop=True)
cumulative_df = pd.DataFrame(cumulative_rows).sort_values(["season", "week"]).reset_index(drop=True)

weekly_df.to_csv(OUT_DIR / "weekly_network_metrics.csv", index=False)
cumulative_df.to_csv(OUT_DIR / "cumulative_network_metrics.csv", index=False)

print("Saved:")
print(OUT_DIR / "weekly_network_metrics.csv")
print(OUT_DIR / "cumulative_network_metrics.csv")
print()
print("Weekly preview:")
print(weekly_df.head(8).to_string(index=False))
print()
print("Cumulative preview:")
print(cumulative_df.head(8).to_string(index=False))