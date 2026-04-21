from pathlib import Path
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
IN_FILE = ROOT / "data" / "processed" / "patriots_master_table.csv"
OUT_DIR = ROOT / "outputs" / "gexf"

df = pd.read_csv(IN_FILE)

required = {
    "season",
    "game_id",
    "passer_player_name",
    "receiver_player_name",
    "receptions",
    "rec_yards",
    "rec_tds",
    "first_downs",
    "total_air_yards",
    "total_yac",
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

for season, s in df.groupby("season"):
    edge_df = (
        s.groupby(["passer_player_name", "receiver_player_name"], as_index=False)
        .agg(
            weight=("receptions", "sum"),
            rec_yards=("rec_yards", "sum"),
            rec_tds=("rec_tds", "sum"),
            first_downs=("first_downs", "sum"),
            total_air_yards=("total_air_yards", "sum"),
            total_yac=("total_yac", "sum"),
            games_together=("game_id", "nunique"),
        )
    )

    G = nx.DiGraph(
        season=int(season),
        team="NE",
        label=f"Patriots Passing Network {season}",
        graph_type="directed_weighted",
    )

    passers = set(edge_df["passer_player_name"])
    receivers = set(edge_df["receiver_player_name"])
    players = passers.union(receivers)

    for player in players:
        if player in passers and player in receivers:
            role = "both"
        elif player in passers:
            role = "passer"
        else:
            role = "receiver"
        G.add_node(player, label=player, role=role)

    for _, row in edge_df.iterrows():
        G.add_edge(
            row["passer_player_name"],
            row["receiver_player_name"],
            weight=float(row["weight"]),
            receptions=float(row["weight"]),
            rec_yards=float(row["rec_yards"]),
            rec_tds=float(row["rec_tds"]),
            first_downs=float(row["first_downs"]),
            total_air_yards=float(row["total_air_yards"]) if pd.notna(row["total_air_yards"]) else 0.0,
            total_yac=float(row["total_yac"]) if pd.notna(row["total_yac"]) else 0.0,
            games_together=int(row["games_together"]),
        )

    out_file = OUT_DIR / f"patriots_passing_{int(season)}.gexf"
    nx.write_gexf(G, out_file)

    print(
        f"Saved {out_file} | "
        f"nodes={G.number_of_nodes()} "
        f"edges={G.number_of_edges()}"
    )