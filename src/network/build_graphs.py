from pathlib import Path
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
IN_FILE = ROOT / "data" / "processed" / "patriots_master_table.csv"
OUT_DIR = ROOT / "outputs" / "gexf"

df = pd.read_csv(IN_FILE)

required = {
    "season", "passer_player_name", "receiver_player_name",
    "receptions", "rec_yards", "rec_tds", "first_downs"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

for season, s in df.groupby("season"):
    edge_df = (
        s.groupby(["passer_player_name", "receiver_player_name"], as_index=False)
         .agg(
             weight=("receptions", "sum"),
             total_yards=("rec_yards", "sum"),
             total_tds=("rec_tds", "sum"),
             total_first_downs=("first_downs", "sum"),
             games_together=("game_id", "nunique")
         )
    )

    G = nx.DiGraph(season=int(season), team="NE", label=f"Patriots Passing Network {season}")

    players = set(edge_df["passer_player_name"]).union(edge_df["receiver_player_name"])
    for p in players:
        G.add_node(p, label=p)

    for _, row in edge_df.iterrows():
        G.add_edge(
            row["passer_player_name"],
            row["receiver_player_name"],
            weight=int(row["weight"]),
            receptions=int(row["weight"]),
            receiving_yards=int(row["total_yards"]),
            touchdowns=int(row["total_tds"]),
            first_downs=int(row["total_first_downs"]),
            games_together=int(row["games_together"])
        )

    out_file = OUT_DIR / f"patriots_passing_{season}.gexf"
    nx.write_gexf(G, out_file)
    print(f"Saved {out_file} | nodes={G.number_of_nodes()} edges={G.number_of_edges()}")