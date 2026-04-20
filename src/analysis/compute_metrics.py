from pathlib import Path
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
IN_FILE = ROOT / "data" / "processed" / "patriots_master_table.csv"
OUT_DIR = ROOT / "outputs" / "tables"

df = pd.read_csv(IN_FILE)
OUT_DIR.mkdir(parents=True, exist_ok=True)

season_summary = []
player_rows = []

for season, s in df.groupby("season"):
    edge_df = (
        s.groupby(["passer_player_name", "receiver_player_name"], as_index=False)
         .agg(
             weight=("receptions", "sum"),
             rec_yards=("rec_yards", "sum"),
             rec_tds=("rec_tds", "sum"),
             first_downs=("first_downs", "sum"),
             games_together=("game_id", "nunique")
         )
    )

    G = nx.DiGraph()
    for _, row in edge_df.iterrows():
        G.add_edge(
            row["passer_player_name"],
            row["receiver_player_name"],
            weight=row["weight"],
            rec_yards=row["rec_yards"],
            rec_tds=row["rec_tds"],
            first_downs=row["first_downs"],
            games_together=row["games_together"]
        )

    n = G.number_of_nodes()
    m = G.number_of_edges()

    density = nx.density(G)
    weak_components = nx.number_weakly_connected_components(G)
    strong_components = nx.number_strongly_connected_components(G)

    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    in_strength = dict(G.in_degree(weight="weight"))
    out_strength = dict(G.out_degree(weight="weight"))

    degree_c = nx.degree_centrality(G)
    betweenness_c = nx.betweenness_centrality(G, weight=None, normalized=True)
    pagerank_c = nx.pagerank(G, weight="weight")

    try:
        assortativity = nx.degree_pearson_correlation_coefficient(G)
    except Exception:
        assortativity = None

    season_summary.append({
        "season": season,
        "nodes": n,
        "edges": m,
        "density": density,
        "weak_components": weak_components,
        "strong_components": strong_components,
        "assortativity": assortativity
    })

    for node in G.nodes():
        player_rows.append({
            "season": season,
            "player": node,
            "in_degree": in_degree.get(node, 0),
            "out_degree": out_degree.get(node, 0),
            "in_strength": in_strength.get(node, 0),
            "out_strength": out_strength.get(node, 0),
            "degree_centrality": degree_c.get(node, 0),
            "betweenness_centrality": betweenness_c.get(node, 0),
            "pagerank": pagerank_c.get(node, 0)
        })

season_df = pd.DataFrame(season_summary).sort_values("season")
players_df = pd.DataFrame(player_rows).sort_values(
    ["season", "pagerank"], ascending=[True, False]
)

season_df.to_csv(OUT_DIR / "season_summary_metrics.csv", index=False)
players_df.to_csv(OUT_DIR / "player_centrality_metrics.csv", index=False)

top_players = (
    players_df.sort_values(["season", "pagerank"], ascending=[True, False])
              .groupby("season")
              .head(10)
)

top_players.to_csv(OUT_DIR / "top10_players_by_season.csv", index=False)

print("Saved:")
print(OUT_DIR / "season_summary_metrics.csv")
print(OUT_DIR / "player_centrality_metrics.csv")
print(OUT_DIR / "top10_players_by_season.csv")
print()
print(season_df.to_string(index=False))