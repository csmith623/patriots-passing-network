from pathlib import Path
import pandas as pd
import networkx as nx
from networkx.algorithms import community

ROOT = Path(__file__).resolve().parents[2]
IN_FILE = ROOT / "data" / "processed" / "patriots_master_table.csv"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_GEXF = ROOT / "outputs" / "gexf"

df = pd.read_csv(IN_FILE)
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_GEXF.mkdir(parents=True, exist_ok=True)

required = {
    "season",
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
            games_together=("game_id", "nunique"),
        )
    )

    G = nx.DiGraph(season=int(season), team="NE")

    for _, row in edge_df.iterrows():
        G.add_edge(
            row["passer_player_name"],
            row["receiver_player_name"],
            weight=float(row["weight"]),
            rec_yards=float(row["rec_yards"]),
            rec_tds=float(row["rec_tds"]),
            first_downs=float(row["first_downs"]),
            games_together=int(row["games_together"]),
        )

    n = G.number_of_nodes()
    m = G.number_of_edges()

    density = nx.density(G) if n > 1 else 0.0
    weak_components = nx.number_weakly_connected_components(G) if n > 0 else 0
    strong_components = nx.number_strongly_connected_components(G) if n > 0 else 0

    # --- GLOBAL METRICS (Distance) ---
    if n > 0:
        G_undirected = G.to_undirected()
        largest_cc_nodes = max(nx.connected_components(G_undirected), key=len)
        G_largest_cc = G_undirected.subgraph(largest_cc_nodes)
        
        if len(G_largest_cc) > 1:
            diameter = nx.diameter(G_largest_cc)
            avg_path_length = nx.average_shortest_path_length(G_largest_cc)
        else:
            diameter = 0
            avg_path_length = 0.0
            
        avg_clustering = nx.average_clustering(G)
    else:
        diameter = 0
        avg_path_length = 0.0
        avg_clustering = 0.0

    # --- COMMUNITY DETECTION (Louvain Method) ---
    if n > 0:
        communities = community.louvain_communities(G_undirected, weight='weight', seed=42)
        modularity_score = community.modularity(G_undirected, communities, weight='weight')
        
        community_map = {}
        for comm_id, comm_nodes in enumerate(communities):
            for node in comm_nodes:
                community_map[node] = comm_id
    else:
        modularity_score = 0.0
        community_map = {}

    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    in_strength = dict(G.in_degree(weight="weight"))
    out_strength = dict(G.out_degree(weight="weight"))

    degree_c = nx.degree_centrality(G) if n > 1 else {node: 0.0 for node in G.nodes()}
    betweenness_c = nx.betweenness_centrality(G, weight=None, normalized=True) if n > 1 else {node: 0.0 for node in G.nodes()}
    pagerank_c = nx.pagerank(G, weight="weight") if n > 0 else {}

    try:
        assortativity = nx.degree_pearson_correlation_coefficient(G) if n > 1 and m > 0 else None
    except Exception:
        assortativity = None

    season_summary.append(
        {
            "season": int(season),
            "nodes": n,
            "edges": m,
            "density": density,
            "weak_components": weak_components,
            "strong_components": strong_components,
            "diameter": diameter,                   
            "avg_path_length": avg_path_length,     
            "avg_clustering": avg_clustering,
            "modularity": modularity_score, 
            "assortativity": assortativity,
        }
    )

    # --- ADD ATTRIBUTES TO GRAPH AND BUILD CSV ROWS ---
    for node in G.nodes():
        # 1. Save data for the CSV table
        player_rows.append(
            {
                "season": int(season),
                "player": node,
                "community_id": community_map.get(node, -1),
                "in_degree": in_degree.get(node, 0),
                "out_degree": out_degree.get(node, 0),
                "in_strength": in_strength.get(node, 0),
                "out_strength": out_strength.get(node, 0),
                "degree_centrality": degree_c.get(node, 0),
                "betweenness_centrality": betweenness_c.get(node, 0),
                "pagerank": pagerank_c.get(node, 0),
            }
        )
        
        # 2. Attach data directly to the NetworkX node for Gephi
        G.nodes[node]['label'] = node
        G.nodes[node]['community_id'] = community_map.get(node, -1)
        G.nodes[node]['pagerank'] = pagerank_c.get(node, 0.0)
        G.nodes[node]['in_strength'] = in_strength.get(node, 0.0)

    # Export the enriched Graph to GEXF
    out_file = OUT_GEXF / f"patriots_passing_{int(season)}.gexf"
    nx.write_gexf(G, out_file)
    print(f"Exported enriched network to {out_file}")

# --- SAVE DATA TABLES ---
season_df = pd.DataFrame(season_summary).sort_values("season")
players_df = pd.DataFrame(player_rows).sort_values(
    ["season", "pagerank"], ascending=[True, False]
)

season_df.to_csv(OUT_TABLES / "season_summary_metrics.csv", index=False)
players_df.to_csv(OUT_TABLES / "player_centrality_metrics.csv", index=False)

top_players = (
    players_df.sort_values(["season", "pagerank"], ascending=[True, False])
    .groupby("season", as_index=False, group_keys=False)
    .head(10)
)

top_players.to_csv(OUT_TABLES / "top10_players_by_season.csv", index=False)

print("\nSaved all metrics and graphs successfully.")