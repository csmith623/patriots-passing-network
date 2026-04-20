from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "data" / "processed" / "patriots_master_table.csv"
SEASON_METRICS = ROOT / "outputs" / "tables" / "season_summary_metrics.csv"
PLAYER_METRICS = ROOT / "outputs" / "tables" / "player_centrality_metrics.csv"
FIG_DIR = ROOT / "outputs" / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)

season_df = pd.read_csv(SEASON_METRICS)
player_df = pd.read_csv(PLAYER_METRICS)
master_df = pd.read_csv(MASTER)

plt.style.use("seaborn-v0_8-whitegrid")

# 1. Season density comparison
plt.figure(figsize=(8, 5))
plt.bar(season_df["season"].astype(str), season_df["density"], color="#1f77b4")
plt.title("Patriots Passing Network Density by Season")
plt.xlabel("Season")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(FIG_DIR / "season_density.png", dpi=300)
plt.close()

# 2. Top 5 players by PageRank for each season
for season in sorted(player_df["season"].unique()):
    s = (
        player_df[player_df["season"] == season]
        .sort_values("pagerank", ascending=False)
        .head(5)
    )

    plt.figure(figsize=(8, 5))
    plt.barh(s["player"], s["pagerank"], color="#2a9d8f")
    plt.gca().invert_yaxis()
    plt.title(f"Top 5 Players by PageRank ({season})")
    plt.xlabel("PageRank")
    plt.ylabel("Player")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"top5_pagerank_{season}.png", dpi=300)
    plt.close()

# 3. Network plot for each season
for season in sorted(master_df["season"].unique()):
    s = (
        master_df[master_df["season"] == season]
        .groupby(["passer_player_name", "receiver_player_name"], as_index=False)
        .agg(weight=("receptions", "sum"))
    )

    G = nx.DiGraph()
    for _, row in s.iterrows():
        G.add_edge(
            row["passer_player_name"],
            row["receiver_player_name"],
            weight=row["weight"]
        )

    pos = nx.spring_layout(G, seed=42, k=1.2)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    node_sizes = [300 + 80 * G.degree(n) for n in G.nodes()]

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#457b9d", alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(
        G, pos,
        width=[0.5 + 0.25 * w for w in weights],
        alpha=0.6,
        edge_color="#999999",
        arrows=True,
        arrowsize=12
    )

    plt.title(f"Patriots Passing Network ({season})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"network_{season}.png", dpi=300)
    plt.close()

print("Saved figures to:", FIG_DIR)