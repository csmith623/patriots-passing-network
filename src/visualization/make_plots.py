from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "data" / "processed" / "patriots_master_table.csv"
SEASON_METRICS = ROOT / "outputs" / "tables" / "season_summary_metrics.csv"
PLAYER_METRICS = ROOT / "outputs" / "tables" / "player_centrality_metrics.csv"
WEEKLY = ROOT / "outputs" / "tables" / "weekly_network_metrics.csv"
CUMULATIVE = ROOT / "outputs" / "tables" / "cumulative_network_metrics.csv"
ROBUSTNESS = ROOT / "outputs" / "tables" / "robustness_results.csv"
FIG_DIR = ROOT / "outputs" / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)

season_df = pd.read_csv(SEASON_METRICS)
player_df = pd.read_csv(PLAYER_METRICS)
master_df = pd.read_csv(MASTER)
weekly_df = pd.read_csv(WEEKLY)
cumulative_df = pd.read_csv(CUMULATIVE)
robust_df = pd.read_csv(ROBUSTNESS)

plt.style.use("seaborn-v0_8-whitegrid")

season_colors = {
    2006: "#1f77b4",
    2007: "#d62728",
    2008: "#2ca02c",
    2016: "#9467bd",
}

plt.figure(figsize=(8, 5))
plt.bar(season_df["season"].astype(str), season_df["density"], color="#1f77b4")
plt.title("Patriots Passing Network Density by Season")
plt.xlabel("Season")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(FIG_DIR / "season_density.png", dpi=300)
plt.close()

for season in sorted(player_df["season"].unique()):
    s = (
        player_df[player_df["season"] == season]
        .sort_values("pagerank", ascending=False)
        .head(5)
        .sort_values("pagerank", ascending=True)
    )

    plt.figure(figsize=(8, 5))
    plt.barh(s["player"], s["pagerank"], color="#2a9d8f")
    plt.title(f"Top 5 Players by PageRank ({season})")
    plt.xlabel("PageRank")
    plt.ylabel("Player")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"top5_pagerank_{season}.png", dpi=300)
    plt.close()

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
            weight=float(row["weight"]),
        )

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=1.2)

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    node_sizes = [350 + 90 * G.degree(n) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#457b9d", alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(
        G,
        pos,
        width=[0.5 + 0.20 * w for w in weights],
        alpha=0.6,
        edge_color="#999999",
        arrows=True,
        arrowsize=12,
    )

    plt.title(f"Patriots Passing Network ({season})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"network_{season}.png", dpi=300)
    plt.close()

plt.figure(figsize=(10, 6))
for season in sorted(weekly_df["season"].unique()):
    s = weekly_df[weekly_df["season"] == season].sort_values("week")
    plt.plot(
        s["week"],
        s["density"],
        marker="o",
        linewidth=2,
        label=str(season),
        color=season_colors.get(season, None),
    )
plt.title("Weekly Passing-Network Density by Season")
plt.xlabel("Week")
plt.ylabel("Density")
plt.xticks(sorted(weekly_df["week"].unique()))
plt.legend(title="Season")
plt.tight_layout()
plt.savefig(FIG_DIR / "weekly_density_by_season.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
for season in sorted(cumulative_df["season"].unique()):
    s = cumulative_df[cumulative_df["season"] == season].sort_values("week")
    plt.plot(
        s["week"],
        s["edges"],
        marker="o",
        linewidth=2,
        label=str(season),
        color=season_colors.get(season, None),
    )
plt.title("Cumulative Distinct Passing Edges by Season")
plt.xlabel("Week")
plt.ylabel("Cumulative Edge Count")
plt.xticks(sorted(cumulative_df["week"].unique()))
plt.legend(title="Season")
plt.tight_layout()
plt.savefig(FIG_DIR / "cumulative_edges_by_season.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
for season in sorted(cumulative_df["season"].unique()):
    s = cumulative_df[cumulative_df["season"] == season].sort_values("week")
    plt.plot(
        s["week"],
        s["max_in_strength"],
        marker="o",
        linewidth=2,
        label=str(season),
        color=season_colors.get(season, None),
    )
plt.title("Cumulative Max Receiver In-Strength by Season")
plt.xlabel("Week")
plt.ylabel("Max In-Strength")
plt.xticks(sorted(cumulative_df["week"].unique()))
plt.legend(title="Season")
plt.tight_layout()
plt.savefig(FIG_DIR / "cumulative_max_in_strength_by_season.png", dpi=300)
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
axes = axes.flatten()

for ax, season in zip(axes, sorted(robust_df["season"].unique())):
    s = robust_df[robust_df["season"] == season].copy()

    for strategy, color in [("random", "#1f77b4"), ("targeted", "#d62728")]:
        sub = s[s["strategy"] == strategy].sort_values("fraction_removed")
        ax.plot(
            sub["fraction_removed"],
            sub["largest_weak_component"],
            marker="o",
            linewidth=2,
            markersize=3,
            label=strategy.capitalize(),
            color=color,
        )

    ax.set_title(f"{season}")
    ax.set_xlabel("Fraction of Nodes Removed")
    ax.set_ylabel("Largest Weak Component Size")
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle("Robustness of Patriots Passing Networks: Random vs Targeted Removal", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "robustness_targeted_vs_random_all_seasons.png", dpi=300, bbox_inches="tight")
plt.close()

player_df["total_degree"] = player_df["in_degree"] + player_df["out_degree"]

plt.figure(figsize=(8, 5))
plt.hist(
    player_df["total_degree"], 
    bins=range(0, int(player_df["total_degree"].max()) + 2), 
    align='left', 
    color="#e76f51", 
    edgecolor="black", 
    alpha=0.9
)
plt.title("Degree Distribution of Patriots Passing Networks (2006-2016)")
plt.xlabel("Node Degree (Total Connections)")
plt.ylabel("Frequency (Number of Players)")
plt.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(FIG_DIR / "degree_distribution.png", dpi=300)
plt.close()

print(f"Saved figures to: {FIG_DIR}")