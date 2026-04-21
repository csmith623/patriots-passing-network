from pathlib import Path
import random
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
IN_FILE = ROOT / "data" / "processed" / "patriots_master_table.csv"
OUT_DIR = ROOT / "outputs" / "tables"

df = pd.read_csv(IN_FILE)
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

def build_graph(season_df):
    edge_df = (
        season_df.groupby(["passer_player_name", "receiver_player_name"], as_index=False)
        .agg(
            weight=("receptions", "sum"),
            rec_yards=("rec_yards", "sum"),
            rec_tds=("rec_tds", "sum"),
            first_downs=("first_downs", "sum"),
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
        )
    return G

def largest_weak_component_size(G):
    if G.number_of_nodes() == 0:
        return 0
    return len(max(nx.weakly_connected_components(G), key=len))

def targeted_attack_curve(G):
    H = G.copy()
    rows = []
    initial_nodes = H.number_of_nodes()

    rows.append(
        {
            "step": 0,
            "fraction_removed": 0.0,
            "largest_weak_component": largest_weak_component_size(H),
            "remaining_nodes": H.number_of_nodes(),
            "strategy": "targeted",
        }
    )

    removed = 0
    while H.number_of_nodes() > 0:
        node_strength = dict(H.degree(weight="weight"))
        target = max(node_strength, key=node_strength.get)
        H.remove_node(target)
        removed += 1

        rows.append(
            {
                "step": removed,
                "fraction_removed": removed / initial_nodes if initial_nodes > 0 else 0.0,
                "largest_weak_component": largest_weak_component_size(H),
                "remaining_nodes": H.number_of_nodes(),
                "strategy": "targeted",
            }
        )

    return pd.DataFrame(rows)

def random_attack_curve(G, runs=100):
    initial_nodes = G.number_of_nodes()
    all_runs = []

    for run in range(runs):
        H = G.copy()
        nodes = list(H.nodes())
        random.shuffle(nodes)

        rows = [
            {
                "step": 0,
                "fraction_removed": 0.0,
                "largest_weak_component": largest_weak_component_size(H),
                "remaining_nodes": H.number_of_nodes(),
                "run": run,
            }
        ]

        removed = 0
        for node in nodes:
            if node in H:
                H.remove_node(node)
                removed += 1
                rows.append(
                    {
                        "step": removed,
                        "fraction_removed": removed / initial_nodes if initial_nodes > 0 else 0.0,
                        "largest_weak_component": largest_weak_component_size(H),
                        "remaining_nodes": H.number_of_nodes(),
                        "run": run,
                    }
                )

        all_runs.append(pd.DataFrame(rows))

    combined = pd.concat(all_runs, ignore_index=True)
    avg = (
        combined.groupby(["step", "fraction_removed"], as_index=False)
        .agg(
            largest_weak_component=("largest_weak_component", "mean"),
            remaining_nodes=("remaining_nodes", "mean"),
        )
    )
    avg["strategy"] = "random"
    return avg

results = []

for season in sorted(df["season"].unique()):
    season_df = df[df["season"] == season].copy()
    G = build_graph(season_df)

    targeted = targeted_attack_curve(G)
    targeted["season"] = int(season)

    random_avg = random_attack_curve(G, runs=100)
    random_avg["season"] = int(season)

    results.append(targeted)
    results.append(random_avg)

robustness_df = pd.concat(results, ignore_index=True)
robustness_df = robustness_df[
    ["season", "strategy", "step", "fraction_removed", "largest_weak_component", "remaining_nodes"]
].sort_values(["season", "strategy", "step"])

robustness_df.to_csv(OUT_DIR / "robustness_results.csv", index=False)

print("Saved:")
print(OUT_DIR / "robustness_results.csv")
print()
print(robustness_df.head(20).to_string(index=False))