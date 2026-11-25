"""Part 1 / issue #9 (RQ2): who attacks whom? Node roles on the core graph.

Outputs
- results/part1_node_roles.csv   one row per core subreddit (node features for Part 3)
- results/part1_roles_tests.csv  Mann-Whitney U tests, attackers / targets vs. the rest
- figures/part1_roles.{pdf,png}  box plots of each metric by group (log axes drop zeros; tests use all values)
Run (from src/): python part1_roles.py

Graph: directed core graph over the whole period (load_graph()).
Metrics per subreddit
- out_degree / in_degree: distinct subreddits linked to / linked from
- links_out / links_in:   hyperlinks sent / received (posts)
- neg_out / neg_in:       negative hyperlinks sent / received
- neg_out_ratio / neg_in_ratio: neg_out / links_out and neg_in / links_in (NaN when 0 links)
- pagerank: weighted by total links per pair, damping 0.85
- betweenness: exact, directed, unweighted, normalized (igraph)
Groups
- attacker: top N_TOP by neg_out; target: top N_TOP by neg_in; rest: neither.
  Ties at the cutoff are all kept, so a group can hold slightly more than N_TOP.
  A subreddit in both top lists counts in both groups (overlap is reported); each group
  is tested against `rest`, so the comparison groups never overlap.
Tests: two-sided Mann-Whitney U, rank-biserial effect size r = 2U / (n1 n2) - 1 (U for the group),
so r > 0 means the group tends to be higher than rest; p is Bonferroni-adjusted over all 12 tests.
"""
import igraph as ig
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from load import ROOT, load_graph

RES = ROOT / "results"
FIG = ROOT / "figures"
N_TOP = 100
METRICS = ["out_degree", "in_degree", "pagerank", "betweenness", "neg_out_ratio", "neg_in_ratio"]
LOG_METRICS = {"out_degree", "in_degree", "pagerank", "betweenness"}


def node_roles(G: nx.DiGraph) -> pd.DataFrame:
    nodes = sorted(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    pairs = [(idx[u], idx[v], d["pos_w"] + d["neg_w"], d["neg_w"]) for u, v, d in G.edges(data=True)]
    src, dst, w, neg = (np.array(x) for x in zip(*pairs))
    n = len(nodes)

    df = pd.DataFrame(index=pd.Index(nodes, name="subreddit"))
    df["out_degree"] = np.bincount(src, minlength=n)
    df["in_degree"] = np.bincount(dst, minlength=n)
    df["links_out"] = np.bincount(src, weights=w, minlength=n).astype(int)
    df["links_in"] = np.bincount(dst, weights=w, minlength=n).astype(int)
    df["neg_out"] = np.bincount(src, weights=neg, minlength=n).astype(int)
    df["neg_in"] = np.bincount(dst, weights=neg, minlength=n).astype(int)
    df["neg_out_ratio"] = df["neg_out"] / df["links_out"].replace(0, np.nan)
    df["neg_in_ratio"] = df["neg_in"] / df["links_in"].replace(0, np.nan)

    g = ig.Graph(n=n, edges=list(zip(src.tolist(), dst.tolist())), directed=True)
    df["pagerank"] = g.pagerank(weights=w.tolist(), damping=0.85, directed=True)
    raw = np.array(g.betweenness(directed=True))
    df["betweenness"] = raw / ((n - 1) * (n - 2))  # normalized like networkx for directed graphs
    return df


def assign_groups(df: pd.DataFrame) -> pd.DataFrame:
    attackers = set(df["neg_out"].nlargest(N_TOP, keep="all").index)
    targets = set(df["neg_in"].nlargest(N_TOP, keep="all").index)
    df["is_attacker"] = df.index.isin(attackers)
    df["is_target"] = df.index.isin(targets)
    df["is_rest"] = ~(df["is_attacker"] | df["is_target"])
    return df


def run_tests(df: pd.DataFrame) -> pd.DataFrame:
    rest = df[df["is_rest"]]
    rows = []
    for group in ("attacker", "target"):
        grp = df[df[f"is_{group}"]]
        for m in METRICS:
            a, b = grp[m].dropna(), rest[m].dropna()
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            rows.append({
                "group": group, "metric": m, "n_group": len(a), "n_rest": len(b),
                "median_group": a.median(), "median_rest": b.median(),
                "U": u, "p": p, "rank_biserial": 2 * u / (len(a) * len(b)) - 1,
            })
    tests = pd.DataFrame(rows)
    tests["p_bonferroni"] = (tests["p"] * len(tests)).clip(upper=1.0)
    return tests


def plot(df: pd.DataFrame) -> None:
    groups = [("attacker", "#c33"), ("target", "#36c"), ("rest", "#999")]
    fig, axes = plt.subplots(2, 3, figsize=(9, 5))
    for ax, m in zip(axes.flat, METRICS):
        data = [df.loc[df[f"is_{gname}"], m].dropna() for gname, _ in groups]
        if m in LOG_METRICS:
            data = [d[d > 0] for d in data]
            ax.set_yscale("log")
        bp = ax.boxplot(data, tick_labels=[gname for gname, _ in groups], showfliers=False, patch_artist=True)
        for patch, (_, color) in zip(bp["boxes"], groups):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_title(m.replace("_", " "), fontsize=10)
    fig.suptitle(f"Top-{N_TOP} attackers and targets vs. the rest (core graph)", fontsize=11)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"part1_roles.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    RES.mkdir(exist_ok=True)
    df = assign_groups(node_roles(load_graph()))
    df.to_csv(RES / "part1_node_roles.csv")
    tests = run_tests(df)
    tests.to_csv(RES / "part1_roles_tests.csv", index=False)
    plot(df)
    both = int((df["is_attacker"] & df["is_target"]).sum())
    print(f"nodes={len(df):,}  attackers={df.is_attacker.sum()}  targets={df.is_target.sum()}  in both={both}")
    with pd.option_context("display.width", 160, "display.float_format", "{:.4g}".format):
        print(tests.drop(columns=["U"]))


if __name__ == "__main__":
    main()
