"""Part 1 / issue #11 (RQ2): signed triad census and structural balance vs. a sign-shuffled null.

Outputs
- results/part1_triads.csv       one row per (tie rule, triangle type), plus balanced / balanced_weak rows
- results/part1_triads_hubs.csv  share of triangles of each type that touch a top-k "negative hub"
- figures/part1_triads.{pdf,png} observed vs. null share per type, and observed / null ratio (log)
Run (from src/): python part1_triads.py

Graph: undirected signed core graph over the whole period (to_undirected_signed(load_graph())).
Each pair has one sign: + if it has more positive than negative links (both directions summed),
- if more negative. Pairs with equal counts ("ties", ~4.5k edges) are handled three ways:
tie_rule = "negative" (main rule, the load.py default), "positive", or "dropped" (tied edges removed,
so their triangles disappear too).
Triangle types by number of negative edges: +++ (0, balanced), ++- (1, unbalanced),
+-- (2, balanced), --- (3, unbalanced under strong balance; balanced under weak balance).
Null: keep the topology, shuffle the signs across edges (N_SHUFFLES permutations, seed SEED), which
keeps the number of negative edges fixed. z = (observed - null mean) / null std; p_emp is the two-sided
empirical p-value (share of shuffles at least as far from the null mean as observed, +1 smoothing;
the floor is 1 / (N_SHUFFLES + 1)). Summary rows: "balanced" = +++ and +-- (strong balance),
"balanced_weak" = everything except ++- (weak balance, Davis 1967).
Hubs: nodes ranked by number of negative edges (main rule); for k in HUB_K, the share of triangles of
each type with at least one top-k node.
"""
import igraph as ig
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from load import ROOT, load_graph, to_undirected_signed

RES = ROOT / "results"
FIG = ROOT / "figures"
N_SHUFFLES = 100
SEED = 42
TYPES = ["+++", "++-", "+--", "---"]  # index = number of negative edges
BALANCED = {"+++": True, "++-": False, "+--": True, "---": False}
TIE_RULES = {"negative": -1, "positive": 1, "dropped": 0}
HUB_K = [10, 50]


def triangles(U: nx.Graph) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """All triangles of U: (T, 3) edge indices, (T, 3) node indices, and the edge list they index."""
    nodes = list(U.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    edges = list(U.edges)
    g = ig.Graph(n=len(nodes), edges=[(idx[u], idx[v]) for u, v in edges])
    tri = np.array(g.list_triangles(), dtype=np.int64).reshape(-1, 3)
    if len(tri) == 0:
        return np.empty((0, 3), dtype=np.int64), tri, edges
    pairs = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]])
    eids = np.array(g.get_eids(pairs.tolist()), dtype=np.int64).reshape(3, -1).T
    return eids, tri, edges


def triangle_edges(U: nx.Graph) -> tuple[np.ndarray, list[tuple]]:
    """All triangles of U as a (T, 3) array of edge indices into the returned edge list."""
    eids, _, edges = triangles(U)
    return eids, edges


def census(neg: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Counts of +++, ++-, +--, --- given a boolean negative flag per edge."""
    return np.bincount(neg[tri].sum(axis=1), minlength=4)


def null_censuses(neg: np.ndarray, tri: np.ndarray, n=N_SHUFFLES, seed=SEED) -> np.ndarray:
    """(n, 4) triangle-type counts under sign shuffles that keep the number of negative edges."""
    rng = np.random.default_rng(seed)
    return np.array([census(rng.permutation(neg), tri) for _ in range(n)])


def compare(obs: np.ndarray, null: np.ndarray, tie_rule: str) -> pd.DataFrame:
    """Observed vs. null shares per type plus the strong / weak balance summary rows."""
    total = obs.sum()
    bal = np.array([BALANCED[t] for t in TYPES])
    weak = np.array([t != "++-" for t in TYPES])
    labels = TYPES + ["balanced", "balanced_weak"]
    obs_c = np.append(obs, [obs[bal].sum(), obs[weak].sum()])
    null_c = np.column_stack([null, null[:, bal].sum(axis=1), null[:, weak].sum(axis=1)])
    obs_s, null_s = obs_c / total, null_c / total
    mean, std = null_s.mean(axis=0), null_s.std(axis=0, ddof=1)
    dev = np.abs(null_s - mean) >= np.abs(obs_s - mean)
    return pd.DataFrame({
        "tie_rule": tie_rule, "type": labels, "n_triangles": total, "observed": obs_c,
        "observed_share": obs_s, "null_mean_share": mean, "null_std_share": std,
        "ratio_to_null": obs_s / mean, "z": (obs_s - mean) / std,
        "p_emp": (dev.sum(axis=0) + 1) / (len(null) + 1),
        "balanced_strong": [BALANCED.get(t) for t in labels],
    })


def hub_shares(neg: np.ndarray, eids: np.ndarray, nodes_tri: np.ndarray, edges_idx: np.ndarray,
               n_nodes: int, ks=HUB_K) -> pd.DataFrame:
    """Share of triangles of each type that contain >= 1 of the top-k nodes by negative-edge count."""
    neg_deg = np.bincount(edges_idx[neg].ravel(), minlength=n_nodes)
    types = neg[eids].sum(axis=1)
    rows = []
    for k in ks:
        hubs = np.zeros(n_nodes, dtype=bool)
        hubs[np.argsort(-neg_deg, kind="stable")[:k]] = True
        touch = hubs[nodes_tri].any(axis=1)
        for i, t in enumerate(TYPES):
            sel = types == i
            rows.append({"k": k, "type": t, "n_triangles": int(sel.sum()), "with_hub": int(touch[sel].sum()),
                         "share_with_hub": touch[sel].mean() if sel.any() else np.nan})
        rows.append({"k": k, "type": "all", "n_triangles": len(types), "with_hub": int(touch.sum()),
                     "share_with_hub": touch.mean()})
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> None:
    d = df[(df["tie_rule"] == "negative") & df["type"].isin(TYPES)]
    x = np.arange(len(d))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    ax1.bar(x - 0.2, d["observed_share"], 0.4, label="observed", color="#c33")
    ax1.bar(x + 0.2, d["null_mean_share"], 0.4, yerr=d["null_std_share"],
            label=f"sign-shuffled null (n={N_SHUFFLES})", color="#999", capsize=3)
    ax1.set_yscale("log")
    ax1.set_ylabel("share of triangles (log)")
    ax1.legend(frameon=False, fontsize=8)
    ticks = [f"{t}\n({'bal.' if BALANCED[t] else 'unbal.'})" for t in d["type"]]
    ax1.set_xticks(x, ticks)
    colors = ["#36c" if BALANCED[t] else "#c33" for t in d["type"]]
    for rule, marker in (("negative", "o"), ("positive", "s"), ("dropped", "^")):
        r = df[(df["tie_rule"] == rule) & df["type"].isin(TYPES)]
        ax2.scatter(x, r["ratio_to_null"], marker=marker, color=colors, label=f"ties {rule}", zorder=3)
    ax2.axhline(1, color="#666", lw=0.8, ls="--")
    ax2.set_yscale("log")
    ax2.set_xticks(x, ticks)
    ax2.set_yticks([0.9, 1, 1.5, 2, 3])
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_ylabel("observed / null share (log)")
    leg = ax2.legend(frameon=False, fontsize=8, title="blue = balanced, red = unbalanced", title_fontsize=7)
    for h in leg.legend_handles:
        h.set_color("#555")
    fig.suptitle("Signed triangles in the core graph (left: ties negative)", fontsize=11)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"part1_triads.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    RES.mkdir(exist_ok=True)
    G = load_graph()
    out, hubs = [], None
    for rule, tie in TIE_RULES.items():
        U = to_undirected_signed(G, tie=tie)
        if tie == 0:
            U.remove_edges_from([(u, v) for u, v, s in U.edges(data="sign") if s == 0])
        eids, nodes_tri, edges = triangles(U)
        neg = np.array([U[u][v]["sign"] < 0 for u, v in edges])
        obs = census(neg, eids)
        out.append(compare(obs, null_censuses(neg, eids), rule))
        print(f"ties {rule}: edges={len(edges):,}  negative edges={neg.mean():.2%}  triangles={len(eids):,}")
        if rule == "negative":
            idx = {n: i for i, n in enumerate(U.nodes)}
            edges_idx = np.array([(idx[u], idx[v]) for u, v in edges])
            hubs = hub_shares(neg, eids, nodes_tri, edges_idx, U.number_of_nodes())
    df = pd.concat(out, ignore_index=True)
    df.to_csv(RES / "part1_triads.csv", index=False)
    assert hubs is not None
    hubs.to_csv(RES / "part1_triads_hubs.csv", index=False)
    plot(df)
    with pd.option_context("display.width", 180, "display.float_format", "{:.4g}".format):
        print(df.drop(columns=["balanced_strong"]))
        print(hubs)


if __name__ == "__main__":
    main()
