"""Part 2 / issue #13: community detection on the core graph with Leiden (main), Louvain and spectral clustering.

Outputs
- results/part2_node_community.csv   subreddit, community_id   (main Leiden partition; Part 3 uses it as a feature)
- results/part2_memberships.csv      subreddit + one column per method (leiden, louvain, spectral, leiden_all_edges)
- results/part2_communities.csv      one row per method: modularity, #communities, size distribution
- results/part2_community_sizes.csv  method, community_id, size
- results/part2_method_ari.csv       adjusted Rand index between every pair of methods
- results/part2_leiden_seeds.csv     Leiden stability over SEEDS (modularity, #communities, ARI to the main run)
- figures/part2_community_sizes.{pdf,png}  rank-size plot of community sizes per method
Run (from src/): python part2_communities.py

Graph: the undirected core graph over the whole period, nodes = core subreddits with at least one core link
(7,854; the 5 core subreddits whose links all leave the core get no community). Main input ("positive"):
edge weight = number of positive links between the pair, both directions summed, so communities are groups that
link to each other approvingly. "all edges" uses every link (positive + negative) as weight and is used by the
negative-edge comparison (#16) and as a robustness check for the polarization metrics (#15).

Methods
- Leiden: leidenalg ModularityVertexPartition, iterated until stable (n_iterations=-1), seed SEED.
- Louvain: igraph community_multilevel (Blondel et al. 2008) with igraph's RNG seeded to SEED.
- Spectral: regularized spectral clustering (Qin & Rohe 2013) on the same weighted graph: k-means on the
  row-normalized top-k eigenvectors of D_tau^-1/2 A D_tau^-1/2, tau = mean weighted degree. Plain normalized
  spectral clustering (sklearn SpectralClustering) is degenerate here: one cluster takes 82-99% of the nodes.
  The main spectral partition uses k = the number of Leiden communities, for a like-for-like comparison;
  the sweep over SPECTRAL_K is in part2_communities.csv (rows "spectral_k=..."). Its modularity rises slowly
  with k (0.35 at k=10, ~0.40 from k=40 on), so k matters less than the method.
Community ids are relabeled by size: 0 is the largest community of each method.
Modularity is always the weighted Newman modularity of the partition on the graph the method ran on.
"""
import random

import igraph as ig
import leidenalg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from load import ROOT, aggregate, load_edges

RES = ROOT / "results"
FIG = ROOT / "figures"
SEED = 42
SEEDS = range(10)                      # Leiden stability check
SPECTRAL_K = [10, 20, 30, 40, 50, 60]  # candidate numbers of clusters for spectral clustering
MAIN = "leiden"


def undirected_weights(edges: pd.DataFrame, kind: str = "positive") -> pd.DataFrame:
    """One row per unordered pair (a < b) with weight w = positive links (kind="positive") or all links ("all").

    Pairs with zero weight are dropped (e.g. negative-only pairs in the positive graph).
    """
    agg = aggregate(edges)
    w = agg["pos_w"] if kind == "positive" else agg["pos_w"] + agg["neg_w"]
    a, b = np.minimum(agg["src"], agg["dst"]), np.maximum(agg["src"], agg["dst"])
    und = pd.DataFrame({"a": a, "b": b, "w": w})
    und = und[(und["a"] != und["b"]) & (und["w"] > 0)]
    return und.groupby(["a", "b"], as_index=False)["w"].sum()


def to_igraph(nodes: list[str], und: pd.DataFrame) -> ig.Graph:
    """Weighted undirected igraph over `nodes` (vertex attribute "name"); nodes without an edge stay isolated."""
    idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=list(zip(und["a"].map(idx), und["b"].map(idx))))
    g.vs["name"] = nodes
    g.es["weight"] = und["w"].astype(float).tolist()
    return g


def relabel_by_size(memb) -> np.ndarray:
    """Renumber communities so 0 is the largest (ties broken by the smallest original id)."""
    memb = np.asarray(memb)
    ids, counts = np.unique(memb, return_counts=True)
    order = ids[np.lexsort((ids, -counts))]
    new = np.empty(ids.max() + 1, dtype=np.int64)
    new[order] = np.arange(len(order))
    return new[memb]


def leiden(g: ig.Graph, seed: int = SEED) -> np.ndarray:
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights="weight",
                                    seed=seed, n_iterations=-1)
    return relabel_by_size(part.membership)


def louvain(g: ig.Graph, seed: int = SEED) -> np.ndarray:
    ig.set_random_number_generator(random.Random(seed))
    try:
        return relabel_by_size(g.community_multilevel(weights="weight").membership)
    finally:
        ig.set_random_number_generator(random)


def spectral(g: ig.Graph, k: int, seed: int = SEED, tau: float | None = None) -> np.ndarray:
    """Regularized spectral clustering (Qin & Rohe 2013) into k clusters.

    Embedding: top-k eigenvectors of D_tau^-1/2 A D_tau^-1/2 with D_tau = D + tau I (tau = mean weighted degree
    by default), rows scaled to unit length, then k-means (n_init=10). Without the tau term the hubs dominate
    the eigenvectors and one cluster swallows 80-99% of the nodes.
    """
    el = np.array(g.get_edgelist())
    w = np.array(g.es["weight"], dtype=float)
    n = g.vcount()
    A = sparse.coo_matrix((np.r_[w, w], (np.r_[el[:, 0], el[:, 1]], np.r_[el[:, 1], el[:, 0]])), shape=(n, n)).tocsr()
    d = np.asarray(A.sum(axis=1)).ravel()
    s = 1 / np.sqrt(d + (d.mean() if tau is None else tau))
    L = sparse.diags(s) @ A @ sparse.diags(s)
    _, vecs = eigsh(L, k=k, which="LA", v0=np.random.default_rng(seed).random(n))
    X = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    return relabel_by_size(KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X))


def summarize(method: str, g: ig.Graph, memb: np.ndarray, graph: str) -> dict:
    sizes = np.bincount(memb)
    return {
        "method": method, "graph": graph, "modularity": g.modularity(memb.tolist(), weights="weight"),
        "n_nodes": len(memb), "n_communities": len(sizes), "n_comm_ge_50": int((sizes >= 50).sum()),
        "n_singletons": int((sizes == 1).sum()), "largest_share": sizes.max() / len(memb),
        "top10_share": np.sort(sizes)[::-1][:10].sum() / len(memb), "median_size": float(np.median(sizes)),
    }


def plot_sizes(memberships: pd.DataFrame, methods: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    for m in methods:
        s = np.sort(np.bincount(memberships[m].to_numpy()))[::-1]
        ax.loglog(np.arange(1, len(s) + 1), s, marker=".", ms=4, lw=1, label=m)
    ax.set_xlabel("community rank")
    ax.set_ylabel("size (subreddits)")
    ax.set_title("Community sizes, core graph", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"part2_community_sizes.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    RES.mkdir(exist_ok=True)
    edges = load_edges(core_only=True)
    nodes = sorted(set(edges["src"]) | set(edges["dst"]))
    g_pos = to_igraph(nodes, undirected_weights(edges, "positive"))
    g_all = to_igraph(nodes, undirected_weights(edges, "all"))

    memb = {"leiden": leiden(g_pos)}
    out = pd.DataFrame({"subreddit": nodes, "community_id": memb["leiden"]})
    out.to_csv(RES / "part2_node_community.csv", index=False)  # written first: Part 3 needs it
    rows = [summarize("leiden", g_pos, memb["leiden"], "positive")]
    memb["louvain"] = louvain(g_pos)
    rows.append(summarize("louvain", g_pos, memb["louvain"], "positive"))

    for k in SPECTRAL_K:
        r = summarize(f"spectral_k={k}", g_pos, spectral(g_pos, k), "positive")
        rows.append(r)
        print(f"spectral k={k}: Q={r['modularity']:.3f}")
    k_main = int(memb["leiden"].max() + 1)
    memb["spectral"] = spectral(g_pos, k_main)
    rows.append(summarize("spectral", g_pos, memb["spectral"], "positive") | {"spectral_k": k_main})
    memb["leiden_all_edges"] = leiden(g_all)
    rows.append(summarize("leiden_all_edges", g_all, memb["leiden_all_edges"], "all edges"))
    comm = pd.DataFrame(rows)
    comm.to_csv(RES / "part2_communities.csv", index=False)

    methods = ["leiden", "louvain", "spectral", "leiden_all_edges"]
    memberships = pd.DataFrame({"subreddit": nodes} | {m: memb[m] for m in methods})
    memberships.to_csv(RES / "part2_memberships.csv", index=False)
    pd.concat([pd.DataFrame({"method": m, "community_id": np.arange(memb[m].max() + 1),
                             "size": np.bincount(memb[m])}) for m in methods]).to_csv(
        RES / "part2_community_sizes.csv", index=False)
    ari = pd.DataFrame([{"method_a": a, "method_b": b, "ari": adjusted_rand_score(memb[a], memb[b])}
                        for i, a in enumerate(methods) for b in methods[i + 1:]])
    ari.to_csv(RES / "part2_method_ari.csv", index=False)

    seeds = []
    for s in SEEDS:
        m = leiden(g_pos, seed=s)
        seeds.append({"seed": s, "modularity": g_pos.modularity(m.tolist(), weights="weight"),
                      "n_communities": int(m.max() + 1), "n_comm_ge_50": int((np.bincount(m) >= 50).sum()),
                      "ari_to_main": adjusted_rand_score(memb["leiden"], m)})
    seeds = pd.DataFrame(seeds)
    seeds.to_csv(RES / "part2_leiden_seeds.csv", index=False)
    plot_sizes(memberships, ["leiden", "louvain", "spectral"])

    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(comm.to_string(index=False))
        print(ari.to_string(index=False))
        print(seeds.describe().loc[["mean", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
