"""Preliminary descriptive + community/polarization analysis for the project update.

Outputs: results/prelim_*.csv, results/prelim_SUMMARY.md, figures/prelim_*.pdf/.png
Run: python src/prelim_analysis.py
"""
import json
from pathlib import Path

import igraph as ig
import leidenalg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from load import ROOT, aggregate, load_core_nodes, load_edges

RES = ROOT / "results"
FIG = ROOT / "figures"
SEED = 42
N_SHUFFLES = 100


def savefig(name):
    for ext in ("pdf", "png"):
        plt.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close()


def describe(edges, core):
    stats = {}
    for label, df in (("full", edges), ("core", edges[edges.src.isin(core) & edges.dst.isin(core)])):
        agg = aggregate(df)
        G = nx.DiGraph()
        G.add_edges_from(zip(agg.src, agg.dst))
        U = G.to_undirected()
        lcc = max(nx.connected_components(U), key=len)
        neg_pairs = set(zip(agg.src[agg.neg_w > 0], agg.dst[agg.neg_w > 0]))
        stats[label] = {
            "nodes": G.number_of_nodes(),
            "links (posts)": len(df),
            "distinct directed pairs": G.number_of_edges(),
            "% negative links": round(100 * (df.sign < 0).mean(), 2),
            "density": G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1)),
            "reciprocity": round(nx.reciprocity(G), 3),
            "avg clustering (undirected)": round(nx.average_clustering(U), 3) if label == "core" else None,
            "largest WCC share": round(len(lcc) / U.number_of_nodes(), 3),
            "neg pairs reciprocated (neg)": round(np.mean([(v, u) in neg_pairs for u, v in neg_pairs]), 3),
        }
    tab = pd.DataFrame(stats)
    tab.to_csv(RES / "prelim_descriptive.csv")
    return tab


def plot_activity(edges):
    import matplotlib.dates as mdates

    m = edges[edges.ts >= "2014-01-01"].set_index("ts").resample("ME")["sign"]
    monthly = pd.DataFrame({"links": m.size(), "neg_share": m.apply(lambda s: (s < 0).mean())})
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.bar(monthly.index, monthly["links"] / 1000, width=25, color="#9ab")
    ax.set_ylabel("links / month (k)")
    ax2 = ax.twinx()
    ax2.plot(monthly.index, 100 * monthly["neg_share"], color="#c33", lw=1.5)
    ax2.set_ylabel("% negative", color="#c33")
    ax2.set_ylim(0, 15)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("Monthly hyperlink volume and negative share")
    savefig("prelim_activity")


def plot_degree(edges):
    agg = aggregate(edges)
    out_deg = agg.groupby("src").size()
    in_deg = agg.groupby("dst").size()
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    for s, lab in ((out_deg, "out"), (in_deg, "in")):
        vc = s.value_counts().sort_index()
        ax.loglog(vc.index, vc.values, ".", ms=3, label=f"{lab}-degree")
    ax.set_xlabel("degree (distinct neighbors)")
    ax.set_ylabel("# subreddits")
    ax.legend(frameon=False)
    ax.set_title("Degree distribution")
    savefig("prelim_degree")


def top_actors(edges):
    neg = edges[edges.sign < 0]
    tab = pd.DataFrame({
        "top_attackers": neg.src.value_counts().head(10).index,
        "neg_sent": neg.src.value_counts().head(10).values,
        "top_targets": neg.dst.value_counts().head(10).index,
        "neg_received": neg.dst.value_counts().head(10).values,
    })
    tab.to_csv(RES / "prelim_top_actors.csv", index=False)
    return tab


def communities(edges, core):
    """Detect communities on the undirected positive core graph; measure where negative links fall."""
    ce = edges[edges.src.isin(core) & edges.dst.isin(core)]
    agg = aggregate(ce)
    pos = agg[agg.pos_w > 0]
    # undirected positive weights
    und = pos.assign(a=np.minimum(pos.src, pos.dst), b=np.maximum(pos.src, pos.dst)).groupby(["a", "b"]).pos_w.sum().reset_index()
    names = sorted(set(und.a) | set(und.b))
    idx = {n: i for i, n in enumerate(names)}
    g = ig.Graph(n=len(names), edges=list(zip(und.a.map(idx), und.b.map(idx))))
    g.es["weight"] = und.pos_w.tolist()

    parts = {}
    leiden = leidenalg.find_partition(
        g, leidenalg.ModularityVertexPartition, weights="weight", seed=SEED, n_iterations=-1
    )
    parts["Leiden"] = (leiden.membership, g.modularity(leiden.membership, weights="weight"))
    louv = g.community_multilevel(weights="weight")
    parts["Louvain"] = (louv.membership, g.modularity(louv.membership, weights="weight"))

    # negative links among nodes that are in the positive graph
    neg = ce[(ce.sign < 0) & ce.src.isin(idx) & ce.dst.isin(idx)]
    allk = ce[ce.src.isin(idx) & ce.dst.isin(idx)]
    rng = np.random.default_rng(SEED)
    rows = []
    for name, (memb, Q) in parts.items():
        memb = np.asarray(memb)
        cs = memb[allk.src.map(idx).values]
        cd = memb[allk.dst.map(idx).values]
        within_all = cs == cd
        is_neg = (allk.sign < 0).values
        obs_neg_within = within_all[is_neg].mean()
        obs_pos_within = within_all[~is_neg].mean()
        null = [within_all[rng.permutation(is_neg)].mean() for _ in range(N_SHUFFLES)]
        sizes = np.bincount(memb)
        rows.append({
            "method": name,
            "modularity": round(Q, 3),
            "n_communities": len(sizes),
            "n_comm_ge_50": int((sizes >= 50).sum()),
            "largest_comm_share": round(sizes.max() / len(memb), 3),
            "% pos links within community": round(100 * obs_pos_within, 1),
            "% neg links within community": round(100 * obs_neg_within, 1),
            "% neg within (sign-shuffle null, mean)": round(100 * np.mean(null), 1),
            "null std": round(100 * np.std(null), 2),
            "neg rate within comm (%)": round(100 * is_neg[within_all].mean(), 2),
            "neg rate between comm (%)": round(100 * is_neg[~within_all].mean(), 2),
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "prelim_communities.csv", index=False)

    # Leiden membership for downstream parts + hostility heatmap among the largest communities
    memb = np.asarray(parts["Leiden"][0])
    pd.DataFrame({"subreddit": names, "community_id": memb}).to_csv(RES / "prelim_leiden_membership.csv", index=False)
    sizes = pd.Series(memb).value_counts()
    top = sizes.index[:10]
    # label each community with its 3 most active subreddits
    act = pd.concat([allk.src, allk.dst]).value_counts()
    lab = {}
    for c in top:
        members = [n for n in names if memb[idx[n]] == c]
        lab[c] = ", ".join(act.reindex(members).sort_values(ascending=False).index[:3])
    cs = pd.Series(memb[allk.src.map(idx).values])
    cd = pd.Series(memb[allk.dst.map(idx).values])
    df = pd.DataFrame({"cs": cs.values, "cd": cd.values, "neg": (allk.sign < 0).values})
    df = df[df.cs.isin(top) & df.cd.isin(top)]
    total = df.groupby(["cs", "cd"]).size().unstack().reindex(index=top, columns=top)
    negs = df[df.neg].groupby(["cs", "cd"]).size().unstack().reindex(index=top, columns=top).fillna(0)
    host = 100 * negs / total
    host.index = [f"C{i}" for i in range(len(top))]
    host.columns = host.index
    host.to_csv(RES / "prelim_hostility_matrix.csv")
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(host.values, cmap="Reds", vmin=0)
    ax.set_xticks(range(len(top)), host.columns, fontsize=7)
    ax.set_yticks(range(len(top)), host.index, fontsize=7)
    ax.set_xlabel("target community")
    ax.set_ylabel("source community")
    ax.set_title("% negative links (Leiden, top 10)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)
    savefig("prelim_hostility")
    with open(RES / "prelim_community_labels.json", "w") as f:
        json.dump({f"C{i}": lab[c] for i, c in enumerate(top)} | {f"C{i}_size": int(sizes[c]) for i, c in enumerate(top)}, f, indent=1)
    return tab, lab


def main():
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    edges = load_edges()
    core = set(load_core_nodes())
    desc = describe(edges, core)
    plot_activity(edges)
    plot_degree(edges)
    top = top_actors(edges)
    comm, labels = communities(edges, core)
    print(desc.to_string(), "\n")
    print(top.to_string(), "\n")
    print(comm.T.to_string(), "\n")
    print(json.load(open(RES / "prelim_community_labels.json")))


if __name__ == "__main__":
    main()
