"""Part 2 / issue #15 (RQ1): is negative linking concentrated between communities? Hostility matrix.

Outputs
- results/part2_polarization.csv         one row per (partition, unit): within/between shares and rates vs. null
- results/part2_hostility_matrix.csv     % negative links, source community (rows) x target community (columns),
                                         for the labeled Leiden communities (part2_community_labels.json)
- results/part2_hostility_long.csv       the same cells with link counts, null mean/std, ratio and z
- figures/part2_hostility.{pdf,png}      heatmaps: % negative, and observed / expected (log2)
Run (from src/, after part2_communities.py and part2_labels.py): python part2_polarization.py

Units. "link": every hyperlink counts once (busy pairs weigh more). "pair": each directed pair (src, dst) counts
once and is negative if it has at least one negative link (the Part 1 definition).
Partitions (from part2_memberships.csv unless noted):
- leiden / louvain / spectral: communities of the positive-link graph (main analysis).
- leiden_all_edges: communities of the graph with every link as weight, blind to sign.
- leiden_pre_2016-07: Leiden (seed 42) on the positive graph of links before CUTOFF, evaluated only on links
  from CUTOFF on between subreddits that it covers (out of sample in time).
Why the robustness rows: communities found on positive links put positive links inside communities by
construction, so "negatives fall between communities more than positives" is partly built in for the main rows.
The sign-blind and out-of-sample partitions do not use the signs of the links they are tested on.
Null: shuffle signs across units (links or pairs) N_SHUFFLES times with seed SEED, which keeps the partition,
the topology and the number of negative units. z = (observed - null mean) / null std; p_emp is two-sided with
+1 smoothing (floor 1 / (N_SHUFFLES + 1)).
Hostility: for source community i and target community j, neg rate = negative links / all links from i to j.
Expected under the null = the overall negative rate among links between labeled communities; ratio = observed /
expected, z from the same shuffles (restricted to links between labeled communities).
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load import ROOT, aggregate, load_edges
from part2_communities import leiden, to_igraph, undirected_weights

RES = ROOT / "results"
FIG = ROOT / "figures"
SEED = 42
N_SHUFFLES = 100
CUTOFF = "2016-07-01"  # Part 3 train / validation boundary
MIN_CELL_LINKS = 100  # hostility cells with fewer links are greyed out in the ratio heatmap
PARTITIONS = ["leiden", "louvain", "spectral", "leiden_all_edges"]


def p_emp(obs: float, null: np.ndarray) -> float:
    dev = np.abs(null - null.mean()) >= abs(obs - null.mean())
    return (dev.sum() + 1) / (len(null) + 1)


def polarization(cs: np.ndarray, cd: np.ndarray, neg: np.ndarray, rng: np.random.Generator,
                 n_shuffles: int = N_SHUFFLES) -> dict:
    """Within/between statistics for units with source / target community cs, cd and a negative flag."""
    within = cs == cd
    neg = np.asarray(neg, dtype=bool)
    obs = within[neg].mean()
    null = np.array([within[rng.permutation(neg)].mean() for _ in range(n_shuffles)])
    rate_w, rate_b = neg[within].mean(), neg[~within].mean()
    return {
        "n_units": len(neg), "n_negative": int(neg.sum()), "share_within_all": within.mean(),
        "share_pos_within": within[~neg].mean(), "share_neg_within": obs,
        "null_share_neg_within": null.mean(), "null_std": null.std(ddof=1),
        "z": (obs - null.mean()) / null.std(ddof=1), "p_emp": p_emp(obs, null),
        "neg_rate_within": rate_w, "neg_rate_between": rate_b, "between_within_ratio": rate_b / rate_w,
    }


def units(edges: pd.DataFrame, memb: pd.Series, unit: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(source community, target community, negative flag) per link or per directed pair, for covered nodes."""
    df = aggregate(edges).assign(neg=lambda d: d["neg_w"] > 0) if unit == "pair" else edges.assign(neg=edges["sign"] < 0)
    df = df[df["src"].isin(memb.index) & df["dst"].isin(memb.index)]
    return df["src"].map(memb).to_numpy(), df["dst"].map(memb).to_numpy(), df["neg"].to_numpy()


def pre_cutoff_partition(edges: pd.DataFrame) -> pd.Series:
    early = edges[edges["ts"] < pd.Timestamp(CUTOFF)]
    nodes = sorted(set(early["src"]) | set(early["dst"]))
    return pd.Series(leiden(to_igraph(nodes, undirected_weights(early, "positive"))), index=nodes)


def hostility(cs: np.ndarray, cd: np.ndarray, neg: np.ndarray, comms: list[int], rng: np.random.Generator,
              n_shuffles: int = N_SHUFFLES) -> pd.DataFrame:
    """Long table of (source, target) cells over `comms` with counts, rate, null mean / std, ratio and z."""
    pos = {c: i for i, c in enumerate(comms)}
    keep = np.isin(cs, comms) & np.isin(cd, comms)
    k = len(comms)
    cell = np.array([pos[c] for c in cs[keep]]) * k + np.array([pos[c] for c in cd[keep]])
    neg = np.asarray(neg, dtype=bool)[keep]
    total = np.bincount(cell, minlength=k * k)
    obs = np.bincount(cell, weights=neg, minlength=k * k)
    null = np.array([np.bincount(cell, weights=rng.permutation(neg), minlength=k * k) for _ in range(n_shuffles)])
    with np.errstate(invalid="ignore", divide="ignore"):
        rate, null_rate = obs / total, null / total
        z = (rate - null_rate.mean(axis=0)) / null_rate.std(axis=0, ddof=1)
    expected = neg.mean()
    return pd.DataFrame({
        "src_comm": np.repeat(comms, k), "dst_comm": np.tile(comms, k), "n_links": total, "n_negative": obs.astype(int),
        "neg_rate": rate, "null_mean_rate": null_rate.mean(axis=0), "null_std_rate": null_rate.std(axis=0, ddof=1),
        "ratio_to_expected": rate / expected, "z": z,
    })


def plot_hostility(long: pd.DataFrame, names: dict[int, str]) -> None:
    comms = list(names)
    ticks = [f"C{c} {names[c]}" for c in comms]
    rate = long.pivot(index="src_comm", columns="dst_comm", values="neg_rate").reindex(index=comms, columns=comms)
    ratio = long.pivot(index="src_comm", columns="dst_comm", values="ratio_to_expected").reindex(index=comms, columns=comms)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    im = axes[0].imshow(100 * rate.values, cmap="Reds", vmin=0)
    for i in range(len(comms)):
        for j in range(len(comms)):
            v = 100 * rate.values[i, j]
            if np.isfinite(v):
                axes[0].text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                             color="white" if v > 0.6 * np.nanmax(100 * rate.values) else "black")
    plt.colorbar(im, ax=axes[0], fraction=0.046, label="% negative links")
    n = long.pivot(index="src_comm", columns="dst_comm", values="n_links").reindex(index=comms, columns=comms).values
    r = ratio.values
    lr = np.log2(np.where((r > 0) & (n >= MIN_CELL_LINKS), r, np.nan))  # small or all-positive cells stay grey
    lim = 2  # colors saturate at 4x / 0.25x the expected rate
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#ddd")
    im2 = axes[1].imshow(np.ma.masked_invalid(lr), cmap=cmap, vmin=-lim, vmax=lim)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, label="log2(observed / expected)")
    titles = ("% negative links (source row -> target column)",
              f"Observed / expected rate (grey: < {MIN_CELL_LINKS} links or 0 negative)")
    for ax, title in zip(axes, titles):
        ax.set_xticks(range(len(comms)), [f"C{c}" for c in comms], fontsize=7)
        ax.set_yticks(range(len(comms)), ticks, fontsize=7)
        ax.set_xlabel("target community")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("source community")
    fig.suptitle("Negative links between the largest Leiden communities (core graph, all years)", fontsize=11)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"part2_hostility.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    edges = load_edges(core_only=True)
    members = pd.read_csv(RES / "part2_memberships.csv").set_index("subreddit")
    late = edges[edges["ts"] >= pd.Timestamp(CUTOFF)]
    parts = {p: (members[p], edges) for p in PARTITIONS}
    parts[f"leiden_pre_{CUTOFF[:7]}"] = (pre_cutoff_partition(edges), late)

    rng = np.random.default_rng(SEED)
    rows = []
    for name, (memb, ev) in parts.items():
        for unit in ("link", "pair"):
            cs, cd, neg = units(ev, memb, unit)
            rows.append({"partition": name, "unit": unit, "links_from": CUTOFF if ev is late else "all",
                         "n_communities": memb.nunique()} | polarization(cs, cd, neg, rng))
    pol = pd.DataFrame(rows)
    pol.to_csv(RES / "part2_polarization.csv", index=False)

    labels = json.load(open(RES / "part2_community_labels.json", encoding="utf-8"))
    names = {r["community_id"]: r["label"] for r in labels}
    cs, cd, neg = units(edges, members["leiden"], "link")
    long = hostility(cs, cd, neg, list(names), rng)
    long.insert(1, "src_label", long["src_comm"].map(names))
    long.insert(3, "dst_label", long["dst_comm"].map(names))
    long.to_csv(RES / "part2_hostility_long.csv", index=False)
    wide = (100 * long.pivot(index="src_comm", columns="dst_comm", values="neg_rate")).reindex(index=list(names), columns=list(names))
    wide.index = [f"C{c} {names[c]}" for c in wide.index]
    wide.columns = [f"C{c} {names[c]}" for c in wide.columns]
    wide.round(2).to_csv(RES / "part2_hostility_matrix.csv")
    plot_hostility(long, names)

    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(pol.drop(columns=["links_from"]).to_string(index=False))
        diag = long[long["src_comm"] == long["dst_comm"]]
        off = long[long["src_comm"] != long["dst_comm"]]
        print(f"\nlabeled communities: within-cell neg rate {diag.n_negative.sum() / diag.n_links.sum():.3%}, "
              f"between-cell {off.n_negative.sum() / off.n_links.sum():.3%}")
        print(off[off["n_links"] >= 200].sort_values("neg_rate", ascending=False).head(12)
              [["src_label", "dst_label", "n_links", "neg_rate", "ratio_to_expected", "z"]].to_string(index=False))


if __name__ == "__main__":
    main()
