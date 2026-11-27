"""Part 2 / issue #15 (RQ1): is negative linking concentrated between communities? Hostility matrix.

Outputs
- results/part2_polarization.csv         one row per (partition, unit, null): within/between shares and rates vs. null
- results/part2_polarization_seeds.csv   the same for Leiden partitions from every seed in SEEDS (seed robustness)
- results/part2_hostility_matrix.csv     % negative links, source community (rows) x target community (columns),
                                         for the labeled Leiden communities (part2_community_labels.json)
- results/part2_hostility_long.csv       the same cells with link counts, ratios, null mean/std, z, Bonferroni p
- figures/part2_hostility.{pdf,png}      heatmaps: % negative, and targeting = cell rate / source row rate (log2)
Run (from src/, after part2_communities.py and part2_labels.py): python part2_polarization.py

Units. "link": every hyperlink counts once (busy pairs weigh more). "pair": each directed pair (src, dst) counts
once and is negative if it has at least one negative link (the Part 1 definition).
Partitions (from part2_memberships.csv unless noted):
- leiden / louvain / spectral: communities of the positive-link graph (main analysis).
- leiden_all_edges: communities of the graph with every link as weight, blind to sign.
- leiden_pre_2016-07: Leiden (seed 42) on the positive graph of links before CUTOFF (subreddits with at least one
  positive link before CUTOFF), evaluated only on links from CUTOFF on between subreddits that it covers
  (out of sample in time).
Why the robustness rows: communities found on positive links put positive links inside communities by
construction, so "negatives fall between communities more than positives" is partly built in for the main rows.
The sign-blind and out-of-sample partitions do not use the signs of the links they are tested on.
Nulls (column "null"), each N_SHUFFLES sign shuffles with seed SEED that keep the partition, the topology and the
number of negative units. z = (observed - null mean) / null std; p_emp is two-sided with +1 smoothing (floor
1 / (N_SHUFFLES + 1)).
- "global": shuffle signs across all units.
- "by_source" (links): shuffle only among links from the same source subreddit, so every subreddit keeps its own
  number of negative links. A few attackers (subredditdrama, drama, ...) send most negative links; this null asks
  whether a subreddit's negative links go between communities more than its own positive links do.
- "by_volume" (pairs): shuffle the negative flag only among pairs with the same number of links (VOLUME_BINS).
  A pair with more links is more likely to contain a negative one and busy pairs sit mostly inside communities,
  so the global pair-level null mixes in link volume (the same confound as Part 1's reciprocity analysis).
Hostility: for source community i and target community j (labeled Leiden communities), neg rate = negative links /
all links from i to j. ratio_to_expected = rate / the negative rate among all links whose two ends are both in
labeled communities (diagonal included). ratio_to_row = rate / i's own negative rate toward all labeled
communities, which shows whom a community targets apart from how negative it is overall (C1 Politics, news &
drama is about twice as negative as average toward everyone). z uses the by_source null (signs shuffled within
each source subreddit), which also removes the row effect; p_bonf = two-sided normal p of z times the number of
cells with a defined z (capped at 1).
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from load import ROOT, aggregate, load_edges
from part2_communities import SEEDS, leiden, to_igraph, undirected_weights

RES = ROOT / "results"
FIG = ROOT / "figures"
SEED = 42
N_SHUFFLES = 100
N_SHUFFLES_SEEDS = 20  # per partition in the seed-robustness table
CUTOFF = "2016-07-01"  # Part 3 train / validation boundary
MIN_CELL_LINKS = 100  # hostility cells with fewer links are greyed out in both heatmaps
VOLUME_BINS = [0, 1, 2, 5, 20, np.inf]  # pair link-count strata: 1, 2, 3-5, 6-20, 21+
PARTITIONS = ["leiden", "louvain", "spectral", "leiden_all_edges"]
NULLS = {"link": ["global", "by_source"], "pair": ["global", "by_volume"]}


def p_emp(obs: float, null: np.ndarray) -> float:
    dev = np.abs(null - null.mean()) >= abs(obs - null.mean())
    return (dev.sum() + 1) / (len(null) + 1)


def shuffle_within(neg: np.ndarray, strata: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    """Permutation of `neg` that only moves values between units of the same stratum (all units if None)."""
    if strata is None:
        return rng.permutation(neg)
    # integer strata + uniform [0, 1) noise sorts by stratum first and randomly within it
    order = np.argsort(strata, kind="stable")
    shuffled = np.empty_like(neg)
    shuffled[order] = neg[np.argsort(strata + rng.random(len(neg)))]
    return shuffled


def polarization(cs: np.ndarray, cd: np.ndarray, neg: np.ndarray, rng: np.random.Generator,
                 n_shuffles: int = N_SHUFFLES, strata: np.ndarray | None = None) -> dict:
    """Within/between statistics for units with source / target community cs, cd and a negative flag.

    The null shuffles `neg` within `strata` (globally if None).
    """
    within = cs == cd
    neg = np.asarray(neg, dtype=bool)
    obs = within[neg].mean()
    null = np.array([within[shuffle_within(neg, strata, rng)].mean() for _ in range(n_shuffles)])
    rate_w, rate_b = neg[within].mean(), neg[~within].mean()
    return {
        "n_units": len(neg), "n_negative": int(neg.sum()), "share_within_all": within.mean(),
        "share_pos_within": within[~neg].mean(), "share_neg_within": obs,
        "null_share_neg_within": null.mean(), "null_std": null.std(ddof=1),
        "z": (obs - null.mean()) / null.std(ddof=1), "p_emp": p_emp(obs, null),
        "neg_rate_within": rate_w, "neg_rate_between": rate_b, "between_within_ratio": rate_b / rate_w,
    }


def units(edges: pd.DataFrame, memb: pd.Series, unit: str) -> pd.DataFrame:
    """One row per link or per directed pair between covered nodes: cs, cd (communities), neg, src, n_links."""
    if unit == "pair":
        df = aggregate(edges).assign(neg=lambda d: d["neg_w"] > 0, n_links=lambda d: d["pos_w"] + d["neg_w"])
    else:
        df = edges.assign(neg=edges["sign"] < 0, n_links=1)
    df = df[df["src"].isin(memb.index) & df["dst"].isin(memb.index)]
    return pd.DataFrame({"cs": df["src"].map(memb).to_numpy(), "cd": df["dst"].map(memb).to_numpy(),
                         "neg": df["neg"].to_numpy(), "src": df["src"].to_numpy(), "n_links": df["n_links"].to_numpy()})


def null_strata(u: pd.DataFrame, null: str) -> np.ndarray | None:
    """Stratum per unit for a null: None (global), source subreddit, or pair link-count bin."""
    if null == "global":
        return None
    if null == "by_source":
        return pd.factorize(u["src"])[0]
    if null == "by_volume":
        return pd.cut(u["n_links"], VOLUME_BINS, labels=False).to_numpy()
    raise ValueError(f"unknown null {null!r}")


def polarization_rows(memb: pd.Series, ev: pd.DataFrame, rng: np.random.Generator,
                      n_shuffles: int = N_SHUFFLES) -> list[dict]:
    """polarization() for every (unit, null) in NULLS."""
    rows = []
    for unit, nulls in NULLS.items():
        u = units(ev, memb, unit)
        for null in nulls:
            rows.append({"unit": unit, "null": null} | polarization(
                u["cs"].to_numpy(), u["cd"].to_numpy(), u["neg"].to_numpy(), rng, n_shuffles, null_strata(u, null)))
    return rows


def pre_cutoff_partition(edges: pd.DataFrame) -> pd.Series:
    """Leiden on the positive graph of links before CUTOFF; covers subreddits with a positive link before CUTOFF."""
    und = undirected_weights(edges[edges["ts"] < pd.Timestamp(CUTOFF)], "positive")
    nodes = sorted(set(und["a"]) | set(und["b"]))
    return pd.Series(leiden(to_igraph(nodes, und)), index=nodes)


def hostility(cs: np.ndarray, cd: np.ndarray, neg: np.ndarray, comms: list[int], rng: np.random.Generator,
              n_shuffles: int = N_SHUFFLES, strata: np.ndarray | None = None) -> pd.DataFrame:
    """Long table of (source, target) cells over `comms`: counts, rate, ratios, null mean / std, z, p_bonf.

    Uses only links whose two ends are in `comms`; the null shuffles signs among them within `strata`.
    """
    pos = {c: i for i, c in enumerate(comms)}
    keep = np.isin(cs, comms) & np.isin(cd, comms)
    k = len(comms)
    row = np.array([pos[c] for c in cs[keep]], dtype=np.int64)
    cell = row * k + np.array([pos[c] for c in cd[keep]], dtype=np.int64)
    neg = np.asarray(neg, dtype=bool)[keep]
    st = None if strata is None else np.asarray(strata)[keep]
    total = np.bincount(cell, minlength=k * k)
    obs = np.bincount(cell, weights=neg, minlength=k * k)
    null = np.array([np.bincount(cell, weights=shuffle_within(neg, st, rng), minlength=k * k)
                     for _ in range(n_shuffles)])
    with np.errstate(invalid="ignore", divide="ignore"):
        row_rate = np.bincount(row, weights=neg, minlength=k) / np.bincount(row, minlength=k)
        rate, null_rate = obs / total, null / total
        sd = null_rate.std(axis=0, ddof=1)
        z = np.where(sd > 0, (rate - null_rate.mean(axis=0)) / sd, np.nan)
        ratio_row = rate / np.repeat(row_rate, k)
    n_tests = int(np.isfinite(z).sum())
    return pd.DataFrame({
        "src_comm": np.repeat(comms, k), "dst_comm": np.tile(comms, k), "n_links": total,
        "n_negative": obs.astype(int), "neg_rate": rate, "ratio_to_expected": rate / neg.mean(),
        "ratio_to_row": ratio_row, "null_mean_rate": null_rate.mean(axis=0), "null_std_rate": sd, "z": z,
        "p_bonf": np.minimum(1, 2 * norm.sf(np.abs(z)) * n_tests),
    })


def plot_hostility(long: pd.DataFrame, names: dict[int, str]) -> None:
    comms = list(names)
    ticks = [f"C{c} {names[c]}" for c in comms]

    def grid(col):
        return long.pivot(index="src_comm", columns="dst_comm", values=col).reindex(index=comms, columns=comms).values

    n, rate, ratio = grid("n_links"), 100 * grid("neg_rate"), grid("ratio_to_row")
    small = n < MIN_CELL_LINKS
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    reds = plt.get_cmap("Reds").copy()
    reds.set_bad("#ddd")
    im = axes[0].imshow(np.ma.masked_where(small, rate), cmap=reds, vmin=0)
    vmax = np.nanmax(rate[~small])
    for i in range(len(comms)):
        for j in range(len(comms)):
            if not small[i, j]:
                axes[0].text(j, i, f"{rate[i, j]:.0f}", ha="center", va="center", fontsize=6,
                             color="white" if rate[i, j] > 0.6 * vmax else "black")
    plt.colorbar(im, ax=axes[0], fraction=0.046, label="% negative links")
    lr = np.log2(np.where((ratio > 0) & ~small, ratio, np.nan))
    rdbu = plt.get_cmap("RdBu_r").copy()
    rdbu.set_bad("#ddd")
    im2 = axes[1].imshow(np.ma.masked_invalid(lr), cmap=rdbu, vmin=-2, vmax=2)  # saturates at 4x / 0.25x
    plt.colorbar(im2, ax=axes[1], fraction=0.046, label="log2(cell rate / source row rate)")
    titles = ("% negative links", "Targeting: cell rate / source community's overall rate")
    for ax, title in zip(axes, titles):
        ax.set_xticks(range(len(comms)), [f"C{c}" for c in comms], fontsize=7)
        ax.set_yticks(range(len(comms)), ticks, fontsize=7)
        ax.set_xlabel("target community")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("source community")
    fig.suptitle("Negative links between the largest Leiden communities (core graph, all years; "
                 f"grey: < {MIN_CELL_LINKS} links or 0 negative)", fontsize=10)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"part2_hostility.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    edges = load_edges(core_only=True)
    members = pd.read_csv(RES / "part2_memberships.csv").set_index("subreddit")
    late = edges[edges["ts"] >= pd.Timestamp(CUTOFF)]
    parts = {p: (members[p], edges, "all") for p in PARTITIONS}
    parts[f"leiden_pre_{CUTOFF[:7]}"] = (pre_cutoff_partition(edges), late, CUTOFF)

    rng = np.random.default_rng(SEED)
    rows = []
    for name, (memb, ev, frm) in parts.items():
        rows += [{"partition": name, "links_from": frm, "n_communities": memb.nunique()} | r
                 for r in polarization_rows(memb, ev, rng)]
    pol = pd.DataFrame(rows)
    pol.to_csv(RES / "part2_polarization.csv", index=False)

    nodes = list(members.index)
    g_pos = to_igraph(nodes, undirected_weights(edges, "positive"))
    seed_rows = []
    for s in SEEDS:
        memb = pd.Series(leiden(g_pos, seed=s), index=nodes)
        seed_rows += [{"seed": s, "n_communities": memb.nunique()} | r
                      for r in polarization_rows(memb, edges, rng, N_SHUFFLES_SEEDS)]
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(RES / "part2_polarization_seeds.csv", index=False)

    labels = json.load(open(RES / "part2_community_labels.json", encoding="utf-8"))
    names = {r["community_id"]: r["label"] for r in labels}
    u = units(edges, members["leiden"], "link")
    long = hostility(u["cs"].to_numpy(), u["cd"].to_numpy(), u["neg"].to_numpy(), list(names), rng,
                     strata=null_strata(u, "by_source"))
    long.insert(1, "src_label", long["src_comm"].map(names))
    long.insert(3, "dst_label", long["dst_comm"].map(names))
    long.to_csv(RES / "part2_hostility_long.csv", index=False)
    wide = (100 * long.pivot(index="src_comm", columns="dst_comm", values="neg_rate")).reindex(
        index=list(names), columns=list(names))
    wide.index = [f"C{c} {names[c]}" for c in wide.index]
    wide.columns = [f"C{c} {names[c]}" for c in wide.columns]
    wide.round(2).to_csv(RES / "part2_hostility_matrix.csv")
    plot_hostility(long, names)

    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(pol.drop(columns=["links_from", "share_within_all"]).to_string(index=False))
        print(seeds.groupby(["unit", "null"])[["z", "between_within_ratio"]].agg(["min", "max"]).to_string())
        diag = long[long["src_comm"] == long["dst_comm"]]
        off = long[long["src_comm"] != long["dst_comm"]]
        print(f"\nlabeled communities: within-cell neg rate {diag.n_negative.sum() / diag.n_links.sum():.3%}, "
              f"between-cell {off.n_negative.sum() / off.n_links.sum():.3%}; "
              f"{(long.p_bonf < 0.05).sum()} of {long.z.notna().sum()} cells significant (Bonferroni)")
        print(off[off["n_links"] >= 200].sort_values("ratio_to_row", ascending=False).head(12)
              [["src_label", "dst_label", "n_links", "neg_rate", "ratio_to_row", "z", "p_bonf"]].to_string(index=False))


if __name__ == "__main__":
    main()
