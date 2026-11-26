"""Part 1 / issue #10 (RQ2): are negative links reciprocated?

Output: results/part1_reciprocity.csv (one row per measure / window / stratum)
Run (from src/): python part1_reciprocity.py

Unit: an ordered pair (A, B) of core subreddits with at least one A->B hyperlink, whole period.
A pair is "negative" if A->B has >= 1 negative link, "positive-only" if it has none.

Static measures (no time order)
- neg_given_neg:  P(B->A negative | A->B negative)          (the conditional of interest)
- neg_baseline:   P(B->A negative)  over all linked pairs   (includes the negative pairs themselves)
- neg_given_pos:  P(B->A negative | A->B positive-only)     (clean contrast)
- any_given_neg / any_baseline: P(B->A any link | ...)      (is there a reverse link at all?)
- neg_given_neg_recip / neg_baseline_recip: the first two restricted to pairs with a reverse link.
- strat_neg_given_neg / strat_neg_given_pos: P(B->A negative | ...) within strata of the number of
  B->A links (`stratum` column), so busy and quiet pairs are compared separately.
lift = rate / baseline (the baseline is named in `baseline_measure`, same window and stratum).

Time-ordered retaliation (first strike)
- Treated: t0 = first A->B negative link; kept only if B had no negative link to A at or before t0
  (A struck first). Outcome: a B->A negative link with ts in (t0, t0 + W].
- Control: pairs whose first A->B link is positive, t0 = that link, same first-strike rule and outcome.
  Defined at t0 only (a control pair may turn negative later), so no future information is used.
- W in {30 d, 90 d, 365 d, until end of data}. For a finite W only pairs with t0 <= data_end - W
  are kept, so no pair is cut off by the end of the data. The unbounded row is right-censored.

Null model: "negative" pairs are busier than average, and busy pairs link back more, so lift > 1 can
arise from link volume alone. We shuffle the sign column across all core links (N_NULL permutations,
seed SEED), which keeps every pair's link count and timing and the overall negative share but breaks
any link between A->B and B->A signs. null_lift_mean / null_lift_std are the lift under shuffles,
z_null = (lift - null_lift_mean) / null_lift_std, and lift_vs_null = lift / null_lift_mean.
"""
import numpy as np
import pandas as pd

from load import ROOT, aggregate, load_edges

RES = ROOT / "results"
WINDOWS = [30, 90, 365, None]  # days; None = until the end of the data
STRATA = [(1, 1), (2, 2), (3, 5), (6, 20), (21, np.inf)]  # number of B->A links
N_NULL = 20
SEED = 42
KEY = ["measure", "window_days", "stratum"]


def _row(name, outcome, base, window=np.nan, stratum=""):
    n, k = int(outcome.size), int(outcome.sum())
    return {"measure": name, "window_days": window, "stratum": stratum, "n_pairs": n, "count": k,
            "rate": k / n if n else np.nan, "baseline_measure": base}


def static_reciprocity(edges: pd.DataFrame) -> pd.DataFrame:
    """Pair-level P(B->A ... | A->B ...) measures; see module docstring."""
    agg = aggregate(edges)
    rev = agg.rename(columns={"src": "dst", "dst": "src", "pos_w": "rev_pos", "neg_w": "rev_neg"})
    p = agg.merge(rev, on=["src", "dst"], how="left").fillna({"rev_pos": 0, "rev_neg": 0})
    neg, rev_n, rev_neg = p["neg_w"] > 0, p["rev_pos"] + p["rev_neg"], p["rev_neg"] > 0
    rev_any = rev_n > 0

    rows = [
        _row("neg_given_neg", rev_neg[neg], "neg_baseline"),
        _row("neg_baseline", rev_neg, None),
        _row("neg_given_pos", rev_neg[~neg], "neg_baseline"),
        _row("any_given_neg", rev_any[neg], "any_baseline"),
        _row("any_baseline", rev_any, None),
        _row("neg_given_neg_recip", rev_neg[neg & rev_any], "neg_baseline_recip"),
        _row("neg_baseline_recip", rev_neg[rev_any], None),
    ]
    for lo, hi in STRATA:
        s = (rev_n >= lo) & (rev_n <= hi)
        label = f"{lo}" if lo == hi else f"{lo}+" if np.isinf(hi) else f"{lo}-{hi}"
        rows.append(_row("strat_neg_given_neg", rev_neg[neg & s], "strat_neg_given_pos", stratum=label))
        rows.append(_row("strat_neg_given_pos", rev_neg[~neg & s], None, stratum=label))
    return _add_lift(pd.DataFrame(rows))


def _first_strike_outcomes(starts: pd.DataFrame, neg_edges: pd.DataFrame) -> pd.DataFrame:
    """starts: (src=A, dst=B, t0). Drops pairs where B->A was negative at or before t0 and adds
    `first_resp`: time of the first B->A negative link after t0 (NaT if none)."""
    back = neg_edges[["src", "dst", "ts"]].rename(columns={"src": "dst", "dst": "src", "ts": "back_ts"})
    m = starts.merge(back, on=["src", "dst"], how="left")
    prior = (m["back_ts"] <= m["t0"]).groupby([m["src"], m["dst"]]).any()
    after = m[m["back_ts"] > m["t0"]].groupby(["src", "dst"])["back_ts"].min()
    out = starts.set_index(["src", "dst"])
    out = out[~prior.reindex(out.index, fill_value=False)]
    out["first_resp"] = after.reindex(out.index)
    return out.reset_index()


def temporal_retaliation(edges: pd.DataFrame, windows=WINDOWS) -> pd.DataFrame:
    """First-strike retaliation rates vs. the positive-first-link control; see module docstring."""
    edges = edges[["src", "dst", "ts", "sign"]]
    neg_edges = edges[edges["sign"] < 0]
    data_end = edges["ts"].max()
    treated = neg_edges.groupby(["src", "dst"], as_index=False)["ts"].min().rename(columns={"ts": "t0"})
    # first link per pair; on a same-timestamp tie the negative link wins, so the pair is not a control
    first = edges.sort_values(["ts", "sign"]).groupby(["src", "dst"], as_index=False).first()
    control = first.loc[first["sign"] > 0, ["src", "dst", "ts"]].rename(columns={"ts": "t0"})

    groups = {"retaliation": _first_strike_outcomes(treated, neg_edges),
              "retaliation_baseline": _first_strike_outcomes(control, neg_edges)}
    rows = []
    for w in windows:
        for name, g in groups.items():
            if w is None:
                hit = g["first_resp"].notna()
            else:
                g = g[g["t0"] <= data_end - pd.Timedelta(days=w)]
                hit = (g["first_resp"] - g["t0"]) <= pd.Timedelta(days=w)
            base = "retaliation_baseline" if name == "retaliation" else None
            rows.append(_row(name, hit, base, window=np.nan if w is None else w))
    return _add_lift(pd.DataFrame(rows))


def _add_lift(df: pd.DataFrame) -> pd.DataFrame:
    key = list(zip(df["measure"], df["window_days"].fillna(-1), df["stratum"]))
    lookup = dict(zip(key, df["rate"]))
    df["baseline"] = [lookup.get((b, w, s), np.nan) if b else np.nan
                      for b, (_, w, s) in zip(df["baseline_measure"], key)]
    df["lift"] = df["rate"] / df["baseline"]
    return df


def all_measures(edges: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([static_reciprocity(edges), temporal_retaliation(edges)], ignore_index=True)


def add_null(obs: pd.DataFrame, edges: pd.DataFrame, n=N_NULL, seed=SEED) -> pd.DataFrame:
    """Attach the lift distribution under link-sign shuffles (see module docstring)."""
    rng = np.random.default_rng(seed)
    lifts = []
    for _ in range(n):
        shuffled = edges.assign(sign=rng.permutation(edges["sign"].to_numpy()))
        lifts.append(all_measures(shuffled).set_index(KEY)["lift"])
    null = pd.concat(lifts, axis=1)
    idx = pd.MultiIndex.from_frame(obs[KEY])
    obs["null_lift_mean"] = null.mean(axis=1).reindex(idx).to_numpy()
    obs["null_lift_std"] = null.std(axis=1, ddof=1).reindex(idx).to_numpy()
    obs["z_null"] = (obs["lift"] - obs["null_lift_mean"]) / obs["null_lift_std"]
    obs["lift_vs_null"] = obs["lift"] / obs["null_lift_mean"]
    return obs


def main() -> None:
    RES.mkdir(exist_ok=True)
    edges = load_edges(core_only=True)[["src", "dst", "ts", "sign"]]
    out = add_null(all_measures(edges), edges)
    out.to_csv(RES / "part1_reciprocity.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4g}".format):
        print(out.drop(columns=["baseline_measure"]))


if __name__ == "__main__":
    main()
