"""Unit tests for Part 1 analysis functions on small hand-built inputs (no data/ needed).

Run from the repo root: python -m pytest tests
"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import part1_reciprocity as rec  # noqa: E402
import part1_triads as tri  # noqa: E402


def _edges(rows):
    return pd.DataFrame(rows, columns=["src", "dst", "ts", "sign"]).assign(ts=lambda d: pd.to_datetime(d["ts"]))


def test_static_reciprocity():
    # a->b neg, b->a neg (reciprocated); c->d neg, no reverse; e->f pos, f->e neg
    e = _edges([("a", "b", "2015-01-01", -1), ("b", "a", "2015-02-01", -1),
                ("c", "d", "2015-01-01", -1),
                ("e", "f", "2015-01-01", 1), ("f", "e", "2015-03-01", -1)])
    r = rec.static_reciprocity(e).set_index("measure")
    # negative pairs: a->b, b->a, c->d, f->e; reverse negative for a->b, b->a only
    assert r.loc["neg_given_neg", "n_pairs"] == 4 and r.loc["neg_given_neg", "count"] == 2
    # all 5 pairs; reverse negative for a->b, b->a, e->f
    assert r.loc["neg_baseline", "count"] == 3 and r.loc["neg_baseline", "n_pairs"] == 5
    assert r.loc["neg_given_pos", "count"] == 1 and r.loc["neg_given_pos", "n_pairs"] == 1
    assert np.isclose(r.loc["neg_given_neg", "lift"], (2 / 4) / (3 / 5))


def test_temporal_retaliation_first_strike():
    e = _edges([
        ("a", "b", "2015-01-01", -1), ("b", "a", "2015-01-11", -1),  # a strikes, b retaliates in 10 d
        ("c", "d", "2015-01-01", -1), ("d", "c", "2015-06-01", -1),  # d answers after ~150 d
        ("x", "y", "2015-01-01", 1), ("y", "x", "2015-01-05", -1),   # positive-only pair answered negatively
        ("z", "z2", "2016-12-31", 1),                                # sets data end
    ])
    r = rec.temporal_retaliation(e, windows=[30, None])
    t30 = r[(r["measure"] == "retaliation") & (r["window_days"] == 30)].iloc[0]
    # first strikers: a->b, c->d, y->x (b->a and d->c had earlier negatives from the other side)
    assert t30["n_pairs"] == 3 and t30["count"] == 1
    tinf = r[(r["measure"] == "retaliation") & r["window_days"].isna()].iloc[0]
    assert tinf["count"] == 2
    base = r[(r["measure"] == "retaliation_baseline") & (r["window_days"] == 30)].iloc[0]
    # control pairs: x->y and z->z2 (y->x is negative, so not a control); z->z2 too late for 30 d
    assert base["n_pairs"] == 1 and base["count"] == 1


def test_triangle_census():
    U = nx.Graph()
    # triangle abc: all positive; triangle acd: a-c +, c-d -, a-d -  (+--)
    U.add_edge("a", "b", s=1); U.add_edge("b", "c", s=1); U.add_edge("a", "c", s=1)
    U.add_edge("c", "d", s=-1); U.add_edge("a", "d", s=-1)
    t, edges = tri.triangle_edges(U)
    neg = np.array([U[u][v]["s"] < 0 for u, v in edges])
    assert len(t) == 2
    assert tri.census(neg, t).tolist() == [1, 0, 1, 0]


def test_null_keeps_sign_count_and_seed():
    rng = np.random.default_rng(0)
    neg = rng.random(50) < 0.2
    t = rng.integers(0, 50, size=(200, 3))
    a = tri.null_censuses(neg, t, n=5, seed=42)
    b = tri.null_censuses(neg, t, n=5, seed=42)
    assert (a == b).all() and (a.sum(axis=1) == 200).all()
    # with one triangle per edge (edge i alone in triangle i, i, i), the --- count equals the number
    # of negative edges, so every shuffle must keep it
    diag = np.repeat(np.arange(50)[:, None], 3, axis=1)
    assert (tri.null_censuses(neg, diag, n=5)[:, 3] == neg.sum()).all()


def test_compare_hand_computed():
    obs = np.array([6, 2, 1, 1])                        # 10 triangles
    null = np.array([[7, 2, 1, 0], [5, 4, 1, 0]])       # null shares +++: 0.7, 0.5
    df = tri.compare(obs, null, "negative").set_index("type")
    assert np.isclose(df.loc["+++", "null_mean_share"], 0.6)
    assert np.isclose(df.loc["+++", "null_std_share"], np.std([0.7, 0.5], ddof=1))
    assert np.isclose(df.loc["+++", "z"], 0.0)
    assert df.loc["balanced", "observed"] == 7 and df.loc["balanced_weak", "observed"] == 8
    assert np.isclose(df.loc["---", "p_emp"], 1 / 3)   # neither shuffle reaches 1 --- triangle
    assert np.isclose(df.loc["+++", "p_emp"], 1.0)


def test_add_lift_matches_window_and_stratum():
    df = pd.DataFrame([
        {"measure": "t", "window_days": 30, "stratum": "", "rate": 0.2, "baseline_measure": "b"},
        {"measure": "b", "window_days": 30, "stratum": "", "rate": 0.1, "baseline_measure": None},
        {"measure": "t", "window_days": np.nan, "stratum": "", "rate": 0.3, "baseline_measure": "b"},
        {"measure": "b", "window_days": np.nan, "stratum": "", "rate": 0.6, "baseline_measure": None},
        {"measure": "t", "window_days": np.nan, "stratum": "2", "rate": 0.4, "baseline_measure": "b"},
        {"measure": "b", "window_days": np.nan, "stratum": "2", "rate": 0.2, "baseline_measure": None},
    ])
    lift = rec._add_lift(df)["lift"].tolist()
    assert np.allclose([lift[0], lift[2], lift[4]], [2.0, 0.5, 2.0]) and np.isnan(lift[1])


def test_same_timestamp_strikes_are_not_first_strikes():
    e = _edges([("a", "b", "2015-01-01", -1), ("b", "a", "2015-01-01", -1),
                ("z", "z2", "2016-12-31", 1)])
    r = rec.temporal_retaliation(e, windows=[None]).set_index("measure")
    # each side's attack is matched by one at the same time from the other side, so neither is kept
    assert r.loc["retaliation", "n_pairs"] == 0
    # unbounded control row keeps late pairs (z->z2), with no reply
    assert r.loc["retaliation_baseline", "n_pairs"] == 1 and r.loc["retaliation_baseline", "count"] == 0


def test_static_strata():
    e = _edges([("a", "b", "2015-01-01", -1), ("b", "a", "2015-02-01", -1),
                ("c", "d", "2015-01-01", 1), ("d", "c", "2015-02-01", 1)])
    r = rec.static_reciprocity(e)
    s1 = r[(r["measure"] == "strat_neg_given_neg") & (r["stratum"] == "1")].iloc[0]
    assert s1["n_pairs"] == 2 and s1["count"] == 2  # a->b and b->a
    p1 = r[(r["measure"] == "strat_neg_given_pos") & (r["stratum"] == "1")].iloc[0]
    assert p1["n_pairs"] == 2 and p1["count"] == 0  # c->d and d->c, both answered positively
