"""Unit tests for Part 2 community / polarization functions on small hand-built inputs (no data/ needed).

Run from the repo root: python -m pytest tests
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import part2_communities as pc  # noqa: E402
import part2_labels as pl  # noqa: E402
import part2_polarization as pp  # noqa: E402


def _edges(rows):
    return pd.DataFrame(rows, columns=["src", "dst", "sign"]).assign(ts=pd.Timestamp("2015-01-01"))


def _two_cliques():
    """Two 5-cliques of positive links joined by one positive bridge; nodes a0..a4 and b0..b4."""
    rows = []
    for g in "ab":
        rows += [(f"{g}{i}", f"{g}{j}", 1) for i in range(5) for j in range(5) if i < j]
    rows.append(("a0", "b0", 1))
    return _edges(rows)


def test_undirected_weights_positive_vs_all():
    e = _edges([("a", "b", 1), ("b", "a", 1), ("a", "b", -1), ("c", "d", -1), ("c", "c", 1)])
    pos = pc.undirected_weights(e, "positive").set_index(["a", "b"])["w"]
    allw = pc.undirected_weights(e, "all").set_index(["a", "b"])["w"]
    assert pos.to_dict() == {("a", "b"): 2}            # negative-only pair and self-loop dropped
    assert allw.to_dict() == {("a", "b"): 3, ("c", "d"): 1}


def test_relabel_by_size():
    assert pc.relabel_by_size([5, 5, 2, 9, 9, 9]).tolist() == [1, 1, 2, 0, 0, 0]
    assert pc.relabel_by_size([3, 1]).tolist() == [1, 0]  # size tie -> smaller original id first


@pytest.mark.parametrize("method", ["leiden", "louvain", "spectral"])
def test_methods_recover_two_cliques(method):
    e = _two_cliques()
    nodes = sorted(set(e["src"]) | set(e["dst"]))
    g = pc.to_igraph(nodes, pc.undirected_weights(e))
    memb = pc.spectral(g, 2) if method == "spectral" else getattr(pc, method)(g)
    m = dict(zip(nodes, memb))
    assert len(set(memb)) == 2
    assert len({m[f"a{i}"] for i in range(5)}) == 1 and len({m[f"b{i}"] for i in range(5)}) == 1


def test_polarization_counts():
    cs = np.array([0, 0, 0, 0, 1, 1])
    cd = np.array([0, 0, 1, 1, 1, 0])
    neg = np.array([False, True, True, True, False, False])
    r = pp.polarization(cs, cd, neg, np.random.default_rng(0), n_shuffles=20)
    # within = links 0, 1, 4; negatives = 1, 2, 3 -> 1 of 3 negatives within
    assert r["share_neg_within"] == pytest.approx(1 / 3)
    assert r["share_pos_within"] == pytest.approx(2 / 3)
    assert r["neg_rate_within"] == pytest.approx(1 / 3) and r["neg_rate_between"] == pytest.approx(2 / 3)
    assert r["between_within_ratio"] == pytest.approx(2)
    assert 1 / 21 <= r["p_emp"] <= 1


def test_units_link_vs_pair():
    e = _edges([("a", "b", -1), ("a", "b", 1), ("a", "b", 1), ("b", "c", 1), ("x", "a", -1)])
    memb = pd.Series({"a": 0, "b": 0, "c": 1})          # x is not covered
    cs, cd, neg = pp.units(e, memb, "link")
    assert len(neg) == 4 and neg.sum() == 1
    cs, cd, neg = pp.units(e, memb, "pair")
    assert sorted(zip(cs, cd, neg)) == [(0, 0, True), (0, 1, False)]


def test_hostility_cells():
    cs = np.array([0, 0, 0, 1, 1, 2])
    cd = np.array([1, 1, 0, 0, 0, 0])
    neg = np.array([True, False, False, True, True, True])
    long = pp.hostility(cs, cd, neg, [0, 1], np.random.default_rng(0), n_shuffles=10).set_index(["src_comm", "dst_comm"])
    assert long.loc[(0, 1), "n_links"] == 2 and long.loc[(0, 1), "neg_rate"] == 0.5
    assert long.loc[(1, 0), "neg_rate"] == 1.0
    assert long.loc[(0, 0), "n_links"] == 1 and long.loc[(1, 1), "n_links"] == 0
    # expected rate: 3 negatives among the 5 links between communities 0 and 1 (link from 2 excluded)
    assert long.loc[(1, 0), "ratio_to_expected"] == pytest.approx(1 / 0.6)


def test_label_communities_anchors():
    memb = pd.Series({"g1": 0, "g2": 0, "g3": 0, "p1": 1, "p2": 1, "s1": 2})
    act = pd.Series({"g1": 5, "g2": 9, "g3": 1, "p1": 3, "p2": 4, "s1": 1})
    internal = pd.Series({"g1": 9, "g2": 1, "p1": 2})
    out = pl.label_communities(memb, act, internal, {"g1": "Games", "p2": "Politics"}, n_label=2, top=2)
    assert [r["label"] for r in out] == ["Games", "Politics"]
    assert out[0]["top_by_activity"] == ["g2", "g1"] and out[0]["top_by_internal"] == ["g1", "g2"]
    assert out[0]["size"] == 3
    with pytest.raises(ValueError, match="both in community"):
        pl.label_communities(memb, act, internal, {"g1": "A", "g2": "B"}, n_label=2)
    with pytest.raises(ValueError, match="not among"):
        pl.label_communities(memb, act, internal, {"s1": "Small"}, n_label=2)
