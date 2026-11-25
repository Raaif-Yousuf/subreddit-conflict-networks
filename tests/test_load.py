"""Smoke tests for the data contract and src/load.py. Needs data/processed/ (run src/preprocess.py first).

Run from the repo root: python -m pytest tests
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import load  # noqa: E402

EDGE_COLUMNS = ["src", "dst", "ts", "sign", "source_file", "post_id"]

pytestmark = pytest.mark.skipif(not (load.PROC / "edges.parquet").exists(), reason="run src/preprocess.py first")


def test_edges_contract():
    e = load.load_edges()
    assert list(e.columns) == EDGE_COLUMNS
    assert set(e["sign"].unique()) <= {-1, 1}
    assert not e.duplicated(["src", "dst", "post_id"]).any()


def test_core_nodes_size():
    assert 5_000 <= len(load.load_core_nodes()) <= 10_000


@pytest.mark.parametrize("quarter", load.list_snapshots() if load.SNAPSHOTS.exists() else [])
def test_each_snapshot_loads(quarter):
    s = load.load_snapshot(quarter)
    assert list(s.columns) == EDGE_COLUMNS
    assert len(s) > 0
    assert (s["ts"].dt.to_period("Q").astype(str) == quarter).all()


def test_snapshots_cover_all_edges():
    total = sum(len(load.load_snapshot(q)) for q in load.list_snapshots())
    assert total == len(load.load_edges())


def test_load_graph_window_matches_date_range():
    Gw = load.load_graph(window="2016Q3")
    Gd = load.load_graph(start="2016-07-01", end="2016-10-01")
    assert Gw.number_of_edges() == Gd.number_of_edges()
    _, _, d = next(iter(Gw.edges(data=True)))
    assert {"pos_w", "neg_w"} <= d.keys()


def test_load_graph_rejects_window_and_range():
    with pytest.raises(ValueError):
        load.load_graph(window="2016Q3", end="2016-10-01")


def test_bad_snapshot_name():
    with pytest.raises(ValueError):
        load.load_snapshot("1999Q1")


def test_to_undirected_signed():
    import networkx as nx

    G = nx.DiGraph()
    G.add_edge("a", "b", pos_w=2, neg_w=0)
    G.add_edge("b", "a", pos_w=0, neg_w=1)  # a-b: pos 2 vs neg 1 -> +
    G.add_edge("a", "c", pos_w=0, neg_w=3)  # a-c: -
    G.add_edge("b", "c", pos_w=1, neg_w=1)  # b-c: tie -> default -1
    U = load.to_undirected_signed(G)
    assert U["a"]["b"] == {"pos_w": 2, "neg_w": 1, "weight": 3, "sign": 1}
    assert U["a"]["c"]["sign"] == -1
    assert U["b"]["c"]["sign"] == -1
    assert load.to_undirected_signed(G, tie=1)["b"]["c"]["sign"] == 1
