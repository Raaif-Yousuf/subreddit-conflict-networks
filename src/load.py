"""Shared loaders for the processed data contract (see PROJECT_PLAN.md, Section 2).

Typical use (run from src/):
    from load import load_edges, load_graph, load_snapshot, to_undirected_signed
    G  = load_graph(end="2016-07-01")          # directed core graph, pos_w / neg_w per edge
    Gq = load_graph(window="2016Q3")           # one quarter
    U  = to_undirected_signed(G)               # undirected, one sign per pair (triads, communities)
"""
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SNAPSHOTS = PROC / "snapshots"


def load_core_nodes() -> list[str]:
    """Analysis subset: subreddits with >= MIN_LINKS total links (see preprocess.py)."""
    return (PROC / "core_nodes.txt").read_text(encoding="utf-8").split()


def _filter_core(df: pd.DataFrame) -> pd.DataFrame:
    core = set(load_core_nodes())
    return df[df["src"].isin(core) & df["dst"].isin(core)]


def load_edges(start=None, end=None, core_only=False) -> pd.DataFrame:
    """Hyperlink rows (src, dst, ts, sign, source_file, post_id), optionally filtered to [start, end)."""
    df = pd.read_parquet(PROC / "edges.parquet")
    if start is not None:
        df = df[df["ts"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["ts"] < pd.Timestamp(end)]
    return _filter_core(df) if core_only else df


def list_snapshots() -> list[str]:
    """Available quarter labels in time order, e.g. ['2013Q4', '2014Q1', ..., '2017Q2']."""
    return sorted(p.stem for p in SNAPSHOTS.glob("*.parquet"))


def load_snapshot(quarter: str, core_only=False) -> pd.DataFrame:
    """Edge rows for one quarter, e.g. load_snapshot('2016Q3'). Same columns as load_edges()."""
    path = SNAPSHOTS / f"{quarter}.parquet"
    if not path.exists():
        raise ValueError(f"no snapshot {quarter!r}; available: {list_snapshots()}")
    df = pd.read_parquet(path)
    return _filter_core(df) if core_only else df


def aggregate(edges: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated links into one row per (src, dst) with pos_w / neg_w counts."""
    g = edges.assign(pos=(edges["sign"] > 0).astype(int), neg=(edges["sign"] < 0).astype(int))
    return g.groupby(["src", "dst"], as_index=False)[["pos", "neg"]].sum().rename(columns={"pos": "pos_w", "neg": "neg_w"})


def load_graph(start=None, end=None, core_only=True, window=None) -> nx.DiGraph:
    """Directed weighted graph with edge attributes pos_w and neg_w (link counts).

    Use either a [start, end) date range or window='YYYYQn' for one quarterly snapshot.
    """
    if window is not None:
        if start is not None or end is not None:
            raise ValueError("pass either window or start/end, not both")
        edges = load_snapshot(window, core_only)
    else:
        edges = load_edges(start, end, core_only)
    agg = aggregate(edges)
    G = nx.DiGraph()
    G.add_edges_from((r.src, r.dst, {"pos_w": r.pos_w, "neg_w": r.neg_w}) for r in agg.itertuples(index=False))
    return G


def to_undirected_signed(G: nx.DiGraph, tie: int = -1) -> nx.Graph:
    """Collapse both directions of each pair into one undirected edge.

    Edge attributes: pos_w and neg_w summed over u->v and v->u, weight = pos_w + neg_w, and
    sign = +1 if pos_w > neg_w, -1 if neg_w > pos_w, `tie` (default -1) when they are equal.
    Ties default to negative because any balanced exchange still contains open conflict.
    """
    U = nx.Graph()
    U.add_nodes_from(G.nodes)
    for u, v, d in G.edges(data=True):
        if U.has_edge(u, v):
            U[u][v]["pos_w"] += d["pos_w"]
            U[u][v]["neg_w"] += d["neg_w"]
        else:
            U.add_edge(u, v, pos_w=d["pos_w"], neg_w=d["neg_w"])
    for _, _, d in U.edges(data=True):
        d["weight"] = d["pos_w"] + d["neg_w"]
        d["sign"] = 1 if d["pos_w"] > d["neg_w"] else -1 if d["neg_w"] > d["pos_w"] else tie
    return U
