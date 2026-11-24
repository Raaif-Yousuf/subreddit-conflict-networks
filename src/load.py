"""Shared loaders for the processed data contract (see PROJECT_PLAN.md, Section 2)."""
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def load_edges(start=None, end=None, core_only=False) -> pd.DataFrame:
    """Hyperlink rows (src, dst, ts, sign, source_file, post_id), optionally filtered to [start, end)."""
    df = pd.read_parquet(PROC / "edges.parquet")
    if start is not None:
        df = df[df["ts"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["ts"] < pd.Timestamp(end)]
    if core_only:
        core = set(load_core_nodes())
        df = df[df["src"].isin(core) & df["dst"].isin(core)]
    return df


def load_core_nodes() -> list[str]:
    return (PROC / "core_nodes.txt").read_text(encoding="utf-8").split()


def aggregate(edges: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated links into one row per (src, dst) with pos_w / neg_w counts."""
    g = edges.assign(pos=(edges["sign"] > 0).astype(int), neg=(edges["sign"] < 0).astype(int))
    return g.groupby(["src", "dst"], as_index=False)[["pos", "neg"]].sum().rename(columns={"pos": "pos_w", "neg": "neg_w"})


def load_graph(start=None, end=None, core_only=True) -> nx.DiGraph:
    """Directed weighted graph with edge attributes pos_w and neg_w."""
    agg = aggregate(load_edges(start, end, core_only))
    G = nx.DiGraph()
    G.add_edges_from((r.src, r.dst, {"pos_w": r.pos_w, "neg_w": r.neg_w}) for r in agg.itertuples(index=False))
    return G
