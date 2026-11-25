"""Build the shared data contract from the raw SNAP Reddit Hyperlink TSVs.

Inputs : data/raw/soc-redditHyperlinks-{body,title}.tsv
Outputs: data/processed/edges.parquet      one row per hyperlink
         data/processed/nodes.parquet      per-subreddit activity stats
         data/processed/core_nodes.txt     analysis subset (>= MIN_LINKS total links)
         data/processed/snapshots/YYYYQn.parquet  quarterly edge lists

Cleaning: body + title files are merged, timestamps parsed, self-links dropped, and a
(src, dst, post_id) triple that appears in both files is kept once.

Core threshold: MIN_LINKS = 20 total links (sent + received) gives 7,859 core subreddits
(7,854 of them link to another core node, so the core graph has 7,854 nodes),
inside the 5-10k target from PROJECT_PLAN.md so Leiden and GNNs stay laptop-sized.

Run: python src/download.py && python src/preprocess.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
MIN_LINKS = 20  # a subreddit is "core" if it sends + receives at least this many links
EDGE_COLUMNS = ["src", "dst", "ts", "sign", "source_file", "post_id"]
NODE_COLUMNS = ["subreddit", "n_out", "n_in", "n_neg_out", "n_neg_in", "first_ts", "last_ts"]


def load_raw() -> pd.DataFrame:
    frames = []
    for kind in ("body", "title"):
        df = pd.read_csv(
            RAW / f"soc-redditHyperlinks-{kind}.tsv",
            sep="\t",
            usecols=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "POST_ID", "TIMESTAMP", "LINK_SENTIMENT"],
        )
        df["source_file"] = kind
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(
        columns={
            "SOURCE_SUBREDDIT": "src",
            "TARGET_SUBREDDIT": "dst",
            "POST_ID": "post_id",
            "TIMESTAMP": "ts",
            "LINK_SENTIMENT": "sign",
        }
    )
    df["ts"] = pd.to_datetime(df["ts"])
    df["sign"] = df["sign"].astype("int8")
    df = df[df["src"] != df["dst"]]  # drop self-links
    df = df.drop_duplicates(subset=["src", "dst", "post_id"])  # same post in body+title counts once
    return df.sort_values("ts").reset_index(drop=True)[EDGE_COLUMNS]


def node_stats(edges: pd.DataFrame) -> pd.DataFrame:
    neg = edges[edges["sign"] < 0]
    out = pd.DataFrame(
        {
            "n_out": edges.groupby("src").size(),
            "n_in": edges.groupby("dst").size(),
            "n_neg_out": neg.groupby("src").size(),
            "n_neg_in": neg.groupby("dst").size(),
        }
    ).fillna(0).astype(int)
    first = pd.concat([edges[["src", "ts"]].rename(columns={"src": "n"}), edges[["dst", "ts"]].rename(columns={"dst": "n"})])
    out["first_ts"] = first.groupby("n")["ts"].min()
    out["last_ts"] = first.groupby("n")["ts"].max()
    out.index.name = "subreddit"
    return out.reset_index()[NODE_COLUMNS]


def validate(edges: pd.DataFrame, nodes: pd.DataFrame) -> None:
    """Fail loudly if the outputs drift from the data contract other parts rely on."""
    assert list(edges.columns) == EDGE_COLUMNS, edges.columns
    assert list(nodes.columns) == NODE_COLUMNS, nodes.columns
    assert pd.api.types.is_datetime64_any_dtype(edges["ts"])
    assert set(edges["sign"].unique()) <= {-1, 1}
    assert set(edges["source_file"].unique()) <= {"body", "title"}
    assert not (edges["src"] == edges["dst"]).any(), "self-links present"
    assert not edges.duplicated(["src", "dst", "post_id"]).any(), "duplicate links present"
    assert edges["ts"].is_monotonic_increasing


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    edges = load_raw()
    nodes = node_stats(edges)
    validate(edges, nodes)
    edges.to_parquet(OUT / "edges.parquet", index=False)
    nodes.to_parquet(OUT / "nodes.parquet", index=False)

    core = nodes.loc[(nodes["n_out"] + nodes["n_in"]) >= MIN_LINKS, "subreddit"].sort_values()
    (OUT / "core_nodes.txt").write_text("\n".join(core) + "\n", encoding="utf-8")

    snap_dir = OUT / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    for q, part in edges.groupby(edges["ts"].dt.to_period("Q")):
        part.to_parquet(snap_dir / f"{q}.parquet", index=False)

    print(f"edges={len(edges):,}  nodes={len(nodes):,}  core_nodes={len(core):,}  "
          f"neg_share={(edges['sign'] < 0).mean():.3%}  range={edges['ts'].min()} -> {edges['ts'].max()}")


if __name__ == "__main__":
    main()
