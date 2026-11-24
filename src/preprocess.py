"""Build the shared data contract from the raw SNAP Reddit Hyperlink TSVs.

Inputs : data/raw/soc-redditHyperlinks-{body,title}.tsv
Outputs: data/processed/edges.parquet      one row per hyperlink
         data/processed/nodes.parquet      per-subreddit activity stats
         data/processed/core_nodes.txt     analysis subset (>= MIN_LINKS total links)
         data/processed/snapshots/YYYYQn.parquet  quarterly edge lists

Run: python src/preprocess.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
MIN_LINKS = 20  # a subreddit is "core" if it sends + receives at least this many links


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
    return df.sort_values("ts").reset_index(drop=True)


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
    return out.reset_index()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    edges = load_raw()
    edges.to_parquet(OUT / "edges.parquet", index=False)

    nodes = node_stats(edges)
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
