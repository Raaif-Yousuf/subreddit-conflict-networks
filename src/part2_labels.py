"""Part 2 / issue #14: name the largest Leiden communities by their biggest subreddits.

Output: results/part2_community_labels.json, a list (largest community first) of
    {community_id, label, anchor, size, top_by_activity, top_by_internal}
- top_by_activity: members with the most core links (sent + received, any sign): the "biggest" subreddits.
- top_by_internal: members with the most positive links to their own community: the most typical members
  (askreddit-style hubs are big everywhere, so this list says more about what the community is about).
Run (from src/, after part2_communities.py): python part2_labels.py

Names are hand-assigned from both lists (checked on the seed-42 partition). Each name is tied to an anchor
subreddit rather than to a community id, so a changed partition cannot silently mislabel a community: the
script fails if an anchor's community is not among the N_LABEL largest or two anchors share a community.
"""
import json

import pandas as pd

from load import ROOT, load_edges
from part2_communities import undirected_weights

RES = ROOT / "results"
N_LABEL = 12
TOP = 10

# anchor subreddit -> community name
ANCHORS = {
    "askreddit": "Default & meta subs",  # also titlegore, shitredditsays, circlebroke2
    "subredditdrama": "Politics, news & drama",
    "gaming": "Gaming",
    "android": "Tech & programming",
    "iama": "Movies, TV & books",
    "science": "Science, drugs & hobbies",  # also space, diy, collapse, math
    "legaladvice": "Advice, relationships & gender",
    "atheism": "Religion, philosophy & history",
    "music": "Music",
    "writingprompts": "Fiction & anime",
    "nfl": "Sports",
    "bitcoin": "Cryptocurrency",  # plus a few strays (e.g. formula1)
}


def community_tables(memb: pd.Series, edges: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Per-subreddit activity (all core links) and positive weight to its own community."""
    activity = pd.concat([edges["src"], edges["dst"]]).value_counts()
    und = undirected_weights(edges, "positive")
    same = und[und["a"].map(memb) == und["b"].map(memb)]
    internal = pd.concat([same.groupby("a")["w"].sum(), same.groupby("b")["w"].sum()]).groupby(level=0).sum()
    return activity, internal


def label_communities(memb: pd.Series, activity: pd.Series, internal: pd.Series,
                      anchors: dict[str, str] = ANCHORS, n_label: int = N_LABEL, top: int = TOP) -> list[dict]:
    """One record per community among the n_label largest that has an anchor; raises on a bad anchor."""
    sizes = memb.value_counts()
    largest = list(sizes.index[:n_label])
    by_comm = {}
    for anchor, name in anchors.items():
        if anchor not in memb.index:
            raise ValueError(f"anchor {anchor!r} is not in the partition")
        c = memb[anchor]
        if c not in largest:
            raise ValueError(f"anchor {anchor!r} is in community {c}, not among the {n_label} largest")
        if c in by_comm:
            raise ValueError(f"anchors {by_comm[c][0]!r} and {anchor!r} are both in community {c}")
        by_comm[c] = (anchor, name)
    out = []
    for c in largest:
        if c not in by_comm:
            continue
        members = memb.index[memb == c]
        out.append({
            "community_id": int(c), "label": by_comm[c][1], "anchor": by_comm[c][0], "size": int(sizes[c]),
            "top_by_activity": list(activity.reindex(members).fillna(0).sort_values(ascending=False, kind="stable").index[:top]),
            "top_by_internal": list(internal.reindex(members).fillna(0).sort_values(ascending=False, kind="stable").index[:top]),
        })
    return out


def main() -> None:
    memb = pd.read_csv(RES / "part2_node_community.csv").set_index("subreddit")["community_id"]
    activity, internal = community_tables(memb, load_edges(core_only=True))
    labels = label_communities(memb, activity, internal)
    with open(RES / "part2_community_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=1)
    covered = sum(r["size"] for r in labels)
    print(f"{len(labels)} labeled communities cover {covered:,} of {len(memb):,} subreddits ({covered / len(memb):.0%})")
    for r in labels:
        print(f"C{r['community_id']:<3}{r['label']:<34}{r['size']:>5}  {', '.join(r['top_by_activity'][:5])}")


if __name__ == "__main__":
    main()
