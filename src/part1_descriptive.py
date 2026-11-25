"""Part 1 / issue #8: descriptive statistics for the full and core hyperlink graphs.

Output: results/part1_descriptive.csv (rows = statistic, columns = full, core)
Run (from src/): python part1_descriptive.py

Definitions
- links: hyperlink rows (one per post); pairs: distinct directed (src, dst) pairs.
- % negative pairs: pairs with at least one negative link.
- density: pairs / (n (n - 1)) on the directed graph.
- reciprocity: share of directed pairs whose reverse pair also exists.
- avg clustering: networkx average clustering on the undirected, unweighted graph.
- largest WCC / SCC share: share of nodes in the largest weakly / strongly connected component.
"""
import networkx as nx
import pandas as pd

from load import ROOT, aggregate, load_core_nodes, load_edges

RES = ROOT / "results"


def graph_stats(edges: pd.DataFrame) -> dict:
    agg = aggregate(edges)
    G = nx.DiGraph()
    G.add_edges_from(zip(agg["src"], agg["dst"]))
    U = G.to_undirected()
    n, m = G.number_of_nodes(), G.number_of_edges()
    return {
        "nodes": n,
        "links": len(edges),
        "distinct directed pairs": m,
        "% negative links": round(100 * (edges["sign"] < 0).mean(), 2),
        "% negative pairs": round(100 * (agg["neg_w"] > 0).mean(), 2),
        "density": m / (n * (n - 1)),
        "reciprocity": round(nx.reciprocity(G), 3),
        "avg clustering (undirected)": round(nx.average_clustering(U), 3),
        "largest WCC share": round(len(max(nx.weakly_connected_components(G), key=len)) / n, 3),
        "largest SCC share": round(len(max(nx.strongly_connected_components(G), key=len)) / n, 3),
    }


def main() -> None:
    RES.mkdir(exist_ok=True)
    edges = load_edges()
    core = set(load_core_nodes())
    core_edges = edges[edges["src"].isin(core) & edges["dst"].isin(core)]
    tab = pd.DataFrame({"full": graph_stats(edges), "core": graph_stats(core_edges)})
    tab.index.name = "statistic"
    tab.to_csv(RES / "part1_descriptive.csv")
    with pd.option_context("display.float_format", "{:,.6g}".format):
        print(tab)


if __name__ == "__main__":
    main()
