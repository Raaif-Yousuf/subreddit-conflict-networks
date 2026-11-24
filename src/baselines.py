"""RQ4 baselines: predict NEW negative links between core subreddits in a future window.

Setup (temporal, no leakage):
  - fit window : graph = links before 2016-07-01, labels = new negative links in 2016-07-01..2016-09-30
  - test window: graph = links before 2016-10-01, labels = new negative links in 2016-10-01..2017-04-30
  "New" = the directed pair (u, v) had no negative link u->v in the graph period.
  Candidate pairs: both endpoints are core nodes active in the graph period.
  Negatives: (a) random pairs with no negative link in the label window (1:1),
             (b) hard: pairs that only had POSITIVE links in the label window (1:1).

Methods: Common Neighbors, Jaccard, Adamic-Adar, Preferential Attachment (undirected, all signs),
         Attacker x Target activity (neg out-degree of u * neg in-degree of v),
         Retaliation (v attacked u before), and Logistic Regression on all of the above.
Run: python src/baselines.py
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from load import ROOT, aggregate, load_core_nodes, load_edges

SEED = 42
RES = ROOT / "results"
WINDOWS = {
    "fit": ("2016-07-01", "2016-10-01"),
    "test": ("2016-10-01", "2017-05-01"),
}


class GraphFeatures:
    def __init__(self, hist: pd.DataFrame, nodes: list[str]):
        self.idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        agg = aggregate(hist)
        s = agg.src.map(self.idx).values
        d = agg.dst.map(self.idx).values
        A = sp.coo_matrix((np.ones(len(s)), (s, d)), shape=(n, n)).tocsr()
        U = ((A + A.T) > 0).astype(np.float64).tocsr()
        self.U = U
        self.deg = np.asarray(U.sum(1)).ravel()
        inv_log = np.zeros(n)
        mask = self.deg > 1
        inv_log[mask] = 1.0 / np.log(self.deg[mask])
        self.U_aa = U @ sp.diags(inv_log)
        neg = agg[agg.neg_w > 0]
        self.neg_pairs = set(zip(neg.src, neg.dst))
        self.neg_out = np.bincount(neg.src.map(self.idx), weights=neg.neg_w, minlength=n)
        self.neg_in = np.bincount(neg.dst.map(self.idx), weights=neg.neg_w, minlength=n)

    def features(self, pairs: pd.DataFrame) -> pd.DataFrame:
        u = pairs.src.map(self.idx).values
        v = pairs.dst.map(self.idx).values
        Uu, Uv = self.U[u], self.U[v]
        cn = np.asarray(Uu.multiply(Uv).sum(1)).ravel()
        union = self.deg[u] + self.deg[v] - cn
        aa = np.asarray(self.U_aa[u].multiply(Uv).sum(1)).ravel()
        return pd.DataFrame({
            "Common Neighbors": cn,
            "Jaccard": np.divide(cn, union, out=np.zeros_like(cn), where=union > 0),
            "Adamic-Adar": aa,
            "Preferential Attachment": self.deg[u] * self.deg[v],
            "Attacker x Target": self.neg_out[u] * self.neg_in[v],
            "Retaliation (v attacked u)": np.array([(b, a) in self.neg_pairs for a, b in zip(pairs.src, pairs.dst)], float),
        })


def build_split(edges, core, start, end, rng):
    hist = edges[edges.ts < start]
    hist = hist[hist.src.isin(core) & hist.dst.isin(core)]
    active = sorted(set(hist.src) | set(hist.dst))
    act_set = set(active)
    fut = edges[(edges.ts >= start) & (edges.ts < end) & edges.src.isin(act_set) & edges.dst.isin(act_set)]
    gf = GraphFeatures(hist, active)

    fut_agg = aggregate(fut)
    pos = fut_agg[(fut_agg.neg_w > 0)]
    pos = pos[[(a, b) not in gf.neg_pairs for a, b in zip(pos.src, pos.dst)]][["src", "dst"]]
    future_neg = set(zip(fut_agg.src[fut_agg.neg_w > 0], fut_agg.dst[fut_agg.neg_w > 0]))

    # (a) random negatives
    arr = np.array(active)
    k = len(pos)
    cand = pd.DataFrame({"src": arr[rng.integers(0, len(arr), 3 * k)], "dst": arr[rng.integers(0, len(arr), 3 * k)]})
    cand = cand[(cand.src != cand.dst)]
    cand = cand[[(a, b) not in future_neg for a, b in zip(cand.src, cand.dst)]].drop_duplicates().head(k)
    # (b) hard negatives: only positive links in the label window
    hard = fut_agg[(fut_agg.neg_w == 0) & (fut_agg.pos_w > 0)][["src", "dst"]]
    hard = hard.sample(n=min(k, len(hard)), random_state=SEED)

    out = {}
    for name, negs in (("random", cand), ("hard", hard)):
        pairs = pd.concat([pos.assign(y=1), negs.assign(y=0)], ignore_index=True)
        X = gf.features(pairs)
        out[name] = (X, pairs.y.values)
    stats = {"graph_nodes": len(active), "graph_links": len(hist), "new_neg_pairs": k, "hard_negs": len(hard)}
    return out, stats


def main():
    rng = np.random.default_rng(SEED)
    edges = load_edges()
    core = set(load_core_nodes())
    fit, fit_stats = build_split(edges, core, *WINDOWS["fit"], rng)
    test, test_stats = build_split(edges, core, *WINDOWS["test"], rng)
    print("fit:", fit_stats, "\ntest:", test_stats)

    rows = []
    for setting in ("random", "hard"):
        Xtr, ytr = fit[setting]
        Xte, yte = test[setting]
        for col in Xte.columns:
            rows.append({"negatives": setting, "method": col,
                         "ROC-AUC": roc_auc_score(yte, Xte[col]), "AP": average_precision_score(yte, Xte[col])})
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        Xtr_l, Xte_l = np.log1p(Xtr), np.log1p(Xte)
        clf.fit(Xtr_l, ytr)
        p = clf.predict_proba(Xte_l)[:, 1]
        rows.append({"negatives": setting, "method": "Logistic Regression (all)",
                     "ROC-AUC": roc_auc_score(yte, p), "AP": average_precision_score(yte, p)})
    tab = pd.DataFrame(rows)
    tab[["ROC-AUC", "AP"]] = tab[["ROC-AUC", "AP"]].round(3)
    wide = tab.pivot(index="method", columns="negatives", values=["ROC-AUC", "AP"])
    wide = wide.reindex(tab.method.unique())
    wide.to_csv(RES / "prelim_baselines.csv")
    pd.DataFrame({"fit": fit_stats, "test": test_stats}).to_csv(RES / "prelim_baseline_split_stats.csv")
    print(wide.to_string())


if __name__ == "__main__":
    main()
