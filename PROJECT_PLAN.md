# CS 4352 Project Plan — Conflict & Polarization Between Reddit Communities

## 0. Requirements & deadlines (from syllabus + Brightspace, checked 11/23)

| Deliverable | Weight (of project) | Due | Requirement |
|---|---|---|---|
| Proposal | 10% | 11/5 — **done**: submitted as the merged Proposal + Update, which Dr. Derr approved by email | 1 page: overview, dataset, methods, weekly timeline |
| **Project Update** | 10% | Tue 11/24, 11:59 PM (Brightspace) — **done** | 1–2 pages: progress, **preliminary results**, changes, updated timeline. Submit as merged Proposal + Update. |
| Project Paper | 40% | TBD, est. Week 15 (~12/8–12/10) | AAAI template, 4–10 pages (aim 6–8), references don't count |
| Presentation | 33% | Week 15 (overflow: final exam slot) | ~12 min + ~2 min Q&A; graded per person |
| Code & docs | 3% | TBD | Reasonably documented code |
| Peer evaluations | 4% | After last presentations | Feedback forms for other groups |

Late policy: first late submission <24h has no penalty (useful safety net for the Update).

## 1. Project summary

**Title (working):** *Who Attacks Whom? Community Structure, Conflict, and Predicting Negative Interactions Between Reddit Communities*

**Data:** Stanford SNAP **Reddit Hyperlink Network** (`soc-RedditHyperlinks`) — https://snap.stanford.edu/data/soc-RedditHyperlinks.html
- Directed edges = a post in subreddit A links to subreddit B. ~56k subreddits, ~860k links, Jan 2014 – Apr 2017.
- Two TSVs: `body` (links in post body) and `title` (links in post title). Columns: `SOURCE_SUBREDDIT, TARGET_SUBREDDIT, POST_ID, TIMESTAMP, LINK_SENTIMENT (+1/−1), PROPERTIES` (86 text features of the post).
- Optional: SNAP subreddit embeddings (300-d) as node features.
- Source paper to cite and compare against: Kumar, Hamilton, Leskovec, Jurafsky, *Community Interaction and Conflict on the Web*, WWW 2018.

**Project type:** A (seed idea: "Community Detection and Polarization" from the ideas deck) with a type-B-style prediction component.

**Research questions**
- **RQ1 (structure):** Do the communities we detect line up with conflict? Is negative linking concentrated *between* detected communities or *within* them?
- **RQ2 (roles):** Who are the attackers and who are the targets? How do centrality, size, and activity differ? Are negative links reciprocated? Do signed triads follow structural balance?
- **RQ3 (time):** Does conflict escalate or stay stable? Do communities and between-community hostility change across quarters?
- **RQ4 (prediction):** Given the network up to time *t*, can we predict which subreddit pairs will have a **new negative link** in the next window? Do embedding/GNN methods beat classic link-prediction heuristics?

**Course topics covered:** centrality, community detection (Louvain/Leiden, modularity), signed networks and balance, link prediction heuristics, node2vec, GNNs (GraphSAGE/GCN in PyG), temporal networks.

## 2. Shared conventions (these make handoffs easy)

**Repo layout** (one shared GitHub repo):
```
data/raw/            # SNAP TSVs (git-ignored; download script instead)
data/processed/      # outputs of Part 1 — the contract other parts read
src/                 # reusable code (loaders, metrics)
notebooks/           # exploration, one per part
results/             # CSV tables produced by each part
figures/             # PNG/PDF figures, named partN_<what>.pdf
paper/               # Overleaf export / LaTeX (AAAI template)
slides/
README.md            # how to reproduce everything
```

**Data contract (produced by Part 1, consumed by Parts 2–3):**
- `data/processed/edges.parquet` — one row per link: `src, dst, ts (datetime), sign (+1/−1), source_file (body/title), post_id`
- `data/processed/nodes.parquet` — `subreddit, n_out, n_in, n_neg_out, n_neg_in, first_ts, last_ts`
- `data/processed/snapshots/` — quarterly edge lists `YYYYQn.parquet`
- `data/processed/core_nodes.txt` — the analysis subset (e.g., subreddits with ≥ k total links, so community detection and GNNs stay fast)
- `src/load.py` with `load_edges()`, `load_graph(signed=True, weighted=True, window=None)` returning a networkx DiGraph with `pos_w`, `neg_w` edge attributes.

**Fixed decisions (agree once, don't relitigate):** Python 3.11, networkx + `python-igraph`/`leidenalg`, pandas, scikit-learn, PyTorch Geometric. Random seed 42. Temporal split: train ≤ 2016-06, validation 2016-07 to 2016-09, test 2016-10 to 2017-04.

**Results contract:** every part saves its numbers to `results/partN_*.csv` and figures to `figures/partN_*.pdf`, and writes a 5–10 line summary in `results/partN_SUMMARY.md` (what was run, key numbers, caveats). The writing part (Part 4) only reads these files.

---

## 3. Tonight: Project Update (due Tue 11/24, 11:59 PM)

Split into two halves that run in parallel and merge in one shared doc.

**Update — Half A (data + preliminary numbers)**
1. Download both SNAP TSVs. Load with pandas.
2. Compute and tabulate: #nodes, #edges, % negative, date range, degree distribution (log-log plot), top 10 subreddits by negative links sent and received.
3. Quick Louvain (networkx `louvain_communities`) on the undirected positive graph of the top ~5k most active subreddits. Report modularity, #communities, and **% of negative links that fall within vs. between communities** — this is the headline preliminary result.
4. Hand off: 2 figures + 1 table + 4–5 bullet takeaways.

**Update — Half B (document)**
1. Set up Overleaf with the AAAI template (reuse it for the final paper).
2. Write: title, motivation (1 paragraph), dataset description, RQ1–RQ4, methods (as in Section 4 below), note that topic changed/was finalized late per email with Dr. Derr, and the updated weekly timeline (Section 5).
3. Drop in Half A's figures/table and takeaways. Keep it to 2 pages. Export PDF, both review, submit.

---

## 4. Main work packages

Four parts of roughly equal effort. Parts 2 and 3 can run fully in parallel once Part 1 delivers the data contract. Part 4 runs alongside all of them. Any split of two parts per person works; the most balanced pairings are **{1, 4}** and **{2, 3}**, or **{1, 3}** and **{2, 4}**. We'll pick together once the plan is approved. Whoever owns a part also writes that part's Methods/Results paragraphs and presents those slides.

### Part 1 — Data pipeline, descriptive analysis, roles (RQ2)
**Goal:** clean data everyone else builds on, plus the "who attacks whom" analysis.
1. `src/download.py` + `src/preprocess.py`: merge body + title files, parse timestamps, dedupe, build the data contract files (Section 2). Pick the `core_nodes` threshold so the core graph is ~5–10k nodes.
2. `src/load.py` helpers + quarterly snapshots.
3. Descriptive stats table (full vs. core graph): nodes, edges, % negative, density, reciprocity, clustering, largest component.
4. **RQ2 roles:** for each subreddit compute in/out degree, PageRank, betweenness (approx.), negative out-ratio and in-ratio. Compare attackers (top negative senders) vs. targets vs. everyone else (box plots + Mann-Whitney tests).
5. Negative reciprocity: P(B→A negative | A→B negative) vs. baseline.
6. Signed triad census on the undirected signed core graph: share of balanced (+++, +−−) vs. unbalanced (++−, −−−) triangles vs. a sign-shuffled null model.

**Handoff out:** data contract files (needed by Parts 2 and 3 — **deliver first, by Sat 11/28**), `results/part1_*.csv`, `figures/part1_*.pdf`, `results/part1_SUMMARY.md`.

### Part 2 — Communities, polarization, and time (RQ1 + RQ3)
**Goal:** show how community structure relates to conflict, and how that changes over time.
1. Community detection on the core graph with **Leiden** (main) and **Louvain** + **spectral clustering** (comparisons). Report modularity and community sizes for each.
2. Label the top ~10 communities by their biggest subreddits (e.g., gaming, politics, sports).
3. **Polarization metrics:** fraction of negative links within vs. between communities; community-to-community "hostility matrix" (negative links / all links) as a heatmap; compare against a sign-shuffled null.
4. Does including negative edges change the communities? Compare positive-only vs. all-edges vs. a signed-modularity variant (adjusted Rand index between partitions).
5. **RQ3 temporal:** run Leiden per quarter, match communities across quarters (Jaccard overlap), and plot hostility between the top community pairs over time. Check whether conflict is bursty (share of negative links in the top 1% of weeks).

**Handoff out:** `results/part2_*.csv` (including a `node → community_id` file, which Part 3 uses as a feature), `figures/part2_*.pdf`, `results/part2_SUMMARY.md`.

### Part 3 — Predicting future negative links (RQ4)
**Goal:** the modeling contribution — an empirical comparison of methods on a temporal split.
1. Task setup: graph = links before the cutoff. Positives = (src, dst) pairs with a **new** negative link in the test window. Negatives = sampled pairs with no link (and a harder variant: pairs with only positive links in the test window). Metrics: ROC-AUC, AP, Hits@50.
2. **Heuristic baselines:** common neighbors, Jaccard, Adamic-Adar, preferential attachment, past negative interaction (A ever attacked B?).
3. **Feature model:** logistic regression or gradient boosting on pair features (heuristics + Part 1 node stats + same/different community from Part 2).
4. **Embeddings:** node2vec on the training graph → Hadamard edge features → logistic regression.
5. **GNN:** GraphSAGE (and optionally GCN/GAT) link predictor in PyG, using the SNAP subreddit embeddings or degree features as node features. Optional stretch: a signed GNN (SGCN).
6. Ablations: with vs. without negative edges in the input graph; with vs. without the community feature.

**Handoff in:** data contract (Part 1); `node → community_id` (Part 2) — if not ready, run without it first and add it later as an ablation.
**Handoff out:** `results/part3_main_table.csv` (method × metric), ablation table, ROC/PR curve figure, `results/part3_SUMMARY.md`, and a runnable `src/predict.py`.

### Part 4 — Paper, related work, presentation, reproducibility
**Goal:** turn the results into the graded deliverables (paper 40%, presentation 33%, code 3%).
1. Related work (6–10 refs): Kumar et al. 2018 (the dataset paper); Leskovec et al. on signed networks, balance and status (2010); Blondel (Louvain) and Traag (Leiden); Grover & Leskovec (node2vec); Hamilton et al. (GraphSAGE); Derr et al. (SGCN, signed GNN); Garimella et al. and other echo-chamber / polarization papers.
2. Paper skeleton in Overleaf (AAAI): Abstract, Introduction, Related Work, Data, Methods, Results (RQ1–RQ4), Discussion & Limitations, Conclusion. Write Intro, Related Work, Data, and Discussion; fill Methods and Results from each part's `SUMMARY.md` plus a short methods blurb written by the person who ran that part.
3. Figure polish: consistent style, captions, a single "pipeline overview" diagram.
4. Slides (~12 min ≈ 10–12 slides): motivation → data → RQ1 → RQ2 → RQ3 → RQ4 → takeaways. Build a speaker-notes split so both people present (presentation is graded per person).
5. Reproducibility: `README.md` (setup, download, run order), `requirements.txt`, docstrings, one `run_all.sh`.

**Handoff in:** every `results/partN_SUMMARY.md` + figures. **Handoff out:** paper PDF, slides, cleaned repo.

---

## 5. Timeline

| When | Milestone | Parts active |
|---|---|---|
| **Mon 11/23 (tonight)** | Repo set up, data downloaded, preliminary numbers | Update A + B |
| **Tue 11/24** | **Submit merged Proposal + Update** | Update A + B |
| Wed 11/25 – Sat 11/28 (Thanksgiving) | Data contract finalized and pushed; Overleaf skeleton + related work started | 1, 4 (2 and 3 start exploring on the preliminary data) |
| Sun 11/29 – Wed 12/2 | Leiden + polarization + hostility matrix; heuristic baselines + node2vec; roles + triads | 1, 2, 3, 4 |
| **Thu 12/3** (class has reading presentations) | **Mid-check:** every part posts a first `SUMMARY.md`; decide what makes the paper | all |
| Fri 12/4 – Sun 12/6 | Temporal analysis; GNN + ablations; methods/results drafted; slides v1 | 2, 3, 4 |
| Mon 12/7 | **Freeze results.** Full paper draft; dry-run presentation (time it) | all |
| Week 15 (12/8 – 12/10) | Present; final paper + code submission (confirm exact date with Dr. Derr) | all |

## 6. Risks & fallbacks

| Risk | Fallback |
|---|---|
| Graph too big for Leiden/GNN on a laptop | Restrict to `core_nodes` (≥ k links); use Colab/ACCRE GPU for PyG |
| Few negative links → class imbalance (~10% negative) | Report AP and Hits@K, not just AUC; balanced negative sampling |
| GNN doesn't beat heuristics | That's a valid finding — explain why (e.g., past-attack feature dominates); keep the ablations |
| Temporal community matching is messy | Fall back to fixed communities (trained on all data) and track hostility over time only |
| Part 2 community IDs late for Part 3 | Part 3 runs without the feature first and adds it as an ablation |

## 7. Definition of done
- [x] Update submitted 11/24
- [ ] Each of RQ1–RQ4 has ≥ 1 table or figure in the paper
- [ ] Main prediction table: ≥ 4 heuristics + feature model + node2vec + GNN, on the temporal test split
- [ ] Paper is 6–8 pages in the AAAI format; every figure is referenced in the text
- [ ] Slides timed to ≤ 12 min; both people present
- [ ] Repo reproduces all results from `README.md`
