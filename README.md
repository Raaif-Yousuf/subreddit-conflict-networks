# Subreddit Conflict Networks

CS 4352 (Social Network Analysis) project: community structure, conflict, and prediction of negative interactions in the SNAP Reddit Hyperlink Network. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full plan.

## Setup
```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
```

## Data
Download the two SNAP TSVs (~650 MB total) from https://snap.stanford.edu/data/soc-RedditHyperlinks.html into `data/raw/`:
```bash
python src/download.py
```

## Run (from `src/`)
| Step | Command | Output |
|---|---|---|
| 1. Build data contract | `python preprocess.py` | `data/processed/` (edges, nodes, core_nodes, quarterly snapshots) |
| 2. Descriptive + communities | `python prelim_analysis.py` | `results/prelim_*.csv`, `figures/prelim_*` |
| 3. Link-prediction baselines | `python baselines.py` | `results/prelim_baselines.csv` |
| 4. Part 1 descriptive stats (#8) | `python part1_descriptive.py` | `results/part1_descriptive.csv` |
| 5. Part 1 roles, RQ2 (#9) | `python part1_roles.py` | `results/part1_node_roles.csv`, `results/part1_roles_tests.csv`, `figures/part1_roles.*` |
| 6. Part 2 communities (#13) | `python part2_communities.py` | `results/part2_node_community.csv` (Leiden, for Part 3), `results/part2_memberships.csv`, `results/part2_communities.csv`, `results/part2_community_sizes.csv`, `results/part2_method_ari.csv`, `results/part2_leiden_seeds.csv`, `figures/part2_community_sizes.*` |
| 7. Part 2 community labels (#14), after 6 | `python part2_labels.py` | `results/part2_community_labels.json` |
| 8. Part 2 polarization, RQ1 (#15), after 6 and 7 | `python part2_polarization.py` (~5 min) | `results/part2_polarization.csv`, `results/part2_polarization_seeds.csv`, `results/part2_hostility_matrix.csv`, `results/part2_hostility_long.csv`, `figures/part2_hostility.*` |

`src/load.py` has the shared loaders: `load_edges`, `load_graph` (date range or `window="2016Q3"`), `load_snapshot`, `list_snapshots`, `load_core_nodes`, `aggregate`, and `to_undirected_signed` (one sign per pair, for triads and community detection).

Check the data contract with `python -m pytest tests` from the repo root.

## Reports
- `report/project_update.pdf`: merged Proposal + Project Update (11/24)
