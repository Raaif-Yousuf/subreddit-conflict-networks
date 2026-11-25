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

`src/load.py` has the shared loaders: `load_edges`, `load_graph` (date range or `window="2016Q3"`), `load_snapshot`, `list_snapshots`, `load_core_nodes`, `aggregate`, and `to_undirected_signed` (one sign per pair, for triads and community detection).

Check the data contract with `python -m pytest tests` from the repo root.

## Reports
- `report/project_update.pdf`: merged Proposal + Project Update (11/24)
