# Stability and External Prognostic Validity of Unsupervised Parkinson's Disease Subtyping Across 11 PPMI Curated Releases

Analysis and figure-generation code evaluating individual subtype assignment stability and held-out two-year prognostic separation across 11 archived official curated releases and chronologically ordered training-size growth in Parkinson's Precision Medicine Initiative (PPMI) baseline clinical subtyping.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Paper:** Jing Wang et al. *Stability and external prognostic validity of unsupervised Parkinson's disease subtyping across 11 PPMI curated releases and chronological sample sizes*. (Author list to be finalized prior to journal submission.)

## Overview

This standalone pipeline benchmarks 27 unsupervised Parkinson's disease (PD) subtyping strategies—three baseline clinical feature panels (11-feature multi-domain P1, 7-feature compact P2, and 12-feature P1 plus olfactory UPSIT P3), three clustering algorithms (K-means, diagonal Gaussian mixture models, and Ward hierarchical clustering), and $K \in \{2, 3, 4\}$—using a single deterministic 30% stratified evaluation reserve fixed across all 11 archived PPMI curated extracts ($n = 333\text{--}335$ PD across panels). All preprocessing is fitted strictly inside development sets, and historical clusters are aligned to the August 2026 reference via leak-free training-centroid Hungarian matching. Trajectory A tracks assignment stability (Adjusted Rand Index, Normalized Mutual Information, individual flip rate, consecutive step ARI, and centroid drift) across all 11 official curated releases (297 strategy--release points); Trajectory B grows nested development sets chronologically within August 2026 from $n = 37$ to full size across 1,000 within-snapshot tie permutations (seed `20261121`). External prognostic validity is evaluated on seven held-out two-year clinical anchors never used in clustering.

## Data Access and Setup

No raw PPMI tables, participant-level derivatives, cluster labels, or fitted model checkpoints are distributed in this repository.

- **PPMI Curated Data Cuts (11 archived releases, 12 June 2023 to 10 August 2026; retrieved 22 October 2026)**: Obtain via the [PPMI data portal](https://www.ppmi-info.org/access-data-specimens/download-data) (RRID:SCR_006431) under the PPMI Data Use Agreement.

Place the 11 official curated files in `data/raw/` (or pass `--data-dir` / set `PPMI_DATA_DIR`). See [data/README.md](data/README.md) for the exact file list.

```bash
export PPMI_DATA_DIR=/path/to/ppmi/curated
# PowerShell: $env:PPMI_DATA_DIR = '/path/to/ppmi/curated'
```

## Environment and Usage

Analyses were executed in Python 3.14.6 with dependencies recorded in [requirements.txt](requirements.txt) and [environment_snapshot.json](environment_snapshot.json).

```bash
python -m pip install -r requirements.txt

# 1. Run 10-round pilot audit (partition isolation, centroid matching, and anchor zero-drift checks)
python run_subtyping_trajectories.py --pilot --workers 4

# 2. Run full 1,000-round dual-trajectory evaluation (27 strategies, 297 release points, 162 tiers)
python run_subtyping_trajectories.py --workers 8

# 3. Generate publication figures and summary tables from aggregate CSV outputs
python build_subtyping_assets.py
```

Run the pure-synthetic unit tests without private PPMI data:

```bash
python -m unittest discover -s tests -t .
```

Aggregate CSV/JSON tables, optional LaTeX table bodies (`--latex-tables-dir`), and trajectory figures are written locally under `outputs/`.

## Repository Structure

```text
ppmi-data-cut-subtyping/
├── README.md                          # Overview and reproduction guide
├── LICENSE                            # MIT license
├── CITATION.cff                       # Citation metadata
├── requirements.txt                   # Pinned Python dependencies
├── environment_snapshot.json          # Recorded runtime package versions
├── subtyping_registry.py              # Curated extract parser, feature panels, stratified reserve
├── run_subtyping_trajectories.py      # Pilot audit and 1,000-round dual-trajectory runner
├── build_subtyping_assets.py          # Aggregate summary tables and publication figures
├── tests/
│   ├── __init__.py
│   └── test_subtyping_trajectories.py # Pure-synthetic unit tests
└── data/
    └── README.md                      # Official PPMI file list and directory setup
```

## License and Contact

Released under the [MIT License](LICENSE). Official PPMI data remain subject to the PPMI Data Use Agreement.

Correspondence: Jing Wang (`wangjing@xynu.edu.cn`).
