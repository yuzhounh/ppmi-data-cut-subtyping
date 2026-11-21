# Data Directory

This repository does not include raw PPMI tables, participant-level derived arrays, subtype assignments, or fitted model checkpoints.

## Required Inputs

Obtain the 11 archived official **PPMI Curated Data Cuts** (retrieved 22 October 2026; RRID:SCR_006431) from the [PPMI data portal](https://www.ppmi-info.org/access-data-specimens/download-data) under the PPMI Data Use Agreement:

1. `PPMI_Curated_Data_Cut_Public_20230612_rev.csv`
2. `PPMI_Curated_Data_Cut_Public_20240129.xlsx`
3. `PPMI_Curated_Data_Cut_Public_20240729.xlsx`
4. `PPMI_Curated_Data_Cut_Public_20241211.xlsx`
5. `PPMI_Curated_Data_Cut_Public_20250321.xlsx`
6. `PPMI_Curated_Data_Cut_Public_20250714.xlsx`
7. `PPMI_Curated_Data_Cut_Public_20251112.xlsx`
8. `PPMI_Curated_Data_Cut_Public_20260223.xlsx`
9. `PPMI_Curated_Data_Cut_Public_20260319.xlsx`
10. `PPMI_Curated_Data_Cut_Public_20260511.xlsx`
11. `PPMI_Curated_Data_Cut_Public_20260810.xlsx`

## Placement

Place all 11 files in `data/raw/` (or set the `PPMI_DATA_DIR` environment variable, or pass `--data-dir <PATH>` to `run_subtyping_trajectories.py`). All files under `data/raw/`, `data/derived/`, and `outputs/` are ignored by `.gitignore`.
