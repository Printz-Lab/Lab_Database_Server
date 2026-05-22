# Lab Characterization Data Browser

A lightweight system for organizing, standardizing, and exploring experimental data from multiple characterization techniques (PL, trPL, PLQY, UV-vis, etc.) using Python and Streamlit.

## Overview

This project has two main components:

1. Data standardization GUI → converts raw instrument exports into a consistent format  
2. Streamlit browser → interactive web app for viewing and comparing data  

---

## Features

### Data Standardization
- GUI-based workflow (no coding required)
- Supports:
  - Steady-state PL
  - Time-resolved PL (trPL)
  - PLQY (raw + analyzed)
  - UV-vis (including Tauc analysis)
- Automatically organizes data into a consistent structure
- Stores:
  - raw data
  - processed data
  - fit parameters
  - metadata

### Data Browser
- Browse experiments by sample
- Search by:
  - sample name
  - formulation
  - batch
- Compare datasets across samples
- Interactive plots (Plotly)
- UV-vis support (absorbance + Tauc + Eg)
- Statistical plots (box plots for Eg, PL peaks, lifetimes)

---

## Project Structure
```
standardized_lab_data/
├── manifests/
│   ├── samples.csv
│   └── experiments.csv
│
├── experiments/
│   ├── sample1__uvvis__001/
│   │   ├── raw_data.csv
│   │   ├── processed_data.csv
│   │   ├── fit_parameters.csv
│   │   └── metadata.json
│   └── ...
```
---

## Installation

pip install -r requirements.txt

Or (conda):

conda create -n labbrowser python=3.11
conda activate labbrowser
pip install -r requirements.txt

---

## Usage

### Standardize Data

python standardize_lab_optics_data.py

### GUI (cross-platform)

The standardizer now uses a **PySide6 (Qt)** desktop GUI, which is compatible with Linux (including AlmaLinux + GNOME), macOS, and Windows.

Use the app buttons to:
- load an existing database
- add files
- edit sample metadata
- choose output folder
- export standardized data

### Launch App

streamlit run streamlit_lab_browser.py

---

## Database Setup

Recommended structure:
```
LabDatabases/
├── main_database/
├── uvvis_tests/
├── archived_data/
```
---

## Notes

- must standardize data and upload to database before using the browser
- Be consistent with sample naming and formulation formatting

---

## Troubleshooting

If Streamlit command fails:

python -m streamlit run streamlit_lab_browser.py

If data does not load:
- Ensure manifests folder exists
- Check samples.csv and experiments.csv

---

## Purpose

- Organize experimental data
- Enable fast comparison across samples
- Support data-driven analysis and plotting

---

## Author
Sean Raglow
Internal lab tool


## SQLite Database (Incremental + Idempotent)

The exporter now builds/updates `lab_data.sqlite` in each database root.

- Incremental updates: new experiments are appended without rebuilding existing tables.
- Idempotent re-runs: existing experiment IDs are upserted and point rows are replaced only for that experiment.
- Ingestion tracking: `ingestion_log` stores `experiment_id`, `source_hash`, `parser_version`, and ingest timestamp.
- Plot data storage: measurement points are stored in per-type tables (`points_<measurement_type>`).

Core SQLite tables:
- `samples`
- `experiments`
- `metrics_long`
- `ingestion_log`
- `points_<measurement_type>` (created on demand)

The Streamlit browser prefers SQLite for manifests and plotted points, and falls back to CSV files if needed.


## Validating "acceptable data file examples"

Use the **Validate example folder** button in the standardizer GUI to scan your examples folder and report:
- files that are recognized by current parsers
- files that are not yet implemented

This makes it easy to confirm that every example format is supported before production use.

## How to add new data types (JV, EQE, XRD, etc.)

1. **Add a parser function** in `standardize_lab_optics_data.py`
   - Create `parse_<type>_<format>(path: Path) -> List[dict]`.
   - Return records with the same schema used by existing parsers:
     - `measurement_type` (e.g. `jv_curve`, `eqe_spectrum`)
     - `experiment_subtype`
     - `sample_guess`, `source_label`, `source_file`
     - `raw_df`, `processed_df`, optional `params_df`, and `metadata`

2. **Register detection logic** in `detect_and_parse(path)`
   - Add format checks (sheet names / filename patterns / CSV columns).
   - Route recognized files to your new parser.

3. **Keep tabular numeric columns in `processed_df`**
   - The exporter stores per-experiment points into SQLite `points_<measurement_type>`.
   - The browser can then plot from SQLite directly.

4. **(Optional) Add display defaults in browser**
   - In `streamlit_lab_browser.py`, extend `MEASUREMENT_CONFIG` with:
     - `pretty_name`
     - `default_x`
     - `default_y`
   - This improves auto-selected axes/series for the new type.

5. **Validate with example files**
   - Put representative files in your examples folder.
   - Run the GUI button **Validate example folder** and ensure unsupported count is zero.

6. **Export once to materialize new tables**
   - On export, new `measurement_type` values automatically create/update SQLite tables:
     - `points_<measurement_type>`
   - Existing experiments remain intact (incremental/idempotent behavior).
