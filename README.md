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
