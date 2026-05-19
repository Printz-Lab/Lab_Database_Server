from __future__ import annotations

import json
import re
import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


APP_TITLE = "Lab Characterization Data Standardizer"


# -----------------------------
# Parsing helpers
# -----------------------------

def clean_sample_name(name: str) -> str:
    s = str(name).strip()
    s = s.replace("_Data", "")
    s = s.replace("test_I_think_", "")
    s = s.replace("gthree", "g3")
    s = s.replace("Blank", "blank")
    s = s.replace("blank_", "blank_")
    # turn g2 -> G2, g10 -> G10
    m = re.fullmatch(r"g(\d+)", s.lower())
    if m:
        return f"G{int(m.group(1))}"
    if s.lower() == "blank":
        return "BLANK"
    return s.strip()




def canonicalize_sample_key(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace("_data", "")
    return re.sub(r"[^a-z0-9]+", "", s)


def slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)




def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def init_sqlite_database(db_path: Path) -> None:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS samples (
        sample_id TEXT PRIMARY KEY,
        formulation TEXT,
        batch TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        sample_id TEXT NOT NULL,
        measurement_type TEXT NOT NULL,
        experiment_subtype TEXT,
        source_label TEXT,
        source_file TEXT,
        raw_data_path TEXT,
        processed_data_path TEXT,
        fit_parameters_path TEXT,
        parser TEXT,
        parser_version TEXT,
        source_hash TEXT,
        ingested_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ingestion_log (
        experiment_id TEXT PRIMARY KEY,
        source_hash TEXT NOT NULL,
        parser_version TEXT NOT NULL,
        ingested_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS metrics_long (
        experiment_id TEXT NOT NULL,
        metric_name TEXT NOT NULL,
        metric_value REAL,
        PRIMARY KEY (experiment_id, metric_name)
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_experiments_sample ON experiments(sample_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_experiments_measurement ON experiments(measurement_type)")
    con.commit()
    con.close()


def table_for_measurement(measurement_type: str) -> str:
    m = re.sub(r"[^a-z0-9]+", "_", str(measurement_type).lower()).strip("_")
    return f"points_{m or 'unknown'}"


def upsert_experiment_to_sqlite(db_path: Path, experiment_row: dict, sample_row: dict, source_hash: str, parser_version: str,
                                processed_df: Optional[pd.DataFrame], raw_df: Optional[pd.DataFrame], params_df: Optional[pd.DataFrame], metadata: dict) -> None:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    cur.execute(
        "INSERT INTO samples(sample_id, formulation, batch) VALUES(?,?,?) ON CONFLICT(sample_id) DO UPDATE SET formulation=excluded.formulation, batch=excluded.batch",
        (sample_row.get("sample_id",""), sample_row.get("formulation",""), sample_row.get("batch",""))
    )

    cur.execute(
        """INSERT INTO experiments(
            experiment_id, sample_id, measurement_type, experiment_subtype, source_label, source_file,
            raw_data_path, processed_data_path, fit_parameters_path, parser, parser_version, source_hash, ingested_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
        ON CONFLICT(experiment_id) DO UPDATE SET
            sample_id=excluded.sample_id, measurement_type=excluded.measurement_type, experiment_subtype=excluded.experiment_subtype,
            source_label=excluded.source_label, source_file=excluded.source_file, raw_data_path=excluded.raw_data_path,
            processed_data_path=excluded.processed_data_path, fit_parameters_path=excluded.fit_parameters_path, parser=excluded.parser,
            parser_version=excluded.parser_version, source_hash=excluded.source_hash, ingested_at=CURRENT_TIMESTAMP
        """,
        (experiment_row.get("experiment_id",""), experiment_row.get("sample_id",""), experiment_row.get("measurement_type",""),
         experiment_row.get("experiment_subtype",""), experiment_row.get("source_label",""), experiment_row.get("source_file",""),
         experiment_row.get("raw_data_path",""), experiment_row.get("processed_data_path",""), experiment_row.get("fit_parameters_path",""),
         experiment_row.get("parser",""), parser_version, source_hash)
    )

    measurement_type = experiment_row.get("measurement_type", "")
    points_table = table_for_measurement(measurement_type)
    cols = []
    df = processed_df if processed_df is not None and not processed_df.empty else raw_df
    if df is not None and not df.empty:
        cur.execute(f'DROP TABLE IF EXISTS "{points_table}"') if False else None
        cols = [c for c in df.columns]
        schema_cols = ", ".join([f'"{c}" REAL' for c in cols])
        cur.execute(f'CREATE TABLE IF NOT EXISTS "{points_table}" (experiment_id TEXT NOT NULL, row_idx INTEGER NOT NULL, {schema_cols}, PRIMARY KEY(experiment_id,row_idx))')
        cur.execute(f'DELETE FROM "{points_table}" WHERE experiment_id=?', (experiment_row.get("experiment_id",""),))
        ins_cols = ', '.join([f'"{c}"' for c in cols])
        placeholders = ', '.join(['?'] * (2 + len(cols)))
        rows=[]
        numeric_df = df.copy()
        for c in cols:
            numeric_df[c] = pd.to_numeric(numeric_df[c], errors='coerce')
        for i, r in numeric_df.reset_index(drop=True).iterrows():
            rows.append((experiment_row.get("experiment_id",""), int(i), *[None if pd.isna(v) else float(v) for v in r.tolist()]))
        cur.executemany(f'INSERT OR REPLACE INTO "{points_table}" (experiment_id, row_idx, {ins_cols}) VALUES ({placeholders})', rows)

    metrics = {}
    if params_df is not None and not params_df.empty:
        first = params_df.iloc[0]
        for c,v in first.items():
            try:
                metrics[str(c)] = float(v)
            except Exception:
                pass
    if isinstance(metadata, dict):
        md = metadata.get('metadata', metadata) if isinstance(metadata.get('metadata', metadata), dict) else metadata
        for c,v in md.items():
            try:
                metrics[f'metadata.{c}'] = float(v)
            except Exception:
                pass
    cur.execute('DELETE FROM metrics_long WHERE experiment_id=?', (experiment_row.get("experiment_id",""),))
    if metrics:
        cur.executemany('INSERT OR REPLACE INTO metrics_long(experiment_id, metric_name, metric_value) VALUES (?,?,?)',
                        [(experiment_row.get("experiment_id",""), k, v) for k,v in metrics.items()])

    cur.execute('INSERT OR REPLACE INTO ingestion_log(experiment_id, source_hash, parser_version, ingested_at) VALUES (?,?,?,CURRENT_TIMESTAMP)',
                (experiment_row.get("experiment_id",""), source_hash, parser_version))
    con.commit(); con.close()

def load_existing_manifests(out_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifests_dir = out_root / "manifests"
    exp_path = manifests_dir / "experiments.csv"
    samp_path = manifests_dir / "samples.csv"

    existing_experiments = pd.read_csv(exp_path).fillna("") if exp_path.exists() else pd.DataFrame()
    existing_samples = pd.read_csv(samp_path).fillna("") if samp_path.exists() else pd.DataFrame()
    return existing_experiments, existing_samples


def make_duplicate_key(sample_id: str, measurement_type: str, subtype: str, source_file: str, source_label: str) -> str:
    return "||".join([
        str(sample_id).strip().lower(),
        str(measurement_type).strip().lower(),
        str(subtype).strip().lower(),
        Path(str(source_file)).name.strip().lower(),
        str(source_label).strip().lower(),
    ])


def experiment_seq_from_id(experiment_id: str) -> int:
    m = re.search(r"__(\d+)$", str(experiment_id).strip())
    return int(m.group(1)) if m else 0


def build_existing_counter_map(existing_experiments: pd.DataFrame) -> Dict[str, int]:
    counters: Dict[str, int] = {}
    if existing_experiments is None or existing_experiments.empty:
        return counters

    required = {"sample_id", "measurement_type", "experiment_id"}
    if not required.issubset(existing_experiments.columns):
        return counters

    for _, row in existing_experiments.iterrows():
        key = f"{str(row.get('sample_id', '')).strip()}__{str(row.get('measurement_type', '')).strip()}"
        counters[key] = max(counters.get(key, 0), experiment_seq_from_id(row.get("experiment_id", "")))
    return counters


def merge_sample_manifests(existing_samples: pd.DataFrame, new_rows: List[dict]) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)
    if existing_samples is None or existing_samples.empty:
        merged = new_df
    elif new_df.empty:
        merged = existing_samples.copy()
    else:
        merged = pd.concat([existing_samples, new_df], ignore_index=True)

    if merged.empty:
        return merged

    for col in ["sample_id", "formulation", "batch"]:
        if col not in merged.columns:
            merged[col] = ""

    merged = merged.fillna("")
    merged = merged.sort_values(["sample_id", "formulation", "batch"], kind="stable")

    # Keep one row per sample_id, preferring rows with more filled metadata.
    merged["_filled_score"] = (
        (merged["formulation"].astype(str).str.strip() != "").astype(int)
        + (merged["batch"].astype(str).str.strip() != "").astype(int)
    )
    merged = merged.sort_values(["sample_id", "_filled_score"], ascending=[True, False], kind="stable")
    merged = merged.drop_duplicates(subset=["sample_id"], keep="first").drop(columns="_filled_score")
    return merged.reset_index(drop=True)


# -----------------------------
# Known-format parsers
# -----------------------------

def parse_sspl_excel(path: Path) -> List[dict]:
    """Parses steady-state PL format like GaussianAllegedly.xlsx."""
    outputs: List[dict] = []
    xl = pd.ExcelFile(path)

    raw_wl = pd.read_excel(path, sheet_name="Raw Data (wavelength)") if "Raw Data (wavelength)" in xl.sheet_names else None
    raw_e = pd.read_excel(path, sheet_name="Raw Data (Energy)") if "Raw Data (Energy)" in xl.sheet_names else None
    fit_wl = pd.read_excel(path, sheet_name="Fit Data") if "Fit Data" in xl.sheet_names else None
    fit_e = pd.read_excel(path, sheet_name="Fit Data (Energy)") if "Fit Data (Energy)" in xl.sheet_names else None
    fit_params = pd.read_excel(path, sheet_name="Fit Parameters") if "Fit Parameters" in xl.sheet_names else None

    if raw_wl is None:
        return outputs

    # columns come in pairs: wavelength, sample_data; fit sheets come in triplets: wavelength, sample_data, fit
    for i in range(0, len(raw_wl.columns), 2):
        x_col = raw_wl.columns[i]
        y_col = raw_wl.columns[i + 1]
        source_label = str(y_col).replace("_Data", "")
        sample_guess = clean_sample_name(source_label)

        raw_df = pd.DataFrame({
            "wavelength_nm": pd.to_numeric(raw_wl[x_col], errors="coerce"),
            "intensity_counts": pd.to_numeric(raw_wl[y_col], errors="coerce"),
        }).dropna()

        processed = raw_df.copy()
        if raw_e is not None and i + 1 < len(raw_e.columns):
            processed["energy_eV"] = pd.to_numeric(raw_e.iloc[:, i], errors="coerce")
            processed["intensity_energy_axis_counts"] = pd.to_numeric(raw_e.iloc[:, i + 1], errors="coerce")

        # look for matching fit triplet in fit_wl
        if fit_wl is not None:
            for j in range(0, len(fit_wl.columns), 3):
                sample_col = fit_wl.columns[j + 1]
                if str(sample_col) == y_col:
                    processed = processed.copy()
                    processed["fit_intensity_counts"] = pd.to_numeric(fit_wl.iloc[:, j + 2], errors="coerce")
                    break
        if fit_e is not None:
            for j in range(0, len(fit_e.columns), 3):
                sample_col = fit_e.columns[j + 1]
                if str(sample_col) == y_col:
                    processed = processed.copy()
                    processed["fit_energy_axis_counts"] = pd.to_numeric(fit_e.iloc[:, j + 2], errors="coerce")
                    break

        params_df = None
        if fit_params is not None and "Sample" in fit_params.columns:
            match = fit_params[fit_params["Sample"].astype(str) == y_col]
            if match.empty:
                match = fit_params[fit_params["Sample"].astype(str) == f"{source_label}_Data"]
            if not match.empty:
                params_df = match.reset_index(drop=True)

        outputs.append({
            "source_file": str(path),
            "parser": "steady_state_pl_excel",
            "measurement_type": "steady_state_pl",
            "experiment_subtype": "gaussian_fit",
            "source_label": source_label,
            "sample_guess": sample_guess,
            "raw_df": raw_df,
            "processed_df": processed,
            "params_df": params_df,
            "metadata": {
                "input_filename": path.name,
                "fit_model": params_df.iloc[0].get("Best Fit Model") if params_df is not None else None,
            },
        })
    return outputs


def parse_trpl_excel(path: Path) -> List[dict]:
    """Parses trPL format like trpl_re-fit.xlsx."""
    outputs: List[dict] = []
    xl = pd.ExcelFile(path)
    raw = pd.read_excel(path, sheet_name="Raw Data") if "Raw Data" in xl.sheet_names else None
    fit = pd.read_excel(path, sheet_name="Fitted Data") if "Fitted Data" in xl.sheet_names else None
    params = pd.read_excel(path, sheet_name="Fit Parameters") if "Fit Parameters" in xl.sheet_names else None
    if raw is None:
        return outputs

    for i in range(0, len(raw.columns), 2):
        t_col = raw.columns[i]
        y_col = raw.columns[i + 1]
        source_label = re.sub(r"^Intensity \((.*)\)$", r"\1", str(y_col))
        source_label = re.sub(r"^Time \((.*)\)$", r"\1", source_label)
        sample_guess = clean_sample_name(source_label)

        raw_df = pd.DataFrame({
            "time_ns": pd.to_numeric(raw[t_col], errors="coerce"),
            "intensity_counts": pd.to_numeric(raw[y_col], errors="coerce"),
        }).dropna()

        processed = raw_df.copy()
        if fit is not None:
            time_match = f"Time ({source_label})"
            int_match = f"Intensity ({source_label})"
            fit_match = f"Fit ({source_label})"
            if all(col in fit.columns for col in [time_match, int_match, fit_match]):
                processed = pd.DataFrame({
                    "time_ns": pd.to_numeric(fit[time_match], errors="coerce"),
                    "intensity_counts": pd.to_numeric(fit[int_match], errors="coerce"),
                    "fit_intensity_counts": pd.to_numeric(fit[fit_match], errors="coerce"),
                }).dropna(how="all")

        params_df = None
        if params is not None and "Sample" in params.columns:
            match = params[params["Sample"].astype(str) == source_label]
            if not match.empty:
                params_df = match.reset_index(drop=True)

        outputs.append({
            "source_file": str(path),
            "parser": "trpl_excel",
            "measurement_type": "time_resolved_pl",
            "experiment_subtype": "multi_exp_fit",
            "source_label": source_label,
            "sample_guess": sample_guess,
            "raw_df": raw_df,
            "processed_df": processed,
            "params_df": params_df,
            "metadata": {
                "input_filename": path.name,
                "average_tau_ns": float(params_df.iloc[0]["Average Tau"]) if params_df is not None and "Average Tau" in params_df.columns else None,
            },
        })
    return outputs


def parse_plqy_analyzed_csv(path: Path) -> List[dict]:
    outputs: List[dict] = []
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        source_label = str(row.get("Sample Name", "")).strip()
        sample_guess = clean_sample_name(source_label)
        summary_df = pd.DataFrame([row])
        outputs.append({
            "source_file": str(path),
            "parser": "plqy_analyzed_csv",
            "measurement_type": "plqy_summary",
            "experiment_subtype": "analyzed_summary",
            "source_label": source_label,
            "sample_guess": sample_guess,
            "raw_df": None,
            "processed_df": summary_df,
            "params_df": None,
            "metadata": {"input_filename": path.name},
        })
    return outputs


def _parse_plqy_trace_name(trace_name: str) -> dict:
    s = str(trace_name).replace("_Data_corrected", "").replace("_Data", "")
    s = s.replace("gthree", "g3").replace("Blank", "blank")
    parts = s.split("_")
    sample = clean_sample_name(parts[0]) if parts else "UNKNOWN"

    mode = None
    angle_deg = None
    corrected = "corrected" in trace_name.lower()

    for p in parts[1:]:
        pl = p.lower()
        if pl in {"exc", "ex"}:
            mode = "excitation"
        elif pl in {"em"}:
            mode = "emission"
        elif pl in {"mc", "combined", "comb"}:
            mode = "combined"
        elif pl.isdigit():
            angle_deg = int(pl)

    return {
        "sample_guess": sample,
        "scan_mode": mode or "unknown",
        "angle_deg": angle_deg,
        "corrected": corrected,
    }


def parse_plqy_raw_excel(path: Path) -> List[dict]:
    outputs: List[dict] = []
    xl = pd.ExcelFile(path)
    raw_wl = pd.read_excel(path, sheet_name="Raw Data (Wavelength)") if "Raw Data (Wavelength)" in xl.sheet_names else None
    raw_e = pd.read_excel(path, sheet_name="Raw Data") if "Raw Data" in xl.sheet_names else None
    if raw_wl is None:
        return outputs

    for i in range(0, len(raw_wl.columns), 2):
        x_col = raw_wl.columns[i]
        y_col = raw_wl.columns[i + 1]
        parsed = _parse_plqy_trace_name(str(y_col))
        source_label = str(y_col)
        sample_guess = parsed["sample_guess"]

        raw_df = pd.DataFrame({
            "wavelength_nm": pd.to_numeric(raw_wl[x_col], errors="coerce"),
            "intensity_counts": pd.to_numeric(raw_wl[y_col], errors="coerce"),
        }).dropna()

        processed = raw_df.copy()
        if raw_e is not None and i + 1 < len(raw_e.columns):
            processed["energy_eV"] = pd.to_numeric(raw_e.iloc[:, i + 1], errors="coerce")
            processed["intensity_energy_axis_counts"] = pd.to_numeric(raw_e.iloc[:, i], errors="coerce")

        outputs.append({
            "source_file": str(path),
            "parser": "plqy_raw_excel",
            "measurement_type": "plqy_raw",
            "experiment_subtype": parsed["scan_mode"],
            "source_label": source_label,
            "sample_guess": sample_guess,
            "raw_df": raw_df,
            "processed_df": processed,
            "params_df": None,
            "metadata": {
                "input_filename": path.name,
                "scan_mode": parsed["scan_mode"],
                "collection_angle_deg": parsed["angle_deg"],
                "corrected": parsed["corrected"],
            },
        })
    return outputs



def parse_uvvis_excel(path: Path) -> List[dict]:
    """Parses UV-vis absorbance / manual Tauc export like bandgaps_manual_export.xlsx."""
    outputs: List[dict] = []
    xl = pd.ExcelFile(path)
    summary = pd.read_excel(path, sheet_name="Summary") if "Summary" in xl.sheet_names else None
    raw = pd.read_excel(path, sheet_name="Raw_UVVis") if "Raw_UVVis" in xl.sheet_names else None
    if raw is None:
        return outputs

    summary_lookup = {}
    if summary is not None and "sample" in summary.columns:
        for _, row in summary.iterrows():
            summary_lookup[canonicalize_sample_key(row.get("sample", ""))] = row

    for i in range(0, len(raw.columns), 2):
        wl_col = raw.columns[i]
        abs_col = raw.columns[i + 1] if i + 1 < len(raw.columns) else None
        if abs_col is None:
            continue

        source_label = str(wl_col)
        base_label = re.sub(r"_wavelength_nm$", "", source_label, flags=re.IGNORECASE)
        base_label = re.sub(r"_absorbance_au$", "", base_label, flags=re.IGNORECASE)
        sample_guess = clean_sample_name(base_label)

        raw_df = pd.DataFrame({
            "wavelength_nm": pd.to_numeric(raw[wl_col], errors="coerce"),
            "absorbance_au": pd.to_numeric(raw[abs_col], errors="coerce"),
        }).dropna()

        processed = raw_df.copy()
        processed["energy_eV"] = 1240.0 / processed["wavelength_nm"]
        processed["tauc_direct_signal"] = (processed["absorbance_au"] * processed["energy_eV"]) ** 2
        processed["tauc_indirect_signal"] = (processed["absorbance_au"] * processed["energy_eV"]) ** 0.5

        params_df = None
        summary_row = summary_lookup.get(canonicalize_sample_key(base_label))
        if summary_row is not None:
            params_df = pd.DataFrame([summary_row])
            sample_guess = clean_sample_name(str(summary_row.get("sample", sample_guess)))

        metadata = {
            "input_filename": path.name,
            "analysis_type": "manual_tauc_export",
            "y_axis": "absorbance_au",
        }
        if params_df is not None:
            for key in ["Eg_eV", "baseline_Emin", "baseline_Emax", "edge_Emin", "edge_Emax", "R2_baseline", "R2_edge", "plot_file"]:
                if key in params_df.columns:
                    val = params_df.iloc[0][key]
                    metadata[key] = None if pd.isna(val) else (float(val) if isinstance(val, (int, float)) or str(type(val)).find("numpy")!=-1 and pd.api.types.is_number(val) else val)

        outputs.append({
            "source_file": str(path),
            "parser": "uvvis_excel",
            "measurement_type": "uv_vis_absorbance",
            "experiment_subtype": "manual_tauc_export",
            "source_label": base_label,
            "sample_guess": sample_guess,
            "raw_df": raw_df,
            "processed_df": processed,
            "params_df": params_df,
            "metadata": metadata,
        })
    return outputs


def detect_and_parse(path: Path) -> List[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        if "plqy" in path.name.lower() and "analyzed" in path.name.lower():
            return parse_plqy_analyzed_csv(path)
        return []

    if suffix in {".xlsx", ".xlsm", ".xls"}:
        xl = pd.ExcelFile(path)
        sheets = set(xl.sheet_names)
        if {"Raw Data (wavelength)", "Fit Parameters", "Fit Data"}.issubset(sheets):
            return parse_sspl_excel(path)
        if {"Raw Data", "Fitted Data", "Fit Parameters"}.issubset(sheets):
            return parse_trpl_excel(path)
        if {"Raw Data", "Raw Data (Wavelength)"}.issubset(sheets) and "plqy" in path.name.lower():
            return parse_plqy_raw_excel(path)
        if {"Summary", "Raw_UVVis"}.issubset(sheets):
            return parse_uvvis_excel(path)
    return []



def build_record_from_existing_row(out_root: Path, row: pd.Series, sample_lookup: Dict[str, dict]) -> dict:
    experiment_id = str(row.get("experiment_id", "")).strip()
    exp_dir = out_root / "experiments" / experiment_id
    metadata_path = exp_dir / "metadata.json"
    metadata_json = json.load(open(str(metadata_path), "r", encoding="utf-8")) if metadata_path.exists() else {}
    nested_metadata = metadata_json.get("metadata", {}) if isinstance(metadata_json.get("metadata", {}), dict) else {}

    sample_id = str(row.get("sample_id", metadata_json.get("sample_id", ""))).strip()
    sample_meta = sample_lookup.get(sample_id, {})
    source_file_name = str(row.get("source_file", "")).strip()
    source_file_value = str(out_root / source_file_name) if source_file_name else str(metadata_json.get("source_file", ""))

    return {
        "source_file": source_file_value,
        "parser": metadata_json.get("parser", "loaded_manifest"),
        "measurement_type": str(row.get("measurement_type", metadata_json.get("measurement_type", ""))).strip(),
        "experiment_subtype": str(row.get("experiment_subtype", metadata_json.get("experiment_subtype", ""))).strip(),
        "source_label": str(row.get("source_label", metadata_json.get("source_label", ""))).strip(),
        "sample_guess": sample_id or clean_sample_name(row.get("source_label", "")),
        "sample_id": sample_id,
        "formulation": str(row.get("formulation", sample_meta.get("formulation", metadata_json.get("formulation", "")))).strip(),
        "batch": str(row.get("batch", sample_meta.get("batch", metadata_json.get("batch", "")))).strip(),
        "user_notes": str(row.get("notes", metadata_json.get("user_notes", ""))).strip(),
        "raw_df": None,
        "processed_df": None,
        "params_df": None,
        "metadata": nested_metadata,
        "experiment_id": experiment_id,
        "existing_experiment_id": experiment_id,
        "loaded_from_database": True,
        "db_existing_paths": {
            "raw_data_path": str(row.get("raw_data_path", "")).strip(),
            "processed_data_path": str(row.get("processed_data_path", "")).strip(),
            "fit_parameters_path": str(row.get("fit_parameters_path", "")).strip(),
        },
    }


def export_records(records: List[dict], out_root: Path) -> tuple[int, int, int]:
    manifests_dir = out_root / "manifests"
    experiments_dir = out_root / "experiments"
    ensure_dir(manifests_dir)
    ensure_dir(experiments_dir)
    existing_experiments, existing_samples = load_existing_manifests(out_root)
    existing_experiments = existing_experiments.fillna("") if existing_experiments is not None else pd.DataFrame()
    existing_samples = existing_samples.fillna("") if existing_samples is not None else pd.DataFrame()
    existing_rows_by_id: Dict[str, dict] = {}
    if not existing_experiments.empty and "experiment_id" in existing_experiments.columns:
        for _, row in existing_experiments.iterrows():
            existing_rows_by_id[str(row.get("experiment_id", "")).strip()] = row.to_dict()
    counters = build_existing_counter_map(existing_experiments)
    existing_duplicate_keys = set()
    if not existing_experiments.empty:
        for _, row in existing_experiments.iterrows():
            existing_duplicate_keys.add(make_duplicate_key(row.get("sample_id", ""), row.get("measurement_type", ""), row.get("experiment_subtype", ""), row.get("source_file", ""), row.get("source_label", "")))
    final_experiment_rows: Dict[str, dict] = {k: v.copy() for k, v in existing_rows_by_id.items()}
    sample_manifest_rows: List[dict] = []
    added_count = 0
    updated_count = 0
    skipped_duplicates = 0
    for rec in records:
        sample_id = str(rec.get("sample_id", "")).strip()
        if not sample_id:
            continue
        measurement_type = str(rec.get("measurement_type", "")).strip()
        subtype = str(rec.get("experiment_subtype", "") or "default").strip()
        source_file_name = Path(str(rec.get("source_file", ""))).name
        source_label = str(rec.get("source_label", "")).strip()
        existing_experiment_id = str(rec.get("existing_experiment_id") or rec.get("experiment_id") or "").strip()
        if existing_experiment_id:
            experiment_id = existing_experiment_id
            existing_row = final_experiment_rows.get(experiment_id, existing_rows_by_id.get(experiment_id, {}))
        else:
            dup_key = make_duplicate_key(sample_id, measurement_type, subtype, source_file_name, source_label)
            if dup_key in existing_duplicate_keys:
                skipped_duplicates += 1
                continue
            key = f"{sample_id}__{measurement_type}"
            counters[key] = counters.get(key, 0) + 1
            experiment_id = f"{slugify(sample_id)}__{slugify(measurement_type)}__{counters[key]:03d}"
            existing_row = {}
            existing_duplicate_keys.add(dup_key)
        exp_dir = experiments_dir / experiment_id
        ensure_dir(exp_dir)
        raw_rel = str(existing_row.get("raw_data_path", "") or rec.get("db_existing_paths", {}).get("raw_data_path", "")).strip() or None
        processed_rel = str(existing_row.get("processed_data_path", "") or rec.get("db_existing_paths", {}).get("processed_data_path", "")).strip() or None
        params_rel = str(existing_row.get("fit_parameters_path", "") or rec.get("db_existing_paths", {}).get("fit_parameters_path", "")).strip() or None
        if rec.get("raw_df") is not None:
            raw_rel = f"experiments/{experiment_id}/raw_data.csv"; rec["raw_df"].to_csv(out_root / raw_rel, index=False)
        if rec.get("processed_df") is not None:
            processed_rel = f"experiments/{experiment_id}/processed_data.csv"; rec["processed_df"].to_csv(out_root / processed_rel, index=False)
        if rec.get("params_df") is not None:
            params_rel = f"experiments/{experiment_id}/fit_parameters.csv"; rec["params_df"].to_csv(out_root / params_rel, index=False)
        metadata_path = exp_dir / "metadata.json"
        old_meta = json.load(open(str(metadata_path), "r", encoding="utf-8")) if metadata_path.exists() else {}
        nested_metadata = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else old_meta.get("metadata", {})
        metadata = {"experiment_id": experiment_id, "sample_id": sample_id, "measurement_type": measurement_type, "experiment_subtype": subtype, "source_label": source_label, "source_file": rec.get("source_file"), "parser": rec.get("parser", old_meta.get("parser")), "formulation": rec.get("formulation", old_meta.get("formulation", "")), "batch": rec.get("batch", old_meta.get("batch", "")), "user_notes": rec.get("user_notes", old_meta.get("user_notes", "")), "metadata": nested_metadata}
        with open(metadata_path, "w", encoding="utf-8") as f: json.dump(metadata, f, indent=2)
        final_experiment_rows[experiment_id] = {"experiment_id": experiment_id, "sample_id": sample_id, "measurement_type": measurement_type, "experiment_subtype": subtype, "source_label": source_label, "source_file": source_file_name, "raw_data_path": raw_rel or "", "processed_data_path": processed_rel or "", "fit_parameters_path": params_rel or "", "formulation": rec.get("formulation", ""), "batch": rec.get("batch", ""), "notes": rec.get("user_notes", "")}
        sample_manifest_rows.append({"sample_id": sample_id, "formulation": rec.get("formulation", ""), "batch": rec.get("batch", "")})
        if existing_experiment_id: updated_count += 1
        else:
            added_count += 1; rec["existing_experiment_id"] = experiment_id; rec["experiment_id"] = experiment_id
    final_experiments_df = pd.DataFrame(list(final_experiment_rows.values()))
    if not final_experiments_df.empty:
        for col in ["experiment_id", "sample_id", "measurement_type", "experiment_subtype", "source_label", "source_file", "raw_data_path", "processed_data_path", "fit_parameters_path", "formulation", "batch", "notes"]:
            if col not in final_experiments_df.columns: final_experiments_df[col] = ""
        final_experiments_df = final_experiments_df.fillna("").sort_values(["sample_id", "measurement_type", "experiment_id"], kind="stable")
    final_experiments_df.to_csv(manifests_dir / "experiments.csv", index=False)
    merged_samples_df = merge_sample_manifests(existing_samples, sample_manifest_rows)
    merged_samples_df.to_csv(manifests_dir / "samples.csv", index=False)
    db_path = out_root / "lab_data.sqlite"; init_sqlite_database(db_path); parser_version = "v1"
    for rec in records:
        experiment_id = str(rec.get("existing_experiment_id") or rec.get("experiment_id") or "").strip()
        if not experiment_id: continue
        exp_row = final_experiment_rows.get(experiment_id, {}).copy()
        if not exp_row: continue
        exp_row["parser"] = rec.get("parser", "")
        source_path = Path(str(rec.get("source_file", "")))
        source_hash = file_sha256(source_path) if source_path.exists() and source_path.is_file() else ""
        metadata_doc = {"metadata": rec.get("metadata", {}) if isinstance(rec.get("metadata"), dict) else {}}
        upsert_experiment_to_sqlite(db_path, exp_row, {"sample_id": exp_row.get("sample_id", ""), "formulation": exp_row.get("formulation", ""), "batch": exp_row.get("batch", "")}, source_hash, parser_version, rec.get("processed_df"), rec.get("raw_df"), rec.get("params_df"), metadata_doc)
    return added_count, updated_count, skipped_duplicates


# -----------------------------
# PySide6 GUI App (cross-platform)
# -----------------------------
from PySide6 import QtCore, QtWidgets


class DataStandardizerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 850)

        self.records: List[dict] = []
        self.output_dir = str(Path.cwd() / "standardized_lab_data")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load existing database")
        self.btn_add = QtWidgets.QPushButton("Add files")
        self.btn_remove = QtWidgets.QPushButton("Remove selected")
        self.btn_autofill = QtWidgets.QPushButton("Autofill sample IDs")
        self.btn_choose_out = QtWidgets.QPushButton("Choose output folder")
        self.btn_export = QtWidgets.QPushButton("Export standardized data")
        for b in [self.btn_load, self.btn_add, self.btn_remove, self.btn_autofill, self.btn_choose_out, self.btn_export]:
            top.addWidget(b)
        root.addLayout(top)

        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(QtWidgets.QLabel("Output folder:"))
        self.out_edit = QtWidgets.QLineEdit(self.output_dir)
        out_row.addWidget(self.out_edit)
        root.addLayout(out_row)

        self.status = QtWidgets.QLabel("Ready.")
        root.addWidget(self.status)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root.addWidget(split, 1)

        left = QtWidgets.QWidget(); left_l = QtWidgets.QVBoxLayout(left)
        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["Origin", "Experiment ID", "Detected sample", "Standard sample ID", "Measurement", "Subtype", "Source file", "Source label"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        left_l.addWidget(self.table)
        split.addWidget(left)

        right = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(right)
        self.sample_edit = QtWidgets.QLineEdit()
        self.formulation_edit = QtWidgets.QLineEdit()
        self.batch_edit = QtWidgets.QLineEdit()
        self.notes_edit = QtWidgets.QLineEdit()
        form.addRow("Standard sample ID", self.sample_edit)
        form.addRow("Formulation", self.formulation_edit)
        form.addRow("Batch / replicate", self.batch_edit)
        form.addRow("Notes", self.notes_edit)
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_apply = QtWidgets.QPushButton("Apply to selected")
        self.btn_clear = QtWidgets.QPushButton("Clear fields")
        btn_row.addWidget(self.btn_apply); btn_row.addWidget(self.btn_clear)
        form.addRow(btn_row)
        info = QtWidgets.QTextEdit(); info.setReadOnly(True)
        info.setPlainText("""Load existing database:
- reads manifests/experiments.csv and samples.csv
- loads existing experiment metadata into the editor
- lets you rename sample IDs / formulation / batch / notes

Export standardized data:
- updates loaded experiments in place when experiment_id already exists
- appends new experiments for newly added source files
- keeps existing raw/processed/fitted CSVs unless new parsed data is present
- merges samples.csv rather than replacing the whole database
""")
        form.addRow(info)
        split.addWidget(right)
        split.setSizes([900, 500])

        self.btn_add.clicked.connect(self.add_files)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_autofill.clicked.connect(self.autofill_sample_ids)
        self.btn_choose_out.clicked.connect(self.choose_output_dir)
        self.btn_export.clicked.connect(self.export_all)
        self.btn_load.clicked.connect(self.load_existing_database)
        self.btn_apply.clicked.connect(self.apply_to_selected)
        self.btn_clear.clicked.connect(self.clear_form)
        self.table.itemSelectionChanged.connect(self.on_select)

    def set_status(self, text: str):
        self.status.setText(text)

    def refresh_table(self):
        active = [r for r in self.records if not r.get("_removed")]
        self.table.setRowCount(len(active))
        for r_idx, rec in enumerate(active):
            vals = [
                "database" if rec.get("loaded_from_database") else "new",
                rec.get("existing_experiment_id") or rec.get("experiment_id", ""),
                rec.get("sample_guess", ""),
                rec.get("sample_id", ""),
                rec.get("measurement_type", ""),
                rec.get("experiment_subtype", ""),
                Path(str(rec.get("source_file", ""))).name,
                rec.get("source_label", ""),
            ]
            for c, v in enumerate(vals):
                self.table.setItem(r_idx, c, QtWidgets.QTableWidgetItem(str(v)))

    def selected_record_indexes(self) -> List[int]:
        rows = sorted(set(i.row() for i in self.table.selectionModel().selectedRows()))
        active_idxs = [i for i, r in enumerate(self.records) if not r.get("_removed")]
        return [active_idxs[r] for r in rows if r < len(active_idxs)]

    def add_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Choose characterization files", "", "Supported files (*.xlsx *.xls *.csv);;All files (*.*)")
        if not paths:
            return
        added = 0; skipped = []
        for pth in paths:
            path = Path(pth)
            try:
                parsed = detect_and_parse(path)
                if not parsed:
                    skipped.append(path.name); continue
                for rec in parsed:
                    rec["sample_id"] = rec.get("sample_guess", "")
                    rec["formulation"] = ""; rec["batch"] = ""; rec["user_notes"] = ""
                    rec["existing_experiment_id"] = ""; rec["loaded_from_database"] = False; rec["db_existing_paths"] = {}
                    self.records.append(rec); added += 1
            except Exception as exc:
                skipped.append(f"{path.name} ({exc})")
        self.refresh_table()
        self.set_status(f"Session now has {len([r for r in self.records if not r.get('_removed')])} active record(s).")
        msg = f"Added {added} record(s)." + ("\n\nSkipped:\n- " + "\n- ".join(skipped) if skipped else "")
        QtWidgets.QMessageBox.information(self, APP_TITLE, msg)

    def remove_selected(self):
        idxs = self.selected_record_indexes()
        for idx in idxs: self.records[idx]["_removed"] = True
        self.refresh_table(); self.set_status(f"Marked {len(idxs)} row(s) as removed for this session.")

    def autofill_sample_ids(self):
        for rec in self.records:
            if not rec.get("_removed"): rec["sample_id"] = clean_sample_name(rec.get("sample_guess", ""))
        self.refresh_table(); self.set_status("Sample IDs autofilled from detected names.")

    def on_select(self):
        idxs = self.selected_record_indexes()
        if not idxs: return
        rec = self.records[idxs[0]]
        self.sample_edit.setText(str(rec.get("sample_id", "")))
        self.formulation_edit.setText(str(rec.get("formulation", "")))
        self.batch_edit.setText(str(rec.get("batch", "")))
        self.notes_edit.setText(str(rec.get("user_notes", "")))

    def apply_to_selected(self):
        idxs = self.selected_record_indexes()
        if not idxs:
            QtWidgets.QMessageBox.warning(self, APP_TITLE, "Select at least one row first.")
            return
        for idx in idxs:
            rec = self.records[idx]
            rec["sample_id"] = self.sample_edit.text().strip() or rec.get("sample_id", "")
            rec["formulation"] = self.formulation_edit.text().strip()
            rec["batch"] = self.batch_edit.text().strip()
            rec["user_notes"] = self.notes_edit.text().strip()
        self.refresh_table(); self.set_status(f"Updated {len(idxs)} selected record(s).")

    def clear_form(self):
        self.sample_edit.clear(); self.formulation_edit.clear(); self.batch_edit.clear(); self.notes_edit.clear()

    def choose_output_dir(self):
        out = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose export folder", self.out_edit.text().strip() or str(Path.cwd()))
        if out:
            self.out_edit.setText(out); self.output_dir = out; self.set_status(f"Output folder set to {out}")

    def load_existing_database(self):
        chosen = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose existing standardized data folder", self.out_edit.text().strip() or str(Path.cwd()))
        if not chosen: return
        out_root = Path(chosen).expanduser().resolve()
        existing_experiments, existing_samples = load_existing_manifests(out_root)
        if existing_experiments.empty:
            QtWidgets.QMessageBox.warning(self, APP_TITLE, f"Could not find existing experiments manifest in:\n{out_root}")
            return
        sample_lookup = {}
        if existing_samples is not None and not existing_samples.empty and "sample_id" in existing_samples.columns:
            for _, row in existing_samples.iterrows():
                sample_lookup[str(row.get("sample_id", "")).strip()] = row.to_dict()
        existing_ids = {str(r.get("existing_experiment_id") or r.get("experiment_id") or "").strip() for r in self.records if not r.get("_removed")}
        added = 0; skipped = 0
        for _, row in existing_experiments.fillna("").iterrows():
            exp_id = str(row.get("experiment_id", "")).strip()
            if not exp_id or exp_id in existing_ids:
                skipped += 1; continue
            self.records.append(build_record_from_existing_row(out_root, row, sample_lookup)); added += 1; existing_ids.add(exp_id)
        self.out_edit.setText(str(out_root)); self.refresh_table()
        self.set_status(f"Loaded {added} existing experiment record(s) from {out_root}.")
        QtWidgets.QMessageBox.information(self, APP_TITLE, f"Loaded {added} experiment record(s).\nSkipped duplicates: {skipped}")

    def export_all(self):
        records = [r for r in self.records if not r.get("_removed")]
        if not records:
            QtWidgets.QMessageBox.warning(self, APP_TITLE, "No records to export."); return
        if any(not str(r.get("sample_id", "")).strip() for r in records):
            QtWidgets.QMessageBox.warning(self, APP_TITLE, "Every record needs a standard sample ID before export."); return
        out_root = Path(self.out_edit.text().strip() or self.output_dir).expanduser().resolve()
        added, updated, skipped = export_records(records, out_root)
        self.refresh_table()
        self.set_status(f"Export complete to {out_root}. Added {added}, updated {updated}, skipped {skipped} duplicate new record(s).")
        QtWidgets.QMessageBox.information(self, APP_TITLE, f"Export complete.\n\nSaved to:\n{out_root}\n\nAdded: {added}\nUpdated: {updated}\nSkipped duplicates: {skipped}")


def main() -> None:
    app = QtWidgets.QApplication([])
    win = DataStandardizerApp()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
