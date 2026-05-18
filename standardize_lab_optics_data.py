from __future__ import annotations

import json
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
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



# -----------------------------
# GUI App
# -----------------------------

class DataStandardizerApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title(APP_TITLE)
        self.master.geometry("1380x800")

        self.records: List[dict] = []
        self.file_paths: List[Path] = []
        self.tree_items: Dict[str, int] = {}
        self.loaded_database_root: Optional[Path] = None

        self.sample_id_var = tk.StringVar()
        self.formulation_var = tk.StringVar()
        self.batch_var = tk.StringVar()
        self.notes_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=str(Path.cwd() / "standardized_lab_data"))
        self.status_var = tk.StringVar(value="Ready.")

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self.master, padding=10)
        top.pack(fill="x")

        ttk.Button(top, text="Load existing database", command=self.load_existing_database).pack(side="left", padx=4)
        ttk.Button(top, text="Add files", command=self.add_files).pack(side="left", padx=4)
        ttk.Button(top, text="Remove selected", command=self.remove_selected).pack(side="left", padx=4)
        ttk.Button(top, text="Autofill sample IDs", command=self.autofill_sample_ids).pack(side="left", padx=4)
        ttk.Button(top, text="Choose output folder", command=self.choose_output_dir).pack(side="left", padx=4)
        ttk.Button(top, text="Export standardized data", command=self.export_all).pack(side="left", padx=12)

        out_frame = ttk.Frame(self.master, padding=(10, 0, 10, 8))
        out_frame.pack(fill="x")
        ttk.Label(out_frame, text="Output folder:").pack(side="left")
        ttk.Entry(out_frame, textvariable=self.output_dir_var, width=100).pack(side="left", fill="x", expand=True, padx=6)

        status_frame = ttk.Frame(self.master, padding=(10, 0, 10, 6))
        status_frame.pack(fill="x")
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor="w")

        mid = ttk.Panedwindow(self.master, orient="horizontal")
        mid.pack(fill="both", expand=True, padx=10, pady=8)

        left = ttk.Frame(mid)
        right = ttk.Frame(mid)
        mid.add(left, weight=4)
        mid.add(right, weight=2)

        self.tree = ttk.Treeview(
            left,
            columns=("origin", "experiment_id", "sample_guess", "sample_id", "measurement_type", "subtype", "source_file", "source_label"),
            show="headings",
            selectmode="extended",
        )
        for col, text, width in [
            ("origin", "Origin", 90),
            ("experiment_id", "Experiment ID", 190),
            ("sample_guess", "Detected sample", 120),
            ("sample_id", "Standard sample ID", 170),
            ("measurement_type", "Measurement", 160),
            ("subtype", "Subtype", 130),
            ("source_file", "Source file", 220),
            ("source_label", "Source label", 180),
        ]:
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, stretch=True)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        form = ttk.LabelFrame(right, text="Edit selected record", padding=10)
        form.pack(fill="x", padx=4, pady=4)

        self._labeled_entry(form, "Standard sample ID", self.sample_id_var, 0)
        self._labeled_entry(form, "Formulation", self.formulation_var, 1)
        self._labeled_entry(form, "Batch / replicate", self.batch_var, 2)
        self._labeled_entry(form, "Notes", self.notes_var, 3)

        btns = ttk.Frame(form)
        btns.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(btns, text="Apply to selected", command=self.apply_to_selected).pack(side="left", padx=4)
        ttk.Button(btns, text="Clear fields", command=self.clear_form).pack(side="left", padx=4)

        info = ttk.LabelFrame(right, text="How edit mode works", padding=10)
        info.pack(fill="both", expand=True, padx=4, pady=4)
        msg = (
            "Load existing database:\n"
            "- reads manifests/experiments.csv and samples.csv\n"
            "- loads existing experiment metadata into the editor\n"
            "- lets you rename sample IDs / formulation / batch / notes\n\n"
            "Export standardized data:\n"
            "- updates loaded experiments in place when experiment_id already exists\n"
            "- appends new experiments for newly added source files\n"
            "- keeps existing raw/processed/fitted CSVs unless new parsed data is present\n"
            "- merges samples.csv rather than replacing the whole database\n"
        )
        ttk.Label(info, text=msg, justify="left").pack(anchor="w")

    def _labeled_entry(self, parent, label, var, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=var, width=40).grid(row=row, column=1, sticky="ew", pady=4)
        parent.grid_columnconfigure(1, weight=1)

    def _set_status(self, text: str):
        self.status_var.set(text)

    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Choose characterization files",
            filetypes=[
                ("Supported files", "*.xlsx *.xls *.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return

        added = 0
        skipped = []
        for p in paths:
            path = Path(p)
            try:
                records = detect_and_parse(path)
                if not records:
                    skipped.append(path.name)
                    continue
                for rec in records:
                    rec["sample_id"] = rec.get("sample_guess", "")
                    rec["formulation"] = ""
                    rec["batch"] = ""
                    rec["user_notes"] = ""
                    rec["existing_experiment_id"] = ""
                    rec["loaded_from_database"] = False
                    rec["db_existing_paths"] = {}
                    self.records.append(rec)
                    self._insert_record(len(self.records) - 1)
                    added += 1
                self.file_paths.append(path)
            except Exception as exc:
                skipped.append(f"{path.name} ({exc})")

        msg = f"Added {added} record(s)."
        if skipped:
            msg += "\n\nSkipped:\n- " + "\n- ".join(skipped)
        self._set_status(f"Session now has {len([r for r in self.records if not r.get('_removed')])} active record(s).")
        messagebox.showinfo(APP_TITLE, msg)

    def load_existing_database(self):
        start_dir = self.output_dir_var.get().strip() or str(Path.cwd())
        chosen = filedialog.askdirectory(title="Choose existing standardized data folder", initialdir=start_dir)
        if not chosen:
            return

        out_root = Path(chosen).expanduser().resolve()
        existing_experiments, existing_samples = load_existing_manifests(out_root)
        if existing_experiments.empty:
            messagebox.showwarning(APP_TITLE, f"Could not find an existing experiments manifest in:\n{out_root}")
            return

        sample_lookup = {}
        if existing_samples is not None and not existing_samples.empty and "sample_id" in existing_samples.columns:
            for _, row in existing_samples.iterrows():
                sample_lookup[str(row.get("sample_id", "")).strip()] = row.to_dict()

        added = 0
        skipped = 0
        existing_ids_in_session = {
            str(r.get("existing_experiment_id") or r.get("experiment_id") or "").strip()
            for r in self.records if not r.get("_removed")
        }

        for _, row in existing_experiments.fillna("").iterrows():
            experiment_id = str(row.get("experiment_id", "")).strip()
            if not experiment_id or experiment_id in existing_ids_in_session:
                skipped += 1
                continue

            exp_dir = out_root / "experiments" / experiment_id
            metadata_path = exp_dir / "metadata.json"
            metadata_json = json.load(open(str(metadata_path), "r", encoding="utf-8")) if metadata_path.exists() else {}
            nested_metadata = metadata_json.get("metadata", {}) if isinstance(metadata_json.get("metadata", {}), dict) else {}

            sample_id = str(row.get("sample_id", metadata_json.get("sample_id", ""))).strip()
            sample_meta = sample_lookup.get(sample_id, {})
            source_file_name = str(row.get("source_file", "")).strip()
            source_file_value = str(out_root / source_file_name) if source_file_name else str(metadata_json.get("source_file", ""))

            rec = {
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
            self.records.append(rec)
            self._insert_record(len(self.records) - 1)
            existing_ids_in_session.add(experiment_id)
            added += 1

        self.loaded_database_root = out_root
        self.output_dir_var.set(str(out_root))
        self._set_status(f"Loaded {added} existing experiment record(s) from {out_root}.")
        messagebox.showinfo(APP_TITLE, f"Loaded {added} experiment record(s) from:\n{out_root}\n\nSkipped duplicates already present in the session: {skipped}")

    def _insert_record(self, idx: int):
        rec = self.records[idx]
        origin = "database" if rec.get("loaded_from_database") else "new"
        item = self.tree.insert(
            "",
            "end",
            values=(
                origin,
                rec.get("existing_experiment_id") or rec.get("experiment_id", ""),
                rec.get("sample_guess", ""),
                rec.get("sample_id", ""),
                rec.get("measurement_type", ""),
                rec.get("experiment_subtype", ""),
                Path(str(rec.get("source_file", ""))).name,
                rec.get("source_label", ""),
            ),
        )
        self.tree_items[item] = idx

    def remove_selected(self):
        selected = list(self.tree.selection())
        if not selected:
            return
        for item in selected:
            idx = self.tree_items.pop(item)
            self.records[idx]["_removed"] = True
            self.tree.delete(item)
        self._set_status(f"Marked {len(selected)} row(s) as removed for this session. They will not be exported.")
        messagebox.showinfo(APP_TITLE, f"Removed {len(selected)} selected row(s) from the current session.")

    def autofill_sample_ids(self):
        for rec in self.records:
            if rec.get("_removed"):
                continue
            rec["sample_id"] = clean_sample_name(rec.get("sample_guess", ""))
        self.refresh_tree()
        self._set_status("Sample IDs autofilled from detected names.")
        messagebox.showinfo(APP_TITLE, "Sample IDs were autofilled from detected names. You can still edit them.")

    def refresh_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.tree_items.clear()
        for idx, rec in enumerate(self.records):
            if rec.get("_removed"):
                continue
            self._insert_record(idx)

    def on_select(self, _event=None):
        selected = self.tree.selection()
        if not selected:
            return
        idx = self.tree_items[selected[0]]
        rec = self.records[idx]
        self.sample_id_var.set(rec.get("sample_id", ""))
        self.formulation_var.set(rec.get("formulation", ""))
        self.batch_var.set(rec.get("batch", ""))
        self.notes_var.set(rec.get("user_notes", ""))

    def apply_to_selected(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning(APP_TITLE, "Select at least one row first.")
            return

        for item in selected:
            idx = self.tree_items[item]
            self.records[idx]["sample_id"] = self.sample_id_var.get().strip() or self.records[idx].get("sample_id", "")
            self.records[idx]["formulation"] = self.formulation_var.get().strip()
            self.records[idx]["batch"] = self.batch_var.get().strip()
            self.records[idx]["user_notes"] = self.notes_var.get().strip()
        self.refresh_tree()
        self._set_status(f"Updated {len(selected)} selected record(s).")

    def clear_form(self):
        self.sample_id_var.set("")
        self.formulation_var.set("")
        self.batch_var.set("")
        self.notes_var.set("")

    def choose_output_dir(self):
        out = filedialog.askdirectory(title="Choose export folder")
        if out:
            self.output_dir_var.set(out)
            self._set_status(f"Output folder set to {out}")

    def export_all(self):
        active = [r for r in self.records if not r.get("_removed")]
        if not active:
            messagebox.showwarning(APP_TITLE, "No records to export.")
            return

        missing = [r for r in active if not str(r.get("sample_id", "")).strip()]
        if missing:
            messagebox.showwarning(APP_TITLE, "Every record needs a standard sample ID before export.")
            return

        out_root = Path(self.output_dir_var.get()).expanduser().resolve()
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
                existing_duplicate_keys.add(
                    make_duplicate_key(
                        row.get("sample_id", ""),
                        row.get("measurement_type", ""),
                        row.get("experiment_subtype", ""),
                        row.get("source_file", ""),
                        row.get("source_label", ""),
                    )
                )

        final_experiment_rows: Dict[str, dict] = {k: v.copy() for k, v in existing_rows_by_id.items()}
        sample_manifest_rows: List[dict] = []
        added_count = 0
        updated_count = 0
        skipped_duplicates = 0

        for rec in active:
            sample_id = str(rec.get("sample_id", "")).strip()
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
                seq = counters[key]
                experiment_id = f"{slugify(sample_id)}__{slugify(measurement_type)}__{seq:03d}"
                existing_row = {}
                existing_duplicate_keys.add(dup_key)

            exp_dir = experiments_dir / experiment_id
            ensure_dir(exp_dir)

            raw_rel = str(existing_row.get("raw_data_path", "") or rec.get("db_existing_paths", {}).get("raw_data_path", "")).strip() or None
            processed_rel = str(existing_row.get("processed_data_path", "") or rec.get("db_existing_paths", {}).get("processed_data_path", "")).strip() or None
            params_rel = str(existing_row.get("fit_parameters_path", "") or rec.get("db_existing_paths", {}).get("fit_parameters_path", "")).strip() or None

            if rec.get("raw_df") is not None:
                raw_rel = f"experiments/{experiment_id}/raw_data.csv"
                rec["raw_df"].to_csv(out_root / raw_rel, index=False)
            if rec.get("processed_df") is not None:
                processed_rel = f"experiments/{experiment_id}/processed_data.csv"
                rec["processed_df"].to_csv(out_root / processed_rel, index=False)
            if rec.get("params_df") is not None:
                params_rel = f"experiments/{experiment_id}/fit_parameters.csv"
                rec["params_df"].to_csv(out_root / params_rel, index=False)

            metadata_path = exp_dir / "metadata.json"
            old_meta = json.load(open(str(metadata_path), "r", encoding="utf-8")) if metadata_path.exists() else {}
            nested_metadata = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else old_meta.get("metadata", {})
            metadata = {
                "experiment_id": experiment_id,
                "sample_id": sample_id,
                "measurement_type": measurement_type,
                "experiment_subtype": subtype,
                "source_label": source_label,
                "source_file": rec.get("source_file"),
                "parser": rec.get("parser", old_meta.get("parser")),
                "formulation": rec.get("formulation", old_meta.get("formulation", "")),
                "batch": rec.get("batch", old_meta.get("batch", "")),
                "user_notes": rec.get("user_notes", old_meta.get("user_notes", "")),
                "metadata": nested_metadata,
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            final_experiment_rows[experiment_id] = {
                "experiment_id": experiment_id,
                "sample_id": sample_id,
                "measurement_type": measurement_type,
                "experiment_subtype": subtype,
                "source_label": source_label,
                "source_file": source_file_name,
                "raw_data_path": raw_rel or "",
                "processed_data_path": processed_rel or "",
                "fit_parameters_path": params_rel or "",
                "formulation": rec.get("formulation", ""),
                "batch": rec.get("batch", ""),
                "notes": rec.get("user_notes", ""),
            }

            sample_manifest_rows.append({
                "sample_id": sample_id,
                "formulation": rec.get("formulation", ""),
                "batch": rec.get("batch", ""),
            })

            if existing_experiment_id:
                updated_count += 1
            else:
                added_count += 1
                rec["existing_experiment_id"] = experiment_id
                rec["experiment_id"] = experiment_id

        final_experiments_df = pd.DataFrame(list(final_experiment_rows.values()))
        if not final_experiments_df.empty:
            for col in ["experiment_id", "sample_id", "measurement_type", "experiment_subtype", "source_label", "source_file", "raw_data_path", "processed_data_path", "fit_parameters_path", "formulation", "batch", "notes"]:
                if col not in final_experiments_df.columns:
                    final_experiments_df[col] = ""
            final_experiments_df = final_experiments_df.fillna("")
            final_experiments_df = final_experiments_df.sort_values(["sample_id", "measurement_type", "experiment_id"], kind="stable")
        final_experiments_df.to_csv(manifests_dir / "experiments.csv", index=False)

        merged_samples_df = merge_sample_manifests(existing_samples, sample_manifest_rows)
        merged_samples_df.to_csv(manifests_dir / "samples.csv", index=False)

        readme = out_root / "README_standardized_format.txt"
        readme.write_text(
            "Standardized lab characterization export\n\n"
            "Top-level folders:\n"
            "- manifests/: tables that index samples and experiments\n"
            "- experiments/: one folder per experiment record\n\n"
            "Inside each experiment folder you may find:\n"
            "- raw_data.csv\n"
            "- processed_data.csv\n"
            "- fit_parameters.csv\n"
            "- metadata.json\n",
            encoding="utf-8",
        )

        self.refresh_tree()
        self._set_status(
            f"Export complete to {out_root}. Added {added_count}, updated {updated_count}, skipped {skipped_duplicates} duplicate new record(s)."
        )
        messagebox.showinfo(
            APP_TITLE,
            f"Export complete.\n\nSaved to:\n{out_root}\n\nAdded new experiments: {added_count}\nUpdated existing experiments: {updated_count}\nSkipped duplicate new records: {skipped_duplicates}",
        )


if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.25)
    except Exception:
        pass
    style = ttk.Style(root)
    for theme in ("vista", "clam", "default"):
        try:
            style.theme_use(theme)
            break
        except Exception:
            continue
    app = DataStandardizerApp(root)
    root.mainloop()
