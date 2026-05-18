from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

APP_TITLE = "Lab Characterization Data Browser"
DEFAULT_ROOT = "/data/databases"
PARENT_DIR = Path("/data/databases")



# -----------------------------
# Database discovery
# -----------------------------

def find_databases(parent: Path) -> Dict[str, str]:
    found: Dict[str, str] = {}
    if not parent.exists():
        return found

    for sub in parent.iterdir():
        if not sub.is_dir():
            continue
        samples = sub / "manifests" / "samples.csv"
        experiments = sub / "manifests" / "experiments.csv"
        if samples.exists() and experiments.exists():
            found[sub.name] = str(sub)
    return dict(sorted(found.items()))


# -----------------------------
# App setup
# -----------------------------

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Data helpers
# -----------------------------

def normalize_path(base: Path, rel: Optional[str]) -> Optional[Path]:
    if rel is None or str(rel).strip() == "" or str(rel).lower() == "nan":
        return None
    return (base / rel).resolve()


@st.cache_data(show_spinner=False)
def load_manifests(root_str: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    root = Path(root_str).expanduser().resolve()
    samples_path = root / "manifests" / "samples.csv"
    experiments_path = root / "manifests" / "experiments.csv"

    if not samples_path.exists() or not experiments_path.exists():
        return pd.DataFrame(), pd.DataFrame(), (
            f"Could not find manifest files at: {root / 'manifests'}\n"
            "Expected both samples.csv and experiments.csv."
        )

    samples = pd.read_csv(samples_path).fillna("")
    experiments = pd.read_csv(experiments_path).fillna("")

    required = {
        "experiment_id",
        "sample_id",
        "measurement_type",
        "experiment_subtype",
        "raw_data_path",
        "processed_data_path",
        "fit_parameters_path",
    }
    missing = [c for c in required if c not in experiments.columns]
    if missing:
        return pd.DataFrame(), pd.DataFrame(), (
            "experiments.csv is missing required columns: " + ", ".join(missing)
        )

    if "sample_id" not in samples.columns:
        return pd.DataFrame(), pd.DataFrame(), "samples.csv is missing the sample_id column."

    return samples, experiments, None


@st.cache_data(show_spinner=False)
def load_csv(path_str: str) -> Optional[pd.DataFrame]:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_json(path_str: str) -> Dict:
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# -----------------------------
# Plot helpers
# -----------------------------

def numeric_columns(df: Optional[pd.DataFrame]) -> List[str]:
    if df is None or df.empty:
        return []
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


PREFERRED_X = [
    "wavelength_nm", "time_ns", "energy_eV", "wavelength", "time", "x"
]
PREFERRED_Y = [
    "fit_intensity_counts",
    "intensity_counts",
    "intensity_energy_axis_counts",
    "absorbance_au",
    "tauc_direct_signal",
    "tauc_indirect_signal",
    "normalized_intensity",
    "plqy",
    "y",
]

MEASUREMENT_CONFIG = {
    "steady_state_pl": {
        "pretty_name": "Steady-state PL",
        "default_x": "wavelength_nm",
        "default_y": ["intensity_counts", "fit_intensity_counts"],
    },
    "time_resolved_pl": {
        "pretty_name": "Time-resolved PL",
        "default_x": "time_ns",
        "default_y": ["intensity_counts", "fit_intensity_counts"],
    },
    "plqy_raw": {
        "pretty_name": "PLQY Raw",
        "default_x": "wavelength_nm",
        "default_y": ["intensity_counts"],
    },
    "plqy_summary": {
        "pretty_name": "PLQY Summary",
    },
    "uv_vis_absorbance": {
        "pretty_name": "UV-vis Absorbance",
        "default_x": "wavelength_nm",
        "default_y": ["absorbance_au"],
        "compare_x": "wavelength_nm",
        "compare_y": "absorbance_au",
    },
}


def pretty_measurement_name(measurement_type: str) -> str:
    return MEASUREMENT_CONFIG.get(measurement_type, {}).get("pretty_name", measurement_type)


def guess_xy(df: Optional[pd.DataFrame], measurement_type: str = "") -> Tuple[Optional[str], Optional[str]]:
    cols = numeric_columns(df)
    if not cols:
        return None, None

    cfg = MEASUREMENT_CONFIG.get(measurement_type, {})
    x = cfg.get("default_x") if cfg.get("default_x") in cols else None
    if x is None:
        x = next((c for c in PREFERRED_X if c in cols), cols[0])

    remaining = [c for c in cols if c != x]
    if not remaining:
        return x, None

    default_y_list = [c for c in cfg.get("default_y", []) if c in remaining]
    if default_y_list:
        return x, default_y_list[0]

    y = next((c for c in PREFERRED_Y if c in remaining), remaining[0])
    return x, y


def build_line_figure(df: pd.DataFrame, x: str, y_cols: List[str], title: str, reverse_x: bool = False):
    plot_df = df[[x] + y_cols].copy()
    plot_df = plot_df.dropna(how="all")
    long_df = plot_df.melt(id_vars=x, value_vars=y_cols, var_name="series", value_name="value")
    long_df = long_df.dropna(subset=[x, "value"])
    fig = px.line(long_df, x=x, y="value", color="series", title=title)
    fig.update_layout(legend_title_text="Series", margin=dict(l=20, r=20, t=50, b=20))
    if reverse_x:
        fig.update_xaxes(autorange="reversed")
    return fig


def safe_float(value) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def apply_axis_limits(fig, x_min, x_max, y_min, y_max, reverse_x: bool = False):
    x_range = None
    if x_min is not None and x_max is not None:
        x_range = [x_max, x_min] if reverse_x else [x_min, x_max]
    elif reverse_x:
        fig.update_xaxes(autorange="reversed")

    if x_range is not None:
        fig.update_xaxes(range=x_range)
    if y_min is not None and y_max is not None:
        fig.update_yaxes(range=[y_min, y_max])
    return fig


def extract_summary_metrics(
    row: pd.Series,
    root_path: Path,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    fit_path = normalize_path(root_path, row.get("fit_parameters_path", ""))
    fit_df = load_csv(str(fit_path)) if fit_path else None
    if fit_df is not None and not fit_df.empty:
        first = fit_df.iloc[0]
        for col, value in first.items():
            val = safe_float(value)
            if val is not None:
                metrics[str(col)] = val

    metadata_path = root_path / "experiments" / str(row.get("experiment_id", "")) / "metadata.json"
    metadata = load_json(str(metadata_path))

    def _flatten(prefix: str, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_prefix = f"{prefix}.{k}" if prefix else str(k)
                yield from _flatten(new_prefix, v)
        else:
            val = safe_float(obj)
            if val is not None:
                yield prefix, val

    for key, value in _flatten("", metadata):
        metrics.setdefault(key, value)

    # Friendly aliases for common metrics
    if "metadata.Eg_eV" in metrics and "Eg_eV" not in metrics:
        metrics["Eg_eV"] = metrics["metadata.Eg_eV"]
    if "Average Tau" in metrics and "average_tau_ns" not in metrics:
        metrics["average_tau_ns"] = metrics["Average Tau"]

    return metrics


def summary_metric_candidates(compare_measurement: str, compare_pool: pd.DataFrame, root_path: Path) -> List[str]:
    candidates = set()
    for _, row in compare_pool.iterrows():
        metrics = extract_summary_metrics(row, root_path)
        candidates.update(metrics.keys())

    preferred = []
    if compare_measurement == "uv_vis_absorbance":
        preferred = ["Eg_eV", "metadata.Eg_eV", "R2", "metadata.R2"]
    elif compare_measurement == "steady_state_pl":
        preferred = [
            "Peak Center (eV)", "Peak Center (nm)",
            "Center", "Center 1", "Center_1",
            "Peak Position", "Peak Position (eV)", "Peak Position (nm)"
        ]
    elif compare_measurement == "time_resolved_pl":
        preferred = ["average_tau_ns", "Average Tau", "Tau 1", "Tau 2", "Tau 3"]

    ordered = [c for c in preferred if c in candidates]
    ordered += sorted(c for c in candidates if c not in ordered)
    return ordered


def choose_default_metric(metric_options: List[str], compare_measurement: str) -> Optional[str]:
    preferred_map = {
        "uv_vis_absorbance": ["Eg_eV", "metadata.Eg_eV"],
        "steady_state_pl": [
            "Peak Center (eV)", "Peak Center (nm)",
            "Center", "Center 1", "Center_1",
            "Peak Position", "Peak Position (eV)", "Peak Position (nm)"
        ],
        "time_resolved_pl": ["average_tau_ns", "Average Tau"],
    }
    for candidate in preferred_map.get(compare_measurement, []):
        if candidate in metric_options:
            return candidate
    return metric_options[0] if metric_options else None


# -----------------------------
# Search helpers
# -----------------------------

def build_sample_search_mask(samples_df: pd.DataFrame, query: str) -> pd.Series:
    q = query.strip().lower()
    if not q:
        return pd.Series([True] * len(samples_df), index=samples_df.index)

    mask = pd.Series(False, index=samples_df.index)
    for col in ["sample_id", "formulation", "batch"]:
        if col in samples_df.columns:
            mask = mask | samples_df[col].astype(str).str.lower().str.contains(q, na=False)
    return mask


def build_experiment_search_mask(experiments_df: pd.DataFrame, query: str) -> pd.Series:
    q = query.strip().lower()
    if not q:
        return pd.Series([True] * len(experiments_df), index=experiments_df.index)

    mask = pd.Series(False, index=experiments_df.index)
    for col in ["sample_id", "source_file", "source_label", "measurement_type", "experiment_subtype", "formulation", "batch", "notes"]:
        if col in experiments_df.columns:
            mask = mask | experiments_df[col].astype(str).str.lower().str.contains(q, na=False)
    return mask


# -----------------------------
# UI helpers
# -----------------------------

def show_metadata_table(metadata: Dict):
    if not metadata:
        st.info("No metadata.json found for this experiment.")
        return

    flat_rows = []
    for key, value in metadata.items():
        if isinstance(value, dict):
            for subk, subv in value.items():
                flat_rows.append({"field": f"{key}.{subk}", "value": subv})
        else:
            flat_rows.append({"field": key, "value": value})
    st.dataframe(pd.DataFrame(flat_rows), width='stretch', hide_index=True)


def show_uvvis_summary(metadata: Dict, fit_df: Optional[pd.DataFrame]):
    eg = None
    if isinstance(metadata, dict):
        eg = metadata.get("metadata", {}).get("Eg_eV") if "metadata" in metadata else metadata.get("Eg_eV")
    if eg is None and fit_df is not None and not fit_df.empty and "Eg_eV" in fit_df.columns:
        eg = safe_float(fit_df.iloc[0]["Eg_eV"])

    baseline = edge = None
    if isinstance(metadata, dict):
        md = metadata.get("metadata", {}) if "metadata" in metadata else metadata
        baseline = (
            md.get("baseline_Emin"),
            md.get("baseline_Emax"),
        ) if md.get("baseline_Emin") is not None and md.get("baseline_Emax") is not None else None
        edge = (
            md.get("edge_Emin"),
            md.get("edge_Emax"),
        ) if md.get("edge_Emin") is not None and md.get("edge_Emax") is not None else None

    cols = st.columns(3)
    cols[0].metric("Band gap Eg (eV)", f"{eg:.3f}" if eg is not None else "—")
    cols[1].metric(
        "Baseline fit window",
        f"{baseline[0]:.2f} to {baseline[1]:.2f} eV" if baseline else "—",
    )
    cols[2].metric(
        "Edge fit window",
        f"{edge[0]:.2f} to {edge[1]:.2f} eV" if edge else "—",
    )


# -----------------------------
# Sidebar
# -----------------------------

st.title(APP_TITLE)
st.caption("Browse standardized steady-state PL, time-resolved PL, PLQY, and UV-vis data without touching file paths.")

with st.sidebar:
    st.header("Data location")

    database_options = find_databases(PARENT_DIR)

    mode = st.radio(
        "Database selection mode",
        ["Choose from known databases", "Manual path"],
    )

    if mode == "Choose from known databases":
        if database_options:
            db_name = st.selectbox("Choose database", list(database_options.keys()))
            root_dir = database_options[db_name]
        else:
            st.warning(f"No valid databases found in {PARENT_DIR}")
            root_dir = DEFAULT_ROOT
    else:
        root_dir = st.text_input("Database path", value=st.session_state.get("root_dir", DEFAULT_ROOT))
        if database_options:
            st.caption("Known databases are still available above if you switch modes.")

    st.session_state["root_dir"] = root_dir

    samples_df, experiments_df, load_error = load_manifests(root_dir)
    root_path = Path(root_dir).expanduser().resolve()

    if load_error:
        st.error(load_error)
    else:
        st.success(f"Loaded {len(samples_df)} samples and {len(experiments_df)} experiments.")

        measurement_options = sorted(experiments_df["measurement_type"].astype(str).unique())
        selected_measurements = st.multiselect(
            "Filter measurement types",
            options=measurement_options,
            default=measurement_options,
            format_func=pretty_measurement_name,
        )

        sample_search = st.text_input(
            "Search sample or formulation",
            help="Search sample_id, formulation, batch, source label, subtype, and related fields.",
        )

if load_error:
    st.stop()

filtered_experiments = experiments_df.copy()
if selected_measurements:
    filtered_experiments = filtered_experiments[
        filtered_experiments["measurement_type"].isin(selected_measurements)
    ]

# Apply unified search across both sample metadata and experiment metadata.
if sample_search.strip():
    sample_mask = build_sample_search_mask(samples_df, sample_search)
    matching_sample_ids = set(samples_df.loc[sample_mask, "sample_id"].astype(str))

    experiment_mask = build_experiment_search_mask(filtered_experiments, sample_search)
    matching_sample_ids.update(filtered_experiments.loc[experiment_mask, "sample_id"].astype(str))

    filtered_experiments = filtered_experiments[
        filtered_experiments["sample_id"].astype(str).isin(sorted(matching_sample_ids))
    ]

visible_samples = sorted(filtered_experiments["sample_id"].astype(str).unique())

with st.sidebar:
    if sample_search.strip():
        st.caption(f"Search matched {len(visible_samples)} sample(s).")

    if visible_samples:
        default_index = 0
        previous_sample = st.session_state.get("selected_sample")
        if previous_sample in visible_samples:
            default_index = visible_samples.index(previous_sample)

        selected_sample = st.selectbox(
            "Primary sample",
            options=visible_samples,
            index=default_index,
        )
        st.session_state["selected_sample"] = selected_sample
    else:
        selected_sample = None
        st.warning("No samples match the current database, measurement, and search filters.")

sample_rows = (
    filtered_experiments[filtered_experiments["sample_id"] == selected_sample].copy()
    if selected_sample is not None
    else filtered_experiments.iloc[0:0].copy()
)


# -----------------------------
# Main layout
# -----------------------------

tab1, tab2, tab3, tab4 = st.tabs(["Sample browser", "Compare", "Experiment table", "Help"])

with tab1:
    left, right = st.columns([1.1, 2.0], gap="large")

    with left:
        st.subheader("Sample summary")
        sample_meta = samples_df[samples_df["sample_id"] == selected_sample] if selected_sample is not None else pd.DataFrame()
        if not sample_meta.empty:
            if "formulation" in sample_meta.columns:
                formulation = str(sample_meta.iloc[0].get("formulation", "")).strip()
                if formulation:
                    st.markdown(f"**Formulation:** `{formulation}`")
            st.dataframe(sample_meta, width='stretch', hide_index=True)
        else:
            st.info("This sample does not have a row in samples.csv.")

        st.subheader("Available experiments")
        if sample_rows.empty:
            st.warning("No experiments match the current filters for this sample.")
            chosen_experiment_id = None
        else:
            listing_df = sample_rows[
                ["experiment_id", "measurement_type", "experiment_subtype", "batch", "source_file", "source_label"]
            ].copy()
            listing_df["measurement_type"] = listing_df["measurement_type"].map(pretty_measurement_name)
            st.dataframe(listing_df, width='stretch', hide_index=True)
            chosen_experiment_id = st.selectbox(
                "Open experiment",
                options=sample_rows["experiment_id"].tolist(),
            )

    with right:
        st.subheader("Experiment viewer")
        if not selected_sample:
            st.info("Choose a sample in the sidebar.")
        elif not chosen_experiment_id:
            st.info("Choose an experiment on the left.")
        else:
            row = sample_rows[sample_rows["experiment_id"] == chosen_experiment_id].iloc[0]
            exp_dir = root_path / "experiments" / chosen_experiment_id
            metadata = load_json(str(exp_dir / "metadata.json"))
            raw_path = normalize_path(root_path, row.get("raw_data_path", ""))
            processed_path = normalize_path(root_path, row.get("processed_data_path", ""))
            fit_path = normalize_path(root_path, row.get("fit_parameters_path", ""))

            raw_df = load_csv(str(raw_path)) if raw_path else None
            processed_df = load_csv(str(processed_path)) if processed_path else None
            fit_df = load_csv(str(fit_path)) if fit_path else None

            measurement_type = row["measurement_type"]
            pretty_name = f"{row['sample_id']} — {pretty_measurement_name(measurement_type)}"
            if str(row.get("experiment_subtype", "")).strip():
                pretty_name += f" ({row['experiment_subtype']})"
            st.markdown(f"### {pretty_name}")

            cols = st.columns(4)
            cols[0].metric("Experiment ID", row["experiment_id"])
            cols[1].metric("Sample ID", row["sample_id"])
            cols[2].metric("Measurement", pretty_measurement_name(measurement_type))
            cols[3].metric("Subtype", row["experiment_subtype"] or "—")

            if measurement_type == "uv_vis_absorbance":
                show_uvvis_summary(metadata, fit_df)

            if processed_df is not None and not processed_df.empty:
                x_default, y_default = guess_xy(processed_df, measurement_type)
                num_cols = numeric_columns(processed_df)
                x_col = st.selectbox(
                    "X axis",
                    options=num_cols,
                    index=num_cols.index(x_default) if x_default in num_cols else 0,
                    key=f"x_{chosen_experiment_id}",
                )
                y_candidates = [c for c in num_cols if c != x_col]

                preferred_default_y = [
                    c for c in MEASUREMENT_CONFIG.get(measurement_type, {}).get("default_y", [])
                    if c in y_candidates
                ]
                if not preferred_default_y and y_default in y_candidates:
                    preferred_default_y = [y_default]

                y_cols = st.multiselect(
                    "Y series",
                    options=y_candidates,
                    default=preferred_default_y or y_candidates[:1],
                    key=f"y_{chosen_experiment_id}",
                )

                reverse_default = measurement_type == "uv_vis_absorbance" and x_col == "wavelength_nm"
                reverse_x = st.checkbox(
                    "Reverse X axis",
                    value=reverse_default,
                    key=f"reverse_{chosen_experiment_id}",
                    help="Useful for UV-vis when plotting from high wavelength to low wavelength.",
                )

                if y_cols:
                    fig = build_line_figure(
                        processed_df,
                        x_col,
                        y_cols,
                        title="Processed data",
                        reverse_x=reverse_x,
                    )
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("Choose at least one Y series to plot.")
            else:
                st.warning("No processed_data.csv found for this experiment.")

            with st.expander("Downloads", expanded=True):
                dl_cols = st.columns(3)
                if raw_path and raw_path.exists():
                    dl_cols[0].download_button(
                        "Download raw data",
                        data=raw_path.read_bytes(),
                        file_name=raw_path.name,
                        mime="text/csv",
                    )
                if processed_path and processed_path.exists():
                    dl_cols[1].download_button(
                        "Download processed data",
                        data=processed_path.read_bytes(),
                        file_name=processed_path.name,
                        mime="text/csv",
                    )
                if fit_path and fit_path.exists():
                    dl_cols[2].download_button(
                        "Download fit parameters",
                        data=fit_path.read_bytes(),
                        file_name=fit_path.name,
                        mime="text/csv",
                    )

            with st.expander("Metadata", expanded=False):
                show_metadata_table(metadata)

            with st.expander("Data preview", expanded=False):
                if processed_df is not None:
                    st.markdown("**processed_data.csv**")
                    st.dataframe(processed_df, width='stretch')
                if raw_df is not None:
                    st.markdown("**raw_data.csv**")
                    st.dataframe(raw_df, width='stretch')
                if fit_df is not None:
                    st.markdown("**fit_parameters.csv**")
                    st.dataframe(fit_df, width='stretch')

with tab2:
    st.subheader("Compare experiments")
    if filtered_experiments.empty:
        st.info("No experiments available for comparison.")
    else:
        compare_measurement = st.selectbox(
            "Measurement type to compare",
            options=sorted(filtered_experiments["measurement_type"].unique()),
            format_func=pretty_measurement_name,
        )
        compare_pool = filtered_experiments[
            filtered_experiments["measurement_type"] == compare_measurement
        ].copy()

        sample_choices = sorted(compare_pool["sample_id"].unique())
        compare_samples = st.multiselect(
            "Samples",
            options=sample_choices,
            default=sample_choices[: min(8, len(sample_choices))],
        )
        compare_pool = compare_pool[compare_pool["sample_id"].isin(compare_samples)]

        subtype_choices = sorted(
            [s for s in compare_pool["experiment_subtype"].unique() if str(s).strip()]
        )
        if subtype_choices:
            compare_subtypes = st.multiselect(
                "Subtype filter",
                options=subtype_choices,
                default=subtype_choices,
            )
            compare_pool = compare_pool[compare_pool["experiment_subtype"].isin(compare_subtypes)]

        compare_mode = st.radio(
            "Comparison view",
            ["Overlay curves", "Box plot of summary metric"],
            horizontal=True,
        )

        if compare_pool.empty:
            st.warning("Nothing matches the current comparison filters.")
        elif compare_mode == "Overlay curves":
            first_df = None
            for _, row in compare_pool.iterrows():
                p = normalize_path(root_path, row.get("processed_data_path", ""))
                df = load_csv(str(p)) if p else None
                if df is not None and not df.empty:
                    first_df = df
                    break

            if first_df is None:
                st.warning("Could not load any processed data for comparison.")
            else:
                x_default, y_default = guess_xy(first_df, compare_measurement)
                cfg = MEASUREMENT_CONFIG.get(compare_measurement, {})
                if cfg.get("compare_x") in numeric_columns(first_df):
                    x_default = cfg["compare_x"]
                if cfg.get("compare_y") in numeric_columns(first_df):
                    y_default = cfg["compare_y"]

                num_cols = numeric_columns(first_df)
                x_col = st.selectbox(
                    "Comparison X axis",
                    options=num_cols,
                    index=num_cols.index(x_default) if x_default in num_cols else 0,
                )
                y_candidates = [c for c in num_cols if c != x_col]
                y_col = st.selectbox(
                    "Comparison Y series",
                    options=y_candidates,
                    index=y_candidates.index(y_default) if y_default in y_candidates else 0,
                )
                reverse_default = compare_measurement == "uv_vis_absorbance" and x_col == "wavelength_nm"
                reverse_x = st.checkbox(
                    "Reverse X axis for comparison",
                    value=reverse_default,
                    key="compare_reverse_x",
                )

                merged = []
                skipped = []
                x_values = []
                y_values = []
                for _, row in compare_pool.iterrows():
                    p = normalize_path(root_path, row.get("processed_data_path", ""))
                    df = load_csv(str(p)) if p else None
                    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
                        skipped.append(row["experiment_id"])
                        continue
                    tmp = df[[x_col, y_col]].copy().dropna()
                    if tmp.empty:
                        skipped.append(row["experiment_id"])
                        continue
                    tmp["label"] = f"{row['sample_id']} | {row['experiment_subtype']} | {row['experiment_id']}"
                    merged.append(tmp)
                    x_values.extend(tmp[x_col].tolist())
                    y_values.extend(tmp[y_col].tolist())

                if not merged:
                    st.warning("No compatible processed_data.csv files were found for that axis selection.")
                else:
                    lim_cols = st.columns(4)
                    x_min_default = float(min(x_values)) if x_values else 0.0
                    x_max_default = float(max(x_values)) if x_values else 1.0
                    y_min_default = float(min(y_values)) if y_values else 0.0
                    y_max_default = float(max(y_values)) if y_values else 1.0

                    x_min = lim_cols[0].number_input("X min", value=x_min_default, key="cmp_x_min")
                    x_max = lim_cols[1].number_input("X max", value=x_max_default, key="cmp_x_max")
                    y_min = lim_cols[2].number_input("Y min", value=y_min_default, key="cmp_y_min")
                    y_max = lim_cols[3].number_input("Y max", value=y_max_default, key="cmp_y_max")

                    plot_df = pd.concat(merged, ignore_index=True)
                    fig = px.line(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        color="label",
                        title=f"Comparison: {pretty_measurement_name(compare_measurement)}",
                    )
                    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                    fig = apply_axis_limits(fig, x_min, x_max, y_min, y_max, reverse_x=reverse_x)
                    st.plotly_chart(fig, width='stretch')

                    if skipped:
                        st.caption(
                            "Skipped experiments that did not contain the selected columns: "
                            + ", ".join(skipped)
                        )
        else:
            metric_options = summary_metric_candidates(compare_measurement, compare_pool, root_path)
            if not metric_options:
                st.warning("No summary metrics were found in fit_parameters.csv or metadata.json for these experiments.")
            else:
                default_metric = choose_default_metric(metric_options, compare_measurement)
                metric_name = st.selectbox(
                    "Summary metric",
                    options=metric_options,
                    index=metric_options.index(default_metric) if default_metric in metric_options else 0,
                )

                rows = []
                for _, row in compare_pool.iterrows():
                    metrics = extract_summary_metrics(row, root_path)
                    if metric_name not in metrics:
                        continue
                    rows.append({
                        "sample_id": row["sample_id"],
                        "experiment_id": row["experiment_id"],
                        "experiment_subtype": row["experiment_subtype"],
                        "metric_name": metric_name,
                        "metric_value": metrics[metric_name],
                    })

                if not rows:
                    st.warning("The selected metric was not available for the current experiment set.")
                else:
                    summary_df = pd.DataFrame(rows)
                    group_by = st.selectbox(
                        "Group box plot by",
                        options=["sample_id", "experiment_subtype"],
                        index=0,
                    )
                    color_by = st.selectbox(
                        "Color points by",
                        options=["experiment_subtype", "sample_id", "None"],
                        index=0,
                    )
                    plot_points = st.checkbox("Overlay individual points", value=True)

                    y_low = float(summary_df["metric_value"].min())
                    y_high = float(summary_df["metric_value"].max())
                    if y_low == y_high:
                        y_low -= 0.5
                        y_high += 0.5
                    lim_cols = st.columns(2)
                    y_min = lim_cols[0].number_input("Metric Y min", value=y_low, key="metric_y_min")
                    y_max = lim_cols[1].number_input("Metric Y max", value=y_high, key="metric_y_max")

                    fig = px.box(
                        summary_df,
                        x=group_by,
                        y="metric_value",
                        color=None if color_by == "None" else color_by,
                        points="all" if plot_points else False,
                        hover_data=["experiment_id", "experiment_subtype"],
                        title=f"{pretty_measurement_name(compare_measurement)}: {metric_name}",
                    )
                    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), yaxis_title=metric_name)
                    fig.update_yaxes(range=[y_min, y_max])
                    st.plotly_chart(fig, width='stretch')

                    st.dataframe(summary_df, width='stretch', hide_index=True)


with tab3:
    st.subheader("Experiment manifest")
    manifest_view = filtered_experiments.copy()
    display_view = manifest_view.copy()
    display_view["measurement_type"] = display_view["measurement_type"].map(pretty_measurement_name)
    quick_search = st.text_input("Search sample ID, source file, or label")
    if quick_search.strip():
        q = quick_search.strip().lower()
        mask = (
            manifest_view["sample_id"].astype(str).str.lower().str.contains(q)
            | manifest_view["source_file"].astype(str).str.lower().str.contains(q)
            | manifest_view["source_label"].astype(str).str.lower().str.contains(q)
            | manifest_view["measurement_type"].astype(str).str.lower().str.contains(q)
        )
        display_view = display_view[mask]
    st.dataframe(display_view, width='stretch', hide_index=True)

with tab4:
    st.subheader("How to use this app")
    st.markdown(
        """
1. In the sidebar, point the app at the standardized lab data folder made by the standardization script.
2. Choose a database, then use the sample/formulation search to narrow down what you want.
3. Choose a primary sample to browse all available experiments for that sample.
4. Use **Sample browser** to inspect one experiment at a time.
5. Use **Compare** to overlay one measurement type across multiple samples.
6. Use **Experiment table** when you need a searchable list of everything in the dataset.

### Recommended installation
```bash
pip install streamlit pandas plotly
streamlit run streamlit_lab_browser_uvvis.py
```

### Notes
- The browser expects the manifest and experiment layout created by the standardization script.
- It reads `metadata.json`, `raw_data.csv`, `processed_data.csv`, and `fit_parameters.csv` when present.
- The formulation search now filters the sample list itself, not just the experiment table.
- UV-vis experiments can be plotted directly as absorbance vs wavelength or switched to energy / Tauc columns.
- The plotting controls are intentionally simple so less code-proficient lab members can still use the app comfortably.
        """
    )
