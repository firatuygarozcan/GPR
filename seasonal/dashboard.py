import json
import joblib
import base64
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import networkx as nx

from config import OUTPUT_ROOT
import config as pipeline_config
import dsse
from dsse import (
    get_reference_buses,
    load_network,
    saved_to_current_bus_map,
    season_for_day,
    validate_simbench_bus_identity,
)
from kernel_diagnostics import (
    compute_kernel_diagnostics,
    train_train_heatmap_figure,
    test_train_heatmap_figure,
    test_test_heatmap_figure,
    true_voltage_state_difference_heatmap_figure,
)

from winter_independent_evaluation import (
    MODEL_TYPE as WINTER_MODEL_TYPE,
    compute_winter_kernel_diagnostics,
    load_winter_models,
)


st.set_page_config(
    page_title="GPR DSSE Dashboard",
    page_icon="⚡",
    layout="wide",
)

CASE_LABELS = {
    "annual_max_voltage": "Maximum Voltage",
    "annual_min_voltage": "Minimum Voltage",
    "annual_max_total_load": "Maximum Total Load",
    "annual_max_net_demand": "Maximum Net Demand",
}


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

def case_label(case_name):
    case_name = str(case_name)
    return CASE_LABELS.get(case_name, case_name.replace("_", " ").title())


def noise_label(use_noise):
    if use_noise is True:
        return "Noisy"
    if use_noise is False:
        return "Clean"
    return "Unknown"


def format_metric(value, unit=""):
    return f"{float(value):.8f}{unit}"


def dashboard_dataframe(frame, **kwargs):
    """Show floating-point table values with six decimals; keep data numeric."""
    columns = {
        column: st.column_config.NumberColumn(format="%.6f")
        for column in frame.select_dtypes(include="floating").columns
    }
    columns.update(kwargs.pop("column_config", {}))
    st.dataframe(frame, column_config=columns, **kwargs)


def dashboard_plotly_chart(figure, **kwargs):
    """Send ordinary JSON arrays, including to older Streamlit Plotly clients."""
    if "width" in kwargs and "width" not in inspect.signature(st.plotly_chart).parameters:
        # Streamlit <1.51 uses use_container_width instead of width="stretch".
        if kwargs.pop("width") == "stretch":
            kwargs["use_container_width"] = True
    def plain(value):
        if isinstance(value, np.ndarray):
            return plain(value.tolist())
        if isinstance(value, dict):
            # Plotly 6+ already encodes NumPy arrays in Figure.to_dict().
            if "bdata" in value and "dtype" in value:
                array = np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
                if "shape" in value:
                    shape = value["shape"]
                    if isinstance(shape, str):
                        shape = tuple(int(size.strip()) for size in shape.split(","))
                    array = array.reshape(shape)
                return array.tolist()
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value

    st.plotly_chart(go.Figure(plain(figure.to_dict())), **kwargs)


def bus_identity_lookup(config):
    return {
        int(row["internal_bus"]): row
        for row in config.get("bus_identity", [])
        if "internal_bus" in row
    }


def bus_display_label(config, bus):
    """Display the exact saved SimBench name; never prefer the old short label."""
    bus = int(bus)
    row = bus_identity_lookup(config).get(bus, {})
    return str(row.get("simbench_name") or row.get("bus_label") or f"Internal bus {bus}")


def bus_full_name(config, bus):
    bus = int(bus)
    row = bus_identity_lookup(config).get(bus, {})
    return str(row.get("simbench_name") or row.get("bus_label") or f"Internal bus {bus}")


def frame_bus_label_lookup(frame):
    """Prefer exact SimBench names already stored in result CSVs."""
    if frame is None or frame.empty or "bus" not in frame.columns:
        return {}
    if "simbench_name" in frame.columns:
        subset = frame[["bus", "simbench_name"]].dropna().drop_duplicates("bus")
        if not subset.empty:
            return {int(row.bus): str(row.simbench_name) for row in subset.itertuples(index=False)}
    if "bus_label" in frame.columns:
        subset = frame[["bus", "bus_label"]].dropna().drop_duplicates("bus")
        return {int(row.bus): str(row.bus_label) for row in subset.itertuples(index=False)}
    return {int(bus): f"Internal bus {int(bus)}" for bus in frame["bus"].astype(int).unique()}


def live_bus_label(net, bus):
    """Exact SimBench bus name from the currently loaded network."""
    bus = int(bus)
    names = validate_simbench_bus_identity(net)
    if bus not in names:
        raise RuntimeError(f"Current network has no bus index {bus}.")
    return names[bus]


def dashboard_bus_context(net, config, evaluation_buses_override=None):
    """Resolve saved experiment roles onto the live topology by exact SimBench name.

    The current network is authoritative for topology and the reference/ext-grid
    connection. Saved evaluation metadata supplies only experiment roles
    (measured/evaluation), which are remapped through exact SimBench names.
    """
    saved_identity = config.get("bus_identity", [])
    saved_to_current = saved_to_current_bus_map(
        net,
        saved_identity,
        context="dashboard evaluation run",
    )

    def remap(values, field):
        result = set()
        for value in values or []:
            old_bus = int(value)
            if old_bus not in saved_to_current:
                raise RuntimeError(
                    f"dashboard evaluation run field {field!r} references unknown saved "
                    f"internal bus {old_bus}."
                )
            result.add(int(saved_to_current[old_bus]))
        return result

    measured_buses = remap(config.get("measured_buses", []), "measured_buses")
    saved_reference_buses = remap(config.get("reference_buses", []), "reference_buses")
    reference_buses = set(get_reference_buses(net))

    # A saved run may only be combined with this topology if it identifies the
    # same physical SimBench reference bus(es). This catches exactly the class
    # of Bus-6/Bus-52 visualization mix-ups that motivated this cleanup.
    if saved_reference_buses != reference_buses:
        saved_names = sorted(live_bus_label(net, bus) for bus in saved_reference_buses)
        live_names = sorted(live_bus_label(net, bus) for bus in reference_buses)
        raise RuntimeError(
            "Saved evaluation reference bus does not match the current SimBench "
            f"external-grid connection. Saved={saved_names}, current={live_names}."
        )

    evaluation_source = (
        evaluation_buses_override
        if evaluation_buses_override is not None
        else config.get("evaluation_buses", [])
    )
    evaluation_buses = remap(evaluation_source, "evaluation_buses")

    overlap = measured_buses & reference_buses
    if overlap:
        names = sorted(live_bus_label(net, bus) for bus in overlap)
        raise RuntimeError(
            "A reference bus is also marked as measured after name-based remapping: "
            + ", ".join(names)
        )

    return {
        "saved_to_current": saved_to_current,
        "measured_buses": measured_buses,
        "reference_buses": reference_buses,
        "evaluation_buses": evaluation_buses,
        "label_for": lambda bus: live_bus_label(net, int(bus)),
    }


def remap_saved_bus_value_dict(net, config, values):
    """Remap a saved ``bus -> value`` dictionary to live bus indices by name."""
    mapping = saved_to_current_bus_map(
        net,
        config.get("bus_identity", []),
        context="dashboard evaluation run",
    )
    result = {}
    for old_bus, value in (values or {}).items():
        old_bus = int(old_bus)
        if old_bus not in mapping:
            raise RuntimeError(f"Dashboard value references unknown saved bus {old_bus}.")
        result[int(mapping[old_bus])] = value
    return result


def bus_order_from_frame(frame):
    if frame is None or frame.empty or "bus" not in frame.columns:
        return []
    cols = ["bus"]
    if "simbench_bus_number" in frame.columns:
        cols.append("simbench_bus_number")
    unique = frame[cols].drop_duplicates("bus").copy()
    if "simbench_bus_number" in unique.columns and unique["simbench_bus_number"].notna().any():
        unique["_sort_number"] = pd.to_numeric(unique["simbench_bus_number"], errors="coerce")
        unique = unique.sort_values(["_sort_number", "bus"], na_position="last")
    else:
        unique = unique.sort_values("bus")
    return unique["bus"].astype(int).tolist()


def hide_internal_bus_columns(frame):
    """Presentation copy with one canonical exact SimBench name per bus.

    Older saved result files may contain both a shortened ``bus_label`` and the
    exact ``simbench_name``.  The exact SimBench name always wins here.
    """
    display = frame.copy()

    exact_name_pairs = [
        ("simbench_name", "bus_label"),
        ("focus_bus_name", "focus_bus_label"),
        ("max_vm_bus_name", "max_vm_bus_label"),
        ("min_vm_bus_name", "min_vm_bus_label"),
    ]
    for exact_column, label_column in exact_name_pairs:
        if exact_column in display.columns:
            if label_column in display.columns:
                display[label_column] = display[exact_column].where(
                    display[exact_column].notna(),
                    display[label_column],
                )
            else:
                display[label_column] = display[exact_column]
            display = display.drop(columns=[exact_column])

    if "bus_label" in display.columns and "bus" in display.columns:
        display = display.drop(columns=["bus"])
    if "simbench_bus_number" in display.columns and "bus_label" in display.columns:
        display = display.drop(columns=["simbench_bus_number"])
    if "focus_bus_label" in display.columns and "focus_bus" in display.columns:
        display = display.drop(columns=["focus_bus"])
    if "critical_bus_label" in display.columns and "critical_bus" in display.columns:
        display = display.drop(columns=["critical_bus"])
    return display


COLUMN_LABELS = {
    "case": "Stress Case",
    "season": "Season",
    "season_label": "Season",
    "evaluation_scope": "Evaluation Scope",
    "evaluation_scope_label": "Evaluation Scope",
    "day": "Day",
    "time": "Time",
    "time_step": "Timestep",
    "bus": "Internal Bus Index",
    "bus_label": "SimBench Bus",
    "simbench_name": "SimBench Bus Name",
    "bus_name": "SimBench Bus Name",
    "simbench_bus_number": "SimBench Bus Number",
    "focus_bus_label": "Focus SimBench Bus",
    "focus_bus_name": "Focus SimBench Bus Name",
    "critical_bus_label": "Critical SimBench Bus",
    "bus_role": "Bus Role",
    "n_cases": "Stress Cases",
    "n_predictions": "Voltage Estimates",
    "n_training_days": "Training Days",
    "n_test_days": "Test Days",
    "n_training_samples": "Training Samples",
    "n_test_timesteps": "Test Timesteps",
    "n_evaluation_buses": "Evaluation Buses",
    "n_evaluated_voltage_predictions": "Evaluated Voltage Estimates",
    "mae_pu": "MAE [p.u.]",
    "rmse_pu": "RMSE [p.u.]",
    "voltage_mae_pu": "Voltage MAE [p.u.]",
    "voltage_rmse_pu": "Voltage RMSE [p.u.]",
    "coverage_95": "95% Coverage",
    "coverage_percent": "95% Coverage [%]",
    "voltage_coverage_95": "Voltage 95% Coverage",
    "mean_interval_width_95_pu": "Mean 95% PI Width [p.u.]",
    "mean_pi_width_pu": "Mean 95% PI Width [p.u.]",
    "voltage_mean_interval_width_95_pu": "Voltage Mean 95% PI Width [p.u.]",
    "mean_sigma_pu": "Mean Predictive Sigma [p.u.]",
    "max_vm_pu": "Maximum Voltage [p.u.]",
    "min_vm_pu": "Minimum Voltage [p.u.]",
    "total_load_mw": "Total Load [MW]",
    "net_demand_mw": "Net Demand [MW]",
    "critical_bus": "Critical Bus",
    "seen_in_training": "Seen in Training",
    "constant_value": "ConstantKernel Value",
    "rbf_length_scale": "RBF Length Scale",
    "white_noise_level": "WhiteKernel Noise Level",
    "alpha": "GPR Alpha",
    "length_scale": "RBF Length Scale",
    "kernel": "Kernel",
    "parameter": "Parameter",
    "value": "Value",
}


def friendly_table(frame):
    """Return a presentation copy with readable column names and values."""
    display = frame.copy()
    if "seen_in_training" in display.columns:
        display["seen_in_training"] = display["seen_in_training"].map(
            lambda value: "Yes" if bool(value) else "No"
        )
    rename = {}
    for column in display.columns:
        if column in COLUMN_LABELS:
            rename[column] = COLUMN_LABELS[column]
        else:
            label = str(column).replace("_", " ").title()
            for old, new in [
                ("Rmse", "RMSE"),
                ("Mae", "MAE"),
                ("Gpr", "GPR"),
                ("Rbf", "RBF"),
                (" Pi ", " PI "),
                ("Vm", "Voltage"),
                ("Mw", "MW"),
                ("Mvar", "MVAr"),
                ("Pu", "p.u."),
            ]:
                label = label.replace(old, new)
            rename[column] = label
    return display.rename(columns=rename)


def add_shaded_interval(fig, bus_order, frame, lower_col, upper_col, name, bus_label_map=None):
    if frame.empty:
        return

    ci = frame[["bus", lower_col, upper_col]].copy()
    ci["bus"] = ci["bus"].astype(int)
    ci = ci.drop_duplicates("bus").set_index("bus")
    predicted = set(ci.index.astype(int))

    segments = []
    current = []
    for bus in bus_order:
        bus = int(bus)
        if bus in predicted:
            current.append(bus)
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)

    legend_used = False
    for segment in segments:
        if len(segment) == 1:
            bus = segment[0]
            fig.add_trace(
                go.Bar(
                    x=[(bus_label_map or {}).get(bus, f"Internal bus {bus}")],
                    y=[float(ci.at[bus, upper_col] - ci.at[bus, lower_col])],
                    base=[float(ci.at[bus, lower_col])],
                    width=0.35,
                    opacity=0.28,
                    marker={"color": "rgba(90, 90, 90, 0.45)"},
                    name=name,
                    legendgroup="pi",
                    showlegend=not legend_used,
                    hoverinfo="skip",
                )
            )
            legend_used = True
            continue

        upper = [float(ci.at[bus, upper_col]) for bus in segment]
        lower = [float(ci.at[bus, lower_col]) for bus in segment]
        fig.add_trace(
            go.Scatter(
                x=[(bus_label_map or {}).get(bus, f"Internal bus {bus}") for bus in segment]
                + [(bus_label_map or {}).get(bus, f"Internal bus {bus}") for bus in reversed(segment)],
                y=upper + list(reversed(lower)),
                mode="lines",
                line={"width": 0},
                fill="toself",
                fillcolor="rgba(90, 90, 90, 0.24)",
                name=name,
                legendgroup="pi",
                showlegend=not legend_used,
                hoverinfo="skip",
            )
        )
        legend_used = True

# -----------------------------------------------------------------------------
# Seasonal GPR evaluation loading and figures
# -----------------------------------------------------------------------------

PRIMARY_SCOPE = "unmeasured_only"
ALL_SCOPE = "all_predicted_buses"
SCOPE_LABELS = {
    PRIMARY_SCOPE: "Unmeasured buses only (primary)",
    ALL_SCOPE: "All predicted buses (measured + unmeasured)",
}

def find_annual_evaluation_runs():
    """Find completed four-season and independent winter runs, newest first."""
    runs = []
    required_files = [
        "run_config.json",
        "pooled_overall_metrics.csv",
        "daily_metrics.csv",
        "unmeasured_bus_metrics.csv",
        "training_days.csv",
        "seasonal_metrics.csv",
    ]
    for run_dir in Path(OUTPUT_ROOT).glob("*/seasonal_evaluation/*"):
        paths = [run_dir / name for name in required_files]
        if not all(path.exists() for path in paths):
            continue
        try:
            with open(run_dir / "run_config.json", encoding="utf-8") as file:
                config = json.load(file)
            overall_table = pd.read_csv(run_dir / "pooled_overall_metrics.csv")
            primary = overall_table
            if "evaluation_scope" in overall_table.columns:
                primary = overall_table[overall_table["evaluation_scope"] == PRIMARY_SCOPE]
            overall = primary.iloc[0]
        except (OSError, json.JSONDecodeError, pd.errors.ParserError, IndexError):
            continue
        if config.get("model_type") not in ("four_season_specific_multioutput_gprs", WINTER_MODEL_TYPE):
            continue
        optional_paths = [
            run_dir / "daily_bus_metrics.csv",
            run_dir / "all_bus_metrics.csv",
            run_dir / "voltage_kernel_hyperparameters.csv",
            run_dir / "test_days.csv",
            run_dir / "bus_role_metrics.csv",
        ]
        modified_candidates = paths + [path for path in optional_paths if path.exists()]
        modified = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size)
                         for path in modified_candidates)
        runs.append((run_dir, config, overall, modified))
    return sorted(runs, key=lambda item: max(stamp[1] for stamp in item[3]), reverse=True)


def select_annual_evaluation_run(runs):
    scenario = st.sidebar.selectbox(
        "Measurement scenario", ["Noisy", "Noise-free"], index=0,
        key="measurement_scenario",
    )
    use_noise = scenario == "Noisy"
    runs = [item for item in runs if item[1].get("use_measurement_noise") is use_noise]
    if not runs and not use_noise:
        st.sidebar.error(
            "No noise-free seasonal GPR evaluation found. "
            "Run `python seasonal_evaluation_noise_free.py` first."
        )
        st.stop()
    if not runs:
        st.sidebar.error(
            "No seasonal GPR evaluation found. Run `python seasonal_evaluation.py` first."
        )
        st.stop()
    st.sidebar.subheader("Seasonal experiment")
    networks = sorted(
        {item[1].get("network", item[0].parent.parent.name) for item in runs}
    )
    selected_network = st.sidebar.selectbox("Network", networks, key="annual_network")
    candidates = [
        item for item in runs
        if item[1].get("network", item[0].parent.parent.name) == selected_network
    ]
    labels = {}
    for run_dir, config, overall, _modified in candidates:
        n_train = int(config.get("n_training_days_per_season", 30))
        n_test_total = int(overall.get("n_test_days", 0))
        noise = noise_label(config.get("use_measurement_noise"))
        description = (
            f"Winter · {int(config['n_models'])} independent GPRs"
            if config.get("model_type") == WINTER_MODEL_TYPE else "4 seasonal GPRs"
        )
        labels[str(run_dir)] = (
            f"{description} · {config.get('kernel', 'Unknown kernel')} · "
            f"{n_train} train/season · {n_test_total} total test days · "
            f"{noise} · {run_dir.name}"
        )
    options = [str(item[0]) for item in candidates]
    selected_path = st.sidebar.selectbox(
        "Evaluation run", options, format_func=lambda path: labels[path], key="annual_run"
    )
    return next(item for item in candidates if str(item[0]) == selected_path)


@st.cache_data(show_spinner=False)
def load_annual_evaluation(path_string, modified=None):
    run_dir = Path(path_string)
    with open(run_dir / "run_config.json", encoding="utf-8") as file:
        config = json.load(file)
    overall_table = pd.read_csv(run_dir / "pooled_overall_metrics.csv")
    seasonal = pd.read_csv(run_dir / "seasonal_metrics.csv")
    daily = pd.read_csv(run_dir / "daily_metrics.csv")
    all_bus_path = run_dir / "all_bus_metrics.csv"
    if all_bus_path.exists():
        all_buses = pd.read_csv(all_bus_path)
    else:
        all_buses = pd.read_csv(run_dir / "unmeasured_bus_metrics.csv")
        if "bus_role" not in all_buses.columns:
            all_buses["bus_role"] = "unmeasured"
    training_days = pd.read_csv(run_dir / "training_days.csv")
    test_days_path = run_dir / "test_days.csv"
    test_days = pd.read_csv(test_days_path) if test_days_path.exists() else pd.DataFrame()
    daily_bus_path = run_dir / "daily_bus_metrics.csv"
    daily_bus = pd.read_csv(daily_bus_path) if daily_bus_path.exists() else pd.DataFrame()
    kernel_path = run_dir / "voltage_kernel_hyperparameters.csv"
    kernel = pd.read_csv(kernel_path) if kernel_path.exists() else pd.DataFrame()
    primary = overall_table
    if "evaluation_scope" in primary.columns:
        primary = primary[primary["evaluation_scope"] == PRIMARY_SCOPE]
    return {
        "run_dir": run_dir,
        "network_dir": run_dir.parent.parent,
        "config": config,
        "overall_table": overall_table,
        "overall": primary.iloc[0],
        "seasonal_all_scopes": seasonal,
        "seasonal": seasonal[seasonal.get("evaluation_scope", PRIMARY_SCOPE) == PRIMARY_SCOPE].copy()
            if "evaluation_scope" in seasonal.columns else seasonal.copy(),
        "daily_all_scopes": daily,
        "daily": daily[daily.get("evaluation_scope", PRIMARY_SCOPE) == PRIMARY_SCOPE].copy()
            if "evaluation_scope" in daily.columns else daily.copy(),
        "all_buses": all_buses.sort_values("bus").reset_index(drop=True),
        "buses": all_buses[all_buses.get("bus_role", "unmeasured") == "unmeasured"].copy().sort_values("bus").reset_index(drop=True)
            if "bus_role" in all_buses.columns else all_buses.sort_values("bus").reset_index(drop=True),
        "daily_bus_all": daily_bus.sort_values(["day", "bus"]).reset_index(drop=True) if not daily_bus.empty else daily_bus,
        "daily_bus": daily_bus[daily_bus.get("bus_role", "unmeasured") == "unmeasured"].copy().sort_values(["day", "bus"]).reset_index(drop=True)
            if (not daily_bus.empty and "bus_role" in daily_bus.columns) else daily_bus,
        "training_days": sorted(training_days["training_day"].astype(int).tolist()),
        "training_days_frame": training_days,
        "test_days_frame": test_days,
        "kernel": kernel,
    }


def seasonal_bus_role_metrics(daily_bus):
    """Pool each season/role over the saved bus-timestep estimates."""
    rows = []
    for (season, role), frame in daily_bus.groupby(["season", "bus_role"], sort=False):
        counts = frame["n_timesteps"].to_numpy(float)
        rows.append({
            "season": season, "bus_role": role, "n_buses": int(frame.bus.nunique()),
            "n_predictions": int(counts.sum()),
            "mae_pu": float(np.average(frame.mae_pu, weights=counts)),
            "rmse_pu": float(np.sqrt(np.average(frame.rmse_pu ** 2, weights=counts))),
            "coverage_95": float(np.average(frame.coverage_95, weights=counts)),
            "mean_interval_width_95_pu": float(np.average(frame.mean_interval_width_95_pu, weights=counts)),
        })
    return pd.DataFrame(rows)


def evaluation_view(base, scope):
    """Return a lightweight scope-filtered view without recomputing predictions."""
    view = dict(base)
    overall = base["overall_table"]
    if "evaluation_scope" in overall.columns:
        overall = overall[overall["evaluation_scope"] == scope]
    if overall.empty:
        raise RuntimeError(f"No saved metrics for evaluation scope '{scope}'.")
    view["overall"] = overall.iloc[0]

    seasonal = base["seasonal_all_scopes"]
    if "evaluation_scope" in seasonal.columns:
        seasonal = seasonal[seasonal["evaluation_scope"] == scope]
    view["seasonal"] = seasonal.copy().reset_index(drop=True)

    daily = base["daily_all_scopes"]
    if "evaluation_scope" in daily.columns:
        daily = daily[daily["evaluation_scope"] == scope]
    view["daily"] = daily.copy().sort_values("day").reset_index(drop=True)

    buses = base["all_buses"].copy()
    daily_bus = base["daily_bus_all"].copy()
    if scope == PRIMARY_SCOPE:
        if "bus_role" in buses.columns:
            buses = buses[buses["bus_role"] == "unmeasured"]
        if not daily_bus.empty and "bus_role" in daily_bus.columns:
            daily_bus = daily_bus[daily_bus["bus_role"] == "unmeasured"]
    view["buses"] = buses.sort_values("bus").reset_index(drop=True)
    view["daily_bus"] = (
        daily_bus.sort_values(["day", "bus"]).reset_index(drop=True)
        if not daily_bus.empty else daily_bus
    )
    view["evaluation_scope"] = scope
    return view


def filter_detail_scope(detail, scope):
    if detail.empty or scope == ALL_SCOPE or "bus_role" not in detail.columns:
        return detail.copy()
    return detail[detail["bus_role"] == "unmeasured"].copy()


def daily_metric_figure(annual_eval, metric, title, y_title, percent=False, nominal=None):
    daily = annual_eval["daily"]
    training_days = annual_eval["training_days"]

    y = daily[metric].astype(float)
    if percent:
        y = 100.0 * y

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["day"],
            y=y,
            mode="lines",
            name=title,
            hovertemplate=(
                "Day %{x}<br>" + y_title + " %{y:.8f}<extra></extra>"
                if not percent
                else "Day %{x}<br>" + y_title + " %{y:.2f}%<extra></extra>"
            ),
        )
    )

    if nominal is not None:
        fig.add_hline(
            y=nominal,
            line_dash="dash",
            annotation_text=f"Nominal {nominal:.0f}%",
            annotation_position="top right",
        )

    for day in training_days:
        fig.add_vline(x=day, line_width=1, opacity=0.10)

    fig.update_layout(
        title=title,
        xaxis_title="Day of year",
        yaxis_title=y_title,
        height=410,
        margin={"l": 60, "r": 25, "t": 65, "b": 55},
        showlegend=False,
    )
    return fig


def bus_metric_figure(annual_eval, metric, title, y_title, percent=False):
    buses = annual_eval["buses"].copy()
    order = bus_order_from_frame(buses)
    buses = buses.set_index("bus").reindex(order).reset_index()
    y = buses[metric].astype(float)
    if percent:
        y = 100.0 * y
    labels = [
        (
            row.get("simbench_name")
            if pd.notna(row.get("simbench_name"))
            else row.get("bus_label")
            if pd.notna(row.get("bus_label"))
            else bus_display_label(annual_eval["config"], row["bus"])
        )
        for _, row in buses.iterrows()
    ]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=y,
            hovertemplate=(
                "%{x}<br>" + y_title + " %{y:.8f}<extra></extra>"
                if not percent
                else "%{x}<br>" + y_title + " %{y:.2f}%<extra></extra>"
            ),
        )
    )
    if percent and metric == "coverage_95":
        fig.add_hline(y=95.0, line_dash="dash", annotation_text="Nominal 95%")

    fig.update_layout(
        title=title,
        xaxis_title="SimBench bus",
        yaxis_title=y_title,
        height=420,
        margin={"l": 60, "r": 25, "t": 65, "b": 55},
    )
    fig.update_xaxes(type="category")
    return fig



def annual_bus_day_heatmap(annual_eval, metric, day_view="All test days"):
    """Bus x day heatmap built from each day's actual SimBench timesteps."""
    frame = annual_eval["daily_bus"].copy()
    if frame.empty:
        return None

    metric_config = {
        "RMSE": {
            "column": "rmse_pu",
            "title": "Bus × Day RMSE",
            "colorbar": "RMSE (p.u.)",
            "hover": "%{z:.8f} p.u.",
            "colorscale": "Viridis",
            "reverse": False,
        },
        "95% Coverage": {
            "column": "coverage_95",
            "title": "Bus × Day 95% Coverage",
            "colorbar": "Coverage (%)",
            "hover": "%{z:.2f}%",
            "colorscale": "RdYlGn",
            "reverse": False,
        },
        "PI Width": {
            "column": "mean_interval_width_95_pu",
            "title": "Bus × Day Mean 95% PI Width",
            "colorbar": "Mean PI width (p.u.)",
            "hover": "%{z:.8f} p.u.",
            "colorscale": "Viridis",
            "reverse": False,
        },
    }
    cfg = metric_config[metric]
    column = cfg["column"]

    if metric == "95% Coverage":
        frame["display_value"] = 100.0 * frame[column].astype(float)
    else:
        frame["display_value"] = frame[column].astype(float)

    chronological = day_view == "All test days"
    if chronological:
        day_order = sorted(frame["day"].astype(int).unique().tolist())
    else:
        daily_summary = annual_eval["daily"].copy()
        if metric == "RMSE":
            day_order = daily_summary.nlargest(30, "rmse_pu")["day"].astype(int).tolist()
        elif metric == "95% Coverage":
            day_order = daily_summary.nsmallest(30, "coverage_95")["day"].astype(int).tolist()
        else:
            day_order = daily_summary.nlargest(
                30, "mean_interval_width_95_pu"
            )["day"].astype(int).tolist()
        frame = frame[frame["day"].astype(int).isin(day_order)].copy()

    bus_order = bus_order_from_frame(frame)
    label_lookup = frame_bus_label_lookup(frame)
    pivot = frame.pivot(index="bus", columns="day", values="display_value")
    pivot = pivot.reindex(index=bus_order, columns=day_order)

    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(float),
            x=day_order,
            y=[label_lookup.get(bus, bus_display_label(annual_eval["config"], bus)) for bus in bus_order],
            colorscale=cfg["colorscale"],
            reversescale=cfg["reverse"],
            colorbar={"title": cfg["colorbar"]},
            hovertemplate=(
                "Day %{x}<br>%{y}<br>" + metric + " " + cfg["hover"] + "<extra></extra>"
            ),
        )
    )

    # Training days are not evaluated and therefore do not appear as cells.
    # On the chronological view, mark their position lightly to make the
    # train/test split visible without adding another chart.
    if chronological:
        for day in annual_eval["training_days"]:
            fig.add_vline(x=int(day), line_width=1, opacity=0.16)

    fig.update_layout(
        title=(
            cfg["title"]
            if chronological
            else cfg["title"] + " — 30 most difficult test days"
        ),
        xaxis_title=(
            "Day of year (vertical markers = training days)"
            if chronological
            else "Test day (ordered by selected difficulty metric)"
        ),
        yaxis_title="SimBench bus",
        height=max(560, 19 * len(bus_order) + 170),
        margin={"l": 90, "r": 35, "t": 75, "b": 65},
    )
    if not chronological:
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=day_order)
    return fig


def selected_run_file(run_path, relative_path):
    """A selected run cannot silently borrow another scenario's result file."""
    root = Path(run_path).resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Result file is outside the selected run: {relative_path}")
    return path


def day_detail_signature(run_path, day):
    root = Path(run_path)
    with (root / "run_config.json").open(encoding="utf-8") as file:
        cfg = json.load(file)
    season = season_for_day(day)
    paths = [root / "run_config.json", root / "daily_metrics.csv",
             selected_run_file(root, cfg.get("daily_diagnostics_directory", "daily_diagnostics")) / f"day_{int(day):03d}.csv.gz",
             selected_run_file(root, cfg.get("trained_model_files", {}).get(season, f"trained_gpr_{season}.joblib")),
             root / "daily_bus_metrics.csv"]
    paths.extend(selected_run_file(root, name) for name in cfg.get("independent_model_files", {}).values())
    paths.extend(dsse.ground_truth_paths(cfg["network"]).values())
    noise = tuple(getattr(pipeline_config, name) for name in
                  ["TEST_NOISE_SEED", "P_RELATIVE_STD", "Q_RELATIVE_STD", "V_RELATIVE_STD", "ANGLE_STD_DEGREE"])
    return (tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size)
                  for path in paths if path.exists()), noise)


@st.cache_data(show_spinner=False)
def dashboard_timeline(network):
    """Authoritative local-clock labels, keyed by unique SimBench row position."""
    timeline = dsse.build_simbench_timeline(load_dashboard_network(network))
    dsse.validate_simbench_timeline(timeline)
    return timeline


def validate_day_grid(frame, day, cfg):
    """Require every saved output bus at every physical step of the selected day."""
    frame = frame.copy()
    for column in ["bus", "time_step"]:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Daily {column} identifiers must be finite integers.")
        frame[column] = values.astype(int)
    if frame.empty or frame.duplicated(["bus", "time_step"]).any():
        raise ValueError("Daily data is empty or contains duplicate bus/timestep pairs.")
    expected = dashboard_timeline(cfg["network"])
    expected = expected[expected.day_of_year.eq(int(day))].set_index("simbench_step")
    if expected.empty:
        raise ValueError("Selected day is not in the SimBench timeline.")
    predicted = set(map(int, cfg["prediction_buses"]))
    identity = bus_identity_lookup(cfg)
    names = [identity.get(bus, {}).get("simbench_name") for bus in predicted]
    if (not predicted or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("Saved output buses require complete, unique SimBench identity metadata.")
    if predicted & set(map(int, cfg.get("reference_buses", []))):
        raise ValueError("Reference buses cannot be GPR output buses.")
    if set(frame.bus) != predicted:
        raise ValueError("Daily data does not contain exactly the selected run's output buses.")
    steps = set(expected.index)
    if any(set(part.time_step) != steps for _, part in frame.groupby("bus")):
        raise ValueError("Daily data is missing or has unexpected SimBench timesteps for an output bus.")
    aligned = expected.loc[frame.time_step]
    canonical = {
        "simbench_step": frame.time_step.to_numpy(),
        "timestamp": aligned.timestamp.to_numpy(),
        "date": aligned.date.to_numpy(),
        "day_of_year": aligned.day_of_year.to_numpy(),
        "step_in_day": aligned.step_in_day.to_numpy(),
        "time": aligned.timestamp.str[-5:].to_numpy(),
    }
    for column, values in canonical.items():
        if column in frame and not np.array_equal(frame[column].to_numpy(), values):
            raise ValueError(f"Daily {column} does not match the selected SimBench day/timestep.")
        frame[column] = values
    return frame


def validate_day_detail(frame, day, cfg):
    required = ["time_step", "time", "bus", "ground_truth_vm_pu", "predicted_vm_pu",
                "lower_95_vm_pu", "upper_95_vm_pu"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Daily state-estimation data is missing columns: {missing}")
    frame = validate_day_grid(frame, day, cfg)
    numbers = frame[["ground_truth_vm_pu", "predicted_vm_pu", "lower_95_vm_pu", "upper_95_vm_pu"]].to_numpy(float)
    if not np.isfinite(numbers).all() or np.any(numbers[:, 2] > numbers[:, 3]):
        raise ValueError("Daily voltage estimates or interval bounds are invalid.")
    if "sigma_vm_pu" in frame:
        sigma = frame.sigma_vm_pu.to_numpy(float)
        multiplier = float(cfg.get("gaussian_95_multiplier", dsse.GAUSSIAN_95_MULTIPLIER))
        if (not np.isfinite(sigma).all() or np.any(sigma < 0)
                or not np.allclose(numbers[:, 2], numbers[:, 1] - multiplier*sigma, rtol=1e-7, atol=1e-10)
                or not np.allclose(numbers[:, 3], numbers[:, 1] + multiplier*sigma, rtol=1e-7, atol=1e-10)):
            raise ValueError("95% interval bounds disagree with the saved GPR standard deviation.")
    if "day_of_year" in frame and not frame.day_of_year.astype(int).eq(int(day)).all():
        raise ValueError("Daily data belongs to a different day.")
    if "season" in frame and not frame.season.astype(str).str.lower().eq(season_for_day(day)).all():
        raise ValueError("Daily data belongs to a different seasonal model.")
    identity = bus_identity_lookup(cfg)
    measured = set(map(int, cfg.get("measured_buses", [])))
    predicted = set(map(int, cfg.get("prediction_buses", [])))
    for bus, part in frame.groupby("bus"):
        bus = int(bus)
        if predicted and bus not in predicted:
            raise ValueError(f"Bus {bus} is not an output of the selected run.")
        name = identity.get(bus, {}).get("simbench_name")
        if name and "simbench_name" in part and not part.simbench_name.eq(name).all():
            raise ValueError(f"SimBench name/index mismatch for bus {bus}.")
        if name:
            frame.loc[part.index, "simbench_name"] = name
            frame.loc[part.index, "bus_label"] = name
            frame.loc[part.index, "simbench_bus_number"] = identity[bus].get("simbench_bus_number")
        expected_role = "measured" if bus in measured else "unmeasured"
        if "bus_role" in part and not part.bus_role.eq(expected_role).all():
            raise ValueError(f"Measurement role mismatch for bus {bus}.")
        frame.loc[part.index, "bus_role"] = expected_role
    # Derive heatmap values from exactly the traces displayed above it. This
    # also avoids treating numeric CSV booleans (0/1) as false strings.
    derived = {
        "absolute_error_vm_pu": np.abs(numbers[:, 1] - numbers[:, 0]),
        "pi_width_pu": numbers[:, 3] - numbers[:, 2],
        "covered_95": (numbers[:, 0] >= numbers[:, 2]) & (numbers[:, 0] <= numbers[:, 3]),
    }
    for column in ["absolute_error_vm_pu", "pi_width_pu"]:
        if column in frame and not np.allclose(frame[column].to_numpy(float), derived[column], rtol=1e-7, atol=1e-10):
            raise ValueError(f"Saved {column} disagrees with the voltage traces.")
    for column, values in derived.items():
        frame[column] = values
    return frame.sort_values(["bus", "time_step"]).reset_index(drop=True)


def verify_day_metrics(frame, run_path, day):
    daily = pd.read_csv(Path(run_path) / "daily_metrics.csv")
    checks = daily[daily.day.astype(int).eq(int(day))]
    if checks.empty:
        raise ValueError("This day is not an evaluated test day of the selected run.")
    for _, row in checks.iterrows():
        part = filter_detail_scope(frame, row.get("evaluation_scope", PRIMARY_SCOPE))
        error = part.predicted_vm_pu - part.ground_truth_vm_pu
        values = {"mae_pu": np.abs(error).mean(), "rmse_pu": np.sqrt(np.mean(error**2)),
                  "coverage_95": part.covered_95.mean(), "mean_interval_width_95_pu": part.pi_width_pu.mean()}
        if any(not np.isclose(values[key], float(row[key]), rtol=1e-6, atol=1e-10) for key in values):
            raise ValueError("Daily state estimates do not match the selected run's saved metrics. Restore this run's daily CSV or its original noise settings; incompatible data will not be plotted.")
    per_bus_path = Path(run_path) / "daily_bus_metrics.csv"
    if per_bus_path.exists():
        per_bus = pd.read_csv(per_bus_path)
        for _, row in per_bus[per_bus.day.astype(int).eq(int(day))].iterrows():
            part = frame[frame.bus.astype(int).eq(int(row.bus))]
            error = part.predicted_vm_pu - part.ground_truth_vm_pu
            values = {"mae_pu": np.abs(error).mean(), "rmse_pu": np.sqrt(np.mean(error**2)),
                      "coverage_95": part.covered_95.mean(), "mean_interval_width_95_pu": part.pi_width_pu.mean()}
            if ("n_timesteps" in row and len(part) != int(row.n_timesteps)) or any(
                    not np.isclose(values[key], float(row[key]), rtol=1e-6, atol=1e-10) for key in values):
                raise ValueError(f"State estimates do not match the saved metrics for bus {int(row.bus)}.")


def recover_day_detail(run_path, day, cfg):
    """Missing CSV: use the saved seasonal GPR; never fit or run power flow."""
    season = season_for_day(day)
    model_path = selected_run_file(run_path, cfg.get("trained_model_files", {}).get(season, f"trained_gpr_{season}.joblib"))
    if not model_path.exists():
        raise ValueError(f"No daily CSV or saved {season} GPR in this run. Generate this scenario's seasonal evaluation first.")
    network = cfg["network"]
    net = load_dashboard_network(network)
    dsse.assert_saved_bus_identity_compatible(net, cfg.get("bus_identity", []),
                                            context="dashboard saved GPR", require_internal_index_match=True)
    cache = dsse.ground_truth_paths(network)
    if cache["directory"].resolve().parent.parent != Path(run_path).resolve().parent.parent:
        raise ValueError("The saved run and configured ground-truth cache use different networks/roots.")
    timeline = dsse.build_simbench_timeline(net)
    if dsse._cached_ground_truth_report(net, network, dsse.validate_simbench_timeline(timeline)) is None:
        raise ValueError("A validated existing ground-truth cache is required; dashboard will not regenerate it.")
    saved_identity = pd.read_csv(cache["bus_identity"])
    canonical = pd.DataFrame(dsse.bus_identity_records(net))
    columns = ["internal_bus", "simbench_bus_number", "simbench_name", "bus_label"]
    if not saved_identity.reindex(columns=columns).equals(canonical[columns]):
        raise ValueError("Ground-truth bus identity is incompatible; dashboard will not rewrite it.")
    truth = dsse.load_ground_truth(net, network, [int(day)])
    X, Y, meta, targets, _ = dsse.build_voltage_dataset(
        truth, cfg["measured_buses"], [int(b) for b in net.bus.index], cfg["reference_buses"])
    if targets != list(map(int, cfg["prediction_buses"])):
        raise ValueError("Saved GPR target ordering differs from the ground truth.")
    if cfg.get("use_measurement_noise") is True:
        # Single-day equivalent of the pipeline's add_noise_by_day; noise-free
        # runs use clean X without changing shared module configuration.
        X = dsse.add_measurement_noise(X, dsse.day_noise_seed(pipeline_config.TEST_NOISE_SEED, int(day)))
    model = (
        load_winter_models(run_path, cfg)
        if cfg.get("model_type") == WINTER_MODEL_TYPE else joblib.load(model_path)
    )
    if list(model.target_columns) != list(Y.columns):
        raise ValueError("Saved GPR output columns differ from the selected run.")
    pred, sigma = model.predict(X)
    records = []
    # In-memory equivalent of seasonal_evaluation.save_daily_diagnostics.
    for bus in targets:
        column = f"bus_{bus}_vm_pu"
        true, mean, std = [f[column].to_numpy(float) for f in [Y, pred, sigma]]
        records.append(pd.DataFrame({
            "season": season, "time_step": Y.index.to_numpy(), "time": meta.time.astype(str).to_numpy(),
            "day_of_year": int(day), "bus": bus, "ground_truth_vm_pu": true,
            "predicted_vm_pu": mean, "sigma_vm_pu": std,
            "lower_95_vm_pu": mean - dsse.GAUSSIAN_95_MULTIPLIER * std,
            "upper_95_vm_pu": mean + dsse.GAUSSIAN_95_MULTIPLIER * std,
        }))
    frame = validate_day_detail(pd.concat(records, ignore_index=True), day, cfg)
    # Old runs did not save noise parameters/seeds. Accept reconstruction only
    # when its metrics reproduce the saved run, not merely the current config.
    verify_day_metrics(frame, run_path, day)
    frame.attrs["detail_source"] = "Verified prediction from saved seasonal GPR (daily CSV missing)"
    return frame


@st.cache_data(show_spinner=False)
def load_annual_day_detail(run_path, day, modified=None):
    with (Path(run_path) / "run_config.json").open(encoding="utf-8") as file:
        cfg = json.load(file)
    directory = selected_run_file(run_path, cfg.get("daily_diagnostics_directory", "daily_diagnostics"))
    path = directory / f"day_{int(day):03d}.csv.gz"
    if not path.exists():
        return recover_day_detail(run_path, day, cfg)
    frame = validate_day_detail(pd.read_csv(path), day, cfg)
    verify_day_metrics(frame, run_path, day)
    frame.attrs["detail_source"] = f"Saved daily diagnostics: {path.name}"
    return frame


def annual_day_heatmap_figure(detail, metric):
    """15-minute bus x timestep drill-down for one annual-evaluation day."""
    frame = detail.copy()
    if frame.empty:
        return None

    if metric == "Absolute Error":
        column = "absolute_error_vm_pu"
        title = "15-Minute Absolute Voltage Error"
        colorbar = "|error| (p.u.)"
        hover_value = "%{z:.8f} p.u."
        colorscale = "Viridis"
    elif metric == "95% PI Width":
        column = "pi_width_pu"
        title = "15-Minute 95% Prediction-Interval Width"
        colorbar = "PI width (p.u.)"
        hover_value = "%{z:.8f} p.u."
        colorscale = "Blues"
    elif metric == "Coverage Miss":
        column = "missed_95"
        frame[column] = (~frame["covered_95"].astype(bool)).astype(int)
        title = "15-Minute 95% Prediction-Interval Misses"
        colorbar = "Miss"
        hover_value = "%{z:.0f}"
        colorscale = [[0.0, "#eef3f8"], [0.5, "#eef3f8"],
                      [0.5, "#b2182b"], [1.0, "#b2182b"]]
    else:
        raise ValueError(f"Unknown daily heatmap metric: {metric}")

    bus_order = bus_order_from_frame(frame)
    label_lookup = frame_bus_label_lookup(frame)

    # Use the unique SimBench step as the x-key.  Local clock labels are not
    # unique on the autumn DST transition (02:00..02:45 occur twice), so pivoting
    # directly on `time` would merge/fail on exactly the day we must preserve.
    step_meta = (
        frame[["time_step", "time", "timestamp", "date"]]
        .drop_duplicates("time_step")
        .sort_values("time_step")
    )
    step_order = step_meta["time_step"].astype(int).tolist()
    time_labels = step_meta["time"].astype(str).tolist()
    pivot = frame.pivot(index="bus", columns="time_step", values=column)
    pivot = pivot.reindex(index=bus_order, columns=step_order)
    ordered = frame.set_index(["bus", "time_step"]).loc[
        pd.MultiIndex.from_product([bus_order, step_order])]
    hover_columns = ["timestamp", "bus_role", "ground_truth_vm_pu", "predicted_vm_pu",
                     "lower_95_vm_pu", "upper_95_vm_pu"]
    custom = ordered[hover_columns].to_numpy(object).reshape(len(bus_order), len(step_order), -1).tolist()
    colorbar_config = {"title": colorbar}
    if metric == "Coverage Miss":
        colorbar_config.update(tickvals=[0, 1], ticktext=["Inside PI (0)", "Outside PI (1)"])

    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(float).tolist(),
            x=step_order,
            y=[label_lookup.get(bus, f"Internal bus {bus}") for bus in bus_order],
            customdata=custom,
            colorscale=colorscale,
            zmin=0 if metric == "Coverage Miss" else None,
            zmax=1 if metric == "Coverage Miss" else None,
            colorbar=colorbar_config,
            hovertemplate=(
                "Local date/time %{customdata[0]}<br>SimBench step %{x}<br>%{y}<br>"
                "Role: %{customdata[1]}<br>Ground truth: %{customdata[2]:.8f} p.u.<br>"
                "GPR mean: %{customdata[3]:.8f} p.u.<br>"
                "95% PI: [%{customdata[4]:.8f}, %{customdata[5]:.8f}] p.u.<br>"
                + metric + " " + hover_value + "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"{title} — {step_meta.date.iloc[0]} · {len(step_order)} timesteps",
        xaxis_title="Local time of day (repeated DST times are separate steps)",
        yaxis_title="SimBench bus",
        height=max(560, 19 * len(bus_order) + 170),
        margin={"l": 90, "r": 35, "t": 75, "b": 65},
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=step_order[::4],
        ticktext=time_labels[::4],
        type="linear",
    )
    return fig


def annual_day_all_buses_figure(detail, time_step):
    """One saved timestep: each output bus's mean, 95% PI and ground truth."""
    frame = detail[detail["time_step"].astype(int).eq(int(time_step))].copy()
    if frame.empty:
        raise ValueError(f"No state-estimation data for timestep {time_step}.")
    bus_order = bus_order_from_frame(detail)
    frame = frame.set_index("bus").loc[bus_order].reset_index()
    label_lookup = frame_bus_label_lookup(frame)
    labels = [label_lookup.get(int(bus), f"Internal bus {int(bus)}") for bus in frame.bus]
    covered = (
        frame.ground_truth_vm_pu.ge(frame.lower_95_vm_pu)
        & frame.ground_truth_vm_pu.le(frame.upper_95_vm_pu)
    )
    hover = [
        [row.bus_role, float(row.ground_truth_vm_pu), float(row.predicted_vm_pu),
         float(row.lower_95_vm_pu), float(row.upper_95_vm_pu),
         float(row.upper_95_vm_pu - row.lower_95_vm_pu),
         float(abs(row.predicted_vm_pu - row.ground_truth_vm_pu)),
         "Inside" if inside else "Outside"]
        for row, inside in zip(frame.itertuples(index=False), covered)
    ]
    hovertemplate = (
        "%{x}<br>Role: %{customdata[0]}"
        "<br>Ground truth: %{customdata[1]:.8f} p.u."
        "<br>GPR mean: %{customdata[2]:.8f} p.u."
        "<br>Lower 95%: %{customdata[3]:.8f} p.u."
        "<br>Upper 95%: %{customdata[4]:.8f} p.u."
        "<br>PI width: %{customdata[5]:.8f} p.u."
        "<br>Absolute error: %{customdata[6]:.8f} p.u."
        "<br>Ground truth: %{customdata[7]} 95% PI<extra></extra>"
    )
    fig = go.Figure()
    for role, color in [("measured", "#2a78d6"), ("unmeasured", "#eb6834")]:
        indices = np.flatnonzero(frame.bus_role.eq(role).to_numpy()).tolist()
        if not indices:
            continue
        part = frame.iloc[indices]
        fig.add_trace(go.Scatter(
            x=[labels[i] for i in indices], y=part.predicted_vm_pu.astype(float).tolist(),
            mode="markers", name=f"{role.title()}: GPR mean + 95% PI",
            marker={"color": color, "size": 9},
            error_y={
                "type": "data", "symmetric": False, "visible": True,
                "array": (part.upper_95_vm_pu - part.predicted_vm_pu).astype(float).tolist(),
                "arrayminus": (part.predicted_vm_pu - part.lower_95_vm_pu).astype(float).tolist(),
                "color": color, "thickness": 2, "width": 5,
            },
            customdata=[hover[i] for i in indices], hovertemplate=hovertemplate,
        ))
    for inside, color, symbol, name in [
        (True, "#303030", "diamond", "Ground truth: inside 95% PI"),
        (False, "#b2182b", "x", "Ground truth: outside 95% PI"),
    ]:
        indices = np.flatnonzero(covered.eq(inside).to_numpy()).tolist()
        if indices:
            fig.add_trace(go.Scatter(
                x=[labels[i] for i in indices],
                y=frame.iloc[indices].ground_truth_vm_pu.astype(float).tolist(),
                mode="markers", name=name,
                marker={"color": color, "symbol": symbol, "size": 8},
                customdata=[hover[i] for i in indices], hovertemplate=hovertemplate,
            ))
    # Keep the same voltage scale while scrubbing through the selected day.
    values = detail[["ground_truth_vm_pu", "predicted_vm_pu",
                     "lower_95_vm_pu", "upper_95_vm_pu"]].to_numpy(float)
    low, high = float(values.min()), float(values.max())
    padding = max((high - low) * 0.08, 1e-5)
    fig.update_layout(
        title=f"All output buses — {frame['time'].iloc[0]} · step {int(time_step)}",
        xaxis_title="SimBench bus", yaxis_title="Voltage magnitude [p.u.]",
        height=660, margin={"l": 65, "r": 25, "t": 120, "b": 180},
        legend={"orientation": "h", "y": 1.03, "yanchor": "bottom", "x": 0},
    )
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=labels,
                     tickmode="array", tickvals=labels, tickangle=-60, tickfont={"size": 10})
    fig.update_yaxes(range=[low - padding, high + padding])
    return fig


def annual_day_bus_timeseries_figure(detail, bus):
    frame = detail[detail["bus"].astype(int) == int(bus)].copy()
    frame = frame.sort_values("time_step")
    if frame.empty:
        raise ValueError(f"No state-estimation data for selected bus {bus}.")
    # Plain lists also render reliably with Streamlit frontends whose bundled
    # Plotly.js predates Plotly's NumPy binary-array transport.
    x = frame["time_step"].astype(int).tolist()
    time_labels = frame["time"].astype(str).tolist()
    hover = [[label] for label in time_labels]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=frame["upper_95_vm_pu"].astype(float).tolist(),
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=frame["lower_95_vm_pu"].astype(float).tolist(),
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(90, 90, 90, 0.20)",
            name="95% Predictive Interval",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=frame["ground_truth_vm_pu"].astype(float).tolist(),
            customdata=hover,
            mode="lines",
            name="Ground Truth",
            line={"color": "#2a78d6", "width": 2.5},
            hovertemplate="Time %{customdata[0]}<br>SimBench step %{x}<br>V %{y:.8f} p.u.<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=frame["predicted_vm_pu"].astype(float).tolist(),
            customdata=hover,
            mode="lines",
            name="GPR Prediction",
            line={"color": "#eb6834", "width": 2.5, "dash": "dash"},
            hovertemplate="Time %{customdata[0]}<br>SimBench step %{x}<br>V %{y:.8f} p.u.<extra></extra>",
        )
    )
    label_lookup = frame_bus_label_lookup(frame)
    display_bus = label_lookup.get(int(bus), f"Internal bus {int(bus)}")
    fig.update_layout(
        title={"text": f"State Estimation — {display_bus}", "x": 0.02, "y": 0.98,
               "xanchor": "left", "yanchor": "top"},
        xaxis_title="Time of day",
        yaxis_title="Voltage magnitude [p.u.]",
        height=500,
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0},
        margin={"l": 60, "r": 25, "t": 105, "b": 55},
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=x[::8],
        ticktext=time_labels[::8],
    )
    return fig


# -----------------------------------------------------------------------------
# Network topology, component structure, and uncertainty map
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_dashboard_network(network_code):
    """Load the SimBench network once for topology visualization."""
    return load_network(network_code)


def topology_graph(net):
    """Build a bus graph from lines, transformers, and closed bus-bus switches."""
    graph = nx.Graph()
    graph.add_nodes_from(int(bus) for bus in net.bus.index)

    if "line" in net and not net.line.empty:
        for line_idx, row in net.line.iterrows():
            if "in_service" in row.index and not bool(row["in_service"]):
                continue
            graph.add_edge(
                int(row["from_bus"]),
                int(row["to_bus"]),
                kind="line",
                element_index=int(line_idx),
            )

    if "trafo" in net and not net.trafo.empty:
        for trafo_idx, row in net.trafo.iterrows():
            if "in_service" in row.index and not bool(row["in_service"]):
                continue
            graph.add_edge(
                int(row["hv_bus"]),
                int(row["lv_bus"]),
                kind="trafo",
                element_index=int(trafo_idx),
            )

    # Most SimBench switches sit on a line and are already represented by the
    # line edge. Only true bus-bus switches need an additional graph edge.
    if "switch" in net and not net.switch.empty:
        for switch_idx, row in net.switch.iterrows():
            if str(row.get("et", "")) != "b" or not bool(row.get("closed", True)):
                continue
            graph.add_edge(
                int(row["bus"]),
                int(row["element"]),
                kind="switch",
                element_index=int(switch_idx),
            )

    return graph


def _geo_coordinates(net):
    """Return complete pandapower/SimBench drawing coordinates when available.

    The dashboard intentionally does not force the raw geographic CSV
    coordinates.  If the packaged network exposes no suitable drawing geodata,
    ``network_positions`` falls back to a deterministic feeder layout.
    """
    buses = [int(bus) for bus in net.bus.index]

    if "bus_geodata" in net and not net.bus_geodata.empty:
        geo = net.bus_geodata
        if {"x", "y"}.issubset(geo.columns) and all(bus in geo.index for bus in buses):
            if not geo.loc[buses, ["x", "y"]].isna().any().any():
                return {
                    bus: (float(geo.at[bus, "x"]), float(geo.at[bus, "y"]))
                    for bus in buses
                }, "SimBench / pandapower schematic geodata"

    if "geo" in net.bus.columns:
        coordinates = {}
        for bus in buses:
            value = net.bus.at[bus, "geo"]
            if pd.isna(value):
                break
            try:
                payload = json.loads(value) if isinstance(value, str) else value
                coords = payload.get("coordinates") if isinstance(payload, dict) else None
                if coords is None and isinstance(payload, dict):
                    coords = payload.get("geometry", {}).get("coordinates")
                if coords is None or len(coords) < 2:
                    break
                coordinates[bus] = (float(coords[0]), float(coords[1]))
            except (TypeError, ValueError, json.JSONDecodeError):
                break
        if len(coordinates) == len(buses):
            return coordinates, "SimBench / pandapower schematic geodata"

    return None, None

def network_positions(net, graph, reference_buses):
    """
    Build a deterministic topology-only schematic layout.

    No geographic / bus_geodata coordinates are used. The drawing is derived
    exclusively from the actual electrical graph (lines, transformer and
    closed bus-bus switches), rooted at the live external-grid/reference bus.

    This prevents physically distinct SimBench buses that share the same
    geographic coordinate from being drawn on top of each other.
    """
    if graph.number_of_nodes() == 0:
        return {}, "No topology"

    references = sorted(
        [int(bus) for bus in reference_buses if int(bus) in graph],
        key=lambda bus: live_bus_label(net, bus),
    )
    root = references[0] if references else min(
        graph.nodes,
        key=lambda bus: live_bus_label(net, int(bus)),
    )

    # The thesis network is radial. A BFS tree therefore reproduces the
    # electrical feeder hierarchy directly. Sorting by exact SimBench bus name
    # makes the layout deterministic across refreshes.
    tree = nx.bfs_tree(
        graph,
        root,
        sort_neighbors=lambda nodes: sorted(
            nodes,
            key=lambda bus: live_bus_label(net, int(bus)),
        ),
    )

    children = {
        int(bus): [int(child) for child in tree.successors(bus)]
        for bus in tree.nodes
    }

    positions = {}
    leaf_cursor = [0.0]
    leaf_spacing = 3.0
    depth_spacing = 1.0

    def place(bus, depth):
        bus = int(bus)
        bus_children = children.get(bus, [])

        if not bus_children:
            x = leaf_cursor[0]
            leaf_cursor[0] += leaf_spacing
        else:
            child_x = [place(child, depth + 1) for child in bus_children]
            x = float(np.mean(child_x))

        positions[bus] = (float(x), float(-depth * depth_spacing))
        return float(x)

    place(root, 0)

    # Defensive handling for an unexpected disconnected component. This does
    # not occur in the validated thesis network, but avoids overlapping nodes
    # if a future network contains more than one component.
    remaining = [int(bus) for bus in graph.nodes if int(bus) not in positions]
    if remaining:
        current_x = max((x for x, _ in positions.values()), default=0.0) + 2 * leaf_spacing

        for component_nodes in sorted(
            nx.connected_components(graph.subgraph(remaining)),
            key=lambda comp: min(live_bus_label(net, int(bus)) for bus in comp),
        ):
            component = graph.subgraph(component_nodes).copy()
            component_root = min(
                component.nodes,
                key=lambda bus: live_bus_label(net, int(bus)),
            )
            component_tree = nx.bfs_tree(
                component,
                component_root,
                sort_neighbors=lambda nodes: sorted(
                    nodes,
                    key=lambda bus: live_bus_label(net, int(bus)),
                ),
            )

            component_children = {
                int(bus): [int(child) for child in component_tree.successors(bus)]
                for bus in component_tree.nodes
            }
            component_cursor = [current_x]

            def place_component(bus, depth):
                bus = int(bus)
                kids = component_children.get(bus, [])
                if not kids:
                    x = component_cursor[0]
                    component_cursor[0] += leaf_spacing
                else:
                    xs = [place_component(child, depth + 1) for child in kids]
                    x = float(np.mean(xs))
                positions[bus] = (float(x), float(-depth * depth_spacing))
                return float(x)

            place_component(component_root, 0)
            current_x = component_cursor[0] + 2 * leaf_spacing

    return (
        positions,
        "Topology-only schematic layout rooted at the exact SimBench external-grid bus",
    )


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def classify_load_profile(row):
    """
    Derive a readable load class from SimBench profile/type/name information.

    SimBench load profiles commonly use H0 for household, G0-G6 for commercial,
    and L0-L2 for agricultural standard-load-profile families. Unknown profiles
    remain explicitly unclassified instead of being guessed.
    """
    profile = _safe_text(row.get("profile", ""))
    profile_prefix = profile.upper().split("-")[0].split("_")[0]

    if profile_prefix == "H0":
        return "Household"
    if profile_prefix in {f"G{i}" for i in range(7)}:
        return "Commercial"
    if profile_prefix in {f"L{i}" for i in range(3)}:
        return "Agriculture"
    if profile_prefix in {"BL", "BW", "WB"}:
        return "Band / other load"

    searchable = " ".join(
        _safe_text(row.get(column, ""))
        for column in ["type", "name", "category"]
    ).lower()
    if any(token in searchable for token in ["household", "residential", "haushalt"]):
        return "Household"
    if any(token in searchable for token in ["commercial", "business", "gewerbe"]):
        return "Commercial"
    if any(token in searchable for token in ["agriculture", "agricultural", "landwirtschaft"]):
        return "Agriculture"
    return "Other / unclassified"


def classify_generation_profile(row):
    """Return a compact generation technology label when the data support it."""
    profile = _safe_text(row.get("profile", ""))
    searchable = " ".join(
        [
            profile,
            _safe_text(row.get("type", "")),
            _safe_text(row.get("name", "")),
        ]
    ).upper()

    if "PV" in searchable or "SOLAR" in searchable:
        return "PV"
    if "WIND" in searchable or profile.upper().startswith("WP"):
        return "Wind"
    if "BIOM" in searchable or profile.upper().startswith("BM"):
        return "Biomass"
    if "HYDRO" in searchable or "WATER" in searchable:
        return "Hydro"
    return "Other generation"


def network_component_summary(net):
    """Build one topology/component summary row per bus for hover information."""
    summary = pd.DataFrame(index=net.bus.index.astype(int))
    summary.index.name = "bus"

    summary["bus_name"] = [
        _safe_text(net.bus.at[bus, "name"]) if "name" in net.bus.columns else ""
        for bus in summary.index
    ]
    summary["vn_kv"] = [
        _safe_float(net.bus.at[bus, "vn_kv"], np.nan) if "vn_kv" in net.bus.columns else np.nan
        for bus in summary.index
    ]
    summary["zone"] = [
        _safe_text(net.bus.at[bus, "zone"]) if "zone" in net.bus.columns else ""
        for bus in summary.index
    ]

    for column, default in [
        ("line_count", 0),
        ("load_count", 0),
        ("load_p_mw", 0.0),
        ("load_q_mvar", 0.0),
        ("sgen_count", 0),
        ("sgen_p_mw", 0.0),
        ("sgen_q_mvar", 0.0),
        ("is_ext_grid", False),
        ("is_trafo_bus", False),
    ]:
        summary[column] = default

    summary["load_classes"] = ""
    summary["load_profiles"] = ""
    summary["generation_types"] = ""
    summary["generation_profiles"] = ""

    # Line degree from the actual line table (transformers handled separately).
    if "line" in net and not net.line.empty:
        for _, row in net.line.iterrows():
            if "in_service" in row.index and not bool(row["in_service"]):
                continue
            for endpoint in ["from_bus", "to_bus"]:
                bus = int(row[endpoint])
                if bus in summary.index:
                    summary.at[bus, "line_count"] += 1

    if "load" in net and not net.load.empty and "bus" in net.load.columns:
        class_map = {bus: [] for bus in summary.index}
        profile_map = {bus: [] for bus in summary.index}
        for _, row in net.load.iterrows():
            if "in_service" in row.index and not bool(row.get("in_service", True)):
                continue
            bus = int(row["bus"])
            if bus not in summary.index:
                continue
            summary.at[bus, "load_count"] += 1
            summary.at[bus, "load_p_mw"] += _safe_float(row.get("p_mw", 0.0))
            summary.at[bus, "load_q_mvar"] += _safe_float(row.get("q_mvar", 0.0))
            class_map[bus].append(classify_load_profile(row))
            profile = _safe_text(row.get("profile", ""))
            if profile:
                profile_map[bus].append(profile)
        for bus in summary.index:
            summary.at[bus, "load_classes"] = ", ".join(sorted(set(class_map[bus])))
            summary.at[bus, "load_profiles"] = ", ".join(sorted(set(profile_map[bus])))

    if "sgen" in net and not net.sgen.empty and "bus" in net.sgen.columns:
        type_map = {bus: [] for bus in summary.index}
        profile_map = {bus: [] for bus in summary.index}
        for _, row in net.sgen.iterrows():
            if "in_service" in row.index and not bool(row.get("in_service", True)):
                continue
            bus = int(row["bus"])
            if bus not in summary.index:
                continue
            summary.at[bus, "sgen_count"] += 1
            summary.at[bus, "sgen_p_mw"] += _safe_float(row.get("p_mw", 0.0))
            summary.at[bus, "sgen_q_mvar"] += _safe_float(row.get("q_mvar", 0.0))
            type_map[bus].append(classify_generation_profile(row))
            profile = _safe_text(row.get("profile", ""))
            if profile:
                profile_map[bus].append(profile)
        for bus in summary.index:
            summary.at[bus, "generation_types"] = ", ".join(sorted(set(type_map[bus])))
            summary.at[bus, "generation_profiles"] = ", ".join(sorted(set(profile_map[bus])))

    if "ext_grid" in net and not net.ext_grid.empty and "bus" in net.ext_grid.columns:
        for bus in net.ext_grid["bus"].dropna().astype(int):
            if bus in summary.index:
                summary.at[bus, "is_ext_grid"] = True

    if "trafo" in net and not net.trafo.empty:
        for _, row in net.trafo.iterrows():
            for endpoint in ["hv_bus", "lv_bus"]:
                if endpoint in row.index and pd.notna(row[endpoint]):
                    bus = int(row[endpoint])
                    if bus in summary.index:
                        summary.at[bus, "is_trafo_bus"] = True

    return summary.reset_index()


def network_inventory_tables(net):
    """Return detailed load, generation, and transformer tables for the dashboard."""
    loads = pd.DataFrame()
    if "load" in net and not net.load.empty:
        loads = net.load.copy().reset_index().rename(columns={"index": "load_id"})
        loads["load_category"] = net.load.apply(classify_load_profile, axis=1).to_numpy()
        wanted = [
            "load_id", "bus", "name", "profile", "load_category",
            "p_mw", "q_mvar", "scaling", "in_service", "type",
        ]
        loads = loads[[column for column in wanted if column in loads.columns]]

    generators = pd.DataFrame()
    if "sgen" in net and not net.sgen.empty:
        generators = net.sgen.copy().reset_index().rename(columns={"index": "sgen_id"})
        generators["generation_type"] = net.sgen.apply(classify_generation_profile, axis=1).to_numpy()
        wanted = [
            "sgen_id", "bus", "name", "profile", "generation_type",
            "p_mw", "q_mvar", "scaling", "in_service", "type",
        ]
        generators = generators[[column for column in wanted if column in generators.columns]]

    trafos = pd.DataFrame()
    if "trafo" in net and not net.trafo.empty:
        trafos = net.trafo.copy().reset_index().rename(columns={"index": "trafo_id"})
        wanted = [
            "trafo_id", "name", "hv_bus", "lv_bus", "sn_mva",
            "vn_hv_kv", "vn_lv_kv", "vk_percent", "vkr_percent",
            "tap_pos", "in_service", "std_type",
        ]
        trafos = trafos[[column for column in wanted if column in trafos.columns]]

    return loads, generators, trafos


def uncertainty_from_annual_bus_metrics(annual_eval):
    frame = annual_eval["buses"]
    if frame.empty or "mean_interval_width_95_pu" not in frame.columns:
        return {}
    return dict(
        zip(
            frame["bus"].astype(int),
            frame["mean_interval_width_95_pu"].astype(float),
        )
    )


def uncertainty_from_daily_mean(annual_eval, day, day_detail=None):
    """Mean 95% PI width per bus over all actual SimBench timesteps of one test day."""
    daily_bus = annual_eval.get("daily_bus", pd.DataFrame())
    if not daily_bus.empty and "mean_interval_width_95_pu" in daily_bus.columns:
        frame = daily_bus[daily_bus["day"].astype(int) == int(day)]
        if not frame.empty:
            return dict(
                zip(
                    frame["bus"].astype(int),
                    frame["mean_interval_width_95_pu"].astype(float),
                )
            )

    if day_detail is None or day_detail.empty:
        return {}
    grouped = day_detail.groupby(day_detail["bus"].astype(int))["pi_width_pu"].mean()
    return {int(bus): float(value) for bus, value in grouped.items()}


def uncertainty_from_timestep(day_detail, time_step):
    """95% PI width per bus for one exact 15-minute operating point."""
    if day_detail.empty:
        return {}
    frame = day_detail[day_detail["time_step"].astype(int) == int(time_step)]
    return dict(zip(frame["bus"].astype(int), frame["pi_width_pu"].astype(float)))


def _safe_color_range(values):
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    lower = float(values.min())
    upper = float(values.max())
    if np.isclose(lower, upper):
        pad = max(abs(lower) * 0.05, 1e-12)
        lower -= pad
        upper += pad
    return lower, upper


def _component_offsets(positions):
    """Return visually stable small offsets for load and generation symbols."""
    if not positions:
        return 0.04, 0.04
    xs = np.asarray([xy[0] for xy in positions.values()], dtype=float)
    ys = np.asarray([xy[1] for xy in positions.values()], dtype=float)
    x_span = max(float(np.ptp(xs)), 1.0)
    y_span = max(float(np.ptp(ys)), 1.0)
    return 0.025 * x_span, 0.035 * y_span


def _bus_hover_text(bus, row, role, uncertainty=None, display_label=None):
    name = _safe_text(row.get("bus_name", "")) or "–"
    vn = row.get("vn_kv", np.nan)
    vn_text = f"{float(vn):.3f} kV" if pd.notna(vn) else "–"
    load_classes = _safe_text(row.get("load_classes", "")) or "–"
    load_profiles = _safe_text(row.get("load_profiles", "")) or "–"
    gen_types = _safe_text(row.get("generation_types", "")) or "–"
    gen_profiles = _safe_text(row.get("generation_profiles", "")) or "–"
    u_text = "–" if uncertainty is None or not np.isfinite(uncertainty) else f"{uncertainty:.8f} p.u."
    label = display_label or f"Internal bus {int(bus)}"
    return (
        f"<b>{label}</b><br>"
        f"SimBench name: {name}<br>"
        f"Voltage level: {vn_text}<br>"
        f"Role: {role}<br>"
        f"Connected lines: {int(row.get('line_count', 0))}<br>"
        f"Loads: {int(row.get('load_count', 0))} "
        f"(P={float(row.get('load_p_mw', 0.0)):.5f} MW, "
        f"Q={float(row.get('load_q_mvar', 0.0)):.5f} MVAr)<br>"
        f"Load class(es): {load_classes}<br>"
        f"Load profile(s): {load_profiles}<br>"
        f"Generators: {int(row.get('sgen_count', 0))} "
        f"(P={float(row.get('sgen_p_mw', 0.0)):.5f} MW)<br>"
        f"Generation type(s): {gen_types}<br>"
        f"Generation profile(s): {gen_profiles}<br>"
        f"95% PI width: {u_text}"
    )


def _bus_load_visual_category(summary_row):
    """Return one compact load category for drawing a bus directly on the network."""
    load_count = int(_safe_float(summary_row.get("load_count", 0), 0))
    if load_count <= 0:
        return "No load"

    classes = [
        item.strip()
        for item in _safe_text(summary_row.get("load_classes", "")).split(",")
        if item.strip()
    ]
    classes = sorted(set(classes))
    if len(classes) == 1:
        return classes[0]
    if len(classes) > 1:
        return "Mixed load"
    return "Other / unclassified"



def _external_grid_marker_positions(positions, ext_buses):
    """
    Draw the external-grid symbol next to its connected bus instead of directly
    on top of it.

    The offset points away from the network centroid, which keeps the symbol
    visually separate without changing the electrical topology.
    """
    if not positions:
        return {}

    xs = np.array([float(x) for x, _ in positions.values()], dtype=float)
    ys = np.array([float(y) for _, y in positions.values()], dtype=float)
    cx, cy = float(np.mean(xs)), float(np.mean(ys))

    span_x = max(float(np.ptp(xs)), 1e-9)
    span_y = max(float(np.ptp(ys)), 1e-9)
    scale = max(span_x, span_y)

    result = {}
    for bus in ext_buses:
        if bus not in positions:
            continue
        x, y = positions[bus]

        # Move outward from the centre of the drawing.
        dx, dy = float(x - cx), float(y - cy)
        norm = (dx * dx + dy * dy) ** 0.5

        if norm < 1e-12:
            # Deterministic fallback if the bus happens to be at the centroid.
            dx, dy, norm = 1.0, -0.25, (1.0 + 0.25**2) ** 0.5

        ux, uy = dx / norm, dy / norm
        offset = 0.065 * scale
        result[bus] = (float(x + ux * offset), float(y + uy * offset))

    return result


def network_professional_figure(
    net,
    annual_config,
    view_mode="Network Structure",
    uncertainty=None,
    title="Electrical Network Structure",
    show_labels=False,
    color_range=None,
    evaluation_buses_override=None,
):
    """
    Compact network visualization.

    The physical topology remains the central object. Load information is encoded
    directly in each bus marker instead of drawing detached load symbols. Measured
    buses are marked by a red ring in every view, so the sensor placement stays
    visible without obscuring the physical network.
    """
    uncertainty = remap_saved_bus_value_dict(net, annual_config, uncertainty or {})
    bus_context = dashboard_bus_context(
        net,
        annual_config,
        evaluation_buses_override=evaluation_buses_override,
    )
    measured_buses = bus_context["measured_buses"]
    reference_buses = bus_context["reference_buses"]
    evaluation_buses = bus_context["evaluation_buses"]
    label_for = bus_context["label_for"]

    graph = topology_graph(net)
    positions, layout_name = network_positions(net, graph, reference_buses)
    summary = network_component_summary(net).set_index("bus")
    fig = go.Figure()

    # ------------------------------------------------------------------
    # Physical lines
    # ------------------------------------------------------------------
    line_x, line_y = [], []
    if "line" in net and not net.line.empty:
        for _, row in net.line.iterrows():
            if "in_service" in row.index and not bool(row["in_service"]):
                continue
            from_bus, to_bus = int(row["from_bus"]), int(row["to_bus"])
            if from_bus not in positions or to_bus not in positions:
                continue
            x0, y0 = positions[from_bus]
            x1, y1 = positions[to_bus]
            line_x.extend([x0, x1, None])
            line_y.extend([y0, y1, None])
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line={"width": 2.0, "color": "#8a949f"},
            name="Line",
            hoverinfo="skip",
        )
    )

    # Closed bus-bus switches.
    switch_x, switch_y = [], []
    if "switch" in net and not net.switch.empty:
        for _, row in net.switch.iterrows():
            if str(row.get("et", "")) != "b" or not bool(row.get("closed", True)):
                continue
            bus_a, bus_b = int(row["bus"]), int(row["element"])
            if bus_a not in positions or bus_b not in positions:
                continue
            x0, y0 = positions[bus_a]
            x1, y1 = positions[bus_b]
            switch_x.extend([x0, x1, None])
            switch_y.extend([y0, y1, None])
    if switch_x:
        fig.add_trace(
            go.Scatter(
                x=switch_x,
                y=switch_y,
                mode="lines",
                line={"width": 1.5, "dash": "dash", "color": "#a7afb8"},
                name="Bus-bus switch",
                hoverinfo="skip",
            )
        )

    # Transformer: keep a compact two-coil symbol directly on the branch.
    trafo_symbol_x, trafo_symbol_y, trafo_hover = [], [], []
    if "trafo" in net and not net.trafo.empty:
        for trafo_idx, row in net.trafo.iterrows():
            if "in_service" in row.index and not bool(row["in_service"]):
                continue
            hv_bus, lv_bus = int(row["hv_bus"]), int(row["lv_bus"])
            if hv_bus not in positions or lv_bus not in positions:
                continue
            x0, y0 = positions[hv_bus]
            x1, y1 = positions[lv_bus]
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line={"width": 2.6, "color": "#303846"},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            vx, vy = x1 - x0, y1 - y0
            norm = max((vx**2 + vy**2) ** 0.5, 1e-12)
            ux, uy = vx / norm, vy / norm
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            separation = 0.018 * norm
            p1 = (mx - ux * separation, my - uy * separation)
            p2 = (mx + ux * separation, my + uy * separation)
            details = [
                f"<b>Transformer {int(trafo_idx)}</b>",
                f"HV bus: {label_for(hv_bus)}",
                f"LV bus: {label_for(lv_bus)}",
            ]
            if "sn_mva" in row.index and pd.notna(row["sn_mva"]):
                details.append(f"Rated power: {float(row['sn_mva']):.4f} MVA")
            if (
                "vn_hv_kv" in row.index
                and "vn_lv_kv" in row.index
                and pd.notna(row["vn_hv_kv"])
                and pd.notna(row["vn_lv_kv"])
            ):
                details.append(
                    f"Voltage: {float(row['vn_hv_kv']):.3f}/{float(row['vn_lv_kv']):.3f} kV"
                )
            hover = "<br>".join(details)
            trafo_symbol_x.extend([p1[0], p2[0]])
            trafo_symbol_y.extend([p1[1], p2[1]])
            trafo_hover.extend([hover, hover])

    if trafo_symbol_x:
        fig.add_trace(
            go.Scatter(
                x=trafo_symbol_x,
                y=trafo_symbol_y,
                mode="markers",
                marker={
                    "symbol": "circle-open",
                    "size": 16,
                    "color": "#202833",
                    "line": {"width": 2.2, "color": "#202833"},
                },
                name="Transformer",
                hovertext=trafo_hover,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

    # ------------------------------------------------------------------
    # Bus markers
    # ------------------------------------------------------------------
    all_buses = sorted(bus for bus in positions if bus in summary.index)

    def hover_for(bus):
        if bus in reference_buses:
            role = "reference / slack bus"
        elif bus in measured_buses:
            role = "measured / sensor bus"
        elif bus in evaluation_buses:
            role = "unmeasured / evaluation bus"
        else:
            role = "other bus"
        return _bus_hover_text(bus, summary.loc[bus], role, uncertainty.get(bus), label_for(bus))

    label_mode = "markers+text" if show_labels else "markers"

    if view_mode == "Network Structure":
        # Encode load type directly at the bus. No detached component symbols.
        load_style = {
            "No load": {"color": "#f3f4f6", "size": 9},
            "Household": {"color": "#4c78a8", "size": 13},
            "Commercial": {"color": "#f28e2b", "size": 13},
            "Agriculture": {"color": "#59a14f", "size": 13},
            "Band / other load": {"color": "#8f63b8", "size": 13},
            "Other / unclassified": {"color": "#9ca3af", "size": 13},
            "Mixed load": {"color": "#b07aa1", "size": 14},
        }
        categories = {
            bus: _bus_load_visual_category(summary.loc[bus]) for bus in all_buses
        }
        for category, style in load_style.items():
            buses = [bus for bus in all_buses if categories[bus] == category]
            if not buses:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[positions[bus][0] for bus in buses],
                    y=[positions[bus][1] for bus in buses],
                    mode=label_mode,
                    text=[label_for(bus) for bus in buses] if show_labels else None,
                    textposition="top center",
                    marker={
                        "size": style["size"],
                        "symbol": "circle",
                        "color": style["color"],
                        "line": {"width": 1.5, "color": "#374151"},
                    },
                    name=category,
                    hovertext=[hover_for(bus) for bus in buses],
                    hovertemplate="%{hovertext}<extra></extra>",
                )
            )

    else:  # GPR Uncertainty
        evaluation = [bus for bus in all_buses if bus in evaluation_buses]
        non_evaluation = [bus for bus in all_buses if bus not in evaluation_buses]

        if non_evaluation:
            fig.add_trace(
                go.Scatter(
                    x=[positions[bus][0] for bus in non_evaluation],
                    y=[positions[bus][1] for bus in non_evaluation],
                    mode=label_mode,
                    text=[label_for(bus) for bus in non_evaluation] if show_labels else None,
                    textposition="top center",
                    marker={
                        "size": 10,
                        "symbol": "circle",
                        "color": "#e5e7eb",
                        "line": {"width": 1.3, "color": "#4b5563"},
                    },
                    name="Non-evaluation bus",
                    hovertext=[hover_for(bus) for bus in non_evaluation],
                    hovertemplate="%{hovertext}<extra></extra>",
                )
            )

        values = [uncertainty.get(bus, np.nan) for bus in evaluation]
        has_uncertainty = any(np.isfinite(value) for value in values)
        marker = {
            "size": 14,
            "symbol": "circle",
            "line": {"width": 1.4, "color": "#27313d"},
        }
        if has_uncertainty:
            marker.update(
                {
                    "color": values,
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": "95% PI width<br>[p.u.]", "thickness": 16},
                }
            )
            if color_range is not None:
                marker["cmin"], marker["cmax"] = color_range
        else:
            marker["color"] = "#d1d5db"

        if evaluation:
            fig.add_trace(
                go.Scatter(
                    x=[positions[bus][0] for bus in evaluation],
                    y=[positions[bus][1] for bus in evaluation],
                    mode=label_mode,
                    text=[label_for(bus) for bus in evaluation] if show_labels else None,
                    textposition="top center",
                    marker=marker,
                    name="Evaluation bus uncertainty",
                    hovertext=[hover_for(bus) for bus in evaluation],
                    hovertemplate="%{hovertext}<extra></extra>",
                )
            )

    # ------------------------------------------------------------------
    # Measurement overlay: always visible, directly on the measured bus.
    # ------------------------------------------------------------------
    measured = sorted(bus for bus in measured_buses if bus in positions)
    if measured:
        fig.add_trace(
            go.Scatter(
                x=[positions[bus][0] for bus in measured],
                y=[positions[bus][1] for bus in measured],
                mode="markers",
                marker={
                    "size": 20,
                    "symbol": "circle-open",
                    "color": "#d62728",
                    "line": {"width": 3.0, "color": "#d62728"},
                },
                name="Measured / sensor bus",
                hovertext=[f"{label_for(bus)}<br>Measured / sensor bus" for bus in measured],
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

    # Reference / external-grid bus:
    # overlay a star directly on the *actual* bus returned by net.ext_grid.
    # No artificial external-grid node, no connector line, and no coordinate
    # shift are introduced.
    ext_buses = sorted(
        bus for bus in all_buses
        if bool(summary.at[bus, "is_ext_grid"])
    )
    if ext_buses:
        fig.add_trace(
            go.Scatter(
                x=[positions[bus][0] for bus in ext_buses],
                y=[positions[bus][1] for bus in ext_buses],
                mode="markers",
                marker={
                    "size": 19,
                    "symbol": "star",
                    "color": "#111827",
                    "line": {"width": 1.2, "color": "#ffffff"},
                },
                name="Reference / external-grid bus",
                hovertext=[
                    f"Reference / external-grid bus<br>{label_for(bus)}"
                    for bus in ext_buses
                ],
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"{title}<br><sup>{layout_name}</sup>",
        template="plotly_white",
        height=760,
        margin={"l": 20, "r": 35, "t": 105, "b": 30},
        xaxis={"visible": False, "showgrid": False, "zeroline": False},
        yaxis={
            "visible": False,
            "showgrid": False,
            "zeroline": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "x": 0,
            "bgcolor": "rgba(255,255,255,0.84)",
        },
        hovermode="closest",
        hoverlabel={"bgcolor": "white", "font_size": 12},
        uirevision="network-topology",
    )
    return fig

# -----------------------------------------------------------------------------
# Physical annual / stress-case loading
# -----------------------------------------------------------------------------

def load_year_analysis(network_dir, run_dir=None):
    analysis_dir = Path(run_dir if run_dir is not None else network_dir) / "year_analysis"

    def read(name):
        path = analysis_dir / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    return {
        "analysis_dir": analysis_dir,
        "yearly": read("yearly_summary.csv"),
        "cases": read("extreme_cases.csv"),
        "profiles": read("extreme_voltage_profiles.csv"),
    }


def load_stress_run(annual_run_dir):
    """Load stress predictions tied to the selected seasonal GPR run."""
    run_dir = Path(annual_run_dir) / "stress_cases"
    summary_path = run_dir / "stress_case_summary.csv"
    pred_path = run_dir / "stress_predictions.csv"
    if not (summary_path.exists() and pred_path.exists()):
        return None
    config = {}
    config_path = run_dir / "stress_config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as file:
            config = json.load(file)
    overall_path = run_dir / "stress_overall_metrics.csv"
    metrics_path = run_dir / "stress_case_metrics.csv"
    measurement_path = run_dir / "stress_measurements.csv"
    return {
        "path": run_dir,
        "summary": pd.read_csv(summary_path),
        "case_metrics": pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame(),
        "predictions": pd.read_csv(pred_path),
        "overall": pd.read_csv(overall_path) if overall_path.exists() else pd.DataFrame(),
        "measurements": pd.read_csv(measurement_path) if measurement_path.exists() else pd.DataFrame(),
        "config": config,
    }


def stress_metrics(stress_run, selected_case, evaluation_scope=PRIMARY_SCOPE):
    if stress_run is None:
        return None
    saved = stress_run.get("case_metrics", pd.DataFrame())
    if not saved.empty and "evaluation_scope" in saved.columns:
        match = saved[(saved["case"] == selected_case) & (saved["evaluation_scope"] == evaluation_scope)]
        if not match.empty:
            row = match.iloc[0]
            return {
                "mae_pu": float(row["voltage_mae_pu"]),
                "rmse_pu": float(row["voltage_rmse_pu"]),
                "coverage_95": float(row["voltage_coverage_95"]),
                "mean_pi_width_pu": float(row["voltage_mean_interval_width_95_pu"]),
            }
    frame = stress_run["predictions"]
    frame = frame[frame["case"] == selected_case].copy()
    if evaluation_scope == PRIMARY_SCOPE and "bus_role" in frame.columns:
        frame = frame[frame["bus_role"] == "unmeasured"]
    if frame.empty:
        return None
    error = frame["predicted_vm_pu"].to_numpy(float) - frame["ground_truth_vm_pu"].to_numpy(float)
    lower = frame["lower_95_vm_pu"].to_numpy(float)
    upper = frame["upper_95_vm_pu"].to_numpy(float)
    true = frame["ground_truth_vm_pu"].to_numpy(float)
    return {
        "mae_pu": float(np.mean(np.abs(error))),
        "rmse_pu": float(np.sqrt(np.mean(error**2))),
        "coverage_95": float(np.mean((true >= lower) & (true <= upper))),
        "mean_pi_width_pu": float(np.mean(upper - lower)),
    }


def stress_overall_from_predictions(stress_run, case_scope="all_cases", evaluation_scope=PRIMARY_SCOPE):
    if stress_run is None:
        return None
    overall = stress_run.get("overall", pd.DataFrame())
    if not overall.empty:
        cols = overall.columns
        mask = pd.Series(True, index=overall.index)
        if "case_scope" in cols:
            mask &= overall["case_scope"].eq(case_scope)
        elif "scope" in cols:
            mask &= overall["scope"].eq(case_scope)
        if "evaluation_scope" in cols:
            mask &= overall["evaluation_scope"].eq(evaluation_scope)
        match = overall[mask]
        if not match.empty:
            return match.iloc[0].to_dict()
    return None


def extreme_figure(stress_run, selected_case, truth_profiles):
    """Voltage profile and 95% PI for one physical stress operating point."""
    frame = stress_run["predictions"]
    frame = frame[frame["case"] == selected_case].copy()
    frame["bus"] = frame["bus"].astype(int)
    frame_order = bus_order_from_frame(frame)
    frame = frame.set_index("bus").reindex(frame_order).reset_index()

    truth = truth_profiles[truth_profiles["case"] == selected_case].copy()
    if truth.empty:
        truth = frame[["bus", "ground_truth_vm_pu"]].rename(
            columns={"ground_truth_vm_pu": "vm_pu"}
        )
    truth["bus"] = truth["bus"].astype(int)
    truth_order = bus_order_from_frame(truth)
    truth = truth.set_index("bus").reindex(truth_order).reset_index()
    frame_label_map = frame_bus_label_lookup(frame)
    truth_label_map = frame_bus_label_lookup(truth)
    label_map = {**frame_label_map, **truth_label_map}

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[label_map.get(int(bus), f"Internal bus {int(bus)}") for bus in truth["bus"]],
            y=truth["vm_pu"],
            mode="lines",
            line={"dash": "dash"},
            name="Ground Truth",
        )
    )
    add_shaded_interval(
        fig,
        truth["bus"].tolist(),
        frame,
        "lower_95_vm_pu",
        "upper_95_vm_pu",
        "95% Predictive Interval",
        bus_label_map=label_map,
    )
    fig.add_trace(
        go.Scatter(
            x=[label_map.get(int(bus), f"Internal bus {int(bus)}") for bus in frame["bus"]],
            y=frame["predicted_vm_pu"],
            mode="lines+markers",
            name="GPR Prediction",
        )
    )

    measured_buses = set(
        int(bus) for bus in stress_run.get("config", {}).get("measured_buses", [])
    )
    if measured_buses:
        measured = frame[frame["bus"].isin(measured_buses)]
        if not measured.empty:
            fig.add_trace(
                go.Scatter(
                    x=[label_map.get(int(bus), f"Internal bus {int(bus)}") for bus in measured["bus"]],
                    y=measured["predicted_vm_pu"],
                    mode="markers",
                    marker={"size": 9, "color": "red"},
                    name="Measured Bus",
                )
            )

    fig.update_layout(
        title=f"{case_label(selected_case)} — voltage estimation",
        xaxis_title="SimBench bus",
        yaxis_title="Voltage (p.u.)",
        height=500,
    )
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=[label_map.get(int(bus), f"Internal bus {int(bus)}") for bus in truth["bus"]],
    )
    return fig

# -----------------------------------------------------------------------------
# Kernel-diagnostic cache
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_kernel_diagnostics_cached(
    run_dir, selected_day, evaluation_scope, model_modified, config_modified
):
    # Modification times and scope are part of the cache key.
    return compute_kernel_diagnostics(
        run_dir, selected_day, evaluation_scope=evaluation_scope
    )


# -----------------------------------------------------------------------------
# Independent winter kernel-diagnostic cache
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_winter_kernel_diagnostics_cached(run_dir, selected_day, target_bus, model_modified, config_modified):
    return compute_winter_kernel_diagnostics(run_dir, selected_day, target_bus)


# App
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------

annual_runs = find_annual_evaluation_runs()
selected_annual_path, _selected_config, _selected_overall, selected_modified = (
    select_annual_evaluation_run(annual_runs)
)
base_eval = load_annual_evaluation(str(selected_annual_path), selected_modified)
annual_config = base_eval["config"]
annual_noise = annual_config.get("use_measurement_noise")
winter_independent = annual_config.get("model_type") == WINTER_MODEL_TYPE
model_description = (
    f"Winter · {int(annual_config['n_models'])} independent GPRs"
    if winter_independent else "4 season-specific GPRs"
)
evaluation_script = "seasonal_evaluation_noise_free.py" if annual_noise is False else "seasonal_evaluation.py"
stress_command = (
    'python -c "from seasonal_evaluation_noise_free import run_stress_analysis; run_stress_analysis()"'
    if annual_noise is False else "python stress_analysis.py"
)
if winter_independent:
    scenario = "noisy" if annual_noise else "noise-free"
    evaluation_script = f"winter_independent_evaluation.py --scenario {scenario}"
    stress_command = f"python {evaluation_script}"
network = annual_config.get("network", base_eval["network_dir"].name)

st.sidebar.markdown("---")
st.sidebar.subheader("Evaluation scope")
selected_scope = st.sidebar.radio(
    "Buses included in metrics",
    [PRIMARY_SCOPE, ALL_SCOPE],
    format_func=lambda scope: SCOPE_LABELS[scope],
    index=0,
    key="evaluation_scope",
)
annual_eval = evaluation_view(base_eval, selected_scope)
annual_overall = annual_eval["overall"]

st.title("⚡ GPR Distribution System State Estimation")
st.caption(
    f"{network} · {model_description} · "
    f"{int(annual_config.get('n_training_days_per_season', 30))} training days per season · "
    "all remaining same-season days are test days · "
    f"{100 * float(annual_config['sensor_ratio']):.0f}% fixed sensors"
)
st.write(
    "**Final system:** P,Q,V → V · "
    f"**Kernel:** {annual_config.get('kernel', 'Unknown')} · "
    f"**Measurement noise:** {'ON' if annual_noise else 'OFF'} · "
    "**Uncertainty:** raw GPR 95% predictive intervals (μ ± 1.96σ)"
)
st.info(
    "Evaluation scope currently shown: **"
    + SCOPE_LABELS[selected_scope]
    + "**. The reference/slack bus is never a GPR output and is therefore not part of either scope."
)

main_tabs = st.tabs(
    [
        "State Estimation & Daily Diagnostics",
        "Seasonal GPR Evaluation",
        "Network Uncertainty Map",
        "Stress Cases",
        "Kernel Diagnostics",
    ]
)

# -----------------------------------------------------------------------------
# Tab 1: Seasonal GPR evaluation
# -----------------------------------------------------------------------------
with main_tabs[1]:
    st.subheader("Seasonal GPR Evaluation")
    if winter_independent:
        st.caption(
            "Winter only: every target bus has its own independently fitted GPR and kernel parameters. "
            f"Each model uses the same {int(annual_config.get('n_training_days_per_season', 30))} winter training days "
            "and the same measured-bus P/Q/V inputs. All remaining winter days are test days. "
            "Compare these results with the Winter results of the four-season run."
        )
    else:
        st.caption(
            "Winter, Spring, Summer and Autumn are fitted independently. Each GPR sees only "
            f"{int(annual_config.get('n_training_days_per_season', 30))} distributed training days from its own season and is tested on every remaining "
            "day of that same season. No day is left unused."
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pooled MAE", format_metric(annual_overall["voltage_mae_pu"], " p.u."))
    m2.metric("Pooled RMSE", format_metric(annual_overall["voltage_rmse_pu"], " p.u."))
    m3.metric("Pooled 95% Coverage", f"{100 * float(annual_overall['voltage_coverage_95']):.2f}%")
    m4.metric(
        "Mean 95% PI Width",
        format_metric(annual_overall["voltage_mean_interval_width_95_pu"], " p.u."),
    )

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Training Days / Season", str(int(annual_config.get("n_training_days_per_season", 30))))
    a2.metric("Total Test Days", str(int(annual_overall["n_test_days"])))
    a3.metric("Evaluation Buses", str(int(annual_overall["n_evaluation_buses"])))
    a4.metric("Evaluated V Estimates", f"{int(annual_overall['n_evaluated_voltage_predictions']):,}")

    st.markdown("### Evaluation-scope comparison")
    scope_compare = base_eval["overall_table"].copy()
    if "evaluation_scope_label" not in scope_compare.columns:
        scope_compare["evaluation_scope_label"] = scope_compare["evaluation_scope"].map(
            lambda x: SCOPE_LABELS.get(x, x)
        )
    scope_compare["coverage_percent"] = 100.0 * scope_compare["voltage_coverage_95"]
    cols = [
        "evaluation_scope_label",
        "n_evaluation_buses",
        "voltage_mae_pu",
        "voltage_rmse_pu",
        "coverage_percent",
        "voltage_mean_interval_width_95_pu",
    ]
    dashboard_dataframe(friendly_table(scope_compare[cols]), hide_index=True, width="stretch")
    st.caption(
        "The second row uses exactly the same predictions and models; only the buses included "
        "in the metric calculation change. This makes the influence of measured buses visible "
        "without retraining anything."
    )

    st.markdown("### Measured vs. unmeasured target buses")
    if winter_independent:
        role_metrics = pd.read_csv(base_eval["run_dir"] / "bus_role_metrics.csv")
        dashboard_dataframe(friendly_table(role_metrics), hide_index=True, width="stretch")
        st.caption("Both groups are evaluated on the same winter test timesteps; each target uses the same input matrix.")
    elif base_eval["daily_bus_all"].empty:
        st.info("Per-season measured/unmeasured summaries require this run's saved daily_bus_metrics.csv.")
    else:
        role_metrics = seasonal_bus_role_metrics(base_eval["daily_bus_all"])
        for season in annual_config.get("season_order", ["winter", "spring", "summer", "autumn"]):
            season_metrics = role_metrics[role_metrics.season.eq(season)].drop(columns="season")
            if not season_metrics.empty:
                st.markdown(f"**{season.title()}**")
                dashboard_dataframe(friendly_table(season_metrics), hide_index=True, width="stretch")
        st.caption(
            "Each table pools the saved test estimates from that season. Measured and unmeasured groups "
            "use the same test timesteps and remain visible in both evaluation scopes. "
            "Metrics are weighted by the number of bus-timestep estimates; RMSE is pooled from squared errors."
        )

    if winter_independent:
        st.markdown("### Independently optimized target-bus kernels")
        dashboard_dataframe(friendly_table(base_eval["kernel"]), hide_index=True, width="stretch")

    st.markdown("### Season-specific results")
    seasonal_display = annual_eval["seasonal"].copy()
    seasonal_display["coverage_percent"] = 100.0 * seasonal_display["voltage_coverage_95"]
    seasonal_cols = [
        "season_label",
        "n_training_days",
        "n_test_days",
        "voltage_mae_pu",
        "voltage_rmse_pu",
        "coverage_percent",
        "voltage_mean_interval_width_95_pu",
        "fit_time_seconds",
        "prediction_time_seconds",
    ]
    dashboard_dataframe(
        friendly_table(seasonal_display[[c for c in seasonal_cols if c in seasonal_display.columns]]),
        hide_index=True,
        width="stretch",
    )

    # Compact visual comparison across seasons.
    chart_left, chart_right = st.columns(2)
    with chart_left:
        fig_rmse = go.Figure(
            go.Bar(
                x=seasonal_display["season_label"],
                y=seasonal_display["voltage_rmse_pu"],
                hovertemplate="%{x}<br>RMSE=%{y:.8f} p.u.<extra></extra>",
            )
        )
        fig_rmse.update_layout(title="Seasonal RMSE", yaxis_title="RMSE [p.u.]", height=380)
        dashboard_plotly_chart(fig_rmse, width="stretch")
    with chart_right:
        fig_cov = go.Figure(
            go.Bar(
                x=seasonal_display["season_label"],
                y=100.0 * seasonal_display["voltage_coverage_95"],
                hovertemplate="%{x}<br>Coverage=%{y:.2f}%<extra></extra>",
            )
        )
        fig_cov.add_hline(y=95.0, line_dash="dash", annotation_text="Nominal 95%")
        fig_cov.update_layout(title="Seasonal 95% Coverage", yaxis_title="Coverage [%]", height=380)
        dashboard_plotly_chart(fig_cov, width="stretch")

    st.markdown("### Daily performance across all seasonal test sets")
    metric_choice = st.radio(
        "Daily metric",
        ["RMSE", "95% Coverage", "Predictive Uncertainty"],
        horizontal=True,
        key="daily_metric_overview",
    )
    if metric_choice == "RMSE":
        fig = daily_metric_figure(annual_eval, "rmse_pu", "Daily RMSE", "RMSE [p.u.]")
    elif metric_choice == "95% Coverage":
        fig = daily_metric_figure(
            annual_eval,
            "coverage_95",
            "Daily raw-GPR 95% coverage",
            "Coverage",
            percent=True,
            nominal=95.0,
        )
    else:
        fig = daily_metric_figure(
            annual_eval,
            "mean_interval_width_95_pu",
            "Daily predictive uncertainty",
            "Mean 95% PI width [p.u.]",
        )
    dashboard_plotly_chart(fig, width="stretch")

    st.markdown("### Spatial-temporal performance")
    if annual_eval["daily_bus"].empty:
        st.info(f"Run `python {evaluation_script}` to generate this scenario's bus × day diagnostics.")
    else:
        h1, h2 = st.columns(2)
        with h1:
            heatmap_metric = st.radio(
                "Heatmap metric",
                ["RMSE", "95% Coverage", "PI Width"],
                horizontal=True,
                key="seasonal_heatmap_metric",
            )
        with h2:
            heatmap_view = st.radio(
                "Days shown",
                ["All test days", "Worst 30 days"],
                horizontal=True,
                key="seasonal_heatmap_view",
            )
        dashboard_plotly_chart(
            annual_bus_day_heatmap(annual_eval, heatmap_metric, day_view=heatmap_view),
            width="stretch",
        )

    with st.expander("Bus-wise pooled summary"):
        left, right = st.columns(2)
        with left:
            dashboard_plotly_chart(
                bus_metric_figure(annual_eval, "rmse_pu", "Pooled RMSE by bus", "RMSE [p.u.]"),
                width="stretch",
            )
        with right:
            dashboard_plotly_chart(
                bus_metric_figure(
                    annual_eval, "coverage_95", "Pooled 95% coverage by bus", "Coverage", percent=True
                ),
                width="stretch",
            )
        bus_table = annual_eval["buses"].copy()
        bus_table["coverage_percent"] = 100.0 * bus_table["coverage_95"]
        dashboard_dataframe(
            friendly_table(hide_internal_bus_columns(bus_table.drop(columns=["coverage_95"], errors="ignore"))),
            hide_index=True,
            width="stretch",
        )

    with st.expander("Training days and fitted GPR models"):
        split_frame = base_eval["training_days_frame"].copy()
        grouped = []
        for season, group in split_frame.groupby("season", sort=False):
            grouped.append(
                {
                    "season": season.title(),
                    "training_days": ", ".join(str(int(x)) for x in group["training_day"]),
                    "n_training_days": len(group),
                    "n_test_days": int((base_eval["test_days_frame"]["season"] == season).sum()),
                }
            )
        dashboard_dataframe(pd.DataFrame(grouped), hide_index=True, width="stretch")
        if not base_eval["kernel"].empty:
            st.markdown("**Fitted kernel per season**")
            dashboard_dataframe(friendly_table(base_eval["kernel"]), hide_index=True, width="stretch")
        st.caption(f"Result directory: {base_eval['run_dir']}")

# -----------------------------------------------------------------------------
# Tab 2: Daily diagnostics
# -----------------------------------------------------------------------------
with main_tabs[0]:
    st.subheader("Daily Diagnostics")
    daily = annual_eval["daily"].copy()
    season_options = ["All seasons"] + [x.title() for x in annual_config.get("season_order", ["winter", "spring", "summer", "autumn"])]
    selected_season_label = st.selectbox("Season", season_options, key="daily_season_filter")
    daily_for_select = daily.copy()
    if selected_season_label != "All seasons":
        daily_for_select = daily_for_select[
            daily_for_select["season"].astype(str).str.lower() == selected_season_label.lower()
        ]
    day_options = daily_for_select["day"].astype(int).tolist()
    if not day_options:
        st.warning("No test days found for this selection.")
        st.stop()
    default_day = int(daily_for_select.loc[daily_for_select["rmse_pu"].idxmax(), "day"])
    selected_day = st.selectbox(
        "Inspect unseen test day",
        day_options,
        index=day_options.index(default_day) if default_day in day_options else 0,
        key="daily_selected_day",
    )
    selected_row = daily_for_select[daily_for_select["day"].astype(int) == int(selected_day)].iloc[0]
    st.caption(
        f"Model used: **{str(selected_row.get('season', season_for_day(selected_day))).title()} GPR**"
    )
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Daily MAE", f"{selected_row['mae_pu']:.8f} p.u.")
    q2.metric("Daily RMSE", f"{selected_row['rmse_pu']:.8f} p.u.")
    q3.metric("95% Coverage", f"{100 * selected_row['coverage_95']:.2f}%")
    q4.metric("Mean 95% PI Width", f"{selected_row['mean_interval_width_95_pu']:.8f} p.u.")

    try:
        raw_detail = load_annual_day_detail(
            str(annual_eval["run_dir"]), int(selected_day),
            day_detail_signature(annual_eval["run_dir"], selected_day))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        st.error(f"State-estimation data could not be verified: {exc}")
        raw_detail = pd.DataFrame()
    detail = filter_detail_scope(raw_detail, selected_scope)

    if detail.empty:
        st.warning("No verified state estimates available for this day and scope in the selected run.")
    else:
        st.markdown("### All-Bus State Estimation at One Timestep")
        step_table = raw_detail[["time_step", "time"]].drop_duplicates("time_step").sort_values("time_step")
        steps = step_table.time_step.astype(int).tolist()
        time_lookup = dict(zip(steps, step_table.time.astype(str)))
        snapshot_context = f"{selected_annual_path}_{selected_day}"
        snapshot_step = st.select_slider(
            "All-bus 15-minute timestep", options=steps, value=steps[0],
            format_func=lambda step: f"{time_lookup[int(step)]} · step {int(step)}",
            key=f"daily_all_bus_step_{snapshot_context}",
        )
        st.caption(
            f"{noise_label(annual_noise)} · Day {selected_day} · "
            f"All {raw_detail.bus.nunique()} GPR output buses (measured + unmeasured). "
            "Dots show GPR means; whiskers show the saved 95% predictive intervals (μ ± 1.96σ). "
            "Diamonds show ground truth; red crosses mark ground truth outside the interval. "
            "The voltage scale stays fixed for this day. The reference/slack bus has no GPR output."
        )
        dashboard_plotly_chart(annual_day_all_buses_figure(raw_detail, snapshot_step),
                               width="stretch", key=f"all_bus_snapshot_{snapshot_context}")
        with st.expander("All-bus values at the selected timestep"):
            snapshot = raw_detail[raw_detail.time_step.astype(int).eq(int(snapshot_step))].copy()
            snapshot = snapshot.set_index("bus").loc[bus_order_from_frame(raw_detail)].reset_index()
            table = pd.DataFrame({
                "SimBench Bus": snapshot.bus.map(frame_bus_label_lookup(raw_detail)),
                "Bus Role": snapshot.bus_role,
                "Ground Truth [p.u.]": snapshot.ground_truth_vm_pu,
                "GPR Mean [p.u.]": snapshot.predicted_vm_pu,
                "Lower 95% [p.u.]": snapshot.lower_95_vm_pu,
                "Upper 95% [p.u.]": snapshot.upper_95_vm_pu,
                "PI Width [p.u.]": snapshot.pi_width_pu,
                "Absolute Error [p.u.]": snapshot.absolute_error_vm_pu,
                "Ground Truth Inside 95% PI": snapshot.covered_95.map({True: "Yes", False: "No"}),
            })
            dashboard_dataframe(table, hide_index=True, width="stretch")

        st.markdown("### Bus-wise State Estimation")
        bus_options = bus_order_from_frame(raw_detail)
        detail_label_lookup = frame_bus_label_lookup(raw_detail)
        selected_bus = st.selectbox(
            "Bus state estimation", bus_options,
            format_func=lambda bus: detail_label_lookup.get(int(bus), bus_display_label(annual_config, bus)),
            index=bus_options.index(bus_order_from_frame(detail)[0]), key="daily_bus",
        )
        role = raw_detail.loc[raw_detail.bus.astype(int).eq(int(selected_bus)), "bus_role"].iloc[0]
        st.caption(
            f"{noise_label(annual_noise)} · {str(selected_row['season']).title()} GPR · "
            f"Day {selected_day} · {role} bus · {raw_detail.attrs.get('detail_source', 'Saved results')}"
        )
        chart_context = f"{selected_annual_path}_{selected_scope}_{selected_day}"
        dashboard_plotly_chart(annual_day_bus_timeseries_figure(raw_detail, selected_bus), width="stretch",
                        key=f"state_estimation_{chart_context}_{selected_bus}")
        st.markdown("### 15-Minute Heatmap")
        drill_metric = st.radio(
            "15-minute heatmap", ["Absolute Error", "95% PI Width", "Coverage Miss"],
            horizontal=True, key="daily_drill_metric",
        )
        st.caption(
            f"{noise_label(annual_noise)} · Day {selected_day} · {SCOPE_LABELS[selected_scope]} · "
            "Heatmap includes only the selected metric scope; the state-estimation selector includes all GPR output buses. "
            "Coverage Miss: 0 = ground truth inside the 95% PI (including bounds); 1 = outside. "
            "Local-clock repeats on the autumn DST day remain separate SimBench steps."
        )
        column = {"Absolute Error": "absolute_error_vm_pu", "95% PI Width": "pi_width_pu"}.get(drill_metric)
        if column:
            st.caption(f"Value range: {detail[column].min():.8f}–{detail[column].max():.8f} p.u. "
                       "Colours are scaled to this selected day's values.")
        dashboard_plotly_chart(annual_day_heatmap_figure(detail, drill_metric), width="stretch",
                        key=f"day_heatmap_{chart_context}_{drill_metric}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Worst 15 test days by RMSE**")
        worst = daily.nlargest(15, "rmse_pu").copy()
        worst["coverage_percent"] = 100.0 * worst["coverage_95"]
        dashboard_dataframe(
            friendly_table(worst[["season", "day", "mae_pu", "rmse_pu", "coverage_percent", "mean_interval_width_95_pu"]]),
            hide_index=True,
            width="stretch",
        )
    with right:
        st.markdown("**Lowest 15 test days by coverage**")
        low = daily.nsmallest(15, "coverage_95").copy()
        low["coverage_percent"] = 100.0 * low["coverage_95"]
        dashboard_dataframe(
            friendly_table(low[["season", "day", "mae_pu", "rmse_pu", "coverage_percent", "mean_interval_width_95_pu"]]),
            hide_index=True,
            width="stretch",
        )

# -----------------------------------------------------------------------------
# Tab 3: Network structure and uncertainty
# -----------------------------------------------------------------------------
with main_tabs[2]:
    st.subheader("Network Structure and Predictive Uncertainty")
    dashboard_net = load_dashboard_network(network)
    n_lines = len(dashboard_net.line) if "line" in dashboard_net else 0
    n_trafos = len(dashboard_net.trafo) if "trafo" in dashboard_net else 0
    n_loads = len(dashboard_net.load) if "load" in dashboard_net else 0
    n_sgens = len(dashboard_net.sgen) if "sgen" in dashboard_net else 0
    overview = st.columns(5)
    overview[0].metric("Buses", len(dashboard_net.bus))
    overview[1].metric("Lines", n_lines)
    overview[2].metric("Transformers", n_trafos)
    overview[3].metric("Loads", n_loads)
    overview[4].metric("Distributed generators", n_sgens)

    # Small live-topology identity audit: these names come directly from the
    # currently loaded SimBench network, not from saved evaluation metadata.
    live_reference_buses = get_reference_buses(dashboard_net)
    live_reference_names = [live_bus_label(dashboard_net, bus) for bus in live_reference_buses]
    topology_note = "External grid / reference: " + ", ".join(live_reference_names)
    if "trafo" in dashboard_net and not dashboard_net.trafo.empty:
        trafo_links = []
        for _, row in dashboard_net.trafo.iterrows():
            trafo_links.append(
                f"{live_bus_label(dashboard_net, int(row['hv_bus']))} ↔ "
                f"{live_bus_label(dashboard_net, int(row['lv_bus']))}"
            )
        topology_note += "  |  Transformer: " + "; ".join(trafo_links)
    st.caption(topology_note)

    left_control, right_control = st.columns([2.2, 1.0])
    with left_control:
        network_view = st.radio(
            "Network view",
            ["Network Structure", "GPR Uncertainty"],
            horizontal=True,
            key="network_view",
        )
    with right_control:
        show_labels = st.checkbox("Show bus labels", value=False, key="network_labels")

    uncertainty = {}
    color_range = None
    map_title = "Electrical Topology with Load Types"
    eval_bus_override = (
        annual_config.get("evaluation_buses", [])
        if selected_scope == PRIMARY_SCOPE
        else annual_config.get("all_evaluation_buses", annual_config.get("prediction_buses", []))
    )

    if network_view == "GPR Uncertainty":
        mode = st.radio(
            "Uncertainty aggregation",
            ["Selected timestep", "Daily mean", "All seasonal test days mean"],
            horizontal=True,
            key="uncertainty_mode",
        )
        if mode == "All seasonal test days mean":
            uncertainty = uncertainty_from_annual_bus_metrics(annual_eval)
            color_range = _safe_color_range(uncertainty.values())
            map_title = "Pooled Mean 95% Prediction-Interval Width"
        else:
            map_days = annual_eval["daily"]["day"].astype(int).tolist()
            default_map_day = int(annual_overall["worst_day_by_rmse"])
            map_day = st.selectbox(
                "Test day",
                map_days,
                index=map_days.index(default_map_day) if default_map_day in map_days else 0,
                key="network_day",
            )
            try:
                map_detail = load_annual_day_detail(str(annual_eval["run_dir"]), int(map_day),
                                                   day_detail_signature(annual_eval["run_dir"], map_day))
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                st.error(f"Network uncertainty data could not be verified: {exc}")
                map_detail = pd.DataFrame()
            map_detail = filter_detail_scope(map_detail, selected_scope)
            if mode == "Daily mean":
                uncertainty = uncertainty_from_daily_mean(annual_eval, map_day, map_detail)
                color_range = _safe_color_range(uncertainty.values())
                map_title = f"Day {int(map_day)} — Mean 95% PI Width"
            elif not map_detail.empty:
                step_table = map_detail[["time_step", "time"]].drop_duplicates().sort_values("time_step")
                lookup = dict(zip(step_table["time_step"].astype(int), step_table["time"].astype(str)))
                steps = step_table["time_step"].astype(int).tolist()
                selected_step = st.select_slider(
                    "15-minute timestep",
                    options=steps,
                    value=steps[0],
                    format_func=lambda step: lookup[int(step)],
                    key="network_step",
                )
                uncertainty = uncertainty_from_timestep(map_detail, selected_step)
                color_range = _safe_color_range(map_detail["pi_width_pu"].astype(float))
                map_title = f"Day {int(map_day)}, {lookup[int(selected_step)]} — 95% PI Width"

        if uncertainty:
            values = pd.Series(uncertainty, dtype=float).dropna()
            if not values.empty:
                u1, u2, u3, u4 = st.columns(4)
                u1.metric("Mean PI Width", f"{values.mean():.8f} p.u.")
                u2.metric("Maximum PI Width", f"{values.max():.8f} p.u.")
                u3.metric("Highest-Uncertainty Bus", bus_display_label(annual_config, int(values.idxmax())))
                u4.metric("Colored Buses", len(values))

    dashboard_plotly_chart(
        network_professional_figure(
            dashboard_net,
            annual_config,
            view_mode=network_view,
            uncertainty=uncertainty,
            title=map_title,
            show_labels=show_labels,
            color_range=color_range,
            evaluation_buses_override=eval_bus_override,
        ),
        width="stretch",
    )
    if network_view == "Network Structure":
        st.caption(
            "Load type is encoded at the bus. The red open ring identifies the fixed measured buses. "
            "There is intentionally no separate Measurement Layout view."
        )
    else:
        st.caption(
            "Bus color represents raw-GPR 95% prediction-interval width for the currently selected evaluation scope."
        )

    with st.expander("Network element details"):
        bus_summary = network_component_summary(dashboard_net)
        load_inventory, generator_inventory, trafo_inventory = network_inventory_tables(dashboard_net)

        detail_context = dashboard_bus_context(
            dashboard_net,
            annual_config,
            evaluation_buses_override=eval_bus_override,
        )
        measured_set = detail_context["measured_buses"]
        reference_set = detail_context["reference_buses"]

        bus_summary["measurement_role"] = bus_summary["bus"].astype(int).map(
            lambda bus: "Reference" if bus in reference_set else "Measured" if bus in measured_set else "Unmeasured"
        )
        bus_summary["bus_label"] = bus_summary["bus"].astype(int).map(
            lambda bus: live_bus_label(dashboard_net, bus)
        )
        dashboard_dataframe(
            friendly_table(hide_internal_bus_columns(bus_summary)),
            hide_index=True,
            width="stretch",
        )
        if not load_inventory.empty:
            st.markdown("**Load elements**")
            load_display = load_inventory.copy()
            load_display["bus_label"] = load_display["bus"].astype(int).map(
                lambda bus: live_bus_label(dashboard_net, bus)
            )
            dashboard_dataframe(
                friendly_table(hide_internal_bus_columns(load_display)),
                hide_index=True,
                width="stretch",
            )
        if not generator_inventory.empty:
            st.markdown("**Distributed generation**")
            generator_display = generator_inventory.copy()
            generator_display["bus_label"] = generator_display["bus"].astype(int).map(
                lambda bus: live_bus_label(dashboard_net, bus)
            )
            dashboard_dataframe(
                friendly_table(hide_internal_bus_columns(generator_display)),
                hide_index=True,
                width="stretch",
            )
        if not trafo_inventory.empty:
            st.markdown("**Transformers**")
            trafo_display = trafo_inventory.copy()
            if "hv_bus" in trafo_display.columns:
                trafo_display["hv_bus"] = trafo_display["hv_bus"].astype(int).map(
                    lambda bus: live_bus_label(dashboard_net, bus)
                )
            if "lv_bus" in trafo_display.columns:
                trafo_display["lv_bus"] = trafo_display["lv_bus"].astype(int).map(
                    lambda bus: live_bus_label(dashboard_net, bus)
                )
            dashboard_dataframe(friendly_table(trafo_display), hide_index=True, width="stretch")

# -----------------------------------------------------------------------------
# Tab 4: Stress cases
# -----------------------------------------------------------------------------
with main_tabs[3]:
    st.subheader("Winter Physical Stress Cases" if winter_independent else "Annual Physical Stress Cases")
    st.caption(
        "Each already identified stress timestep is routed by day-of-year to the matching saved seasonal GPR. "
        "A case is explicitly marked as seen if its day belongs to that season's saved training days."
    )
    physical = load_year_analysis(
        base_eval["network_dir"],
        run_dir=base_eval["run_dir"] if winter_independent or annual_noise is False else None,
    )
    stress_run = load_stress_run(selected_annual_path)
    if physical["cases"].empty:
        st.info(f"Run `{stress_command}` to create this scenario's annual physical stress-case files.")
    else:
        case_table = physical["cases"].copy()
        case_table["season"] = case_table["day"].astype(int).map(season_for_day)
        if stress_run is not None and not stress_run["summary"].empty:
            seen_lookup = stress_run["summary"].set_index("case")["seen_in_training"].to_dict()
            case_table["seen_in_training"] = case_table["case"].map(seen_lookup).fillna(False)
        dashboard_dataframe(
            friendly_table(case_table[[c for c in ["case", "season", "day", "time", "time_step", "seen_in_training"] if c in case_table.columns]]),
            hide_index=True,
            width="stretch",
        )

        if stress_run is None:
            st.info(f"Run `{stress_command}` to evaluate these timesteps with the matching seasonal GPRs.")
        else:
            case_names = stress_run["summary"]["case"].tolist()
            selected_case = st.selectbox(
                "Stress case",
                case_names,
                format_func=case_label,
                key="stress_case",
            )
            row = stress_run["summary"][stress_run["summary"]["case"] == selected_case].iloc[0]
            status = "SEEN in seasonal training" if bool(row["seen_in_training"]) else "UNSEEN"
            st.caption(
                f"Routing: Day {int(row['day'])} → **{str(row['season']).title()} GPR** · status: **{status}**"
            )
            metrics = stress_metrics(stress_run, selected_case, selected_scope)
            if metrics is not None:
                g1, g2, g3, g4 = st.columns(4)
                g1.metric("MAE", f"{metrics['mae_pu']:.8f} p.u.")
                g2.metric("RMSE", f"{metrics['rmse_pu']:.8f} p.u.")
                g3.metric("95% Coverage", f"{100 * metrics['coverage_95']:.2f}%")
                g4.metric("Mean 95% PI Width", f"{metrics['mean_pi_width_pu']:.8f} p.u.")
            dashboard_plotly_chart(
                extreme_figure(stress_run, selected_case, physical["profiles"]),
                width="stretch",
            )
            if not stress_run.get("case_metrics", pd.DataFrame()).empty:
                display = stress_run["case_metrics"]
                display = display[display["evaluation_scope"] == selected_scope].copy()
                display["coverage_percent"] = 100.0 * display["voltage_coverage_95"]
                cols = [
                    "case", "season", "day", "time", "seen_in_training", "n_predictions",
                    "voltage_mae_pu", "voltage_rmse_pu", "coverage_percent",
                    "voltage_mean_interval_width_95_pu",
                ]
                dashboard_dataframe(friendly_table(display[cols]), hide_index=True, width="stretch")

# -----------------------------------------------------------------------------
# Tab 5: Kernel diagnostics
# -----------------------------------------------------------------------------
with main_tabs[4]:
    st.subheader("Kernel Covariance Diagnostics")
    if winter_independent:
        st.caption("Choose a measured or unmeasured target bus to inspect its own fitted winter kernel. No model is retrained here.")
    st.caption(
        "The fitted kernel is ConstantKernel × RBF + WhiteKernel. The selected test day "
        "determines which saved seasonal GPR and scaler are inspected; no retraining is "
        "performed here. Similarity heatmaps use only the fitted ConstantKernel × RBF "
        "signal component, while the WhiteKernel noise level is reported separately."
    )
    kernel_daily = annual_eval["daily"].copy()
    kernel_days = kernel_daily["day"].astype(int).tolist()
    default_kernel_day = int(kernel_daily.iloc[len(kernel_daily) // 2]["day"])
    kernel_day = st.selectbox(
        "Inspect unseen test day",
        kernel_days,
        index=kernel_days.index(default_kernel_day) if default_kernel_day in kernel_days else 0,
        key="kernel_day",
    )
    kernel_season = season_for_day(kernel_day)
    model_file = annual_config.get("trained_model_files", {}).get(kernel_season, f"trained_gpr_{kernel_season}.joblib")
    if winter_independent:
        winter_target_bus = st.selectbox(
            "Kernel target bus", annual_config["prediction_buses"], key="winter_kernel_bus",
            format_func=lambda bus: (
                f"{bus_display_label(annual_config, bus)} · "
                + ("Measured" if int(bus) in annual_config["measured_buses"] else "Unmeasured")
            ),
        )
        model_file = annual_config["independent_model_files"][f"bus_{int(winter_target_bus)}_vm_pu"]
    model_path = Path(annual_eval["run_dir"]) / model_file
    config_path = Path(annual_eval["run_dir"]) / "run_config.json"
    if not model_path.exists():
        st.error(f"Saved {kernel_season} GPR is missing. Run `python {evaluation_script}`.")
    else:
        if winter_independent:
            diag = load_winter_kernel_diagnostics_cached(
                str(annual_eval["run_dir"]), int(kernel_day), int(winter_target_bus),
                model_path.stat().st_mtime, config_path.stat().st_mtime,
            )
        else:
            diag = load_kernel_diagnostics_cached(
                str(annual_eval["run_dir"]),
                int(kernel_day),
                selected_scope,
                model_path.stat().st_mtime,
                config_path.stat().st_mtime if config_path.exists() else None,
            )
        st.markdown("### Fitted kernel")
        st.code(diag["kernel_string"], language="text")
        k1, k2, k3 = st.columns(3)
        k1.metric("Constant value", f"{diag['constant_value']:.6g}")
        k2.metric("RBF length scale", f"{diag['rbf_length_scale']:.6g}")
        k3.metric("White noise level", f"{diag['white_noise_level']:.6g}")
        if winter_independent:
            st.caption(
                f"Day {kernel_day}: independent winter GPR for {bus_display_label(annual_config, winter_target_bus)}. "
                "True-state comparison uses this model's single target voltage."
            )
        else:
            st.caption(
                f"Day {kernel_day} belongs to the {kernel_season.title()} GPR. True-state comparison uses "
                f"{diag['n_evaluation_buses']} buses from the selected evaluation scope."
            )
        st.markdown("### 1. Train–Train covariance")
        dashboard_plotly_chart(train_train_heatmap_figure(diag), width="stretch")
        st.markdown("### 2. Test–Train covariance")
        dashboard_plotly_chart(test_train_heatmap_figure(diag), width="stretch")
        st.markdown("### 3. Test–Test covariance vs. true voltage-state structure")
        left, right = st.columns(2)
        with left:
            dashboard_plotly_chart(test_test_heatmap_figure(diag), width="stretch")
        with right:
            dashboard_plotly_chart(true_voltage_state_difference_heatmap_figure(diag), width="stretch")
        st.metric(
            "Test–Test covariance vs. true voltage-state difference (Spearman)",
            f"{diag['test_test_structure_spearman']:.3f}",
        )
        st.caption(
            "Signal-kernel sanity check: maximum symmetry error = "
            f"{diag['test_test_symmetry_error']:.3e}; diagonal range = "
            f"[{diag['test_test_diagonal_min']:.6g}, {diag['test_test_diagonal_max']:.6g}]. "
            "For ConstantKernel × RBF, this signal-kernel diagonal should equal the "
            "fitted ConstantKernel value."
        )

st.sidebar.markdown("---")
st.sidebar.write(
    f"**Primary result:** Winter · {int(annual_config['n_models'])} independent GPRs"
    if winter_independent else "**Primary result:** Four seasonal GPRs"
)
st.sidebar.write(f"**Kernel:** {annual_config.get('kernel', 'Unknown')}")
st.sidebar.write(f"**Noise:** {noise_label(annual_noise)}")
st.sidebar.caption(
    f"{int(annual_config.get('n_training_days_per_season', 30))} train days per season; all remaining same-season days are test days"
)
