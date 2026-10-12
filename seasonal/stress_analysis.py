"""Identify annual physical stress points and evaluate them with the matching seasonal GPR."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import simbench as sb

from config import (
    NETWORK,
    OUTPUT_ROOT,
    TEST_NOISE_SEED,
    USE_MEASUREMENT_NOISE,
)
from dsse import (
    GAUSSIAN_95_MULTIPLIER,
    add_noise_by_day,
    seasonal_run_directory,
    season_for_day,
    build_voltage_dataset,
    bus_identity_records,
    evaluate,
    get_reference_buses,
    load_ground_truth,
    load_full_year_powerflow_state,
    load_network,
    profile_key,
    safe_name,
)

CASE_LABELS = {
    "annual_max_voltage": "Maximum Voltage",
    "annual_min_voltage": "Minimum Voltage",
    "annual_max_total_load": "Maximum Total Load",
    "annual_max_net_demand": "Maximum Net Demand",
}


def case_label(case_name):
    return CASE_LABELS.get(str(case_name), str(case_name).replace("_", " ").title())


def year_analysis_directory():
    path = Path(OUTPUT_ROOT) / safe_name(NETWORK) / "year_analysis"
    path.mkdir(parents=True, exist_ok=True)
    return path


def seasonal_config():
    run_dir = seasonal_run_directory(NETWORK)
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found. Run `python seasonal_evaluation.py` first."
        )
    with open(config_path, encoding="utf-8") as file:
        return run_dir, json.load(file)


# ---------------------------------------------------------------------------
# 1. Full-year physical operating-point analysis
# ---------------------------------------------------------------------------


def _absolute_total_series(absolute_values, element, column, steps):
    """Sum an existing SimBench absolute profile over all elements of one type."""
    for key, frame in absolute_values.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        if profile_key(key) == (str(element), str(column)):
            if len(frame) != len(steps):
                raise RuntimeError(
                    f"Absolute profile {element}.{column} has {len(frame)} rows; "
                    f"expected {len(steps)}."
                )
            return pd.Series(
                frame.to_numpy(dtype=float).sum(axis=1),
                index=steps,
                dtype=float,
            )
    return pd.Series(0.0, index=steps, dtype=float)


def load_or_simulate_year():
    """Build annual physical summaries from the already computed clean PF cache.

    Despite the legacy function name, this function no longer runs any power
    flow.  The full-year AC power flow is produced exactly once by
    build_full_year_ground_truth()/load_full_year_powerflow_state().
    """
    net = load_network(NETWORK)
    payload = load_full_year_powerflow_state(net, NETWORK)
    timeline = payload["timeline"].copy().sort_values("simbench_step")
    steps = timeline["simbench_step"].astype(int).tolist()

    identity_records = bus_identity_records(net)
    identity_map = {int(row["internal_bus"]): row for row in identity_records}
    references = set(get_reference_buses(net))
    analysis_buses = [int(bus) for bus in net.bus.index if int(bus) not in references]

    vm = payload["states"]["vm_pu"].loc[steps, analysis_buses].copy()
    max_vm_bus = vm.idxmax(axis=1).astype(int)
    min_vm_bus = vm.idxmin(axis=1).astype(int)

    # Reuse SimBench's own absolute values for element totals.  This does not
    # rerun a power flow and does not invent any storage/SOC behavior.
    absolute_values = sb.get_absolute_values(
        net,
        profiles_instead_of_study_cases=True,
    )
    total_load_mw = _absolute_total_series(absolute_values, "load", "p_mw", steps)
    total_load_mvar = _absolute_total_series(absolute_values, "load", "q_mvar", steps)
    total_sgen_mw = _absolute_total_series(absolute_values, "sgen", "p_mw", steps)

    summary = pd.DataFrame(
        {
            "time_step": timeline["simbench_step"].astype(int).to_numpy(),
            "simbench_step": timeline["simbench_step"].astype(int).to_numpy(),
            "profile_index": timeline["simbench_step"].astype(str).to_numpy(),
            "timestamp": timeline["timestamp"].astype(str).to_numpy(),
            "date": timeline["date"].astype(str).to_numpy(),
            "day_of_year": timeline["day_of_year"].astype(int).to_numpy(),
            "step_in_day": timeline["step_in_day"].astype(int).to_numpy(),
            "day": timeline["day_of_year"].astype(int).to_numpy(),
            "time": timeline["timestamp"].astype(str).str[-5:].to_numpy(),
            "max_vm_pu": vm.max(axis=1).to_numpy(dtype=float),
            "max_vm_bus": max_vm_bus.to_numpy(dtype=int),
            "min_vm_pu": vm.min(axis=1).to_numpy(dtype=float),
            "min_vm_bus": min_vm_bus.to_numpy(dtype=int),
            "total_load_mw": total_load_mw.loc[steps].to_numpy(dtype=float),
            "total_load_mvar": total_load_mvar.loc[steps].to_numpy(dtype=float),
            "total_sgen_mw": total_sgen_mw.loc[steps].to_numpy(dtype=float),
        }
    )
    summary["net_demand_mw"] = summary["total_load_mw"] - summary["total_sgen_mw"]
    summary["max_vm_bus_label"] = summary["max_vm_bus"].map(
        lambda bus: identity_map[int(bus)]["bus_label"]
    )
    summary["max_vm_bus_name"] = summary["max_vm_bus"].map(
        lambda bus: identity_map[int(bus)]["simbench_name"]
    )
    summary["min_vm_bus_label"] = summary["min_vm_bus"].map(
        lambda bus: identity_map[int(bus)]["bus_label"]
    )
    summary["min_vm_bus_name"] = summary["min_vm_bus"].map(
        lambda bus: identity_map[int(bus)]["simbench_name"]
    )

    # These fields were informational only in the previous stress summary.  We
    # do not rerun PF merely to reconstruct them; the voltage stress analysis
    # below is entirely based on the cached clean full-year result.
    summary["max_line_loading_percent"] = np.nan
    summary["max_trafo_loading_percent"] = np.nan

    return summary, net, payload, analysis_buses


def find_physical_cases(summary):
    definitions = [
        ("annual_max_voltage", "max_vm_pu", "idxmax", "max_vm_bus"),
        ("annual_min_voltage", "min_vm_pu", "idxmin", "min_vm_bus"),
        ("annual_max_total_load", "total_load_mw", "idxmax", None),
        ("annual_max_net_demand", "net_demand_mw", "idxmax", None),
    ]
    rows = []
    for case_name, column, operation, bus_column in definitions:
        row = summary.loc[getattr(summary[column], operation)()]
        rows.append(
            {
                "case": case_name,
                "time_step": int(row["time_step"]),
                "profile_index": row["profile_index"],
                "day": int(row["day"]),
                "time": row["time"],
                "critical_bus": int(row[bus_column]) if bus_column else None,
                "critical_bus_label": (
                    row.get(f"{bus_column}_label") if bus_column else None
                ),
                "max_vm_pu": float(row["max_vm_pu"]),
                "max_vm_bus": int(row["max_vm_bus"]),
                "max_vm_bus_label": row.get("max_vm_bus_label"),
                "max_vm_bus_name": row.get("max_vm_bus_name"),
                "min_vm_pu": float(row["min_vm_pu"]),
                "min_vm_bus": int(row["min_vm_bus"]),
                "min_vm_bus_label": row.get("min_vm_bus_label"),
                "min_vm_bus_name": row.get("min_vm_bus_name"),
                "total_load_mw": float(row["total_load_mw"]),
                "total_sgen_mw": float(row["total_sgen_mw"]),
                "net_demand_mw": float(row["net_demand_mw"]),
            }
        )
    return pd.DataFrame(rows)


def physical_voltage_profiles(net, payload, cases, analysis_buses):
    """Select stress-case voltages from the clean annual PF cache; never rerun PF."""
    identity_map = {
        int(row["internal_bus"]): row for row in bus_identity_records(net)
    }
    vm = payload["states"]["vm_pu"]
    va = payload["states"]["va_degree"]
    rows = []
    for _, case in cases.iterrows():
        step = int(case["time_step"])
        for bus in analysis_buses:
            rows.append(
                {
                    "case": case["case"],
                    "time_step": step,
                    "day": int(case["day"]),
                    "time": case["time"],
                    "bus": int(bus),
                    "bus_label": identity_map[int(bus)]["bus_label"],
                    "simbench_name": identity_map[int(bus)]["simbench_name"],
                    "simbench_bus_number": identity_map[int(bus)]["simbench_bus_number"],
                    "vm_pu": float(vm.at[step, bus]),
                    "va_degree": float(va.at[step, bus]),
                }
            )
    return pd.DataFrame(rows)


def save_physical_analysis(summary, cases, profiles):
    out_dir = year_analysis_directory()
    summary.to_csv(out_dir / "yearly_summary.csv", index=False)
    cases.to_csv(out_dir / "extreme_cases.csv", index=False)
    profiles.to_csv(out_dir / "extreme_voltage_profiles.csv", index=False)


# ---------------------------------------------------------------------------
# 2. Matching season-specific GPR evaluated at the four physical stress points
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 2. Matching season-specific GPR evaluated at the four physical stress points
# ---------------------------------------------------------------------------

PRIMARY_SCOPE = "unmeasured_only"
ALL_SCOPE = "all_predicted_buses"
SCOPE_LABELS = {
    PRIMARY_SCOPE: "Unmeasured buses only",
    ALL_SCOPE: "All predicted buses (measured + unmeasured)",
}


def focus_bus_for_case(case):
    if pd.notna(case["critical_bus"]):
        return int(case["critical_bus"])
    return int(case["min_vm_bus"])


def prepare_stress_inputs(cases, config):
    net = load_network(NETWORK)
    all_buses = [int(bus) for bus in net.bus.index]
    measured_buses = [int(bus) for bus in config["measured_buses"]]
    reference_buses = [int(bus) for bus in config["reference_buses"]]
    expected_targets = [int(bus) for bus in config["prediction_buses"]]
    expected_unmeasured = [int(bus) for bus in config["evaluation_buses"]]

    case_days = sorted(set(int(day) for day in cases["day"]))
    truth = load_ground_truth(net, NETWORK, case_days)
    X_clean, y_voltage, metadata, target_buses, unmeasured_buses = build_voltage_dataset(
        truth,
        measured_buses,
        all_buses,
        reference_buses,
    )
    if target_buses != expected_targets or unmeasured_buses != expected_unmeasured:
        raise RuntimeError("Stress-case bus setup does not match the seasonal evaluation.")

    selected_steps = list(dict.fromkeys(int(step) for step in cases["time_step"]))
    missing = [step for step in selected_steps if step not in X_clean.index]
    if missing:
        raise RuntimeError(f"Selected stress timesteps not found: {missing}")

    X_case_clean = X_clean.loc[selected_steps].copy()
    case_meta = metadata.loc[selected_steps].copy()
    X_case = add_noise_by_day(X_case_clean, case_meta, TEST_NOISE_SEED)

    training_days_by_season = {
        str(season): sorted(int(day) for day in days)
        for season, days in config["training_days_by_season"].items()
    }
    identity_records = config.get("bus_identity") or bus_identity_records(net)
    return {
        "reference_buses": reference_buses,
        "measured_buses": measured_buses,
        "target_buses": target_buses,
        "unmeasured_buses": unmeasured_buses,
        "training_days_by_season": training_days_by_season,
        "bus_identity": identity_records,
        "X_case": X_case,
        "y_voltage": y_voltage,
    }


def pooled_metrics(frame):
    if frame.empty:
        return {
            "n_predictions": 0,
            "mae_pu": float("nan"),
            "rmse_pu": float("nan"),
            "coverage_95": float("nan"),
            "mean_interval_width_95_pu": float("nan"),
        }
    error = frame["predicted_vm_pu"].to_numpy(float) - frame[
        "ground_truth_vm_pu"
    ].to_numpy(float)
    true = frame["ground_truth_vm_pu"].to_numpy(float)
    lower = frame["lower_95_vm_pu"].to_numpy(float)
    upper = frame["upper_95_vm_pu"].to_numpy(float)
    return {
        "n_predictions": int(len(frame)),
        "mae_pu": float(np.mean(np.abs(error))),
        "rmse_pu": float(np.sqrt(np.mean(error**2))),
        "coverage_95": float(np.mean((true >= lower) & (true <= upper))),
        "mean_interval_width_95_pu": float(np.mean(upper - lower)),
    }


def build_stress_results(cases, data, prediction, sigma):
    identity_map = {
        int(row["internal_bus"]): row for row in data.get("bus_identity", [])
    }
    measured = set(data["measured_buses"])
    unmeasured = set(data["unmeasured_buses"])
    target_buses = data["target_buses"]
    case_rows = []
    prediction_rows = []

    for _, case in cases.iterrows():
        step = int(case["time_step"])
        day = int(case["day"])
        season = str(case["season"])
        training_days = set(data["training_days_by_season"][season])
        seen = bool(day in training_days)
        focus_bus = focus_bus_for_case(case)
        focus_col = f"bus_{focus_bus}_vm_pu"
        true_focus = float(data["y_voltage"].at[step, focus_col])
        pred_focus = float(prediction.at[step, focus_col])
        sigma_focus = float(sigma.at[step, focus_col])
        lower_focus = pred_focus - GAUSSIAN_95_MULTIPLIER * sigma_focus
        upper_focus = pred_focus + GAUSSIAN_95_MULTIPLIER * sigma_focus

        for bus in target_buses:
            column = f"bus_{bus}_vm_pu"
            true = float(data["y_voltage"].at[step, column])
            pred = float(prediction.at[step, column])
            std = float(sigma.at[step, column])
            lower = pred - GAUSSIAN_95_MULTIPLIER * std
            upper = pred + GAUSSIAN_95_MULTIPLIER * std
            prediction_rows.append(
                {
                    "case": case["case"],
                    "case_label": case_label(case["case"]),
                    "day": day,
                    "season": season,
                    "time": case["time"],
                    "time_step": step,
                    "seen_in_training": seen,
                    "bus": int(bus),
                    "bus_label": identity_map.get(int(bus), {}).get("bus_label", f"Internal bus {int(bus)}"),
                    "simbench_name": identity_map.get(int(bus), {}).get("simbench_name"),
                    "simbench_bus_number": identity_map.get(int(bus), {}).get("simbench_bus_number"),
                    "bus_role": "measured" if int(bus) in measured else "unmeasured",
                    "is_focus_bus": bool(int(bus) == focus_bus),
                    "ground_truth_vm_pu": true,
                    "predicted_vm_pu": pred,
                    "sigma_vm_pu": std,
                    "lower_95_vm_pu": lower,
                    "upper_95_vm_pu": upper,
                    "covered_95": bool(lower <= true <= upper),
                    "absolute_error_vm_pu": abs(pred - true),
                }
            )

        base = {
            "case": case["case"],
            "case_label": case_label(case["case"]),
            "day": day,
            "season": season,
            "time": case["time"],
            "time_step": step,
            "seen_in_training": seen,
            "focus_bus": focus_bus,
            "focus_bus_label": identity_map.get(focus_bus, {}).get("bus_label", f"Internal bus {focus_bus}"),
            "focus_bus_name": identity_map.get(focus_bus, {}).get("simbench_name"),
            "focus_bus_role": "measured" if focus_bus in measured else "unmeasured",
            "annual_max_vm_pu": float(case["max_vm_pu"]),
            "annual_min_vm_pu": float(case["min_vm_pu"]),
            "total_load_mw": float(case["total_load_mw"]),
            "net_demand_mw": float(case["net_demand_mw"]),
            "focus_true_vm_pu": true_focus,
            "focus_predicted_vm_pu": pred_focus,
            "focus_absolute_error_vm_pu": abs(pred_focus - true_focus),
            "focus_sigma_vm_pu": sigma_focus,
            "focus_lower_95_vm_pu": lower_focus,
            "focus_upper_95_vm_pu": upper_focus,
            "focus_covered_95_voltage": bool(lower_focus <= true_focus <= upper_focus),
        }
        case_rows.append(base)

    predictions = pd.DataFrame(prediction_rows)
    case_summary = pd.DataFrame(case_rows)

    metric_rows = []
    for _, case in case_summary.iterrows():
        case_frame = predictions[predictions["case"] == case["case"]].copy()
        for scope, label in SCOPE_LABELS.items():
            scoped = (
                case_frame[case_frame["bus_role"] == "unmeasured"].copy()
                if scope == PRIMARY_SCOPE
                else case_frame
            )
            metrics = pooled_metrics(scoped)
            metric_rows.append(
                {
                    "evaluation_scope": scope,
                    "evaluation_scope_label": label,
                    **case.to_dict(),
                    "n_predictions": metrics["n_predictions"],
                    "voltage_mae_pu": metrics["mae_pu"],
                    "voltage_rmse_pu": metrics["rmse_pu"],
                    "voltage_coverage_95": metrics["coverage_95"],
                    "voltage_mean_interval_width_95_pu": metrics[
                        "mean_interval_width_95_pu"
                    ],
                }
            )
    return case_summary, pd.DataFrame(metric_rows), predictions


def stress_overall_metrics(case_metrics, predictions):
    rows = []
    unique_cases = case_metrics.drop_duplicates("case")
    for evaluation_scope, evaluation_label in SCOPE_LABELS.items():
        frame = (
            predictions[predictions["bus_role"] == "unmeasured"].copy()
            if evaluation_scope == PRIMARY_SCOPE
            else predictions.copy()
        )
        all_cases = unique_cases["case"].tolist()
        unseen_cases = unique_cases.loc[
            ~unique_cases["seen_in_training"].astype(bool), "case"
        ].tolist()
        groups = [("all_cases", all_cases)]
        if unseen_cases:
            groups.append(("unseen_cases", unseen_cases))

        scope_metrics = case_metrics[
            case_metrics["evaluation_scope"] == evaluation_scope
        ]
        for case_scope, case_names in groups:
            scoped_predictions = frame[frame["case"].isin(case_names)].copy()
            metrics = pooled_metrics(scoped_predictions)
            scoped_cases = scope_metrics[scope_metrics["case"].isin(case_names)]
            worst = scoped_cases.loc[scoped_cases["voltage_rmse_pu"].idxmax()]
            low_cov = scoped_cases.loc[
                scoped_cases["voltage_coverage_95"].idxmin()
            ]
            rows.append(
                {
                    "evaluation_scope": evaluation_scope,
                    "evaluation_scope_label": evaluation_label,
                    "case_scope": case_scope,
                    "n_cases": int(len(case_names)),
                    **metrics,
                    "worst_case_by_rmse": str(worst["case"]),
                    "worst_case_rmse_pu": float(worst["voltage_rmse_pu"]),
                    "lowest_coverage_case": str(low_cov["case"]),
                    "lowest_case_coverage_95": float(low_cov["voltage_coverage_95"]),
                }
            )
    return pd.DataFrame(rows)


def evaluate_stress_cases(cases):
    run_dir, config = seasonal_config()
    model_files = config.get("trained_model_files", {})
    if not model_files:
        raise RuntimeError(
            "run_config.json has no seasonal model mapping. "
            "Run `python seasonal_evaluation.py` first."
        )

    data = prepare_stress_inputs(cases, config)
    expected_columns = [f"bus_{bus}_vm_pu" for bus in data["target_buses"]]

    cases_with_season = cases.copy()
    cases_with_season["season"] = cases_with_season["day"].astype(int).map(season_for_day)
    cases_with_season["seen_in_training"] = cases_with_season.apply(
        lambda row: int(row["day"]) in set(
            data["training_days_by_season"][str(row["season"])]
        ),
        axis=1,
    )

    prediction_parts = []
    sigma_parts = []
    kernel_parts = []
    for season, season_cases in cases_with_season.groupby("season", sort=False):
        model_filename = model_files.get(str(season))
        if not model_filename:
            raise FileNotFoundError(f"No saved model is configured for season '{season}'.")
        model_path = run_dir / model_filename
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run `python seasonal_evaluation.py` first."
            )

        model = joblib.load(model_path)
        if list(getattr(model, "target_columns", []) or []) != expected_columns:
            raise RuntimeError(
                f"Saved {season} GPR does not match the current stress-case bus setup."
            )

        steps = [int(step) for step in season_cases["time_step"].tolist()]
        print(f"Reusing fitted {season} GPR: {model_path}")
        pred, std = model.predict(data["X_case"].loc[steps])
        prediction_parts.append(pred)
        sigma_parts.append(std)

        k = model.kernel_summary().copy()
        k.insert(0, "season", season)
        kernel_parts.append(k)

    prediction = pd.concat(prediction_parts).sort_index()
    sigma = pd.concat(sigma_parts).sort_index()
    case_summary, case_metrics, predictions = build_stress_results(
        cases_with_season, data, prediction, sigma
    )
    overall = stress_overall_metrics(case_metrics, predictions)

    out_dir = run_dir / "stress_cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    case_summary.to_csv(out_dir / "stress_case_summary.csv", index=False)
    case_metrics.to_csv(out_dir / "stress_case_metrics.csv", index=False)
    predictions.to_csv(out_dir / "stress_predictions.csv", index=False)
    overall.to_csv(out_dir / "stress_overall_metrics.csv", index=False)
    pd.concat(kernel_parts, ignore_index=True).to_csv(
        out_dir / "voltage_kernel_hyperparameters.csv", index=False
    )

    measurements = data["X_case"].copy()
    measurements.index.name = "time_step"
    measurements.reset_index().to_csv(out_dir / "stress_measurements.csv", index=False)

    stress_config = {
        "network": NETWORK,
        "model_type": "four_season_specific_multioutput_gprs",
        "model_sources": model_files,
        "routing": "day -> season -> matching saved seasonal GPR",
        "kernel": config.get("kernel", "Unknown"),
        "n_training_days_per_season": config.get("n_training_days_per_season"),
        "training_days_by_season": data["training_days_by_season"],
        "output_state": "voltage_magnitude",
        "use_measurement_noise": USE_MEASUREMENT_NOISE,
        "noise_strategy": "deterministic_day_specific_seeds",
        "uncertainty_method": "raw_gpr_gaussian_95",
        "gaussian_95_multiplier": GAUSSIAN_95_MULTIPLIER,
        "measured_buses": data["measured_buses"],
        "reference_buses": data["reference_buses"],
        "prediction_buses": data["target_buses"],
        "bus_identity": data.get("bus_identity", []),
        "bus_display_policy": "SimBench name/number for all user-facing output; pandapower index remains internal",
        "evaluation_buses": data["unmeasured_buses"],
        "all_evaluation_buses": data["target_buses"],
        "evaluation_scopes": list(SCOPE_LABELS),
        "stress_case_definition": "four_annual_physical_extremes_only",
        "cases": cases_with_season[
            ["case", "season", "day", "time", "time_step", "seen_in_training"]
        ].to_dict("records"),
    }
    with open(out_dir / "stress_config.json", "w", encoding="utf-8") as file:
        json.dump(stress_config, file, indent=2)

    return case_summary, case_metrics, overall, out_dir


def print_summary(cases, case_summary, case_metrics, overall, out_dir):
    print("\n" + "=" * 82)
    print("ANNUAL PHYSICAL STRESS CASES — SEASON-ROUTED GPRS")
    print("=" * 82)
    for _, case in case_summary.iterrows():
        status = "SEEN IN TRAINING" if bool(case["seen_in_training"]) else "UNSEEN"
        print(
            f"{case_label(case['case'])}: {str(case['season']).title()} · "
            f"Day {int(case['day'])}, {case['time']} (step {int(case['time_step'])}) · {status}"
        )

    for scope, label in SCOPE_LABELS.items():
        print(f"\n{label.upper()}")
        subset = case_metrics[case_metrics["evaluation_scope"] == scope]
        for _, row in subset.iterrows():
            print(
                f"  {row['case_label']}: RMSE={row['voltage_rmse_pu']:.8f}, "
                f"coverage={100 * row['voltage_coverage_95']:.2f}%, "
                f"PI width={row['voltage_mean_interval_width_95_pu']:.8f}"
            )
        pooled = overall[
            (overall["evaluation_scope"] == scope)
            & (overall["case_scope"] == "all_cases")
        ].iloc[0]
        print(
            f"  POOLED: RMSE={pooled['rmse_pu']:.8f}, "
            f"coverage={100 * pooled['coverage_95']:.2f}%"
        )
    print(f"\nSaved: {out_dir}")


def main():
    summary, net, payload, analysis_buses = load_or_simulate_year()
    cases = find_physical_cases(summary)
    voltage_profiles = physical_voltage_profiles(net, payload, cases, analysis_buses)
    save_physical_analysis(summary, cases, voltage_profiles)

    case_summary, case_metrics, overall, out_dir = evaluate_stress_cases(cases)
    print_summary(cases, case_summary, case_metrics, overall, out_dir)


if __name__ == "__main__":
    main()
