"""Winter-only experiment: one independent GPR per non-reference target bus.

Run from seasonal/: python winter_independent_evaluation.py
Optional: --scenario noisy, noise-free, or both. The default follows config.py.
Requires the existing validated ground-truth cache; never rebuilds that cache.
Every run creates a new output directory and leaves the four-season runs alone.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import simbench as sb
from scipy.stats import spearmanr

from config import (
    GPR_ALPHA, KERNEL_NAME, KERNEL_RUN_LABEL, MEASUREMENTS, NETWORK,
    N_TRAINING_DAYS_PER_SEASON, OUTPUT_ROOT, RESOLUTION_MINUTES,
    SENSOR_RATIO, SENSOR_SEED, TEST_BATCH_DAYS, TEST_NOISE_SEED,
    TRAIN_NOISE_SEED, USE_MEASUREMENT_NOISE,
)
from dsse import (
    DSSEGaussianProcess, GAUSSIAN_95_MULTIPLIER, add_measurement_noise,
    assert_saved_bus_identity_compatible, attach_bus_identity_columns,
    build_simbench_timeline, build_voltage_dataset,
    bus_identity_records, day_noise_seed, evaluate, get_reference_buses,
    load_full_year_powerflow_state, load_ground_truth, load_network, safe_name,
    seasonal_train_test_split, select_measured_buses,
)
from kernel_diagnostics import _even_indices, _pairwise_euclidean
from seasonal_evaluation import (
    ALL_SCOPE, PRIMARY_SCOPE, SCOPE_LABELS, add_metrics_to_accumulator,
    chunked, daily_bus_metrics, empty_bus_accumulator, empty_metric_accumulator,
    finalize_bus_metrics, finalize_metric_accumulator, save_daily_diagnostics,
    update_bus_accumulator,
)
from seasonal_evaluation_noise_free import require_existing_ground_truth
from stress_analysis import (
    _absolute_total_series, build_stress_results, find_physical_cases,
    physical_voltage_profiles, stress_overall_metrics,
)

MODEL_TYPE = "winter_independent_singleoutput_gprs"


def measurement_inputs(clean, metadata, use_noise, seed):
    """Use the existing noise formula and day seeds without changing config."""
    if not use_noise:
        return clean.copy()
    result = clean.copy()
    for day in sorted(metadata.loc[clean.index, "day"].astype(int).unique()):
        indices = metadata.index[metadata["day"].astype(int).eq(day)]
        indices = clean.index.intersection(indices, sort=False)
        result.loc[indices] = add_measurement_noise(clean.loc[indices], day_noise_seed(seed, day))
    return result


def _run_file(run_dir, name):
    root = Path(run_dir).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("A winter model file must belong to its selected run.")
    return path


class WinterModelCollection:
    """Predict with independently saved scalar models, loading one at a time."""

    def __init__(self, run_dir, config):
        self.run_dir = Path(run_dir)
        manifest_path = _run_file(run_dir, config["trained_model_files"]["winter"])
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.target_columns = self.manifest["target_columns"]
        self.input_columns = self.manifest["input_columns"]
        expected = [f"bus_{int(bus)}_vm_pu" for bus in config["prediction_buses"]]
        if self.target_columns != expected:
            raise ValueError("Winter manifest has a different target-bus ordering.")
        if set(self.manifest["model_files"]) != set(expected):
            raise ValueError("Winter manifest must contain one model for every target bus.")

    def load_model(self, column):
        model = joblib.load(_run_file(self.run_dir, self.manifest["model_files"][column]))
        if model.target_columns != [column]:
            raise ValueError(f"Saved model does not exclusively predict {column}.")
        if list(model.scaler.feature_names_in_) != self.input_columns:
            raise ValueError("Saved model does not use this run's P/Q/V input ordering.")
        return model

    def predict(self, inputs):
        if list(inputs.columns) != self.input_columns:
            raise ValueError("Winter predictions require the same input columns as training.")
        means, deviations = {}, {}
        for column in self.target_columns:
            prediction, sigma = self.load_model(column).predict(inputs)
            means[column], deviations[column] = prediction[column], sigma[column]
        return pd.DataFrame(means, index=inputs.index), pd.DataFrame(deviations, index=inputs.index)


def load_winter_models(run_dir, config):
    return WinterModelCollection(run_dir, config)


def _new_run_directory(use_noise):
    parent = Path(OUTPUT_ROOT) / safe_name(NETWORK) / "seasonal_evaluation"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    scenario = "noisy" if use_noise else "clean"
    name = (f"winter_independent_pqv_{KERNEL_RUN_LABEL}_"
            f"train_{N_TRAINING_DAYS_PER_SEASON:02d}_{scenario}_{stamp}")
    directory = parent / name
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def save_winter_stress(run_dir, config, net, winter_days):
    """Use only winter extrema from the existing clean physical cache."""
    payload = load_full_year_powerflow_state(net, NETWORK)
    timeline = payload["timeline"]
    timeline = timeline[timeline["day_of_year"].astype(int).isin(winter_days)].copy()
    steps = timeline["simbench_step"].astype(int).tolist()
    buses = config["prediction_buses"]
    voltage = payload["states"]["vm_pu"].loc[steps, buses]
    absolute = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    profile_steps = build_simbench_timeline(net)["simbench_step"].astype(int).tolist()
    loads = _absolute_total_series(absolute, "load", "p_mw", profile_steps)
    reactive_loads = _absolute_total_series(absolute, "load", "q_mvar", profile_steps)
    generation = _absolute_total_series(absolute, "sgen", "p_mw", profile_steps)
    summary = timeline.copy()
    summary["time_step"] = summary["simbench_step"]
    summary["profile_index"] = summary["simbench_step"].astype(str)
    summary["day"] = summary["day_of_year"]
    summary["time"] = summary["timestamp"].astype(str).str[-5:]
    summary["max_vm_pu"] = voltage.max(axis=1).to_numpy()
    summary["min_vm_pu"] = voltage.min(axis=1).to_numpy()
    summary["max_vm_bus"] = voltage.idxmax(axis=1).to_numpy()
    summary["min_vm_bus"] = voltage.idxmin(axis=1).to_numpy()
    summary["total_load_mw"] = loads.loc[steps].to_numpy()
    summary["total_load_mvar"] = reactive_loads.loc[steps].to_numpy()
    summary["total_sgen_mw"] = generation.loc[steps].to_numpy()
    summary["net_demand_mw"] = summary["total_load_mw"] - summary["total_sgen_mw"]
    identities = {int(row["internal_bus"]): row for row in config["bus_identity"]}
    for kind in ("max", "min"):
        for suffix, field in (("label", "bus_label"), ("name", "simbench_name")):
            summary[f"{kind}_vm_bus_{suffix}"] = summary[f"{kind}_vm_bus"].map(
                lambda bus: identities[int(bus)][field])
    cases = find_physical_cases(summary)
    cases["season"] = "winter"
    cases["seen_in_training"] = cases["day"].isin(config["training_days"])
    profiles = physical_voltage_profiles(net, payload, cases, buses)
    physical_dir = Path(run_dir) / "year_analysis"
    physical_dir.mkdir()
    for filename, frame in (("yearly_summary.csv", summary), ("extreme_cases.csv", cases),
                            ("extreme_voltage_profiles.csv", profiles)):
        frame.to_csv(physical_dir / filename, index=False)

    truth = load_ground_truth(net, NETWORK, sorted(set(cases["day"].astype(int))))
    X, Y, metadata, targets, unmeasured = build_voltage_dataset(
        truth, config["measured_buses"], [int(bus) for bus in net.bus.index], config["reference_buses"])
    selected = list(dict.fromkeys(cases["time_step"].astype(int)))
    inputs = measurement_inputs(X.loc[selected], metadata.loc[selected],
                                config["use_measurement_noise"], TEST_NOISE_SEED)
    prediction, sigma = load_winter_models(run_dir, config).predict(inputs)
    data = {"measured_buses": config["measured_buses"], "unmeasured_buses": unmeasured,
            "target_buses": targets, "training_days_by_season": config["training_days_by_season"],
            "bus_identity": config["bus_identity"], "y_voltage": Y}
    case_summary, case_metrics, predictions = build_stress_results(cases, data, prediction, sigma)
    directory = Path(run_dir) / "stress_cases"
    directory.mkdir()
    for filename, frame in (("stress_case_summary.csv", case_summary),
                            ("stress_case_metrics.csv", case_metrics),
                            ("stress_predictions.csv", predictions),
                            ("stress_overall_metrics.csv", stress_overall_metrics(case_metrics, predictions))):
        frame.to_csv(directory / filename, index=False)
    inputs.rename_axis("time_step").reset_index().to_csv(directory / "stress_measurements.csv", index=False)
    stress_config = {**config, "stress_case_definition": "four_winter_physical_extremes_only"}
    (directory / "stress_config.json").write_text(json.dumps(stress_config, indent=2), encoding="utf-8")


def run_experiment(use_noise=USE_MEASUREMENT_NOISE):
    if MEASUREMENTS != ["p_mw", "q_mvar", "vm_pu"]:
        raise ValueError("This comparison requires the existing P/Q/V-only measurement configuration.")
    net = load_network(NETWORK)
    report = require_existing_ground_truth(net)
    split = seasonal_train_test_split(int(report["total_days"]))["winter"]
    train_days, test_days = split["training_days"], split["test_days"]
    identities = bus_identity_records(net)
    all_buses = [int(bus) for bus in net.bus.index]
    measured, references = select_measured_buses(net), get_reference_buses(net)
    training_truth = load_ground_truth(net, NETWORK, train_days)
    X_clean, Y, metadata, targets, unmeasured = build_voltage_dataset(
        training_truth, measured, all_buses, references)
    X = measurement_inputs(X_clean, metadata, use_noise, TRAIN_NOISE_SEED)
    output = _new_run_directory(use_noise)
    models_dir = output / "models"
    models_dir.mkdir()
    files, kernel_rows, fit_times = {}, [], {}
    print(f"Winter independent GPRs: {len(targets)} models; {X.shape[1]} P/Q/V inputs each")
    print(f"Training days: {train_days}; test days: {test_days}")
    for position, bus in enumerate(targets, 1):
        column = f"bus_{bus}_vm_pu"
        print(f"[{position}/{len(targets)}] Fitting target bus {bus}", flush=True)
        started = time.perf_counter()
        model = DSSEGaussianProcess().fit(X, Y[[column]])
        fit_times[bus] = time.perf_counter() - started
        filename = f"models/trained_gpr_winter_bus_{bus}.joblib"
        joblib.dump(model, output / filename)
        files[column] = filename
        row = model.kernel_summary().iloc[0].to_dict()
        row.update(season="winter", bus=bus, bus_role="measured" if bus in measured else "unmeasured",
                   model_type="independent_singleoutput_gpr", fit_time_seconds=fit_times[bus])
        kernel_rows.append(row)
    manifest = {"input_columns": list(X.columns), "target_columns": list(Y.columns), "model_files": files}
    (output / "winter_models.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    config = {
        "network": NETWORK, "model_type": MODEL_TYPE, "experiment": "winter_one_gpr_per_target_bus",
        "season_order": ["winter"], "season_definition": {"winter": "day 1-60 and day 336-366"},
        "n_models": len(targets), "n_training_days_per_season": N_TRAINING_DAYS_PER_SEASON,
        "available_complete_days": int(report["total_days"]), "ground_truth_validation": report,
        "ground_truth_schema_version": report.get("schema_version"),
        "ground_truth_total_timesteps": report.get("total_timesteps"),
        "training_days_by_season": {"winter": train_days}, "test_days_by_season": {"winter": test_days},
        "training_days": train_days, "test_days": test_days, "unused_days": [],
        "n_test_days_by_season": {"winter": len(test_days)},
        "training_day_strategy": "distributed_within_each_season",
        "test_day_strategy": "all_remaining_days_from_the_same_season",
        "evaluation_window": "winter_only", "resolution_minutes": RESOLUTION_MINUTES,
        "sensor_ratio": SENSOR_RATIO, "sensor_seed": SENSOR_SEED,
        "measured_buses": measured, "reference_buses": references, "prediction_buses": targets,
        "bus_identity": identities, "evaluation_buses": unmeasured, "all_evaluation_buses": targets,
        "evaluation_scopes": {PRIMARY_SCOPE: unmeasured, ALL_SCOPE: targets},
        "primary_evaluation_scope": PRIMARY_SCOPE, "outputs_correlated": False,
        "kernel": KERNEL_NAME, "kernel_run_label": KERNEL_RUN_LABEL, "gpr_alpha": GPR_ALPHA,
        "kernel_signal_component": "ConstantKernel * RBF", "kernel_noise_component": "WhiteKernel",
        "alpha_role": "numerical_jitter_only", "uncertainty_method": "raw_gpr_gaussian_95",
        "gaussian_95_multiplier": GAUSSIAN_95_MULTIPLIER, "use_measurement_noise": bool(use_noise),
        "noise_strategy": "deterministic_day_specific_seeds" if use_noise else "none_clean_measurements",
        "input_columns": list(X.columns), "measurements": list(MEASUREMENTS),
        "trained_model_files": {"winter": "winter_models.json"}, "independent_model_files": files,
        "daily_bus_metrics_file": "daily_bus_metrics.csv", "daily_diagnostics_directory": "daily_diagnostics",
        "seasonal_metrics_file": "seasonal_metrics.csv", "pooled_metrics_file": "pooled_overall_metrics.csv",
    }
    collection = load_winter_models(output, config)
    scopes = {PRIMARY_SCOPE: unmeasured, ALL_SCOPE: targets}
    accumulators = {scope: empty_metric_accumulator() for scope in scopes}
    bus_accumulator = empty_bus_accumulator(targets)
    daily_rows, bus_rows = [], []
    test_steps = 0
    started = time.perf_counter()
    for batch in chunked(test_days, TEST_BATCH_DAYS):
        truth = load_ground_truth(net, NETWORK, batch)
        X_batch, Y_batch, meta, batch_targets, batch_unmeasured = build_voltage_dataset(
            truth, measured, all_buses, references)
        if (batch_targets, batch_unmeasured) != (targets, unmeasured):
            raise ValueError("Winter test data changed its bus ordering.")
        inputs = measurement_inputs(X_batch, meta, use_noise, TEST_NOISE_SEED)
        prediction, sigma = collection.predict(inputs)
        test_steps += len(inputs)
        update_bus_accumulator(bus_accumulator, Y_batch, prediction, sigma, targets)
        bus_rows.extend(daily_bus_metrics(Y_batch, prediction, sigma, meta, targets, batch, "winter", measured))
        save_daily_diagnostics(Y_batch, prediction, sigma, meta, targets, measured, batch,
                               "winter", output / "daily_diagnostics", identities)
        for scope, scope_buses in scopes.items():
            columns = [f"bus_{bus}_vm_pu" for bus in scope_buses]
            add_metrics_to_accumulator(accumulators[scope], Y_batch[columns], prediction[columns], sigma[columns])
            for day in batch:
                indices = meta.index[meta["day"].astype(int).eq(day)]
                metrics = evaluate(Y_batch.loc[indices, columns], prediction.loc[indices, columns], sigma.loc[indices, columns])
                daily_rows.append({"evaluation_scope": scope, "evaluation_scope_label": SCOPE_LABELS[scope],
                                   "season": "winter", "season_label": "Winter", "day": day,
                                   "n_evaluation_buses": len(scope_buses), "mae_pu": metrics["mae"],
                                   "rmse_pu": metrics["rmse"], "coverage_95": metrics["coverage_95"],
                                   "mean_interval_width_95_pu": metrics["mean_interval_width_95"]})
    prediction_seconds = time.perf_counter() - started
    timeline = build_simbench_timeline(net)
    expected_steps = len(timeline[timeline["day_of_year"].astype(int).isin(split["all_days"])])
    if len(X) + test_steps != expected_steps:
        raise RuntimeError("Winter training/test samples do not cover exactly the winter timeline.")
    daily = pd.DataFrame(daily_rows).sort_values(["evaluation_scope", "day"])
    daily_bus = attach_bus_identity_columns(pd.DataFrame(bus_rows).sort_values(["day", "bus"]), identities)
    bus_metrics = attach_bus_identity_columns(finalize_bus_metrics(bus_accumulator, measured), identities)
    kernels = attach_bus_identity_columns(pd.DataFrame(kernel_rows), identities)
    season_rows, pooled_rows = [], []
    for scope, scope_buses in scopes.items():
        metrics = finalize_metric_accumulator(accumulators[scope])
        scoped = daily[daily.evaluation_scope.eq(scope)]
        row = {"evaluation_scope": scope, "evaluation_scope_label": SCOPE_LABELS[scope],
               "season": "winter", "season_label": "Winter", "n_models": len(targets),
               "n_available_days": len(split["all_days"]), "n_available_season_days": len(split["all_days"]),
               "n_training_days_per_season": N_TRAINING_DAYS_PER_SEASON, "n_training_days": len(train_days),
               "n_test_days": len(test_days), "n_unused_days": 0, "n_training_samples": len(X),
               "n_test_timesteps": test_steps, "n_prediction_buses": len(targets),
               "n_evaluation_buses": len(scope_buses), "n_evaluated_voltage_predictions": metrics["n_predictions"],
               "voltage_mae_pu": metrics["mae_pu"], "voltage_rmse_pu": metrics["rmse_pu"],
               "voltage_coverage_95": metrics["coverage_95"],
               "voltage_mean_interval_width_95_pu": metrics["mean_interval_width_95_pu"],
               "mean_daily_rmse_pu": float(scoped.rmse_pu.mean()), "std_daily_rmse_pu": float(scoped.rmse_pu.std(ddof=0)),
               "mean_daily_coverage_95": float(scoped.coverage_95.mean()),
               "std_daily_coverage_95": float(scoped.coverage_95.std(ddof=0)),
               "worst_day_by_rmse": int(scoped.loc[scoped.rmse_pu.idxmax(), "day"]),
               "worst_day_rmse_pu": float(scoped.rmse_pu.max()),
               "lowest_coverage_day": int(scoped.loc[scoped.coverage_95.idxmin(), "day"]),
               "lowest_daily_coverage_95": float(scoped.coverage_95.min()),
               "fit_time_seconds": sum(fit_times.values()), "prediction_time_seconds": prediction_seconds}
        season_rows.append(row)
        pooled_rows.append(row)
    role_rows = []
    for role, frame in bus_metrics.groupby("bus_role", sort=False):
        counts = frame.n_predictions.to_numpy(float)
        role_rows.append({"bus_role": role, "n_buses": len(frame), "n_predictions": int(counts.sum()),
                          "mae_pu": float(np.average(frame.mae_pu, weights=counts)),
                          "rmse_pu": float(np.sqrt(np.average(frame.rmse_pu ** 2, weights=counts))),
                          "coverage_95": float(np.average(frame.coverage_95, weights=counts)),
                          "mean_interval_width_95_pu": float(np.average(frame.mean_interval_width_95_pu, weights=counts))})
    for filename, frame in (("pooled_overall_metrics.csv", pd.DataFrame(pooled_rows)),
                            ("seasonal_metrics.csv", pd.DataFrame(season_rows)), ("daily_metrics.csv", daily),
                            ("daily_bus_metrics.csv", daily_bus), ("all_bus_metrics.csv", bus_metrics),
                            ("unmeasured_bus_metrics.csv", bus_metrics[bus_metrics.bus_role.eq("unmeasured")]),
                            ("voltage_kernel_hyperparameters.csv", kernels), ("bus_role_metrics.csv", pd.DataFrame(role_rows)),
                            ("bus_identity.csv", pd.DataFrame(identities)),
                            ("training_days.csv", pd.DataFrame({"season": "winter", "training_day": train_days})),
                            ("test_days.csv", pd.DataFrame({"season": "winter", "test_day": test_days}))):
        frame.to_csv(output / filename, index=False)
    save_winter_stress(output, config, net, split["all_days"])
    # Publish run metadata last, so incomplete runs are not listed by the dashboard.
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Completed winter-only run: {output}")
    print(pd.DataFrame(role_rows).to_string(index=False))
    return output


def compute_winter_kernel_diagnostics(run_dir, selected_day, target_bus, max_train_overview=320):
    """Inspect the actual independently optimized kernel for the selected bus."""
    config = json.loads((Path(run_dir) / "run_config.json").read_text(encoding="utf-8"))
    day, bus = int(selected_day), int(target_bus)
    if day not in config["test_days_by_season"]["winter"] or bus not in config["prediction_buses"]:
        raise ValueError("Select a saved winter test day and a target bus from this run.")
    collection = load_winter_models(run_dir, config)
    column = f"bus_{bus}_vm_pu"
    model = collection.load_model(column)
    net = load_network(config["network"])
    assert_saved_bus_identity_compatible(net, config["bus_identity"], context="independent winter model")
    all_buses = [int(value) for value in net.bus.index]
    matrices = []
    for days, seed in ((config["training_days"], TRAIN_NOISE_SEED), ([day], TEST_NOISE_SEED)):
        truth = load_ground_truth(net, config["network"], days)
        X, Y, metadata, _, _ = build_voltage_dataset(truth, config["measured_buses"], all_buses, config["reference_buses"])
        X = measurement_inputs(X, metadata, config["use_measurement_noise"], seed)
        if list(X.columns) != collection.input_columns:
            raise ValueError("Kernel diagnostics input columns do not match the fitted winter model.")
        matrices.append((X, Y, metadata))
    X_train, _, train_meta = matrices[0]
    X_test, y_test, test_meta = matrices[1]
    train_scaled, test_scaled = model.scaler.transform(X_train), model.scaler.transform(X_test)
    fitted = model.model.kernel_
    signal, white = fitted.k1, fitted.k2
    overview = _even_indices(len(X_train), max_train_overview)
    K_train = signal(train_scaled[overview])
    K_cross = signal(test_scaled, train_scaled)
    K_test = signal(test_scaled)
    differences = _pairwise_euclidean(y_test[[column]].to_numpy(float))
    upper = np.triu_indices_from(K_test, k=1)
    covariance, distances = K_test[upper], differences[upper]
    correlation = (float(spearmanr(covariance, distances).statistic)
                   if len(covariance) > 1 and np.std(covariance) > 0 and np.std(distances) > 0 else float("nan"))
    return {"season": "winter", "selected_day": day, "target_bus": bus,
            "evaluation_scope": "selected_target_bus", "n_evaluation_buses": 1,
            "model_file": collection.manifest["model_files"][column],
            "kernel_string": str(fitted), "signal_kernel_string": str(signal),
            "constant_value": float(signal.k1.constant_value), "rbf_length_scale": float(signal.k2.length_scale),
            "white_noise_level": float(white.noise_level), "n_training_samples": len(X_train),
            "n_train_overview_samples": len(overview), "training_days": config["training_days"],
            "train_overview_days": train_meta.loc[X_train.index, "day"].astype(int).to_numpy()[overview],
            "test_times": test_meta.loc[X_test.index, "time"].astype(str).tolist(),
            "train_train_covariance": K_train, "test_train_covariance": K_cross, "test_test_covariance": K_test,
            "true_voltage_state_difference": differences, "test_test_structure_spearman": correlation,
            "test_test_symmetry_error": float(np.max(np.abs(K_test - K_test.T))),
            "test_test_diagonal_min": float(np.diag(K_test).min()),
            "test_test_diagonal_max": float(np.diag(K_test).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("noisy", "noise-free", "both"),
                        default="noisy" if USE_MEASUREMENT_NOISE else "noise-free")
    args = parser.parse_args()
    scenarios = (True, False) if args.scenario == "both" else (args.scenario == "noisy",)
    for use_noise in scenarios:
        run_experiment(use_noise)


if __name__ == "__main__":
    main()
