"""Noise-free counterpart of seasonal_evaluation.py, using the same model and splits.

Run from the project directory: python seasonal_evaluation_noise_free.py
Uses the existing validated ground-truth cache; never creates or updates it.
All experiment outputs (including stress/physical diagnostics) use the existing
parallel *_clean run directory. The noisy entry points/configuration are unchanged.

The evaluation orchestration is intentionally preserved from the noisy entry
point; shared dataset, GPR, metrics, diagnostics and stress helpers are imported.
Only the measurement inputs bypass noise injection. Kernel optimization is unchanged.
"""

import json
import shutil
import time

import joblib
import numpy as np
import pandas as pd

from config import (
    GPR_ALPHA,
    KERNEL_NAME,
    KERNEL_RUN_LABEL,
    NETWORK,
    N_TRAINING_DAYS_PER_SEASON,
    RESOLUTION_MINUTES,
    SEASON_LABELS,
    SEASON_ORDER,
    SENSOR_RATIO,
    SENSOR_SEED,
    TEST_BATCH_DAYS,
)
from dsse import (
    DSSEGaussianProcess,
    GAUSSIAN_95_MULTIPLIER,
    build_voltage_dataset,
    bus_identity_records,
    attach_bus_identity_columns,
    evaluate,
    get_reference_buses,
    load_ground_truth,
    load_network,
    seasonal_run_directory,
    season_for_day,
    seasonal_train_test_split,
    select_measured_buses,
)

import dsse
from seasonal_evaluation import (
    PRIMARY_SCOPE,
    ALL_SCOPE,
    SCOPE_LABELS,
    chunked,
    empty_metric_accumulator,
    add_metrics_to_accumulator,
    finalize_metric_accumulator,
    empty_bus_accumulator,
    update_bus_accumulator,
    finalize_bus_metrics,
    daily_bus_metrics,
    save_daily_diagnostics,
)
from stress_analysis import (
    load_or_simulate_year,
    find_physical_cases,
    physical_voltage_profiles,
    build_stress_results,
    stress_overall_metrics,
    print_summary,
)

USE_MEASUREMENT_NOISE = False


def require_existing_ground_truth(net):
    """Check the existing cache without allowing rebuilds or identity rewrites."""
    timeline = dsse.build_simbench_timeline(net)
    checks = dsse.validate_simbench_timeline(timeline)
    report = dsse._cached_ground_truth_report(net, NETWORK, checks)
    if report is None:
        raise RuntimeError(
            "Noise-free baseline requires the existing validated clean ground-truth "
            "cache and its original SimBench/pandapower versions. No ground truth "
            "was generated or modified. Use the same cache/environment as the noisy run."
        )
    identity_path = dsse.ground_truth_paths(NETWORK)["bus_identity"]
    saved = pd.read_csv(identity_path)
    canonical = pd.DataFrame(bus_identity_records(net))
    dsse.assert_saved_bus_identity_compatible(
        net, saved.to_dict(orient="records"),
        context="noise-free ground truth", require_internal_index_match=True,
    )
    comparable = ["internal_bus", "simbench_bus_number", "simbench_name", "bus_label"]
    if not saved.reindex(columns=comparable).equals(canonical[comparable]):
        raise RuntimeError(
            "The existing ground-truth bus identity would require a metadata rewrite. "
            "The noise-free baseline refuses to modify ground-truth files."
        )
    return report


def main():
    net = load_network(NETWORK)

    # Reuse the validated clean cache; never regenerate it for this baseline.
    ground_truth_report = require_existing_ground_truth(net)
    n_available_days = int(ground_truth_report["total_days"])

    identity_records = bus_identity_records(net)
    identity_map = {int(row["internal_bus"]): row for row in identity_records}
    all_buses = [int(bus) for bus in net.bus.index]
    reference_buses = get_reference_buses(net)
    measured_buses = select_measured_buses(net)

    if n_available_days < 366:
        raise RuntimeError(
            "The requested seasonal split uses day 366, but the loaded SimBench "
            f"profiles contain only {n_available_days} complete days."
        )

    splits = seasonal_train_test_split(n_available_days)
    out_dir = seasonal_run_directory(NETWORK, use_noise=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = out_dir / "daily_diagnostics"
    if diagnostics_dir.exists():
        shutil.rmtree(diagnostics_dir)
    # Stress predictions depend on the saved fitted models. Remove stale stress
    # results whenever the four seasonal models are refitted.
    stale_stress = out_dir / "stress_cases"
    if stale_stress.exists():
        shutil.rmtree(stale_stress)

    print("=" * 88)
    print("FOUR SEASON-SPECIFIC GPR EVALUATION")
    print("=" * 88)
    print(f"Network                    : {NETWORK}")
    print(f"Available complete days    : {n_available_days}")
    print(f"Training days per season   : {N_TRAINING_DAYS_PER_SEASON}")
    print("Test-day rule              : every remaining day in the same season")
    print("Winter definition          : Day 1-60 + Day 336-366")
    print("Spring definition          : Day 61-152")
    print("Summer definition          : Day 153-244")
    print("Autumn definition          : Day 245-335")
    print("Input                       : P,Q,V at fixed measured buses")
    print("Output                      : V at all non-reference buses")
    print(f"Kernel                      : {KERNEL_NAME}")
    print(f"GPR alpha                   : {GPR_ALPHA:.1e} (numerical jitter only)")
    print("Primary evaluation          : unmeasured buses only")
    print("Secondary evaluation        : all predicted buses incl. measured buses")
    print("95% PI                      : mean ± 1.96 sigma (raw GPR)")
    print(f"Measurement noise           : {'ON' if USE_MEASUREMENT_NOISE else 'OFF'}")
    print("Bus display                 : original SimBench bus identity")

    # Fix bus ordering once for all four models.
    probe_day = splits[SEASON_ORDER[0]]["training_days"][0]
    probe_truth = load_ground_truth(net, NETWORK, [probe_day])
    _, _, _, target_buses, unmeasured_buses = build_voltage_dataset(
        probe_truth,
        measured_buses,
        all_buses,
        reference_buses,
    )
    target_columns = [f"bus_{bus}_vm_pu" for bus in target_buses]
    scope_bus_map = {
        PRIMARY_SCOPE: list(unmeasured_buses),
        ALL_SCOPE: list(target_buses),
    }
    scope_column_map = {
        scope: [f"bus_{bus}_vm_pu" for bus in buses]
        for scope, buses in scope_bus_map.items()
    }

    all_bus_accumulator = empty_bus_accumulator(target_buses)
    pooled_accumulators = {
        scope: empty_metric_accumulator() for scope in SCOPE_LABELS
    }
    all_daily_rows = []
    all_daily_bus_rows = []
    season_rows = []
    kernel_rows = []
    model_files = {}
    total_fit_time = 0.0
    total_prediction_time = 0.0

    for season in SEASON_ORDER:
        label = SEASON_LABELS.get(season, season.title())
        training_days = splits[season]["training_days"]
        test_days = splits[season]["test_days"]

        print("\n" + "-" * 88)
        print(f"{label.upper()} GPR")
        print("-" * 88)
        print(f"Training days ({len(training_days)}): {training_days}")
        print(f"Test days     ({len(test_days)}): {test_days}")
        print("Unused days   (0): none")

        training_truth = load_ground_truth(net, NETWORK, training_days)
        X_train_clean, y_train, training_meta, batch_targets, batch_eval = (
            build_voltage_dataset(
                training_truth,
                measured_buses,
                all_buses,
                reference_buses,
            )
        )
        if batch_targets != target_buses or batch_eval != unmeasured_buses:
            raise RuntimeError(f"Bus ordering changed in {season} training data.")

        X_train = X_train_clean
        Y_train = y_train[target_columns]

        fit_start = time.perf_counter()
        model = DSSEGaussianProcess().fit(X_train, Y_train)
        fit_time = time.perf_counter() - fit_start
        total_fit_time += fit_time
        print(f"Fit time                   : {fit_time:.1f} s")
        print(f"Optimized kernel           : {model.model.kernel_}")

        model_filename = f"trained_gpr_{season}.joblib"
        joblib.dump(model, out_dir / model_filename)
        model_files[season] = model_filename

        kernel_summary = model.kernel_summary().copy()
        kernel_summary.insert(0, "season", season)
        kernel_rows.append(kernel_summary)

        season_accumulators = {
            scope: empty_metric_accumulator() for scope in SCOPE_LABELS
        }
        season_daily_by_scope = {scope: [] for scope in SCOPE_LABELS}
        prediction_start = time.perf_counter()
        processed_days = 0
        season_test_timestep_count = 0

        for batch_days in chunked(test_days, TEST_BATCH_DAYS):
            print(
                f"Evaluating {label} days {batch_days[0]}-{batch_days[-1]} "
                f"({processed_days}/{len(test_days)} completed) ..."
            )
            truth = load_ground_truth(net, NETWORK, batch_days)
            X_test_clean, y_test, test_meta, batch_targets, batch_eval = (
                build_voltage_dataset(
                    truth,
                    measured_buses,
                    all_buses,
                    reference_buses,
                )
            )
            if batch_targets != target_buses or batch_eval != unmeasured_buses:
                raise RuntimeError(f"Bus ordering changed in {season} test batches.")

            season_test_timestep_count += int(len(test_meta))
            X_test = X_test_clean
            prediction, sigma = model.predict(X_test)
            Y_test = y_test[target_columns]

            update_bus_accumulator(
                all_bus_accumulator,
                Y_test,
                prediction,
                sigma,
                target_buses,
            )
            all_daily_bus_rows.extend(
                daily_bus_metrics(
                    Y_test,
                    prediction,
                    sigma,
                    test_meta,
                    target_buses,
                    batch_days,
                    season,
                    measured_buses,
                )
            )
            save_daily_diagnostics(
                Y_test,
                prediction,
                sigma,
                test_meta,
                target_buses,
                measured_buses,
                batch_days,
                season,
                diagnostics_dir,
                identity_records,
            )

            for scope in SCOPE_LABELS:
                columns = scope_column_map[scope]
                Y_scope = Y_test[columns]
                pred_scope = prediction[columns]
                sigma_scope = sigma[columns]
                add_metrics_to_accumulator(
                    season_accumulators[scope], Y_scope, pred_scope, sigma_scope
                )
                add_metrics_to_accumulator(
                    pooled_accumulators[scope], Y_scope, pred_scope, sigma_scope
                )

                for day in batch_days:
                    day_index = test_meta.index[
                        test_meta["day"].astype(int).eq(int(day))
                    ]
                    metrics = evaluate(
                        Y_scope.loc[day_index],
                        pred_scope.loc[day_index],
                        sigma_scope.loc[day_index],
                    )
                    row = {
                        "evaluation_scope": scope,
                        "evaluation_scope_label": SCOPE_LABELS[scope],
                        "season": season,
                        "season_label": label,
                        "day": int(day),
                        "n_evaluation_buses": len(scope_bus_map[scope]),
                        "mae_pu": metrics["mae"],
                        "rmse_pu": metrics["rmse"],
                        "coverage_95": metrics["coverage_95"],
                        "mean_interval_width_95_pu": metrics[
                            "mean_interval_width_95"
                        ],
                    }
                    all_daily_rows.append(row)
                    season_daily_by_scope[scope].append(row)

            processed_days += len(batch_days)

        prediction_time = time.perf_counter() - prediction_start
        total_prediction_time += prediction_time

        for scope in SCOPE_LABELS:
            season_metrics = finalize_metric_accumulator(season_accumulators[scope])
            season_daily = pd.DataFrame(season_daily_by_scope[scope]).sort_values("day")
            season_rows.append(
                {
                    "evaluation_scope": scope,
                    "evaluation_scope_label": SCOPE_LABELS[scope],
                    "season": season,
                    "season_label": label,
                    "n_available_season_days": len(splits[season]["all_days"]),
                    "n_training_days": len(training_days),
                    "n_test_days": len(test_days),
                    "n_unused_days": 0,
                    "n_training_samples": len(X_train),
                    "n_test_timesteps": int(season_test_timestep_count),
                    "n_prediction_buses": len(target_buses),
                    "n_evaluation_buses": len(scope_bus_map[scope]),
                    "n_evaluated_voltage_predictions": season_metrics["n_predictions"],
                    "voltage_mae_pu": season_metrics["mae_pu"],
                    "voltage_rmse_pu": season_metrics["rmse_pu"],
                    "voltage_coverage_95": season_metrics["coverage_95"],
                    "voltage_mean_interval_width_95_pu": season_metrics[
                        "mean_interval_width_95_pu"
                    ],
                    "mean_daily_rmse_pu": float(season_daily["rmse_pu"].mean()),
                    "std_daily_rmse_pu": float(season_daily["rmse_pu"].std(ddof=0)),
                    "mean_daily_coverage_95": float(
                        season_daily["coverage_95"].mean()
                    ),
                    "std_daily_coverage_95": float(
                        season_daily["coverage_95"].std(ddof=0)
                    ),
                    "worst_day_by_rmse": int(
                        season_daily.loc[season_daily["rmse_pu"].idxmax(), "day"]
                    ),
                    "worst_day_rmse_pu": float(season_daily["rmse_pu"].max()),
                    "lowest_coverage_day": int(
                        season_daily.loc[season_daily["coverage_95"].idxmin(), "day"]
                    ),
                    "lowest_daily_coverage_95": float(
                        season_daily["coverage_95"].min()
                    ),
                    "fit_time_seconds": fit_time,
                    "prediction_time_seconds": prediction_time,
                    "trained_model_file": model_filename,
                }
            )

    daily = pd.DataFrame(all_daily_rows).sort_values(
        ["evaluation_scope", "day"]
    ).reset_index(drop=True)
    daily_bus = pd.DataFrame(all_daily_bus_rows).sort_values(
        ["day", "bus"]
    ).reset_index(drop=True)
    daily_bus = attach_bus_identity_columns(
        daily_bus, identity_records, bus_column="bus"
    )
    bus_metrics = finalize_bus_metrics(all_bus_accumulator, measured_buses)
    bus_metrics = attach_bus_identity_columns(
        bus_metrics, identity_records, bus_column="bus"
    )
    seasonal_metrics = pd.DataFrame(season_rows)
    kernels = pd.concat(kernel_rows, ignore_index=True)

    pooled_rows = []
    total_training_days = sum(len(splits[s]["training_days"]) for s in SEASON_ORDER)
    total_test_days = sum(len(splits[s]["test_days"]) for s in SEASON_ORDER)
    total_training_samples = sum(
        int(row["n_training_samples"])
        for row in season_rows
        if row["evaluation_scope"] == PRIMARY_SCOPE
    )
    total_test_timesteps = sum(
        int(row["n_test_timesteps"])
        for row in season_rows
        if row["evaluation_scope"] == PRIMARY_SCOPE
    )
    expected_full_year_steps = int(ground_truth_report["total_timesteps"])
    if total_training_samples + total_test_timesteps != expected_full_year_steps:
        raise RuntimeError(
            "Seasonal train/test timestep accounting does not cover the full SimBench "
            f"timeline: train={total_training_samples}, test={total_test_timesteps}, "
            f"full_year={expected_full_year_steps}."
        )

    for scope in SCOPE_LABELS:
        metrics = finalize_metric_accumulator(pooled_accumulators[scope])
        daily_scope = daily[daily["evaluation_scope"] == scope].copy()
        pooled_rows.append(
            {
                "evaluation_scope": scope,
                "evaluation_scope_label": SCOPE_LABELS[scope],
                "n_models": len(SEASON_ORDER),
                "n_available_days": n_available_days,
                "n_training_days_per_season": N_TRAINING_DAYS_PER_SEASON,
                "n_training_days": total_training_days,
                "n_test_days": total_test_days,
                "n_unused_days": 0,
                "n_training_samples": total_training_samples,
                "n_test_timesteps": total_test_timesteps,
                "n_prediction_buses": len(target_buses),
                "n_evaluation_buses": len(scope_bus_map[scope]),
                "n_evaluated_voltage_predictions": metrics["n_predictions"],
                "voltage_mae_pu": metrics["mae_pu"],
                "voltage_rmse_pu": metrics["rmse_pu"],
                "voltage_coverage_95": metrics["coverage_95"],
                "voltage_mean_interval_width_95_pu": metrics[
                    "mean_interval_width_95_pu"
                ],
                "mean_daily_rmse_pu": float(daily_scope["rmse_pu"].mean()),
                "std_daily_rmse_pu": float(daily_scope["rmse_pu"].std(ddof=0)),
                "mean_daily_coverage_95": float(daily_scope["coverage_95"].mean()),
                "std_daily_coverage_95": float(
                    daily_scope["coverage_95"].std(ddof=0)
                ),
                "worst_day_by_rmse": int(
                    daily_scope.loc[daily_scope["rmse_pu"].idxmax(), "day"]
                ),
                "worst_day_rmse_pu": float(daily_scope["rmse_pu"].max()),
                "lowest_coverage_day": int(
                    daily_scope.loc[daily_scope["coverage_95"].idxmin(), "day"]
                ),
                "lowest_daily_coverage_95": float(
                    daily_scope["coverage_95"].min()
                ),
                "fit_time_seconds": total_fit_time,
                "prediction_time_seconds": total_prediction_time,
            }
        )
    pooled_overall = pd.DataFrame(pooled_rows)

    seasonal_metrics.to_csv(out_dir / "seasonal_metrics.csv", index=False)
    pooled_overall.to_csv(out_dir / "pooled_overall_metrics.csv", index=False)
    # Compatibility file keeps the primary thesis scope only.
    pooled_overall[pooled_overall["evaluation_scope"] == PRIMARY_SCOPE].to_csv(
        out_dir / "annual_overall_metrics.csv", index=False
    )
    daily.to_csv(out_dir / "daily_metrics.csv", index=False)
    daily_bus.to_csv(out_dir / "daily_bus_metrics.csv", index=False)
    bus_metrics.to_csv(out_dir / "all_bus_metrics.csv", index=False)
    bus_metrics[bus_metrics["bus_role"] == "unmeasured"].to_csv(
        out_dir / "unmeasured_bus_metrics.csv", index=False
    )
    kernels.to_csv(out_dir / "voltage_kernel_hyperparameters.csv", index=False)
    pd.DataFrame(identity_records).to_csv(out_dir / "bus_identity.csv", index=False)

    pd.DataFrame(
        [
            {"season": season, "training_day": day}
            for season in SEASON_ORDER
            for day in splits[season]["training_days"]
        ]
    ).to_csv(out_dir / "training_days.csv", index=False)
    pd.DataFrame(
        [
            {"season": season, "test_day": day}
            for season in SEASON_ORDER
            for day in splits[season]["test_days"]
        ]
    ).to_csv(out_dir / "test_days.csv", index=False)

    stale_unused = out_dir / "unused_days.csv"
    if stale_unused.exists():
        stale_unused.unlink()

    training_days_by_season = {
        season: splits[season]["training_days"] for season in SEASON_ORDER
    }
    test_days_by_season = {
        season: splits[season]["test_days"] for season in SEASON_ORDER
    }
    run_config = {
        "network": NETWORK,
        "experiment": "four_season_specific_constant_rbf_white_gprs_30_train_all_remaining_test_days",
        "available_complete_days": n_available_days,
        "ground_truth_schema_version": ground_truth_report.get("schema_version"),
        "ground_truth_total_timesteps": ground_truth_report.get("total_timesteps"),
        "ground_truth_validation": ground_truth_report,
        "season_definition": {
            "winter": "day 1-60 and day 336-366",
            "spring": "day 61-152",
            "summer": "day 153-244",
            "autumn": "day 245-335",
        },
        "season_order": SEASON_ORDER,
        "n_models": 4,
        "n_training_days_per_season": N_TRAINING_DAYS_PER_SEASON,
        "training_day_strategy": "distributed_within_each_season",
        "test_day_strategy": "all_remaining_days_from_the_same_season",
        "unused_day_strategy": "none_all_366_days_are_training_or_test",
        "training_days_by_season": training_days_by_season,
        "test_days_by_season": test_days_by_season,
        "n_test_days_by_season": {
            season: len(test_days_by_season[season]) for season in SEASON_ORDER
        },
        "training_days": sorted(
            day for values in training_days_by_season.values() for day in values
        ),
        "test_days": sorted(
            day for values in test_days_by_season.values() for day in values
        ),
        "unused_days": [],
        "resolution_minutes": RESOLUTION_MINUTES,
        "sensor_ratio": SENSOR_RATIO,
        "sensor_seed": SENSOR_SEED,
        "measured_buses": measured_buses,
        "reference_buses": reference_buses,
        "prediction_buses": target_buses,
        "bus_identity": identity_records,
        "bus_display_policy": "SimBench name/number for all user-facing output; pandapower index remains internal",
        "measured_bus_labels": [identity_map[bus]["bus_label"] for bus in measured_buses],
        "reference_bus_labels": [identity_map[bus]["bus_label"] for bus in reference_buses],
        "prediction_bus_labels": [identity_map[bus]["bus_label"] for bus in target_buses],
        "evaluation_buses": unmeasured_buses,
        "all_evaluation_buses": target_buses,
        "evaluation_scopes": {
            PRIMARY_SCOPE: unmeasured_buses,
            ALL_SCOPE: target_buses,
        },
        "primary_evaluation_scope": PRIMARY_SCOPE,
        "model_type": "four_season_specific_multioutput_gprs",
        "outputs_correlated": False,
        "kernel": KERNEL_NAME,
        "kernel_run_label": KERNEL_RUN_LABEL,
        "kernel_signal_component": "ConstantKernel * RBF",
        "kernel_noise_component": "WhiteKernel",
        "gpr_alpha": GPR_ALPHA,
        "alpha_role": "numerical_jitter_only",
        "kernel_structure_fixed_for_seasonal_experiment": True,
        "uncertainty_method": "raw_gpr_gaussian_95",
        "gaussian_95_multiplier": GAUSSIAN_95_MULTIPLIER,
        "use_measurement_noise": USE_MEASUREMENT_NOISE,
        "noise_strategy": "none_clean_measurements",
        "trained_model_files": model_files,
        "daily_bus_metrics_file": "daily_bus_metrics.csv",
        "daily_diagnostics_directory": "daily_diagnostics",
        "seasonal_metrics_file": "seasonal_metrics.csv",
        "pooled_metrics_file": "pooled_overall_metrics.csv",
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2)

    print("\n" + "=" * 88)
    print("SEASONAL EVALUATION SUMMARY")
    print("=" * 88)
    for scope in SCOPE_LABELS:
        print(f"\n{SCOPE_LABELS[scope].upper()}")
        scope_seasons = seasonal_metrics[
            seasonal_metrics["evaluation_scope"] == scope
        ]
        for _, row in scope_seasons.iterrows():
            print(
                f"{row['season_label']:>8}: "
                f"MAE={row['voltage_mae_pu']:.8f}, "
                f"RMSE={row['voltage_rmse_pu']:.8f}, "
                f"coverage={100 * row['voltage_coverage_95']:.2f}%, "
                f"PI width={row['voltage_mean_interval_width_95_pu']:.8f}"
            )
        pooled = pooled_overall[pooled_overall["evaluation_scope"] == scope].iloc[0]
        print(
            f"  POOLED: MAE={pooled['voltage_mae_pu']:.8f}, "
            f"RMSE={pooled['voltage_rmse_pu']:.8f}, "
            f"coverage={100 * pooled['voltage_coverage_95']:.2f}%, "
            f"PI width={pooled['voltage_mean_interval_width_95_pu']:.8f}"
        )

    print(f"\nTotal fit time          : {total_fit_time:.1f} s")
    print(f"Total prediction time   : {total_prediction_time:.1f} s")
    print(f"Saved                   : {out_dir}")


def seasonal_config():
    run_dir = seasonal_run_directory(NETWORK, use_noise=False)
    with open(run_dir / "run_config.json", encoding="utf-8") as file:
        return run_dir, json.load(file)


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
    X_case = X_case_clean

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
        "noise_strategy": "none_clean_measurements",
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


def run_stress_analysis():
    """Produce the corresponding dashboard stress data without touching noisy outputs."""
    # Physical extrema are independent of measurement noise. Reuse their saved
    # selection where available; otherwise use the unchanged physical helpers.
    shared_dir = seasonal_run_directory(NETWORK, use_noise=False).parent.parent / "year_analysis"
    filenames = ["yearly_summary.csv", "extreme_cases.csv", "extreme_voltage_profiles.csv"]
    if all((shared_dir / name).exists() for name in filenames):
        summary, cases, profiles = [pd.read_csv(shared_dir / name) for name in filenames]
    else:
        summary, net, payload, analysis_buses = load_or_simulate_year()
        cases = find_physical_cases(summary)
        profiles = physical_voltage_profiles(net, payload, cases, analysis_buses)

    physical_dir = seasonal_run_directory(NETWORK, use_noise=False) / "year_analysis"
    physical_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(filenames, [summary, cases, profiles]):
        frame.to_csv(physical_dir / name, index=False)

    case_summary, case_metrics, overall, out_dir = evaluate_stress_cases(cases)
    print_summary(cases, case_summary, case_metrics, overall, out_dir)


if __name__ == "__main__":
    main()
    run_stress_analysis()
