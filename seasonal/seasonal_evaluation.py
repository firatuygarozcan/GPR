"""Four season-specific GPRs with learned residual-noise modeling and two evaluation scopes.

Each seasonal model uses exactly 30 distributed whole training days from its own
season. Every remaining day of that season is an unseen test day. The same four
models are evaluated in two ways without retraining:

1) unmeasured_only: only genuinely unmeasured, non-reference buses
2) all_predicted_buses: all non-reference output buses, including measured buses

The GPR target itself remains voltage magnitude at all non-reference buses.
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
    TEST_NOISE_SEED,
    TRAIN_NOISE_SEED,
    USE_MEASUREMENT_NOISE,
)
from dsse import (
    DSSEGaussianProcess,
    GAUSSIAN_95_MULTIPLIER,
    add_noise_by_day,
    build_full_year_ground_truth,
    build_voltage_dataset,
    bus_identity_records,
    attach_bus_identity_columns,
    evaluate,
    get_reference_buses,
    load_ground_truth,
    load_network,
    seasonal_run_directory,
    seasonal_train_test_split,
    select_measured_buses,
)

PRIMARY_SCOPE = "unmeasured_only"
ALL_SCOPE = "all_predicted_buses"
SCOPE_LABELS = {
    PRIMARY_SCOPE: "Unmeasured buses only",
    ALL_SCOPE: "All predicted buses (measured + unmeasured)",
}


def chunked(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def empty_metric_accumulator():
    return {
        "n": 0,
        "abs_error_sum": 0.0,
        "squared_error_sum": 0.0,
        "covered_sum": 0,
        "interval_width_sum": 0.0,
    }


def add_metrics_to_accumulator(accumulator, y_true, y_pred, sigma):
    true = y_true.to_numpy(float)
    pred = y_pred.to_numpy(float)
    std = sigma.to_numpy(float)
    error = pred - true
    lower = pred - GAUSSIAN_95_MULTIPLIER * std
    upper = pred + GAUSSIAN_95_MULTIPLIER * std

    accumulator["n"] += int(true.size)
    accumulator["abs_error_sum"] += float(np.abs(error).sum())
    accumulator["squared_error_sum"] += float((error**2).sum())
    accumulator["covered_sum"] += int(((true >= lower) & (true <= upper)).sum())
    accumulator["interval_width_sum"] += float((upper - lower).sum())


def finalize_metric_accumulator(accumulator):
    n = int(accumulator["n"])
    if n == 0:
        return {
            "n_predictions": 0,
            "mae_pu": float("nan"),
            "rmse_pu": float("nan"),
            "coverage_95": float("nan"),
            "mean_interval_width_95_pu": float("nan"),
        }
    return {
        "n_predictions": n,
        "mae_pu": accumulator["abs_error_sum"] / n,
        "rmse_pu": float(np.sqrt(accumulator["squared_error_sum"] / n)),
        "coverage_95": accumulator["covered_sum"] / n,
        "mean_interval_width_95_pu": accumulator["interval_width_sum"] / n,
    }


def empty_bus_accumulator(buses):
    return {int(bus): empty_metric_accumulator() for bus in buses}


def update_bus_accumulator(accumulator, y_true, y_pred, sigma, buses):
    for bus in buses:
        column = f"bus_{int(bus)}_vm_pu"
        add_metrics_to_accumulator(
            accumulator[int(bus)],
            y_true[[column]],
            y_pred[[column]],
            sigma[[column]],
        )


def finalize_bus_metrics(accumulator, measured_buses):
    measured_set = set(int(bus) for bus in measured_buses)
    rows = []
    for bus, values in sorted(accumulator.items()):
        metrics = finalize_metric_accumulator(values)
        rows.append(
            {
                "bus": int(bus),
                "bus_role": "measured" if int(bus) in measured_set else "unmeasured",
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def daily_bus_metrics(y_true, y_pred, sigma, metadata, buses, days, season, measured_buses):
    measured_set = set(int(bus) for bus in measured_buses)
    rows = []
    day_series = metadata["day"].astype(int)
    for day in days:
        day_index = metadata.index[day_series.eq(int(day))]
        for bus in buses:
            column = f"bus_{int(bus)}_vm_pu"
            metrics = evaluate(
                y_true.loc[day_index, [column]],
                y_pred.loc[day_index, [column]],
                sigma.loc[day_index, [column]],
            )
            rows.append(
                {
                    "season": season,
                    "day": int(day),
                    "bus": int(bus),
                    "bus_role": "measured" if int(bus) in measured_set else "unmeasured",
                    "n_timesteps": int(len(day_index)),
                    "mae_pu": metrics["mae"],
                    "rmse_pu": metrics["rmse"],
                    "coverage_95": metrics["coverage_95"],
                    "mean_interval_width_95_pu": metrics["mean_interval_width_95"],
                }
            )
    return rows


def save_daily_diagnostics(
    y_true,
    y_pred,
    sigma,
    metadata,
    buses,
    measured_buses,
    days,
    season,
    out_dir,
    identity_records,
):
    """Save all predicted non-reference buses once; dashboard filters evaluation scope."""
    out_dir.mkdir(parents=True, exist_ok=True)
    measured_set = set(int(bus) for bus in measured_buses)
    day_series = metadata["day"].astype(int)

    for day in days:
        day_index = metadata.index[day_series.eq(int(day))]
        meta = metadata.loc[day_index]
        records = []
        for bus in buses:
            column = f"bus_{int(bus)}_vm_pu"
            true = y_true.loc[day_index, column].to_numpy(float)
            pred = y_pred.loc[day_index, column].to_numpy(float)
            std = sigma.loc[day_index, column].to_numpy(float)
            lower = pred - GAUSSIAN_95_MULTIPLIER * std
            upper = pred + GAUSSIAN_95_MULTIPLIER * std
            records.append(
                pd.DataFrame(
                    {
                        "season": season,
                        "time_step": day_index.to_numpy(),
                        "simbench_step": (
                            meta["simbench_step"].astype(int).to_numpy()
                            if "simbench_step" in meta.columns
                            else day_index.to_numpy()
                        ),
                        "timestamp": (
                            meta["timestamp"].astype(str).to_numpy()
                            if "timestamp" in meta.columns
                            else meta["time"].astype(str).to_numpy()
                        ),
                        "date": (
                            meta["date"].astype(str).to_numpy()
                            if "date" in meta.columns
                            else np.repeat("", len(meta))
                        ),
                        "day_of_year": (
                            meta["day_of_year"].astype(int).to_numpy()
                            if "day_of_year" in meta.columns
                            else meta["day"].astype(int).to_numpy()
                        ),
                        "step_in_day": (
                            meta["step_in_day"].astype(int).to_numpy()
                            if "step_in_day" in meta.columns
                            else np.arange(len(meta), dtype=int)
                        ),
                        "time": meta["time"].astype(str).to_numpy(),
                        "bus": int(bus),
                        "bus_role": (
                            "measured" if int(bus) in measured_set else "unmeasured"
                        ),
                        "ground_truth_vm_pu": true,
                        "predicted_vm_pu": pred,
                        "sigma_vm_pu": std,
                        "lower_95_vm_pu": lower,
                        "upper_95_vm_pu": upper,
                        "absolute_error_vm_pu": np.abs(pred - true),
                        "covered_95": (true >= lower) & (true <= upper),
                        "pi_width_pu": upper - lower,
                    }
                )
            )
        day_frame = pd.concat(records, ignore_index=True)
        day_frame = attach_bus_identity_columns(
            day_frame, identity_records, bus_column="bus"
        )
        day_frame.to_csv(
            out_dir / f"day_{int(day):03d}.csv.gz",
            index=False,
            compression="gzip",
        )


def scope_buses(scope, target_buses, unmeasured_buses):
    if scope == PRIMARY_SCOPE:
        return list(unmeasured_buses)
    if scope == ALL_SCOPE:
        return list(target_buses)
    raise KeyError(f"Unknown evaluation scope: {scope}")


def main():
    net = load_network(NETWORK)

    # Physical ground truth is built first and is completely independent of
    # sensor placement, measurement noise and GPR.  All later calls merely
    # select days from this immutable full-year power-flow cache.
    ground_truth_report = build_full_year_ground_truth(net, NETWORK)
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
    out_dir = seasonal_run_directory(NETWORK)
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

        X_train = add_noise_by_day(X_train_clean, training_meta, TRAIN_NOISE_SEED)
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
            X_test = add_noise_by_day(X_test_clean, test_meta, TEST_NOISE_SEED)
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
        "noise_strategy": "deterministic_day_specific_seeds",
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


if __name__ == "__main__":
    main()
