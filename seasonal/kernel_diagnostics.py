"""Kernel diagnostics for the saved season-specific GPR models."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import spearmanr

from config import NETWORK, TEST_NOISE_SEED, TRAIN_NOISE_SEED
from dsse import (
    add_noise_by_day,
    build_voltage_dataset,
    load_ground_truth,
    load_network,
    season_for_day,
)


def _load_config(run_dir):
    run_dir = Path(run_dir)
    with open(run_dir / "run_config.json", encoding="utf-8") as file:
        return json.load(file)


def _pairwise_euclidean(values):
    values = np.asarray(values, dtype=float)
    diff = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _even_indices(n, max_points):
    if n <= max_points:
        return np.arange(n, dtype=int)
    return np.unique(np.rint(np.linspace(0, n - 1, max_points)).astype(int))


def compute_kernel_diagnostics(
    run_dir, selected_day, max_train_overview=320, evaluation_scope="unmeasured_only"
):
    run_dir = Path(run_dir)
    config = _load_config(run_dir)
    day = int(selected_day)
    season = season_for_day(day)

    configured_test_days = {
        int(d)
        for d in config.get("test_days_by_season", {}).get(season, [])
    }
    if configured_test_days and day not in configured_test_days:
        raise ValueError(
            f"Day {day} is not a saved {season} test day for this experiment."
        )

    model_file = config.get("trained_model_files", {}).get(season)
    if not model_file:
        raise FileNotFoundError(f"No model file configured for season '{season}'.")
    model_path = run_dir / model_file
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model = joblib.load(model_path)

    net = load_network(config.get("network", NETWORK))
    all_buses = [int(bus) for bus in net.bus.index]
    measured_buses = [int(bus) for bus in config["measured_buses"]]
    reference_buses = [int(bus) for bus in config["reference_buses"]]
    unmeasured_buses = [int(bus) for bus in config["evaluation_buses"]]
    all_evaluation_buses = [
        int(bus) for bus in config.get("all_evaluation_buses", config["prediction_buses"])
    ]
    if evaluation_scope == "unmeasured_only":
        evaluation_buses = unmeasured_buses
    elif evaluation_scope == "all_predicted_buses":
        evaluation_buses = all_evaluation_buses
    else:
        raise KeyError(f"Unknown evaluation scope: {evaluation_scope}")
    training_days = [
        int(d) for d in config["training_days_by_season"][season]
    ]

    training_truth = load_ground_truth(
        net, config.get("network", NETWORK), training_days
    )
    X_train_clean, y_train, train_meta, target_buses, eval_buses_check = (
        build_voltage_dataset(
            training_truth,
            measured_buses,
            all_buses,
            reference_buses,
        )
    )
    if eval_buses_check != unmeasured_buses:
        raise RuntimeError("Unmeasured-bus ordering does not match the saved run.")
    X_train = (
        add_noise_by_day(X_train_clean, train_meta, TRAIN_NOISE_SEED)
        if config.get("use_measurement_noise", True)
        else X_train_clean
    )

    test_truth = load_ground_truth(net, config.get("network", NETWORK), [day])
    X_test_clean, y_test, test_meta, test_targets, test_eval = build_voltage_dataset(
        test_truth,
        measured_buses,
        all_buses,
        reference_buses,
    )
    if test_targets != target_buses or test_eval != unmeasured_buses:
        raise RuntimeError("Test-day bus ordering does not match the seasonal model.")
    X_test = (
        add_noise_by_day(X_test_clean, test_meta, TEST_NOISE_SEED)
        if config.get("use_measurement_noise", True)
        else X_test_clean
    )

    X_train_scaled = model.scaler.transform(X_train)
    X_test_scaled = model.scaler.transform(X_test)

    fitted_kernel = model.model.kernel_
    # Fixed fitted structure: (ConstantKernel * RBF) + WhiteKernel.
    # The heatmaps are intended to visualize operating-point similarity, so
    # they use only the smooth signal component. WhiteKernel is reported
    # separately because it represents iid residual/output noise, not
    # cross-operating-point similarity.
    signal_kernel = fitted_kernel.k1
    white_kernel = fitted_kernel.k2

    overview_idx = _even_indices(len(X_train_scaled), max_train_overview)
    X_train_overview = X_train_scaled[overview_idx]
    K_train_train = signal_kernel(X_train_overview)
    K_test_train = signal_kernel(X_test_scaled, X_train_scaled)
    K_test_test = signal_kernel(X_test_scaled)

    eval_columns = [f"bus_{bus}_vm_pu" for bus in evaluation_buses]
    true_state = y_test[eval_columns].to_numpy(float)
    true_state_difference = _pairwise_euclidean(true_state)

    upper = np.triu_indices_from(K_test_test, k=1)
    cov_values = K_test_test[upper]
    diff_values = true_state_difference[upper]
    if len(cov_values) > 1 and np.std(cov_values) > 0 and np.std(diff_values) > 0:
        structure_corr = float(spearmanr(cov_values, diff_values).statistic)
    else:
        structure_corr = float("nan")

    symmetry_error = float(np.max(np.abs(K_test_test - K_test_test.T)))
    diag = np.diag(K_test_test)

    train_day_lookup = train_meta.loc[X_train.index, "day"].astype(int).to_numpy()
    overview_days = train_day_lookup[overview_idx]
    test_times = test_meta.loc[X_test.index, "time"].astype(str).tolist()

    return {
        "season": season,
        "selected_day": day,
        "evaluation_scope": evaluation_scope,
        "n_evaluation_buses": len(evaluation_buses),
        "model_file": model_file,
        "kernel_string": str(fitted_kernel),
        "signal_kernel_string": str(signal_kernel),
        "constant_value": float(signal_kernel.k1.constant_value),
        "rbf_length_scale": float(signal_kernel.k2.length_scale),
        "white_noise_level": float(white_kernel.noise_level),
        "n_training_samples": int(len(X_train)),
        "n_train_overview_samples": int(len(overview_idx)),
        "training_days": training_days,
        "train_overview_days": overview_days,
        "test_times": test_times,
        "train_train_covariance": K_train_train,
        "test_train_covariance": K_test_train,
        "test_test_covariance": K_test_test,
        "true_voltage_state_difference": true_state_difference,
        "test_test_structure_spearman": structure_corr,
        "test_test_symmetry_error": symmetry_error,
        "test_test_diagonal_min": float(np.min(diag)),
        "test_test_diagonal_max": float(np.max(diag)),
    }


def train_train_heatmap_figure(diag):
    matrix = np.asarray(diag["train_train_covariance"], dtype=float)
    days = [int(x) for x in np.asarray(diag["train_overview_days"]).tolist()]
    labels = [f"D{day}" for day in days]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Viridis",
            colorbar={"title": "Covariance"},
            hovertemplate="Train %{y} ↔ %{x}<br>K=%{z:.6g}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{diag['season'].title()} GPR — Train–Train Covariance",
        xaxis_title="Downsampled training operating points",
        yaxis_title="Downsampled training operating points",
        height=620,
    )
    return fig


def test_train_heatmap_figure(diag):
    matrix = np.asarray(diag["test_train_covariance"], dtype=float)
    train_days = diag["training_days"]
    samples_per_day = matrix.shape[1] // max(len(train_days), 1)
    tickvals = []
    ticktext = []
    for i, day in enumerate(train_days):
        tickvals.append(i * samples_per_day + max(samples_per_day - 1, 0) / 2)
        ticktext.append(f"Day {day}")
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=np.arange(matrix.shape[1]),
            y=diag["test_times"],
            colorscale="Viridis",
            colorbar={"title": "Covariance"},
            hovertemplate="Test %{y}<br>Training sample %{x}<br>K=%{z:.6g}<extra></extra>",
        )
    )
    fig.update_layout(
        title=(
            f"{diag['season'].title()} GPR — Day {diag['selected_day']} "
            "Test–Train Covariance"
        ),
        xaxis_title="Training samples grouped by training day",
        yaxis_title="Selected test day",
        height=560,
    )
    if tickvals:
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
    return fig


def test_test_heatmap_figure(diag):
    matrix = np.asarray(diag["test_test_covariance"], dtype=float)
    times = diag["test_times"]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=times,
            y=times,
            colorscale="Viridis",
            colorbar={"title": "Covariance"},
            hovertemplate="%{y} ↔ %{x}<br>K=%{z:.6g}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Test–Test Kernel Covariance",
        xaxis_title="Time",
        yaxis_title="Time",
        height=560,
    )
    fig.update_xaxes(tickmode="array", tickvals=times[::8], ticktext=times[::8])
    fig.update_yaxes(tickmode="array", tickvals=times[::8], ticktext=times[::8])
    return fig


def true_voltage_state_difference_heatmap_figure(diag):
    matrix = np.asarray(diag["true_voltage_state_difference"], dtype=float)
    times = diag["test_times"]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=times,
            y=times,
            colorscale="Viridis",
            reversescale=True,
            colorbar={"title": "State distance"},
            hovertemplate="%{y} ↔ %{x}<br>Distance=%{z:.6g}<extra></extra>",
        )
    )
    fig.update_layout(
        title=(
            "True Voltage-State Difference — "
            + ("Unmeasured Buses" if diag.get("evaluation_scope") == "unmeasured_only" else "All Predicted Buses")
        ),
        xaxis_title="Time",
        yaxis_title="Time",
        height=560,
    )
    fig.update_xaxes(tickmode="array", tickvals=times[::8], ticktext=times[::8])
    fig.update_yaxes(tickmode="array", tickvals=times[::8], ticktext=times[::8])
    return fig
