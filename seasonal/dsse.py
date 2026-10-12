"""Shared DSSE building blocks used by the final thesis workflow."""

import copy
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp
import simbench as sb
from pandapower.timeseries import run_timeseries
from pandapower.timeseries.output_writer import OutputWriter
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from config import (
    ANGLE_STD_DEGREE,
    GPR_ALPHA,
    KERNEL_RUN_LABEL,
    MEASUREMENTS,
    N_TRAINING_DAYS,
    N_TRAINING_DAYS_PER_SEASON,
    SEASON_DAY_RANGES,
    SEASON_ORDER,
    OUTPUT_ROOT,
    P_RELATIVE_STD,
    Q_RELATIVE_STD,
    RESOLUTION_MINUTES,
    SENSOR_RATIO,
    SENSOR_SEED,
    USE_MEASUREMENT_NOISE,
    V_RELATIVE_STD,
)

GAUSSIAN_95_MULTIPLIER = 1.96
STATE_COLUMNS = ["p_mw", "q_mvar", "vm_pu", "va_degree"]


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))




_SIMBENCH_BUS_NUMBER_RE = re.compile(r"\bBus\s+(\d+)\s*$", re.IGNORECASE)
BUS_IDENTITY_SCHEMA_VERSION = "simbench_exact_name_v1"


def validate_simbench_bus_identity(net):
    """Return ``internal_bus -> exact SimBench name`` after strict validation.

    The pandapower integer bus index is the numerical calculation key.  The
    original SimBench ``net.bus["name"]`` value is the sole user-facing/stable
    identity.  Missing or duplicate names are rejected so a bus can never be
    silently confused with another bus.
    """
    if "name" not in net.bus.columns:
        raise RuntimeError("Loaded SimBench network has no net.bus['name'] column.")

    mapping = {}
    for bus in net.bus.index:
        bus = int(bus)
        value = net.bus.at[bus, "name"]
        if pd.isna(value) or not str(value).strip():
            raise RuntimeError(f"SimBench bus {bus} has no valid name.")
        mapping[bus] = str(value).strip()

    names = list(mapping.values())
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise RuntimeError(
            "SimBench bus names are not unique; identity would be ambiguous: "
            + ", ".join(duplicated[:10])
        )
    return mapping


def bus_identity_records(net):
    """Return the canonical bus identity table used throughout the project.

    ``internal_bus`` remains the pandapower calculation key. ``simbench_name``
    and ``bus_label`` are deliberately identical and preserve the *exact*
    SimBench name (for example ``LV4.101 Bus 6``).  Short labels such as
    ``Bus 6`` are no longer generated because they can hide the subnet/voltage
    context and caused ambiguity in the dashboard.
    """
    names = validate_simbench_bus_identity(net)
    records = []
    for bus in net.bus.index:
        bus = int(bus)
        name = names[bus]
        match = _SIMBENCH_BUS_NUMBER_RE.search(name)
        number = int(match.group(1)) if match else None
        records.append(
            {
                "internal_bus": bus,
                "simbench_bus_number": number,
                "simbench_name": name,
                "bus_label": name,
            }
        )
    return records


def bus_identity_maps(net):
    records = bus_identity_records(net)
    return {int(row["internal_bus"]): row for row in records}


def simbench_name_to_bus(net):
    """Return the inverse exact-name map ``SimBench name -> current bus index``."""
    mapping = validate_simbench_bus_identity(net)
    return {name: bus for bus, name in mapping.items()}


def saved_to_current_bus_map(net, saved_identity_records, context="saved run"):
    """Map saved internal indices to current indices using exact SimBench names.

    This is intended for presentation layers such as the dashboard.  It allows
    an old run to be displayed safely even if pandapower's internal integer
    indices were reordered, while the stable SimBench names stayed unchanged.
    """
    records = list(saved_identity_records or [])
    if not records:
        raise RuntimeError(
            f"{context} has no bus_identity metadata. Regenerate that evaluation run "
            "before combining it with the current network."
        )

    current_by_name = simbench_name_to_bus(net)
    saved_by_index = {}
    saved_names = set()
    for row in records:
        if "internal_bus" not in row:
            raise RuntimeError(f"{context} contains a bus_identity row without internal_bus.")
        old_bus = int(row["internal_bus"])
        name = row.get("simbench_name")
        if name is None or not str(name).strip():
            raise RuntimeError(
                f"{context} bus {old_bus} has no exact simbench_name metadata."
            )
        name = str(name).strip()
        if name in saved_names:
            raise RuntimeError(f"{context} contains duplicate SimBench bus name {name!r}.")
        saved_names.add(name)
        if name not in current_by_name:
            raise RuntimeError(
                f"{context} contains SimBench bus {name!r}, which is absent from "
                "the currently loaded network."
            )
        saved_by_index[old_bus] = int(current_by_name[name])

    current_names = set(current_by_name)
    if saved_names != current_names:
        missing = sorted(current_names - saved_names)
        extra = sorted(saved_names - current_names)
        raise RuntimeError(
            f"{context} bus identity set does not match the current network. "
            f"Missing={missing[:5]}, extra={extra[:5]}"
        )
    return saved_by_index


def remap_saved_bus_indices(net, saved_buses, saved_identity_records, context="saved run"):
    """Resolve saved bus IDs onto the current network by exact SimBench name."""
    mapping = saved_to_current_bus_map(net, saved_identity_records, context=context)
    result = []
    for bus in saved_buses:
        bus = int(bus)
        if bus not in mapping:
            raise RuntimeError(f"{context} references unknown saved internal bus {bus}.")
        result.append(int(mapping[bus]))
    return result


def assert_saved_bus_identity_compatible(
    net,
    saved_identity_records,
    *,
    context="saved run",
    require_internal_index_match=True,
):
    """Fail fast if saved numerical bus semantics differ from the current network.

    Numerical model features/targets are keyed by pandapower bus indices.  For
    model reuse (kernel diagnostics, stress analysis, cached ground truth), we
    therefore require the saved ``index -> exact SimBench name`` mapping to be
    identical.  Presentation code may instead use ``saved_to_current_bus_map``.
    """
    mapping = saved_to_current_bus_map(net, saved_identity_records, context=context)
    if require_internal_index_match:
        changed = [
            (old_bus, new_bus)
            for old_bus, new_bus in sorted(mapping.items())
            if int(old_bus) != int(new_bus)
        ]
        if changed:
            preview = ", ".join(f"{old}->{new}" for old, new in changed[:8])
            raise RuntimeError(
                f"{context} uses a different pandapower bus-index mapping than the "
                f"current SimBench network ({preview}). Refusing numerical reuse. "
                "Regenerate the affected evaluation/cache with the current environment."
            )
    return True


def attach_bus_identity_columns(frame, identity_records, bus_column="bus"):
    """Add SimBench identity columns without changing the internal bus index column."""
    result = frame.copy()
    if result.empty or bus_column not in result.columns:
        return result
    mapping = {int(row["internal_bus"]): row for row in identity_records}
    buses = result[bus_column].astype(int)
    result["bus_label"] = buses.map(
        lambda bus: mapping.get(int(bus), {}).get("bus_label", f"Internal bus {int(bus)}")
    )
    result["simbench_name"] = buses.map(
        lambda bus: mapping.get(int(bus), {}).get("simbench_name")
    )
    result["simbench_bus_number"] = buses.map(
        lambda bus: mapping.get(int(bus), {}).get("simbench_bus_number")
    )
    return result


def annual_run_directory(network, use_noise=USE_MEASUREMENT_NOISE):
    """Legacy annual-run location kept only for backward compatibility."""
    noise = "noisy" if use_noise else "clean"
    return (
        Path(OUTPUT_ROOT)
        / safe_name(network)
        / "annual_evaluation"
        / f"train_{N_TRAINING_DAYS:02d}_distributed_{noise}"
    )


def seasonal_run_directory(network, use_noise=USE_MEASUREMENT_NOISE):
    """Output directory for the four season-specific GPR experiment."""
    noise = "noisy" if use_noise else "clean"
    return (
        Path(OUTPUT_ROOT)
        / safe_name(network)
        / "seasonal_evaluation"
        / (
            f"4gpr_{KERNEL_RUN_LABEL}_"
            f"train_{N_TRAINING_DAYS_PER_SEASON:02d}_"
            f"all_remaining_test_{noise}"
        )
    )


def load_network(network_code):
    net = sb.get_simbench_net(network_code)
    # Validate only identity metadata; no electrical element or parameter is
    # modified here.
    validate_simbench_bus_identity(net)

    # Keep the originating SimBench code as harmless project metadata.  This is
    # only used by the compatibility simulate_day() wrapper; pandapower ignores
    # unknown top-level net keys during the power flow.
    net["_masterarbeit_simbench_code"] = str(network_code)
    return net


def get_reference_buses(net):
    if len(net.ext_grid) == 0:
        raise ValueError("Network has no ext_grid/reference bus.")
    return list(dict.fromkeys(int(bus) for bus in net.ext_grid.bus))


def select_measured_buses(net, ratio=SENSOR_RATIO, seed=SENSOR_SEED):
    """Deterministic sensor placement; reference buses are never sensors."""
    buses = [int(bus) for bus in net.bus.index]
    references = set(get_reference_buses(net))
    non_reference = [bus for bus in buses if bus not in references]

    n_measured = max(1, int(len(buses) * ratio))
    n_measured = min(n_measured, len(non_reference))

    rng = np.random.default_rng(seed)
    selected = list(rng.choice(buses, size=n_measured, replace=False))

    # Preserve the original sampling behavior, replacing a reference bus only
    # if the deterministic draw happens to select one.
    for ref in [bus for bus in selected if bus in references]:
        selected.remove(ref)
        candidates = [bus for bus in non_reference if bus not in selected]
        selected.append(int(rng.choice(candidates)))

    return sorted(int(bus) for bus in selected)


# ---------------------------------------------------------------------------
# SimBench time-series / clean full-year power-flow ground truth
# ---------------------------------------------------------------------------

GROUND_TRUTH_SCHEMA_VERSION = "simbench_timeline_v2"
EXPECTED_SIMBENCH_YEAR = 2016
EXPECTED_YEAR_STEPS = 35136
EXPECTED_YEAR_DAYS = 366
EXPECTED_SPRING_DST_DATE = "2016-03-27"
EXPECTED_AUTUMN_DST_DATE = "2016-10-30"

_FULL_YEAR_STATE_CACHE = {}


def package_version(module, package_name):
    """Return an installed package version without making it a hard dependency."""
    value = getattr(module, "__version__", None)
    if value:
        return str(value)
    try:
        from importlib.metadata import version

        return str(version(package_name))
    except Exception:
        return "unknown"


def profile_key(key):
    if isinstance(key, tuple) and len(key) == 2:
        return str(key[0]), str(key[1])
    if isinstance(key, str) and "." in key:
        element, column = key.split(".", 1)
        return str(element), str(column)
    raise ValueError(f"Unsupported SimBench profile key: {key!r}")


def available_steps(profiles):
    lengths = [
        len(frame)
        for frame in profiles.values()
        if isinstance(frame, pd.DataFrame) and not frame.empty
    ]
    if not lengths:
        raise RuntimeError("No SimBench time-series profiles found.")
    if len(set(lengths)) != 1:
        raise RuntimeError(
            f"SimBench absolute profile lengths are inconsistent: {sorted(set(lengths))}"
        )
    return int(lengths[0])


def _profile_time_frames(net):
    """Return all SimBench profile frames which expose the original `time` column."""
    profiles = net.get("profiles", None)
    if not isinstance(profiles, dict):
        raise RuntimeError(
            "The loaded SimBench network does not expose net.profiles as a dictionary."
        )

    frames = []
    for name, frame in profiles.items():
        if isinstance(frame, pd.DataFrame) and "time" in frame.columns and not frame.empty:
            frames.append((str(name), frame))
    if not frames:
        raise RuntimeError(
            "No SimBench profile table with a `time` column was found in net.profiles."
        )
    return frames


def build_simbench_timeline(net):
    """Build the authoritative timestep table from SimBench's own profile `time` column.

    The row position is the unique ``simbench_step``.  We intentionally do *not*
    use the local timestamp as the unique key because the autumn DST transition
    contains four duplicated local clock labels (02:00..02:45), each representing
    a distinct SimBench timestep.
    """
    time_frames = _profile_time_frames(net)
    reference_name, reference_frame = time_frames[0]
    labels = reference_frame["time"].astype(str).reset_index(drop=True)

    # Every profile family used by this network must describe the same physical
    # timeline.  Never align profile values by a separately generated calendar.
    for name, frame in time_frames[1:]:
        candidate = frame["time"].astype(str).reset_index(drop=True)
        if not candidate.equals(labels):
            raise RuntimeError(
                "SimBench profile time axes are inconsistent between "
                f"'{reference_name}' and '{name}'."
            )

    parsed = pd.to_datetime(
        labels,
        format="%d.%m.%Y %H:%M",
        errors="raise",
    )
    timeline = pd.DataFrame(
        {
            "simbench_step": np.arange(len(labels), dtype=int),
            # Preserve the exact SimBench local-clock label, including the four
            # duplicated labels on the autumn DST transition.
            "timestamp": labels.to_numpy(),
            "date": parsed.dt.strftime("%Y-%m-%d").to_numpy(),
            "day_of_year": parsed.dt.dayofyear.astype(int).to_numpy(),
        }
    )
    timeline["step_in_day"] = (
        timeline.groupby("date", sort=False).cumcount().astype(int)
    )
    return timeline


def validate_simbench_timeline(timeline):
    """Validate the derived 2016 SimBench timeline; do not manufacture DST counts."""
    if timeline.empty:
        raise RuntimeError("SimBench timeline is empty.")

    expected_steps = np.arange(len(timeline), dtype=int)
    actual_steps = timeline["simbench_step"].to_numpy(dtype=int)
    if not np.array_equal(actual_steps, expected_steps):
        raise RuntimeError("simbench_step must be the unique contiguous row position 0..N-1.")

    if timeline["simbench_step"].duplicated().any():
        raise RuntimeError("Duplicate simbench_step values found.")

    total_steps = int(len(timeline))
    total_days = int(timeline["date"].nunique())
    years = sorted(pd.to_datetime(timeline["date"]).dt.year.unique().tolist())
    day_counts = timeline.groupby("date", sort=False).size().astype(int)

    if years != [EXPECTED_SIMBENCH_YEAR]:
        raise RuntimeError(
            f"Expected the SimBench profile year {EXPECTED_SIMBENCH_YEAR}, got {years}."
        )
    if total_steps != EXPECTED_YEAR_STEPS:
        raise RuntimeError(
            f"Expected {EXPECTED_YEAR_STEPS} SimBench timesteps, got {total_steps}."
        )
    if total_days != EXPECTED_YEAR_DAYS:
        raise RuntimeError(
            f"Expected {EXPECTED_YEAR_DAYS} calendar days, got {total_days}."
        )

    spring_count = int(day_counts.get(EXPECTED_SPRING_DST_DATE, -1))
    autumn_count = int(day_counts.get(EXPECTED_AUTUMN_DST_DATE, -1))
    if spring_count != 92:
        raise RuntimeError(
            f"Expected 92 timesteps on {EXPECTED_SPRING_DST_DATE}, got {spring_count}."
        )
    if autumn_count != 100:
        raise RuntimeError(
            f"Expected 100 timesteps on {EXPECTED_AUTUMN_DST_DATE}, got {autumn_count}."
        )

    normal_counts = day_counts.drop(
        [EXPECTED_SPRING_DST_DATE, EXPECTED_AUTUMN_DST_DATE],
        errors="ignore",
    )
    bad_normal = normal_counts[normal_counts != 96]
    if not bad_normal.empty:
        raise RuntimeError(
            "All non-DST days must contain 96 SimBench timesteps. Bad days: "
            + ", ".join(f"{date}={count}" for date, count in bad_normal.items())
        )

    # The repeated autumn local-clock labels are intentional and must survive.
    duplicate_timestamp_rows = int(timeline["timestamp"].duplicated(keep=False).sum())
    duplicate_timestamp_labels = int(
        (timeline["timestamp"].value_counts() > 1).sum()
    )
    if duplicate_timestamp_labels != 4:
        raise RuntimeError(
            "Expected four duplicated local-clock labels on the autumn DST transition, "
            f"got {duplicate_timestamp_labels}."
        )

    return {
        "total_timesteps": total_steps,
        "total_days": total_days,
        "spring_dst_timesteps": spring_count,
        "autumn_dst_timesteps": autumn_count,
        "normal_day_timesteps": 96,
        "duplicate_local_timestamp_labels": duplicate_timestamp_labels,
        "rows_participating_in_duplicate_local_timestamps": duplicate_timestamp_rows,
    }


def available_complete_days(net):
    """Return calendar-day count from the actual SimBench profile timeline."""
    timeline = build_simbench_timeline(net)
    return int(timeline["date"].nunique())


def distributed_training_days(n_available_days, n_training_days=N_TRAINING_DAYS):
    """Deterministic, approximately equally spaced whole training days."""
    if n_training_days < 1 or n_training_days >= n_available_days:
        raise ValueError(
            "n_training_days must be at least 1 and smaller than available days."
        )

    selected = set(
        int(day)
        for day in np.rint(
            np.linspace(1, n_available_days, n_training_days)
        ).astype(int)
    )

    # Defensive fallback if rounding creates duplicates.
    while len(selected) < n_training_days:
        candidates = set(range(1, n_available_days + 1)) - selected
        new_day = max(
            candidates,
            key=lambda day: (
                min(abs(day - chosen) for chosen in selected),
                -day,
            ),
        )
        selected.add(int(new_day))

    selected = sorted(selected)
    if len(selected) > n_training_days:
        indices = np.rint(
            np.linspace(0, len(selected) - 1, n_training_days)
        ).astype(int)
        selected = [selected[index] for index in indices]
    return selected


def final_training_days(net):
    n_available_days = available_complete_days(net)
    return distributed_training_days(n_available_days), n_available_days


def season_days(season, n_available_days=366):
    """Return ordered day-of-year values belonging to one season."""
    if season not in SEASON_DAY_RANGES:
        raise KeyError(f"Unknown season: {season}")
    days = []
    for start, end in SEASON_DAY_RANGES[season]:
        days.extend(range(int(start), int(end) + 1))
    missing = [day for day in days if day > int(n_available_days)]
    if missing:
        raise ValueError(
            f"Season '{season}' requires day {max(days)}, but only "
            f"{n_available_days} profile days are available."
        )
    return days


def season_for_day(day):
    """Map a 1-based day-of-year to winter/spring/summer/autumn."""
    day = int(day)
    for season in SEASON_ORDER:
        for start, end in SEASON_DAY_RANGES[season]:
            if int(start) <= day <= int(end):
                return season
    raise ValueError(f"Day {day} is outside the configured seasonal ranges.")


def distributed_days_from_candidates(candidates, n_days):
    """Select whole days approximately evenly over an ordered candidate list."""
    candidates = [int(day) for day in candidates]
    if n_days < 1 or n_days > len(candidates):
        raise ValueError(
            f"Cannot select {n_days} days from {len(candidates)} candidates."
        )
    if n_days == len(candidates):
        return candidates.copy()

    indices = np.rint(np.linspace(0, len(candidates) - 1, n_days)).astype(int)
    selected_indices = sorted(set(int(index) for index in indices))
    # With n_days <= len(candidates), linspace normally gives unique rounded
    # indices. Keep a deterministic fallback in case a future split changes.
    if len(selected_indices) < n_days:
        available = [i for i in range(len(candidates)) if i not in selected_indices]
        while len(selected_indices) < n_days:
            index = max(
                available,
                key=lambda i: (
                    min(abs(i - chosen) for chosen in selected_indices),
                    -i,
                ),
            )
            selected_indices.append(index)
            available.remove(index)
    selected_indices = sorted(selected_indices[:n_days])
    return [candidates[index] for index in selected_indices]


def seasonal_train_test_split(
    n_available_days,
    n_training_days=N_TRAINING_DAYS_PER_SEASON,
):
    """Create deterministic seasonal splits with no unevaluated calendar days."""
    splits = {}
    for season in SEASON_ORDER:
        candidates = season_days(season, n_available_days)
        if n_training_days >= len(candidates):
            raise ValueError(
                f"Season '{season}' has {len(candidates)} days, so "
                f"{n_training_days} training days would leave no unseen test day."
            )
        training = distributed_days_from_candidates(candidates, n_training_days)
        training_set = set(training)
        test = [day for day in candidates if day not in training_set]
        splits[season] = {
            "all_days": candidates,
            "training_days": training,
            "test_days": test,
            "unused_days": [],
        }
    return splits


def day_noise_seed(base_seed, day):
    sequence = np.random.SeedSequence([int(base_seed), int(day)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def ground_truth_directory(network_code):
    return (
        Path(OUTPUT_ROOT)
        / safe_name(network_code)
        / "ground_truth"
        / GROUND_TRUTH_SCHEMA_VERSION
    )


def ground_truth_paths(network_code):
    directory = ground_truth_directory(network_code)
    return {
        "directory": directory,
        "timeline": directory / "simbench_timeline.csv",
        "voltage_ground_truth": directory / "annual_vm_pu_ground_truth.csv.gz",
        "powerflow_state": directory / "annual_powerflow_state.pkl.gz",
        "bus_identity": directory / "bus_identity.csv",
        "validation": directory / "ground_truth_validation.json",
    }


def _absolute_profile_summary(absolute_values, expected_steps):
    summary = []
    for key, frame in absolute_values.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        element, column = profile_key(key)
        if len(frame) != expected_steps:
            raise RuntimeError(
                f"Absolute SimBench profile {element}.{column} has {len(frame)} rows; "
                f"expected {expected_steps}."
            )
        expected_index = np.arange(expected_steps, dtype=int)
        try:
            actual_index = frame.index.to_numpy(dtype=int)
        except Exception as exc:
            raise RuntimeError(
                f"Absolute profile {element}.{column} does not use numeric row indices."
            ) from exc
        if not np.array_equal(actual_index, expected_index):
            raise RuntimeError(
                f"Absolute profile {element}.{column} is not indexed by the SimBench "
                "row positions 0..N-1. Refusing to remap values implicitly."
            )
        summary.append(
            {
                "element": element,
                "column": column,
                "n_timesteps": int(len(frame)),
                "n_elements": int(frame.shape[1]),
            }
        )
    if not summary:
        raise RuntimeError("SimBench get_absolute_values() returned no usable profile frames.")
    return summary


def _normalize_logged_frame(frame, steps, columns, label):
    if frame is None or not isinstance(frame, pd.DataFrame):
        raise RuntimeError(f"OutputWriter did not return a DataFrame for {label}.")
    result = frame.copy()
    if len(result) != len(steps):
        raise RuntimeError(
            f"OutputWriter {label} has {len(result)} rows; expected {len(steps)}."
        )

    # OutputWriter normally preserves time_steps as the DataFrame index.  If a
    # compatible pandapower version returns a simple positional index, row order
    # is still exactly the requested time_steps; normalize only that index label.
    if set(steps).issubset(set(result.index)):
        result = result.reindex(steps)
    else:
        result.index = pd.Index(steps, name="simbench_step")

    lookup = {str(column): column for column in result.columns}
    missing = [column for column in columns if str(column) not in lookup]
    if missing:
        raise RuntimeError(f"OutputWriter {label} is missing columns {missing}.")
    result = result[[lookup[str(column)] for column in columns]].copy()
    result.columns = [int(column) for column in columns]
    result.index = pd.Index(steps, name="simbench_step")
    return result.astype(float)


def _writer_parameters(output_writer):
    frame = output_writer.output.get("Parameters")
    return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _bool_series(values):
    return values.map(
        lambda value: bool(value)
        if isinstance(value, (bool, np.bool_))
        else str(value).strip().lower() in {"1", "true", "yes"}
    )


def _current_versions():
    return {
        "simbench_version": package_version(sb, "simbench"),
        "pandapower_version": package_version(pp, "pandapower"),
    }


def _element_counts(net):
    return {
        "bus_count": int(len(net.bus)),
        "load_count": int(len(net.load)) if "load" in net else 0,
        "sgen_count": int(len(net.sgen)) if "sgen" in net else 0,
        "gen_count": int(len(net.gen)) if "gen" in net else 0,
        "storage_count": int(len(net.storage)) if "storage" in net else 0,
    }


def _cached_ground_truth_report(net, network_code, timeline_checks):
    paths = ground_truth_paths(network_code)
    if not all(paths[name].exists() for name in ("timeline", "voltage_ground_truth", "powerflow_state", "bus_identity", "validation")):
        return None
    try:
        with open(paths["validation"], encoding="utf-8") as file:
            report = json.load(file)
    except Exception:
        return None

    expected = {
        "schema_version": GROUND_TRUTH_SCHEMA_VERSION,
        "network": str(network_code),
        **_current_versions(),
        **_element_counts(net),
        **timeline_checks,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            return None
    return report


def build_full_year_ground_truth(net, network_code, force=False, verbose=True):
    """Calculate the clean SimBench full-year AC power-flow ground truth once.

    No sensor placement, measurement masking, noise, normalization or GPR code is
    touched here.  SimBench builds the absolute full-year element profiles and
    ``apply_const_controllers`` binds them to the network.  pandapower's official
    ``run_timeseries`` then executes the balanced AC power flow for every unique
    ``simbench_step``.  The canonical ground-truth CSV contains only clean bus
    voltage magnitudes plus timeline metadata.
    """
    timeline = build_simbench_timeline(net)
    timeline_checks = validate_simbench_timeline(timeline)
    paths = ground_truth_paths(network_code)

    if not force:
        cached = _cached_ground_truth_report(net, network_code, timeline_checks)
        if cached is not None:
            print(f"Using validated full-year ground truth: {paths['voltage_ground_truth']}")
            return cached

    paths["directory"].mkdir(parents=True, exist_ok=True)
    work_net = copy.deepcopy(net)
    bus_order = [int(bus) for bus in work_net.bus.index]
    steps = timeline["simbench_step"].astype(int).tolist()

    # Important: default time_as_index=False keeps the unique numeric SimBench row
    # index for the controllers.  The local-clock labels remain separate metadata,
    # so the duplicated 30-Oct 02:xx labels are never deduplicated.
    absolute_values = sb.get_absolute_values(
        work_net,
        profiles_instead_of_study_cases=True,
    )
    absolute_profile_summary = _absolute_profile_summary(
        absolute_values,
        expected_steps=len(timeline),
    )

    # This is SimBench's official profile-to-network binding.  It creates
    # ConstControl objects for every absolute profile key SimBench supplies
    # (loads, sgens, gens, storages, ...); we do not invent storage/SOC behavior.
    sb.apply_const_controllers(work_net, absolute_values)

    output_writer = OutputWriter(
        work_net,
        time_steps=steps,
        output_path=None,
        log_variables=[],
    )
    for variable in STATE_COLUMNS:
        output_writer.log_variable("res_bus", variable, index=bus_order)

    print("=" * 88)
    print("CLEAN FULL-YEAR SIMBENCH POWER-FLOW GROUND TRUTH")
    print("=" * 88)
    print(f"Network                    : {network_code}")
    print(f"SimBench version           : {_current_versions()['simbench_version']}")
    print(f"pandapower version         : {_current_versions()['pandapower_version']}")
    print(f"Buses                      : {len(work_net.bus)}")
    print(f"Loads                      : {len(work_net.load) if 'load' in work_net else 0}")
    print(f"Sgens                      : {len(work_net.sgen) if 'sgen' in work_net else 0}")
    print(f"Gens                       : {len(work_net.gen) if 'gen' in work_net else 0}")
    print(f"Storages                   : {len(work_net.storage) if 'storage' in work_net else 0}")
    print(f"Timesteps                  : {len(timeline)}")
    print(f"Calendar days              : {timeline_checks['total_days']}")
    print("Profile application        : SimBench get_absolute_values + apply_const_controllers")
    print("Power flow                 : pandapower run_timeseries -> balanced AC runpp")
    print("continue_on_divergence     : False")
    print("Ground-truth noise/sensors : NONE")

    try:
        run_timeseries(
            work_net,
            time_steps=steps,
            continue_on_divergence=False,
            verbose=bool(verbose),
            init="auto",
        )
    except Exception as exc:
        parameters = _writer_parameters(output_writer)
        failed_steps = []
        if not parameters.empty and "powerflow_failed" in parameters.columns:
            failed_mask = _bool_series(parameters["powerflow_failed"])
            failed_steps = parameters.loc[failed_mask].index.astype(int).tolist()
        suffix = f" Failed timestep(s) recorded: {failed_steps[:10]}." if failed_steps else ""
        raise RuntimeError(
            "Full-year pandapower time-series power flow aborted; divergence/failure "
            f"is not allowed.{suffix}"
        ) from exc

    parameters = _writer_parameters(output_writer)
    failed_steps = []
    unstable_steps = []
    if not parameters.empty:
        if "powerflow_failed" in parameters.columns:
            failed_mask = _bool_series(parameters["powerflow_failed"])
            failed_steps = parameters.loc[failed_mask].index.astype(int).tolist()
        if "controller_unstable" in parameters.columns:
            unstable_mask = _bool_series(parameters["controller_unstable"])
            unstable_steps = parameters.loc[unstable_mask].index.astype(int).tolist()
    if failed_steps:
        raise RuntimeError(f"Power-flow failures were recorded at timesteps {failed_steps[:10]}.")
    if unstable_steps:
        raise RuntimeError(f"Controller instability was recorded at timesteps {unstable_steps[:10]}.")

    state_frames = {
        variable: _normalize_logged_frame(
            output_writer.output.get(f"res_bus.{variable}"),
            steps,
            bus_order,
            f"res_bus.{variable}",
        )
        for variable in STATE_COLUMNS
    }
    voltage = state_frames["vm_pu"]
    voltage_values = voltage.to_numpy(dtype=float)
    invalid_voltage_rows = ~np.isfinite(voltage_values).all(axis=1)
    n_invalid_voltage_timesteps = int(invalid_voltage_rows.sum())
    if n_invalid_voltage_timesteps:
        bad = np.flatnonzero(invalid_voltage_rows)[:10].tolist()
        raise RuntimeError(
            "Ground-truth vm_pu contains NaN/inf values at SimBench steps "
            f"{bad}."
        )
    if voltage.shape[1] != len(work_net.bus):
        raise RuntimeError(
            f"Ground-truth voltage matrix has {voltage.shape[1]} bus columns, "
            f"but network has {len(work_net.bus)} buses."
        )

    # Canonical thesis ground truth: metadata + clean voltage magnitudes only.
    voltage_ground_truth = timeline.copy()
    for bus in bus_order:
        voltage_ground_truth[f"bus_{bus}"] = voltage[bus].to_numpy(dtype=float)

    timeline.to_csv(paths["timeline"], index=False)
    voltage_ground_truth.to_csv(
        paths["voltage_ground_truth"],
        index=False,
        compression="gzip",
    )
    pd.DataFrame(bus_identity_records(work_net)).to_csv(paths["bus_identity"], index=False)

    # Clean PF states are stored separately only so the existing downstream
    # measurement simulation can later copy P/Q/V/theta without rerunning PF.
    # They are not sensor-masked, noised or normalized ground truth.
    state_payload = {
        "schema_version": GROUND_TRUTH_SCHEMA_VERSION,
        "network": str(network_code),
        "timeline": timeline,
        "bus_order": bus_order,
        "states": state_frames,
    }
    pd.to_pickle(state_payload, paths["powerflow_state"], compression="gzip")

    day_counts = timeline.groupby("date", sort=False).size().astype(int)
    versions = _current_versions()
    counts = _element_counts(work_net)
    report = {
        "schema_version": GROUND_TRUTH_SCHEMA_VERSION,
        "network": str(network_code),
        **counts,
        **timeline_checks,
        "example_normal_date": "2016-01-01",
        "example_normal_day_timesteps": int(day_counts.loc["2016-01-01"]),
        "failed_powerflow_timesteps": 0,
        "controller_unstable_timesteps": 0,
        "nan_or_inf_voltage_timesteps": n_invalid_voltage_timesteps,
        "ground_truth_matrix_shape": [int(voltage.shape[0]), int(voltage.shape[1])],
        "ground_truth_dataset_shape": [
            int(voltage_ground_truth.shape[0]),
            int(voltage_ground_truth.shape[1]),
        ],
        "voltage_column_count": int(voltage.shape[1]),
        **versions,
        "absolute_profile_keys": [
            f"{row['element']}.{row['column']}" for row in absolute_profile_summary
        ],
        "absolute_profile_summary": absolute_profile_summary,
        "profile_binding": "simbench.apply_const_controllers",
        "powerflow_engine": "pandapower.timeseries.run_timeseries / runpp",
        "continue_on_divergence": False,
        "balanced_ac_power_flow": True,
        "ground_truth_contains_noise": False,
        "ground_truth_contains_sensor_masking": False,
        "ground_truth_contains_gpr": False,
        "voltage_ground_truth_file": paths["voltage_ground_truth"].name,
        "timeline_file": paths["timeline"].name,
        "powerflow_state_file": paths["powerflow_state"].name,
    }
    with open(paths["validation"], "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    _FULL_YEAR_STATE_CACHE.pop(str(paths["powerflow_state"].resolve()), None)

    print("\nGROUND-TRUTH VALIDATION")
    print(f"Total timesteps             : {report['total_timesteps']}")
    print(f"Total days                  : {report['total_days']}")
    print(f"27.03.2016 timesteps        : {report['spring_dst_timesteps']}")
    print(f"30.10.2016 timesteps        : {report['autumn_dst_timesteps']}")
    print(f"01.01.2016 timesteps        : {report['example_normal_day_timesteps']}")
    print(f"Failed power flows          : {report['failed_powerflow_timesteps']}")
    print(f"NaN/inf voltage timesteps   : {report['nan_or_inf_voltage_timesteps']}")
    print(f"Voltage matrix shape        : {tuple(report['ground_truth_matrix_shape'])}")
    print(f"Saved ground truth          : {paths['voltage_ground_truth']}")
    return report


def _assert_ground_truth_bus_identity_compatible(net, network_code):
    """Verify that cached PF columns still refer to the same exact SimBench buses."""
    identity_path = ground_truth_paths(network_code)["bus_identity"]
    if not identity_path.exists():
        raise RuntimeError(
            "Ground-truth bus_identity.csv is missing. Refusing to use the cached "
            "power-flow state without a verifiable bus mapping."
        )
    saved_frame = pd.read_csv(identity_path)
    saved_identity = saved_frame.to_dict(orient="records")
    assert_saved_bus_identity_compatible(
        net,
        saved_identity,
        context="cached ground truth",
        require_internal_index_match=True,
    )

    # Identity metadata can be modernized without rerunning a single power flow:
    # after compatibility is proven, rewrite only bus_identity.csv so bus_label
    # equals the exact SimBench name in existing caches as well.
    canonical = pd.DataFrame(bus_identity_records(net))
    comparable = ["internal_bus", "simbench_bus_number", "simbench_name", "bus_label"]
    if not saved_frame.reindex(columns=comparable).equals(canonical[comparable]):
        canonical.to_csv(identity_path, index=False)


def load_full_year_powerflow_state(net, network_code):
    """Load the immutable clean PF cache; build it first if it does not exist."""
    build_full_year_ground_truth(net, network_code, force=False, verbose=True)

    # The clean state matrices are keyed by pandapower integer bus indices.
    # Before using an existing cache, prove that every index still resolves to
    # the same exact SimBench bus name in the current network.
    _assert_ground_truth_bus_identity_compatible(net, network_code)

    path = ground_truth_paths(network_code)["powerflow_state"]
    key = str(path.resolve())
    if key not in _FULL_YEAR_STATE_CACHE:
        payload = pd.read_pickle(path, compression="gzip")
        if payload.get("schema_version") != GROUND_TRUTH_SCHEMA_VERSION:
            raise RuntimeError("Cached power-flow state has the wrong schema version.")
        _FULL_YEAR_STATE_CACHE[key] = payload
    return _FULL_YEAR_STATE_CACHE[key]


def _long_ground_truth_from_payload(payload, days):
    timeline = payload["timeline"].copy()
    days = sorted(set(int(day) for day in days))
    available = set(timeline["day_of_year"].astype(int).unique())
    invalid = [day for day in days if day not in available]
    if invalid:
        raise ValueError(f"Requested day(s) are unavailable in the SimBench timeline: {invalid}")

    selected_timeline = timeline[timeline["day_of_year"].astype(int).isin(days)].copy()
    selected_timeline = selected_timeline.sort_values("simbench_step")
    steps = selected_timeline["simbench_step"].astype(int).tolist()
    buses = [int(bus) for bus in payload["bus_order"]]
    n_buses = len(buses)

    repeated_meta = selected_timeline.loc[
        selected_timeline.index.repeat(n_buses)
    ].reset_index(drop=True)
    result = repeated_meta.copy()
    result["time_step"] = result["simbench_step"].astype(int)
    result["profile_index"] = result["simbench_step"].astype(str)
    result["day"] = result["day_of_year"].astype(int)
    result["time"] = result["timestamp"].astype(str).str[-5:]
    result["bus"] = np.tile(np.asarray(buses, dtype=int), len(steps))

    for variable in STATE_COLUMNS:
        frame = payload["states"][variable]
        values = frame.loc[steps, buses].to_numpy(dtype=float).reshape(-1)
        result[variable] = values

    return result


def load_ground_truth(net, network_code, days):
    """Select clean day(s) from the already computed full-year PF cache.

    This function never runs a power flow.  Day selection is based on the real
    SimBench timeline and therefore returns 92 rows-per-bus on 27 March, 100 on
    30 October, and 96 on ordinary days.
    """
    days = sorted(set(int(day) for day in days))
    if not days:
        raise ValueError("At least one day is required.")
    payload = load_full_year_powerflow_state(net, network_code)
    ground_truth = _long_ground_truth_from_payload(payload, days)
    ground_truth = attach_bus_identity_columns(
        ground_truth,
        bus_identity_records(net),
        bus_column="bus",
    )
    return ground_truth.sort_values(["simbench_step", "bus"]).reset_index(drop=True)


def get_day_ground_truth(net, network_code, date_or_day):
    """Convenience selector using either YYYY-MM-DD or a day-of-year integer."""
    timeline = build_simbench_timeline(net)
    if isinstance(date_or_day, str) and "-" in date_or_day:
        rows = timeline[timeline["date"] == str(date_or_day)]
        if rows.empty:
            raise ValueError(f"Date {date_or_day!r} is not present in the SimBench timeline.")
        day = int(rows["day_of_year"].iloc[0])
    else:
        day = int(date_or_day)
    return load_ground_truth(net, network_code, [day])


def simulate_day(net, profiles, profile_reference, day):
    """Backward-compatible wrapper; no longer performs any power-flow calculation."""
    network_code = net.get("_masterarbeit_simbench_code")
    if not network_code:
        raise RuntimeError(
            "simulate_day() compatibility wrapper needs a network loaded by load_network()."
        )
    return load_ground_truth(net, network_code, [int(day)])


# ---------------------------------------------------------------------------
# Dataset / noise
# ---------------------------------------------------------------------------


def build_voltage_dataset(
    ground_truth,
    measured_buses,
    all_buses,
    reference_buses,
    measurements=MEASUREMENTS,
):
    """Build measured-bus inputs and voltage targets for non-reference buses."""
    measured_buses = [int(bus) for bus in measured_buses]
    all_buses = [int(bus) for bus in all_buses]
    references = set(int(bus) for bus in reference_buses)

    pivots = {
        state: ground_truth.pivot(index="time_step", columns="bus", values=state)
        .reindex(columns=all_buses)
        for state in STATE_COLUMNS
    }

    X = pd.DataFrame(
        {
            f"bus_{bus}_{state}": pivots[state][bus]
            for bus in measured_buses
            for state in measurements
        }
    )
    if X.isna().any().any():
        raise RuntimeError("NaN values found in the measurement matrix.")

    target_buses = [bus for bus in all_buses if bus not in references]
    measured_set = set(measured_buses)
    evaluation_buses = [bus for bus in target_buses if bus not in measured_set]

    y_voltage = pd.DataFrame(
        {f"bus_{bus}_vm_pu": pivots["vm_pu"][bus] for bus in target_buses}
    )
    metadata_columns = [
        column
        for column in [
            "time_step",
            "simbench_step",
            "timestamp",
            "date",
            "day_of_year",
            "step_in_day",
            "profile_index",
            "day",
            "time",
        ]
        if column in ground_truth.columns
    ]
    metadata = (
        ground_truth[metadata_columns]
        .drop_duplicates("time_step")
        .set_index("time_step")
        .reindex(X.index)
    )
    return X, y_voltage, metadata, target_buses, evaluation_buses


def add_measurement_noise(X, seed):
    noisy = X.copy()
    rng = np.random.default_rng(seed)

    for column in noisy.columns:
        values = noisy[column].to_numpy(float)
        if column.endswith("_p_mw"):
            std = (
                np.maximum(np.abs(values) * P_RELATIVE_STD, 1e-6)
                if P_RELATIVE_STD > 0
                else np.zeros_like(values)
            )
        elif column.endswith("_q_mvar"):
            std = (
                np.maximum(np.abs(values) * Q_RELATIVE_STD, 1e-6)
                if Q_RELATIVE_STD > 0
                else np.zeros_like(values)
            )
        elif column.endswith("_vm_pu"):
            std = np.abs(values) * V_RELATIVE_STD
        elif column.endswith("_va_degree"):
            std = np.full(values.shape, ANGLE_STD_DEGREE, dtype=float)
        else:
            continue
        noisy[column] = values + rng.normal(0.0, std)
    return noisy


def add_noise_by_day(X_clean, metadata, base_seed):
    """Apply measurement noise with deterministic, day-specific seeds."""
    if not USE_MEASUREMENT_NOISE:
        return X_clean.copy()

    noisy = X_clean.copy()
    meta = metadata.loc[X_clean.index]
    for day in sorted(meta["day"].astype(int).unique()):
        mask = meta["day"].astype(int).eq(int(day))
        X_day = X_clean.loc[mask].copy()
        X_day_noisy = add_measurement_noise(X_day, day_noise_seed(base_seed, day))
        noisy.loc[X_day_noisy.index, X_day_noisy.columns] = X_day_noisy
    return noisy


# ---------------------------------------------------------------------------
# GPR and metrics
# ---------------------------------------------------------------------------


def build_kernel():
    """Smooth signal kernel plus learned iid residual/output noise.

    ConstantKernel * RBF models the smooth mapping from the measured-bus
    operating point to the network voltage magnitudes. WhiteKernel learns an
    additional iid residual/output-noise level. Numerical matrix stability is
    handled separately by the very small ``GPR_ALPHA`` value.
    """
    return (
        ConstantKernel(
            constant_value=1.0,
            constant_value_bounds=(1e-3, 1e3),
        )
        * RBF(
            length_scale=1.0,
            length_scale_bounds=(1e-2, 1e3),
        )
        + WhiteKernel(
            noise_level=1e-3,
            noise_level_bounds=(1e-8, 1e0),
        )
    )


class DSSEGaussianProcess:
    """One fitted multi-output GPR used by one season-specific DSSE model."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.target_columns = None

    @staticmethod
    def _new_regressor():
        return GaussianProcessRegressor(
            kernel=build_kernel(),
            alpha=GPR_ALPHA,
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=42,
        )

    @staticmethod
    def _as_2d(array, n_outputs):
        values = np.asarray(array, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.shape[1] != n_outputs:
            raise RuntimeError(
                f"Unexpected GPR output shape {values.shape}; "
                f"expected {n_outputs} output columns."
            )
        return values

    def fit(self, X, Y):
        if not isinstance(Y, pd.DataFrame):
            Y = pd.DataFrame(Y, index=X.index)
        if len(X) != len(Y):
            raise ValueError("X and Y must contain the same number of samples.")

        self.target_columns = list(Y.columns)
        print("\nTraining one season-specific multi-output GPR on all training samples")
        X_scaled = self.scaler.fit_transform(X)
        self.model = self._new_regressor()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(X_scaled, Y.to_numpy(float))

        print(
            f"  fitted one GPR object: {X.shape[1]} inputs → "
            f"{Y.shape[1]} voltage outputs"
        )
        return self

    def predict(self, X):
        if self.model is None or self.target_columns is None:
            raise RuntimeError("Call fit() before predict().")

        mean, std = self.model.predict(
            self.scaler.transform(X), return_std=True
        )
        mean = self._as_2d(mean, len(self.target_columns))
        std = self._as_2d(std, len(self.target_columns))
        return (
            pd.DataFrame(mean, index=X.index, columns=self.target_columns),
            pd.DataFrame(std, index=X.index, columns=self.target_columns),
        )

    def kernel_summary(self):
        if self.model is None:
            return pd.DataFrame()

        fitted_kernel = self.model.kernel_
        # Expected fixed structure:
        # (ConstantKernel * RBF) + WhiteKernel
        signal_kernel = fitted_kernel.k1
        constant_kernel = signal_kernel.k1
        rbf_kernel = signal_kernel.k2
        white_kernel = fitted_kernel.k2

        return pd.DataFrame(
            [
                {
                    "model_type": "single_multioutput_gpr",
                    "n_outputs": len(self.target_columns),
                    "optimized_kernel": str(fitted_kernel),
                    "constant_value": float(constant_kernel.constant_value),
                    "rbf_length_scale": float(rbf_kernel.length_scale),
                    "white_noise_level": float(white_kernel.noise_level),
                    "alpha": float(self.model.alpha),
                    "log_marginal_likelihood": float(
                        self.model.log_marginal_likelihood(self.model.kernel_.theta)
                    ),
                }
            ]
        )


def evaluate(y_true, y_pred, sigma):
    true = y_true.to_numpy(float)
    pred = y_pred.to_numpy(float)
    std = sigma.to_numpy(float)
    error = pred - true
    lower = pred - GAUSSIAN_95_MULTIPLIER * std
    upper = pred + GAUSSIAN_95_MULTIPLIER * std
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "coverage_95": float(np.mean((true >= lower) & (true <= upper))),
        "mean_interval_width_95": float(np.mean(upper - lower)),
    }


# Old saved joblib models were pickled as gpr.DSSEGaussianProcess. Registering
# this alias lets the cleaned project reuse those existing model artifacts.
sys.modules.setdefault("gpr", sys.modules[__name__])
