from pathlib import Path

# Network and measurements
NETWORK = "1-LV-semiurb4--2-sw"
SENSOR_RATIO = 0.30
SENSOR_SEED = 42
MEASUREMENTS = ["p_mw", "q_mvar", "vm_pu"]
RESOLUTION_MINUTES = 15

# Seasonal thesis experiment
# SimBench uses its original 2016 full-year local-clock timeline.
# Days are selected by day-of-year from that timeline; no fixed 96-step/day
# slicing is used (27 Mar has 92 steps, 30 Oct has 100, normal days have 96).
# Each season uses exactly 30 distributed training days; every remaining day
# of that same season is evaluated as an unseen test day.
N_TRAINING_DAYS_PER_SEASON = 30
TEST_BATCH_DAYS = 7

# Day-of-year definition used throughout the project.
# Winter is intentionally split into Jan-Feb and December but belongs to one GPR.
SEASON_DAY_RANGES = {
    "winter": [(1, 60), (336, 366)],
    "spring": [(61, 152)],
    "summer": [(153, 244)],
    "autumn": [(245, 335)],
}
SEASON_ORDER = ["winter", "spring", "summer", "autumn"]
SEASON_LABELS = {
    "winter": "Winter",
    "spring": "Spring",
    "summer": "Summer",
    "autumn": "Autumn",
}

# Backward-compatible aliases for older helper code.
N_TRAINING_DAYS = N_TRAINING_DAYS_PER_SEASON


# Kernel configuration for the seasonal experiment.
# The smooth signal is modeled by ConstantKernel * RBF.
# WhiteKernel learns an additional iid residual/output-noise level.
# GPR_ALPHA is only a very small numerical jitter for matrix stability.
KERNEL_RUN_LABEL = "constant_rbf_white_alpha1e-10"
KERNEL_NAME = "ConstantKernel * RBF + WhiteKernel"
GPR_ALPHA = 1e-10

# Measurement noise
USE_MEASUREMENT_NOISE = True
P_RELATIVE_STD = 0.01 / 3
Q_RELATIVE_STD = 0.01 / 3
V_RELATIVE_STD = 0.005 / 3
ANGLE_STD_DEGREE = 0.05
TRAIN_NOISE_SEED = 7001
TEST_NOISE_SEED = 8001

# Outputs
OUTPUT_ROOT = Path("outputs")
