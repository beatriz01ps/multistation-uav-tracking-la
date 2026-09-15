"""Frozen constants for the GAP_HORIZON study, registered before any execution."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIOS_DIR = REPO_ROOT / "src" / "network_simulator" / "scenarios"

# 4 scenarios of the GAP_HORIZON family - see each YAML for its full
#geometric rationale.
GAP_HORIZON_SCENARIOS = [
    "gap_horizon_1uav_straight",
    "gap_horizon_1uav_maneuver",
    "gap_horizon_2uav_crossing",
    "gap_horizon_5uav_dense_crossing",
]

# uav_1 is always the target under test in the 4 scenarios (the only one
# subjected to an artificial blackout). see each YAML's docstring.
TARGET_UNDER_TEST_NAME = "uav_1"

# the same seeds used across FULL_STACK validation/literature baselines -
# never overlapping FINAL_SEEDS (2001-2020) or DEV_RESERVED_SEEDS (1001-1005).
SEEDS = list(range(5001, 5021))
MA_EXPLORATORY_SEEDS = list(range(5001, 5011))  # 10 pre-registered seeds, exploratory only

# The fixed reacquisition instant -> blackout(d) = [80-d, 80).
REACQUISITION_ANCHOR_S = 80.0

# Ppre-registered precision thresholds.
PRECISION_THRESHOLDS_M = [1.0, 2.0, 5.0, 10.0]

# Scenarios used in the exploratory comparison against MA 1 maneuvering
# UAV, 2 UAV crossing, 5 UAV dense covers 1/2/several targets.
MA_COMPARISON_SCENARIOS = [
    "gap_horizon_1uav_maneuver",
    "gap_horizon_2uav_crossing",
    "gap_horizon_5uav_dense_crossing",
]

# The dense-crossing scenario reports 8 sentinel gaps instead of the
# other 3 scenarios full 1..32 range (Table 4's reported case count).
DENSE_SCENARIO = "gap_horizon_5uav_dense_crossing"
G1_G2_G3_SCENARIOS = [s for s in GAP_HORIZON_SCENARIOS if s != DENSE_SCENARIO]
PRIMARY_GAP_RANGE = list(range(1, 33))  # G1-G3 run 1..32
DENSE_SCENARIO_SENTINEL_GAPS = [1, 2, 3, 4, 5, 10, 20, 32]
REDUCED_MA_GAP_RANGE = list(range(1, 25))  # reduced MA range -> 1..24 (crosses tau_g=20)


def scenario_path(scenario_name: str) -> Path:
    return SCENARIOS_DIR / f"{scenario_name}.yaml"


def gap_window(duration_s: float) -> tuple[float, float]:
    """blackout(d) = [80-d, 80)."""
    return REACQUISITION_ANCHOR_S - duration_s, REACQUISITION_ANCHOR_S
