"""Reproduce Table 4 gap-duration robustness results."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiment.gap_horizon.case_data import load_cases
from experiment.gap_horizon.horizon_analysis import recovery_horizon, recovery_rate_by_gap
from experiment.gap_horizon.precision_horizon import build_full_breakdown, build_h_precision_table
from experiment.gap_horizon.recovery_failure_cause import build_audit
from experiment.gap_horizon.study_config import (
    DENSE_SCENARIO,
    DENSE_SCENARIO_SENTINEL_GAPS,
    G1_G2_G3_SCENARIOS,
    PRIMARY_GAP_RANGE,
    SEEDS,
)

EXPECTED_DIR = REPO_ROOT / "expected" / "table4"
ALL_SCENARIOS_GAPS = {s: PRIMARY_GAP_RANGE for s in G1_G2_G3_SCENARIOS}
ALL_SCENARIOS_GAPS[DENSE_SCENARIO] = DENSE_SCENARIO_SENTINEL_GAPS


def _sha256(obj) -> str:
    
    return hashlib.sha256(json.dumps(obj, indent=2, default=str).encode()).hexdigest()


def h_recovery_95(scenario: str, gaps: list[int]) -> int | str:

    cases = load_cases()
    outcomes = {gap: {seed: c.recovery_outcome for seed in SEEDS if (c := cases.get((scenario, seed, float(gap)))) is not None} for gap in gaps}
    rate_by_gap = recovery_rate_by_gap(outcomes)
    h = recovery_horizon(rate_by_gap, 0.95)
    return h if h is not None else "NOT_ACHIEVED"


def main() -> None:

    rows = build_full_breakdown()
    h_table = build_h_precision_table(rows)
    recovery_audit = build_audit()

    h_table_sha = _sha256({"full_breakdown_per_scenario_gap_threshold": rows, "H_precision_canonical_table": h_table})
    recovery_sha = _sha256(recovery_audit)
    expected_h_sha = hashlib.sha256((EXPECTED_DIR / "h_precision.json").read_text(encoding="utf-8").encode()).hexdigest()
    expected_recovery_sha = hashlib.sha256((EXPECTED_DIR / "recovery_failure_cause.json").read_text(encoding="utf-8").encode()).hexdigest()
    h_table_match = h_table_sha == expected_h_sha
    recovery_match = recovery_sha == expected_recovery_sha

    print("Table 4")
    print("-" * 7)
    print(f"{'scenario':<32} {'H_recovery,95':>14} {'H_5m,95':>8} {'H_10m,95':>9} {'<=10m@10s':>10}")
    for scenario, gaps in ALL_SCENARIOS_GAPS.items():
        h_rec = h_recovery_95(scenario, gaps)
        h5 = h_table[scenario]["H_5m"]["H_95"]
        h10 = h_table[scenario]["H_10m"]["H_95"]
        row10 = next(r for r in rows if r["scenario"] == scenario and r["gap"] == 10 and r["threshold_m"] == 10.0)
        rate10 = row10["success_rate_over_evaluable"]
        rate10_str = f"{rate10:.0%}" if rate10 is not None else "n/a"
        print(f"{scenario:<32} {str(h_rec):>14} {str(h5):>8} {str(h10):>9} {rate10_str:>10}")

    print()
    print(f"H-precision table  vs expected/table4/h_precision.json:         {'MATCH' if h_table_match else 'DIFFERS'}")
    print(f"Recovery-failure-cause vs expected/table4/recovery_failure_cause.json: {'MATCH' if recovery_match else 'DIFFERS'}")

    all_ok = h_table_match and recovery_match
    print(f"\nOVERALL: {'MATCH' if all_ok else 'DIFFERS'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
