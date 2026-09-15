"""H-precision table (Table 4): the recovery horizon where a fixed fraction of
cases have position error under a threshold. Uses the exact reacquisition-instant error when available, else the last persisted 1s bucket; a case with no pre-gap track is excluded."""

from __future__ import annotations

from experiment.gap_horizon.case_data import load_cases
from experiment.gap_horizon.horizon_analysis import recovery_horizon
from experiment.gap_horizon.study_config import (
    DENSE_SCENARIO,
    DENSE_SCENARIO_SENTINEL_GAPS,
    G1_G2_G3_SCENARIOS,
    PRECISION_THRESHOLDS_M,
    PRIMARY_GAP_RANGE,
    SEEDS,
)

ALL_SCENARIOS_GAPS = {s: PRIMARY_GAP_RANGE for s in G1_G2_G3_SCENARIOS}
ALL_SCENARIOS_GAPS[DENSE_SCENARIO] = DENSE_SCENARIO_SENTINEL_GAPS


def _sample_and_method(scenario: str, seed: int, gap: int) -> tuple[dict | None, str | None]:

    """Returns (sample, method). method=None if no track was alive before the gap;
    sample=None with method=POINTWISE_1S_BUCKET_APPROXIMATION means no data at bucketed resolution."""

    case = load_cases().get((scenario, seed, float(gap)))
    if case is None or not case.pre_gap_track:
        return None, None

    if case.exact_position_error_m is not None:
        return {"position_error_m": case.exact_position_error_m}, "EXACT_NATIVE_RESOLUTION"

    if case.last_bucket_position_error_m is not None:
        return {"position_error_m": case.last_bucket_position_error_m}, "POINTWISE_1S_BUCKET_APPROXIMATION"
    return None, "POINTWISE_1S_BUCKET_APPROXIMATION"


def _breakdown_row(scenario: str, gap: int, threshold: float) -> dict:

    n_expected = len(SEEDS)
    n_no_pre_gap_track = 0
    n_insufficient_resolution = 0
    n_success = 0
    n_evaluable = 0
    methods_used = set()

    for seed in SEEDS:
        sample, method = _sample_and_method(scenario, seed, gap)
        if method is None:
            n_no_pre_gap_track += 1
            continue
        methods_used.add(method)
        if sample is None:
            n_insufficient_resolution += 1
            continue
        n_evaluable += 1
        if sample["position_error_m"] <= threshold:
            n_success += 1

    return {
        "scenario": scenario, "gap": gap, "threshold_m": threshold,
        "method": sorted(methods_used) if methods_used else None,
        "N_expected": n_expected,
        "N_no_pre_gap_track": n_no_pre_gap_track,
        "N_insufficient_resolution": n_insufficient_resolution,
        "N_evaluable": n_evaluable,
        "N_success": n_success,
        "success_rate_over_evaluable": (n_success / n_evaluable) if n_evaluable else None,
        "success_rate_over_expected": n_success / n_expected,
    }


def build_full_breakdown() -> list[dict]:

    rows = []
    for scenario, gaps in ALL_SCENARIOS_GAPS.items():
        for gap in gaps:
            for threshold in PRECISION_THRESHOLDS_M:
                rows.append(_breakdown_row(scenario, gap, threshold))
    return rows


def build_h_precision_table(rows: list[dict]) -> dict:
    
    table = {}
    for scenario, gaps in ALL_SCENARIOS_GAPS.items():
        table[scenario] = {}
        for threshold in PRECISION_THRESHOLDS_M:
            rate_by_gap = {}
            n_evaluable_by_gap = {}
            for gap in gaps:
                row = next(r for r in rows if r["scenario"] == scenario and r["gap"] == gap and r["threshold_m"] == threshold)
                n_evaluable_by_gap[gap] = row["N_evaluable"]
                if row["success_rate_over_evaluable"] is not None:
                    rate_by_gap[gap] = row["success_rate_over_evaluable"]
            key = f"H_{int(threshold)}m"
            table[scenario][key] = {
                "rate_by_gap": rate_by_gap,
                "n_evaluable_by_gap": n_evaluable_by_gap,
                "all_points_N20": all(n == len(SEEDS) for n in n_evaluable_by_gap.values()),
                "H_95": recovery_horizon(rate_by_gap, 0.95) or "NOT_ACHIEVED",
                "H_90": recovery_horizon(rate_by_gap, 0.90) or "NOT_ACHIEVED",
                "H_50": recovery_horizon(rate_by_gap, 0.50) or "NOT_ACHIEVED",
            }
    return table


if __name__ == "__main__":
    rows = build_full_breakdown()
    h_table = build_h_precision_table(rows)

    all_n20_global = True
    for scenario, per_th in h_table.items():
        for th_key, data in per_th.items():
            if not data["all_points_N20"]:
                all_n20_global = False
            print(f"  {scenario} {th_key}: H95={data['H_95']} H90={data['H_90']} H50={data['H_50']} all_points_N20={data['all_points_N20']}")
    print(f"\nDo ALL points used have N=20? {all_n20_global}")
