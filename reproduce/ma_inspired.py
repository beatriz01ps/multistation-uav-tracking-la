"""Reproduce the Ma-inspired baseline."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiment.gap_horizon.campaign import (
    CaseTask,
    reduced_ma_case_tasks,
    run_ma_case_standalone,
)
from _prepare_ma_inputs import prepare_case

EXPECTED_CSV = (
    REPO_ROOT
    / "expected"
    / "ma_inspired"
    / "expected_cases.csv"
)

REPRESENTATIVE_CASES = [
    ("gap_horizon_1uav_maneuver", 5001, 10.0), #single
    ("gap_horizon_2uav_crossing", 5001, 10.0), #mukti
]

FIELDS = [
    "reconciliation_outcome",
    "pre_gap_final_id",
    "post_gap_final_id",
    "post_gap_majority_target",
    "number_of_segments",
    "number_of_gtm_candidates",
    "number_of_gtm_assignments",
    "number_of_reconciliations",
    "number_of_reconstructed_epochs",
]


def _load_expected() -> dict[tuple[str, int, float], dict]:

    expected = {}

    with EXPECTED_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["scenario"],
                int(row["seed"]),
                float(row["gap_s"]),
            )
            expected[key] = row

    return expected


def _compare(result: dict, expected_row: dict) -> tuple[bool, list[str]]:
    diffs = []

    for field in FIELDS:
        fresh = str(result[field])
        expected = expected_row[field]

        if fresh != expected:
            diffs.append(
                f"{field}: fresh={fresh} expected={expected}"
            )

    reconstruction = result.get("offline_reconstruction_error")
    expected_reconstruction = expected_row[
        "recon_mean_position_error_m"
    ]

    fresh_reconstruction = (
        ""
        if reconstruction is None
        else str(reconstruction["mean_position_error_m"])
    )

    if fresh_reconstruction != expected_reconstruction:
        diffs.append(
            "recon_mean_position_error_m: "
            f"fresh={fresh_reconstruction} "
            f"expected={expected_reconstruction}"
        )

    return not diffs, diffs


def run_case(scenario: str, seed: int, gap: float, expected: dict) -> bool:

    prepare_case(scenario, seed)

    task = CaseTask(
        scenario,
        seed,
        gap,
    )

    result = run_ma_case_standalone(task)

    expected_row = expected.get(
        (scenario, seed, gap)
    )

    if expected_row is None:
        print(
            f"{scenario} seed{seed} "
            f"gap{gap:g}s: NO EXPECTED ROW"
        )
        return False

    ok, diffs = _compare(
        result,
        expected_row,
    )

    print(
        f"{scenario} seed{seed} "
        f"gap{gap:g}s: "
        f"{'MATCH' if ok else 'DIFFERS'}"
    )

    for diff in diffs:
        print(f"    {diff}")

    return ok


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="run all 720 cases instead of the 2 representative ones",
    )
    args = parser.parse_args()

    expected = _load_expected()

    print("Ma-inspired baseline")
    print("-" * 21)

    all_ok = True

    if args.full:
        for task in reduced_ma_case_tasks():
            ok = run_case(
                task.scenario,
                task.seed,
                task.gap_duration,
                expected,
            )
            all_ok = all_ok and ok
    else:
        for scenario, seed, gap in REPRESENTATIVE_CASES:
            ok = run_case(
                scenario,
                seed,
                gap,
                expected,
            )
            all_ok = all_ok and ok

    print(f"\nOVERALL: {'MATCH' if all_ok else 'DIFFERS'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()