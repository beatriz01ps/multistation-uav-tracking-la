"""Reproduce Table 3 and the complete-architecture evaluation."""

from __future__ import annotations

import csv
import json
import statistics
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from _reanalyze_main_campaign import (
    DEFAULT_INPUT,
    _read_ci_summary,
    run_reanalysis,
)

EXPECTED_DIR = REPO_ROOT / "expected" / "table3"
COMPLETE_ARCHITECTURE_CSV = (
    REPO_ROOT
    / "data"
    / "complete_architecture"
    / "run_metrics.csv"
)
COMPLETE_ARCHITECTURE_EXPECTED = (
    REPO_ROOT
    / "expected"
    / "complete_architecture"
    / "summary.json"
)

COMPARISONS = [ #formatting easier to see
    (
        "no_fusion_comparison",
        "fusion_local_only_vs_information",
        "Local-Only vs IF",
    ),
    (
        "paired_comparisons",
        "t2tf_ci_vs_reference",
        "CI vs IF",
    ),
    (
        "paired_comparisons",
        "temporal_ekf_vs_reference",
        "EKF vs UKF",
    ),
    (
        "motion_model_comparison",
        "motion_model_cv_vs_ct",
        "CV vs CT",
    ),
    (
        "paired_comparisons",
        "association_euclidean_vs_reference",
        "Euclidean vs Mahalanobis",
    ),
    (
        "full_state_comparison",
        "association_full_state_vs_position_only",
        "6D vs 3D",
    ),
    (
        "paired_comparisons",
        "identity_merger_off_vs_reference",
        "Merger OFF vs ON",
    ),
]

TOLERANCE = 1e-6

def _comparison_matches(generated: dict, bundled: dict, comparison: str) -> bool:

    generated_rows = {
        row["metric"]: row
        for (current_comparison, _metric), row in generated.items()
        if current_comparison == comparison
    }

    bundled_rows = {
        row["metric"]: row
        for (current_comparison, _metric), row in bundled.items()
        if current_comparison == comparison
    }

    if set(generated_rows) != set(bundled_rows):
        return False

    for metric, generated_row in generated_rows.items():
        bundled_row = bundled_rows[metric]

        for field in ("mean", "median", "num_pairs"):
            try:
                if (
                    abs(
                        float(generated_row[field])
                        - float(bundled_row[field])
                    )
                    > TOLERANCE
                ):
                    return False
            except (TypeError, ValueError):
                if generated_row[field] != bundled_row[field]:
                    return False

    return True


def _read_holm(path: Path) -> dict[tuple[str, str], str]:

    with path.open(encoding="utf-8") as handle:
        return {
            (
                row["comparison"],
                row["metric"],
            ): row["p_value_holm_adjusted"]
            for row in csv.DictReader(handle)
        }


def _holm_matches(generated_dir: Path) -> bool:

    generated = _read_holm(
        generated_dir / "HOLM_ADJUSTED_PVALUES.csv"
    )
    bundled = _read_holm(
        EXPECTED_DIR / "HOLM_ADJUSTED_PVALUES.csv"
    )

    if set(generated) != set(bundled):
        return False

    for key in generated:
        generated_value = generated[key]
        bundled_value = bundled[key]

        if generated_value == "" or bundled_value == "":
            if generated_value != bundled_value:
                return False
            continue

        if (
            abs(float(generated_value) - float(bundled_value))
            > TOLERANCE
        ):
            return False

    return True


def _complete_architecture_matches() -> bool:

    with COMPLETE_ARCHITECTURE_CSV.open(
        encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    expected = json.loads(
        COMPLETE_ARCHITECTURE_EXPECTED.read_text(
            encoding="utf-8"
        )
    )

    def fnum(value: str) -> float:
        return float(value)

    checks = [
        (
            "total runs = 260",
            len(rows) == expected["total_runs"],
        ),
        (
            "all runs completed",
            all(
                row["status"] == "completed"
                for row in rows
            )
            == expected["all_runs_completed"],
        ),
        (
            "median association accuracy = 1.0",
            statistics.median(
                fnum(row["association_accuracy"])
                for row in rows
            )
            == expected["median_association_accuracy"],
        ),
        (
            "median epoch-level identity switches = 0",
            statistics.median(
                fnum(row["identity_switches_epoch_level"])
                for row in rows
            )
            == expected[
                "median_identity_switches_epoch_level"
            ],
        ),
        (
            "fragmentation events = 0 in every run",
            max(
                fnum(row["fragmentation_events_total"])
                for row in rows
            )
            == expected["max_fragmentation_events_total"],
        ),
    ]

    print()
    print("Complete architecture")
    print("-" * 22)

    all_ok = True

    for label, ok in checks:
        all_ok = all_ok and ok
        print(
            f"{label:<70} "
            f"{'MATCH' if ok else 'DIFFERS'}"
        )

    # Descriptive only. NEES does not affect MATCH.
    noise_rows = [
        row
        for row in rows
        if row["scenario_id"]
        == "final_extreme_noise_and_dropout"
    ]

    nees_by_seed = sorted(
        (
            (
                int(row["seed"]),
                fnum(row["mean_nees_6d"]),
            )
            for row in noise_rows
        ),
        key=lambda item: item[1],
    )

    nees_values = [
        value
        for _seed, value in nees_by_seed
    ]

    print()
    print(
        "extreme_noise_and_dropout mean_nees_6d "
        "(descriptive, not part of MATCH): "
        f"min={min(nees_values):.2f} "
        f"median={statistics.median(nees_values):.2f} "
        f"max={max(nees_values):.2f}"
    )

    for seed, value in nees_by_seed:
        print(f"    seed{seed}: {value:.2f}")

    return all_ok


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_dir:
        output_dir = Path(temporary_dir)
        run_reanalysis(
            DEFAULT_INPUT,
            output_dir,
        )

        print("Table 3")
        print("-" * 8)

        all_ok = True

        for group, comparison, label in COMPARISONS:
            generated = _read_ci_summary(
                output_dir
                / f"{group}_ci_summary.csv"
            )

            bundled = _read_ci_summary(
                EXPECTED_DIR
                / f"{group}_ci_summary.csv"
            )

            ok = _comparison_matches(
                generated,
                bundled,
                comparison,
            )

            all_ok = all_ok and ok

            print(
                f"{label:<28} "
                f"{'MATCH' if ok else 'DIFFERS'}"
            )

        holm_ok = _holm_matches(output_dir)
        all_ok = all_ok and holm_ok

        print(
            f"{'Holm-adjusted p-values':<28} "
            f"{'MATCH' if holm_ok else 'DIFFERS'}"
        )

    complete_architecture_ok = (
        _complete_architecture_matches()
    )
    all_ok = all_ok and complete_architecture_ok

    print(
        f"\nOVERALL: "
        f"{'MATCH' if all_ok else 'DIFFERS'}"
    )

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()