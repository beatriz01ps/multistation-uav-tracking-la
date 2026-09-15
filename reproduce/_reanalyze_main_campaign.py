"""
Recompute the statistical analyses used by Table 3 from`data/table3/run_metrics.csv`.
Internal helper used by `reproduce/table3.py`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import aggregation.holm_by_research_question as holm_module 
from aggregation.report import ( 
    
    METRIC_COLUMNS,
    build_full_state_comparison_rows,
    build_motion_model_comparison_rows,
    build_no_fusion_comparison_rows,
    build_ofat_ci_summary,
    build_pairing_coverage_summary,
    build_paired_comparisons_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "table3" / "run_metrics.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "table3" / "_generated"

_BOOL_TRUE = {"true", "1", "yes"}


def load_run_metrics(csv_path: Path) -> list[dict]:
    """Load run metrics with the types required by the aggregation code."""
    rows = []
    with csv_path.open(encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            row["merger_enabled"] = raw.get("merger_enabled", "").strip().lower() in _BOOL_TRUE
            for col in METRIC_COLUMNS:
                value = raw.get(col, "")
                row[col] = float(value) if value not in ("", "None", None) else None
            rows.append(row)
    return rows


def build_all_paired_rows(rows: list[dict]) -> dict[str, list[dict]]:
    """Build the four paired-comparison groups used by Table 3."""
    return {
        "paired_comparisons": build_paired_comparisons_table(rows),
        "full_state_comparison": build_full_state_comparison_rows(rows),
        "motion_model_comparison": build_motion_model_comparison_rows(rows),
        "no_fusion_comparison": build_no_fusion_comparison_rows(rows),
    }


def write_csv(path: Path, rows: list[dict]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_reanalysis(input_csv: Path, output_dir: Path) -> dict[str, Path]:

    rows = load_run_metrics(input_csv)
    grouped = build_all_paired_rows(rows)

    written: dict[str, Path] = {}
    for group_name, paired_rows in grouped.items():
        raw_path = output_dir / f"{group_name}.csv"
        write_csv(raw_path, paired_rows)
        written[group_name] = raw_path

        ci_rows = build_ofat_ci_summary(paired_rows)
        ci_path = output_dir / f"{group_name}_ci_summary.csv"
        write_csv(ci_path, ci_rows)
        written[f"{group_name}_ci_summary"] = ci_path

    coverage_rows = build_pairing_coverage_summary(rows)
    coverage_path = output_dir / "pairing_coverage.csv"
    write_csv(coverage_path, coverage_rows)
    written["pairing_coverage"] = coverage_path

    original_dir = holm_module.AGGREGATED_DIR
    holm_module.AGGREGATED_DIR = output_dir
    try:
        holm_rows = holm_module.build_holm_table()
    finally:
        holm_module.AGGREGATED_DIR = original_dir

    holm_path = output_dir / "HOLM_ADJUSTED_PVALUES.csv"
    write_csv(holm_path, holm_rows)
    written["holm_adjusted_pvalues"] = holm_path

    return written

def _read_ci_summary(path: Path) -> dict[tuple[str, str], dict]:

    if not path.exists():
        return {}

    with path.open(encoding="utf-8") as handle:
        return {
            (row["comparison"], row["metric"]): row
            for row in csv.DictReader(handle)
        }

def compare_against(generated_dir: Path, bundled_dir: Path, tolerance: float = 1e-6,) -> bool:
    """Compare generated CI summaries with the bundled expected results."""
    all_ok = True

    groups = [
        "paired_comparisons",
        "full_state_comparison",
        "motion_model_comparison",
        "no_fusion_comparison",
    ]

    for group in groups:
        generated = _read_ci_summary(generated_dir / f"{group}_ci_summary.csv")
        bundled = _read_ci_summary(bundled_dir / f"{group}_ci_summary.csv")

        keys = sorted(set(generated) | set(bundled))
        for key in keys:
            g = generated.get(key)
            b = bundled.get(key)

            if g is None or b is None:
                print(
                    f"  MISSING  {group} {key}: "
                    f"generated={g is not None} bundled={b is not None}"
                )
                all_ok = False
                continue

            for field in ("mean", "median", "num_pairs"):
                gv = g.get(field)
                bv = b.get(field)

                try:
                    ok = abs(float(gv) - float(bv)) <= tolerance
                except (TypeError, ValueError):
                    ok = gv == bv

                if not ok:
                    all_ok = False
                    print(
                        f"  FAIL  {group} {key} {field}: "
                        f"generated={gv} bundled={bv}"
                    )

    return all_ok


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compare-to", type=Path, default=None)
    args = parser.parse_args()

    written = run_reanalysis(args.input, args.output_dir)

    print(f"reanalysis written to {args.output_dir}:")
    for name, path in written.items():
        print(f"  {name}: {path}")

    if args.compare_to:
        print(f"\ncomparing against {args.compare_to} (tolerance=1e-6):")
        ok = compare_against(args.output_dir, args.compare_to)
        print("\nALL MATCHED" if ok else "\nSOME MISMATCHES FOUND (see above)")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":

    main()