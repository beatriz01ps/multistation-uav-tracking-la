"""Apply Holm correction within each Research Question family."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from aggregation.paired import holm_correction  # noqa: E402

AGGREGATED_DIR = REPO_ROOT / "results" / "final" / "aggregated"

# Each entry: (readable_label, source_csv_file, comparison_name, metric).
RQ1_FUSION_TEMPORAL_ESTIMATION = [
    ("local_only_vs_information__position_rmse_update_m", "no_fusion_comparison_ci_summary.csv", "fusion_local_only_vs_information", "position_rmse_update_m"),
    ("information_vs_covariance_intersection__position_rmse_overall_m", "paired_comparisons_ci_summary.csv", "t2tf_ci_vs_reference", "position_rmse_overall_m"),
    ("ekf_vs_ukf__position_rmse_update_m", "paired_comparisons_ci_summary.csv", "temporal_ekf_vs_reference", "position_rmse_update_m"),
    ("cv_vs_ct__position_rmse_overall_m", "motion_model_comparison_ci_summary.csv", "motion_model_cv_vs_ct", "position_rmse_overall_m"),
]

RQ2_ASSOCIATION = [
    ("euclidean_vs_mahalanobis_position_only__association_accuracy", "paired_comparisons_ci_summary.csv", "association_euclidean_vs_reference", "association_accuracy"),
    ("euclidean_vs_mahalanobis_position_only__identity_switches_epoch_level", "paired_comparisons_ci_summary.csv", "association_euclidean_vs_reference", "identity_switches_epoch_level"),
    ("position_only_vs_full_state__association_accuracy", "full_state_comparison_ci_summary.csv", "association_full_state_vs_position_only", "association_accuracy"),
    ("position_only_vs_full_state__identity_switches_epoch_level", "full_state_comparison_ci_summary.csv", "association_full_state_vs_position_only", "identity_switches_epoch_level"),
]

RQ3_IDENTITY_MANAGEMENT = [
    ("merger_off_vs_on__identity_switches_epoch_level", "paired_comparisons_ci_summary.csv", "identity_merger_off_vs_reference", "identity_switches_epoch_level"),
    ("merger_off_vs_on__fragmentation_events_total", "paired_comparisons_ci_summary.csv", "identity_merger_off_vs_reference", "fragmentation_events_total"),
]

FAMILIES = {
    "RQ1_fusion_temporal_estimation": RQ1_FUSION_TEMPORAL_ESTIMATION,
    "RQ2_association": RQ2_ASSOCIATION,
    "RQ3_identity_management": RQ3_IDENTITY_MANAGEMENT,
}


def _load_csv(name: str) -> list[dict]:

    path = AGGREGATED_DIR / name
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _find_p_value(rows: list[dict], comparison: str, metric: str) -> float | None:

    for row in rows:
        if row.get("comparison") == comparison and row.get("metric") == metric:
            raw = row.get("p_value_wilcoxon")
            return float(raw) if raw not in (None, "", "None") else None
    return None


def build_holm_table() -> list[dict]:

    csv_cache: dict[str, list[dict]] = {}
    rows_out = []

    for family_name, entries in FAMILIES.items():
        p_values_raw: dict[str, float] = {}
        entry_meta: dict[str, tuple[str, str, str]] = {}

        for label, source_csv, comparison, metric in entries:
            if source_csv not in csv_cache:
                csv_cache[source_csv] = _load_csv(source_csv)
            p_value = _find_p_value(csv_cache[source_csv], comparison, metric)
            entry_meta[label] = (source_csv, comparison, metric)
            if p_value is not None:
                p_values_raw[label] = p_value

        adjusted = holm_correction(p_values_raw) if p_values_raw else {}

        for label, (source_csv, comparison, metric) in entry_meta.items():
            rows_out.append(
                {
                    "research_question": family_name,
                    "contrast_metric_label": label,
                    "source_csv": source_csv,
                    "comparison": comparison,
                    "metric": metric,
                    "p_value_wilcoxon_raw": p_values_raw.get(label),
                    "p_value_holm_adjusted": adjusted.get(label),
                    "family_size": len(entry_meta),
                    "missing_raw_p_value": label not in p_values_raw,
                }
            )
    return rows_out


def _write_csv(path: Path, rows: list[dict]) -> None:
    
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    rows = build_holm_table()
    out_path = AGGREGATED_DIR / "HOLM_ADJUSTED_PVALUES.csv"
    _write_csv(out_path, rows)
    print(f"wrote {out_path} ({len(rows)} rows)")
    for row in rows:
        flag = " [MISSING RAW P-VALUE]" if row["missing_raw_p_value"] else ""
        print(
            f"  {row['research_question']:28s} {row['contrast_metric_label']:65s} "
            f"raw={row['p_value_wilcoxon_raw']}  holm={row['p_value_holm_adjusted']}{flag}"
        )
