"""Build the paired OFAT/Lego contrasts and statistical summaries."""

from __future__ import annotations

from typing import Any

from aggregation.paired import (
    build_paired_comparisons_rows,
    hierarchical_bootstrap_ci,
    pairing_coverage,
    wilcoxon_signed_rank_p_value,
)

LOWER_IS_BETTER = [
    "position_rmse_update_m", "position_rmse_coasting_m", "position_rmse_overall_m",
    "velocity_rmse_update_mps", "velocity_rmse_coasting_mps", "velocity_rmse_overall_mps",
    "id_switches", "identity_switches_epoch_level", "fragmentation_events_total", "excess_global_tracks_total",
    "recovery_identity_recovery_latency_mean",
]
HIGHER_IS_BETTER = ["association_accuracy", "track_availability_any_active_mean", "track_availability_confirmed_mean"]
CONSISTENCY_ONLY = ["mean_nees_6d", "mean_nis"]

METRIC_COLUMNS = LOWER_IS_BETTER + HIGHER_IS_BETTER + CONSISTENCY_ONLY

#pin all reference factors to keep supplementary runs out of OFAT comparisons
REFERENCE_FACTORS = {
    "estimator": "ukf", "fusion": "information", "association": "mahalanobis", "merger_enabled": True,
    "association_mode": "position_only", "motion_model": "coordinated_turn",
}
OFAT_CONTRASTS = {
    "temporal_ekf_vs_reference": {**REFERENCE_FACTORS, "estimator": "ekf"},
    "t2tf_ci_vs_reference": {**REFERENCE_FACTORS, "fusion": "covariance_intersection"},
    "association_euclidean_vs_reference": {**REFERENCE_FACTORS, "association": "euclidean"},
    "identity_merger_off_vs_reference": {**REFERENCE_FACTORS, "merger_enabled": False},
}


def _filter_pipeline(rows: list[dict], factors: dict[str, Any]) -> list[dict]:

    return [r for r in rows if all(r.get(k) == v for k, v in factors.items())]


def build_paired_comparisons_table(rows: list[dict]) -> list[dict]:

    reference_rows = _filter_pipeline(rows, REFERENCE_FACTORS)
    all_rows = []
    for comparison_name, factors in OFAT_CONTRASTS.items():
        variant_rows = _filter_pipeline(rows, factors)
        all_rows.extend(build_paired_comparisons_rows(comparison_name, reference_rows, variant_rows, METRIC_COLUMNS))
    return all_rows


MOTION_MODEL_CONTRAST = {
    "motion_model_cv_vs_ct": {**REFERENCE_FACTORS, "motion_model": "constant_velocity"},
}
FULL_STATE_CONTRAST = {
    "association_full_state_vs_position_only": {**REFERENCE_FACTORS, "association_mode": "full_state"},
}
NO_FUSION_CONTRASTS = {
    "fusion_local_only_vs_information": {**REFERENCE_FACTORS, "fusion": "local_only"},
}


def _supplementary_comparison_rows(rows: list[dict], contrasts: dict[str, dict]) -> list[dict]:

    reference_rows = _filter_pipeline(rows, REFERENCE_FACTORS)
    all_rows = []
    for comparison_name, factors in contrasts.items():
        variant_rows = _filter_pipeline(rows, factors)
        all_rows.extend(build_paired_comparisons_rows(comparison_name, reference_rows, variant_rows, METRIC_COLUMNS))
    return all_rows


def build_motion_model_comparison_rows(rows: list[dict]) -> list[dict]:

    """Build the CV vs CT comparison."""

    return _supplementary_comparison_rows(rows, MOTION_MODEL_CONTRAST)


def build_full_state_comparison_rows(rows: list[dict]) -> list[dict]:

    """Build the FULL_STATE vs POSITION_ONLY comparison."""

    return _supplementary_comparison_rows(rows, FULL_STATE_CONTRAST)


def build_no_fusion_comparison_rows(rows: list[dict]) -> list[dict]:

    """Build the local-only vs information-fusion comparison."""

    return _supplementary_comparison_rows(rows, NO_FUSION_CONTRASTS)


def build_pairing_coverage_summary(rows: list[dict]) -> list[dict]:

    """Build pairing coverage for each OFAT contrast."""

    reference_rows = _filter_pipeline(rows, REFERENCE_FACTORS)
    out = []
    for comparison_name, factors in OFAT_CONTRASTS.items():
        variant_rows = _filter_pipeline(rows, factors)
        out.append({"comparison": comparison_name, **pairing_coverage(reference_rows, variant_rows)})
    return out


def build_ofat_ci_summary(paired_rows: list[dict]) -> list[dict]:
    
    """Build bootstrap CI and raw Wilcoxon summaries."""

    groups: dict[tuple[str, str], list[dict]] = {}
    for row in paired_rows:
        groups.setdefault((row["comparison"], row["metric"]), []).append(row)

    out = []
    for (comparison, metric), group_rows in groups.items():
        deltas = [r["delta"] for r in group_rows]
        scenario_ids = [r["scenario_id"] for r in group_rows]
        ci = hierarchical_bootstrap_ci(deltas, scenario_ids)
        p_value = wilcoxon_signed_rank_p_value(deltas)
        out.append({"comparison": comparison, "metric": metric, "p_value_wilcoxon": p_value, **ci})
    return out