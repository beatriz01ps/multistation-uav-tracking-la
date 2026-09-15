"""Compute paired comparisons and statistical summaries."""

from __future__ import annotations

import random
import statistics
from typing import Any, Optional

DEFAULT_BOOTSTRAP_REPLICATES = 10_000


def paired_deltas(reference_rows: list[dict], variant_rows: list[dict], metric: str) -> list[dict[str, Any]]:

    """Compute paired deltas matched by scenario and seed."""

    ref_by_key = {(r["scenario_id"], r["seed"]): r for r in reference_rows if r.get("status") == "completed"}
    deltas = []
    for variant_row in variant_rows:
        if variant_row.get("status") != "completed":
            continue
        key = (variant_row["scenario_id"], variant_row["seed"])
        ref_row = ref_by_key.get(key)
        if ref_row is None:
            continue
        ref_value, variant_value = ref_row.get(metric), variant_row.get(metric)
        if ref_value is None or variant_value is None:
            continue
        #exclude NaN values
        if ref_value != ref_value or variant_value != variant_value:
            continue
        deltas.append(
            {
                "scenario_id": key[0],
                "seed": key[1],
                "metric": metric,
                "reference_value": ref_value,
                "variant_value": variant_value,
                "delta": variant_value - ref_value,
            }
        )
    return deltas


def pairing_coverage(reference_rows: list[dict], variant_rows: list[dict]) -> dict[str, Any]:

    """Summarize attempted, surviving, and failed pairs."""

    attempted_keys = {(r["scenario_id"], r["seed"]) for r in reference_rows} | {
        (r["scenario_id"], r["seed"]) for r in variant_rows
    }
    reference_completed = {(r["scenario_id"], r["seed"]) for r in reference_rows if r.get("status") == "completed"}
    variant_completed = {(r["scenario_id"], r["seed"]) for r in variant_rows if r.get("status") == "completed"}
    surviving_pairs = reference_completed & variant_completed

    return {
        "num_pairs_attempted": len(attempted_keys),
        "num_pairs_surviving": len(surviving_pairs),
        "num_pairs_lost_to_failure": len(attempted_keys) - len(surviving_pairs),
        "num_lost_due_to_reference_failure": len(attempted_keys - reference_completed),
        "num_lost_due_to_variant_failure": len(attempted_keys - variant_completed),
    }


def build_paired_comparisons_rows(comparison_name: str, reference_rows: list[dict], variant_rows: list[dict], metric_columns: list[str]) -> list[dict[str, Any]]:
    rows = []
    for metric in metric_columns:
        for delta in paired_deltas(reference_rows, variant_rows, metric):
            rows.append({"comparison": comparison_name, **delta})
    return rows


def hierarchical_bootstrap_ci(deltas: list[float], scenario_ids: list[str], *, num_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = 12345, confidence: float = 0.95) -> dict[str, Any]:

    """Compute a hierarchical bootstrap confidence interval."""

    by_scenario: dict[str, list[float]] = {}
    for scenario_id, delta in zip(scenario_ids, deltas):
        by_scenario.setdefault(scenario_id, []).append(delta)
    scenarios = list(by_scenario.keys())

    if not scenarios:
        return {
            "mean": None, "ci_lower": None, "ci_upper": None,
            "num_replicates": num_replicates, "num_scenarios": 0, "num_pairs": 0,
        }

    rng = random.Random(seed)
    replicate_means = []
    for _ in range(num_replicates):
        resampled_scenarios = [scenarios[rng.randrange(len(scenarios))] for _ in scenarios]
        scenario_means = []
        for scenario_id in resampled_scenarios:
            seed_values = by_scenario[scenario_id]
            resampled = [seed_values[rng.randrange(len(seed_values))] for _ in seed_values]
            scenario_means.append(sum(resampled) / len(resampled))
        replicate_means.append(sum(scenario_means) / len(scenario_means))

    replicate_means.sort()
    alpha = 1.0 - confidence
    lower_idx = max(0, int((alpha / 2) * num_replicates))
    upper_idx = min(num_replicates - 1, int((1 - alpha / 2) * num_replicates) - 1)

    return { #me
        "mean": sum(deltas) / len(deltas),
        "median": statistics.median(deltas),
        "ci_lower": replicate_means[lower_idx],
        "ci_upper": replicate_means[upper_idx],
        "num_replicates": num_replicates,
        "num_scenarios": len(scenarios),
        "num_pairs": len(deltas),
    }


def wilcoxon_signed_rank_p_value(deltas: list[float]) -> Optional[float]:

    """Compute the Wilcoxon signed-rank p-value."""

    nonzero = [d for d in deltas if d != 0]
    if not nonzero:
        return None
    from scipy import stats as scipy_stats

    try:
        _, p_value = scipy_stats.wilcoxon(nonzero)
    except ValueError:
        return None
    return float(p_value)


def holm_correction(p_values: dict[str, float]) -> dict[str, float]:

    """Apply Holm's step-down correction."""
    
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for i, (label, p_value) in enumerate(items):
        corrected = min((m - i) * p_value, 1.0)
        running_max = max(running_max, corrected)
        adjusted[label] = running_max
    return adjusted