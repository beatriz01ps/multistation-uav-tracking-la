"""Runs MA_INSPIRED over GAP_HORIZON derived inputs"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiment.gap_horizon.blackout import BlackoutSpec, build_blackout_derived_input, write_blackout_derived_sim_dir
from experiment.gap_horizon.metrics import classify_recovery, offline_reconstruction_error
from experiment.gap_horizon.sanity_checks import verify_blackout_contract
from experiment.gap_horizon.study_config import (
    MA_COMPARISON_SCENARIOS,
    MA_EXPLORATORY_SEEDS,
    REDUCED_MA_GAP_RANGE,
    REPO_ROOT,
    TARGET_UNDER_TEST_NAME,
    gap_window,
    scenario_path,
)

BASE_INPUTS_DIR = REPO_ROOT / "results" / "gap_horizon_base_inputs" / "unclassified"
DERIVED_DIR = REPO_ROOT / "results" / "gap_horizon_derived"


def _find_base_sim_dir(scenario: str, seed: int) -> Path:
    candidates = list(BASE_INPUTS_DIR.glob(f"{scenario}__seed{seed}__*"))
    if not candidates:
        raise FileNotFoundError(f"base input not found for {scenario}/seed{seed} in {BASE_INPUTS_DIR}")
    return candidates[0] / "sim"


def derived_sim_dir_for(scenario: str, seed: int, gap_duration: float) -> Path:
    return DERIVED_DIR / scenario / f"seed{seed}" / f"gap_{gap_duration:g}s" / "sim"


@dataclass(frozen=True)
class DerivedInputTask:
    scenario: str
    seed: int
    gap_duration: float


def prepare_one_derived_input(task: DerivedInputTask) -> Path:

    """Builds the derived (blackout) input for (scenario, seed, gap) at its shared path and returns the sim_dir."""

    sim_dir = derived_sim_dir_for(task.scenario, task.seed, task.gap_duration)
    base_sim_dir = _find_base_sim_dir(task.scenario, task.seed)
    gap_start, gap_end = gap_window(task.gap_duration)

    spec = BlackoutSpec(target_true_name=TARGET_UNDER_TEST_NAME, gap_start_s=gap_start, gap_end_s=gap_end)
    result = build_blackout_derived_input(base_sim_dir, spec)
    check = verify_blackout_contract(base_sim_dir, result, spec)
    if not check.passed:
        # Never proceed with a derived input that violates the contract - this
        # would be an infrastructure bug, not a scientific result, so it fails loudly.
        raise RuntimeError(
            f"blackout contract failed for {task.scenario}/seed{task.seed}/gap{task.gap_duration}s: {check}"
        )

    write_blackout_derived_sim_dir(base_sim_dir, result, sim_dir)
    return sim_dir


def ma_output_dir(scenario: str, seed: int, gap_duration: float) -> Path:
    return DERIVED_DIR / scenario / f"seed{seed}" / f"gap_{gap_duration:g}s" / "ma_inspired"


@dataclass(frozen=True)
class CaseTask:
    scenario: str
    seed: int
    gap_duration: float


def reduced_ma_case_tasks() -> list[CaseTask]:

    """The public MA_INSPIRED sweep - 3 scenarios x gaps 1..24 x seeds 5001..5010 = 3x24x10 = 720."""

    return [
        CaseTask(scenario, seed, float(gap))
        for scenario in MA_COMPARISON_SCENARIOS
        for seed in MA_EXPLORATORY_SEEDS
        for gap in REDUCED_MA_GAP_RANGE
    ]


def _rmtree_if_exists(path: Path) -> None:
    if path.is_dir():
        import shutil

        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)


def cleanup_shared_derived_input(scenario: str, seed: int, gap_duration: float) -> None:

    """Deletes the derived input's bulky files after a case consumes them cheap to rebuild from the (never-deleted) base input if needed again."""

    sim_dir = derived_sim_dir_for(scenario, seed, gap_duration)
    for filename in ("sent_messages.jsonl", "tracklet_origins.jsonl", "ground_truth.jsonl", "responses.jsonl"):
        (sim_dir / filename).unlink(missing_ok=True)


_RECONCILIATION_LABEL = {
    "CORRECT_RECOVERY": "CORRECT_RECONCILIATION",
    "NO_RECOVERY": "NO_RECONCILIATION",
    "FALSE_RECOVERY": "FALSE_RECONCILIATION",
    "NEVER_REDETECTED": "NEVER_RECONCILED",
}


def run_ma_case(task: CaseTask) -> dict:

    from evaluation.association_metrics import local_track_truth_map
    from evaluation.offline.loaders import load_association_events, load_ground_truth, load_track_history, load_tracklet_origins
    from experiment.literature_baselines.ma_inspired_pipeline import run_ma_inspired

    sim_dir = prepare_one_derived_input(DerivedInputTask(task.scenario, task.seed, task.gap_duration))
    output_dir = ma_output_dir(task.scenario, task.seed, task.gap_duration)
    gap_start, gap_end = gap_window(task.gap_duration)

    pipeline_result = run_ma_inspired(sim_dir, scenario_path(task.scenario), output_dir)

    final_sim_dir = output_dir / "final" / "sim"
    final_tracks_dir = output_dir / "final" / "tracks"
    truth_labels = load_tracklet_origins(final_sim_dir / "tracklet_origins.jsonl")
    local_track_truth = local_track_truth_map(truth_labels)
    events = load_association_events(final_sim_dir / "responses.jsonl")
    recovery = classify_recovery(events, local_track_truth, TARGET_UNDER_TEST_NAME, gap_start, gap_end)

    reconstruction_summary = None
    if recovery.pre_gap_id is not None:
        rows = load_track_history(final_tracks_dir / "track_history.jsonl")
        ground_truth = load_ground_truth(final_sim_dir / "ground_truth.jsonl")
        samples = offline_reconstruction_error(rows, ground_truth[TARGET_UNDER_TEST_NAME], recovery.pre_gap_id, gap_start, gap_end)
        if samples:
            errors = [s.position_error_m for s in samples]
            reconstruction_summary = {"num_samples": len(samples), "mean_position_error_m": sum(errors) / len(errors), "max_position_error_m": max(errors)}

    result = {
        "scenario": task.scenario, "seed": task.seed, "gap_duration_s": task.gap_duration,
        "gap_start_s": gap_start, "gap_end_s": gap_end,
        "reconciliation_outcome": _RECONCILIATION_LABEL[recovery.outcome],
        "pre_gap_final_id": recovery.pre_gap_id, "post_gap_final_id": recovery.post_gap_id,
        "post_gap_majority_target": recovery.post_gap_majority_target,
        "number_of_segments": pipeline_result["number_of_segments"],
        "number_of_gtm_candidates": pipeline_result["number_of_gtm_candidates"],
        "number_of_gtm_assignments": pipeline_result["number_of_gtm_assignments"],
        "number_of_reconciliations": pipeline_result["number_of_reconciliations"],
        "number_of_reconstructed_epochs": pipeline_result["number_of_reconstructed_epochs"],
        "offline_reconstruction_error": reconstruction_summary,
        "causal_prediction_horizon": "N/A - MA_INSPIRED has no causal prediction",
    }
    cleanup_ma_private_outputs(output_dir)
    return result


def cleanup_ma_private_outputs(output_dir: Path) -> None:

    """Deletes run_ma_inspired's already-consumed outputs. Never read again once run_ma_case has returned. Preserves final/metrics.json."""

    for name in ("raw_segments", "gtm"):
        _rmtree_if_exists(output_dir / name)
    final_sim_dir = output_dir / "final" / "sim"
    for filename in ("sent_messages.jsonl", "ground_truth.jsonl", "tracklet_origins.jsonl", "responses.jsonl"):
        (final_sim_dir / filename).unlink(missing_ok=True)
    _rmtree_if_exists(output_dir / "final" / "tracks")


def run_ma_case_standalone(task: CaseTask) -> dict:

    """Runs one MA_INSPIRED case and cleans up the shared derived input
    afterward (`prepare_one_derived_input` rebuilds it deterministically
    from the base input if a later case needs it again)."""

    result = run_ma_case(task)
    cleanup_shared_derived_input(task.scenario, task.seed, task.gap_duration)
    return result
