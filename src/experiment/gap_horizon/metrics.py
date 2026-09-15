"""GAP_HORIZON metrics: identity recovery classification and offline (post-GTM)
reconstruction error, computed read-only over already-persisted artifacts. GT only enters here as **evaluator/diagnostic*, never operationally."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from evaluation.association_metrics import AssociationEvent, LocalTrackTruth, majority_label_per_global_id
from evaluation.offline.loaders import GroundTruthTrajectory, TrackHistoryRow

RecoveryOutcome = Literal["CORRECT_RECOVERY", "NO_RECOVERY", "FALSE_RECOVERY", "NEVER_REDETECTED"]


@dataclass(frozen=True)
class RecoveryResult:
    outcome: RecoveryOutcome
    pre_gap_id: int | None
    post_gap_id: int | None
    post_gap_majority_target: str | None
    first_post_gap_detection_timestamp: float | None


def classify_recovery(events: list[AssociationEvent], local_track_truth: LocalTrackTruth, target_true_name: str,
    gap_start_s: float, gap_end_s: float) -> RecoveryResult:

    """events = every AssociationEvent for the run (real, or MA's synthesized
    equivalent - see ma_inspired_pipeline._write_association_events). local_track_truth is built from the deruved tracklet_origins.jsonl (post-gap ids)."""

    majority_labels = majority_label_per_global_id(events, local_track_truth)

    pre_gap_events = sorted(
        (e for e in events if e.timestamp < gap_start_s and local_track_truth.get((e.station_id, e.local_track_id)) == target_true_name),
        key=lambda e: e.timestamp,
    )
    pre_gap_id = pre_gap_events[-1].global_track_id if pre_gap_events else None

    post_gap_events = sorted(
        (e for e in events if e.timestamp >= gap_end_s and local_track_truth.get((e.station_id, e.local_track_id)) == target_true_name),
        key=lambda e: e.timestamp,
    )
    if not post_gap_events:
        return RecoveryResult("NEVER_REDETECTED", pre_gap_id, None, None, None)

    first_post_gap = post_gap_events[0]
    post_gap_id = first_post_gap.global_track_id
    post_gap_majority_target = majority_labels.get(post_gap_id)

    if post_gap_majority_target != target_true_name:
        # the global_track_id that "inherited" the post-gap label actually
        # represents (in majority) another physical target - a wrong fusion.
        outcome: RecoveryOutcome = "FALSE_RECOVERY"
    elif pre_gap_id is not None and post_gap_id == pre_gap_id:
        outcome = "CORRECT_RECOVERY"
    else:
        outcome = "NO_RECOVERY"

    return RecoveryResult(outcome, pre_gap_id, post_gap_id, post_gap_majority_target, first_post_gap.timestamp)


@dataclass(frozen=True)
class OfflineReconstructionSample:
    kind: str  # always "OFFLINE_RECONSTRUCTED" never confused with a causal prediction
    timestamp: float
    position_error_m: float
    velocity_error_mps: float


def offline_reconstruction_error( rows: list[TrackHistoryRow], ground_truth: GroundTruthTrajectory, final_id: int, gap_start_s: float, gap_end_s: float) -> list[OfflineReconstructionSample]:

    """final_id = the post-GTM-reconciliation identity of the pre-gap fragment.
    Only prediction_only=True rows (retrodiction, never a real measurement) within [gap_start_s, gap_end_s) count."""
    
    samples: list[OfflineReconstructionSample] = []
    for row in sorted(rows, key=lambda r: r.timestamp):
        if row.global_track_id != final_id or not row.prediction_only:
            continue
        if not (gap_start_s <= row.timestamp < gap_end_s):
            continue
        true_state = ground_truth.at(row.timestamp)
        if true_state is None:
            continue
        samples.append(
            OfflineReconstructionSample(
                kind="OFFLINE_RECONSTRUCTED",
                timestamp=row.timestamp,
                position_error_m=float(np.linalg.norm(row.state[0:3] - true_state[0:3])),
                velocity_error_mps=float(np.linalg.norm(row.state[3:6] - true_state[3:6])),
            )
        )
    return samples
