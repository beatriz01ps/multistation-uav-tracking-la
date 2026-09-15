"""Recovery metrics: distinguishes target reacquisition (represented again by any
GlobalTrack, always right-censored with no horizon H) from identity-preserving recovery (same global_track_id, with a definitive outcome via TrackTerminalEvent 
**DELETED/MERGED are confirmed failures, no terminal event is censored)."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from evaluation.association_metrics import AssociationEvent, LocalTrackTruth, TruthLabels
from evaluation.offline.loaders import TrackHistoryRow
from tracking.tracker import TrackTerminalEvent


@dataclass(frozen=True)
class RecoveryEvent:
    true_target_id: str
    event_id: int  # sequential 
    pre_gap_global_track_id: int
    last_measurement_timestamp: float

    first_available_timestamp: float | None

    target_reacquired: bool

    target_reacquisition_censored: bool
    first_reacquisition_timestamp: float | None
    reacquired_global_track_id: int | None
    target_reacquisition_latency: float | None

    identity_preserved: bool

    identity_recovery_censored: bool
    first_identity_recovery_timestamp: float | None
    identity_recovery_latency: float | None
    identity_recovery_penalty: float | None

    physical_gap_duration: float | None


def _detect_loss_events(rows: list[TrackHistoryRow], majority_labels: dict[int, str]) -> list[tuple[str, int, float]]:

    by_track: dict[int, list[TrackHistoryRow]] = {}
    for row in rows:
        by_track.setdefault(row.global_track_id, []).append(row)

    events: list[tuple[str, int, float]] = []
    for global_track_id, track_rows in by_track.items():
        target_id = majority_labels.get(global_track_id)
        if target_id is None:
            continue
        track_rows.sort(key=lambda r: r.timestamp)

        last_real_update_timestamp: float | None = None
        for previous, current in zip(track_rows, track_rows[1:]):
            if not previous.prediction_only:
                last_real_update_timestamp = previous.timestamp
            if previous.status == "CONFIRMED" and current.status == "COASTING":
                gap_start = last_real_update_timestamp if last_real_update_timestamp is not None else previous.timestamp
                events.append((target_id, global_track_id, gap_start))
    return events


def _first_available_timestamp_after(truth_labels: TruthLabels, target_id: str, after_timestamp: float) -> float | None:

    candidates = [t for (_, _, t), tid in truth_labels.items() if tid == target_id and t > after_timestamp]
    return min(candidates) if candidates else None


def _first_matching_event_after(events: list[AssociationEvent], local_track_truth: LocalTrackTruth, target_id: str,
    after_timestamp: float, required_global_track_id: int | None = None) -> AssociationEvent | None:

    candidates = [
        event
        for event in events
        if event.timestamp > after_timestamp
        and local_track_truth.get((event.station_id, event.local_track_id)) == target_id
        and (required_global_track_id is None or event.global_track_id == required_global_track_id)
    ]
    return min(candidates, key=lambda e: e.timestamp) if candidates else None


def compute_recovery_events(rows: list[TrackHistoryRow], events: list[AssociationEvent], truth_labels: TruthLabels,
    local_track_truth: LocalTrackTruth, majority_labels: dict[int, str], terminal_events: dict[int, TrackTerminalEvent]) -> list[RecoveryEvent]:

    event_counter: dict[str, int] = {}
    recovery_events: list[RecoveryEvent] = []

    for target_id, pre_gap_global_track_id, last_measurement_timestamp in _detect_loss_events(rows, majority_labels):
        event_counter[target_id] = event_counter.get(target_id, 0) + 1

        first_available_timestamp = _first_available_timestamp_after(truth_labels, target_id, last_measurement_timestamp)

        reacquisition = _first_matching_event_after(events, local_track_truth, target_id, last_measurement_timestamp)
        target_reacquired = reacquisition is not None
        target_reacquisition_censored = not target_reacquired

        identity_recovery = _first_matching_event_after(
            events, local_track_truth, target_id, last_measurement_timestamp, required_global_track_id=pre_gap_global_track_id
        )
        identity_preserved = identity_recovery is not None
        identity_recovery_censored = (
            False if (identity_preserved or pre_gap_global_track_id in terminal_events) else True
        )

        target_reacquisition_latency = (
            reacquisition.timestamp - first_available_timestamp
            if target_reacquired and first_available_timestamp is not None
            else None
        )
        identity_recovery_latency = (
            identity_recovery.timestamp - first_available_timestamp
            if identity_preserved and first_available_timestamp is not None
            else None
        )
        identity_recovery_penalty = (
            identity_recovery_latency - target_reacquisition_latency
            if identity_recovery_latency is not None and target_reacquisition_latency is not None
            else None
        )
        physical_gap_duration = (
            first_available_timestamp - last_measurement_timestamp if first_available_timestamp is not None else None
        )

        recovery_events.append(
            RecoveryEvent(
                true_target_id=target_id,
                event_id=event_counter[target_id],
                pre_gap_global_track_id=pre_gap_global_track_id,
                last_measurement_timestamp=last_measurement_timestamp,
                first_available_timestamp=first_available_timestamp,
                target_reacquired=target_reacquired,
                target_reacquisition_censored=target_reacquisition_censored,
                first_reacquisition_timestamp=reacquisition.timestamp if reacquisition else None,
                reacquired_global_track_id=reacquisition.global_track_id if reacquisition else None,
                target_reacquisition_latency=target_reacquisition_latency,
                identity_preserved=identity_preserved,
                identity_recovery_censored=identity_recovery_censored,
                first_identity_recovery_timestamp=identity_recovery.timestamp if identity_recovery else None,
                identity_recovery_latency=identity_recovery_latency,
                identity_recovery_penalty=identity_recovery_penalty,
                physical_gap_duration=physical_gap_duration,
            )
        )

    return recovery_events


@dataclass(frozen=True)
class RecoveryStatistics:
    num_events_total: int

    num_target_reacquisitions_observed: int
    num_target_reacquisition_censored: int
    target_reacquisition_latency_mean: float
    target_reacquisition_latency_median: float
    target_reacquisition_latency_stdev: float

    num_identity_recovery_censored: int
    num_identity_recovery_eligible: int
    identity_preserving_recovery_rate: float
    identity_recovery_latency_mean: float
    identity_recovery_latency_median: float
    identity_recovery_latency_stdev: float

    identity_preservation_given_reacquisition: float

    identity_recovery_penalty_mean: float
    identity_recovery_penalty_median: float


def _stats(values: list[float]) -> tuple[float, float, float]:

    if not values:
        return float("nan"), float("nan"), float("nan")
    mean = statistics.fmean(values)
    median = statistics.median(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, median, stdev


def compute_recovery_statistics(events: list[RecoveryEvent]) -> RecoveryStatistics:
    
    reacquired = [e for e in events if e.target_reacquired]

    identity_eligible = [e for e in events if not e.identity_recovery_censored]
    identity_recovered = [e for e in identity_eligible if e.identity_preserved]

    identity_rate = len(identity_recovered) / len(identity_eligible) if identity_eligible else float("nan")
    identity_given_reacquisition = len(identity_recovered) / len(reacquired) if reacquired else float("nan")

    reacq_mean, reacq_median, reacq_std = _stats([e.target_reacquisition_latency for e in reacquired])
    id_mean, id_median, id_std = _stats([e.identity_recovery_latency for e in identity_recovered])
    penalty_values = [e.identity_recovery_penalty for e in identity_recovered if e.identity_recovery_penalty is not None]
    penalty_mean, penalty_median, _ = _stats(penalty_values)

    return RecoveryStatistics(
        num_events_total=len(events),
        num_target_reacquisitions_observed=len(reacquired),
        num_target_reacquisition_censored=len(events) - len(reacquired),
        target_reacquisition_latency_mean=reacq_mean,
        target_reacquisition_latency_median=reacq_median,
        target_reacquisition_latency_stdev=reacq_std,
        num_identity_recovery_censored=len(events) - len(identity_eligible),
        num_identity_recovery_eligible=len(identity_eligible),
        identity_preserving_recovery_rate=identity_rate,
        identity_recovery_latency_mean=id_mean,
        identity_recovery_latency_median=id_median,
        identity_recovery_latency_stdev=id_std,
        identity_preservation_given_reacquisition=identity_given_reacquisition,
        identity_recovery_penalty_mean=penalty_mean,
        identity_recovery_penalty_median=penalty_median,
    )
