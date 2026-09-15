"""Identity metrics over real runs, reusing evaluation.association_metrics directly"""

from __future__ import annotations

from collections import Counter

from evaluation.association_metrics import (
    AssociationEvent,
    LocalTrackTruth,
    TruthLabels,
    association_accuracy,
    id_switches,
    local_track_truth_map,
    majority_label_per_global_id,
    track_fragmentation,
)
from evaluation.offline.loaders import TrackHistoryRow

__all__ = [
    "AssociationEvent",
    "TruthLabels",
    "LocalTrackTruth",
    "local_track_truth_map",
    "association_accuracy",
    "id_switches",
    "majority_label_per_global_id",
    "excess_global_tracks",
    "fragmentation_events",
    "track_availability",
    "epoch_level_identity_switches",
]


def excess_global_tracks(majority_labels: dict[int, str]) -> dict[str, int]:

    return track_fragmentation(majority_labels)


def _count_matched_unmatched_matched_transitions(sequence: list[bool]) -> int:

    transitions = 0
    was_matched = False
    in_gap = False
    for matched in sequence:
        if matched:
            if in_gap and was_matched:
                transitions += 1
            was_matched = True
            in_gap = False
        elif was_matched:
            in_gap = True
    return transitions


def fragmentation_events(rows: list[TrackHistoryRow], majority_labels: dict[int, str]) -> dict[str, int]:

    all_epochs = sorted({row.timestamp for row in rows})
    active_global_ids_by_epoch: dict[float, set[int]] = {}
    for row in rows:
        active_global_ids_by_epoch.setdefault(row.timestamp, set()).add(row.global_track_id)

    targets_by_global_id: dict[int, str] = majority_labels
    global_ids_by_target: dict[str, set[int]] = {}
    for global_track_id, target in targets_by_global_id.items():
        global_ids_by_target.setdefault(target, set()).add(global_track_id)

    result: dict[str, int] = {}
    for target, target_global_ids in global_ids_by_target.items():
        sequence = [
            bool(active_global_ids_by_epoch.get(epoch, set()) & target_global_ids) for epoch in all_epochs
        ]
        result[target] = _count_matched_unmatched_matched_transitions(sequence)
    return result


def track_availability(rows: list[TrackHistoryRow], majority_labels: dict[int, str]) -> dict[str, dict[str, float]]:

    """Fraction of evaluation epochs (track_history timestamps, never AssociationEvent's) at which each real target has a valid GlobalTrack"""

    all_epochs = sorted({row.timestamp for row in rows})
    if not all_epochs:
        return {}

    any_active_by_epoch: dict[float, set[int]] = {}
    confirmed_by_epoch: dict[float, set[int]] = {}
    for row in rows:
        any_active_by_epoch.setdefault(row.timestamp, set()).add(row.global_track_id)
        if row.status == "CONFIRMED":
            confirmed_by_epoch.setdefault(row.timestamp, set()).add(row.global_track_id)

    global_ids_by_target: dict[str, set[int]] = {}
    for global_track_id, target in majority_labels.items():
        global_ids_by_target.setdefault(target, set()).add(global_track_id)

    num_epochs = len(all_epochs)
    result: dict[str, dict[str, float]] = {}
    for target, target_global_ids in global_ids_by_target.items():
        any_hits = sum(1 for epoch in all_epochs if any_active_by_epoch.get(epoch, set()) & target_global_ids)
        confirmed_hits = sum(
            1 for epoch in all_epochs if confirmed_by_epoch.get(epoch, set()) & target_global_ids
        )
        result[target] = {
            "availability_any_active": any_hits / num_epochs,
            "availability_confirmed": confirmed_hits / num_epochs,
        }
    return result


def epoch_level_identity_switches(events: list[AssociationEvent], local_track_truth: LocalTrackTruth) -> dict:

    by_uav_epoch: dict[str, dict[float, Counter]] = {}
    for event in events:
        true_label = local_track_truth.get((event.station_id, event.local_track_id))
        if true_label is None:
            continue
        by_uav_epoch.setdefault(true_label, {}).setdefault(event.timestamp, Counter())[event.global_track_id] += 1

    total_epochs = 0
    disagreement_epochs = 0
    total_switches = 0
    per_uav_switches: dict[str, int] = {}

    for true_label, epochs_by_ts in by_uav_epoch.items():
        representative_sequence = []
        for timestamp in sorted(epochs_by_ts):
            votes = epochs_by_ts[timestamp]
            total_epochs += 1
            if len(votes) > 1:
                disagreement_epochs += 1
            max_votes = max(votes.values())
            tied = sorted(gid for gid, count in votes.items() if count == max_votes)
            representative_sequence.append(tied[0])

        switches = sum(
            1 for prev, curr in zip(representative_sequence, representative_sequence[1:]) if prev != curr
        )
        per_uav_switches[true_label] = switches
        total_switches += switches

    return {
        "epoch_level_identity_switches_total": total_switches,
        "epoch_level_identity_switches_per_uav": per_uav_switches,
        "has_epoch_level_identity_switch": total_switches > 0,
        "total_epochs": total_epochs,
        "same_timestamp_disagreement_epochs": disagreement_epochs,
        "same_timestamp_disagreement_occurred": disagreement_epochs > 0,
    }
