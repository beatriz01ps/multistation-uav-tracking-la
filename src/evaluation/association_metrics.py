"""Association metrics from the tracker's associations vs the simulator's truth_labels"""

from __future__ import annotations

from collections import Counter
from typing import NamedTuple

TruthLabels = dict[tuple[str, str, float], str]
# Ground truth collapsed by stable identity (station_id, local_track_id)
# -> true_target_id - see `local_track_truth_map`.
LocalTrackTruth = dict[tuple[str, str], str]


class AssociationEvent(NamedTuple):
    station_id: str
    local_track_id: str
    global_track_id: int
    timestamp: float


def local_track_truth_map(truth_labels: TruthLabels) -> LocalTrackTruth:

    """Collapses truth_labels to (station_id, local_track_id) -> true_target_id.
    Raises if the non-reuse contract is violated (same key, two targets)."""

    result: LocalTrackTruth = {}
    for (station_id, local_track_id, _timestamp), true_target_id in truth_labels.items():
        key = (station_id, local_track_id)
        existing = result.get(key)
        if existing is not None and existing != true_target_id:
            raise ValueError(
                f"local_track_id CONTRACT VIOLATED: (station_id, local_track_id)={key} "
                f"mapped to different true_target_id values: {existing!r} and {true_target_id!r} "
                "- the result cannot be used without investigating the source of the violation."
            )
        result[key] = true_target_id
    return result


def validate_association_event_truth_coverage(events: list[AssociationEvent], local_track_truth: LocalTrackTruth, *, context: str = "") -> None:

    """Fail-fast if any event has no truth mapping - should never happen with valid data.
    Called once before any metric; downstream None-checks are just a safeguard."""

    missing = [
        event for event in events if (event.station_id, event.local_track_id) not in local_track_truth
    ]
    if missing:
        first = missing[0]
        raise ValueError(
            f"TRUTH COVERAGE FAILED{f' ({context})' if context else ''}: "
            f"{len(missing)} of {len(events)} AssociationEvent(s) with no ground-truth mapping "
            f"for (station_id, local_track_id). First case: station_id={first.station_id!r}, "
            f"local_track_id={first.local_track_id!r}, timestamp={first.timestamp}, "
            f"global_track_id={first.global_track_id}. This should NEVER happen in a controlled "
            "synthetic experiment - investigate tracklet_origins.jsonl before trusting this run."
        )


def majority_label_per_global_id(events: list[AssociationEvent], local_track_truth: LocalTrackTruth) -> dict[int, str]:

    votes: dict[int, Counter] = {}
    for event in events:
        true_label = local_track_truth.get((event.station_id, event.local_track_id))
        if true_label is None:
            continue
        votes.setdefault(event.global_track_id, Counter())[true_label] += 1
    return {gid: counter.most_common(1)[0][0] for gid, counter in votes.items()}


def association_accuracy(events: list[AssociationEvent], local_track_truth: LocalTrackTruth, majority_labels: dict[int, str]) -> float:

    total = correct = 0
    for event in events:
        true_label = local_track_truth.get((event.station_id, event.local_track_id))
        if true_label is None:
            continue
        total += 1
        if majority_labels.get(event.global_track_id) == true_label:
            correct += 1
    return correct / total if total else float("nan")


def track_fragmentation(majority_labels: dict[int, str]) -> dict[str, int]:

    """Distinct global_track_id values each real UAV received, in total."""

    counts: dict[str, int] = {}
    for uav in majority_labels.values():
        counts[uav] = counts.get(uav, 0) + 1
    return counts


def id_switches(events: list[AssociationEvent], local_track_truth: LocalTrackTruth) -> int:

    """How many times each real UAV's global_track_id changed between consecutive observations, in chronological order."""

    by_uav: dict[str, list[tuple[float, int]]] = {}
    for event in events:
        true_label = local_track_truth.get((event.station_id, event.local_track_id))
        if true_label is None:
            continue
        by_uav.setdefault(true_label, []).append((event.timestamp, event.global_track_id))

    switches = 0
    for observations in by_uav.values():
        observations.sort(key=lambda item: item[0])
        for (_, previous_gid), (_, current_gid) in zip(observations, observations[1:]):
            if current_gid != previous_gid:
                switches += 1
    return switches
