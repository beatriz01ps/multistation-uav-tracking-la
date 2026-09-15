"""Orchestrates the evaluation/offline/ loaders + metrics for one real run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from evaluation.association_metrics import (
    association_accuracy,
    id_switches,
    local_track_truth_map,
    majority_label_per_global_id,
    validate_association_event_truth_coverage,
)
from evaluation.offline.consistency import average_nees, nees_series, nis_series
from evaluation.offline.identity import excess_global_tracks, fragmentation_events, track_availability
from evaluation.offline.loaders import (
    load_association_events,
    load_ground_truth,
    load_terminal_events,
    load_tracklet_origins,
    load_track_history,
)
from evaluation.offline.recovery import RecoveryEvent, RecoveryStatistics, compute_recovery_events, compute_recovery_statistics
from evaluation.offline.rmse import RmseReport, compute_rmse

PathLike = Union[str, Path]


@dataclass(frozen=True)
class OfflineEvaluationReport:
    num_real_targets: int
    num_global_tracks_created: int
    association_accuracy: float
    id_switches: int
    excess_global_tracks: dict[str, int]
    fragmentation_events: dict[str, int]
    track_availability: dict[str, dict[str, float]]
    rmse: RmseReport
    mean_nees_6d: float
    mean_nis: float
    num_nis_samples: int
    recovery_events: list[RecoveryEvent]
    recovery_statistics: RecoveryStatistics


def evaluate_run(sim_dir: PathLike, tracks_dir: PathLike) -> OfflineEvaluationReport:
    sim_dir = Path(sim_dir)
    tracks_dir = Path(tracks_dir)

    ground_truth = load_ground_truth(sim_dir / "ground_truth.jsonl")
    truth_labels = load_tracklet_origins(sim_dir / "tracklet_origins.jsonl")

    local_track_truth = local_track_truth_map(truth_labels)
    events = load_association_events(sim_dir / "responses.jsonl")
    rows = load_track_history(tracks_dir / "track_history.jsonl")
    terminal_events = load_terminal_events(tracks_dir / "track_terminal_events.jsonl")

    validate_association_event_truth_coverage(events, local_track_truth, context=str(sim_dir))

    majority_labels = majority_label_per_global_id(events, local_track_truth)

    rmse_report = compute_rmse(rows, ground_truth, majority_labels)
    nees_values = nees_series(rows, ground_truth, majority_labels)
    nis_values = nis_series(rows)
    recovery_events = compute_recovery_events(
        rows, events, truth_labels, local_track_truth, majority_labels, terminal_events
    )

    return OfflineEvaluationReport(
        num_real_targets=len(ground_truth),
        num_global_tracks_created=len({row.global_track_id for row in rows}),
        association_accuracy=association_accuracy(events, local_track_truth, majority_labels),
        id_switches=id_switches(events, local_track_truth),
        excess_global_tracks=excess_global_tracks(majority_labels),
        fragmentation_events=fragmentation_events(rows, majority_labels),
        track_availability=track_availability(rows, majority_labels),
        rmse=rmse_report,
        mean_nees_6d=average_nees(nees_values),
        mean_nis=average_nees(nis_values),  # same formula, a generic name
        num_nis_samples=len(nis_values),
        recovery_events=recovery_events,
        recovery_statistics=compute_recovery_statistics(recovery_events),
    )
