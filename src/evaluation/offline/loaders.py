"""Parsing of the artifacts of a real run: converts JSONL/CSV into simple Python structures, computing no metric here."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from evaluation.association_metrics import AssociationEvent, TruthLabels
from tracking.tracker import AssociationCandidateEvent, TrackTerminalEvent

PathLike = Union[str, Path]


class GroundTruthTrajectory:

    """The true state (6D) of one target over time, linearly interpolated between known samples."""

    def __init__(self, samples: list[tuple[float, np.ndarray]]) -> None:
        self._samples = sorted(samples, key=lambda item: item[0])

    def at(self, t: float) -> np.ndarray | None:
        samples = self._samples
        if not samples:
            return None
        if t <= samples[0][0]:
            return samples[0][1]
        if t >= samples[-1][0]:
            return samples[-1][1]
        for (t0, s0), (t1, s1) in zip(samples, samples[1:]):
            if t0 <= t <= t1:
                if t1 == t0:
                    return s0
                frac = (t - t0) / (t1 - t0)
                return s0 + frac * (s1 - s0)
        return None  # unreachable given the saturation above


@dataclass
class TrackHistoryRow:
    global_track_id: int
    timestamp: float
    state: np.ndarray  # 6D: x,y,z,vx,vy,vz
    covariance: np.ndarray  # 6x6
    status: str
    prediction_only: bool
    nis: float | None


def load_ground_truth(path: PathLike) -> dict[str, GroundTruthTrajectory]:

    """true_target_id -> trajectory (see GroundTruthTrajectory)."""

    by_target: dict[str, list[tuple[float, np.ndarray]]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            state = np.array(record["true_state"], dtype=float)
            by_target.setdefault(record["true_target_id"], []).append((record["timestamp"], state))
    return {target_id: GroundTruthTrajectory(samples) for target_id, samples in by_target.items()}


def load_tracklet_origins(path: PathLike) -> TruthLabels:

    labels: TruthLabels = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            key = (record["station_id"], record["local_track_id"], record["timestamp"])
            labels[key] = record["true_target_id"]
    return labels


def load_association_events(path: PathLike) -> list[AssociationEvent]:

    events = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            events.append(
                AssociationEvent(
                    station_id=record["station_id"],
                    local_track_id=record["local_track_id"],
                    global_track_id=record["global_track_id"],
                    timestamp=record["timestamp"],
                )
            )
    return events


def load_track_history(path: PathLike) -> list[TrackHistoryRow]:

    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            state = record["state"]
            rows.append(
                TrackHistoryRow(
                    global_track_id=record["global_track_id"],
                    timestamp=record["timestamp"],
                    state=np.array([state["x"], state["y"], state["z"], state["vx"], state["vy"], state["vz"]]),
                    covariance=np.array(record["covariance"]),
                    status=record["status"],
                    prediction_only=record["prediction_only"],
                    nis=record["nis"],
                )
            )
    return rows


def load_terminal_events(path: PathLike) -> dict[int, TrackTerminalEvent]:

    events: dict[int, TrackTerminalEvent] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            events[record["global_track_id"]] = TrackTerminalEvent(
                global_track_id=record["global_track_id"],
                timestamp=record["timestamp"],
                event_type=record["event_type"],
                merged_into_global_track_id=record["merged_into_global_track_id"],
            )
    return events


def load_association_candidates(path: PathLike) -> list[AssociationCandidateEvent]:

    events = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            events.append(
                AssociationCandidateEvent(
                    global_track_id=record["global_track_id"],
                    station_id=record["station_id"],
                    local_track_id=record["local_track_id"],
                    timestamp=record["timestamp"],
                    predicted_position=tuple(record["predicted_position"]),
                    tracklet_position=tuple(record["tracklet_position"]),
                    euclidean_distance_m=record["euclidean_distance_m"],
                )
            )
    return events


def load_track_history_csv(path: PathLike) -> list[dict[str, str]]:
    
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
