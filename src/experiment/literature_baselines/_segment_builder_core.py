"""Shared data structures for the Ma-inspired segment builder: a SegmentTrack is
open or closed (no lifecycle/merger/identity-persistence, no ground truth)"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from evaluation.offline.loaders import TrackHistoryRow
from network_simulator.scenario import ScenarioConfig
from tracking.tracker import TrackTerminalEvent

PathLike = Union[str, Path]


def max_expected_inter_message_gap_seconds(scenario: ScenarioConfig) -> float:

    """Uses the SLOWEST cadence among the scenario's stations."""

    if not scenario.stations:
        raise ValueError("scenario has no stations - cannot derive max_expected_inter_message_gap_seconds")
    return max(1.0 / station.frequency_hz for station in scenario.stations)


@dataclass(frozen=True)
class SegmentTrack:
    segment_id: int                # generated operationally (internal counter). *not related to te others tracks ids*
    birth_time: float
    last_detection_time: float
    state: np.ndarray              # 6D at last_detection_time (post-fusion)
    covariance: np.ndarray         # 6x6 at last_detection_time
    state_history: tuple[tuple[float, np.ndarray], ...]
    covariance_history: tuple[tuple[float, np.ndarray], ...]
    contributing_stations: frozenset
    contributing_local_tracks: frozenset 

@dataclass(frozen=True)
class SegmentTerminalEvent:
    segment_id: int
    birth_time: float
    last_detection_time: float
    close_time: float
    close_reason: str  


@dataclass(frozen=True)
class SegmentHistoryRow:
    segment_id: int
    timestamp: float
    state: np.ndarray
    covariance: np.ndarray
    contributing_stations: frozenset
    contributing_local_tracks: frozenset


@dataclass(frozen=True)
class SegmentBuilderOutput:
    segment_history: tuple[SegmentHistoryRow, ...]
    terminal_events: tuple[SegmentTerminalEvent, ...]


def to_ma_gtm_input(output: SegmentBuilderOutput) -> tuple[list[TrackHistoryRow], dict[int, TrackTerminalEvent]]:
    
    rows = [
        TrackHistoryRow(
            global_track_id=row.segment_id,
            timestamp=row.timestamp,
            state=row.state,
            covariance=row.covariance,
            status="CONFIRMED",  
            prediction_only=False, 
            nis=None,
        )
        for row in output.segment_history
    ]
    terminal_events = {
        event.segment_id: TrackTerminalEvent(
            global_track_id=event.segment_id,
            timestamp=event.close_time,
            event_type="DELETED", 
            merged_into_global_track_id=None,  
        )
        for event in output.terminal_events
    }
    return rows, terminal_events


def write_segment_builder_outputs(output: SegmentBuilderOutput, output_dir: PathLike) -> None:

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "segment_history.jsonl").open("w", encoding="utf-8") as handle:
        for row in output.segment_history:
            handle.write(
                json.dumps(
                    {
                        "segment_id": row.segment_id,
                        "timestamp": row.timestamp,
                        "state": {
                            "x": float(row.state[0]), "y": float(row.state[1]), "z": float(row.state[2]),
                            "vx": float(row.state[3]), "vy": float(row.state[4]), "vz": float(row.state[5]),
                        },
                        "covariance": np.asarray(row.covariance).tolist(),
                        "contributing_stations": sorted(row.contributing_stations),
                        "contributing_local_tracks": sorted(list(row.contributing_local_tracks)),
                        "measurement_supported": True,
                    }
                )
                + "\n"
            )

    with (out / "segment_terminal_events.jsonl").open("w", encoding="utf-8") as handle:
        for event in output.terminal_events:
            handle.write(
                json.dumps(
                    {
                        "segment_id": event.segment_id,
                        "birth_time": event.birth_time,
                        "last_detection_time": event.last_detection_time,
                        "close_time": event.close_time,
                        "close_reason": event.close_reason,
                    }
                )
                + "\n"
            )
