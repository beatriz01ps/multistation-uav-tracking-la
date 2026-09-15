"""Pieces shared by the two replay strategies (event_driven.py, periodic_virtual.py)"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from io_.track_history_logger import TrackHistoryLogger
from models.local_tracklet import LocalTracklet
from tracking.tracker import Tracker, TrackUpdateEvent

PathLike = Union[str, Path]


class ReplayClock:
    """An injectable clock that only moves forward"""

    def __init__(self) -> None:
        self._now = 0.0

    def advance_to(self, timestamp: float) -> None:
        self._now = max(self._now, timestamp)

    def __call__(self) -> float:
        return self._now


@dataclass
class ReplayDiagnostics:
    """Never hide late-drops"""

    total_input_messages: int = 0
    accepted_messages: int = 0
    dropped_late_messages: int = 0
    per_station_total: dict[str, int] = field(default_factory=dict)
    per_station_accepted: dict[str, int] = field(default_factory=dict)
    per_station_dropped: dict[str, int] = field(default_factory=dict)

    @property
    def drop_rate(self) -> float:
        return self.dropped_late_messages / self.total_input_messages if self.total_input_messages else 0.0

    def per_station_drop_rate(self) -> dict[str, float]:
        return {
            station: (self.per_station_dropped.get(station, 0) / total if total else 0.0)
            for station, total in self.per_station_total.items()
        }

    def to_dict(self) -> dict:
        return {
            "total_input_messages": self.total_input_messages,
            "accepted_messages": self.accepted_messages,
            "dropped_late_messages": self.dropped_late_messages,
            "drop_rate": self.drop_rate,
            "per_station_total": self.per_station_total,
            "per_station_accepted": self.per_station_accepted,
            "per_station_dropped": self.per_station_dropped,
            "per_station_drop_rate": self.per_station_drop_rate(),
        }


def load_frozen_tracklet_sequence(sent_messages_path: PathLike) -> list[LocalTracklet]:

    """Reads sent_messages.jsonl from an already-executed real run"""

    tracklets = []
    with Path(sent_messages_path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            tracklets.append(
                LocalTracklet(
                    station_id=record["station_id"],
                    local_track_id=record["local_track_id"],
                    timestamp=record["timestamp"],
                    state=record["state"],
                    covariance=record["covariance"],
                )
            )
    return tracklets


class ReplaySink:
    """Post-cycle accounting shared by both strategies: late-drop diagnostics,
    responses.jsonl (same format as io_/receiver.py), and incremental track_history/terminal_events logging."""

    def __init__(self, tracks_dir: PathLike, *, responses_path: Optional[PathLike] = None) -> None:
        self.diagnostics = ReplayDiagnostics()
        self._logger = TrackHistoryLogger(tracks_dir)
        self._terminal_events_flushed = 0
        self._candidates_flushed = 0
        self._responses_file = Path(responses_path).open("w", encoding="utf-8") if responses_path is not None else None

    def record_ingest(self, tracklet: LocalTracklet, accepted: bool) -> None:
        self.diagnostics.total_input_messages += 1
        self.diagnostics.per_station_total[tracklet.station_id] = (
            self.diagnostics.per_station_total.get(tracklet.station_id, 0) + 1
        )
        if accepted:
            self.diagnostics.accepted_messages += 1
            self.diagnostics.per_station_accepted[tracklet.station_id] = (
                self.diagnostics.per_station_accepted.get(tracklet.station_id, 0) + 1
            )
        else:
            self.diagnostics.dropped_late_messages += 1
            self.diagnostics.per_station_dropped[tracklet.station_id] = (
                self.diagnostics.per_station_dropped.get(tracklet.station_id, 0) + 1
            )

    def flush_tick(self, tracker: Tracker, events: list[TrackUpdateEvent]) -> None:

        """Call after every tracker.tick()"""

        self._write_responses(events)
        self._logger.log_snapshot(tracker.track_manager.active_tracks())
        new_terminal = tracker.terminal_events[self._terminal_events_flushed :]
        if new_terminal:
            self._logger.log_terminal_events(new_terminal)
            self._terminal_events_flushed = len(tracker.terminal_events)
        new_candidates = tracker.association_candidates[self._candidates_flushed :]
        if new_candidates:
            self._logger.log_association_candidates(new_candidates)
            self._candidates_flushed = len(tracker.association_candidates)

    def _write_responses(self, events: list[TrackUpdateEvent]) -> None:

        if self._responses_file is None:
            return
        for event in events:
            self._responses_file.write(
                json.dumps(
                    {
                        "station_id": event.station_id,
                        "local_track_id": event.local_track_id,
                        "global_track_id": event.global_track_id,
                        "timestamp": event.timestamp,
                    }
                )
                + "\n"
            )

    def close(self) -> None:
        
        self._logger.close()
        if self._responses_file is not None:
            self._responses_file.close()
