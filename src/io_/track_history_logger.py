"""Continuous history of GlobalTracks."""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from typing import Any, Union

from io_.serializer import serialize_global_track
from models.global_track import GlobalTrack
from tracking.tracker import AssociationCandidateEvent, TrackTerminalEvent

_CSV_FIELDS = [
    "global_track_id",
    "timestamp",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "status",
    "prediction_only",
    "hit_count",
    "miss_count",
    "source_tracks",
    "known_local_tracks",
    "nis",
]


class TrackHistoryLogger:
    def __init__(self, out_dir: Union[str, Path]) -> None:

        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        self._csv_file = (self._out_dir / "track_history.csv").open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=_CSV_FIELDS)
        self._csv_writer.writeheader()

        self._jsonl_file = (self._out_dir / "track_history.jsonl").open("w", encoding="utf-8")
        self._terminal_events_file = (self._out_dir / "track_terminal_events.jsonl").open("w", encoding="utf-8")
        self._association_candidates_file = (
            self._out_dir / "association_candidates.jsonl"
        ).open("w", encoding="utf-8")

    def log_snapshot(self, tracks: list[GlobalTrack]) -> None:

        """Writes one row per active track, using each track's own last_prediction_timestamp"""

        with self._lock:
            for track in tracks:
                record = serialize_global_track(track, timestamp=track.last_prediction_timestamp)
                self._jsonl_file.write(json.dumps(record) + "\n")
                self._csv_writer.writerow(self._to_csv_row(record))
            self._jsonl_file.flush()
            self._csv_file.flush()

    def log_terminal_events(self, events: list[TrackTerminalEvent]) -> None:

        """Called incrementally (only new events since the last call)"""

        if not events:
            return
        with self._lock:
            for event in events:
                self._terminal_events_file.write(
                    json.dumps(
                        {
                            "global_track_id": event.global_track_id,
                            "timestamp": event.timestamp,
                            "event_type": event.event_type,
                            "merged_into_global_track_id": event.merged_into_global_track_id,
                        }
                    )
                    + "\n"
                )
            self._terminal_events_file.flush()

    def log_association_candidates(self, events: list[AssociationCandidateEvent]) -> None:

        """Called incrementally (only new events), same pattern as log_terminal_events."""

        if not events:
            return
        with self._lock:
            for event in events:
                self._association_candidates_file.write(
                    json.dumps(
                        {
                            "global_track_id": event.global_track_id,
                            "station_id": event.station_id,
                            "local_track_id": event.local_track_id,
                            "timestamp": event.timestamp,
                            "predicted_position": list(event.predicted_position),
                            "tracklet_position": list(event.tracklet_position),
                            "euclidean_distance_m": event.euclidean_distance_m,
                        }
                    )
                    + "\n"
                )
            self._association_candidates_file.flush()

    @staticmethod
    def _to_csv_row(record: dict[str, Any]) -> dict[str, Any]:

        return {
            "global_track_id": record["global_track_id"],
            "timestamp": record["timestamp"],
            "x": record["state"]["x"],
            "y": record["state"]["y"],
            "z": record["state"]["z"],
            "vx": record["state"]["vx"],
            "vy": record["state"]["vy"],
            "vz": record["state"]["vz"],
            "status": record["status"],
            "prediction_only": record["prediction_only"],
            "hit_count": record["hit_count"],
            "miss_count": record["miss_count"],
            "source_tracks": _join_local_ids(record["source_tracks"]),
            "known_local_tracks": _join_local_ids(record["known_local_tracks"]),
            "nis": record["nis"] if record["nis"] is not None else "",
        }

    def close(self) -> None:
        
        with self._lock:
            self._csv_file.close()
            self._jsonl_file.close()
            self._terminal_events_file.close()
            self._association_candidates_file.close()


def _join_local_ids(entries: list[dict[str, str]]) -> str:
    return "|".join(f"{entry['station_id']}:{entry['local_track_id']}" for entry in entries)
