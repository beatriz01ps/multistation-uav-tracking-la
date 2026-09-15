"""On-disk record in four files: sent_messages.jsonl (exactly what the tracker
received), ground_truth.jsonl (true state, offline-only), responses.jsonl (tracker's local -> global equivalence),
and tracklet_origins.jsonl (which target generated each message"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Union


class ScenarioLogger:
    def __init__(self, out_dir: Union[str, Path]) -> None:

        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._sent = (self._out_dir / "sent_messages.jsonl").open("w", encoding="utf-8")
        self._truth = (self._out_dir / "ground_truth.jsonl").open("w", encoding="utf-8")
        self._responses = (self._out_dir / "responses.jsonl").open("w", encoding="utf-8")
        self._origins = (self._out_dir / "tracklet_origins.jsonl").open("w", encoding="utf-8")

    def log_sent(self, message: dict[str, Any]) -> None:

        self._write(self._sent, message)

    def log_truth(self, true_target_id: str, timestamp: float, true_state) -> None:

        self._write(
            self._truth,
            {"timestamp": timestamp, "true_target_id": true_target_id, "true_state": list(map(float, true_state))},
        )

    def log_response(self, response: dict[str, Any]) -> None:

        self._write(self._responses, response)

    def log_tracklet_origin(self, station_id: str, local_track_id: str, timestamp: float, true_target_id: str) -> None:

        self._write(
            self._origins,
            {
                "station_id": station_id,
                "local_track_id": local_track_id,
                "timestamp": timestamp,
                "true_target_id": true_target_id,
            },
        )

    def _write(self, handle, record: dict[str, Any]) -> None:

        with self._lock:
            handle.write(json.dumps(record) + "\n")
            handle.flush()

    def close(self) -> None:
        
        with self._lock:
            self._sent.close()
            self._truth.close()
            self._responses.close()
            self._origins.close()
