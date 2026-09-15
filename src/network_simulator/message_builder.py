"""Serializes a LocalTracklet into the same JSON contract io_/parser.py accepts."""

from __future__ import annotations

from typing import Any

from models.local_tracklet import LocalTracklet


def build_tracklet_message(tracklet: LocalTracklet) -> dict[str, Any]:
    
    return {
        "station_id": tracklet.station_id,
        "local_track_id": tracklet.local_track_id,
        "timestamp": tracklet.timestamp,
        "state": tracklet.state.tolist(),
        "covariance": tracklet.covariance.tolist(),
    }
