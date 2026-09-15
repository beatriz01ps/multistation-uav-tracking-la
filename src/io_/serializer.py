"""Serializes system output"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from models.enums import UpdateKind
from models.global_track import GlobalTrack
from models.validation import regularize_for_inversion


def _nis(innovation: Optional[np.ndarray], innovation_covariance: Optional[np.ndarray]) -> Optional[float]:

    """Normalized Innovation Squared"""

    if innovation is None or innovation_covariance is None:
        return None
    precision = np.linalg.inv(regularize_for_inversion(innovation_covariance))
    return float(innovation @ precision @ innovation)


def serialize_global_track(track: GlobalTrack, timestamp: float) -> dict[str, Any]:

    x, y, z, vx, vy, vz = track.state.tolist()
    return {
        "global_track_id": track.global_track_id,
        "timestamp": timestamp,
        "state": {"x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz},
        "covariance": track.covariance.tolist(),
        "status": track.status.value.upper(),
        "prediction_only": track.last_update_kind is UpdateKind.PREDICTION_ONLY,
        "hit_count": track.hit_count,
        "miss_count": track.miss_count,
        "source_tracks": [
            {"station_id": station_id, "local_track_id": local_id}
            for station_id, local_id in track.current_contributors
        ],
        "known_local_tracks": [
            {"station_id": station_id, "local_track_id": local_id}
            for station_id, local_id in track.associated_local_tracks.items()
        ],
        "innovation": track.last_innovation.tolist() if track.last_innovation is not None else None,
        "innovation_covariance": (
            track.last_innovation_covariance.tolist() if track.last_innovation_covariance is not None else None
        ),
        "nis": _nis(track.last_innovation, track.last_innovation_covariance),
    }


def serialize_all_tracks(tracks: list[GlobalTrack], timestamp: float) -> list[dict[str, Any]]:
    
    return [serialize_global_track(track, timestamp) for track in tracks]
