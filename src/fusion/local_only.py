"""Baseline without multi-station fusion"""

from __future__ import annotations

import numpy as np

from models.fused_measurement import FusedMeasurement
from models.local_tracklet import LocalTracklet

_POSITION_DIMS = slice(0, 3)


def _position_uncertainty(covariance: np.ndarray) -> float:

    return float(np.trace(covariance[_POSITION_DIMS, _POSITION_DIMS]))


class LocalOnlyFusion:

    """Returns the entry with the lowest trace(P_position)"""

    def fuse_states(self, entries: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        if not entries:
            raise ValueError("fuse_states() needs at least one entry")
        best_state, best_covariance = min(entries, key=lambda entry: _position_uncertainty(entry[1]))
        return best_state, best_covariance

    def fuse(self, tracklets: list[LocalTracklet]) -> FusedMeasurement:
        if not tracklets:
            raise ValueError("fuse() needs at least one tracklet")
        best = min(tracklets, key=lambda t: _position_uncertainty(t.covariance))
        return FusedMeasurement(
            state=best.state,
            covariance=best.covariance,
            timestamp=max(t.timestamp for t in tracklets),
            
            contributing_tracklets=[(t.station_id, t.local_track_id) for t in tracklets],
        )
