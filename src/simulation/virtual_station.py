"""Simulates a ground station"""

from __future__ import annotations

from typing import Optional

import numpy as np

from models.local_tracklet import LocalTracklet
from simulation.noise import add_gaussian_noise, diagonal_covariance


class VirtualStation:
    def __init__(self, station_id: str, position_std: float = 2.0, velocity_std: float = 0.5,
        rng: Optional[np.random.Generator] = None) -> None:

        self.station_id = station_id
        self._position_std = position_std
        self._velocity_std = velocity_std
        self._rng = rng if rng is not None else np.random.default_rng()
        self._local_ids: dict[str, str] = {}
        self._next_local_id_number = 1

    def _local_id_for(self, uav_name: str) -> str:

        if uav_name not in self._local_ids:
            self._local_ids[uav_name] = f"{self.station_id[-1].upper()}{self._next_local_id_number:03d}"
            self._next_local_id_number += 1
        return self._local_ids[uav_name]

    def force_new_local_id(self, uav_name: str) -> None:

        self._local_ids.pop(uav_name, None)

    def observe(self, uav_name: str, true_state: np.ndarray, timestamp: float, scenario_id: Optional[str] = None) -> LocalTracklet:
        
        noisy_state = add_gaussian_noise(true_state, self._position_std, self._velocity_std, self._rng)
        covariance = diagonal_covariance(self._position_std, self._velocity_std)
        return LocalTracklet(
            station_id=self.station_id,
            local_track_id=self._local_id_for(uav_name),
            timestamp=timestamp,
            state=noisy_state,
            covariance=covariance,
            scenario_id=scenario_id,
            ground_truth=true_state,
        )
