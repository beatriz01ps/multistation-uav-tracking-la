"""Contract: the local tracklet produced by a station. state+covariance are
required; ground_truth exists only for offline evaluation"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from models.validation import coerce_covariance_matrix, coerce_optional_state_vector, coerce_state_vector


class LocalTracklet(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    station_id: str
    local_track_id: str
    timestamp: float

    state: np.ndarray  # [x, y, z, vx, vy, vz]
    covariance: np.ndarray  # 6x6 same order as state

    scenario_id: Optional[str] = None
    sensor_mode: Optional[str] = None

    pos_uncertainty: Optional[float] = None
    vel_uncertainty: Optional[float] = None

    ground_truth: Optional[np.ndarray] = None # EVALUATION ONLY 

    @field_validator("state", mode="before")
    @classmethod
    def _validate_state(cls, v):

        return coerce_state_vector(v)

    @field_validator("covariance", mode="before")
    @classmethod
    def _validate_covariance(cls, v):

        return coerce_covariance_matrix(v)

    @field_validator("ground_truth", mode="before")
    @classmethod
    def _validate_ground_truth(cls, v):

        return coerce_optional_state_vector(v)

    @property
    def position(self) -> np.ndarray:

        return self.state[:3]

    @property
    def velocity(self) -> np.ndarray:
        
        return self.state[3:6]
