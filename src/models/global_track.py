"""GlobalTrack data model"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from models.enums import TrackStatus, UpdateKind
from models.validation import coerce_covariance_matrix, coerce_state_vector


class GlobalTrack(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    global_track_id: int

    state: np.ndarray
    covariance: np.ndarray

    last_innovation: Optional[np.ndarray] = None
    last_innovation_covariance: Optional[np.ndarray] = None

    last_prediction_timestamp: float
    last_measurement_timestamp: float
    created_at: float

    status: TrackStatus = TrackStatus.TENTATIVE
    last_update_kind: UpdateKind = UpdateKind.PREDICTION_ONLY

    age: int = 0
    hit_count: int = 0
    miss_count: int = 0

    # station_id -> most recently known local_track_id
    associated_local_tracks: dict[str, str] = Field(default_factory=dict)

    associated_local_track_timestamps: dict[str, float] = Field(default_factory=dict)

    current_contributors: list[tuple[str, str]] = Field(default_factory=list)

    local_track_history: set[tuple[str, str]] = Field(default_factory=set)

    @field_validator("state", mode="before")
    @classmethod
    def _validate_state(cls, v):
        return coerce_state_vector(v)

    @field_validator("covariance", mode="before")
    @classmethod
    def _validate_covariance(cls, v):

        return coerce_covariance_matrix(v)

    @property
    def position(self) -> np.ndarray:

        return self.state[:3]

    @property
    def velocity(self) -> np.ndarray:

        return self.state[3:6]

    def time_since_measurement(self, timestamp: float) -> float:
        
        return timestamp - self.last_measurement_timestamp
