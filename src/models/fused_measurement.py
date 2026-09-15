"""Result of fusing N local tracklets associated with the same GlobalTrack in the same cycle"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from models.validation import coerce_covariance_matrix, coerce_state_vector


class FusedMeasurement(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: np.ndarray
    covariance: np.ndarray
    timestamp: float
    contributing_tracklets: list[tuple[str, str]]  # (station_id, local_track_id)

    @field_validator("state", mode="before")
    @classmethod
    def _validate_state(cls, v):
        return coerce_state_vector(v)

    @field_validator("covariance", mode="before")
    @classmethod
    def _validate_covariance(cls, v):
        return coerce_covariance_matrix(v)
