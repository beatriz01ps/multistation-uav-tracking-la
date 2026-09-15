"""Common interface between UkfTracker and EkfTracker """

from __future__ import annotations

from typing import Protocol

import numpy as np


class FilterTracker(Protocol):
    def predict(self, dt: float) -> None: ...

    def update(self, measurement: np.ndarray, measurement_covariance: np.ndarray) -> None: ...

    def set_state(self, state: np.ndarray, covariance: np.ndarray) -> None: ...

    @property
    def state(self) -> np.ndarray: ...

    @property
    def covariance(self) -> np.ndarray: ...

    @property
    def last_innovation(self) -> np.ndarray | None:

        """Innovation (y = measurement - prediction) from the LAST real update() """

        ...

    @property
    def last_innovation_covariance(self) -> np.ndarray | None:

        """Innovation covariance (S) from the LAST real update()"""
        
        ...
