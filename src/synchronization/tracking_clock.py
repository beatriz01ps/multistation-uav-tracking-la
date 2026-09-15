"""Separates the measurement_timestamp domain from the local wait/monotonic clock"""

from __future__ import annotations

import time as time_module
from typing import Optional


class TrackingClock:
    def __init__(self, time_scale: float = 1.0, monotonic=time_module.monotonic) -> None:

        self._time_scale = time_scale
        self._monotonic = monotonic
        self._measurement_anchor: Optional[float] = None
        self._monotonic_anchor: Optional[float] = None

    def anchor(self, measurement_timestamp: float) -> None:

        self._measurement_anchor = measurement_timestamp
        self._monotonic_anchor = self._monotonic()

    @property
    def is_anchored(self) -> bool:
        return self._measurement_anchor is not None

    def now(self) -> float:

        """"Now" in the measurement_timestamp domain, extrapolated from the last real anchor. Only call if is_anchored."""

        if self._measurement_anchor is None or self._monotonic_anchor is None:
            raise RuntimeError("TrackingClock.now() called before any anchor() (no real measurement yet)")
        elapsed_monotonic = self._monotonic() - self._monotonic_anchor
        return self._measurement_anchor + elapsed_monotonic * self._time_scale
