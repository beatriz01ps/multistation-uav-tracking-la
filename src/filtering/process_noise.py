"""Process noise for Constant Velocity"""

from __future__ import annotations

import numpy as np


def constant_velocity_process_noise(dt: float, acceleration_std: float) -> np.ndarray:

    variance = acceleration_std**2
    block = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * variance
    q = np.zeros((6, 6))
    for axis in range(3):
        idx = [axis, axis + 3]
        q[np.ix_(idx, idx)] = block
    return q
