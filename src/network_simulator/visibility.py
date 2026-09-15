"""Geometric line-of-sight occlusion"""

from __future__ import annotations

import numpy as np


def line_of_sight_blocked(observer_xy, target_xy, obstacle_center_xy, obstacle_radius: float) -> bool:

    observer = np.asarray(observer_xy, dtype=float)
    target = np.asarray(target_xy, dtype=float)
    center = np.asarray(obstacle_center_xy, dtype=float)

    segment = target - observer
    segment_length_sq = float(segment @ segment)

    if segment_length_sq == 0.0:
        closest_point = observer
    else:
        t = float((center - observer) @ segment / segment_length_sq)
        t_clamped = max(0.0, min(1.0, t))
        closest_point = observer + t_clamped * segment

    distance = float(np.linalg.norm(closest_point - center))
    return distance <= obstacle_radius
