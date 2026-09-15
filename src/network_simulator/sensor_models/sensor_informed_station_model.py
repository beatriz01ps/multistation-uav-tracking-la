"""The sensor_informed_v1 station model"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from models.local_tracklet import LocalTracklet

PathLike = Union[str, Path]


def _local_frame_from_los(los_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    reference = np.array([0.0, 0.0, 1.0]) if abs(los_unit[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = np.cross(reference, los_unit)
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(los_unit, t1)
    return t1, t2, los_unit


def _angular_direction(station_xyz: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:

    """3D unit vector station -> target"""

    delta = target_xyz - station_xyz
    norm = np.linalg.norm(delta)
    return delta / norm if norm > 0 else delta


def _angle_between_deg(u: np.ndarray, v: np.ndarray) -> float:

    cosine = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _is_within_visibility_sector(station_xyz: np.ndarray, target_xyz: np.ndarray, boresight_direction_xyz: np.ndarray, half_angle_deg: float) -> bool:

    """Purely geometric test"""

    direction = _angular_direction(station_xyz, target_xyz)
    return _angle_between_deg(direction, np.asarray(boresight_direction_xyz)) <= half_angle_deg


@dataclass(frozen=True)
class SensorInformedProfile:

    """Frozen parameters loaded from sensor_informed_v1.json"""

    camera_depth_a: float
    camera_depth_b: float
    camera_transverse_1_a: float
    camera_transverse_1_b: float
    camera_transverse_2_a: float
    camera_transverse_2_b: float
    camera_visibility_half_angle_deg: float
    artifact_path: str
    artifact_version: str

    @classmethod
    def from_artifact(cls, artifact_path: PathLike) -> "SensorInformedProfile":
        data = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        camera = data["camera"]
        return cls(
            camera_depth_a=camera["depth"]["pooled"]["a"],
            camera_depth_b=camera["depth"]["pooled"]["b"],
            camera_transverse_1_a=camera["transverse_1"]["pooled"]["a"],
            camera_transverse_1_b=camera["transverse_1"]["pooled"]["b"],
            camera_transverse_2_a=camera["transverse_2"]["pooled"]["a"],
            camera_transverse_2_b=camera["transverse_2"]["pooled"]["b"],
            camera_visibility_half_angle_deg=data["camera_visibility_model"]["half_angle_deg_pooled_mean"],
            artifact_path=str(artifact_path),
            artifact_version=data["version"],
        )


def camera_inspired_covariance(profile: SensorInformedProfile, station_xyz: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:

    """P_local = diag(sigma_t1^2, sigma_t2^2, sigma_depth^2) in the (t1, t2, los) frame, rotated back to the global frame."""

    delta = target_xyz - station_xyz
    range_m = float(np.linalg.norm(delta))
    los_unit = delta / range_m if range_m > 0 else np.array([1.0, 0.0, 0.0])
    t1, t2, los = _local_frame_from_los(los_unit)
    rotation = np.stack([t1, t2, los], axis=1)

    sigma_t1 = profile.camera_transverse_1_a * range_m**profile.camera_transverse_1_b
    sigma_t2 = profile.camera_transverse_2_a * range_m**profile.camera_transverse_2_b
    sigma_depth = profile.camera_depth_a * range_m**profile.camera_depth_b

    covariance_local = np.diag([sigma_t1**2, sigma_t2**2, sigma_depth**2])
    return rotation @ covariance_local @ rotation.T


class SensorInformedStationModel:

    """station_position_xyz assumes Z=0 (ground station). camera_boresight_xyz is passed explicitly by the caller (StationRunner)"""

    def __init__(self, station_id: str, station_position_xyz: np.ndarray, camera_boresight_xyz: np.ndarray,
        profile: SensorInformedProfile, velocity_std: float, rng: Optional[np.random.Generator] = None) -> None:

        self.station_id = station_id
        self._station_position = np.asarray(station_position_xyz, dtype=float)
        self._boresight = np.asarray(camera_boresight_xyz, dtype=float)
        self._boresight = self._boresight / np.linalg.norm(self._boresight)
        self._profile = profile
        self._velocity_std = velocity_std
        self._rng = rng if rng is not None else np.random.default_rng()

        self._local_ids: dict[str, str] = {}
        self._next_local_id_number = 1
        self._last_visual: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}  # uav_name -> (t, position, covariance)

    def _local_id_for(self, uav_name: str) -> str:

        if uav_name not in self._local_ids:
            self._local_ids[uav_name] = f"{self.station_id[-1].upper()}{self._next_local_id_number:03d}"
            self._next_local_id_number += 1
        return self._local_ids[uav_name]

    def force_new_local_id(self, uav_name: str) -> None:

        self._local_ids.pop(uav_name, None)

    def _is_camera_visible(self, target_xyz: np.ndarray) -> bool:

        return _is_within_visibility_sector(
            self._station_position, target_xyz, self._boresight, self._profile.camera_visibility_half_angle_deg
        )

    def observe(self, uav_name: str, true_state: np.ndarray, timestamp: float, scenario_id: Optional[str] = None) -> LocalTracklet:

        true_position = true_state[:3]
        true_velocity = true_state[3:6]
        noisy_velocity = true_velocity + self._rng.normal(0.0, self._velocity_std, size=3)

        if self._is_camera_visible(true_position):
            covariance_position = camera_inspired_covariance(self._profile, self._station_position, true_position)
            noise = self._rng.multivariate_normal(np.zeros(3), covariance_position)
            noisy_position = true_position + noise
            self._last_visual[uav_name] = (timestamp, noisy_position.copy(), covariance_position.copy())
        else:
            if uav_name in self._last_visual:
                last_t, last_position, last_covariance = self._last_visual[uav_name]
                dt = max(0.0, timestamp - last_t)
                noisy_position = last_position + noisy_velocity * dt
                growth = (self._velocity_std * dt) ** 2
                covariance_position = last_covariance + growth * np.eye(3)
            else:
                return None  
            
        covariance = np.zeros((6, 6))
        covariance[:3, :3] = covariance_position
        covariance[3:6, 3:6] = np.eye(3) * (self._velocity_std**2)

        return LocalTracklet(
            station_id=self.station_id,
            local_track_id=self._local_id_for(uav_name),
            timestamp=timestamp,
            state=np.concatenate([noisy_position, noisy_velocity]),
            covariance=covariance,
            scenario_id=scenario_id,
            ground_truth=true_state,
        )
