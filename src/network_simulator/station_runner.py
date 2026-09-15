"""Each StationRunner runs independently at its own frequency_hz, with no sync between stations"""

from __future__ import annotations

import logging
import threading
import time as time_module

import numpy as np

from network_simulator.message_builder import build_tracklet_message
from network_simulator.scenario import ScenarioConfig, StationSpec
from network_simulator.scenario_logger import ScenarioLogger
from network_simulator.trajectories import true_state_for_uav
from network_simulator.udp_sender import UdpSender
from network_simulator.visibility import line_of_sight_blocked
from simulation.virtual_station import VirtualStation

logger = logging.getLogger(__name__)


def _build_station_model(station_spec: StationSpec, scenario: ScenarioConfig, rng):

    """Dispatches on scenario.station_error_model.type"""

    model_type = scenario.station_error_model.type
    if model_type == "legacy":
        return VirtualStation(
            station_id=station_spec.station_id,
            position_std=station_spec.position_std,
            velocity_std=station_spec.velocity_std,
            rng=rng,
        )
    if model_type == "sensor_informed_v1":

        from network_simulator.sensor_models.sensor_informed_station_model import (
            SensorInformedProfile,
            SensorInformedStationModel,
        )

        profile_path = scenario.station_error_model.calibration_profile
        if not profile_path:
            raise ValueError(
                "station_error_model.type=sensor_informed_v1 requires "
                "station_error_model.calibration_profile (path to the frozen artifact)"
            )
        profile = SensorInformedProfile.from_artifact(profile_path)

        station_xyz = np.array([*station_spec.position, 0.0])

        initial_positions = np.array([uav.initial_state[:3] for uav in scenario.uavs])
        centroid = initial_positions.mean(axis=0)
        boresight = centroid - station_xyz
        if np.linalg.norm(boresight) == 0:
            boresight = np.array([1.0, 0.0, 0.0])

        return SensorInformedStationModel(
            station_id=station_spec.station_id,
            station_position_xyz=station_xyz,
            camera_boresight_xyz=boresight,
            profile=profile,
            velocity_std=station_spec.velocity_std,
            rng=rng,
        )
    raise ValueError(f"unknown station_error_model.type: {model_type!r}")


class StationRunner:
    def __init__(self, station_spec: StationSpec, scenario: ScenarioConfig, sender: UdpSender,
        scenario_logger: ScenarioLogger, rng, realtime: bool = True) -> None:

        self._spec = station_spec
        self._scenario = scenario
        self._sender = sender
        self._scenario_logger = scenario_logger
        self._realtime = realtime
        self._station = _build_station_model(station_spec, scenario, rng)
        self._already_switched: set = set()

    def _is_visible(self, uav, t: float, target_xy) -> bool:

        if any(window.covers(self._spec.station_id, t) for window in uav.dropout_windows):
            return False
        return not any(
            line_of_sight_blocked(self._spec.position, target_xy, obstacle.center, obstacle.radius)
            for obstacle in self._scenario.obstacles
        )

    def _apply_pending_id_switches(self, uav, t: float) -> None:

        for event in uav.id_switch_events:
            if event.station_id != self._spec.station_id:
                continue
            marker = (uav.name, event.at_s)
            if t >= event.at_s and marker not in self._already_switched:
                self._station.force_new_local_id(uav.name)
                self._already_switched.add(marker)

    def run(self, stop_event: threading.Event) -> None:

        period = 1.0 / self._spec.frequency_hz
        start_monotonic = time_module.monotonic()
        tick = 0

        while not stop_event.is_set():
            t = tick * period
            if t > self._scenario.duration_s:
                break

            for uav in self._scenario.uavs:
                self._apply_pending_id_switches(uav, t)

                true_state = true_state_for_uav(uav, t)
                if not self._is_visible(uav, t, target_xy=true_state[:2]):
                    continue

                tracklet = self._station.observe(uav.name, true_state, t, scenario_id=self._scenario.scenario_id)
                if tracklet is None:
                    continue
                message = build_tracklet_message(tracklet)

                self._sender.send(message)
                self._scenario_logger.log_sent(message)
                self._scenario_logger.log_tracklet_origin(tracklet.station_id, tracklet.local_track_id, t, uav.name)
                logger.info(
                    "SENT station=%s local=%s t=%.3f", tracklet.station_id, tracklet.local_track_id, t
                )

            tick += 1
            if self._realtime:
                target = start_monotonic + tick * period
                remaining = target - time_module.monotonic()
                if remaining > 0:
                    stop_event.wait(remaining)

        logger.info("station %s finished", self._spec.station_id)
