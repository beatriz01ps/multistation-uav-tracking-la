"""Ground truth logging"""

from __future__ import annotations

import logging
import threading
import time as time_module

from network_simulator.scenario import ScenarioConfig
from network_simulator.scenario_logger import ScenarioLogger
from network_simulator.trajectories import true_state_for_uav

logger = logging.getLogger(__name__)


class TruthRunner:
    def __init__(self, scenario: ScenarioConfig, scenario_logger: ScenarioLogger, realtime: bool = True) -> None:

        self._scenario = scenario
        self._scenario_logger = scenario_logger
        self._realtime = realtime
        self._frequency_hz = max(station.frequency_hz for station in scenario.stations)

    def run(self, stop_event: threading.Event) -> None:
        
        period = 1.0 / self._frequency_hz
        start_monotonic = time_module.monotonic()
        tick = 0

        while not stop_event.is_set():
            t = tick * period
            if t > self._scenario.duration_s:
                break

            for uav in self._scenario.uavs:
                self._scenario_logger.log_truth(uav.name, t, true_state_for_uav(uav, t))

            tick += 1
            if self._realtime:
                target = start_monotonic + tick * period
                remaining = target - time_module.monotonic()
                if remaining > 0:
                    stop_event.wait(remaining)

        logger.info("truth runner finished")
