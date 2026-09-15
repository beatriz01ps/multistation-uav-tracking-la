"""Listens for the station_id/local_track_id/global_track_id/timestamp responses the
receiver sends back over UDP and just records them"""

from __future__ import annotations

import logging
import threading

from network_simulator.scenario_logger import ScenarioLogger
from network_simulator.udp_sender import UdpSender

logger = logging.getLogger(__name__)


def listen_for_responses(sender: UdpSender, scenario_logger: ScenarioLogger, stop_event: threading.Event) -> None:
    
    while not stop_event.is_set():
        response = sender.try_receive()
        if response is None:
            continue
        scenario_logger.log_response(response)
        logger.info(
            "RESPONSE %s/%s -> global_track_id=%s",
            response.get("station_id"),
            response.get("local_track_id"),
            response.get("global_track_id"),
        )
