"""CLI: sends a synthetic scenario over the tracker's real UDP/JSON interface"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path

_SRC_ROOT = str(Path(__file__).resolve().parents[1])
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from network_simulator.response_listener import listen_for_responses
from network_simulator.scenario import load_scenario, station_rngs
from network_simulator.scenario_logger import ScenarioLogger
from network_simulator.station_runner import StationRunner
from network_simulator.truth_runner import TruthRunner
from network_simulator.udp_sender import UdpSender

def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True, help="scenario YAML (see scenario.py)")
    parser.add_argument("--host", default="127.0.0.1", help="tracker address (UDP receiver)")
    parser.add_argument("--port", type=int, default=9999, help="tracker port (same default as src/main.py)")
    parser.add_argument("--duration", type=float, default=None, help="overrides the scenario's simulation.duration (s)")
    parser.add_argument("--seed", type=int, default=None, help="overrides the scenario's simulation.seed")
    parser.add_argument("--realtime", dest="realtime", action="store_true", default=True, help="respects the scenario's real time (default)")
    parser.add_argument("--no-realtime", dest="realtime", action="store_false", help="sends as fast as possible, with no waiting")
    parser.add_argument("--out-dir", default="data/synthetic/network_simulator_run", help="where to save sent_messages/ground_truth/responses .jsonl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("network_simulator")

    scenario = load_scenario(args.scenario)
    if args.duration is not None:
        scenario.duration_s = args.duration
    if args.seed is not None:
        scenario.seed = args.seed

    sender = UdpSender(args.host, args.port)
    scenario_logger = ScenarioLogger(args.out_dir)
    stop_event = threading.Event()

    rngs = station_rngs(scenario)
    runners = [
        StationRunner(station_spec, scenario, sender, scenario_logger, rngs[station_spec.station_id], realtime=args.realtime)
        for station_spec in scenario.stations
    ]

    listener_thread = threading.Thread(
        target=listen_for_responses, args=(sender, scenario_logger, stop_event), name="response-listener", daemon=True
    )

    truth_runner = TruthRunner(scenario, scenario_logger, realtime=args.realtime)
    truth_thread = threading.Thread(target=truth_runner.run, args=(stop_event,), name="truth-runner")
    station_threads = [
        threading.Thread(target=runner.run, args=(stop_event,), name=f"station-{spec.station_id}")
        for runner, spec in zip(runners, scenario.stations)
    ]

    logger.info(
        "scenario=%s duration=%.1fs seed=%d stations=%s -> %s:%s (realtime=%s, out_dir=%s)",
        args.scenario,
        scenario.duration_s,
        scenario.seed,
        [s.station_id for s in scenario.stations],
        args.host,
        args.port,
        args.realtime,
        args.out_dir,
    )

    listener_thread.start()
    truth_thread.start()
    for thread in station_threads:
        thread.start()
    try:
        for thread in station_threads:
            thread.join()
        truth_thread.join()
    finally:
        stop_event.set()
        listener_thread.join(timeout=2.0)
        sender.close()
        scenario_logger.close()

    logger.info("scenario finished. logs at %s", args.out_dir)


if __name__ == "__main__":
    main()
