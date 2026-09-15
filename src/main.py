"""
CLI entry point: brings up the central tracker listening over UDP/JSON.

Usage:
    python src/main.py --port 9999
    python src/main.py --port 9999 --config src/config/default.yaml
    python src/main.py --port 9999 --track-log-dir data/track_history
"""

from __future__ import annotations

import argparse
import logging
import time

from config.loader import load_config
from io_.receiver import DEFAULT_PROCESS_INTERVAL_S, UdpReceiver
from io_.track_history_logger import TrackHistoryLogger
from tracking.tracker import Tracker


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--config", default=None, help="path to a config YAML")
    parser.add_argument(
        "--process-interval", type=float, default=DEFAULT_PROCESS_INTERVAL_S, help="seconds between tracker cycles"
    )
    parser.add_argument(
        "--track-log-dir",
        default=None,
        help="if set, writes the complete state (position, velocity, covariance, status) "
        "of every GlobalTrack on every cycle, as CSV + JSONL, into this directory",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("main")

    config = load_config(args.config)
    tracker = Tracker(config)
    track_logger = TrackHistoryLogger(args.track_log_dir) if args.track_log_dir else None
    receiver = UdpReceiver(
        tracker,
        host=args.host,
        port=args.port,
        process_interval_s=args.process_interval,
        track_logger=track_logger,
    )

    receiver.start()
    logger.info("central tracker listening on %s:%s", args.host, args.port)

    association_gate = (
        f"euclidean_gate_distance={config.association.euclidean_gate_distance}"
        if config.association.metric.value == "euclidean"
        else f"chi_square_probability={config.association.chi_square_probability}"
    )
    logger.info(
        "active config: association(metric=%s, mode=%s, %s) | "
        "fusion(strategy=%s) | filter(type=%s, motion_model=%s) | "
        "duplicate_merger(enabled=%s, fusion_strategy=%s) | track_log_dir=%s",
        config.association.metric.value,
        config.association.mode.value,
        association_gate,
        config.fusion.strategy.value,
        config.filter.type.value,
        config.filter.motion_model.value,
        config.duplicate_merger.enabled,
        config.duplicate_merger.fusion_strategy.value,
        args.track_log_dir or "disabled",
    )

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("shutting down...")
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
