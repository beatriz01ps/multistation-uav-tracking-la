"""UDP/JSON receiver"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from typing import TYPE_CHECKING, Optional

from io_.parser import parse_local_tracklet
from tracking.tracker import Tracker

if TYPE_CHECKING:
    from io_.track_history_logger import TrackHistoryLogger

logger = logging.getLogger(__name__)

Address = tuple[str, int]

DEFAULT_PROCESS_INTERVAL_S = 0.5


class UdpReceiver:
    def __init__(self, tracker: Tracker, host: str = "0.0.0.0", port: int = 9999, process_interval_s: float = DEFAULT_PROCESS_INTERVAL_S,
        max_packet_size: int = 65536, track_logger: Optional["TrackHistoryLogger"] = None, receive_threads: int = 2) -> None:

        self._tracker = tracker
        self._host = host
        self._port = port
        self._process_interval_s = process_interval_s
        self._max_packet_size = max_packet_size
        self._track_logger = track_logger 
        self._terminal_events_flushed = 0
        self._association_candidates_flushed = 0
        self._receive_thread_count = receive_threads

        self._socket: Optional[socket.socket] = None
        self._receive_threads: list[threading.Thread] = []
        self._process_thread: Optional[threading.Thread] = None
        self._running = threading.Event()

        self._address_lock = threading.Lock()
        self._last_address_by_source: dict[tuple[str, str], Address] = {}

    def start(self) -> None:

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((self._host, self._port))
        self._socket.settimeout(0.5)
        self._running.set()

        # All receiving threads read from the same socket concurrently - the
        # OS dispatches each datagram to one of them; they share the same stateless _receive_loop.
        self._receive_threads = [
            threading.Thread(target=self._receive_loop, name=f"udp-receive-{i}", daemon=True)
            for i in range(self._receive_thread_count)
        ]
        self._process_thread = threading.Thread(target=self._process_loop, name="tracker-tick", daemon=True)
        for thread in self._receive_threads:
            thread.start()
        self._process_thread.start()

    def stop(self) -> None:

        self._running.clear()
        for thread in self._receive_threads:
            thread.join(timeout=2.0)
        if self._process_thread is not None:
            self._process_thread.join(timeout=2.0)
        if self._socket is not None:
            self._socket.close()
        if self._track_logger is not None:
            self._track_logger.close()

    def _receive_loop(self) -> None:

        assert self._socket is not None
        while self._running.is_set():
            try:
                raw, addr = self._socket.recvfrom(self._max_packet_size)
            except socket.timeout:
                continue
            except OSError:
                break  # socket closes

            try:
                payload = json.loads(raw.decode("utf-8"))
                tracklet = parse_local_tracklet(payload)
            except Exception:
                logger.exception("invalid packet discarded")
                continue

            with self._address_lock:
                self._last_address_by_source[(tracklet.station_id, tracklet.local_track_id)] = addr

            self._tracker.ingest(tracklet)

    def _process_loop(self) -> None:

        while self._running.is_set():
            try:
                
                events = self._tracker.tick()
                self._send_responses(events)
                if self._track_logger is not None:
                    self._track_logger.log_snapshot(self._tracker.track_manager.active_tracks())
                    new_terminal_events = self._tracker.terminal_events[self._terminal_events_flushed :]
                    if new_terminal_events:
                        self._track_logger.log_terminal_events(new_terminal_events)
                        self._terminal_events_flushed = len(self._tracker.terminal_events)
                    new_candidates = self._tracker.association_candidates[self._association_candidates_flushed :]
                    if new_candidates:
                        self._track_logger.log_association_candidates(new_candidates)
                        self._association_candidates_flushed = len(self._tracker.association_candidates)
            except Exception:
                
                logger.exception("error processing cycle; tracker keeps running")
            time.sleep(self._process_interval_s)

    def _send_responses(self, events) -> None:
        
        if not events or self._socket is None:
            return
        with self._address_lock:
            addresses = dict(self._last_address_by_source)

        for event in events:
            addr = addresses.get((event.station_id, event.local_track_id))
            if addr is None:
                continue
            response = {
                "station_id": event.station_id,
                "local_track_id": event.local_track_id,
                "global_track_id": event.global_track_id,
                "timestamp": event.timestamp,
            }
            try:
                self._socket.sendto(json.dumps(response).encode("utf-8"), addr)
            except OSError:
                logger.exception("failed to send response to %s", addr)
