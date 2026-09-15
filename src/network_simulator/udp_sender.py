"""Thin UDP client: sends JSON to the real receiver and listens for responses on the same socket"""

from __future__ import annotations

import json
import socket
import threading
from typing import Any, Optional


class UdpSender:
    def __init__(self, host: str, port: int, local_port: int = 0, timeout_s: float = 0.5) -> None:

        self._target = (host, port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind(("0.0.0.0", local_port))
        self._socket.settimeout(timeout_s)
        self._send_lock = threading.Lock()

    def send(self, message: dict[str, Any]) -> None:

        payload = json.dumps(message).encode("utf-8")
        with self._send_lock:
            self._socket.sendto(payload, self._target)

    def try_receive(self) -> Optional[dict[str, Any]]:
        
        try:
            raw, _addr = self._socket.recvfrom(65536)
        except (socket.timeout, OSError):
            return None
        return json.loads(raw.decode("utf-8"))

    def close(self) -> None:
        
        self._socket.close()
