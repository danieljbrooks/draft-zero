"""A Python client for the dzg socket protocol (SPEC §3), for tests and benchmarks."""
from __future__ import annotations

import socket
import struct

import numpy as np

from . import packfmt as pf


def recv_exact(sock: socket.socket, n: int, buf: bytearray | None = None) -> bytearray | None:
    """n bytes from sock (None on a clean EOF before the first byte); handles partial reads."""
    buf = bytearray(n) if buf is None else buf
    mv = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(mv[got:], n - got)
        if k == 0:
            if got == 0:
                return None
            raise ConnectionError(f"connection closed after {got} of {n} bytes")
        got += k
    return buf


def connect(socket_path: str | None = None, tcp: str | None = None, timeout: float | None = None) -> socket.socket:
    if (socket_path is None) == (tcp is None):
        raise ValueError("give exactly one of socket_path, tcp")
    if socket_path is not None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(socket_path)
    else:
        host, _, port = tcp.rpartition(":")
        s = socket.create_connection((host or "127.0.0.1", int(port)), timeout=timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return s


class Client:
    """One connection; requests are sent one at a time and answered in order."""

    def __init__(self, socket_path: str | None = None, tcp: str | None = None, timeout: float | None = 60.0):
        self.sock = connect(socket_path, tcp, timeout)

    def request_bytes(self, msg: bytes) -> tuple[np.ndarray, np.ndarray]:
        self.sock.sendall(msg)
        hdr = recv_exact(self.sock, 4)
        if hdr is None:
            raise ConnectionError("server closed the connection")
        (n,) = struct.unpack("<I", hdr)
        body = recv_exact(self.sock, n)
        if body is None:
            raise ConnectionError("server closed the connection")
        return pf.decode_response(body)

    def evaluate(self, pack: dict, want=None) -> tuple[np.ndarray, np.ndarray]:
        """(value [B] = P(the searching seat wins), score [no] option logits, 0 where want = 0)."""
        return self.request_bytes(pf.encode_request(pack, want))

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
