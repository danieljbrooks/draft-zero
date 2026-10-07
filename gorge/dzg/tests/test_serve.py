import os
import socket
import struct
import tempfile
import threading
import time

import numpy as np
import pytest
import torch

from dzg import data as D
from dzg import models as M
from dzg import packfmt as pf
from dzg.client import Client, recv_exact
from dzg.serve import Server

from .conftest import TINY, edge_records


@pytest.fixture
def server():
    torch.manual_seed(0)
    m = M.build("transformer", TINY["transformer"]).eval()
    with torch.no_grad():
        torch.nn.init.normal_(m.policy[-1].weight, std=0.5)
        torch.nn.init.normal_(m.value[-1].weight, std=0.5)
    srv = Server(m, torch.device("cpu"), amp="off", log_every=3600)
    d = tempfile.mkdtemp(prefix="dzgt")   # short: unix socket paths are limited to ~104 bytes
    path = os.path.join(d, "s.sock")
    srv.bind(socket_path=path)
    srv.start()
    yield srv, m, path
    srv.close()


def _state_pack(pack, sel):
    sub = D.take(pack, sel, targets=False)
    return {k: np.ascontiguousarray(v, dtype=pf.SPECS[k][0]) for k, v in sub.items()}


def test_roundtrip(server, syn):
    srv, m, path = server
    pack, _ = syn
    sub = _state_pack(pack, np.arange(30))
    want = (np.arange(30) % 4 != 1).astype(np.uint8)
    with torch.no_grad():
        s_ref, v_ref = m(D.batch_from_wire(sub))
    nopt = np.diff(sub["opt_off"])
    wanted_opt = np.repeat(want, nopt).astype(bool)
    with Client(socket_path=path) as c:
        for _ in range(3):   # several requests on one connection
            v, s = c.evaluate(sub, want)
            assert v.shape == (30,) and s.shape == (int(sub["opt_off"][-1]),)
            assert np.allclose(v, torch.sigmoid(v_ref).numpy(), atol=1e-6)
            assert np.allclose(s[wanted_opt], s_ref.numpy()[wanted_opt], atol=1e-5)
            assert np.all(s[~wanted_opt] == 0)
        # edge states (zero cards, options, sparse rows) and a one-state request
        e = pf.pack_records(edge_records())
        e = {k: e[k] for k in pf.STATE_NAMES}
        v, s = c.evaluate(e)
        assert v.shape == (6,) and np.all((v > 0) & (v < 1)) and s.shape == (int(e["opt_off"][-1]),)
        v, s = c.evaluate(_state_pack(pack, np.array([5])))
        assert v.shape == (1,)
        v, s = c.evaluate(_state_pack(pack, np.arange(0)))   # an empty request
        assert v.shape == (0,) and s.shape == (0,)


def test_concurrent_connections(server, syn):
    srv, m, path = server
    pack, _ = syn
    reqs = [_state_pack(pack, np.arange(i * 10, i * 10 + 10)) for i in range(4)]
    with torch.no_grad():
        refs = [m(D.batch_from_wire(r)) for r in reqs]
    errors = []

    def worker(i):
        try:
            with Client(socket_path=path) as c:
                for _ in range(5):
                    v, s = c.evaluate(reqs[i])
                    assert np.allclose(s, refs[i][0].numpy(), atol=1e-5)
                    assert np.allclose(v, torch.sigmoid(refs[i][1]).numpy(), atol=1e-6)
        except Exception as e:   # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    assert not errors, errors
    st = srv.log_stats()
    assert st["requests"] == 20


def test_partial_writes_and_bad_request(server, syn):
    srv, m, path = server
    pack, _ = syn
    msg = pf.encode_request(_state_pack(pack, np.arange(3)))
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(10)
    s.connect(path)
    for i in range(0, len(msg), 7):   # dribble the request in small pieces (header split too)
        s.sendall(msg[i:i + 7])
        if i % 70 == 0:
            time.sleep(0.001)
    (n,) = struct.unpack("<I", recv_exact(s, 4))
    v, sc = pf.decode_response(recv_exact(s, n))
    assert v.shape == (3,)
    # a bad magic closes the connection
    bad = bytearray(msg)
    bad[4:8] = b"XXXX"
    s.sendall(bytes(bad))
    assert s.recv(4) == b""
    s.close()


class _Gated(torch.nn.Module):
    """A model that sleeps `delay` seconds per forward, holds its first forward until `gate` is set, and
    counts its forwards and their batch sizes."""

    def __init__(self, inner, delay: float = 0.0):
        super().__init__()
        self.inner, self.delay = inner, delay
        self.gate, self.entered = threading.Event(), threading.Event()
        self.sizes = []

    def forward(self, b):
        self.entered.set()
        if not self.sizes:
            self.gate.wait(30)
        self.sizes.append(b.B)
        if self.delay:
            time.sleep(self.delay)
        return self.inner(b)


def _tiny_model():
    torch.manual_seed(0)
    m = M.build("transformer", TINY["transformer"]).eval()
    with torch.no_grad():
        torch.nn.init.normal_(m.policy[-1].weight, std=0.5)
        torch.nn.init.normal_(m.value[-1].weight, std=0.5)
    return m


def _start(model, **kw):
    srv = Server(model, torch.device("cpu"), amp="off", log_every=3600, **kw)
    path = os.path.join(tempfile.mkdtemp(prefix="dzgt"), "s.sock")
    srv.bind(socket_path=path)
    srv.start()
    return srv, path


def _wait_queued(srv, n, timeout=20.0):
    t = time.time()
    while srv._queue.qsize() < n:
        assert time.time() - t < timeout, "requests never queued"
        time.sleep(0.005)


def test_concurrent_requests_share_a_forward(syn):
    pack, _ = syn
    inner = _tiny_model()
    g = _Gated(inner)
    srv, path = _start(g)
    try:
        # a first request holds the device; five more, of different sizes and wants, queue behind it
        reqs = [_state_pack(pack, np.arange(i * 20, i * 20 + 3 + 4 * i)) for i in range(6)]
        wants = [(np.arange(len(r["dense"])) % (i + 2) != 1).astype(np.uint8) for i, r in enumerate(reqs)]
        out, errors = [None] * 6, []

        def ask(i):
            try:
                with Client(socket_path=path) as c:
                    out[i] = c.evaluate(reqs[i], wants[i])
            except Exception as e:   # noqa: BLE001
                errors.append(e)

        ts = [threading.Thread(target=ask, args=(0,))]
        ts[0].start()
        assert g.entered.wait(20)
        for i in range(1, 6):
            ts.append(threading.Thread(target=ask, args=(i,)))
            ts[-1].start()
        _wait_queued(srv, 5)
        g.gate.set()
        for t in ts:
            t.join(30)
        assert not errors, errors
        assert g.sizes == [len(reqs[0]["dense"]), sum(len(r["dense"]) for r in reqs[1:])]
        for i, (v, s) in enumerate(out):
            with torch.no_grad():
                s_ref, v_ref = inner(D.batch_from_wire(reqs[i]))
            keep = np.repeat(wants[i], np.diff(reqs[i]["opt_off"])).astype(bool)
            assert np.allclose(v, torch.sigmoid(v_ref).numpy(), atol=1e-5)
            assert np.allclose(s[keep], s_ref.numpy()[keep], atol=1e-5) and np.all(s[~keep] == 0)
        st = srv.log_stats(log=False)
        assert st["requests"] == 6 and st["forwards"] == 2 and st["requests_per_forward"] == 3
    finally:
        srv.close()


def test_merge_cap(syn):
    pack, _ = syn
    g = _Gated(_tiny_model())
    srv, path = _start(g, max_merge=25)
    try:
        reqs = [_state_pack(pack, np.arange(i * 10, i * 10 + 10)) for i in range(5)]

        def ask(r):
            with Client(socket_path=path) as c:
                c.evaluate(r)

        ts = [threading.Thread(target=ask, args=(r,)) for r in reqs]
        ts[0].start()
        assert g.entered.wait(20)   # the first is in the (gated) forward
        for t in ts[1:]:
            t.start()
        _wait_queued(srv, 4)
        g.gate.set()
        for t in ts:
            t.join(30)
        assert g.sizes == [10, 20, 20]   # 25 states at most: two requests of 10 per forward
    finally:
        srv.close()


def test_a_bad_request_fails_alone(syn):
    """Without validation a bad request reaches the model; it fails its own connection only, even when
    it was merged with good ones."""
    pack, _ = syn
    inner = _tiny_model()
    g = _Gated(inner)
    srv, path = _start(g, validate=False)
    try:
        good = [_state_pack(pack, np.arange(i * 7, i * 7 + 7)) for i in range(4)]
        bad = _state_pack(pack, np.arange(40, 45))
        bad["sp_row"] = bad["sp_row"].copy()
        bad["sp_row"][0] = 60000   # outside the 16,384-row table
        res, errs = {}, {}

        def ask(k, p):
            try:
                with Client(socket_path=path) as c:
                    res[k] = c.evaluate(p)
            except Exception as e:   # noqa: BLE001
                errs[k] = e

        ts = [threading.Thread(target=ask, args=(0, good[0]))]
        ts[0].start()
        assert g.entered.wait(20)
        for k, p in [(1, good[1]), ("bad", bad), (2, good[2]), (3, good[3])]:
            ts.append(threading.Thread(target=ask, args=(k, p)))
            ts[-1].start()
        _wait_queued(srv, 4)
        g.gate.set()
        for t in ts:
            t.join(30)
        assert set(res) == {0, 1, 2, 3} and set(errs) == {"bad"}
        assert isinstance(errs["bad"], ConnectionError)
        # the four queued requests went in one forward, which failed, then one at a time
        assert g.sizes[1] == 26 and sorted(g.sizes[2:6]) == [5, 7, 7, 7], g.sizes
        for k in range(4):
            with torch.no_grad():
                s_ref, v_ref = inner(D.batch_from_wire(good[k]))
            assert np.allclose(res[k][0], torch.sigmoid(v_ref).numpy(), atol=1e-5)
            assert np.allclose(res[k][1], s_ref.numpy(), atol=1e-5)
        # the server still answers
        with Client(socket_path=path) as c:
            v, _ = c.evaluate(good[0])
            assert v.shape == (7,)
    finally:
        srv.close()


@pytest.mark.parametrize("merge", [True, False])
def test_forward_ms_excludes_waiting(syn, merge):
    """forward_ms is the time of a forward, not of a forward plus the wait for the device: with 4
    connections queueing on a 40 ms model it stays near 40 ms, and the queueing shows in wait_ms."""
    pack, _ = syn
    delay = 0.04
    g = _Gated(_tiny_model(), delay=delay)
    g.gate.set()
    srv, path = _start(g, merge=merge)
    try:
        req = _state_pack(pack, np.arange(8))

        def ask():
            with Client(socket_path=path) as c:
                for _ in range(4):
                    c.evaluate(req)

        srv.log_stats(log=False)
        t0 = time.perf_counter()
        ts = [threading.Thread(target=ask) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(60)
        wall = time.perf_counter() - t0
        st = srv.log_stats(log=False)
        assert st["requests"] == 16
        assert delay * 1000 <= st["forward_ms"] < 2.5 * delay * 1000, st
        assert st["wait_ms"] > 0.5 * delay * 1000, st
        assert st["busy"] <= 1.05 and st["forwards"] * delay <= wall
        if not merge:
            assert st["forwards"] == 16
    finally:
        srv.close()
