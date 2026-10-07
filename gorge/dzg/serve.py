"""Serve a dzg network to gorge's search over a Unix or TCP socket (SPEC §3).

  python -m dzg.serve --ckpt OUT/best.pt --socket /tmp/dzg.sock
  python -m dzg.serve --arch gnn --random-init --tcp 127.0.0.1:7070      # speed tests

One thread per connection reads whole requests, decodes and validates them and queues them. One forward
thread owns the device: it takes every request waiting in the queue (up to --max-merge states), runs
them as one forward pass and splits the outputs back, so concurrent connections share a forward instead
of each waiting for its own. --no-merge runs one forward per request instead, under a lock. Each request
is answered with value = sigmoid(value logit) per state and the option logits of the states whose `want`
is 1 (0 elsewhere).

Throughput is logged every --log-every seconds. Per request: parse_ms (decode, validate), wait_ms (from
queued until its forward starts) and reply_ms. Per forward: prep_ms (merge the requests, build the
batch) and forward_ms (copy to the device, forward, copy back). A request's latency is about parse +
wait + prep + forward + reply. busy is the share of wall time the forward thread spent in prep and
forward (with --no-merge, the share the device lock was held): near 1 means this process is the limit
(run more server processes per GPU, SERVERS in dzg_lib.sh), well below 1 means the clients are.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import socket
import struct
import sys
import threading
import time

import numpy as np
import torch

from . import data as D
from . import models as M
from . import packfmt as pf
from .client import recv_exact

MAX_REQUEST = 1 << 31


def _log(msg: str) -> None:
    print(f"[dzg.serve {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


class _Req:
    """One decoded request on its way through the forward thread."""
    __slots__ = ("pack", "kept", "B", "no", "nk", "t_queued", "done", "value", "score", "error")

    def __init__(self, pack, kept, B: int, no: int):
        self.pack, self.kept, self.B, self.no = pack, kept, B, no
        self.nk = int(pack["opt_off"][-1])   # options sent to the model (after dropping want = 0 states')
        self.t_queued = 0.0
        self.done = threading.Event()
        self.value = self.score = self.error = None


class Server:
    def __init__(self, model, device, amp: str = "bf16", max_batch: int = 4096, log_every: float = 30.0,
                 validate: bool = True, merge: bool = True, max_merge: int = 4096):
        self.model, self.device, self.amp = model.eval(), device, amp
        self.max_batch, self.log_every, self.validate = max_batch, log_every, validate
        self.merge, self.max_merge = merge, max(1, max_merge)
        self.lock = threading.Lock()          # the device: one forward at a time
        self.stats_lock = threading.Lock()
        self._reset_stats()
        self.listener = None
        self.address = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._warned = 0.0
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._worker = None
        if merge:
            self._worker = threading.Thread(target=self._work, daemon=True, name="dzg-forward")
            self._worker.start()

    def _reset_stats(self):
        self.stats = {"requests": 0, "states": 0, "options": 0, "parse": 0.0, "wait": 0.0, "reply": 0.0,
                      "forwards": 0, "prep": 0.0, "forward": 0.0, "t": time.time(), "max_states": 0,
                      "max_forward_states": 0}

    # -- one request ----------------------------------------------------------------------------
    def handle(self, body) -> bytes:
        """The response to one request body (the bytes after nbytes). Thread-safe; with merging, the
        request shares a forward with whatever else is queued."""
        t0 = time.perf_counter()
        pack, want = pf.decode_request(body)
        if self.validate:
            pf.validate(pack, train=False)
        B, no = len(want), int(pack["opt_off"][-1])
        if B > self.max_batch and time.time() - self._warned > 10:
            self._warned = time.time()
            _log(f"warning: request of {B} states > --max-batch {self.max_batch}")
        if B == 0:
            return pf.encode_response(np.zeros(0, np.float32), np.zeros(no, np.float32))
        sub, kept = D.drop_unwanted_options(pack, want)
        r = _Req(sub, kept, B, no)
        t1 = r.t_queued = time.perf_counter()
        if self._worker is not None and self._worker.is_alive() and not self._stop.is_set():
            self._queue.put(r)
            while not r.done.wait(1.0):
                if not self._worker.is_alive():
                    raise RuntimeError("the forward thread stopped")
        else:
            self._forward_group([r])
        if r.error is not None:
            raise r.error
        t2 = time.perf_counter()
        if kept is None:
            score = r.score
        else:
            score = np.zeros(no, np.float32)
            score[kept] = r.score
        reply = pf.encode_response(r.value, score)
        t3 = time.perf_counter()
        with self.stats_lock:
            st = self.stats
            st["requests"] += 1
            st["states"] += B
            st["options"] += no
            st["max_states"] = max(st["max_states"], B)
            st["parse"] += t1 - t0
            st["reply"] += t3 - t2
        return reply

    # -- the forward thread ---------------------------------------------------------------------
    def _work(self) -> None:
        carry = None
        while True:
            r = carry if carry is not None else self._queue.get()
            carry = None
            if r is None:   # close()
                break
            group, n, stopping = [r], r.B, False
            while n < self.max_merge:
                try:
                    q = self._queue.get_nowait()
                except queue.Empty:
                    break
                if q is None:
                    stopping = True
                    break
                if n + q.B > self.max_merge:
                    carry = q
                    break
                group.append(q)
                n += q.B
            self._forward_group(group)
            if stopping:
                break
        # requests left behind by close(): fail them rather than leave their connections waiting
        rest = [carry] if carry is not None else []
        while True:
            try:
                q = self._queue.get_nowait()
            except queue.Empty:
                break
            if q is not None:
                rest.append(q)
        for q in rest:
            q.error = RuntimeError("server stopped")
            q.done.set()

    def _forward_group(self, group: list[_Req]) -> None:
        """One forward over the group's requests; on an error, each request alone, so one bad request
        (possible with --no-validate) fails only itself."""
        try:
            self._run(group)
        except Exception as e:   # noqa: BLE001 - reported to the request's connection
            if len(group) == 1:
                group[0].error = e
            else:
                for r in group:
                    try:
                        self._run([r])
                    except Exception as e1:   # noqa: BLE001
                        r.error = e1
        for r in group:
            r.done.set()

    def _run(self, group: list[_Req]) -> None:
        ta = time.perf_counter()
        pack = group[0].pack if len(group) == 1 else D.concat_packs([r.pack for r in group])
        b = D.batch_from_wire(pack)
        tb = time.perf_counter()
        with self.lock:
            tc = time.perf_counter()
            b = b.to(self.device)
            with torch.inference_mode(), M.autocast(self.device, self.amp):
                scores, vlogit = self.model(b)
            value = torch.sigmoid(vlogit.float()).cpu().numpy()
            s = scores.float().cpu().numpy()
            td = time.perf_counter()
        vo = so = 0
        for r in group:
            r.value, r.score = value[vo:vo + r.B], s[so:so + r.nk]
            vo += r.B
            so += r.nk
        with self.stats_lock:
            st = self.stats
            st["forwards"] += 1
            st["prep"] += tb - ta
            st["forward"] += td - tc
            st["wait"] += sum(tc - r.t_queued for r in group) - len(group) * (tb - ta)   # prep is its own line
            st["max_forward_states"] = max(st["max_forward_states"], vo)

    # -- connections ----------------------------------------------------------------------------
    def _conn(self, conn: socket.socket, peer) -> None:
        try:
            while not self._stop.is_set():
                hdr = recv_exact(conn, 4)
                if hdr is None:
                    break
                (n,) = struct.unpack("<I", hdr)
                if n < 32 or n > MAX_REQUEST:
                    _log(f"bad request size {n} from {peer}; closing")
                    break
                body = recv_exact(conn, n)
                if body is None:
                    break
                try:
                    reply = self.handle(body)
                except ValueError as e:
                    _log(f"bad request from {peer}: {e}; closing")
                    break
                except Exception as e:   # noqa: BLE001 - log it here rather than kill the thread silently
                    _log(f"error on a request from {peer}: {type(e).__name__}: {e}; closing")
                    break
                conn.sendall(reply)
        except (ConnectionError, OSError) as e:
            if not self._stop.is_set():
                _log(f"connection {peer}: {e}")
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def bind(self, socket_path: str | None = None, tcp: str | None = None) -> None:
        if (socket_path is None) == (tcp is None):
            raise ValueError("give exactly one of --socket, --tcp")
        if socket_path is not None:
            if os.path.exists(socket_path):
                os.unlink(socket_path)
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.bind(socket_path)
            self.address = socket_path
        else:
            host, _, port = tcp.rpartition(":")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host or "0.0.0.0", int(port)))
            self.address = f"{host}:{s.getsockname()[1]}"
        s.listen(256)
        self.listener = s

    def serve_forever(self) -> None:
        logger = threading.Thread(target=self._log_loop, daemon=True)
        logger.start()
        self.listener.settimeout(0.5)   # so close() from another thread ends the loop on every OS
        while not self._stop.is_set():
            try:
                conn, peer = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.settimeout(None)
            if conn.family != socket.AF_UNIX:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            t = threading.Thread(target=self._conn, args=(conn, peer or "unix"), daemon=True)
            t.start()
            self._threads = [x for x in self._threads if x.is_alive()] + [t]

    def start(self) -> threading.Thread:
        """serve_forever in a background thread (tests, benchmarks)."""
        t = threading.Thread(target=self.serve_forever, daemon=True)
        t.start()
        return t

    def close(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._queue.put(None)
        if self.listener is not None:
            try:
                self.listener.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.listener.close()
        if self.address and os.path.sep in str(self.address) and os.path.exists(self.address):
            try:
                os.unlink(self.address)
            except OSError:
                pass

    def _log_loop(self) -> None:
        while not self._stop.wait(self.log_every):
            self.log_stats()

    def log_stats(self, log: bool = True) -> dict:
        """The stats since the last call (and reset them); logged when there were requests and log."""
        with self.stats_lock:
            st = self.stats
            self._reset_stats()
        dt = max(time.time() - st["t"], 1e-9)
        r = max(st["requests"], 1)
        f = max(st["forwards"], 1)
        out = {"requests_per_s": st["requests"] / dt, "states_per_s": st["states"] / dt,
               "forwards_per_s": st["forwards"] / dt, "requests_per_forward": st["requests"] / f,
               "states_per_forward": st["states"] / f, "max_forward_states": st["max_forward_states"],
               "mean_states": st["states"] / r, "max_states": st["max_states"], "mean_options": st["options"] / r,
               "parse_ms": 1000 * st["parse"] / r, "wait_ms": 1000 * st["wait"] / r,
               "reply_ms": 1000 * st["reply"] / r, "prep_ms": 1000 * st["prep"] / f,
               "forward_ms": 1000 * st["forward"] / f,
               "busy": ((st["prep"] if self.merge else 0.0) + st["forward"]) / dt,
               "requests": st["requests"], "forwards": st["forwards"]}
        if st["requests"] and log:
            _log(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()}))
        return out

    def warmup(self, sizes=(1, 64)) -> None:
        for n in sizes:
            pack, _ = pf.synthetic(n, seed=n)
            msg = pf.encode_request({k: pack[k] for k in pf.STATE_NAMES})
            self.handle(bytearray(msg[4:]))
        self._reset_stats()


def build_model(args, device):
    if args.ckpt:
        model, ck = M.load(args.ckpt, device)
        return model, {"ckpt": args.ckpt, "arch": ck["arch"], "meta_step": ck.get("meta", {}).get("step")}
    if not (args.arch and args.random_init):
        raise SystemExit("give --ckpt, or --arch with --random-init")
    torch.manual_seed(0)
    model = M.build(args.arch, json.loads(args.config) if args.config else None).to(device).eval()
    return model, {"arch": args.arch, "random_init": True}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="dzg.serve", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt")
    ap.add_argument("--arch", choices=M.ARCHS)
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--config", default=None, help="JSON architecture overrides (with --random-init)")
    ap.add_argument("--socket")
    ap.add_argument("--tcp", metavar="HOST:PORT")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", choices=("bf16", "off"), default="bf16")
    ap.add_argument("--max-batch", type=int, default=4096, help="warn on requests with more states")
    ap.add_argument("--max-merge", type=int, default=4096,
                    help="states per merged forward (a larger request still runs, alone)")
    ap.add_argument("--no-merge", action="store_true", help="one forward per request, under a lock")
    ap.add_argument("--log-every", type=float, default=30.0, help="seconds between throughput lines")
    ap.add_argument("--no-validate", action="store_true", help="skip request validation")
    ap.add_argument("--torch-threads", type=int, default=None)
    args = ap.parse_args(argv)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    device = M.pick_device(args.device)
    model, info = build_model(args, device)
    srv = Server(model, device, args.amp, args.max_batch, args.log_every, validate=not args.no_validate,
                 merge=not args.no_merge, max_merge=args.max_merge)
    srv.warmup()
    srv.bind(args.socket, args.tcp)

    def stop(*_):
        srv.close()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    _log(f"listening on {srv.address}: {json.dumps(info)} params {model.parameter_counts()['total']:,} "
         f"device {device} amp {args.amp if device.type == 'cuda' else 'off'} "
         f"{'one forward per request' if args.no_merge else f'merging up to {args.max_merge} states'}")
    srv.serve_forever()
    _log("stopped")


if __name__ == "__main__":
    main()
