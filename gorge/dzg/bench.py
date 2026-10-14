"""Throughput of a dzg network: collate, the training loader, training steps, and inference (direct and
through the socket server).

  python -m dzg.bench --arch mlp [--device auto] [--shards DIR] [--out bench.json]

Collate: one thread over the records in RAM. Loader: the training loader over the records written as
--loader-shards synthetic shards (memory-mapped or merged in RAM; threads or worker processes), states/s
after its first batches. Training: states/s of forward + backward + AdamW step at --batch (default 512)
after warmup, on batches collated beforehand. Inference: states/s and ms per request at request sizes
1, 16, 64, 256: the forward pass alone on a batch already on the device ("model"), the server's whole
request handling in-process (decode, validate, batch, forward, reply: "direct"), and through a socket
server in this process with one client connection ("socket"). Connections ("conns"): client processes,
each one connection sending requests of --conn-states states back to back to a server in this process,
as gorge's 4 connections a server do; states/s, latency, and the server's own stats (states per
forward, busy).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import tempfile
import time

import numpy as np
import torch

from . import data as D
from . import models as M
from . import packfmt as pf
from .client import Client
from .serve import Server


def sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _timed(fn, min_time: float, max_iters: int, min_iters: int = 3) -> tuple[float, int]:
    """(seconds per call, calls) running fn until min_time has passed (at least min_iters calls)."""
    n, t0 = 0, time.perf_counter()
    while n < max_iters and (n < min_iters or time.perf_counter() - t0 < min_time):
        fn()
        n += 1
    return (time.perf_counter() - t0) / n, n


def bench_collate(pack, n_batches: int, batch: int, seed: int = 0) -> dict:
    B = len(pack["dense"])
    rng = np.random.default_rng(seed)
    sels = [np.sort(rng.choice(B, min(batch, B), replace=False)) for _ in range(n_batches)]
    D.collate(pack, sels[0])
    t0 = time.perf_counter()
    for s in sels:
        D.collate(pack, s)
    dt = time.perf_counter() - t0
    return {"states_per_s": n_batches * len(sels[0]) / dt, "ms_per_batch": 1000 * dt / n_batches, "batch": len(sels[0])}


def bench_loader(records: int, batch: int, configs: list[str], shards: int, n_batches: int = 200, skip: int = 10,
                 seed: int = 123, sparse_mean: int = 30) -> dict:
    """The training Loader over synthetic shards: configs like "threads2-mmap", "threads2-ram", "procs4-mmap".
    states/s from batch `skip` to the last; worker start (first_batch_s) and shutdown (close_s) apart."""
    out = {"shards": shards, "records": records}
    d = tempfile.mkdtemp(prefix="dzgl")
    try:
        pf.write_synthetic(d, records, seed=seed, shard_size=-(-records // shards), sparse_mean=sparse_mean)
        rng = np.random.default_rng(seed)
        for cfg in configs:
            m = re.fullmatch(r"(threads|procs)(\d+)-(mmap|ram)", cfg)
            if not m:
                raise SystemExit(f"--loader {cfg!r}: expected threadsN-mmap, threadsN-ram, procsN-mmap or procsN-ram")
            kind, n, store = m.group(1), int(m.group(2)), m.group(3)
            ds = D.ShardSet(d, in_ram=store == "ram")
            bl = [np.sort(rng.choice(len(ds), min(batch, len(ds)), replace=False)) for _ in range(n_batches)]
            kw = {"procs": n} if kind == "procs" else {"workers": n}
            t0 = time.perf_counter()
            for i, _ in enumerate(D.Loader(ds, bl, prefetch=8, **kw)):
                if i == 0:
                    t1 = time.perf_counter()
                if i == skip:
                    t2 = time.perf_counter()
                t3 = time.perf_counter()
            t4 = time.perf_counter()
            out[cfg] = {"states_per_s": (n_batches - skip - 1) * len(bl[0]) / (t3 - t2), "first_batch_s": t1 - t0,
                        "close_s": t4 - t3}
            print(f"  loader {cfg} ({shards} shards): {out[cfg]['states_per_s']:.0f} states/s (first batch after "
                  f"{t1 - t0:.2f} s, closed in {t4 - t3:.2f} s)", flush=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return out


def _conn_client(path: str, msgs: list, ready, go, stop, results) -> None:
    """One client process: one connection, requests back to back from `go` until `stop`."""
    from .client import Client
    rng = np.random.default_rng(os.getpid())
    c = Client(socket_path=path, timeout=600)
    c.request_bytes(msgs[0][1])
    ready.put(1)
    go.wait()
    lat, states = [], 0
    while not stop.is_set():
        n, m = msgs[int(rng.integers(len(msgs)))]
        t = time.perf_counter()
        c.request_bytes(m)
        lat.append(time.perf_counter() - t)
        states += n
    c.close()
    results.put((states, lat))


def bench_conns(model, pack, device, amp: str, conns: list[int], lo: int, hi: int, min_time: float,
                merge: bool = True, seed: int = 0) -> dict:
    """Requests of lo..hi states from `conns` client processes at once."""
    rng = np.random.default_rng(seed)
    B = len(pack["dense"])
    msgs = []
    for _ in range(16):
        n = int(rng.integers(lo, hi + 1))
        sub = D.take(pack, np.sort(rng.choice(B, min(n, B), replace=False)), targets=False)
        sub = {k: np.ascontiguousarray(v, dtype=pf.SPECS[k][0]) for k, v in sub.items()}
        msgs.append((len(sub["dense"]), pf.encode_request(sub)))
    srv = Server(model, device, amp, log_every=3600, merge=merge)
    for _, m in msgs:   # every request shape once (MPS compiles per shape)
        srv.handle(bytearray(m[4:]))
    d = tempfile.mkdtemp(prefix="dzgc")
    path = os.path.join(d, "c.sock")
    srv.bind(socket_path=path)
    srv.start()
    ctx = mp.get_context("spawn")
    out = {}
    try:
        for k in conns:
            ready, results = ctx.Queue(), ctx.Queue()
            go, stop = ctx.Event(), ctx.Event()
            ps = [ctx.Process(target=_conn_client, args=(path, msgs, ready, go, stop, results), daemon=True)
                  for _ in range(k)]
            for p in ps:
                p.start()
            for _ in ps:
                ready.get(timeout=300)
            srv.log_stats(log=False)
            go.set()
            t0 = time.perf_counter()
            time.sleep(min_time)
            stop.set()
            got = [results.get(timeout=300) for _ in ps]
            dt = time.perf_counter() - t0
            st = srv.log_stats(log=False)
            for p in ps:
                p.join(timeout=30)
            lat = np.concatenate([np.asarray(g[1]) for g in got]) * 1000
            r = {"requests_per_s": len(lat) / dt, "states_per_s": sum(g[0] for g in got) / dt,
                 "p50_ms": float(np.percentile(lat, 50)), "p90_ms": float(np.percentile(lat, 90)),
                 "states_per_forward": st["states_per_forward"], "requests_per_forward": st["requests_per_forward"],
                 "forward_ms": st["forward_ms"], "prep_ms": st["prep_ms"], "wait_ms": st["wait_ms"], "busy": st["busy"]}
            out[str(k)] = r
            print(f"  conns {k} ({'merge' if merge else 'no merge'}, {lo}-{hi} states): {r['requests_per_s']:.1f} req/s, "
                  f"{r['states_per_s']:.0f} states/s, p50 {r['p50_ms']:.1f} ms; {r['requests_per_forward']:.2f} "
                  f"requests/forward, forward {r['forward_ms']:.1f} ms, busy {r['busy']:.2f}", flush=True)
    finally:
        srv.close()
        shutil.rmtree(d, ignore_errors=True)
    return out


def bench_train(model, pack, device, batch: int, steps: int, warmup: int, amp: str, seed: int = 0) -> dict:
    from .train import losses, _optimizer

    class A:   # the loss arguments train.py would pass
        policy_target, visits_temp, cq_c_visit, cq_c_scale, value_blend, value_weight = "visits", 1.0, 50.0, 0.1, 0.0, 1.0

    B = len(pack["dense"])
    rng = np.random.default_rng(seed)
    nb = min(steps + warmup, 16)
    bs = [D.collate(pack, np.sort(rng.choice(B, min(batch, B), replace=False))).to(device) for _ in range(nb)]
    model.train()
    opt = _optimizer(model, 1e-4, 0.01, device)

    def step(b):
        with M.autocast(device, amp):
            total, *_ = losses(model, b, A)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for i in range(warmup):
        step(bs[i % nb])
    sync(device)
    t0 = time.perf_counter()
    for i in range(steps):
        step(bs[(warmup + i) % nb])
    sync(device)
    dt = time.perf_counter() - t0
    model.eval()
    return {"states_per_s": steps * bs[0].B / dt, "ms_per_step": 1000 * dt / steps, "batch": bs[0].B, "steps": steps}


def bench_infer(model, pack, device, sizes, amp: str, min_time: float, socket_dir: str | None) -> dict:
    out = {"model": {}, "direct": {}, "socket": {}}
    srv = Server(model, device, amp, log_every=3600)
    d = socket_dir or tempfile.mkdtemp(prefix="dzgb")
    path = os.path.join(d, "bench.sock")
    srv.bind(socket_path=path)
    srv.start()
    cli = Client(socket_path=path)
    try:
        for n in sizes:
            sub = D.take(pack, np.arange(min(n, len(pack["dense"]))), targets=False)
            sub = {k: np.ascontiguousarray(v, dtype=pf.SPECS[k][0]) for k, v in sub.items()}
            msg = pf.encode_request(sub)
            body = bytearray(msg[4:])
            b = D.batch_from_wire(sub).to(device)

            def fwd():
                with torch.inference_mode(), M.autocast(device, amp):
                    model(b)
                sync(device)

            fwd()
            sec, iters = _timed(fwd, min_time, 5000)
            out["model"][str(n)] = {"ms_per_request": 1000 * sec, "states_per_s": n / sec, "iters": iters}
            srv.handle(body)
            sec, iters = _timed(lambda: srv.handle(body), min_time, 5000)
            out["direct"][str(n)] = {"ms_per_request": 1000 * sec, "states_per_s": n / sec, "iters": iters}
            cli.request_bytes(msg)
            sec, iters = _timed(lambda: cli.request_bytes(msg), min_time, 5000)
            out["socket"][str(n)] = {"ms_per_request": 1000 * sec, "states_per_s": n / sec, "iters": iters}
            print(f"  infer {n:4d}: model {out['model'][str(n)]['ms_per_request']:.2f} ms, "
                  f"direct {out['direct'][str(n)]['ms_per_request']:.2f} ms, "
                  f"socket {out['socket'][str(n)]['ms_per_request']:.2f} ms/request "
                  f"({out['socket'][str(n)]['states_per_s']:.0f} states/s)", flush=True)
    finally:
        cli.close()
        srv.close()
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(prog="dzg.bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=M.ARCHS, default="mlp")
    ap.add_argument("--config", default=None, help="JSON architecture overrides")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", choices=("bf16", "off"), default="bf16")
    ap.add_argument("--shards", nargs="*", default=None, help="shard directories (default: synthetic data)")
    ap.add_argument("--records", type=int, default=20000, help="synthetic records (or shard records loaded)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--train-steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--sizes", default="1,16,64,256")
    ap.add_argument("--min-time", type=float, default=2.0, help="seconds per inference measurement")
    ap.add_argument("--conns", default="1,4", help="client processes (connections) for the conns test")
    ap.add_argument("--conn-states", default="32,256", help="min,max states per request in the conns test")
    ap.add_argument("--conn-modes", default="merge", help="comma list: merge (default server), nomerge (--no-merge)")
    ap.add_argument("--loader-shards", type=int, default=10, help="shards for the loader test")
    ap.add_argument("--loader", default="threads2-mmap,threads2-ram,procs4-mmap",
                    help="loader configs: threadsN or procsN, then -mmap or -ram")
    ap.add_argument("--sparse-mean", type=int, default=30, help="synthetic sparse rows per state")
    ap.add_argument("--skip", default="", help="comma list of parts to skip: collate,loader,train,infer,conns")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    device = M.pick_device(args.device)
    torch.manual_seed(0)
    if args.shards:
        ds = D.ShardSet(args.shards, in_ram=False)
        pack = ds.take(np.arange(min(args.records, len(ds))))
        pack = {k: np.asarray(v, dtype=pf.SPECS[k][0]) if k in pf.SPECS else v for k, v in pack.items()}
        source = [str(x) for x in ds.dirs]
    else:
        pack, _ = pf.synthetic(args.records, seed=123, sparse_mean=args.sparse_mean)
        source = "synthetic"
    model = M.build(args.arch, json.loads(args.config) if args.config else None).to(device).eval()
    c = pf.counts(pack)
    res = {"arch": args.arch, "config": model.config, "params": model.parameter_counts()["total"],
           "device": str(device), "amp": args.amp if device.type == "cuda" else "off", "torch": torch.__version__,
           "data": source, "per_state": {k: v / c["B"] for k, v in c.items() if k != "B"},
           "threads": torch.get_num_threads()}
    skip = set(filter(None, args.skip.split(",")))
    print(f"dzg.bench {args.arch} ({res['params']:,} params) on {device}", flush=True)
    if "collate" not in skip:
        res["collate"] = bench_collate(pack, 40, args.batch)
        print(f"  collate: {res['collate']['states_per_s']:.0f} states/s", flush=True)
    if "loader" not in skip and not args.shards and args.loader:
        res["loader"] = bench_loader(args.records, args.batch, args.loader.split(","), args.loader_shards,
                                     sparse_mean=args.sparse_mean)
    if "train" not in skip:
        res["train"] = bench_train(model, pack, device, args.batch, args.train_steps, args.warmup, args.amp)
        print(f"  train: {res['train']['states_per_s']:.0f} states/s ({res['train']['ms_per_step']:.1f} ms/step)", flush=True)
    if "infer" not in skip:
        sizes = [int(s) for s in args.sizes.split(",") if s]
        res["infer"] = bench_infer(model, pack, device, sizes, args.amp, args.min_time, None)
    if "conns" not in skip:
        lo, hi = (int(x) for x in args.conn_states.split(","))
        conns = [int(k) for k in args.conns.split(",") if k]
        res["conns"] = {m: bench_conns(model, pack, device, args.amp, conns, lo, hi, args.min_time, merge=m == "merge")
                        for m in args.conn_modes.split(",") if m}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
    print(json.dumps({k: res[k] for k in ("arch", "params", "device") if k in res}
                     | {"collate_states_per_s": res.get("collate", {}).get("states_per_s"),
                        "loader_states_per_s": {k: v["states_per_s"] for k, v in res.get("loader", {}).items()
                                                if isinstance(v, dict)},
                        "train_states_per_s": res.get("train", {}).get("states_per_s"),
                        "server_states_per_s": {f"{m}/{k}": v["states_per_s"] for m, r in res.get("conns", {}).items()
                                                for k, v in r.items()}}), flush=True)
    return res


if __name__ == "__main__":
    main()
