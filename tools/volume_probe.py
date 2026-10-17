"""volume_probe.py — measure what a run's store relies on, on a folder shared by several machines (docs/035).

Run the two halves of a probe on two pods that mount the same RunPod network volume, at the same time:

    python tools/volume_probe.py mount /workspace                     # filesystem type, options, free space
    python tools/volume_probe.py replace-write /workspace/probe --seconds 60     # pod A: rename a JSON file into place 10x/s
    python tools/volume_probe.py replace-read  /workspace/probe --seconds 60     # pod B: read it 10x/s
    python tools/volume_probe.py appear-write  /workspace/probe --n 20          # pod A: a new file every 1.5 s
    python tools/volume_probe.py appear-read   /workspace/probe --n 20          # pod B: how long until it shows
    python tools/volume_probe.py listing /workspace/probe --files 2000          # make 2,000 small files, list them

replace-read counts reads that found no file, reads of a half-written file (invalid JSON) and how far behind the
writer its reads were; appear-read reports the delay from the writer's write to the reader's first sight (the two
clocks are NTP-synced, so a few ms of it is clock offset). Each prints one JSON line.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _replace(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def mount(a) -> dict:
    out = subprocess.run(["findmnt", "-J", "-T", a.dir], capture_output=True, text=True).stdout
    st = os.statvfs(a.dir)
    return {"findmnt": json.loads(out) if out.strip() else None,
            "free_gb": round(st.f_bavail * st.f_frsize / 1e9, 1), "size_gb": round(st.f_blocks * st.f_frsize / 1e9, 1)}


def replace_write(a) -> dict:
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    path, pad = d / "control.json", "x" * a.pad
    n, end, ms = 0, time.time() + a.seconds, []
    while time.time() < end:
        n += 1
        t0 = time.perf_counter()
        _replace(path, json.dumps({"i": n, "t": time.time(), "pad": pad}).encode())
        ms.append((time.perf_counter() - t0) * 1000)
        time.sleep(max(0.0, 1 / a.hz - ms[-1] / 1000))
    return {"writes": n, "write_ms_median": round(statistics.median(ms), 1), "write_ms_max": round(max(ms), 1)}


def replace_read(a) -> dict:
    path = Path(a.dir) / "control.json"
    reads = missing = partial = 0
    lags, ms = [], []
    end = time.time() + a.seconds
    while time.time() < end:
        t0 = time.perf_counter()
        try:
            b = path.read_bytes()
        except FileNotFoundError:
            missing += 1
        else:
            reads += 1
            try:
                lags.append(time.time() - json.loads(b)["t"])
            except (ValueError, KeyError):
                partial += 1
        ms.append((time.perf_counter() - t0) * 1000)
        time.sleep(max(0.0, 1 / a.hz - ms[-1] / 1000))
    q = sorted(lags)
    return {"reads": reads, "missing": missing, "partial": partial,
            "staleness_s_median": round(statistics.median(q), 3) if q else None,
            "staleness_s_p99": round(q[int(0.99 * (len(q) - 1))], 3) if q else None,
            "read_ms_median": round(statistics.median(ms), 1)}


def appear_write(a) -> dict:
    d = Path(a.dir) / "appear"
    d.mkdir(parents=True, exist_ok=True)
    for k in range(a.n):
        _replace(d / f"f{k:04d}.json", json.dumps({"t": time.time()}).encode())
        time.sleep(a.every)
    return {"written": a.n}


def appear_read(a) -> dict:
    d = Path(a.dir) / "appear"
    delays, k, end = [], 0, time.time() + a.n * a.every + 60
    while k < a.n and time.time() < end:
        p = d / f"f{k:04d}.json"
        if p.exists():
            try:
                delays.append(time.time() - json.loads(p.read_bytes())["t"])
                k += 1
                continue
            except (FileNotFoundError, ValueError):
                pass
        time.sleep(0.01)
    q = sorted(delays)
    return {"seen": len(q), "of": a.n, "delay_s_median": round(statistics.median(q), 3) if q else None,
            "delay_s_max": round(q[-1], 3) if q else None}


def listing(a) -> dict:
    d = Path(a.dir) / "many"
    t0 = time.perf_counter()
    for i in range(a.files):
        p = d / f"{i // 3:06d}" / f"f{i % 3}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        _replace(p, b"{}")
    made = time.perf_counter() - t0
    t0 = time.perf_counter()
    n = sum(len(f) for _, _, f in os.walk(d))
    walk = time.perf_counter() - t0
    t0 = time.perf_counter()
    m = sum(1 for p in d.rglob("*") if p.is_file())
    rglob = time.perf_counter() - t0
    return {"files": n, "rglob_files": m, "create_ms_each": round(made / a.files * 1000, 2),
            "os_walk_s": round(walk, 3), "rglob_stat_s": round(rglob, 3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["mount", "replace-write", "replace-read", "appear-write", "appear-read", "listing"])
    ap.add_argument("dir")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--hz", type=float, default=10)
    ap.add_argument("--pad", type=int, default=65536, help="bytes of padding in the replaced file")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--every", type=float, default=1.5)
    ap.add_argument("--files", type=int, default=2000)
    a = ap.parse_args(argv)
    fn = {"mount": mount, "replace-write": replace_write, "replace-read": replace_read, "appear-write": appear_write,
          "appear-read": appear_read, "listing": listing}[a.what]
    print(json.dumps({"probe": a.what, "host": os.uname().nodename, **fn(a)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
