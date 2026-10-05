"""Load-test a graph inference server (tools/imitation_scale/graph_server.py, docs/022 §5): K client
threads, each sending one real decision state at a time as a search worker does (GraphNet's synchronous
calls), for T seconds. Reports states a second and per-call latency, to size how many search workers
one server keeps fed.

    python tools/imitation_scale/graph_load.py --tables-dir data/imitation_graph/h5 --port 50062 \\
        --clients 1,8,28,56 --seconds 20 --json runs/gnn_bench/load.json
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import msgpack  # noqa: E402

from draftzero.gameplay import graph_tables as gt  # noqa: E402


def load_states(path: Path, n: int) -> list[bytes]:
    """n states of a graph file as /evaluate request bodies (raw ids: the server maps them)."""
    import h5py
    out = []
    with h5py.File(path, "r") as f:
        node_ptr, edge_ptr = f["node_ptr"][:], f["edge_ptr"][:]
        rows = np.linspace(0, len(node_ptr) - 2, min(n, len(node_ptr) - 1)).astype(int)
        ids, vals = f["node_ids"][:node_ptr[rows[-1] + 1]], f["node_values"][:node_ptr[rows[-1] + 1]]
        ch, pa, lb = (f[k][:edge_ptr[rows[-1] + 1]] for k in ("edge_child", "edge_parent", "edge_label"))
        for r in rows:
            a, b = node_ptr[r], node_ptr[r + 1]
            c, d = edge_ptr[r], edge_ptr[r + 1]
            out.append(msgpack.packb({"indices": ids[a:b].tolist(), "values": vals[a:b].astype(int).tolist(),
                                      "offsets": [0], "edge_child": ch[c:d].astype(int).tolist(),
                                      "edge_parent": pa[c:d].astype(int).tolist(),
                                      "edge_label": lb[c:d].tolist(), "edge_offsets": [0]}, use_bin_type=True))
    return out


def run(port: int, bodies: list[bytes], clients: int, seconds: float) -> dict:
    url = f"http://127.0.0.1:{port}/evaluate"
    lat: list[float] = []
    lock = threading.Lock()
    stop = time.perf_counter() + seconds

    def client(k: int):
        i, mine = k, []
        while time.perf_counter() < stop:
            req = urllib.request.Request(url, data=bodies[i % len(bodies)],
                                         headers={"Content-Type": "application/x-msgpack"})
            t0 = time.perf_counter()
            urllib.request.urlopen(req, timeout=60).read()
            mine.append(time.perf_counter() - t0)
            i += clients
        with lock:
            lat.extend(mine)
    th = [threading.Thread(target=client, args=(k,)) for k in range(clients)]
    t0 = time.perf_counter()
    for t in th:
        t.start()
    for t in th:
        t.join()
    el = time.perf_counter() - t0
    a = np.asarray(lat) * 1000
    return {"clients": clients, "states": len(a), "states_per_s": round(len(a) / el, 1),
            "latency_ms_p50": round(float(np.median(a)), 2), "latency_ms_p90": round(float(np.percentile(a, 90)), 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables-dir", required=True)
    ap.add_argument("--table", default="replay_priority_val")
    ap.add_argument("--port", type=int, default=50062)
    ap.add_argument("--clients", default="1,8,28")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--states", type=int, default=2000)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    bodies = load_states(Path(a.tables_dir) / f"{a.table}{gt.GRAPH_SUFFIX}", a.states)
    res = []
    for c in (int(x) for x in a.clients.split(",")):
        r = run(a.port, bodies, c, a.seconds)
        stats = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/stats").read())
        r["server_states_per_batch"] = round(stats["states_per_batch"], 2)
        print(json.dumps(r), flush=True)
        res.append(r)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
