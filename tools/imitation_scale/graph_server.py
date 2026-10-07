"""An inference server for MageZero's graph network (docs/022), for the games' graph bots (play.py's gnn
bots, java/mzbridge GraphNet) and the policy-only references.

The protocol is MageZero's graph server's (WillWroble/MageZero graph-encoder, server.py), so either
side can be swapped: msgpack POST /evaluate with the states' nodes and edges, concatenated,

    indices, values                       per node (feature id, numeric value)
    offsets                               per state, its first node
    edge_child, edge_parent, edge_label   per edge; child and parent local to the state
    edge_offsets                          per state, its first edge

answered with one map per state: policy_priority and policy_target (one score per node, in the
order sent; NaN where a head doesn't read the node), policy_binary [no, yes], value.
GET /healthz, GET /stats. Requests are batched across connections (up to --max-batch states a forward pass),
as tools/search_bench/value_server.py does for the flat networks. One process serves ~650 states a second on
an RTX 3090 under 28+ concurrent searches, limited by its per-request Python work rather than the GPU (3,465
states a second at batch 32): run two or three replicas per pod on separate ports (play.py --graph-ports).

It loads any checkpoint graph_supervised.py saves (any `arch`), and MageZero's own graph checkpoints.
--value-model takes the value from a second checkpoint with the same vocab (another of the run's checkpoints, e.g.
the epoch its value head was best: docs/024 §5), at the cost of a second forward pass.

    python tools/imitation_scale/graph_server.py --model runs/gnn/main/best.pt.gz --port 50062 [--device cpu] \
        [--value-model runs/gnn/main/best_value.pt.gz]
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import msgpack  # noqa: E402
import torch  # noqa: E402
import waitress  # noqa: E402
from flask import Flask, Response, request  # noqa: E402

from draftzero.gameplay import graph_net as gn  # noqa: E402
from draftzero.gameplay import graph_supervised as gs  # noqa: E402

MODEL = VALUE_MODEL = VOCAB = EDGE_VOCAB = DEVICE = DTYPE = None
MAX_BATCH = 64
LINGER_S = 0.0
Q: Queue = Queue(maxsize=8192)
STATS = {"requests": 0, "states": 0, "batches": 0, "forward_s": 0.0}
app = Flask(__name__)


def request_states(data: dict) -> list[tuple]:
    """An /evaluate request -> per-state (node_type, node_row, values, edge_child, edge_parent, edge_row)."""
    ids = np.asarray(data["indices"], np.int32)
    values = np.asarray(data["values"], np.int64)
    child = np.asarray(data["edge_child"], np.int64)
    parent = np.asarray(data["edge_parent"], np.int64)
    labels = np.asarray(data["edge_label"], np.int32)
    ntype = gn.node_types(ids)
    nrow = np.where(ntype == gn.NodeType.LEAF, VOCAB.lookup(ids), -1).astype(np.int64)
    erow = (EDGE_VOCAB.lookup(labels) + 1).astype(np.int64)
    ns = np.append(np.asarray(data["offsets"], np.int64), len(ids))
    es = np.append(np.asarray(data["edge_offsets"], np.int64), len(child))
    return [(ntype[ns[i]:ns[i + 1]], nrow[ns[i]:ns[i + 1]], values[ns[i]:ns[i + 1]], child[es[i]:es[i + 1]],
             parent[es[i]:es[i + 1]], erow[es[i]:es[i + 1]]) for i in range(len(ns) - 1)]


def collate(states: list[tuple]) -> gn.Graphs:
    starts = np.cumsum([0] + [len(s[0]) for s in states])
    cat = lambda k: torch.from_numpy(np.concatenate([s[k] for s in states]).astype(np.int64))   # noqa: E731
    shift = lambda k: torch.from_numpy(np.concatenate([s[k] + o for s, o in zip(states, starts[:-1])]).astype(np.int64))  # noqa: E731
    return gn.Graphs(cat(0), cat(1), cat(2), torch.from_numpy(starts.astype(np.int64)), shift(3), shift(4), cat(5))


class Pending:
    __slots__ = ("states", "evt", "out", "err")

    def __init__(self, states):
        self.states, self.evt, self.out, self.err = states, threading.Event(), None, None


def _floats(x: np.ndarray) -> list:
    return np.where(np.isfinite(x), x, np.nan).astype(np.float32).tolist()


def run_batch(batch: list[Pending]) -> None:
    states = [s for p in batch for s in p.states]
    t0 = time.perf_counter()
    g = collate(states).to(DEVICE)
    with torch.no_grad(), (torch.autocast(DEVICE.type, dtype=DTYPE) if DTYPE is not None else torch.autocast("cpu", enabled=False)):
        o = MODEL(g)
        v = o.value if VALUE_MODEL is None else VALUE_MODEL(g).value
    pri, tgt = o.priority.float().cpu().numpy(), o.target.float().cpu().numpy()
    use, val = o.use.float().cpu().numpy(), v.float().cpu().numpy()
    STATS["forward_s"] += time.perf_counter() - t0
    STATS["batches"] += 1
    STATS["states"] += len(states)
    off = np.asarray(g.node_offsets.cpu())
    row = 0
    for p in batch:
        p.out = []
        for _ in p.states:
            a, b = off[row], off[row + 1]
            p.out.append({"policy_priority": _floats(pri[a:b]), "policy_target": _floats(tgt[a:b]),
                          "policy_binary": use[row].tolist(), "value": float(val[row])})
            row += 1


def worker_loop() -> None:
    while True:
        batch = [Q.get()]
        n = len(batch[0].states)
        deadline = time.perf_counter() + LINGER_S
        while n < MAX_BATCH:
            try:
                left = deadline - time.perf_counter()
                p = Q.get(timeout=left) if left > 0 else Q.get(block=False)
            except Empty:
                break
            batch.append(p)
            n += len(p.states)
        try:
            run_batch(batch)
        except Exception as e:     # noqa: BLE001 - fail these requests, keep serving
            for p in batch:
                p.err = f"{type(e).__name__}: {e}"
        for p in batch:
            p.evt.set()


@app.post("/evaluate")
def evaluate():
    data = msgpack.unpackb(request.data, raw=False)
    p = Pending(request_states(data))
    STATS["requests"] += 1
    Q.put(p)
    p.evt.wait()
    if p.err is not None:
        return Response(p.err, status=500, mimetype="text/plain")
    return Response(msgpack.packb(p.out, use_bin_type=True, use_single_float=True), mimetype="application/x-msgpack")


@app.get("/healthz")
def healthz():
    return "ok", 200


@app.get("/stats")
def stats():
    return {**STATS, "states_per_batch": STATS["states"] / max(1, STATS["batches"])}


def load_value_model(path: str, device):
    """A second checkpoint for the value only: its vocabs must be the policy model's, since both read one encoding."""
    model, vocab, edge_vocab, _ = gs.load_checkpoint(path, device)
    if not (np.array_equal(vocab.ids, VOCAB.ids) and np.array_equal(edge_vocab.ids, EDGE_VOCAB.ids)):
        raise SystemExit(f"--value-model {path}: its vocab differs from --model's")
    return model.eval()


def main() -> None:
    global MODEL, VALUE_MODEL, VOCAB, EDGE_VOCAB, DEVICE, DTYPE, MAX_BATCH, LINGER_S
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a graph checkpoint (path or hf://)")
    ap.add_argument("--value-model", help="take the value from this checkpoint instead (same vocab)")
    ap.add_argument("--port", type=int, default=50062)
    ap.add_argument("--threads", type=int, default=16, help="HTTP threads")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--torch-threads", type=int, default=1, help="CPU inference threads")
    ap.add_argument("--max-batch", type=int, default=MAX_BATCH)
    ap.add_argument("--linger-ms", type=float, default=0.0, help="wait this long for more requests per batch")
    a = ap.parse_args()
    torch.set_num_threads(a.torch_threads)
    DEVICE = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else
                          "cpu" if a.device == "auto" else a.device)
    DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else None
    MAX_BATCH, LINGER_S = a.max_batch, a.linger_ms / 1000.0
    MODEL, VOCAB, EDGE_VOCAB, meta = gs.load_checkpoint(a.model, DEVICE)
    if a.value_model:
        VALUE_MODEL = load_value_model(a.value_model, DEVICE)
    threading.Thread(target=worker_loop, daemon=True).start()
    print(f"[graph_server] {a.model} on :{a.port} device={DEVICE} leaves={len(VOCAB)} edge_labels={len(EDGE_VOCAB)} "
          f"arch={meta['arch']}" + (f" value from {a.value_model}" if a.value_model else ""), flush=True)
    waitress.serve(app, host="127.0.0.1", port=a.port, threads=a.threads, _quiet=True)


if __name__ == "__main__":
    main()
