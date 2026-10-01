"""A lean inference server for the search benchmark and experiment #4's games, with MageZero v0.2's
model, batching and protocol (msgpack POST /evaluate, GET /healthz).

The benchmark searches with priors off, as experiment #2 did, so only the value head is read.
MageZero's server.py returns all four 1,024-wide policy heads per state as Python lists and logs
three lines per request, which caps one replica at a few hundred requests a second. This one runs
the same forward pass (fp16 autocast, batches of up to 64 across requests) and returns the value
plus the two-way binary head, which RemoteModelEvaluator requires to be present.

--policy also returns the three 1,024-wide policy heads (player, opponent, target), as MageZero's
server does, for searches that read the priors (docs/017's imitation prior, play.py's network
bots). The heads come from the same forward pass and go out as float32 (the client reads either
width; fp16 outputs are exact in float32), with no per-request logging. --linger-ms waits that long
for more requests before running a batch.

    MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv python tools/search_bench/value_server.py \\
        --model models/FDN_exp2/ver1/gen18.pt.gz --port 50052 --threads 16 [--policy]
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from queue import Empty, Queue

import magezero

sys.path.insert(0, list(magezero.__path__)[0])

import msgpack  # noqa: E402
import torch  # noqa: E402
import waitress  # noqa: E402
from flask import Flask, Response, request  # noqa: E402
from model import GLOBAL_MAX, NetTransformer, load_model, require_policy_width  # noqa: E402
from vocab import FeatureVocab  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)
MAX_BATCH = 64
MODEL = None
VOCAB = None
POLICY = False                # --policy: return the policy heads too
LINGER_S = 0.0                # --linger-ms
Q: Queue = Queue(maxsize=8192)
app = Flask(__name__)


class Pending:
    __slots__ = ("idx", "off", "nb", "evt", "out")

    def __init__(self, indices, offsets):
        rows, new_off = VOCAB.map_bags(indices, offsets or [0])
        self.idx = torch.as_tensor(rows, dtype=torch.long)
        self.off = torch.as_tensor(new_off, dtype=torch.long)
        self.nb = len(new_off)
        self.evt = threading.Event()
        self.out = None


def worker_loop() -> None:
    while True:
        batch = [Q.get()]
        deadline = time.perf_counter() + LINGER_S
        while len(batch) < MAX_BATCH:
            try:
                wait = deadline - time.perf_counter()
                batch.append(Q.get(timeout=wait) if wait > 0 else Q.get(block=False))
            except Empty:
                break
        try:
            run_batch(batch)
        except Exception as e:  # noqa: BLE001 - never let the batching thread die
            print(f"[value_server] batch of {len(batch)} failed: {e!r}", flush=True)
            for p in batch:
                if p.out is None:
                    p.out = _fallback(p.nb)
                    p.evt.set()


def _fallback(nb: int):
    """Neutral outputs (zero logits: uniform priors) for a state the model cannot run on."""
    o = {"value": 0.0, "policy_binary": [0.0, 0.0]}
    if POLICY:
        z = [0.0] * POLICY_WIDTH
        o.update(policy_player=z, policy_opponent=z, policy_target=z)
    return o if nb == 1 else [dict(o) for _ in range(nb)]


POLICY_WIDTH = 1024


def run_batch(batch: list) -> None:
    empty = [p for p in batch if len(p.idx) == 0]
    for p in empty:  # no known feature: the model cannot run on it (MageZero's server crashes)
        p.out = _fallback(p.nb)
        p.evt.set()
    batch = [p for p in batch if len(p.idx) > 0]
    if not batch:
        return
    if len(batch) == 1:
        idx, off = batch[0].idx, batch[0].off
    else:
        idx = torch.cat([p.idx for p in batch])
        lens = torch.tensor([len(p.idx) for p in batch])
        starts = torch.cat([torch.tensor([0]), lens.cumsum(0)[:-1]])
        off = torch.cat([p.off for p in batch]) + torch.repeat_interleave(starts, torch.tensor([p.nb for p in batch]))
    with torch.no_grad(), torch.amp.autocast("cuda"):
        pa, pb, tgt, bin2, val = MODEL(idx.to(DEVICE, non_blocking=True), off.to(DEVICE, non_blocking=True))
    val = val.float().cpu().reshape(-1).tolist()
    bin2 = bin2.float().cpu().tolist()
    heads = torch.stack([pa, pb, tgt], 1).float().cpu().tolist() if POLICY else None
    row = 0
    for p in batch:
        outs = [{"value": val[row + i], "policy_binary": bin2[row + i]} for i in range(p.nb)]
        if POLICY:
            for i, o in enumerate(outs):
                h = heads[row + i]
                o.update(policy_player=h[0], policy_opponent=h[1], policy_target=h[2])
        p.out = outs[0] if p.nb == 1 else outs
        row += p.nb
        p.evt.set()


@app.post("/evaluate")
def evaluate():
    data = msgpack.unpackb(request.data, raw=False)
    p = Pending(data.get("indices", []), data.get("offsets", []))
    Q.put(p)
    p.evt.wait()
    return Response(msgpack.packb(p.out, use_bin_type=True, use_single_float=POLICY),
                    mimetype="application/x-msgpack")


@app.get("/healthz")
def healthz():
    return "ok", 200


def main() -> None:
    global MODEL, VOCAB, POLICY, LINGER_S, MAX_BATCH, POLICY_WIDTH
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=50052)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--policy", action="store_true", help="also return the policy heads (MageZero's server's fields)")
    ap.add_argument("--linger-ms", type=float, default=0.0, help="wait this long for more requests per batch")
    ap.add_argument("--max-batch", type=int, default=MAX_BATCH)
    a = ap.parse_args()
    POLICY, LINGER_S, MAX_BATCH = a.policy, a.linger_ms / 1000.0, a.max_batch
    ckpt = load_model(a.model)
    VOCAB = FeatureVocab.from_state_dict(ckpt["feature_vocab"])
    VOCAB.require_encoding(GLOBAL_MAX)
    require_policy_width(ckpt["model_state_dict"], a.model)
    MODEL = NetTransformer(len(VOCAB)).to(DEVICE).eval()
    MODEL.load_state_dict(ckpt["model_state_dict"])
    POLICY_WIDTH = int(ckpt["model_state_dict"]["player_priority_head.2.weight"].shape[0]) \
        if "player_priority_head.2.weight" in ckpt["model_state_dict"] else POLICY_WIDTH
    threading.Thread(target=worker_loop, daemon=True).start()
    print(f"[value_server] {a.model} on :{a.port} device={DEVICE} threads={a.threads} policy={POLICY} "
          f"linger={a.linger_ms} ms max_batch={MAX_BATCH}", flush=True)
    waitress.serve(app, host="127.0.0.1", port=a.port, threads=a.threads, _quiet=True)


if __name__ == "__main__":
    main()
