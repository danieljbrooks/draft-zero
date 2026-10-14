"""Write an `mlp` checkpoint as a .dzgw file, the weights gorge's in-process Go network reads
(gorge/cmd/dzgorge/gonet.go, the policy key gonet=PATH).

  python -m dzg.export OUT/best.pt OUT/best.dzgw

Layout (all little-endian):

  bytes 0..7    magic b"DZGW0001"
  bytes 8..15   u64 H, the header's length in bytes
  H bytes       the header, UTF-8 JSON (below)
  zero padding to a multiple of 64 bytes from the start of the file: the data starts there
  data          each tensor as float32, C order, at its `offset` from the data start (64-byte aligned)

The header:

  {"format": "dzgw-1", "arch": "mlp", "config": {...the full models.DEFAULTS config...},
   "consts": {"DENSE_W": 68, "RAW_W": 68, "OPT_DENSE_W": 24, "TABLE_ROWS": 16384, "SLOT_ROWS": 128, "GROUPS": 4},
   "layernorm_eps": {"blocks.0.norm": 1e-05, ..., "norm": 1e-05, "opt_norm": 1e-05},
   "tensors": [{"name": "table.weight", "shape": [16384, 128], "offset": 0, "nbytes": 8388608}, ...],
   "source": "<checkpoint path>", "meta": {...the checkpoint's meta, when it is JSON...}}

Tensor names are the PyTorch state_dict's (a Linear's weight is [out, in]). Eval-mode semantics only: dropout
is the identity, and the LayerNorms' eps are the modules'. Only the mlp has a Go forward pass; other archs are
refused.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

from . import models as M
from . import packfmt as pf

MAGIC = b"DZGW0001"
FORMAT = "dzgw-1"
ALIGN = 64
SUPPORTED = ("mlp",)


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def _json_or_none(x):
    try:
        json.dumps(x)
        return x
    except (TypeError, ValueError):
        return None


def export(model: M._Net, out, source: str = "", meta: dict | None = None) -> dict:
    """Write model (an MLPNet) to out; returns the header."""
    if model.arch not in SUPPORTED:
        raise ValueError(f"arch {model.arch!r} has no Go forward pass: dzg.export writes only {list(SUPPORTED)} "
                         f"(gonet.go implements the mlp alone; a transformer or gnn needs attention in Go first)")
    eps = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.LayerNorm):
            if not mod.elementwise_affine or mod.weight is None or mod.bias is None:
                raise ValueError(f"LayerNorm {name} has no affine weight and bias")
            if tuple(mod.normalized_shape) != (mod.weight.shape[0],):
                raise ValueError(f"LayerNorm {name}: normalized_shape {mod.normalized_shape}")
            eps[name] = float(mod.eps)
    sd = model.state_dict()
    tensors, blobs, off = [], [], 0
    for name, t in sd.items():
        a = np.ascontiguousarray(t.detach().cpu().to(torch.float32).numpy(), dtype="<f4")
        tensors.append({"name": name, "shape": list(a.shape), "offset": off, "nbytes": a.nbytes})
        blobs.append((off, a))
        off = _align(off + a.nbytes)
    header = {"format": FORMAT, "arch": model.arch, "config": dict(model.config),
              "consts": {"DENSE_W": pf.DENSE_W, "RAW_W": pf.RAW_W, "OPT_DENSE_W": pf.OPT_DENSE_W,
                         "TABLE_ROWS": pf.TABLE_ROWS, "SLOT_ROWS": pf.SLOT_ROWS, "GROUPS": pf.GROUPS},
              "layernorm_eps": eps, "tensors": tensors, "source": str(source),
              "meta": _json_or_none(meta or {})}
    hb = json.dumps(header, sort_keys=False).encode("utf-8")
    data_start = _align(16 + len(hb))
    with open(out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(b"\0" * (data_start - 16 - len(hb)))
        pos = 0
        for o, a in blobs:
            f.write(b"\0" * (o - pos))
            f.write(a.tobytes())
            pos = o + a.nbytes
    return header


def read(path) -> tuple[dict, dict]:
    """(header, {name: numpy array}) of a .dzgw file (tests, checks)."""
    b = Path(path).read_bytes()
    if b[:8] != MAGIC:
        raise ValueError(f"{path}: not a .dzgw file")
    (h,) = struct.unpack("<Q", b[8:16])
    header = json.loads(b[16:16 + h])
    start = _align(16 + h)
    out = {}
    for t in header["tensors"]:
        a = np.frombuffer(b, "<f4", count=t["nbytes"] // 4, offset=start + t["offset"])
        out[t["name"]] = a.reshape(t["shape"])
    return header, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dzg.export", description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpt", help="a dzg checkpoint (models.save), arch mlp")
    ap.add_argument("out", help="the .dzgw file to write")
    a = ap.parse_args(argv)
    model, ck = M.load(a.ckpt)
    if model.arch not in SUPPORTED:
        print(f"dzg.export: {a.ckpt} is a {model.arch!r} network; only {list(SUPPORTED)} can be exported "
              f"(gonet.go has no {model.arch} forward pass)", file=sys.stderr)
        return 2
    h = export(model, a.out, source=a.ckpt, meta=ck.get("meta"))
    n = sum(t["nbytes"] for t in h["tensors"]) // 4
    print(f"dzg.export: {a.out}: {h['arch']} {h['config']}, {len(h['tensors'])} tensors, {n:,} floats")
    return 0


if __name__ == "__main__":
    sys.exit(main())
