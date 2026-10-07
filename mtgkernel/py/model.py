"""DzkNet: the policy/value network over dzk-enc-v2 decisions (see docs/encoding.md), and its export to the Rust
format `dzk-net-v1` (src/nn.rs is the same function; `dzk parity` checks it).

Architecture (DeepSets, ReLU everywhere):
  object token  [E(card) ; onehot(zone, NZ) ; feat(F)] -> H -> H
  pools         per group (my hand, my battlefield, opponent's battlefield, the rest): mean ; max (empty -> 0)
  state         h = MLP([global(G) ; pools(8H)] -> S -> H)
  action token  [onehot(kind, NK) ; E(src_card) ; objH[src] ; E(tgt_card) ; objH[tgt] ; onehot(tgt_player, 3) ;
                 feat(A)] -> H -> H          (objH of a missing object = 0; card 0 embeds to 0)
  policy        logit_i = Linear(H, 1)(ReLU(Linear(2H, H)([h ; a_i]))), softmax over the decision's actions
  value         v = tanh(Linear(H, 1)(ReLU(Linear(H, H)(h)))), from the acting player's seat

Batches are flat (no padding): objects and actions of every decision in the batch are concatenated, with the
decision index of each (`obj_dec`, `act_dec`); `act_src`/`act_tgt` are global object indices, or the number of
objects in the batch for "none".
"""

import hashlib
import json
import struct
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ENC_VERSION = "dzk-enc-v2"
NET_FORMAT = "dzk-net-v1"
MAGIC = b"DZKNET01"
# encoder dimensions (src/encode.rs)
ENC_DIMS = {"V": 512, "G": 96, "F": 72, "A": 32, "NZ": 16, "NK": 40, "NP": 3}
# zone category -> pooling group (0 my hand, 1 my battlefield, 2 opponent's battlefield, 3 the rest)
ZONE_GROUP = [0, 1, 2] + [3] * 13


class DzkNet(nn.Module):
    def __init__(self, E=32, H=128, S=256):
        super().__init__()
        d = ENC_DIMS
        self.dims = dict(d, E=E, H=H, S=S)
        self.E, self.H, self.S = E, H, S
        self.emb = nn.Embedding(d["V"], E, padding_idx=0)
        self.obj1 = nn.Linear(E + d["NZ"] + d["F"], H)
        self.obj2 = nn.Linear(H, H)
        self.st1 = nn.Linear(d["G"] + 8 * H, S)
        self.st2 = nn.Linear(S, H)
        self.act1 = nn.Linear(d["NK"] + E + H + E + H + d["NP"] + d["A"], H)
        self.act2 = nn.Linear(H, H)
        self.pol1 = nn.Linear(2 * H, H)
        self.pol2 = nn.Linear(H, 1)
        self.val1 = nn.Linear(H, H)
        self.val2 = nn.Linear(H, 1)
        self.register_buffer("zone_group", torch.tensor(ZONE_GROUP, dtype=torch.long), persistent=False)

    def forward(self, b):
        """b: dict of flat tensors (see data.collate). Returns (logits [n_act], value [B])."""
        d = ENC_DIMS
        B = b["global"].shape[0]
        H = self.H
        oin = torch.cat([self.emb(b["obj_card"]), F.one_hot(b["obj_zone"], d["NZ"]).float(), b["obj_feat"]], -1)
        oh = F.relu(self.obj2(F.relu(self.obj1(oin))))
        seg = b["obj_dec"] * 4 + self.zone_group[b["obj_zone"]]
        sums = oh.new_zeros(B * 4, H).index_add_(0, seg, oh)
        cnt = oh.new_zeros(B * 4).index_add_(0, seg, torch.ones_like(seg, dtype=oh.dtype))
        mean = sums / cnt.clamp(min=1.0)[:, None]
        mx = oh.new_zeros(B * 4, H).scatter_reduce_(0, seg[:, None].expand(-1, H), oh, "amax", include_self=True)
        pools = torch.stack([mean, mx], 1).reshape(B, 8 * H)
        h = F.relu(self.st2(F.relu(self.st1(torch.cat([b["global"], pools], -1)))))
        v = torch.tanh(self.val2(F.relu(self.val1(h)))).squeeze(-1)
        ohz = torch.cat([oh, oh.new_zeros(1, H)], 0)
        ain = torch.cat([
            F.one_hot(b["act_kind"], d["NK"]).float(),
            self.emb(b["act_src_card"]),
            ohz[b["act_src"]],
            self.emb(b["act_tgt_card"]),
            ohz[b["act_tgt"]],
            F.one_hot(b["act_tgt_player"], d["NP"]).float(),
            b["act_feat"],
        ], -1)
        a = F.relu(self.act2(F.relu(self.act1(ain))))
        logit = self.pol2(F.relu(self.pol1(torch.cat([h[b["act_dec"]], a], -1)))).squeeze(-1)
        return logit, v


def segment_log_softmax(logit, seg, n_seg):
    """log softmax of `logit` within each segment (seg[i] = decision of action i)."""
    m = logit.new_full((n_seg,), float("-inf")).scatter_reduce(0, seg, logit, "amax", include_self=True)
    z = logit - m[seg]
    lse = torch.log(logit.new_zeros(n_seg).index_add_(0, seg, torch.exp(z)))
    return z - lse[seg]


# ---------------------------------------------------------------------------------------------- checkpoints

def save_checkpoint(path, model, extra=None):
    torch.save({"encoder": ENC_VERSION, "dims": model.dims, "state_dict": model.state_dict(), **(extra or {})}, path)


def load_checkpoint(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    if ck.get("encoder") != ENC_VERSION:
        raise ValueError(f"{path}: encoder {ck.get('encoder')!r} is not {ENC_VERSION!r}")
    dims = ck["dims"]
    for k, v in ENC_DIMS.items():
        if dims.get(k) != v:
            raise ValueError(f"{path}: dims.{k} = {dims.get(k)}, the encoder has {v}")
    model = DzkNet(E=dims["E"], H=dims["H"], S=dims["S"])
    model.load_state_dict(ck["state_dict"])
    return model, ck


# ---------------------------------------------------------------------------------------------- dzk-net-v1

TENSOR_ORDER = ["emb.weight", "obj1.weight", "obj1.bias", "obj2.weight", "obj2.bias", "st1.weight", "st1.bias",
                "st2.weight", "st2.bias", "act1.weight", "act1.bias", "act2.weight", "act2.bias", "pol1.weight",
                "pol1.bias", "pol2.weight", "pol2.bias", "val1.weight", "val1.bias", "val2.weight", "val2.bias"]


def export_dzkn(model, path, meta=None):
    """Write the Rust format: magic, u64 header length, JSON header, f32 LE blob. Returns the blob sha256."""
    sd = model.state_dict()
    tensors, chunks, off = [], [], 0
    for name in TENSOR_ORDER:
        t = sd[name].detach().to("cpu", torch.float32).contiguous()
        tensors.append({"name": name, "shape": list(t.shape), "offset": off, "numel": t.numel()})
        chunks.append(t.numpy().astype("<f4").tobytes())
        off += t.numel()
    blob = b"".join(chunks)
    digest = hashlib.sha256(blob).hexdigest()
    header = {"format": NET_FORMAT, "encoder": ENC_VERSION, "dims": model.dims, "tensors": tensors,
              "blob_sha256": digest, "meta": dict(meta or {}, exported_unix=int(time.time()))}
    h = json.dumps(header, sort_keys=True).encode()
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<Q", len(h)) + h + blob)
    return digest


def load_dzkn(path):
    """Read a dzk-net-v1 file back into a DzkNet (checks the format, encoder and hash)."""
    import numpy as np
    raw = open(path, "rb").read()
    if raw[:8] != MAGIC:
        raise ValueError(f"{path}: not a dzk-net-v1 file")
    (hl,) = struct.unpack_from("<Q", raw, 8)
    header = json.loads(raw[16:16 + hl])
    blob = raw[16 + hl:]
    if header["format"] != NET_FORMAT or header["encoder"] != ENC_VERSION:
        raise ValueError(f"{path}: {header['format']} / {header['encoder']}")
    if hashlib.sha256(blob).hexdigest() != header["blob_sha256"]:
        raise ValueError(f"{path}: blob hash mismatch")
    dims = header["dims"]
    model = DzkNet(E=dims["E"], H=dims["H"], S=dims["S"])
    floats = np.frombuffer(blob, dtype="<f4")
    sd = {}
    for t in header["tensors"]:
        sd[t["name"]] = torch.from_numpy(floats[t["offset"]:t["offset"] + t["numel"]].copy()).reshape(t["shape"])
    model.load_state_dict(sd)
    return model, header
