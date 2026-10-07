"""Reader for dzk-data-v1 training data (`dzk play --data-out DIR`; format in src/data.rs) and the flat mini-batches
DzkNet takes.

    games = list(iter_dir("runs/x/data"))                  # [(header, [Decision, ...]), ...]  (one game at a time)
    P = load_packed(["runs/x/data"], sources={"search"})   # every decision, packed into flat arrays
    for batch in P.batches(idx, 256, shuffle=True, seed=0): ...

A shard is a sequence of gzip members (one per game); a torn last member (a killed run) is dropped. A game that
appears twice (overlapping runs on one data directory) is kept once, by its `game_key`.

`Packed` holds every decision in flat arrays with per-decision offsets (no Python object per decision); the float
features are stored as float16 by default (they are clamped to [-8, 8] and mostly 0/1 flags or small fractions; the
rounding is below 5e-4 relative) and widened to float32 per batch. `Decision` and `collate` (one Python object per
decision) remain for small jobs such as py/parity.py.
"""

import glob
import json
import os
import struct
import zlib
from dataclasses import dataclass

import numpy as np

ENC_VERSION = "dzk-enc-v2"
DATA_FORMAT = "dzk-data-v1"
G, F, A = 96, 72, 32
FAMILY_OFFSET = 56  # global one-hot of the decision family (src/encode.rs FAMILIES)
FAMILIES = ["priority", "target", "attack", "block", "damage", "mulligan", "bottom", "trigger_order", "legend", "other"]
SOURCES = ["search", "net", "random", "rule", "other"]
FLAG_TRUNCATED, FLAG_INVALID, FLAG_DRAW = 1, 2, 4
_REC = struct.Struct("<IIBBbBHHIIf")


@dataclass
class Decision:
    seat: int
    source: int
    z: int
    flags: int
    turn: int
    decision: int
    chosen: int
    root_q: float
    glob: np.ndarray       # f32[G]
    obj_card: np.ndarray   # u16[n_obj] -> int64 on collate
    obj_zone: np.ndarray   # u8[n_obj]
    obj_feat: np.ndarray   # f32[n_obj, F]
    act_kind: np.ndarray   # u8[n_act]
    act_src: np.ndarray    # i16[n_act]
    act_src_card: np.ndarray
    act_tgt: np.ndarray
    act_tgt_card: np.ndarray
    act_tgt_player: np.ndarray
    act_feat: np.ndarray   # f32[n_act, A]
    target: np.ndarray     # f32[n_act]
    game: int = 0          # index of the game in the load order

    @property
    def n_act(self):
        return len(self.act_kind)

    @property
    def value_ok(self):
        """The result is a valid value target (not halted/error, not truncated)."""
        return not (self.flags & (FLAG_INVALID | FLAG_TRUNCATED))

    def copy(self):
        """A copy owning its arrays (a parsed Decision's arrays are views into its member's buffer)."""
        import dataclasses
        return dataclasses.replace(self, **{f.name: np.array(getattr(self, f.name)) for f in dataclasses.fields(self)
                                            if isinstance(getattr(self, f.name), np.ndarray)})

    @property
    def family(self):
        return int(np.argmax(self.glob[FAMILY_OFFSET:FAMILY_OFFSET + len(FAMILIES)]))


def iter_members(raw):
    """Decompressed gzip members of a shard's bytes, in order (linear time: fed in 1 MB chunks). Yields
    (member bytes, None) and, if the last member is torn, finally (None, "torn")."""
    mv = memoryview(raw)
    pos, CH = 0, 1 << 20
    while pos < len(raw):
        d = zlib.decompressobj(wbits=31)
        parts, p = [], pos
        try:
            while not d.eof and p < len(raw):
                chunk = mv[p:p + CH]
                p += len(chunk)
                parts.append(d.decompress(chunk))
        except zlib.error:
            yield None, "torn"
            return
        if not d.eof:
            yield None, "torn"
            return
        pos = p - len(d.unused_data)
        yield b"".join(parts), None


def parse_member(buf):
    if buf[:4] != b"DZKG":
        raise ValueError("bad member magic")
    version, hl = struct.unpack_from("<II", buf, 4)
    if version != 1:
        raise ValueError(f"member version {version}")
    header = json.loads(buf[12:12 + hl])
    if header.get("encoder") != ENC_VERSION:
        raise ValueError(f"encoder {header.get('encoder')!r} is not {ENC_VERSION!r}")
    off = 12 + hl
    recs = []
    fb = np.frombuffer
    for _ in range(header["records"]):
        n_obj, n_act, seat, source, z, flags, turn, _pad, decision, chosen, root_q = _REC.unpack_from(buf, off)
        off += _REC.size

        def take(dtype, n):
            nonlocal off
            a = fb(buf, dtype=dtype, count=n, offset=off)
            off += a.nbytes
            return a

        glob_ = take("<f4", G)
        obj_card = take("<u2", n_obj)
        obj_zone = take("u1", n_obj)
        obj_feat = take("<f4", n_obj * F).reshape(n_obj, F)
        act_kind = take("u1", n_act)
        act_src = take("<i2", n_act)
        act_src_card = take("<u2", n_act)
        act_tgt = take("<i2", n_act)
        act_tgt_card = take("<u2", n_act)
        act_tgt_player = take("u1", n_act)
        act_feat = take("<f4", n_act * A).reshape(n_act, A)
        target = take("<f4", n_act)
        recs.append(Decision(seat, source, z, flags, turn, decision, chosen, root_q, glob_, obj_card, obj_zone,
                             obj_feat, act_kind, act_src, act_src_card, act_tgt, act_tgt_card, act_tgt_player,
                             act_feat, target))
    return header, recs


def record_bytes(d):
    """One record in the shard layout (the inverse of parse_member's per-record read)."""
    n_obj, n_act = len(d.obj_card), d.n_act
    parts = [_REC.pack(n_obj, n_act, d.seat, d.source, d.z, d.flags, d.turn, 0, d.decision, d.chosen, d.root_q)]
    for a, dt in [(d.glob, "<f4"), (d.obj_card, "<u2"), (d.obj_zone, "u1"), (d.obj_feat, "<f4"), (d.act_kind, "u1"),
                  (d.act_src, "<i2"), (d.act_src_card, "<u2"), (d.act_tgt, "<i2"), (d.act_tgt_card, "<u2"),
                  (d.act_tgt_player, "u1"), (d.act_feat, "<f4"), (d.target, "<f4")]:
        parts.append(np.ascontiguousarray(a, dtype=dt).tobytes())
    return b"".join(parts)


def member_bytes(header, decs):
    """An uncompressed game member holding `decs` (header copied, its record count set)."""
    h = json.dumps(dict(header, records=len(decs))).encode()
    return b"DZKG" + struct.pack("<II", 1, len(h)) + h + b"".join(record_bytes(d) for d in decs)


def iter_shard(path):
    """(header, decisions) per complete game member of one shard (a torn last member is skipped)."""
    with open(path, "rb") as f:
        raw = f.read()
    for m, torn in iter_members(raw):
        if m is not None:
            yield parse_member(m)


def shard_paths(d):
    if os.path.isfile(d):
        return [d]
    meta = os.path.join(d, "meta.json")
    if os.path.exists(meta):
        m = json.load(open(meta))
        if m.get("encoder") != ENC_VERSION:
            raise ValueError(f"{meta}: encoder {m.get('encoder')!r} is not {ENC_VERSION!r}")
    return sorted(glob.glob(os.path.join(d, "*.dzd.gz")))


def iter_dir(*dirs):
    """(header, decisions) per game over data directories (or shard files), each game once (by game_key)."""
    seen = set()
    for d in dirs:
        for p in shard_paths(d):
            for header, recs in iter_shard(p):
                k = header.get("game_key")
                if k in seen:
                    continue
                seen.add(k)
                yield header, recs


def search_budget(spec):
    """N of a uniform-prior search bot spec (`mcts:N...`, `heuristic@N...`), else None (pmcts has priors)."""
    head = (spec or "").split(",")[0].strip()
    for p in ("mcts:", "heuristic@"):
        if head.startswith(p):
            try:
                return int(head[len(p):])
            except ValueError:
                return None
    return None


def value_targets(z, flags, root_q, how):
    """(target, weight) arrays of the value head per --value-target: `z` (the game's result), `rootq` (the
    search's root value at the decision) or `mix:L` (L z + (1 - L) root_q). A missing part falls back to the other;
    neither (a truncated or halted game's decision without a search value) has weight 0."""
    z_ok = (flags & (FLAG_INVALID | FLAG_TRUNCATED)) == 0
    q_ok = np.isfinite(root_q)
    q = np.where(q_ok, root_q, 0.0)
    zf = z.astype(np.float32)
    if how == "z":
        return zf, z_ok.astype(np.float32)
    if how == "rootq":
        lam = 0.0
    elif how.startswith("mix:"):
        lam = float(how[4:])
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"bad value target {how!r}")
    else:
        raise ValueError(f"bad value target {how!r}")
    t = np.where(z_ok & q_ok, lam * zf + (1 - lam) * q, np.where(q_ok, q, zf))
    return t.astype(np.float32), (z_ok | q_ok).astype(np.float32)


def _ranges(starts, lens):
    """Concatenated aranges [starts[i], starts[i] + lens[i])."""
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, np.int64)
    return np.repeat(starts - (np.cumsum(lens) - lens), lens) + np.arange(total, dtype=np.int64)


class Packed:
    """Every decision of a load in flat arrays. Per decision i: objects obj_off[i]:obj_off[i+1], actions
    act_off[i]:act_off[i+1]; scalars seat, source, z, flags, turn, chosen, root_q, game (index into `games`),
    pw (policy weight), vt / vw (value target / weight; see set_value_target)."""

    SCALARS = [("seat", np.uint8), ("source", np.uint8), ("z", np.int8), ("flags", np.uint8), ("turn", np.uint16),
               ("chosen", np.int64), ("root_q", np.float32), ("game", np.int32), ("pw", np.float32)]
    OBJ = [("obj_card", np.uint16), ("obj_zone", np.uint8)]
    ACT = [("act_kind", np.uint8), ("act_src", np.int16), ("act_src_card", np.uint16), ("act_tgt", np.int16),
           ("act_tgt_card", np.uint16), ("act_tgt_player", np.uint8)]

    def __init__(self, feat_dtype=np.float16):
        self.feat_dtype = np.dtype(feat_dtype)
        self.games = []
        self._parts = {k: [] for k in ["glob", "obj_feat", "act_feat", "target", "n_obj", "n_act"]
                       + [k for k, _ in self.SCALARS + self.OBJ + self.ACT]}
        self.n = 0

    def add_game(self, header, decs, pw=None):
        """Append one game's decisions (pw: their policy weights, default 1)."""
        gi = len(self.games)
        self.games.append(header)
        if not decs:
            return
        p, fd = self._parts, self.feat_dtype
        p["glob"].append(np.stack([d.glob for d in decs]).astype(fd))
        p["obj_feat"].append(np.concatenate([d.obj_feat for d in decs]).astype(fd).reshape(-1, F))
        p["act_feat"].append(np.concatenate([d.act_feat for d in decs]).astype(fd).reshape(-1, A))
        p["target"].append(np.concatenate([d.target for d in decs]).astype(np.float32))
        p["n_obj"].append(np.array([len(d.obj_card) for d in decs], np.int64))
        p["n_act"].append(np.array([d.n_act for d in decs], np.int64))
        for k, dt in self.OBJ + self.ACT:
            p[k].append(np.concatenate([getattr(d, k) for d in decs]).astype(dt))
        vals = {"game": [gi] * len(decs), "pw": pw if pw is not None else [1.0] * len(decs)}
        for k, dt in self.SCALARS:
            p[k].append(np.array(vals[k] if k in vals else [getattr(d, k) for d in decs], dtype=dt))
        self.n += len(decs)

    def finish(self):
        """Concatenate the parts (each field's parts are freed as soon as it is built)."""
        p = self._parts
        empty = {"glob": (0, G), "obj_feat": (0, F), "act_feat": (0, A)}
        for k in list(p):
            parts = p.pop(k)
            if parts:
                v = np.concatenate(parts)
            else:
                v = np.zeros(empty.get(k, (0,)), self.feat_dtype if k in empty else np.float32)
            setattr(self, k, v)
        self.obj_off = np.concatenate([[0], np.cumsum(self.n_obj)]).astype(np.int64)
        self.act_off = np.concatenate([[0], np.cumsum(self.n_act)]).astype(np.int64)
        self.act_dec_all = np.repeat(np.arange(self.n), self.n_act)
        self.set_value_target("z")
        return self

    def nbytes(self):
        return sum(v.nbytes for v in vars(self).values() if isinstance(v, np.ndarray))

    # ---- targets
    def set_value_target(self, how):
        self.vt, self.vw = value_targets(self.z, self.flags, self.root_q, how)

    def set_policy_target(self, how):
        """Replace the policy targets (in place): visits (the stored target), top (one-hot of the target's largest
        entry, ties to the lower index: the search's most-visited move, never an exploration sample), chosen
        (one-hot of the played move; in pmcts self-play that includes sampled and noised moves), or sharp:T
        (target^(1/T), renormalized)."""
        if how == "visits":
            return
        t = self.target.astype(np.float64)
        if how == "top":
            first = segment_first_argmax(t, self.act_dec_all, self.n)
            t = np.zeros_like(t)
            t[first] = 1.0
        elif how == "chosen":
            t = np.zeros_like(t)
            t[self.act_off[:-1] + self.chosen] = 1.0
        elif how.startswith("sharp:"):
            t = t ** (1.0 / float(how[6:]))
            s = np.add.reduceat(t, self.act_off[:-1]) if self.n else np.zeros(0)
            t = t / np.repeat(s, self.n_act)
        else:
            raise ValueError(f"bad policy target {how!r}")
        self.target = t.astype(np.float32)

    # ---- batches
    def batch(self, idx, device="cpu"):
        """Flat batch tensors for DzkNet.forward (float32) plus targets, for the decisions `idx`."""
        import torch
        idx = np.asarray(idx, np.int64)
        B = len(idx)
        no, na = self.n_obj[idx], self.n_act[idx]
        oi = _ranges(self.obj_off[idx], no)
        ai = _ranges(self.act_off[idx], na)
        total_obj = len(oi)
        obj_boff = np.cumsum(no) - no
        act_boff = np.cumsum(na) - na
        act_dec = np.repeat(np.arange(B), na)

        def link(local):
            local = local.astype(np.int64)
            return np.where(local >= 0, local + obj_boff[act_dec], total_obj)

        b = {
            "global": self.glob[idx].astype(np.float32),
            "obj_card": self.obj_card[oi].astype(np.int64),
            "obj_zone": self.obj_zone[oi].astype(np.int64),
            "obj_feat": self.obj_feat[oi].astype(np.float32),
            "obj_dec": np.repeat(np.arange(B), no),
            "act_kind": self.act_kind[ai].astype(np.int64),
            "act_src": link(self.act_src[ai]),
            "act_src_card": self.act_src_card[ai].astype(np.int64),
            "act_tgt": link(self.act_tgt[ai]),
            "act_tgt_card": self.act_tgt_card[ai].astype(np.int64),
            "act_tgt_player": self.act_tgt_player[ai].astype(np.int64),
            "act_feat": self.act_feat[ai].astype(np.float32),
            "act_dec": act_dec,
            "target": self.target[ai],
            "chosen": self.chosen[idx] + act_boff,
            "z": self.z[idx].astype(np.float32),
            "value_ok": ((self.flags[idx] & (FLAG_INVALID | FLAG_TRUNCATED)) == 0).astype(np.float32),
            "vt": self.vt[idx],
            "vw": self.vw[idx],
            "pw": self.pw[idx],
        }
        return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in b.items()}

    def batches(self, idx, batch_size, shuffle=False, seed=0, device="cpu"):
        idx = np.array(idx, np.int64)
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)
        for i in range(0, len(idx), batch_size):
            yield self.batch(idx[i:i + batch_size], device)


def segment_first_argmax(x, seg, n_seg):
    """Per segment, the global index of its largest entry (ties to the lower index)."""
    m = np.full(n_seg, -np.inf)
    np.maximum.at(m, seg, x)
    pos = np.where(x >= m[seg], np.arange(len(x)), len(x))
    first = np.full(n_seg, len(x))
    np.minimum.at(first, seg, pos)
    return first


def load_packed(dirs, sources=("search",), max_actions=512, include_invalid=True, flat_k_frac=0.25,
                feat_dtype=np.float16):
    """Every decision of the given sources (names in SOURCES), skipping decisions with more than `max_actions`
    actions, packed. `flat_k_frac`: a decision searched by a uniform-prior `mcts:N` with more than flat_k_frac * N
    actions gets policy weight 0 (its visit target is flat or "index 0" by the search's tie-break; its value
    target is kept); 0 disables. Returns (Packed, stats)."""
    want = {SOURCES.index(s) for s in sources}
    P = Packed(feat_dtype)
    stats = {"games": 0, "records": 0, "kept": 0, "skipped_source": 0, "skipped_large": 0, "skipped_invalid": 0,
             "flat_policy_zeroed": 0}
    for header, recs in iter_dir(*dirs):
        stats["games"] += 1
        budgets = [search_budget(header.get("botA")), search_budget(header.get("botB"))]
        keep, pw = [], []
        for r in recs:
            stats["records"] += 1
            if r.source not in want:
                stats["skipped_source"] += 1
                continue
            if r.n_act > max_actions:
                stats["skipped_large"] += 1
                continue
            if not include_invalid and (r.flags & FLAG_INVALID):
                stats["skipped_invalid"] += 1
                continue
            w = 1.0
            n = budgets[r.seat] if r.seat < 2 else None
            if flat_k_frac and r.source == 0 and n and r.n_act > flat_k_frac * n:
                w = 0.0
                stats["flat_policy_zeroed"] += 1
            keep.append(r)
            pw.append(w)
        P.add_game(header, keep, pw)
        stats["kept"] += len(keep)
    P.finish()
    return P, stats


def load_decisions(dirs, sources=("search",), max_actions=512, include_invalid=True):
    """Every decision of the given sources as Decision objects (small jobs; training uses load_packed). Returns
    (decisions, games headers, stats)."""
    want = {SOURCES.index(s) for s in sources}
    decs, games = [], []
    stats = {"games": 0, "records": 0, "kept": 0, "skipped_source": 0, "skipped_large": 0, "skipped_invalid": 0}
    for header, recs in iter_dir(*dirs):
        gi = len(games)
        games.append(header)
        stats["games"] += 1
        for r in recs:
            stats["records"] += 1
            if r.source not in want:
                stats["skipped_source"] += 1
                continue
            if r.n_act > max_actions:
                stats["skipped_large"] += 1
                continue
            if not include_invalid and (r.flags & FLAG_INVALID):
                stats["skipped_invalid"] += 1
                continue
            r.game = gi
            decs.append(r)
            stats["kept"] += 1
    return decs, games, stats


def collate(decs, device="cpu"):
    """Flat batch tensors for DzkNet.forward from Decision objects (float32, as stored), plus targets."""
    import torch
    n_obj = [len(d.obj_card) for d in decs]
    n_act = [d.n_act for d in decs]
    obj_off = np.concatenate([[0], np.cumsum(n_obj)[:-1]]).astype(np.int64)
    total_obj = int(sum(n_obj))

    def link(idx, off):
        idx = idx.astype(np.int64)
        return np.where(idx >= 0, idx + off, total_obj)

    B = len(decs)
    b = {
        "global": np.stack([d.glob for d in decs]).astype(np.float32),
        "obj_card": np.concatenate([d.obj_card for d in decs]).astype(np.int64),
        "obj_zone": np.concatenate([d.obj_zone for d in decs]).astype(np.int64),
        "obj_feat": np.concatenate([d.obj_feat for d in decs]).astype(np.float32).reshape(-1, F),
        "obj_dec": np.repeat(np.arange(B), n_obj).astype(np.int64),
        "act_kind": np.concatenate([d.act_kind for d in decs]).astype(np.int64),
        "act_src": np.concatenate([link(d.act_src, o) for d, o in zip(decs, obj_off)]),
        "act_src_card": np.concatenate([d.act_src_card for d in decs]).astype(np.int64),
        "act_tgt": np.concatenate([link(d.act_tgt, o) for d, o in zip(decs, obj_off)]),
        "act_tgt_card": np.concatenate([d.act_tgt_card for d in decs]).astype(np.int64),
        "act_tgt_player": np.concatenate([d.act_tgt_player for d in decs]).astype(np.int64),
        "act_feat": np.concatenate([d.act_feat for d in decs]).astype(np.float32).reshape(-1, A),
        "act_dec": np.repeat(np.arange(B), n_act).astype(np.int64),
        "target": np.concatenate([d.target for d in decs]).astype(np.float32),
        "chosen": (np.array([d.chosen for d in decs], dtype=np.int64)
                   + np.concatenate([[0], np.cumsum(n_act)[:-1]]).astype(np.int64)),
        "z": np.array([d.z for d in decs], dtype=np.float32),
        "value_ok": np.array([d.value_ok for d in decs], dtype=np.float32),
    }
    return {k: torch.from_numpy(v).to(device) for k, v in b.items()}


def batches(decs, batch_size, shuffle=False, seed=0, device="cpu"):
    idx = np.arange(len(decs))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield collate([decs[j] for j in idx[i:i + batch_size]], device)
