"""The packed columnar format (SPEC.md §2-3): shard directories, wire messages, synthetic data.

A *pack* is a dict of numpy arrays named as in SPEC.md §2. Offsets are CSR and start at 0. A pack
holds `B` states; a training pack also carries the per-state targets and the candidates.
"""
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np

DENSE_W = 68          # state dense width (gorge DenseWidth)
RAW_W = 68            # card raw width (gorge EntityRawWidth)
OPT_DENSE_W = 24      # option dense width (gorge OptionDenseWidth)
TABLE_ROWS = 16384    # shared hashed table (gorge TableRows)
SLOT_ROWS = 128       # option slot table (gorge OptionSlotWidth)
GROUPS = 4            # card groups: my battlefield, opponent's battlefield, my hand, stack

REQ_MAGIC = 0x31425A47    # "GZB1"
RESP_MAGIC = 0x31525A47   # "GZR1"
FORMAT = "dzg-pack-1"
HOLDOUT_MOD = 20          # holdout = games with crc32(id) % 20 == 0

F32, I32, U16, U8 = np.dtype("<f4"), np.dtype("<i4"), np.dtype("<u2"), np.dtype("u1")

# (name, dtype, length symbol, row width). Length symbols: B, B+1, nsp, nc, nc+1, ncr, no, no+1, nos, noh.
# The order is the wire order (SPEC §3), after `want`.
STATE_ARRAYS = [
    ("dense", F32, "B", DENSE_W),
    ("sp_off", I32, "B+1", 1), ("sp_row", U16, "nsp", 1), ("sp_val", F32, "nsp", 1),
    ("card_off", I32, "B+1", 1), ("card_group", U8, "nc", 1), ("card_raw", F32, "nc", RAW_W),
    ("cr_off", I32, "nc+1", 1), ("cr_row", U16, "ncr", 1), ("cr_val", F32, "ncr", 1),
    ("opt_off", I32, "B+1", 1), ("opt_dense", F32, "no", OPT_DENSE_W), ("opt_bot", U8, "no", 1),
    ("opt_ea", I32, "no", 1), ("opt_eb", I32, "no", 1),
    ("os_off", I32, "no+1", 1), ("os_row", U16, "nos", 1), ("os_val", F32, "nos", 1),
    ("oh_off", I32, "no+1", 1), ("oh_row", U16, "noh", 1), ("oh_val", F32, "noh", 1),
]
TRAIN_ARRAYS = [
    ("outcome", F32, "B", 1), ("root_value", F32, "B", 1), ("turn", I32, "B", 1), ("seat", U8, "B", 1),
    ("subset", U8, "B", 1), ("kind", U8, "B", 1), ("sims", I32, "B", 1), ("choice", I32, "B", 1),
    ("game", I32, "B", 1),
    ("cand_off", I32, "B+1", 1), ("cand_visits", F32, "ncand", 1), ("cand_prior", F32, "ncand", 1),
    ("cand_q", F32, "ncand", 1), ("co_off", I32, "ncand+1", 1), ("co_opt", I32, "nco", 1),
]
ALL_ARRAYS = STATE_ARRAYS + TRAIN_ARRAYS
SPECS = {name: (dt, ln, w) for name, dt, ln, w in ALL_ARRAYS}
STATE_NAMES = [a[0] for a in STATE_ARRAYS]
TRAIN_NAMES = [a[0] for a in TRAIN_ARRAYS]
# CSR structure: offsets name -> (level it indexes, arrays it covers)
CSR = {
    "sp_off": ("B", ("sp_row", "sp_val")),
    "card_off": ("B", ("card_group", "card_raw", "cr_off")),
    "cr_off": ("nc", ("cr_row", "cr_val")),
    "opt_off": ("B", ("opt_dense", "opt_bot", "opt_ea", "opt_eb", "os_off", "oh_off")),
    "os_off": ("no", ("os_row", "os_val")),
    "oh_off": ("no", ("oh_row", "oh_val")),
    "cand_off": ("B", ("cand_visits", "cand_prior", "cand_q", "co_off")),
    "co_off": ("ncand", ("co_opt",)),
}
HEADER_COUNTS = ("B", "nsp", "nc", "ncr", "no", "nos", "noh")


def pad8(n: int) -> int:
    return (n + 7) & ~7


def counts(pack: dict) -> dict:
    """The length symbols of a pack (B, nsp, nc, ...), read from its offsets."""
    c = {"B": int(len(pack["dense"]))}
    c["nsp"] = int(pack["sp_off"][-1])
    c["nc"] = int(pack["card_off"][-1])
    c["ncr"] = int(pack["cr_off"][-1]) if len(pack["cr_off"]) else 0
    c["no"] = int(pack["opt_off"][-1])
    c["nos"] = int(pack["os_off"][-1]) if len(pack["os_off"]) else 0
    c["noh"] = int(pack["oh_off"][-1]) if len(pack["oh_off"]) else 0
    if "cand_off" in pack:
        c["ncand"] = int(pack["cand_off"][-1])
        c["nco"] = int(pack["co_off"][-1]) if len(pack["co_off"]) else 0
    return c


def _length(sym: str, c: dict) -> int:
    if sym.endswith("+1"):
        return c[sym[:-2]] + 1
    return c[sym]


def validate(pack: dict, train: bool | None = None) -> None:
    """Check names, dtypes, shapes, offsets, row ranges and ea/eb references; raise ValueError."""
    names = STATE_NAMES + (TRAIN_NAMES if (train if train is not None else "cand_off" in pack) else [])
    missing = [n for n in names if n not in pack]
    if missing:
        raise ValueError(f"missing arrays {missing}")
    c = counts(pack)
    for n in names:
        dt, ln, w = SPECS[n]
        a = pack[n]
        if a.dtype != dt:
            raise ValueError(f"{n}: dtype {a.dtype}, want {dt}")
        want = (_length(ln, c),) if w == 1 else (_length(ln, c), w)
        if a.shape != want:
            raise ValueError(f"{n}: shape {a.shape}, want {want}")
    for off in CSR:
        if off not in pack:
            continue
        o = np.asarray(pack[off])
        if len(o) and (o[0] != 0 or np.any(np.diff(o) < 0)):
            raise ValueError(f"{off}: offsets must start at 0 and not decrease")
    for n, lim in (("sp_row", TABLE_ROWS), ("cr_row", TABLE_ROWS), ("oh_row", TABLE_ROWS), ("os_row", SLOT_ROWS)):
        if len(pack[n]) and int(pack[n].max()) >= lim:
            raise ValueError(f"{n}: row {int(pack[n].max())} >= {lim}")
    if len(pack["card_group"]) and int(pack["card_group"].max()) >= GROUPS:
        raise ValueError("card_group >= 4")
    ncards = np.diff(pack["card_off"])
    nopt = np.diff(pack["opt_off"])
    st = np.repeat(np.arange(c["B"]), nopt)
    for n in ("opt_ea", "opt_eb"):
        e = pack[n]
        if len(e) and (np.any(e < 0) or np.any(e > ncards[st])):
            raise ValueError(f"{n}: reference outside the state's cards")
    if "co_off" in pack and c["nco"]:
        cst = np.repeat(np.arange(c["B"]), np.diff(pack["cand_off"]))
        ost = np.repeat(cst, np.diff(pack["co_off"]))
        co = pack["co_opt"]
        if np.any(co < 0) or np.any(co >= nopt[ost]):
            raise ValueError("co_opt: option index outside the state's options")


# ------------------------------------------------------------------------------------------------
# shards
# ------------------------------------------------------------------------------------------------

def write_shard(dirpath, pack: dict, games: list[str], source: str = "synthetic") -> None:
    """One shard directory: <name>.npy per array plus meta.json (SPEC §2)."""
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    for n in STATE_NAMES + TRAIN_NAMES:
        np.save(d / f"{n}.npy", np.ascontiguousarray(pack[n], dtype=SPECS[n][0]))
    meta = {"format": FORMAT, "n": int(len(pack["dense"])), "games": list(games), "source": source}
    (d / "meta.json").write_text(json.dumps(meta))


def read_shard(dirpath, in_ram: bool = False) -> tuple[dict, dict]:
    """(pack, meta) of one shard directory; memory-mapped unless in_ram."""
    d = Path(dirpath)
    meta = json.loads((d / "meta.json").read_text())
    if meta.get("format") != FORMAT:
        raise ValueError(f"{d}: format {meta.get('format')!r}, want {FORMAT!r}")
    pack = {}
    for n in STATE_NAMES + TRAIN_NAMES:
        p = d / f"{n}.npy"
        if not p.exists():
            raise ValueError(f"{d}: missing {n}.npy")
        a = np.load(p, mmap_mode=None if in_ram else "r")
        pack[n] = a
    if len(pack["dense"]) != meta["n"]:
        raise ValueError(f"{d}: meta n={meta['n']} but dense has {len(pack['dense'])} rows")
    return pack, meta


def shard_dirs(path) -> list[Path]:
    """A shard directory itself, or the shard subdirectories (00000, 00001, ...) of a pack output."""
    p = Path(path)
    if (p / "meta.json").exists():
        return [p]
    subs = sorted(q for q in p.iterdir() if (q / "meta.json").exists())
    if not subs:
        raise ValueError(f"{p}: no shard (meta.json) found")
    return subs


def holdout_game(game_id: str) -> bool:
    return zlib.crc32(game_id.encode("utf-8")) % HOLDOUT_MOD == 0


# ------------------------------------------------------------------------------------------------
# wire messages (SPEC §3)
# ------------------------------------------------------------------------------------------------

def encode_request(pack: dict, want=None) -> bytes:
    """u32 nbytes, the 32-byte header, then want and the state arrays, each zero-padded to 8 bytes."""
    c = counts(pack)
    if want is None:
        want = np.ones(c["B"], U8)
    parts = [struct.pack("<8I", REQ_MAGIC, *(c[k] for k in HEADER_COUNTS))]
    for name, dt, _, _ in [("want", U8, "B", 1)] + STATE_ARRAYS:
        a = want if name == "want" else pack[name]
        b = np.ascontiguousarray(a, dtype=dt).tobytes()
        parts.append(b)
        if len(b) % 8:
            parts.append(b"\0" * (8 - len(b) % 8))
    body = b"".join(parts)
    return struct.pack("<I", len(body)) + body


def decode_request(body) -> tuple[dict, np.ndarray]:
    """(pack, want) from a request body (the bytes after nbytes); arrays are views into `body`.
    The last array's padding may be absent."""
    mv = memoryview(body)
    if len(mv) < 32:
        raise ValueError(f"request body {len(mv)} bytes, shorter than the 32-byte header")
    magic, *cnt = struct.unpack_from("<8I", mv, 0)
    if magic != REQ_MAGIC:
        raise ValueError(f"bad request magic {magic:#x}")
    c = dict(zip(HEADER_COUNTS, cnt))
    pos = 32
    pack, want = {}, None
    layout = [("want", U8, "B", 1)] + STATE_ARRAYS
    for i, (name, dt, ln, w) in enumerate(layout):
        n = _length(ln, c) * w
        nb = n * dt.itemsize
        if pos + nb > len(mv):
            raise ValueError(f"request truncated at {name}: need {pos + nb} bytes, have {len(mv)}")
        a = np.frombuffer(mv, dtype=dt, count=n, offset=pos)
        if w > 1:
            a = a.reshape(-1, w)
        if name == "want":
            want = a
        else:
            pack[name] = a
        pos += pad8(nb) if i < len(layout) - 1 else nb
    if len(mv) > pad8(pos):
        raise ValueError(f"request has {len(mv) - pad8(pos)} trailing bytes")
    return pack, want


def request_size(c: dict) -> int:
    """The body size (bytes after nbytes) of a request with counts c, last array padded."""
    n = 32
    for _, dt, ln, w in [("want", U8, "B", 1)] + STATE_ARRAYS:
        n += pad8(_length(ln, c) * w * dt.itemsize)
    return n


def encode_response(value: np.ndarray, score: np.ndarray) -> bytes:
    """u32 nbytes, magic, B, no, 0, value f32[B] zero-padded to 8 bytes, score f32[no] (unpadded)."""
    value = np.ascontiguousarray(value, dtype=F32)
    score = np.ascontiguousarray(score, dtype=F32)
    vb = value.tobytes()
    body = b"".join([struct.pack("<4I", RESP_MAGIC, len(value), len(score), 0), vb,
                     b"\0" * (pad8(len(vb)) - len(vb)), score.tobytes()])
    return struct.pack("<I", len(body)) + body


def decode_response(body) -> tuple[np.ndarray, np.ndarray]:
    mv = memoryview(body)
    magic, B, no, _ = struct.unpack_from("<4I", mv, 0)
    if magic != RESP_MAGIC:
        raise ValueError(f"bad response magic {magic:#x}")
    value = np.frombuffer(mv, F32, B, 16)
    score = np.frombuffer(mv, F32, no, 16 + pad8(4 * B))
    return value, score


# ------------------------------------------------------------------------------------------------
# records <-> packs (the naive per-record path: tests and small tools)
# ------------------------------------------------------------------------------------------------

def _slice(a, off, i):
    return np.asarray(a[int(off[i]):int(off[i + 1])])


def get_record(pack: dict, i: int) -> dict:
    """Record i of a pack as a dict of per-record arrays (offsets local), the naive reference."""
    r = {"dense": np.asarray(pack["dense"][i]),
         "sp_row": _slice(pack["sp_row"], pack["sp_off"], i), "sp_val": _slice(pack["sp_val"], pack["sp_off"], i)}
    c0, c1 = int(pack["card_off"][i]), int(pack["card_off"][i + 1])
    r["card_group"] = np.asarray(pack["card_group"][c0:c1])
    r["card_raw"] = np.asarray(pack["card_raw"][c0:c1])
    r["cr_row"] = [_slice(pack["cr_row"], pack["cr_off"], c) for c in range(c0, c1)]
    r["cr_val"] = [_slice(pack["cr_val"], pack["cr_off"], c) for c in range(c0, c1)]
    o0, o1 = int(pack["opt_off"][i]), int(pack["opt_off"][i + 1])
    for n in ("opt_dense", "opt_bot", "opt_ea", "opt_eb"):
        r[n] = np.asarray(pack[n][o0:o1])
    for p in ("os", "oh"):
        r[f"{p}_row"] = [_slice(pack[f"{p}_row"], pack[f"{p}_off"], o) for o in range(o0, o1)]
        r[f"{p}_val"] = [_slice(pack[f"{p}_val"], pack[f"{p}_off"], o) for o in range(o0, o1)]
    if "cand_off" in pack:
        for n in ("outcome", "root_value", "turn", "seat", "subset", "kind", "sims", "choice", "game"):
            r[n] = pack[n][i].item()
        k0, k1 = int(pack["cand_off"][i]), int(pack["cand_off"][i + 1])
        for n in ("cand_visits", "cand_prior", "cand_q"):
            r[n] = np.asarray(pack[n][k0:k1])
        r["cands"] = [_slice(pack["co_opt"], pack["co_off"], k) for k in range(k0, k1)]
    return r


def pack_records(records: list[dict]) -> dict:
    """The inverse of get_record: a list of records to one pack (training arrays when present)."""
    def csr(lens):
        o = np.zeros(len(lens) + 1, I32)
        np.cumsum(np.asarray(lens, np.int64), out=o[1:])
        return o

    def cat(xs, dt, w=1):
        xs = [np.asarray(x, dt).reshape(-1, w) if w > 1 else np.asarray(x, dt).reshape(-1) for x in xs]
        if not xs:
            return np.zeros((0, w) if w > 1 else 0, dt)
        return np.concatenate(xs).astype(dt)

    p = {"dense": cat([r["dense"] for r in records], F32, DENSE_W)}
    p["sp_off"] = csr([len(r["sp_row"]) for r in records])
    p["sp_row"] = cat([r["sp_row"] for r in records], U16)
    p["sp_val"] = cat([r["sp_val"] for r in records], F32)
    p["card_off"] = csr([len(r["card_group"]) for r in records])
    p["card_group"] = cat([r["card_group"] for r in records], U8)
    p["card_raw"] = cat([r["card_raw"] for r in records], F32, RAW_W)
    crs = [x for r in records for x in r["cr_row"]]
    p["cr_off"] = csr([len(x) for x in crs])
    p["cr_row"] = cat(crs, U16)
    p["cr_val"] = cat([x for r in records for x in r["cr_val"]], F32)
    p["opt_off"] = csr([len(r["opt_bot"]) for r in records])
    p["opt_dense"] = cat([r["opt_dense"] for r in records], F32, OPT_DENSE_W)
    for n, dt in (("opt_bot", U8), ("opt_ea", I32), ("opt_eb", I32)):
        p[n] = cat([r[n] for r in records], dt)
    for q in ("os", "oh"):
        rows = [x for r in records for x in r[f"{q}_row"]]
        p[f"{q}_off"] = csr([len(x) for x in rows])
        p[f"{q}_row"] = cat(rows, U16)
        p[f"{q}_val"] = cat([x for r in records for x in r[f"{q}_val"]], F32)
    if records and "cands" in records[0]:
        for n in ("outcome", "root_value", "turn", "seat", "subset", "kind", "sims", "choice", "game"):
            p[n] = np.array([r[n] for r in records], SPECS[n][0])
        p["cand_off"] = csr([len(r["cands"]) for r in records])
        for n in ("cand_visits", "cand_prior", "cand_q"):
            p[n] = cat([r[n] for r in records], F32)
        cands = [x for r in records for x in r["cands"]]
        p["co_off"] = csr([len(x) for x in cands])
        p["co_opt"] = cat(cands, I32)
    return p


# ------------------------------------------------------------------------------------------------
# synthetic data
# ------------------------------------------------------------------------------------------------

_LATENT_SEED = 0x5EED  # the latent "rules" are the same for every synthetic set, so they generalise
_N_NAMES = 400


def _latents():
    r = np.random.default_rng(_LATENT_SEED)
    return {
        "name_row": r.choice(TABLE_ROWS, _N_NAMES, replace=False).astype(np.int64),
        "name_strength": r.standard_normal(_N_NAMES),
        "kind_value": r.standard_normal(24) * 0.8,
        "kind_row": r.choice(TABLE_ROWS, 24, replace=False).astype(np.int64),
        "sparse_vocab": r.choice(TABLE_ROWS, 3000, replace=False).astype(np.int64),
    }


_LATENTS = _latents()


def _csr_from_lens(lens) -> np.ndarray:
    o = np.zeros(len(lens) + 1, np.int64)
    np.cumsum(lens, out=o[1:])
    return o


def _seg_softmax(x, seg, n):
    m = np.full(n, -np.inf)
    np.maximum.at(m, seg, x)
    e = np.exp(x - m[seg])
    s = np.zeros(n)
    np.add.at(s, seg, e)
    return e / s[seg]


def synthetic(n: int, seed: int = 0, max_cards: int = 40, min_opts: int = 1, max_opts: int = 10,
              sparse_mean: int = 30, subset_frac: float = 0.3, game_len: int = 30,
              game_prefix: str | None = None) -> tuple[dict, list[str]]:
    """(pack, games): n random training records with a learnable structure.

    Value: the outcome is Bernoulli(sigmoid(2 dense[0] + 0.5 (sum of my battlefield cards' strength -
    the opponent's))), where a card's strength is a property of its name (its first identity row) and
    is also visible as 0.25 x strength in raw[10] (power). Policy: an option's goodness is
    1.5 opt_dense[0] + its kind's value (first slot row = kind, last hashed row = the kind's row) + 0.5 x its ea
    card's strength; the visits follow softmax(candidate goodness), candidate goodness = the sum over
    member options. Candidate 0 is the bot's answer (noisy goodness)."""
    rng = np.random.default_rng(seed)
    L = _LATENTS
    B = n
    dense = (rng.standard_normal((B, DENSE_W)) * 0.5).astype(np.float32)
    # sparse bag
    nsp = rng.integers(0, 2 * sparse_mean + 1, B)
    nsp[rng.random(B) < 0.03] = 0
    sp_off = _csr_from_lens(nsp)
    sp_row = L["sparse_vocab"][rng.integers(0, len(L["sparse_vocab"]), sp_off[-1])].astype(np.uint16)
    sp_val = np.where(rng.random(sp_off[-1]) < 0.7, 1.0, rng.random(sp_off[-1]) * 3).astype(np.float32)
    # cards, grouped in order 0..3 within each state
    ncs = rng.integers(0, max_cards + 1, B)
    ncs[rng.random(B) < 0.03] = 0
    card_off = _csr_from_lens(ncs)
    nc = int(card_off[-1])
    cstate = np.repeat(np.arange(B), ncs)
    group = rng.integers(0, GROUPS, nc)
    order = np.lexsort((group, cstate))
    group = group[order].astype(np.uint8)
    name = rng.integers(0, _N_NAMES, nc)
    strength = L["name_strength"][name]
    raw = (rng.random((nc, RAW_W)) < 0.15).astype(np.float32)
    raw[:, 10] = (0.25 * strength + 0.05 * rng.standard_normal(nc)).astype(np.float32)
    raw[:, 11] = rng.integers(0, 6, nc) / 4
    raw[:, 59] = (group != 1).astype(np.float32)
    raw[:, 60:63] = 0
    raw[np.arange(nc), 60 + np.minimum(group, 2)] = 1
    raw[:, 67] = 1
    ncr = rng.integers(1, 4, nc)
    ncr[rng.random(nc) < 0.05] = 0
    cr_off = _csr_from_lens(ncr)
    cr_row = rng.integers(0, TABLE_ROWS, cr_off[-1])
    first = cr_off[:-1][ncr > 0]
    cr_row[first] = L["name_row"][name[ncr > 0]]
    cr_val = np.ones(cr_off[-1], np.float32)
    # options
    nos_ = rng.integers(min_opts, max_opts + 1, B)
    opt_off = _csr_from_lens(nos_)
    no = int(opt_off[-1])
    ostate = np.repeat(np.arange(B), nos_)
    opt_dense = (rng.standard_normal((no, OPT_DENSE_W)) * 0.5).astype(np.float32)
    ncard_o = ncs[ostate]
    has_a = (rng.random(no) < 0.7) & (ncard_o > 0)
    has_b = (rng.random(no) < 0.3) & (ncard_o > 0)
    ea = np.where(has_a, 1 + (rng.random(no) * np.maximum(ncard_o, 1)).astype(np.int64), 0)
    eb = np.where(has_b, 1 + (rng.random(no) * np.maximum(ncard_o, 1)).astype(np.int64), 0)
    kind = rng.integers(0, 24, no)
    ns = rng.integers(2, 8, no)
    os_off = _csr_from_lens(ns)
    os_row = rng.integers(24, SLOT_ROWS, os_off[-1])
    os_row[os_off[:-1]] = kind
    os_val = np.ones(os_off[-1], np.float32)
    nh = rng.integers(1, 4, no)
    oh_off = _csr_from_lens(nh)
    oh_row = rng.integers(0, TABLE_ROWS, oh_off[-1])
    oh_row[oh_off[1:] - 1] = L["kind_row"][kind]
    oh_val = np.ones(oh_off[-1], np.float32)
    ea_str = np.where(ea > 0, strength[np.where(ea > 0, card_off[ostate] + ea - 1, 0)] if nc else 0, 0)
    good = 1.5 * opt_dense[:, 0] + L["kind_value"][kind] + 0.5 * ea_str
    noisy = good + 0.7 * rng.standard_normal(no)
    # candidates
    subset = (rng.random(B) < subset_frac).astype(np.uint8)
    sub_o = subset[ostate].astype(bool)
    # singleton states: candidate j = the option of rank j by noisy goodness (candidate 0 = the bot's)
    rank_order = np.lexsort((-noisy, ostate))
    single_opts = rank_order[~sub_o[rank_order]]
    # subset states: k candidates, candidate 0 = options with noisy goodness > 0, others random subsets
    k_sub = np.where(subset == 1, rng.integers(1, 6, B), 0)
    k_single = np.where(subset == 1, 0, nos_)
    ncand_s = k_sub + k_single
    cand_off = _csr_from_lens(ncand_s)
    ncand = int(cand_off[-1])
    cand_state = np.repeat(np.arange(B), ncand_s)
    cand_j = np.arange(ncand) - cand_off[cand_state]
    is_sub_c = subset[cand_state].astype(bool)
    # member lists: singletons have one member; subset candidates have each option with p=0.5 (cand 0: bot)
    sub_c = np.nonzero(is_sub_c)[0]
    o_per = nos_[cand_state[sub_c]]
    pair_c = np.repeat(sub_c, o_per)
    pair_o = (np.arange(len(pair_c)) - np.repeat(_csr_from_lens(o_per)[:-1], o_per))
    glob_o = opt_off[cand_state[pair_c]] + pair_o
    bot_c = cand_j[pair_c] == 0
    take = np.where(bot_c, noisy[glob_o] > 0, rng.random(len(pair_c)) < 0.5)
    pair_c, pair_o, glob_o = pair_c[take], pair_o[take], glob_o[take]
    sing_c = np.nonzero(~is_sub_c)[0]
    sing_o = single_opts  # same order: states ascending, rank ascending == candidates ascending
    mem_c = np.concatenate([pair_c, sing_c])
    mem_glob = np.concatenate([glob_o, sing_o])
    o2 = np.lexsort((mem_glob, mem_c))
    mem_c, mem_glob = mem_c[o2], mem_glob[o2]
    co_off = _csr_from_lens(np.bincount(mem_c, minlength=ncand))
    co_opt = (mem_glob - opt_off[ostate[mem_glob]]).astype(np.int32)
    opt_bot = np.zeros(no, np.uint8)
    opt_bot[mem_glob[cand_j[mem_c] == 0]] = 1
    cgood = np.zeros(ncand)
    np.add.at(cgood, mem_c, good[mem_glob])
    # outcome and value
    my = (group == 0)
    opp = (group == 1)
    adv = np.zeros(B)
    np.add.at(adv, cstate, np.where(my, strength, 0) - np.where(opp, strength, 0))
    logit = 2.0 * dense[:, 0] + 0.5 * adv
    p = 1 / (1 + np.exp(-logit))
    outcome = (rng.random(B) < p).astype(np.float32)
    outcome[rng.random(B) < 0.03] = 0.5
    outcome[rng.random(B) < 0.05] = -1
    root_value = (1 / (1 + np.exp(-(logit + 0.3 * rng.standard_normal(B))))).astype(np.float32)
    sims = rng.choice(np.array([16, 32, 64, 128]), B).astype(np.int32)
    pv = _seg_softmax(cgood, cand_state, B)
    visits = np.floor(sims[cand_state] * pv + rng.random(ncand)).astype(np.float32)
    prior = _seg_softmax(0.5 * (cgood + 0.5 * rng.standard_normal(ncand)), cand_state, B).astype(np.float32)
    q = 1 / (1 + np.exp(-(0.5 * cgood + 0.5 * logit[cand_state] + 0.2 * rng.standard_normal(ncand))))
    q = np.where(visits > 0, q, 0).astype(np.float32)
    best = np.full(B, -1.0)
    np.maximum.at(best, cand_state, visits)
    is_best = visits == best[cand_state]
    choice = np.full(B, 0, np.int64)
    first_best = np.nonzero(is_best)[0][::-1]   # reversed so the first best candidate wins the assignment
    choice[cand_state[first_best]] = cand_j[first_best]
    # games: consecutive runs of ~game_len records
    gid = np.arange(B) // game_len
    games = [f"{game_prefix or f'syn{seed}'}-{g}" for g in range(int(gid[-1]) + 1 if B else 0)]
    turn = (np.arange(B) % game_len) // 2 + 1
    pack = {
        "dense": dense, "sp_off": sp_off.astype(I32), "sp_row": sp_row, "sp_val": sp_val,
        "card_off": card_off.astype(I32), "card_group": group, "card_raw": raw,
        "cr_off": cr_off.astype(I32), "cr_row": cr_row.astype(U16), "cr_val": cr_val,
        "opt_off": opt_off.astype(I32), "opt_dense": opt_dense, "opt_bot": opt_bot,
        "opt_ea": ea.astype(I32), "opt_eb": eb.astype(I32),
        "os_off": os_off.astype(I32), "os_row": os_row.astype(U16), "os_val": os_val,
        "oh_off": oh_off.astype(I32), "oh_row": oh_row.astype(U16), "oh_val": oh_val,
        "outcome": outcome, "root_value": root_value, "turn": turn.astype(I32),
        "seat": rng.integers(0, 2, B).astype(U8), "subset": subset, "kind": rng.integers(0, 12, B).astype(U8),
        "sims": sims, "choice": choice.astype(I32), "game": gid.astype(I32),
        "cand_off": cand_off.astype(I32), "cand_visits": visits, "cand_prior": prior, "cand_q": q,
        "co_off": co_off.astype(I32), "co_opt": co_opt,
    }
    return pack, games


def write_synthetic(root, n: int, seed: int = 0, shard_size: int = 200000, **kw) -> list[Path]:
    """Synthetic shards as `dzgorge pack` lays them out: root/00000, root/00001, ..."""
    out = []
    root = Path(root)
    i = 0
    while i * shard_size < n or (n == 0 and i == 0):
        m = min(shard_size, n - i * shard_size)
        pack, games = synthetic(m, seed=seed * 1000 + i, game_prefix=f"syn{seed}-{i}", **kw)
        d = root / f"{i:05d}"
        write_shard(d, pack, games)
        out.append(d)
        i += 1
    return out
