"""Shards -> batches: a dataset over shard directories, a split by game, a vectorised collate, the
same Batch from a wire request, and a prefetching loader (collate threads, or worker processes)."""
from __future__ import annotations

import os
import queue
import sys
import threading
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from . import packfmt as pf


# ------------------------------------------------------------------------------------------------
# CSR gathers
# ------------------------------------------------------------------------------------------------

def csr_take(off, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(new offsets [len(sel)+1], positions into the covered arrays) of the CSR rows `sel`."""
    s = off[sel].astype(np.int64)
    lens = off[sel + 1].astype(np.int64) - s
    new = np.zeros(len(sel) + 1, np.int64)
    np.cumsum(lens, out=new[1:])
    pos = np.arange(new[-1], dtype=np.int64)
    pos += np.repeat(s - new[:-1], lens)
    return new, pos


def _owner(off: np.ndarray, n: int) -> np.ndarray:
    """Row of each element of a CSR with offsets `off` (n = len(off) - 1)."""
    return np.repeat(np.arange(n, dtype=np.int64), np.diff(off))


def take(pack: dict, sel, targets: bool = True) -> dict:
    """The records `sel` of a pack as a new pack (int64 offsets starting at 0), fully vectorised."""
    sel = np.asarray(sel, np.int64)
    out = {"dense": np.asarray(pack["dense"][sel])}
    out["sp_off"], p = csr_take(pack["sp_off"], sel)
    out["sp_row"], out["sp_val"] = pack["sp_row"][p], pack["sp_val"][p]
    out["card_off"], cp = csr_take(pack["card_off"], sel)
    out["card_group"], out["card_raw"] = pack["card_group"][cp], pack["card_raw"][cp]
    out["cr_off"], p = csr_take(pack["cr_off"], cp)
    out["cr_row"], out["cr_val"] = pack["cr_row"][p], pack["cr_val"][p]
    out["opt_off"], op = csr_take(pack["opt_off"], sel)
    for n in ("opt_dense", "opt_bot", "opt_ea", "opt_eb"):
        out[n] = pack[n][op]
    for q in ("os", "oh"):
        out[f"{q}_off"], p = csr_take(pack[f"{q}_off"], op)
        out[f"{q}_row"], out[f"{q}_val"] = pack[f"{q}_row"][p], pack[f"{q}_val"][p]
    if targets and "cand_off" in pack:
        for n in ("outcome", "root_value", "subset", "choice", "game", "kind", "turn", "seat", "sims"):
            out[n] = pack[n][sel]
        out["cand_off"], kp = csr_take(pack["cand_off"], sel)
        for n in ("cand_visits", "cand_prior", "cand_q"):
            out[n] = pack[n][kp]
        out["co_off"], p = csr_take(pack["co_off"], kp)
        out["co_opt"] = pack["co_opt"][p]
    return out


# offsets name -> the count its values index (to shift when concatenating packs)
_SHIFT = {"sp_off": "nsp", "card_off": "nc", "cr_off": "ncr", "opt_off": "no", "os_off": "nos", "oh_off": "noh",
          "cand_off": "ncand", "co_off": "nco"}


def merge_shards(packs: list[dict], metas: list[dict]) -> tuple[dict, dict]:
    """Shards (memory-mapped) as one pack in RAM: offsets int64 and shifted, `game` re-indexed into the
    concatenated games list. Each array is read once into its final buffer, so the peak memory is the
    result's."""
    out = {}
    for n in packs[0]:
        parts = [p[n] for p in packs]
        if n in _SHIFT:
            total = sum(len(a) - 1 for a in parts) + 1
            o = np.empty(total, np.int64)
            pos = base = 0
            for a in parts:
                k = len(a) - 1
                o[pos:pos + k] = a[:k]
                o[pos:pos + k] += base
                pos += k
                base += int(a[k])
            o[pos] = base
        else:
            o = np.empty((sum(len(a) for a in parts),) + parts[0].shape[1:], parts[0].dtype)
            pos = 0
            for a in parts:
                o[pos:pos + len(a)] = a
                pos += len(a)
            if n == "game":
                bases = np.cumsum([0] + [len(m["games"]) for m in metas[:-1]])
                o += np.repeat(bases, [len(a) for a in parts]).astype(o.dtype)
        out[n] = o
    meta = {"format": pf.FORMAT, "n": int(sum(m["n"] for m in metas)), "games": [g for m in metas for g in m["games"]],
            "source": [m.get("source") for m in metas]}
    return out, meta


def concat_packs(packs: list[dict]) -> dict:
    """Packs (offsets starting at 0) to one pack; state-local references (ea, eb, co_opt) need no shift."""
    if len(packs) == 1:
        return packs[0]
    out = {}
    for n in packs[0]:
        if n in _SHIFT:
            parts, base = [np.zeros(1, np.int64)], 0
            for p in packs:
                o = np.asarray(p[n], np.int64)
                parts.append(o[1:] + base)
                base += int(o[-1])
            out[n] = np.concatenate(parts)
        else:
            out[n] = np.concatenate([np.asarray(p[n]) for p in packs])
    return out


# ------------------------------------------------------------------------------------------------
# Batch
# ------------------------------------------------------------------------------------------------

@dataclass
class Batch:
    """A batch of states as torch tensors. Global indices are batch-wide; `pos` are within a state."""
    B: int
    dense: torch.Tensor        # [B, 68] f32
    sp_row: torch.Tensor       # [nsp] i64
    sp_val: torch.Tensor       # [nsp] f32
    sp_off: torch.Tensor       # [B+1] i64 (F.embedding_bag(..., include_last_offset=True))
    sp_state: torch.Tensor     # [nsp] i64 state of each sparse row
    card_raw: torch.Tensor     # [nc, 68] f32
    card_group: torch.Tensor   # [nc] i64
    card_state: torch.Tensor   # [nc] i64
    card_pos: torch.Tensor     # [nc] i64 position within the state's cards
    cr_row: torch.Tensor       # [ncr] i64 card identity rows
    cr_val: torch.Tensor       # [ncr] f32
    cr_off: torch.Tensor       # [nc+1] i64
    cr_card: torch.Tensor      # [ncr] i64 card of each identity row
    Cmax: int
    card_mask: torch.Tensor    # [B, Cmax] bool, True = a card
    opt_dense: torch.Tensor    # [no, 24] f32
    opt_bot: torch.Tensor      # [no] f32
    opt_state: torch.Tensor    # [no] i64
    opt_pos: torch.Tensor      # [no] i64
    Omax: int
    opt_ea: torch.Tensor       # [no] i64 batch-wide card index, -1 = none
    opt_eb: torch.Tensor       # [no] i64
    os_row: torch.Tensor       # [nos] i64 (< 128)
    os_val: torch.Tensor
    os_off: torch.Tensor       # [no+1]
    oh_row: torch.Tensor       # [noh] i64 (< 16384)
    oh_val: torch.Tensor
    oh_off: torch.Tensor       # [no+1]
    # training targets (None for a wire batch)
    ncand: int = 0
    cand_state: torch.Tensor | None = None    # [ncand] i64
    cand_visits: torch.Tensor | None = None   # [ncand] f32
    cand_prior: torch.Tensor | None = None
    cand_q: torch.Tensor | None = None
    co_cand: torch.Tensor | None = None       # [nco] i64 candidate of each membership
    co_opt: torch.Tensor | None = None        # [nco] i64 batch-wide option of each membership
    root_value: torch.Tensor | None = None    # [B] f32
    outcome: torch.Tensor | None = None       # [B] f32 (-1 unknown)
    subset: torch.Tensor | None = None        # [B] u8
    want: torch.Tensor | None = None          # [B] bool (wire batches)

    @property
    def no(self) -> int:
        return int(self.opt_dense.shape[0])

    @property
    def nc(self) -> int:
        return int(self.card_raw.shape[0])

    def to(self, device, non_blocking: bool = True) -> "Batch":
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            kw[f.name] = v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v
        return Batch(**kw)

    def pin_memory(self) -> "Batch":
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            kw[f.name] = v.pin_memory() if isinstance(v, torch.Tensor) else v
        return Batch(**kw)

    def flatten(self) -> "FlatBatch":
        """The tensors concatenated into one buffer per dtype (one copy each to a device or between
        processes, instead of one per field)."""
        layout, parts, size = [], {}, {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, torch.Tensor):
                k = size.get(v.dtype, 0)
                layout.append((f.name, v.dtype, k, tuple(v.shape)))
                parts.setdefault(v.dtype, []).append(v.reshape(-1))
                size[v.dtype] = k + v.numel()
            else:
                layout.append((f.name, None, v, None))
        return FlatBatch(layout, {dt: torch.cat(ps) for dt, ps in parts.items()})


class FlatBatch:
    """A Batch as one buffer per dtype plus the layout; unflatten() gives the Batch back as views."""

    def __init__(self, layout: list, bufs: dict):
        self.layout, self.bufs = layout, bufs

    def pin_memory(self) -> "FlatBatch":
        return FlatBatch(self.layout, {dt: b.pin_memory() for dt, b in self.bufs.items()})

    def to(self, device, non_blocking: bool = True) -> "FlatBatch":
        return FlatBatch(self.layout, {dt: b.to(device, non_blocking=non_blocking) for dt, b in self.bufs.items()})

    def unflatten(self) -> Batch:
        kw = {}
        for name, dt, k, shape in self.layout:
            if dt is None:
                kw[name] = k
            else:
                n = 1
                for d in shape:
                    n *= d
                kw[name] = self.bufs[dt][k:k + n].view(shape)
        return Batch(**kw)


def _t(a, dtype=None) -> torch.Tensor:
    a = np.asarray(a) if dtype is None else np.asarray(a, dtype=dtype)
    if not a.flags.writeable or not a.flags.c_contiguous:
        a = np.array(a, copy=True, order="C")
    return torch.from_numpy(a)


def to_batch(p: dict, targets: bool = True, want=None) -> Batch:
    """A pack (offsets starting at 0) to a Batch. Rows are taken modulo nothing: out-of-range rows are
    the caller's error (validate wire input with packfmt.validate). ea/eb outside the state's cards
    become -1 (none)."""
    i64, f32 = np.int64, np.float32
    B = int(len(p["dense"]))
    card_off = np.asarray(p["card_off"], i64)
    opt_off = np.asarray(p["opt_off"], i64)
    nc, no = int(card_off[-1]), int(opt_off[-1])
    ncards = np.diff(card_off)
    nopts = np.diff(opt_off)
    card_state = np.repeat(np.arange(B, dtype=i64), ncards)
    card_pos = np.arange(nc, dtype=i64) - card_off[:-1][card_state]
    opt_state = np.repeat(np.arange(B, dtype=i64), nopts)
    opt_pos = np.arange(no, dtype=i64) - opt_off[:-1][opt_state]
    Cmax = int(ncards.max()) if B else 0
    Omax = int(nopts.max()) if B else 0
    card_mask = np.arange(Cmax, dtype=i64)[None, :] < ncards[:, None]
    nco = ncards[opt_state]
    base = card_off[:-1][opt_state]

    def ref(e):
        e = np.asarray(e, i64) - 1
        return np.where((e >= 0) & (e < nco), base + e, -1)

    sp_off = np.asarray(p["sp_off"], i64)
    cr_off = np.asarray(p["cr_off"], i64)
    b = Batch(
        B=B, dense=_t(p["dense"], f32),
        sp_row=_t(p["sp_row"], i64), sp_val=_t(p["sp_val"], f32), sp_off=_t(sp_off), sp_state=_t(_owner(sp_off, B)),
        card_raw=_t(p["card_raw"], f32).reshape(nc, pf.RAW_W), card_group=_t(p["card_group"], i64),
        card_state=_t(card_state), card_pos=_t(card_pos),
        cr_row=_t(p["cr_row"], i64), cr_val=_t(p["cr_val"], f32), cr_off=_t(cr_off), cr_card=_t(_owner(cr_off, nc)),
        Cmax=Cmax, card_mask=_t(card_mask),
        opt_dense=_t(p["opt_dense"], f32).reshape(no, pf.OPT_DENSE_W), opt_bot=_t(p["opt_bot"], f32),
        opt_state=_t(opt_state), opt_pos=_t(opt_pos), Omax=Omax,
        opt_ea=_t(ref(p["opt_ea"])), opt_eb=_t(ref(p["opt_eb"])),
        os_row=_t(p["os_row"], i64), os_val=_t(p["os_val"], f32), os_off=_t(p["os_off"], i64),
        oh_row=_t(p["oh_row"], i64), oh_val=_t(p["oh_val"], f32), oh_off=_t(p["oh_off"], i64),
    )
    if targets and "cand_off" in p:
        cand_off = np.asarray(p["cand_off"], i64)
        co_off = np.asarray(p["co_off"], i64)
        ncand = int(cand_off[-1])
        cand_state = _owner(cand_off, B)
        co_cand = _owner(co_off, ncand)
        b.ncand = ncand
        b.cand_state = _t(cand_state)
        b.cand_visits = _t(p["cand_visits"], f32)
        b.cand_prior = _t(p["cand_prior"], f32)
        b.cand_q = _t(p["cand_q"], f32)
        b.co_cand = _t(co_cand)
        b.co_opt = _t(opt_off[:-1][cand_state[co_cand]] + np.asarray(p["co_opt"], i64))
        b.root_value = _t(p["root_value"], f32)
        b.outcome = _t(p["outcome"], f32)
        b.subset = _t(p["subset"], np.uint8)
    if want is not None:
        b.want = _t(np.asarray(want) != 0)
    return b


def collate(pack: dict, sel, targets: bool = True) -> Batch:
    """Records `sel` of one pack to a Batch."""
    return to_batch(take(pack, sel, targets), targets)


def batch_from_wire(pack: dict, want=None) -> Batch:
    """A decoded request (packfmt.decode_request) to a Batch (no targets)."""
    return to_batch(pack, targets=False, want=want)


def drop_unwanted_options(pack: dict, want: np.ndarray) -> tuple[dict, np.ndarray | None]:
    """(pack without the options of want=0 states, positions of the kept options) — options are
    scored independently, so the others' scores are unchanged. Returns (pack, None) when nothing drops."""
    nopt = np.diff(np.asarray(pack["opt_off"], np.int64))
    drop = (np.asarray(want) == 0) & (nopt > 0)
    if not drop.any():
        return pack, None
    B = len(nopt)
    keep_states = np.nonzero(want != 0)[0]
    new_off, kept = csr_take(np.asarray(pack["opt_off"], np.int64), keep_states)
    lens = np.zeros(B, np.int64)
    lens[keep_states] = np.diff(new_off)
    out = dict(pack)
    out["opt_off"] = np.concatenate([[0], np.cumsum(lens)])
    for n in ("opt_dense", "opt_bot", "opt_ea", "opt_eb"):
        out[n] = pack[n][kept]
    for q in ("os", "oh"):
        out[f"{q}_off"], p = csr_take(np.asarray(pack[f"{q}_off"], np.int64), kept)
        out[f"{q}_row"], out[f"{q}_val"] = pack[f"{q}_row"][p], pack[f"{q}_val"][p]
    return out, kept


# ------------------------------------------------------------------------------------------------
# dataset
# ------------------------------------------------------------------------------------------------

class ShardSet:
    """Records of one or more shard directories (each a shard or a `dzgorge pack` output directory),
    memory-mapped, or loaded with in_ram, which merges several shards into one pack so that a batch is
    one gather however many shards it spans. Global record index = shards in order."""

    def __init__(self, paths, in_ram: bool = False):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.dirs = [d for p in paths for d in pf.shard_dirs(p)]
        self.in_ram = in_ram
        self.packs, self.metas = [], []
        for d in self.dirs:
            pack, meta = pf.read_shard(d, in_ram=in_ram and len(self.dirs) == 1)
            self.packs.append(pack)
            self.metas.append(meta)
        if in_ram and len(self.packs) > 1:
            pack, meta = merge_shards(self.packs, self.metas)
            self.packs, self.metas = [pack], [meta]
        sizes = [m["n"] for m in self.metas]
        self.starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.n = int(self.starts[-1])

    def __getstate__(self):   # worker processes started with spawn reopen the shards
        return {"dirs": [str(d) for d in self.dirs], "in_ram": self.in_ram}

    def __setstate__(self, st):
        self.__init__(st["dirs"], in_ram=st["in_ram"])

    def __len__(self) -> int:
        return self.n

    def holdout_mask(self) -> np.ndarray:
        """[n] True for records of holdout games (crc32(game id) % 20 == 0)."""
        out = []
        for pack, meta in zip(self.packs, self.metas):
            flags = np.array([pf.holdout_game(g) for g in meta["games"]] or [False], bool)
            out.append(flags[np.asarray(pack["game"])] if len(pack["game"]) else np.zeros(0, bool))
        return np.concatenate(out) if out else np.zeros(0, bool)

    def split(self, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(train indices, holdout indices) over the first `limit` records."""
        m = self.holdout_mask()
        if limit is not None:
            m = m[:limit]
        idx = np.arange(len(m), dtype=np.int64)
        return idx[~m], idx[m]

    def array(self, name: str, idx=None) -> np.ndarray:
        """A per-record array (outcome, game, ...) over the given global indices (default: all)."""
        a = np.concatenate([np.asarray(p[name]) for p in self.packs])
        return a if idx is None else a[idx]

    def take(self, idx, targets: bool = True) -> dict:
        idx = np.asarray(idx, np.int64)
        if len(self.packs) == 1:
            return take(self.packs[0], idx, targets)
        shard = np.searchsorted(self.starts, idx, side="right") - 1
        order = np.argsort(shard, kind="stable")
        idx, shard = idx[order], shard[order]
        parts = []
        for s in np.unique(shard):
            parts.append(take(self.packs[s], idx[shard == s] - self.starts[s], targets))
        return concat_packs(parts)

    def collate(self, idx, targets: bool = True) -> Batch:
        """Records idx to a Batch. With several memory-mapped shards the states come grouped by shard
        (in the given order within each shard); with one shard or in_ram, in the given order."""
        return to_batch(self.take(idx, targets), targets)


def batches(indices: np.ndarray, batch_size: int, shuffle: bool, seed: int = 0, drop_last: bool = False,
            sort_within: bool = True) -> list[np.ndarray]:
    idx = np.asarray(indices, np.int64)
    if shuffle:
        idx = np.random.default_rng(seed).permutation(idx)
    out = [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
    if drop_last and out and len(out[-1]) < batch_size:
        out.pop()
    if sort_within:   # memmap locality; a batch is a set
        out = [np.sort(b) for b in out]
    return out


class _Batches(torch.utils.data.Dataset):
    """Index batch k -> its collated Batch (the dataset of a process-based Loader)."""

    def __init__(self, ds: ShardSet, index_batches: list[np.ndarray], targets: bool):
        self.ds, self.batches, self.targets = ds, index_batches, targets

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, k: int) -> FlatBatch:   # a few shared-memory buffers to send, not ~30
        return self.ds.collate(self.batches[k], self.targets).flatten()


def _as_is(x):
    return x


def _check_shm(need: int) -> None:
    try:
        st = os.statvfs("/dev/shm")
    except OSError:
        return
    free = st.f_bavail * st.f_frsize
    if free < need:
        print(f"dzg.data: warning: /dev/shm has {free / 2**20:.0f} MB free, and worker processes may need about "
              f"{need / 2**20:.0f} MB for batches in flight; use fewer --loader-procs or a larger /dev/shm",
              file=sys.stderr, flush=True)


class Loader:
    """Prefetching over precomputed index batches, delivered in order. With procs = 0, `workers` threads
    collate batches k, k+workers, ... (they share the GIL: past about 2 threads they stop helping). With
    procs > 0, that many worker processes collate (a torch DataLoader) and send each batch back as a few
    shared-memory buffers (/dev/shm). Workers fork on Linux. Elsewhere they spawn and reopen the shards
    (with in_ram, each loads its own copy), and on macOS starting and stopping them costs seconds per
    Loader (each worker stops on torch's 5 s join timeout), so procs is for Linux training machines.
    `start_method` overrides the choice."""

    def __init__(self, ds: ShardSet, index_batches: list[np.ndarray], prefetch: int = 4, workers: int = 1,
                 pin: bool = False, targets: bool = True, procs: int = 0, start_method: str | None = None):
        self.ds, self.batches = ds, index_batches
        self.prefetch, self.workers, self.pin, self.targets = prefetch, max(1, workers), pin, targets
        self.procs = max(0, procs)
        self.start_method = start_method or ("fork" if sys.platform.startswith("linux") else "spawn")

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        if self.procs:
            return self._iter_procs()
        return self._iter_threads()

    def _iter_procs(self):
        import multiprocessing as mp
        per = max(2, -(-self.prefetch // self.procs))
        if sys.platform.startswith("linux") and self.batches:
            per_batch = 16000 * max(len(b) for b in self.batches)   # bytes: a collated state is about 10 KB
            _check_shm(per_batch * self.procs * (per + 1))
        dl = torch.utils.data.DataLoader(
            _Batches(self.ds, self.batches, self.targets), batch_size=None, shuffle=False, num_workers=self.procs,
            collate_fn=_as_is, pin_memory=self.pin, prefetch_factor=per,
            multiprocessing_context=mp.get_context(self.start_method))
        for fb in dl:
            yield fb.unflatten()

    def _iter_threads(self):
        stop = threading.Event()
        qs = [queue.Queue(maxsize=max(1, self.prefetch // self.workers + 1)) for _ in range(self.workers)]

        def work(w: int):
            try:
                for k in range(w, len(self.batches), self.workers):
                    if stop.is_set():
                        return
                    b = self.ds.collate(self.batches[k], self.targets)
                    if self.pin:
                        b = b.pin_memory()
                    while not stop.is_set():
                        try:
                            qs[w].put((k, b), timeout=0.1)
                            break
                        except queue.Full:
                            continue
            except BaseException as e:  # surface collate errors to the consumer
                qs[w].put((-1, e))

        threads = [threading.Thread(target=work, args=(w,), daemon=True) for w in range(self.workers)]
        for t in threads:
            t.start()
        try:
            for k in range(len(self.batches)):
                got_k, b = qs[k % self.workers].get()
                if got_k == -1:
                    raise b
                yield b
        finally:
            stop.set()
            for t in threads:
                t.join(timeout=5)
