"""Experiment #4's imitation tables (docs/017 §2): top players' games, every decision turn replayed
with every stop that offers a real choice, the player's block decisions, and the heuristic's score
on every row.

    python tools/imitation_scale/build.py splits [--sbv1 <items.jsonl.gz>]   # row -> exp4 split
    python tools/imitation_scale/build.py build --workers 28 [--limit N]     # shards
    python tools/imitation_scale/build.py tables                             # shards -> HDF5

Splits. The components of docs/011 (drafts joined by a mirrored game) hashed with #2b's salt, so
the hash is #2b's. #2b split it 0.82 / 0.05 / 0.13 (train / val / test); experiment #4 takes
val = [0.82, 0.87), test = [0.87, 0.92) and train = the rest (90 / 5 / 5). The new test split lies
inside #2b's old test range, so #2b's network can be scored on it as a baseline (#2b's own split
lacked the mirrored pairs, so score it only on rows its own hash put in test). Every draft of
experiment #3's benchmark games (sb-v1), and of their mirrored partners, is held out (-1), at the
draft level as docs/016 §9.1 asked. Top players (user_game_win_rate_bucket >= 0.60) are chosen at
build time.

Records (one shard each under data/imitation_scale/shards, a gzip pickle stream of per-game lists):
  ts   turn starts: imitation.turn_start_record with the heuristic's score;
  rp   every decision turn replayed (imitation.replay_records, allStops, heuristic): the player's
       priority decisions at every stop with a real choice, attacks, targets;
  bl   blocks: the first block question of each opponent attack (reconstruct.state_after_user_turn
       at declare_attackers), labelled with coach.human_17lands (exact pairings only);
  op   every opponent turn after a decision turn replayed (turnreplay.replay_opp_turn, the user
       recorded): the user's priority decisions at every stop with a real choice in it (instants,
       flash, abilities, answers to the stack), and every block question.

17lands public data is CC BY 4.0 (https://www.17lands.com/public_datasets).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from draftzero.gameplay import imitation as im

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "imitation_scale"
SPLIT_FILE = OUT / "row_split_exp4.npy"
SBV1 = REPO / "data" / "search_bench" / "sb-v1" / "items.jsonl.gz"
TOP = 0.60
IMPUTED_WEIGHT = 0.5          # training weight of a row whose label is imputed (docs/017 §2.2)
SPLITS = im.SPLITS


# ================================================================================================
# splits
# ================================================================================================

def component_hash(key: str) -> float:
    """#2b's salted hash of a component key, in [0, 1)."""
    return int.from_bytes(hashlib.sha1((im.SPLIT_SALT + "|" + key).encode()).digest()[:8], "big") / 2 ** 64


def exp4_split(h: float) -> int:
    """0 train / 1 val / 2 test: val [0.82, 0.87) and test [0.87, 0.92), #2b's old val range and the
    first part of its old test range; train is everything else."""
    if 0.82 <= h < 0.87:
        return 1
    if 0.87 <= h < 0.92:
        return 2
    return 0


def sbv1_rows(path: Path) -> set[int]:
    with gzip.open(path, "rt") as f:
        return {json.loads(line)["row"] for line in f if line.strip()}


def make_splits(sbv1: Path | None, path=None) -> tuple[np.ndarray, dict]:
    from draftzero.gameplay import pairs as pm
    drafts = im.draft_ids_by_row(path)
    prs = [(p.row_a, p.row_b) for p in pm.load_pairs()]
    uf = im._UnionFind()
    for a, b in prs:
        if drafts[a] and drafts[b]:
            uf.union(drafts[a], drafts[b])

    def comp(i: int) -> str:
        d = drafts[i]
        if not d:
            return f"row:{i}"
        return uf.find(d) if d in uf.parent else d
    # held out at the draft level (docs/016 §9.1): each benchmark game's draft (its player and deck)
    # and its mirrored partner's draft. Components chain drafts through mirrored games (the largest
    # holds about 10k rows), so holding out whole components would drop ~10% of the file.
    held = set()
    if sbv1 is not None:
        partner = {}
        for a, b in prs:
            partner[a], partner[b] = b, a
        for r in sbv1_rows(sbv1):
            held.add(drafts[r] or f"row:{r}")
            if r in partner:
                held.add(drafts[partner[r]] or f"row:{partner[r]}")
    out = np.empty(len(drafts), np.int8)
    cache: dict[str, int] = {}
    for i in range(len(drafts)):
        if (drafts[i] or f"row:{i}") in held:
            out[i] = -1
            continue
        c = comp(i)
        s = cache.get(c)
        if s is None:
            s = cache[c] = exp4_split(component_hash(c))
        out[i] = s
    stats = {"rows": len(drafts), "held_out_drafts": len(held), "sbv1": str(sbv1) if sbv1 else None,
             "rows_by_split": {str(k): int(v) for k, v in zip(*np.unique(out, return_counts=True))}}
    return out, stats


# ================================================================================================
# records
# ================================================================================================

def block_record(g, n: int, b, ids, split: int = -1) -> dict:
    """The player's first block question in the opponent's turn after user turn n, or a status."""
    from draftzero.gameplay import coach as co
    from draftzero.gameplay import reconstruct as rc
    from draftzero.gameplay.bridge import BridgeError
    rec = {"turn": n, "split": split, **im._decision_meta(g)}
    try:
        spec = rc.state_after_user_turn(g, n, "declare_attackers", ids=ids, labels=True)
    except ValueError as e:
        rec.update(status="no_attack" if "did not attack" in str(e) else "spec_error", error=str(e)[:200])
        return rec
    except Exception as e:           # noqa: BLE001 - a reconstruction failure is data, not a crash
        rec.update(status="spec_error", error=f"{type(e).__name__}: {e}"[:300])
        return rec
    pv = spec.provenance
    rec.update(tier=pv.tier if pv else None, flags=list(pv.flags) if pv else [],
               global_turn=spec.labels.get("global_turn"))
    opts = dict(spec.labels.get("bridge") or {})
    try:
        r = b.encode(spec, perfectInfo=False, heuristic=True, dump=True, **opts)
    except BridgeError as e:
        rec.update(status="bridge_error", error=str(e).splitlines()[0][:300])
        return rec
    d = r.get("decision")
    if not d:
        rec.update(status="no_decision", error=(r.get("noDecision") or "")[:200])
        return rec
    w = d.get("where") or {}
    if d.get("type") != "CHOOSE_TARGET" or w.get("step") != "DECLARE_BLOCKERS":
        rec.update(status="wrong_decision", error=f"{d.get('type')} at {w.get('step')}")
        return rec
    legal = d.get("legal") or []
    rec["legal"] = [x["label"] for x in legal]
    rec["legal_idx"] = [int(x.get("idx", -1)) for x in legal]
    decision = {"type": d.get("type"), "text": d.get("text") or "", "turn": w.get("turn"), "step": w.get("step")}
    ha = co._as_human(co.human_17lands(spec.labels), decision, co.alias_names(r.get("dump")))
    if ha is None or not ha.labels:
        rec.update(status="no_label", error=str(getattr(ha, "reason", None))[:200])
        return rec
    S = list(dict.fromkeys(m for m in (co.match_label(a, rec["legal"], "CHOOSE_TARGET") for a in ha.labels) if m))
    if not S:
        rec.update(status="unmatched", error=f"{ha.labels} not in {rec['legal'][:6]}"[:200])
        return rec
    rec.update(status="ok", text=decision["text"], S=S, heuristic=r.get("heuristic"), features=im._feat(r))
    return rec


def opp_replay_records(g, n: int, b, ids, split: int = -1) -> dict:
    """The opponent's turn right after user turn n replayed (turnreplay.replay_opp_turn): the user's
    decisions in it, at every stop with a real choice (its instants, flash, abilities, answers to
    the stack) and every block question, in imitation.replay_records' shape."""
    from draftzero.gameplay import turnreplay as tr
    out = {"turn": n, "split": split, "side": "opp", **im._decision_meta(g)}
    try:
        r = tr.replay_opp_turn(b, g, n, ids, encode=True, allStops=True, heuristic=True)
    except Exception as e:           # noqa: BLE001 - a failed replay is data, not a crash
        out.update(reproduced=False, error=f"{type(e).__name__}: {e}"[:300], decisions=[])
        return out
    out.update(reproduced=bool(r.get("reproduced")), attempt=r.get("attempt"), tier=(r.get("meta") or {}).get("tier"),
               diffKeys=(r.get("diffKeys") or [])[:6], attack_ctx={"blockers": []})
    out["decisions"] = [{"type": d.get("type"), "text": d.get("text"),
                         "legal": [x["label"] for x in d.get("legal") or []],
                         "legal_idx": [int(x.get("idx", -1)) for x in d.get("legal") or []],
                         "chosen": d.get("chosen"), "label_kind": d.get("label_kind"), "evidence": d.get("evidence"),
                         "set": d.get("set"), "step": (d.get("where") or {}).get("step"),
                         "stack": (d.get("where") or {}).get("stack"), "heuristic": d.get("heuristic"),
                         "features": im._feat(d) if out["reproduced"] else None}
                        for d in r.get("decisions") or []]
    return out


_W: dict = {}


def _work(task: tuple) -> dict:
    """One game: its turn starts, every decision turn replayed, and its block questions."""
    from draftzero.gameplay.replay import parse_game
    row, line, split = task
    t0 = time.monotonic()
    try:
        g = parse_game(next(csv.reader([line])), im._W["H"], row)
    except Exception as e:           # noqa: BLE001
        return {"row": row, "error": f"parse: {e}", "ts": [], "rp": [], "bl": [], "op": []}
    ids, b = im._W["ids"], im._bridge()
    turns = g.decision_turns()
    ts = [im.turn_start_record(g, n, b, ids, split=split, heuristic=True) for n in turns]
    t1 = time.monotonic()
    rp = [im.replay_records(g, n, b, ids, split=split, allStops=True, heuristic=True) for n in turns]
    t2 = time.monotonic()
    bl = []
    for t in g.turns:
        if t.side == "user" and t.played:
            q = g.next_slot(t.n)
            if q is not None and q.side == "oppo" and q.played and q.L("creatures_attacked"):
                bl.append(block_record(g, t.n, b, ids, split=split))
    t3 = time.monotonic()
    op = []
    for n in turns:
        q = g.next_slot(n)
        if q is not None and q.side == "oppo" and q.played and not q.terminal:
            op.append(opp_replay_records(g, n, b, ids, split=split))
    return {"row": row, "ts": ts, "rp": rp, "bl": bl, "op": op,
            "sec": {"ts": t1 - t0, "rp": t2 - t1, "bl": t3 - t2, "op": time.monotonic() - t3}}


def build(workers: int, limit: int | None = None, heap: str = "2500m", top: float = TOP, log=print) -> dict:
    """Every top player's game in a split (not held out), `workers` processes with one bridge JVM
    each. Shards ts.pkl / rp.pkl / bl.pkl under data/imitation_scale/shards."""
    from draftzero.gameplay.replay import open_lines, replay_path
    splits = np.load(SPLIT_FILE)
    out_dir = OUT / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    H, lines = open_lines(replay_path())
    wr_col = H.cols.index("user_game_win_rate_bucket")
    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    seen = Counter()

    def tasks():
        k = 0
        for i, line in enumerate(lines):
            seen["rows"] += 1
            if splits[i] < 0:
                seen["held_out"] += 1
                continue
            v = line.split(",")[wr_col]       # data rows hold no quoted commas (checked on 2,000 rows)
            if not v or float(v) < top:
                continue
            if limit is not None and k >= limit:
                break
            k += 1
            yield (i, line, int(splits[i]))

    stats = Counter()
    sec = Counter()
    t0 = time.monotonic()
    # gzip level 1: the features are int arrays that compress about 3x, at little cost in time
    files = {k: gzip.open(out_dir / f"{k}.pkl.gz", "wb", compresslevel=1) for k in ("ts", "rp", "bl", "op")}
    try:
        with ctx.Pool(workers, initializer=im._worker_init,
                      initargs=(H.cols, str(out_dir / "runtime"), heap, counter)) as pool:
            for res in pool.imap_unordered(_work, tasks(), chunksize=2):
                for k, f in files.items():
                    pickle.dump(res[k], f, protocol=pickle.HIGHEST_PROTOCOL)
                stats["games"] += 1
                stats["errors"] += "error" in res
                stats["ts"] += len(res["ts"])
                stats["ts_ok"] += sum(r.get("status") == "ok" for r in res["ts"])
                stats["rp"] += len(res["rp"])
                stats["rp_ok"] += sum(bool(r.get("reproduced")) for r in res["rp"])
                stats["rp_decisions"] += sum(len(r.get("decisions") or []) for r in res["rp"] if r.get("reproduced"))
                stats["bl"] += len(res["bl"])
                stats["bl_ok"] += sum(r.get("status") == "ok" for r in res["bl"])
                stats["op"] += len(res["op"])
                stats["op_ok"] += sum(bool(r.get("reproduced")) for r in res["op"])
                stats["op_decisions"] += sum(len(r.get("decisions") or []) for r in res["op"] if r.get("reproduced"))
                for k, v in (res.get("sec") or {}).items():
                    sec[k] += v
                if stats["games"] % 500 == 0:
                    el = time.monotonic() - t0
                    log(f"{stats['games']} games in {el:.0f} s ({stats['games'] / el:.1f}/s): {stats['ts_ok']} turn starts, "
                        f"{stats['rp_ok']}/{stats['rp']} turns reproduced ({stats['rp_decisions']} decisions), "
                        f"{stats['bl_ok']}/{stats['bl']} blocks, {stats['op_ok']}/{stats['op']} opponent turns "
                        f"({stats['op_decisions']} decisions)", flush=True)
                    for f in files.values():
                        f.flush()
    finally:
        for f in files.values():
            f.close()
        lines.close()
    el = time.monotonic() - t0
    out = {**stats, "seconds": round(el, 1), "games_per_s": round(stats["games"] / max(el, 1e-9), 2),
           "worker_seconds": {k: round(v, 1) for k, v in sec.items()}, "workers": workers, "top": top,
           "rows_scanned": seen["rows"], "rows_held_out": seen["held_out"]}
    (out_dir / "build_stats.json").write_text(json.dumps(out, indent=1))
    return out


# ================================================================================================
# tables
# ================================================================================================

def load_shard(path: Path) -> list:
    """A shard written by `build`: a gzip pickle stream of per-game lists (a partly written one
    loads up to its last complete game)."""
    out = []
    with gzip.open(path, "rb") as f:
        while True:
            try:
                out.extend(pickle.load(f))
            except (EOFError, pickle.UnpicklingError, OSError):
                break
    return out


STEPS = ("UNTAP", "UPKEEP", "DRAW", "PRECOMBAT_MAIN", "BEGIN_COMBAT", "DECLARE_ATTACKERS", "DECLARE_BLOCKERS",
         "FIRST_COMBAT_DAMAGE", "COMBAT_DAMAGE", "END_COMBAT", "POSTCOMBAT_MAIN", "END_TURN", "CLEANUP")


def _append(path: Path, **arrays) -> None:
    import h5py
    with h5py.File(path, "a") as f:
        for k, v in arrays.items():
            if k in f:
                del f[k]
            f.create_dataset(k, data=v)


def _heur(xs) -> np.ndarray:
    return np.asarray([x if x is not None else np.nan for x in xs], np.float32)


def _target_table(recs: list[dict], keep, prefix: str, h5: Path, log) -> None:
    """CHOOSE_TARGET decisions of reproduced replayed turns that `keep` accepts -> <prefix>_<split>.h5
    (one-hot labels in the set CSR, as block_* and turnstart_*)."""
    tg = defaultdict(list)
    for t in recs:
        if not t.get("reproduced"):
            continue
        for d in t.get("decisions") or []:
            if d["type"] != "CHOOSE_TARGET" or d.get("features") is None or not keep(d):
                continue
            tg[t.get("split")].append({**{k: t.get(k) for k in ("row", "turn", "split", "won", "wr_bucket",
                                                                 "n_games_bucket", "on_play", "tier")},
                                       "legal": d["legal"], "legal_idx": d["legal_idx"], "S": [d.get("chosen")],
                                       "features": d["features"], "heuristic": d.get("heuristic"),
                                       "label_status": "set", "acts": [], "attacked": False, "status": "ok"})
    for i, s in enumerate(SPLITS):
        rs = [r for r in tg.get(i, []) if r["S"][0] in r["legal"] and len(set(r["legal"])) >= 2]
        if not rs:
            log(f"{prefix}_{s}: no rows")
            continue
        t = im.ts_table(rs)
        t["lab_idx"] = [r["legal_idx"] for r in rs]
        path = h5 / f"{prefix}_{s}.h5"
        im._save_table(t, path, action_type=im.ACTION_TYPE["CHOOSE_TARGET"])
        _append(path, **{"meta/heuristic": _heur(r.get("heuristic") for r in rs)})
        log(f"{prefix}_{s}: {len(rs)} rows")


def _priority_tables(pri: dict, prefix: str, h5: Path, log) -> None:
    """imitation.replay_tables' priority part -> <prefix>_<split>.h5, with meta/step, meta/stack,
    meta/heuristic and a per-row weight (exact 1, imputed IMPUTED_WEIGHT)."""
    if not pri:
        return
    sp = np.asarray(pri["meta/split"])
    for i, s in enumerate(SPLITS):
        sel = np.flatnonzero(sp == i)
        if not len(sel):
            log(f"{prefix}_{s}: no rows")
            continue
        t = im.priority_table_from_replay(pri, sel)
        t["z"] = np.where(np.asarray([pri["meta/won"][j] for j in sel], bool), 1.0, -1.0).astype(np.float32)
        t["S"] = [[] for _ in sel]
        for k in ("won", "n_games_bucket", "tier", "on_play"):
            vals = [pri["meta/" + k][j] for j in sel]
            if k == "tier":
                vals = [im.TIERS.index(v) if v in im.TIERS else -1 for v in vals]
            t["meta/" + k] = np.asarray([v if v is not None else -1 for v in vals], np.int32)
        t["meta/wr_bucket"] = np.asarray([pri["meta/wr_bucket"][j] if pri["meta/wr_bucket"][j] is not None
                                          else np.nan for j in sel], np.float32)
        path = h5 / f"{prefix}_{s}.h5"
        im._save_table(t, path)
        kind = [pri["kind"][j] for j in sel]
        _append(path, weight=np.asarray([1.0 if k == "exact" else IMPUTED_WEIGHT for k in kind], np.float32),
                **{"meta/step": np.asarray([STEPS.index(pri["step"][j]) if pri["step"][j] in STEPS else -1 for j in sel], np.int32),
                   "meta/stack": np.asarray([pri["stack"][j] for j in sel], np.int32),
                   "meta/heuristic": _heur(pri["heuristic"][j] for j in sel)})
        log(f"{prefix}_{s}: {len(sel)} rows")


def tables(log=print) -> dict:
    """Shards -> HDF5 tables in imitation.write_h5's layout, per split: turnstart_*, replay_priority_*
    (with meta/step, meta/stack and a per-row weight: exact 1, imputed IMPUTED_WEIGHT),
    replay_attack_*, replay_target_* (spell targets the outcome settles) and block_* (both
    CHOOSE_TARGET, legal and set CSR); meta/heuristic on every row."""
    import h5py
    from draftzero.gameplay.ids import Ids
    ids = Ids.load()
    sh, h5 = OUT / "shards", OUT / "h5"
    out: dict = {}
    # turn starts
    ts = load_shard(sh / "ts.pkl.gz")
    out["turnstart_status"] = dict(Counter(r.get("status") for r in ts))
    for i, s in enumerate(SPLITS):
        rs = [r for r in ts if r.get("split") == i and r.get("status") == "ok"]
        if not rs:
            log(f"turnstart_{s}: no rows")
            continue
        t = im.ts_table(rs)
        t["lab_idx"] = [r["legal_idx"] for r in rs]
        im._save_table(t, h5 / f"turnstart_{s}.h5")
        _append(h5 / f"turnstart_{s}.h5", **{"meta/heuristic": _heur(r.get("heuristic") for r in rs)})
        log(f"turnstart_{s}: {len(rs)} rows")
    del ts
    # replayed turns: the user's own (rp) and the opponent's after them (op)
    rp = load_shard(sh / "rp.pkl.gz")
    op = load_shard(sh / "op.pkl.gz") if (sh / "op.pkl.gz").exists() else []
    for name, recs in (("replay", rp), ("opp", op)):
        out[f"{name}_turns"] = len(recs)
        out[f"{name}_reproduced"] = sum(bool(t.get("reproduced")) for t in recs)
        kinds = Counter()
        for t in recs:
            if t.get("reproduced"):
                for d in t.get("decisions") or []:
                    kinds[f"{d['type']}:{d.get('label_kind')}:{d.get('step')}"] += 1
        out[f"{name}_decision_kinds"] = dict(kinds.most_common())

    def is_block(d):
        return d["type"] == "CHOOSE_TARGET" and (d.get("text") or "").startswith("choose which creature to block for")
    # spell targets the outcome settles, in either turn; the user's blocks in the opponent's turns
    _target_table([t for t in rp + op], lambda d: d.get("label_kind") == "exact" and not is_block(d)
                  and d.get("step") != "DECLARE_BLOCKERS", "replay_target", h5, log)
    _target_table(op, lambda d: d.get("label_kind") == "exact" and is_block(d), "opp_block", h5, log)
    rt = im.replay_tables(rp, ids)
    del rp
    att = rt["attack"]
    sp = np.asarray(att["meta/split"])
    for i, s in enumerate(SPLITS):
        sel = np.flatnonzero(sp == i)
        if not len(sel):
            log(f"replay_attack_{s}: no rows")
            continue
        ind, off = im.csr([att["features"][j] for j in sel])
        path = h5 / f"replay_attack_{s}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("indices", data=ind, compression="gzip", compression_opts=1)
            f.create_dataset("offsets", data=off)
            row = np.zeros((len(sel), im.A_DIM + 4), np.float32)
            y = np.asarray(att["y"])[sel]
            row[np.arange(len(sel)), y] = 1.0
            won = np.asarray(att["meta/won"])[sel].astype(bool)
            row[:, im.A_DIM] = np.where(won, 1.0, -1.0)
            row[:, im.A_DIM + 2] = 1.0
            row[:, im.A_DIM + 3] = im.ACTION_TYPE["CHOOSE_USE"]
            f.create_dataset("row", data=row, compression="gzip", compression_opts=1)
            f.create_dataset("y", data=y)
            f.create_dataset("heur", data=np.asarray(att["heur"])[sel])
            f.create_dataset("power", data=np.asarray(att["power"])[sel])
            for k in ("row", "turn", "won", "n_games_bucket", "on_play"):
                f.create_dataset("meta/" + k, data=np.asarray([att["meta/" + k][j] or 0 for j in sel], np.int32))
            f.create_dataset("meta/wr_bucket", data=np.asarray(
                [att["meta/wr_bucket"][j] if att["meta/wr_bucket"][j] is not None else np.nan for j in sel], np.float32))
            f.create_dataset("meta/heuristic", data=_heur(att["heuristic"][j] for j in sel))
            f.attrs["layout"] = "LabeledStateWriter (CHOOSE_USE rows) + y/heur (draftzero.gameplay.imitation)"
        log(f"replay_attack_{s}: {len(sel)} rows")
    _priority_tables(rt["priority"], "replay_priority", h5, log)
    del rt, att
    if op:
        _priority_tables(im.replay_tables(op, ids)["priority"], "opp_priority", h5, log)
    del op
    # blocks
    bl = load_shard(sh / "bl.pkl.gz")
    out["block_status"] = dict(Counter(r.get("status") for r in bl))
    for i, s in enumerate(SPLITS):
        rs = [r for r in bl if r.get("split") == i and r.get("status") == "ok"]
        if not rs:
            log(f"block_{s}: no rows")
            continue
        recs = [{**r, "label_status": "set", "acts": [], "attacked": False} for r in rs]
        t = im.ts_table(recs)
        t["lab_idx"] = [r["legal_idx"] for r in rs]
        path = h5 / f"block_{s}.h5"
        im._save_table(t, path, action_type=im.ACTION_TYPE["CHOOSE_TARGET"])
        _append(path, **{"meta/heuristic": _heur(r.get("heuristic") for r in rs)})
        log(f"block_{s}: {len(rs)} rows")
    (OUT / "tables_stats.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("splits")
    s.add_argument("--sbv1", type=Path, default=SBV1, help="experiment #3's items, whose drafts are held out")
    b = sub.add_parser("build")
    b.add_argument("--workers", type=int, default=3)
    b.add_argument("--limit", type=int)
    b.add_argument("--heap", default="2500m")
    sub.add_parser("tables")
    a = ap.parse_args(argv)
    if a.cmd == "splits":
        codes, stats = make_splits(a.sbv1 if a.sbv1 and a.sbv1.exists() else None)
        if stats["sbv1"] is None:
            raise SystemExit(f"no sb-v1 items at {a.sbv1}: its drafts must be held out (pass --sbv1)")
        OUT.mkdir(parents=True, exist_ok=True)
        np.save(SPLIT_FILE, codes)
        (OUT / "row_split_exp4.stats.json").write_text(json.dumps(stats, indent=1))
        print(json.dumps(stats, indent=1))
    elif a.cmd == "build":
        print(json.dumps(build(a.workers, a.limit, a.heap), indent=1))
    else:
        print(json.dumps(tables(), indent=1, default=str))


if __name__ == "__main__":
    main()
