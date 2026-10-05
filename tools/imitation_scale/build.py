"""Experiment #4's imitation tables (docs/017 §2): top players' games, every decision turn replayed
with every stop that offers a real choice, the player's block decisions, and the heuristic's score
on every row.

    python tools/imitation_scale/build.py splits [--sbv1 <items.jsonl.gz>]   # row -> exp4 split
    python tools/imitation_scale/build.py build --workers 28 [--limit N]     # shard parts; re-run resumes
    python tools/imitation_scale/build.py tables                             # shard parts -> HDF5; re-run resumes

With --graph, build also records every decision as MageZero's graph encoder sees it, and tables writes
a graph file beside each table, row for row (draftzero.gameplay.graph_tables; docs/022). --out puts a
build somewhere other than data/imitation_scale (the splits file stays there):

    python tools/imitation_scale/build.py build --graph --out data/imitation_graph --workers 28
    python tools/imitation_scale/build.py tables --graph --out data/imitation_graph

Two tools for such a build (docs/022 §4.1):

    python tools/imitation_scale/build.py compare data/imitation_graph/h5 data/imitation_scale/h5   # the same rows?
    python tools/imitation_scale/build.py slim data/imitation_graph/h5 data/imitation_graph/slim    # labels + graphs

`compare` checks two builds' flat tables hold the same rows game by game (the order of games differs between
runs: the workers finish them in any order). `slim` copies the graph files and the flat tables without their
features (`indices`, `row`), which is all the graph trainer reads: about a third of the disk.

Splits. The components of docs/011 (drafts joined by a mirrored game) hashed with #2b's salt, so
the hash is #2b's. #2b split it 0.82 / 0.05 / 0.13 (train / val / test); experiment #4 takes
val = [0.82, 0.87), test = [0.87, 0.92) and train = the rest (90 / 5 / 5). The new test split lies
inside #2b's old test range, so #2b's network can be scored on it as a baseline (#2b's own split
lacked the mirrored pairs, so score it only on rows its own hash put in test). Every draft of
experiment #3's benchmark games (sb-v1), and of their mirrored partners, is held out (-1), at the
draft level as docs/016 §9.1 asked. Top players (user_game_win_rate_bucket >= 0.60) are chosen at
build time.

Records (under data/imitation_scale/shards, in parts of PART_GAMES games: <kind>.<part>.pkl.gz, each a
gzip pickle stream of per-game lists, and part.<part>.json once the part is finished):
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
import os
import pickle
import shutil
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

def block_record(g, n: int, b, ids, split: int = -1, graph: bool = False) -> dict:
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
        r = b.encode(spec, perfectInfo=False, heuristic=True, dump=True, **({"graph": True} if graph else {}), **opts)
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
    if graph:
        rec["graph"] = im._graph(r)
    return rec


def opp_replay_records(g, n: int, b, ids, split: int = -1, graph: bool = False) -> dict:
    """The opponent's turn right after user turn n replayed (turnreplay.replay_opp_turn): the user's
    decisions in it, at every stop with a real choice (its instants, flash, abilities, answers to
    the stack) and every block question, in imitation.replay_records' shape."""
    from draftzero.gameplay import turnreplay as tr
    out = {"turn": n, "split": split, "side": "opp", **im._decision_meta(g)}
    try:
        r = tr.replay_opp_turn(b, g, n, ids, encode=True, allStops=True, heuristic=True,
                               **({"graph": True} if graph else {}))
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
                         "features": im._feat(d) if out["reproduced"] else None,
                         **({"graph": im._graph(d) if out["reproduced"] else None} if graph else {})}
                        for d in r.get("decisions") or []]
    return out


_W: dict = {}


def _work(task: tuple) -> dict:
    """One game: its turn starts, every decision turn replayed, and its block questions."""
    from draftzero.gameplay.replay import parse_game
    row, line, split = task[:3]
    graph = len(task) > 3 and bool(task[3])
    gopt = {"graph": True} if graph else {}
    t0 = time.monotonic()
    try:
        g = parse_game(next(csv.reader([line])), im._W["H"], row)
    except Exception as e:           # noqa: BLE001
        return {"row": row, "error": f"parse: {e}", "ts": [], "rp": [], "bl": [], "op": []}
    ids, b = im._W["ids"], im._bridge()
    turns = g.decision_turns()
    ts = [im.turn_start_record(g, n, b, ids, split=split, heuristic=True, **gopt) for n in turns]
    t1 = time.monotonic()
    rp = [im.replay_records(g, n, b, ids, split=split, allStops=True, heuristic=True, **gopt) for n in turns]
    t2 = time.monotonic()
    bl = []
    for t in g.turns:
        if t.side == "user" and t.played:
            q = g.next_slot(t.n)
            if q is not None and q.side == "oppo" and q.played and q.L("creatures_attacked"):
                bl.append(block_record(g, t.n, b, ids, split=split, graph=graph))
    t3 = time.monotonic()
    op = []
    for n in turns:
        q = g.next_slot(n)
        if q is not None and q.side == "oppo" and q.played and not q.terminal:
            op.append(opp_replay_records(g, n, b, ids, split=split, graph=graph))
    return {"row": row, "ts": ts, "rp": rp, "bl": bl, "op": op,
            "sec": {"ts": t1 - t0, "rp": t2 - t1, "bl": t3 - t2, "op": time.monotonic() - t3}}


SHARD_KINDS = ("ts", "rp", "bl", "op")
PART_GAMES = 5000             # games per shard part: `tables` holds one part in memory (~1.3 MB a game)


def _game_stats(res: dict) -> Counter:
    s = Counter(games=1, errors=int("error" in res))
    s["ts"] += len(res["ts"])
    s["ts_ok"] += sum(r.get("status") == "ok" for r in res["ts"])
    s["rp"] += len(res["rp"])
    s["rp_ok"] += sum(bool(r.get("reproduced")) for r in res["rp"])
    s["rp_decisions"] += sum(len(r.get("decisions") or []) for r in res["rp"] if r.get("reproduced"))
    s["bl"] += len(res["bl"])
    s["bl_ok"] += sum(r.get("status") == "ok" for r in res["bl"])
    s["op"] += len(res["op"])
    s["op_ok"] += sum(bool(r.get("reproduced")) for r in res["op"])
    s["op_decisions"] += sum(len(r.get("decisions") or []) for r in res["op"] if r.get("reproduced"))
    return s


def _part_marker(out_dir: Path, i: int) -> Path:
    return out_dir / f"part.{i:05d}.json"


def completed_parts(out_dir: Path) -> list[dict]:
    """The finished shard parts in `out_dir`, in order: {"i", "paths": {kind: path}, "rows", "stats",
    "sec", "seconds"}. A part is finished once its marker is written, after its files are closed. A
    build from before parts existed (ts.pkl.gz, ...) is one part with no rows recorded."""
    parts = []
    for m in sorted(out_dir.glob("part.*.json")):
        d = json.loads(m.read_text())
        d["paths"] = {k: out_dir / f"{k}.{d['i']:05d}.pkl.gz" for k in SHARD_KINDS}
        parts.append(d)
    if not parts and (out_dir / "ts.pkl.gz").exists():
        parts.append({"i": -1, "rows": [], "stats": {}, "sec": {}, "seconds": None, "legacy": True,
                      "paths": {k: out_dir / f"{k}.pkl.gz" for k in SHARD_KINDS}})
    return parts


class PartWriter:
    """Writes `build`'s per-game results as shard parts of `part_games` games each: <kind>.<i>.pkl.gz
    for kind in ts / rp / bl / op (gzip pickle streams of per-game lists), then part.<i>.json (rows,
    stats). A stopped build loses only its unfinished part: resuming drops the unmarked files and
    skips the rows of the marked parts."""

    def __init__(self, out_dir: Path, part_games: int = PART_GAMES):
        self.dir, self.part_games = out_dir, part_games
        out_dir.mkdir(parents=True, exist_ok=True)
        done = completed_parts(out_dir)
        if any(p.get("legacy") for p in done):
            raise SystemExit(f"{out_dir} holds a build from before shard parts (ts.pkl.gz ...): move it away first")
        keep = {p["i"] for p in done}
        for k in SHARD_KINDS:                       # an unfinished part's files
            for f in out_dir.glob(f"{k}.*.pkl.gz"):
                if int(f.name.split(".")[1]) not in keep:
                    f.unlink()
        self.done_rows = {r for p in done for r in p["rows"]}
        self.next_i = max(keep, default=-1) + 1
        self.files = None

    def _open(self) -> None:
        self.i = self.next_i
        self.next_i += 1
        # gzip level 1: the features are int arrays that compress about 3x, at little cost in time
        self.files = {k: gzip.open(self.dir / f"{k}.{self.i:05d}.pkl.gz", "wb", compresslevel=1) for k in SHARD_KINDS}
        self.rows, self.stats, self.sec, self.t0 = [], Counter(), Counter(), time.monotonic()

    def add(self, res: dict) -> None:
        if self.files is None:
            self._open()
        for k, f in self.files.items():
            pickle.dump(res[k], f, protocol=pickle.HIGHEST_PROTOCOL)
        self.rows.append(int(res["row"]))
        self.stats.update(_game_stats(res))
        self.sec.update(res.get("sec") or {})
        if len(self.rows) >= self.part_games:
            self.close()

    def close(self) -> None:
        if self.files is None:
            return
        for f in self.files.values():
            f.close()
        m = _part_marker(self.dir, self.i)
        tmp = m.with_suffix(".tmp")
        tmp.write_text(json.dumps({"i": self.i, "rows": self.rows, "stats": dict(self.stats),
                                   "sec": {k: round(v, 1) for k, v in self.sec.items()},
                                   "seconds": round(time.monotonic() - self.t0, 1)}))
        os.replace(tmp, m)
        self.files = None


def build(workers: int, limit: int | None = None, heap: str = "2500m", top: float = TOP, log=print,
          part_games: int = PART_GAMES, graph: bool = False, out_root: Path | None = None) -> dict:
    """Every top player's game in a split (not held out), `workers` processes with one bridge JVM
    each. Shard parts under data/imitation_scale/shards (PartWriter); re-running resumes, skipping
    the games of finished parts."""
    from draftzero.gameplay.replay import open_lines, replay_path
    splits = np.load(SPLIT_FILE)
    out_dir = (out_root or OUT) / "shards"
    writer = PartWriter(out_dir, part_games)
    if writer.done_rows:
        log(f"resuming: {len(writer.done_rows)} games in {writer.next_i} finished parts", flush=True)
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
            if i in writer.done_rows:
                seen["resumed"] += 1
                continue
            yield (i, line, int(splits[i]), graph)

    stats = Counter()
    t0 = time.monotonic()
    try:
        with ctx.Pool(workers, initializer=im._worker_init,
                      initargs=(H.cols, str(out_dir / "runtime"), heap, counter)) as pool:
            for res in pool.imap_unordered(_work, tasks(), chunksize=2):
                writer.add(res)
                stats.update(_game_stats(res))
                if stats["games"] % 500 == 0:
                    el = time.monotonic() - t0
                    log(f"{stats['games']} games in {el:.0f} s ({stats['games'] / el:.1f}/s): {stats['ts_ok']} turn starts, "
                        f"{stats['rp_ok']}/{stats['rp']} turns reproduced ({stats['rp_decisions']} decisions), "
                        f"{stats['bl_ok']}/{stats['bl']} blocks, {stats['op_ok']}/{stats['op']} opponent turns "
                        f"({stats['op_decisions']} decisions)", flush=True)
    finally:
        writer.close()
        lines.close()
    el = time.monotonic() - t0
    parts = completed_parts(out_dir)
    total, sec = Counter(), Counter()
    for p in parts:
        total.update(p["stats"])
        sec.update(p["sec"])
    out = {**total, "parts": len(parts), "seconds_this_run": round(el, 1),
           "games_this_run": stats["games"], "games_per_s": round(stats["games"] / max(el, 1e-9), 2),
           "worker_seconds": {k: round(v, 1) for k, v in sec.items()}, "workers": workers, "top": top, "graph": graph,
           "rows_scanned": seen["rows"], "rows_held_out": seen["held_out"], "rows_resumed": seen["resumed"]}
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


def _graph_file(path: Path, graphs: list, labels: list[list[str]], sets: list, rows, turns) -> None:
    """The graph file beside table `path`, row for row: each row's graph and, per legal label, whether
    it is in the row's label set (graph_tables)."""
    from draftzero.gameplay import graph_tables as gt
    flags = [gt.option_labels(lab, st) for lab, st in zip(labels, sets)]
    gt.write(gt.graph_path(path), graphs, [f[0] for f in flags], [f[1] for f in flags], rows, turns)


def _target_table(recs: list[dict], keep, prefix: str, h5: Path, log, graph: bool = False) -> None:
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
                                       "label_status": "set", "acts": [], "attacked": False, "status": "ok",
                                       "graph": d.get("graph")})
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
        if graph:
            _graph_file(path, [r["graph"] for r in rs], [r["legal"] for r in rs], [r["S"] for r in rs],
                        [r["row"] for r in rs], [r["turn"] for r in rs])
        log(f"{prefix}_{s}: {len(rs)} rows")


def _priority_tables(pri: dict, prefix: str, h5: Path, log, graph: bool = False) -> None:
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
        if graph:
            _graph_file(path, [pri["graph"][j] for j in sel], [pri["labels"][j] for j in sel],
                        [pri["S_labels"][j] for j in sel], [pri["meta/row"][j] for j in sel],
                        [pri["meta/turn"][j] for j in sel])
        log(f"{prefix}_{s}: {len(sel)} rows")


CSR_PTRS = {"offsets": "indices", "legal_indptr": "legal_idx", "set_indptr": "set_idx",
            # graph files (graph_tables.GRAPH_PTRS)
            "node_ptr": "node_ids", "edge_ptr": "edge_child", "row_opt_ptr": "opt_in_set", "opt_ptr": "opt_node"}


def _h5_datasets(f) -> list[str]:
    import h5py
    names: list[str] = []
    f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
    return names


def merge_h5(paths: list[Path], out: Path, block_elems: int = 1 << 24) -> int:
    """Concatenate tables of one layout (one table's parts, in order) into `out`, a block at a time.
    The CSR pointers (offsets, legal_indptr, set_indptr) are shifted by the rows before them; every
    other dataset is per row or a CSR's data, so it is concatenated on its first axis. Returns rows."""
    import h5py
    srcs = [h5py.File(p, "r") for p in paths]
    try:
        keys = _h5_datasets(srcs[0])
        for s, p in zip(srcs[1:], paths[1:]):
            if set(_h5_datasets(s)) != set(keys):
                raise ValueError(f"{p}: datasets {sorted(set(_h5_datasets(s)) ^ set(keys))} differ from {paths[0]}")
        tmp = out.with_name(out.name + ".tmp")
        with h5py.File(tmp, "w") as g:
            for k in keys:
                d0 = srcs[0][k]
                if d0.shape == ():
                    raise ValueError(f"{paths[0]}: scalar dataset {k}")
                if k in CSR_PTRS:
                    total = sum(s[k].shape[0] - 1 for s in srcs) + 1
                    ds = g.create_dataset(k, shape=(total,), dtype=d0.dtype)
                    ds[0] = 0
                    pos, base = 1, 0
                    for s in srcs:
                        p = s[k][:].astype(np.int64)
                        ds[pos:pos + len(p) - 1] = p[1:] - p[0] + base
                        pos += len(p) - 1
                        base += int(p[-1] - p[0])
                    continue
                total = sum(s[k].shape[0] for s in srcs)
                is_str = h5py.check_string_dtype(d0.dtype) is not None
                kw = {}
                if d0.compression and total:
                    kw = dict(compression=d0.compression, compression_opts=d0.compression_opts, chunks=True)
                ds = g.create_dataset(k, shape=(total,) + d0.shape[1:],
                                      dtype=h5py.string_dtype() if is_str else d0.dtype, **kw)
                rows_per_block = max(1, block_elems // max(1, int(np.prod(d0.shape[1:]))))
                pos = 0
                for s in srcs:
                    d = s[k]
                    n = d.shape[0]
                    for a in range(0, n, rows_per_block):
                        b = min(n, a + rows_per_block)
                        ds[pos + a:pos + b] = d.asstr()[a:b] if is_str else d[a:b]
                    pos += n
            for name in [""] + [n for n in _h5_datasets(srcs[0])]:
                src = srcs[0][name] if name else srcs[0]
                dst = g[name] if name else g
                for a, v in src.attrs.items():
                    dst.attrs[a] = v
            ptr = "offsets" if "offsets" in g else "node_ptr" if "node_ptr" in g else None
            n_rows = int(g[ptr].shape[0] - 1) if ptr else None
        os.replace(tmp, out)
        return n_rows
    finally:
        for s in srcs:
            s.close()


def _sum_stats(a: dict, b: dict) -> dict:
    """Part stats summed: counts add, dicts of counts add key by key."""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict):
            c = Counter(out.get(k) or {})
            c.update(v)
            out[k] = dict(c.most_common())
        elif isinstance(v, (int, float)):
            out[k] = out.get(k, 0) + v
        else:
            out[k] = v
    return out


def tables(log=print, keep_parts: bool = False, graph: bool = False, out_root: Path | None = None) -> dict:
    """Shard parts -> HDF5 tables (tables_part), one part at a time into h5/parts/<i>/, then each
    table's parts merged into h5/<table>_<split>.h5 (merge_h5). A finished part is marked, so a
    re-run resumes. Memory: one part (~1.3 MB a game) rather than the whole build."""
    root = out_root or OUT
    sh, h5 = root / "shards", root / "h5"
    parts = completed_parts(sh)
    if not parts:
        raise SystemExit(f"no finished shard parts in {sh}")
    pdir = h5 / "parts"
    out: dict = {}
    t0 = time.monotonic()
    for j, p in enumerate(parts):
        d = pdir / f"{max(p['i'], 0):05d}"
        done = d / "done.json"
        if done.exists():
            st = json.loads(done.read_text())
        else:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
            st = tables_part(p["paths"], d, log=lambda *a, **k: None, graph=graph)
            done.write_text(json.dumps(st, default=str))
        out = _sum_stats(out, st)
        log(f"tables: part {j + 1}/{len(parts)} done ({time.monotonic() - t0:.0f} s)", flush=True)
    names = sorted({f.name for p in parts for f in (pdir / f"{max(p['i'], 0):05d}").glob("*.h5")})
    for name in names:
        srcs = [pdir / f"{max(p['i'], 0):05d}" / name for p in parts]
        srcs = [s for s in srcs if s.exists()]
        n = merge_h5(srcs, h5 / name)
        log(f"{name[:-3]}: {n} rows from {len(srcs)} parts", flush=True)
    out["parts"] = len(parts)
    (root / "tables_stats.json").write_text(json.dumps(out, indent=1, default=str))
    if not keep_parts:
        shutil.rmtree(pdir)
    return out


def tables_part(paths: dict, h5: Path, log=print, graph: bool = False) -> dict:
    """One shard part ({kind: path}) -> HDF5 tables in imitation.write_h5's layout under `h5`, per
    split: turnstart_*, replay_priority_* and opp_priority_* (with meta/step, meta/stack and a
    per-row weight: exact 1, imputed IMPUTED_WEIGHT), replay_attack_*, replay_target_* (spell
    targets the outcome settles), opp_block_* and block_* (CHOOSE_TARGET, legal and set CSR);
    meta/heuristic on every row."""
    import h5py
    from draftzero.gameplay.ids import Ids
    ids = Ids.load()
    out: dict = {}
    # turn starts
    ts = load_shard(paths["ts"])
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
        if graph:
            _graph_file(h5 / f"turnstart_{s}.h5", [r["graph"] for r in rs], [r["legal"] for r in rs],
                        [r["S"] if r.get("label_status") != "unreachable" else [] for r in rs],
                        [r["row"] for r in rs], [r["turn"] for r in rs])
        log(f"turnstart_{s}: {len(rs)} rows")
    del ts
    # replayed turns: the user's own (rp) and the opponent's after them (op)
    rp = load_shard(paths["rp"])
    op = load_shard(paths["op"]) if paths["op"].exists() else []
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
                  and d.get("step") != "DECLARE_BLOCKERS", "replay_target", h5, log, graph)
    _target_table(op, lambda d: d.get("label_kind") == "exact" and is_block(d), "opp_block", h5, log, graph)
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
        if graph:   # the graph asks "attack with X?" as a target choice: options [no = Stop Choosing, yes = the defender]
            yl = ["yes" if v else "no" for v in y]
            _graph_file(path, [att["graph"][j] for j in sel], [["no", "yes"]] * len(sel), [[v] for v in yl],
                        [att["meta/row"][j] for j in sel], [att["meta/turn"][j] for j in sel])
        log(f"replay_attack_{s}: {len(sel)} rows")
    _priority_tables(rt["priority"], "replay_priority", h5, log, graph)
    del rt, att
    if op:
        _priority_tables(im.replay_tables(op, ids)["priority"], "opp_priority", h5, log, graph)
    del op
    # blocks
    bl = load_shard(paths["bl"])
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
        if graph:
            _graph_file(path, [r["graph"] for r in rs], [r["legal"] for r in rs], [r["S"] for r in rs],
                        [r["row"] for r in rs], [r["turn"] for r in rs])
        log(f"block_{s}: {len(rs)} rows")
    return out


# ================================================================================================
# comparing builds, slim tables
# ================================================================================================

_M1, _M2 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9)


def row_fingerprints(path: Path, block: int = 1 << 16) -> tuple[np.ndarray, np.ndarray]:
    """(game, fingerprint) per row of a flat table: a hash of the row's turn, features, legal labels, label set and
    yes/no label, independent of where the row sits in the file."""
    import h5py
    with h5py.File(path, "r") as f:
        off = f["offsets"][:].astype(np.int64)
        n = len(off) - 1
        game = f["meta/row"][:].astype(np.int64)
        turn = f["meta/turn"][:].astype(np.uint64) if "meta/turn" in f else np.zeros(n, np.uint64)
        fp = np.zeros(n, np.uint64)
        with np.errstate(over="ignore"):
            for a in range(0, n, block):
                b = min(n, a + block)
                x = f["indices"][off[a]:off[b]].astype(np.int64).astype(np.uint64)
                pos = np.repeat(np.arange(b - a), np.diff(off[a:b + 1]))
                h = (x * _M1) ^ ((x + np.uint64(1)) * _M2)
                s = np.zeros(b - a, np.uint64)
                np.add.at(s, pos, h)
                fp[a:b] = s ^ (np.diff(off[a:b + 1]).astype(np.uint64) * _M2)
            fp ^= turn * _M1
            if "legal_labels_json" in f:
                fp ^= np.asarray([hash(x) & 0xFFFFFFFFFFFF for x in f["legal_labels_json"].asstr()[:]], np.uint64)
            if "set_indptr" in f:
                sp, si = f["set_indptr"][:].astype(np.int64), f["set_idx"][:].astype(np.uint64)
                s = np.zeros(n, np.uint64)
                np.add.at(s, np.repeat(np.arange(n), np.diff(sp)), (si + np.uint64(7)) * _M2)
                fp ^= s
            if "y" in f:
                fp ^= f["y"][:].astype(np.uint64) * np.uint64(0x94D049BB133111EB)
    return game, fp


def compare(a_dir: Path, b_dir: Path, names: list[str] | None = None, log=print) -> dict:
    """Per flat table present in both directories: whether every game has the same rows, in the same order within
    the game. A game's rows are contiguous and in decision order in either build; only the order of games differs."""
    out = {}
    names = names or sorted(p.name for p in a_dir.glob("*.h5") if not p.name.endswith(".graph.h5")
                            and (b_dir / p.name).exists())
    for name in names:
        ga, fa = row_fingerprints(a_dir / name)
        gb, fb = row_fingerprints(b_dir / name)

        def per_game(g, f):
            o = np.argsort(g, kind="stable")
            u, st = np.unique(g[o], return_index=True)
            return {int(k): f[o][s:e].tobytes() for k, s, e in zip(u, st, list(st[1:]) + [len(o)])}
        pa, pb = per_game(ga, fa), per_game(gb, fb)
        diff = [k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k)]
        out[name] = {"rows": [len(ga), len(gb)], "games": [len(pa), len(pb)], "games_differing": len(diff),
                     "examples": sorted(diff)[:5], "same": not diff}
        log(f"compare {name}: {len(ga)} / {len(gb)} rows, {len(pa)} / {len(pb)} games, "
            f"{'identical' if not diff else f'{len(diff)} games differ'}", flush=True)
    return out


SLIM_DROP = ("indices", "row")


def slim(src: Path, dst: Path, log=print) -> dict:
    """The graph files, and the flat tables without their features: what graph_supervised.py reads."""
    import h5py
    dst.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for p in sorted(src.glob("*.h5")):
        q = dst / p.name
        if p.name.endswith(".graph.h5"):
            shutil.copy(p, q)
        else:
            with h5py.File(p, "r") as f, h5py.File(q, "w") as g:
                for k in _h5_datasets(f):
                    if k in SLIM_DROP:
                        continue
                    f.copy(f[k], g, name=k)
                for a, v in f.attrs.items():
                    g.attrs[a] = v
                g.attrs["slim"] = "features dropped (build.py slim): labels and metadata only"
        sizes[p.name] = q.stat().st_size
        log(f"slim {p.name}: {p.stat().st_size / 1e6:.0f} -> {q.stat().st_size / 1e6:.0f} MB", flush=True)
    return {"files": len(sizes), "bytes": sum(sizes.values())}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("splits")
    s.add_argument("--sbv1", type=Path, default=SBV1, help="experiment #3's items, whose drafts are held out")
    b = sub.add_parser("build")
    b.add_argument("--workers", type=int, default=3)
    b.add_argument("--limit", type=int)
    b.add_argument("--heap", default="2500m")
    b.add_argument("--part-games", type=int, default=PART_GAMES, help="games per shard part")
    b.add_argument("--graph", action="store_true", help="also record MageZero's graph encoding of every decision (docs/022)")
    b.add_argument("--out", type=Path, help=f"build root (default {OUT})")
    t = sub.add_parser("tables")
    t.add_argument("--keep-parts", action="store_true", help="keep h5/parts/ (the per-part tables)")
    t.add_argument("--graph", action="store_true", help="also write <table>_<split>.graph.h5 (needs a --graph build)")
    t.add_argument("--out", type=Path, help=f"build root (default {OUT})")
    c = sub.add_parser("compare", help="do two builds' flat tables hold the same rows, game by game?")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.add_argument("--tables", default=None, help="comma-separated file names (default: every table in both)")
    c.add_argument("--json", type=Path)
    sl = sub.add_parser("slim", help="graph files + flat tables without features, for the graph trainer")
    sl.add_argument("src", type=Path)
    sl.add_argument("dst", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        res = compare(a.a, a.b, a.tables.split(",") if a.tables else None)
        if a.json:
            a.json.write_text(json.dumps(res, indent=1))
        print(json.dumps({k: v["same"] for k, v in res.items()}, indent=1))
        raise SystemExit(0 if all(v["same"] for v in res.values()) else 1)
    if a.cmd == "slim":
        print(json.dumps(slim(a.src, a.dst), indent=1))
        return
    if a.cmd == "splits":
        codes, stats = make_splits(a.sbv1 if a.sbv1 and a.sbv1.exists() else None)
        if stats["sbv1"] is None:
            raise SystemExit(f"no sb-v1 items at {a.sbv1}: its drafts must be held out (pass --sbv1)")
        OUT.mkdir(parents=True, exist_ok=True)
        np.save(SPLIT_FILE, codes)
        (OUT / "row_split_exp4.stats.json").write_text(json.dumps(stats, indent=1))
        print(json.dumps(stats, indent=1))
    elif a.cmd == "build":
        print(json.dumps(build(a.workers, a.limit, a.heap, part_games=a.part_games, graph=a.graph, out_root=a.out),
                         indent=1))
    else:
        print(json.dumps(tables(keep_parts=a.keep_parts, graph=a.graph, out_root=a.out), indent=1, default=str))


if __name__ == "__main__":
    main()
