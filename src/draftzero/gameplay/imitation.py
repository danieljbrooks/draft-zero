"""
imitation.py — human 17lands play as MageZero training targets, and how predictable it is
(critique.md E1, plus the offline parts of E3/E4).

Two datasets, both split by 17lands event so no player's event straddles train and test:

  turn start   every non-terminal user turn of a sample of FDN games: the eot_rollover spec
               (reconstruct), A's first precombat-main PRIORITY decision reached with the spec's
               labels["bridge"] options, encoded by the bridge (MageZero's StateEncoder,
               perfectInfo=false: human data has no opponent hand). The policy label is a SET
               (17lands has no order inside a turn): S = the vocab keys of the turn's land plays,
               casts (flashback-aware) and activated non-mana abilities that are legal at that
               decision. A turn with no land, cast or activation has S = {Pass}. A turn whose
               actions are all illegal there (the land/spell came later: drawn, mana from a spell,
               a creature that died first ...) is 'unreachable': counted, never trained on.
               The value target is z = +1 / -1 from the game's `won`.
  replayed     turnreplay.replay_turn(..., encode=True) on a subset of the same games' turns: every
               decision A made in the turn, with label_kind exact / imputed_order /
               guessed_target; only turns with reproduced=True are used. Attack questions
               (CHOOSE_USE 'attack with: X?') have exact labels.

Splits: `component_splits` joins drafts linked by a mirrored pair (pairs.py) into one component
(union-find), then hashes the component to train / val / test, so both halves of a mirrored game
and every game of each of their events land in one split. Draft ids stay in memory; the per-row
split array written under data/gameplay/imitation/ holds none.

Storage (HDF5, `write_h5`): MageZero's LabeledStateWriter layout (/indices int32 sorted feature
ids, /offsets int64 [N+1], /row float32 [N, A+4] = policy(A), resultLabel, stateScore, isPlayer,
actionType) so the unmodified `magezero.dataset.H5Indexed` loads it (policy = the uniform
distribution over S, resultLabel = z, stateScore 0, isPlayer 1, actionType PRIORITY 0 or
CHOOSE_USE 5), plus /legal_indptr + /legal_idx (CSR legal set), /set_indptr + /set_idx (CSR
human set), /z, /label_kind (0 exact one-hot, 1 set-valued), /weight, /source (1 = 17lands)
and /meta/* columns (turn, tier, on_play, win-rate and n-games buckets, row, ...).

Models (`stage_*`, `report`): all policies are softmaxes MASKED to the legal set; set-valued
NLL = -log sum_{a in S} p(a); value = MSE(v, z) with weight 0.1. gen33 is MageZero's experiment #1
checkpoint (trained with perfectInfo=true and only the binary prior on in search). Its features are
mapped through its own dense vocab (unknown ids dropped, as its inference server does). Heads-only
models re-train gen33's priority / value / binary heads on its frozen trunk embeddings; full-network
runs (fine-tune, from scratch) train the whole NetTransformer with token-budgeted, length-bucketed
batches (a 64 x 2048-token attention batch took ~20 GB of unified memory on the 16 GB machine).
Results: data/gameplay/imitation/results.json and results.md (+ interpretation.md, kept by hand).

    python -m draftzero.gameplay.imitation splits            # pre-pass: row -> split (~10 s)
    python -m draftzero.gameplay.imitation build --every 48 --workers 2     # ~15 min, 2 JVMs
    python -m draftzero.gameplay.imitation tables             # shards -> HDF5 + decision statistics
    python -m draftzero.gameplay.imitation extras             # test-split perfectInfo=true encodings (1 JVM)
    python -m draftzero.gameplay.imitation gen33 | gen33_attack | gen33_extras | gen33_train --n-train 30000
    python -m draftzero.gameplay.imitation heads              # heads-only fine-tunes (+ binary head)
    python -m draftzero.gameplay.imitation finetune --budget 480
    python -m draftzero.gameplay.imitation scratch --budget 480 [--n-train 3000 --test-rows 5000]
    python -m draftzero.gameplay.imitation replay_heads --n-train 12000
    python -m draftzero.gameplay.imitation report             # every metric -> results.json / .md

17lands public data is CC BY 4.0 (https://www.17lands.com/public_datasets).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import pickle
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "data" / "gameplay" / "imitation"
GEN33 = REPO / "models" / "FDN_generalist" / "ver1" / "gen33.pt.gz"
A_DIM = 1024                         # FDN action vocabulary width (assets/vocab/FDN_SPG.tsv)
SPLITS = ("train", "val", "test")
SPLIT_FRACS = (0.82, 0.05, 0.13)     # cumulative 0.82 / 0.87 / 1.0 of components
SPLIT_SALT = "draftzero-imitation-v1"
TIERS = ("T0", "T1", "T2", "T3")
ACTION_TYPE = {"PRIORITY": 0, "CHOOSE_TARGET": 3, "CHOOSE_USE": 5}
LABEL_STATUS = ("pass", "set", "unreachable")


# ================================================================================================
# splits
# ================================================================================================

def split_for_key(key: str, salt: str = SPLIT_SALT) -> int:
    """0 train / 1 val / 2 test for a component key, by a salted SHA-1 (stable across runs)."""
    h = int.from_bytes(hashlib.sha1((salt + "|" + key).encode()).digest()[:8], "big") / 2 ** 64
    acc = 0.0
    for i, f in enumerate(SPLIT_FRACS):
        acc += f
        if h < acc:
            return i
    return len(SPLIT_FRACS) - 1


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        p = self.parent.setdefault(x, x)
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        self.parent[x] = p
        return p

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra      # the smaller draft id is the root: order-independent key


def component_splits(draft_by_row: list[str], pairs: Iterable[tuple[int, int]]) -> tuple[np.ndarray, dict]:
    """Per-row split codes (int8: 0 train, 1 val, 2 test). Drafts joined by a mirrored pair form
    one component, keyed by its smallest draft id; the component key is hashed to a split. Rows
    without a draft id get their own key (row index). Returns (codes, stats)."""
    uf = _UnionFind()
    for a, b in pairs:
        da, db = draft_by_row[a], draft_by_row[b]
        if da and db:
            uf.union(da, db)
    cache: dict[str, int] = {}
    out = np.empty(len(draft_by_row), dtype=np.int8)
    for i, d in enumerate(draft_by_row):
        if not d:
            out[i] = split_for_key(f"row:{i}")
            continue
        r = uf.find(d) if d in uf.parent else d
        s = cache.get(r)
        if s is None:
            s = cache[r] = split_for_key(r)
        out[i] = s
    comps = Counter(uf.find(d) for d in uf.parent)
    stats = {"rows": len(draft_by_row), "components_joined_by_pairs": len(comps),
             "largest_component_drafts": max(comps.values()) if comps else 0,
             "rows_by_split": {SPLITS[k]: int(v) for k, v in zip(*np.unique(out, return_counts=True))}}
    return out, stats


def draft_ids_by_row(path=None) -> list[str]:
    """draft_id of every data row (column 2; the first two columns hold no commas). Memory only."""
    from draftzero.gameplay.replay import replay_path
    path = path or replay_path()
    out = []
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split(",")
        col = head.index("draft_id")
        for line in f:
            out.append(line.split(",", col + 1)[col])
    return out


def row_splits(path=None, cache: Path | None = None, rebuild: bool = False) -> np.ndarray:
    """Split code per data row, cached in data/gameplay/imitation/row_split.npy (no draft ids)."""
    cache = cache or OUT_DIR / "row_split.npy"
    if cache.exists() and not rebuild and path is None:
        return np.load(cache)
    from draftzero.gameplay import pairs as pm
    drafts = draft_ids_by_row(path)
    try:
        prs = [(p.row_a, p.row_b) for p in pm.load_pairs()] if path is None else []
    except OSError:
        prs = []
    codes, stats = component_splits(drafts, prs)
    if path is None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, codes)
        (cache.parent / "row_split.stats.json").write_text(json.dumps(stats, indent=1))
    return codes


# ================================================================================================
# labels (pure)
# ================================================================================================

def turn_actions(labels: dict) -> list[str]:
    """The vocab keys of a 17lands turn's land plays, casts and keyed activated abilities."""
    acts = [x["key"] for x in labels.get("lands", []) if x.get("key")]
    acts += [x["key"] for x in labels.get("casts", []) if x.get("key")]
    acts += [x["key"] for x in labels.get("activations", []) if x.get("key")]
    return list(dict.fromkeys(acts))


def human_set(labels: dict, legal: list[str]) -> tuple[str, list[str], list[str]]:
    """(status, S, acts) for the turn's first main-phase PRIORITY decision with legal labels `legal`.

    status 'pass': the human took no land, cast or activation this turn, S = ['Pass'];
           'set': S = the turn's actions that are legal at this decision (matched with
                  coach.match_label: exact, modulo reminder text, 17lands ability text without the
                  cost, case);
           'unreachable': the human acted, but none of it is legal here (S empty)."""
    from draftzero.gameplay.coach import match_label
    acts = turn_actions(labels)
    unkeyed = [x for x in labels.get("activations", []) if not x.get("key")]
    if not acts and not unkeyed:
        return "pass", (["Pass"] if "Pass" in legal else []), acts
    S = []
    for a in acts:
        m = match_label(a, legal, "PRIORITY")
        if m is not None and m not in S:
            S.append(m)
    return ("set" if S else "unreachable"), S, acts


def cost_mv(cost: str) -> int:
    """Mana value of a mana cost string such as '{2}{U}{U}' ({X} counts 0)."""
    mv = 0
    for sym in re.findall(r"\{([^}]*)\}", cost or ""):
        mv += int(sym) if sym.isdigit() else (0 if sym in ("X", "T", "Q") else 1)
    return mv


def heuristic_scores(legal: list[str], mv_of) -> np.ndarray:
    """Baseline (b): play a land if one is legal, else cast the highest-mana-value legal spell, else
    Pass. Returns a score per legal option (higher first): lands 1000, casts 100 + MV, Pass 1,
    anything else (activations, flashback) 0. Ties are broken by the caller."""
    out = np.zeros(len(legal))
    for i, lab in enumerate(legal):
        if lab.startswith("Play "):
            out[i] = 1000
        elif lab.startswith("Cast "):
            out[i] = 100 + (mv_of(lab[5:]) or 0)
        elif lab == "Pass":
            out[i] = 1
    return out


# ================================================================================================
# masked softmax and the set loss (torch)
# ================================================================================================

def masked_log_softmax(logits, legal_mask):
    """log softmax over the legal entries only (illegal entries get -inf)."""
    import torch
    x = logits.masked_fill(~legal_mask, float("-inf"))
    return torch.log_softmax(x, dim=-1)


def set_nll(logp, set_mask):
    """-log sum_{a in S} p(a) per row, from masked log-probabilities (S must be legal and
    non-empty)."""
    import torch
    x = logp.masked_fill(~set_mask, float("-inf"))
    return -torch.logsumexp(x, dim=-1)


def masks_from_csr(indptr: np.ndarray, idx: np.ndarray, rows: np.ndarray, width: int = A_DIM):
    """Boolean [len(rows), width] mask from a CSR list (numpy)."""
    m = np.zeros((len(rows), width), dtype=bool)
    for j, r in enumerate(rows):
        a, b = indptr[r], indptr[r + 1]
        m[j, idx[a:b]] = True
    return m


def topk_in_set(scores: np.ndarray, legal: np.ndarray, S: np.ndarray, k: int, rng=None) -> float:
    """1 if one of the k best-scored legal options is in S. Exact ties are broken uniformly at
    random (rng) - pass rng=None to use the expected value over tie orders."""
    order_scores = scores[legal]
    if rng is not None:
        jitter = rng.random(len(legal)) * 1e-9
        top = legal[np.argsort(-(order_scores + jitter), kind="stable")[:k]]
        return float(np.isin(top, S).any())
    # expected value over random tie-breaks (k = 1 exactly; k > 1 via inclusion)
    srt = np.sort(order_scores)[::-1]
    thr = srt[min(k, len(srt)) - 1]
    above = legal[order_scores > thr]
    tied = legal[order_scores == thr]
    if np.isin(above, S).any():
        return 1.0
    slots = min(k, len(legal)) - len(above)
    n_tied, n_good = len(tied), int(np.isin(tied, S).sum())
    if n_good == 0 or slots <= 0:
        return 0.0
    # P(at least one good among `slots` drawn without replacement from the tied group)
    p_none = math.comb(n_tied - n_good, slots) / math.comb(n_tied, slots)
    return 1.0 - p_none


# ================================================================================================
# dataset building
# ================================================================================================

_W: dict[str, Any] = {}


def _decision_meta(g) -> dict:
    m = g.meta
    return {"row": g.row_index, "on_play": bool(m.get("on_play")), "won": bool(m.get("won")),
            "wr_bucket": m.get("user_game_win_rate_bucket"), "n_games_bucket": m.get("user_n_games_bucket"),
            "rank": m.get("rank"), "num_turns": m.get("num_turns")}


def _feat(r: dict) -> np.ndarray:
    return np.asarray(r.get("features") or [], dtype=np.int32)


def turn_start_record(g, n: int, b, ids, *, perfect_info: bool = False, spec=None, split: int = -1) -> dict:
    """One turn-start decision: spec -> encode (labels['bridge'] options) -> labels. Never raises
    for a bridge or reconstruction problem: status says what happened."""
    from draftzero.gameplay import reconstruct as rc
    from draftzero.gameplay.bridge import BridgeError
    rec = {"turn": n, "split": split, **_decision_meta(g)}
    try:
        spec = spec or rc.state_at_user_turn(g, n, labels=True, ids=ids)
    except Exception as e:           # noqa: BLE001 - a reconstruction failure is data, not a crash
        rec.update(status="spec_error", error=f"{type(e).__name__}: {e}"[:300])
        return rec
    pv = spec.provenance
    rec.update(tier=pv.tier if pv else None, flags=list(pv.flags) if pv else [],
               global_turn=spec.labels.get("global_turn"))
    lab = spec.labels
    rec["attacked"] = any(lab.get("attacks", {}).values())
    rec["acts"] = turn_actions(lab)
    rec["n_unkeyed_activations"] = sum(1 for x in lab.get("activations", []) if not x.get("key"))
    opts = dict(lab.get("bridge") or {})
    try:
        r = b.encode(spec, perfectInfo=perfect_info, **opts)
    except BridgeError as e:
        rec.update(status="bridge_error", error=str(e).splitlines()[0][:300])
        return rec
    d = r.get("decision")
    if not d:
        rec.update(status="no_decision", error=(r.get("noDecision") or "")[:200])
        return rec
    w = d.get("where") or {}
    df = opts.get("decideFrom") or {}
    rec.update(dec_type=d.get("type"), dec_step=w.get("step"), dec_turn=w.get("turn"),
               passed_before=w.get("passedBefore"), warnings=len(r.get("warnings") or []))
    ok_place = d.get("type") == "PRIORITY" and w.get("step") == "PRECOMBAT_MAIN" and \
        (not df or w.get("turn") == df.get("turn"))
    legal = d.get("legal") or []
    rec["legal"] = [x["label"] for x in legal]
    rec["legal_idx"] = [int(x.get("idx", -1)) for x in legal]
    rec["legal_n"] = [int(x.get("n", 1)) for x in legal]
    if not ok_place:
        rec["status"] = "wrong_decision"
        return rec
    st, S, _ = human_set(lab, rec["legal"])
    rec["label_status"] = st
    rec["S"] = S
    rec["status"] = "ok"
    rec["features"] = _feat(r)
    return rec


def _attack_context(spec: dict, ids) -> dict:
    """What the attack baseline needs from the turn-start spec: B's untapped creatures (printed
    toughness, flying/reach) - they stay untapped through A's turn unless something taps them."""
    blockers = []
    for p in spec["players"]["B"].get("battlefield", []):
        name = p.get("name")
        if not name or p.get("tapped"):
            continue
        info = ids.info(name)
        if info.toughness is None:
            continue
        blockers.append({"t": info.toughness, "fly": bool({"Flying", "Reach"} & set(info.keywords))})
    return {"blockers": blockers}


def replay_records(g, n: int, b, ids, split: int = -1) -> dict:
    """replay_turn(encode=True) for user turn n: the turn verdict and every decision of A's."""
    from draftzero.gameplay import turnreplay as tr
    from draftzero.gameplay.reconstruct import state_at_user_turn
    out = {"turn": n, "split": split, **_decision_meta(g)}
    try:
        r = tr.replay_turn(b, g, n, ids, encode=True)
    except Exception as e:           # noqa: BLE001
        out.update(reproduced=False, error=f"{type(e).__name__}: {e}"[:300], decisions=[])
        return out
    out.update(reproduced=bool(r.get("reproduced")), attempt=r.get("attempt"), tier=(r.get("meta") or {}).get("tier"),
               diffKeys=(r.get("diffKeys") or [])[:6])
    try:
        spec = state_at_user_turn(g, n, ids=ids).to_dict()
        out["attack_ctx"] = _attack_context(spec, ids)
    except Exception:                # noqa: BLE001
        out["attack_ctx"] = {"blockers": []}
    decs = []
    for d in r.get("decisions") or []:
        decs.append({"type": d.get("type"), "text": d.get("text"), "legal": [x["label"] for x in d.get("legal") or []],
                     "legal_idx": [int(x.get("idx", -1)) for x in d.get("legal") or []],
                     "chosen": d.get("chosen"), "label_kind": d.get("label_kind"), "evidence": d.get("evidence"),
                     "set": d.get("set"), "step": (d.get("where") or {}).get("step"),
                     "features": _feat(d) if out["reproduced"] else None})
    out["decisions"] = decs
    return out


def _worker_init(header_cols: list[str], runtime_root: str, heap: str, wid_counter) -> None:
    from draftzero.gameplay.bridge import Bridge
    from draftzero.gameplay.ids import Ids
    from draftzero.gameplay.replay import Header
    with wid_counter.get_lock():
        wid = wid_counter.value
        wid_counter.value += 1
    _W["H"] = Header(header_cols)
    _W["ids"] = Ids.load()
    _W["name"] = f"imi{wid}"
    _W["bridge"] = Bridge(_W["name"], heap=heap, runtime_root=Path(runtime_root))


def _bridge():
    b = _W["bridge"]
    if b.proc is None or b.proc.poll() is not None:
        b.start()
    return b


def _work(task: tuple) -> dict:
    """One game: every decision turn's turn-start record, and replay_turn on the chosen turns."""
    import csv
    from draftzero.gameplay.replay import parse_game
    row, line, split, replay_frac, seed = task
    t0 = time.monotonic()
    try:
        g = parse_game(next(csv.reader([line])), _W["H"], row)
    except Exception as e:           # noqa: BLE001
        return {"row": row, "error": f"parse: {e}", "ts": [], "rp": [], "sec": time.monotonic() - t0}
    ids = _W["ids"]
    turns = g.decision_turns()
    ts = [turn_start_record(g, n, _bridge(), ids, split=split) for n in turns]
    t1 = time.monotonic()
    rng = random.Random(seed * 1_000_003 + row)
    chosen = []
    if turns:
        chosen.append(rng.choice(turns))
        rest = [n for n in turns if n not in chosen]
        if rest and rng.random() < replay_frac - 1.0:
            chosen.append(rng.choice(rest))
    rp = [replay_records(g, n, _bridge(), ids, split=split) for n in sorted(chosen)]
    return {"row": row, "ts": ts, "rp": rp, "sec_ts": t1 - t0, "sec_rp": time.monotonic() - t1}


def build(every: int = 48, start: int = 5, workers: int = 2, limit: int | None = None, replay_frac: float = 1.25,
          seed: int = 0, out_dir: Path | None = None, runtime_root: Path | None = None, heap: str = "2500m",
          path=None, splits: np.ndarray | None = None, log=print) -> dict:
    """The main pass: every `every`-th row from `start`, `workers` processes with one bridge JVM
    each. Writes shards ts.pkl / rp.pkl (a pickle stream of per-game records) under out_dir."""
    import multiprocessing as mp
    from draftzero.gameplay.replay import open_lines, replay_path
    out_dir = Path(out_dir or OUT_DIR / "shards")
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_root = Path(runtime_root or out_dir / "runtime")
    if splits is None:
        splits = row_splits(path)
    H, lines = open_lines(path or replay_path())
    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)

    def tasks():
        k = 0
        for i, line in enumerate(lines):
            if i < start or (i - start) % every:
                continue
            if limit is not None and k >= limit:
                break
            k += 1
            yield (i, line, int(splits[i]) if i < len(splits) else -1, replay_frac, seed)

    stats = Counter()
    t0 = time.monotonic()
    fts = open(out_dir / "ts.pkl", "wb")
    frp = open(out_dir / "rp.pkl", "wb")
    try:
        with ctx.Pool(workers, initializer=_worker_init,
                      initargs=(H.cols, str(runtime_root), heap, counter)) as pool:
            for res in pool.imap_unordered(_work, tasks(), chunksize=2):
                pickle.dump(res["ts"], fts, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(res["rp"], frp, protocol=pickle.HIGHEST_PROTOCOL)
                stats["games"] += 1
                stats["ts"] += len(res["ts"])
                stats["ts_ok"] += sum(1 for r in res["ts"] if r.get("status") == "ok")
                stats["rp"] += len(res["rp"])
                stats["rp_ok"] += sum(1 for r in res["rp"] if r.get("reproduced"))
                stats["sec_ts"] += res.get("sec_ts", 0)
                stats["sec_rp"] += res.get("sec_rp", 0)
                if stats["games"] % 500 == 0:
                    el = time.monotonic() - t0
                    log(f"{stats['games']} games, {stats['ts']} turn-start ({stats['ts_ok']} ok), "
                        f"{stats['rp']} replays ({stats['rp_ok']} reproduced) in {el:.0f} s: "
                        f"{stats['ts'] / el:.1f} decisions/s")
                    fts.flush()
                    frp.flush()
    finally:
        fts.close()
        frp.close()
        lines.close()
    el = time.monotonic() - t0
    out = {**stats, "seconds": round(el, 1), "decisions_per_s": round(stats["ts"] / max(el, 1e-9), 2),
           "every": every, "start": start, "workers": workers, "replay_frac": replay_frac, "seed": seed}
    (out_dir / "build_stats.json").write_text(json.dumps(out, indent=1))
    return out


def load_shard(path: Path) -> list:
    out = []
    with open(path, "rb") as f:
        while True:
            try:
                out.extend(pickle.load(f))
            except (EOFError, pickle.UnpicklingError):    # end, or a shard still being written
                break
    return out


# ================================================================================================
# test-split extras: perfect-information encodings for gen33
# ================================================================================================

def build_extras(ts: list[dict], *, max_belief_rows: int = 3000, seed: int = 0, out: Path | None = None,
                 runtime_root: Path | None = None, heap: str = "2500m", path=None, log=print) -> dict:
    """gen33 was trained with the opponent's hand visible (perfectInfo=true). For test-split
    decisions, re-encode with perfectInfo=true after (a) a belief-model determinization of the
    opponent's hand and deck (belief.OpponentModel with hand retention, hindsight-free colours
    opp_colors=None, the row's draft and its mirrored partner's draft held out of the pool) and
    (b) for mirrored pairs, the TRUE opponent deck and hand (reconstruct.exact_spec), both rows of
    each pair. Partner rows that are not in the main sample get their perfectInfo=false record too.
    Writes a pickle list of records {row, turn, variant, ...turn_start_record fields}."""
    from draftzero.gameplay import pairs as pm, reconstruct as rc
    from draftzero.gameplay.belief import determinize
    from draftzero.gameplay.bridge import Bridge
    from draftzero.gameplay.coach import opponent_model
    from draftzero.gameplay.ids import Ids
    from draftzero.gameplay.replay import read_games
    out = Path(out or OUT_DIR / "shards" / "extras.pkl")
    ids = Ids.load()
    test = [r for r in ts if r.get("split") == 2 and r.get("status") == "ok"]
    by_row: dict[int, list[int]] = defaultdict(list)
    for r in test:
        by_row[r["row"]].append(r["turn"])
    partners = pm.partner_map(pm.load_pairs())
    pair_rows = {r: partners[r] for r in by_row if r in partners}
    rng = random.Random(seed)
    others = [r for r in sorted(by_row) if r not in pair_rows]
    rng.shuffle(others)
    n_pair_dec = sum(len(by_row[r]) for r in pair_rows)
    belief_rows, k = [], n_pair_dec
    for r in others:
        if k >= max_belief_rows:
            break
        belief_rows.append(r)
        k += len(by_row[r])
    want = set(pair_rows) | set(pair_rows.values()) | set(belief_rows)
    t0 = time.monotonic()
    games = read_games(sorted(want), path)
    log(f"extras: read {len(games)} games in {time.monotonic() - t0:.0f} s "
        f"({len(pair_rows)} pair rows + partners, {len(belief_rows)} belief-only rows)")
    recs, stats = [], Counter()
    b = Bridge("imi-extras", heap=heap, runtime_root=Path(runtime_root or OUT_DIR / "shards" / "runtime"))
    try:
        jobs = [(r, None) for r in belief_rows] + [(r, pair_rows[r]) for r in sorted(pair_rows)]
        # the partner rows' own decisions (both directions of each pair)
        jobs += [(p, r) for r, p in sorted(pair_rows.items()) if p not in by_row]
        for row, prow in jobs:
            g = games.get(row)
            partner = games.get(prow) if prow is not None else None
            if g is None:
                stats["missing_game"] += 1
                continue
            turns = by_row.get(row) or g.decision_turns()
            model = opponent_model(rc.holdout_drafts(g, partner))
            ana = rc.analyze(g, ids)
            for n in turns:
                if row not in by_row:        # a partner row: its own perfectInfo=false record
                    rec = turn_start_record(g, n, b, ids, split=2)
                    rec["variant"] = "pi_false"
                    recs.append(rec)
                    stats["pi_false"] += 1
                try:
                    spec = rc.state_at_user_turn(g, n, labels=True, ids=ids)
                    slot = g.prev_slot(n)
                    seen = Counter(ana.states[slot.seq].revealed) if slot is not None else Counter()
                    det = determinize(spec, model, 1, seed=seed * 7919 + n, opp_colors=None, seen=seen)[0]
                    det.labels = spec.labels
                    rec = turn_start_record(g, n, b, ids, perfect_info=True, spec=det, split=2)
                except Exception as e:      # noqa: BLE001
                    rec = {"row": row, "turn": n, "status": "belief_error", "error": str(e)[:200]}
                rec["variant"] = "belief_pi_true"
                rec["pair"] = partner is not None
                recs.append(rec)
                stats["belief_" + rec.get("status", "?")] += 1
                if partner is not None:
                    try:
                        ex = rc.exact_spec(g, partner, n, ids=ids, labels=True)
                        rec = turn_start_record(g, n, b, ids, perfect_info=True, spec=ex, split=2)
                    except Exception as e:  # noqa: BLE001
                        rec = {"row": row, "turn": n, "status": "exact_error", "error": str(e)[:200]}
                    rec["variant"] = "true_pi_true"
                    rec["pair"] = True
                    rec["exact_flags"] = rec.get("flags")
                    recs.append(rec)
                    stats["true_" + rec.get("status", "?")] += 1
            if len(recs) % 500 < len(turns) * 3:
                log(f"extras: {len(recs)} records, {dict(stats)}, {time.monotonic() - t0:.0f} s")
    finally:
        b.close()
    with open(out, "wb") as f:
        pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)
    stats["seconds"] = round(time.monotonic() - t0, 1)
    return dict(stats)


# ================================================================================================
# HDF5 (MageZero LabeledStateWriter layout + extras)
# ================================================================================================

def _uniq_legal(labels: list[str], idx: list[int]) -> tuple[list[int], dict[str, int]]:
    """Unique vocab indices of the legal options (labels sharing an index are one option to the
    network) and label -> index. Labels with no vocab index (-1) are dropped."""
    lab2i, out = {}, []
    for lab, i in zip(labels, idx):
        if 0 <= i < A_DIM:
            lab2i[lab] = i
            if i not in out:
                out.append(i)
    return out, lab2i


def ts_table(recs: list[dict]) -> dict:
    """Turn-start records (status ok) -> arrays: feature CSR, legal CSR, set CSR, z, meta."""
    recs = [r for r in recs if r.get("status") == "ok"]
    feats, f_ptr = [], [0]
    legal, l_ptr, sets, s_ptr = [], [0], [], [0]
    meta = defaultdict(list)
    for r in recs:
        f = r["features"]
        feats.append(f)
        f_ptr.append(f_ptr[-1] + len(f))
        li, lab2i = _uniq_legal(r["legal"], r["legal_idx"])
        legal.extend(li)
        l_ptr.append(len(legal))
        si = sorted({lab2i[s] for s in r.get("S", []) if s in lab2i})
        st = r["label_status"]
        if st != "unreachable" and not si:
            st = "unreachable"        # every S member lacks a vocab index (none measured so far)
        sets.extend(si if st != "unreachable" else [])
        s_ptr.append(len(sets))
        meta["label_status"].append(LABEL_STATUS.index(st))
        meta["turn"].append(r["turn"])
        meta["global_turn"].append(r.get("global_turn") or -1)
        meta["tier"].append(TIERS.index(r["tier"]) if r.get("tier") in TIERS else -1)
        meta["on_play"].append(int(r["on_play"]))
        meta["won"].append(int(r["won"]))
        meta["wr_bucket"].append(r["wr_bucket"] if r["wr_bucket"] is not None else np.nan)
        meta["n_games_bucket"].append(r["n_games_bucket"] if r["n_games_bucket"] is not None else -1)
        meta["row"].append(r["row"])
        meta["split"].append(r["split"])
        meta["n_legal"].append(len(li))
        meta["n_nonpass"].append(sum(1 for x in r["legal"] if x != "Pass"))
        meta["attacked"].append(int(r.get("attacked", False)))
        meta["n_acts"].append(len(r.get("acts", [])))
    out = {"indices": np.concatenate(feats).astype(np.int32) if feats else np.zeros(0, np.int32),
           "offsets": np.asarray(f_ptr, np.int64),
           "legal_indptr": np.asarray(l_ptr, np.int64), "legal_idx": np.asarray(legal, np.int32),
           "set_indptr": np.asarray(s_ptr, np.int64), "set_idx": np.asarray(sets, np.int32)}
    for k, v in meta.items():
        out["meta/" + k] = np.asarray(v, dtype=np.float32 if k == "wr_bucket" else np.int32)
    out["z"] = np.where(out["meta/won"] > 0, 1.0, -1.0).astype(np.float32)
    out["labels"] = [r["legal"] for r in recs]
    out["S"] = [r.get("S", []) for r in recs]
    out["flags"] = [r.get("flags", []) for r in recs]
    return out


def write_h5(t: dict, path: Path, action_type: int = 0) -> None:
    """LabeledStateWriter layout (/indices, /offsets, /row [N, A+4]) + the CSR/meta extras."""
    import h5py
    n = len(t["offsets"]) - 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("indices", data=t["indices"], compression="gzip", compression_opts=1)
        f.create_dataset("offsets", data=t["offsets"])
        row = f.create_dataset("row", shape=(n, A_DIM + 4), dtype="f4", chunks=(min(max(n, 1), 512), A_DIM + 4),
                               compression="gzip", compression_opts=1)
        for a in range(0, n, 4096):
            b = min(n, a + 4096)
            blk = np.zeros((b - a, A_DIM + 4), np.float32)
            for j in range(a, b):
                s = t["set_idx"][t["set_indptr"][j]:t["set_indptr"][j + 1]]
                if len(s):
                    blk[j - a, s] = 1.0 / len(s)
            blk[:, A_DIM] = t["z"][a:b]              # resultLabel = z (lambda = 1: pure outcome)
            blk[:, A_DIM + 1] = 0.0                  # stateScore: no search was run
            blk[:, A_DIM + 2] = 1.0                  # isPlayer: the human is the agent
            blk[:, A_DIM + 3] = action_type
            row[a:b] = blk
        for k in ("legal_indptr", "legal_idx", "set_indptr", "set_idx", "z"):
            f.create_dataset(k, data=t[k])
        ls = t["meta/label_status"]
        f.create_dataset("label_kind", data=np.ones(n, np.int8))          # 1 = set-valued
        f.create_dataset("weight", data=(ls != LABEL_STATUS.index("unreachable")).astype(np.float32))
        f.create_dataset("source", data=np.ones(n, np.int8))              # 1 = 17lands
        for k, v in t.items():
            if k.startswith("meta/"):
                f.create_dataset(k, data=v)
        f.attrs["layout"] = "LabeledStateWriter + legal/set CSR (draftzero.gameplay.imitation)"
        f.attrs["perfectInfo"] = False
        f.attrs["action_dim"] = A_DIM


# ================================================================================================
# models: gen33, trunk embeddings, training
# ================================================================================================

def _torch_env() -> None:
    os.environ.setdefault("MZ_PAD_BUCKET", "64")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _require_exp1_engine() -> None:
    """These experiments are built on experiment #1's engine: gen33, its NetTransformer signature,
    and v0.1 state encodings (2M-bin feature hash). draft-zero itself now runs on MageZero v0.2,
    which can load none of them, so run them in an environment with exp #1's engine."""
    import magezero.model as mzm
    if not hasattr(mzm, "build_model_from_checkpoint"):
        raise RuntimeError(
            "draftzero.gameplay.imitation needs experiment #1's engine, not MageZero v0.2. Use a "
            "separate venv: pip install 'magezero @ git+https://github.com/danieljbrooks/MageZero@bcc76de' "
            "&& pip install -e . --no-deps, with the v0.1 XMage bundle (danieljbrooks/mage "
            "exp1-fdn-generalist) for the bridge.")


def load_gen33(path: Path = GEN33):
    """(NetTransformer, FeatureVocab) of the experiment #1 checkpoint."""
    _torch_env()
    _require_exp1_engine()
    from magezero.model import build_model_from_checkpoint, load_model
    from magezero.vocab import FeatureVocab
    ck = load_model(str(path))
    return build_model_from_checkpoint(ck), FeatureVocab.from_state_dict(ck["feature_vocab"])


def map_features(vocab, indices: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw StateEncoder ids (CSR) -> embedding rows of `vocab`; unknown ids dropped (as the
    MageZero inference server does)."""
    return vocab.map_csr(indices.astype(np.int64), offsets.astype(np.int64))


def _batch(rows: np.ndarray, ptr: np.ndarray, sel: np.ndarray, device, token_dropout: float = 0.0, rng=None):
    import torch
    parts = [rows[ptr[i]:ptr[i + 1]] for i in sel]
    if token_dropout > 0:            # MageZero's training-time token dropout (tokens removed)
        parts = [p[rng.random(len(p)) >= token_dropout] for p in parts]
    lens = np.array([len(p) for p in parts])
    if (lens == 0).any():                              # an empty bag: one dummy row 0
        parts = [p if len(p) else np.zeros(1, np.int64) for p in parts]
        lens = np.maximum(lens, 1)
    idx = torch.as_tensor(np.concatenate(parts), dtype=torch.long, device=device)
    off = torch.as_tensor(np.concatenate([[0], np.cumsum(lens)[:-1]]), dtype=torch.long, device=device)
    return idx, off


BUCKETS = (256, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 3072)   # fixed pad lengths (MPS compiles per shape)


def bucket_len(n: int) -> int:
    return next((b for b in BUCKETS if n <= b), BUCKETS[-1])


REF_LEN = 768     # batch sizes are set for this padded length and scaled by (REF_LEN / L)^2 above it:
                  # attention memory is batch x heads x L^2 per layer, and on the 16 GB machine a
                  # 64 x 2048 batch took ~20 GB of unified memory and pushed it into swap


def batch_for(L: int, base: int) -> int:
    return max(1, min(base, int(base * (REF_LEN / max(L, 1)) ** 2)))


def length_batches(lengths: np.ndarray, base: int) -> list[np.ndarray]:
    """Index batches in increasing length order, each padded to one BUCKETS size and sized by
    batch_for (a token budget)."""
    order = np.argsort(lengths, kind="stable")
    out, k = [], 0
    while k < len(order):
        L = bucket_len(int(lengths[order[min(k + base, len(order)) - 1]]))
        b = batch_for(L, base)
        L = bucket_len(int(lengths[order[min(k + b, len(order)) - 1]]))
        out.append(order[k:k + b])
        k += b
    return out


def trunk(model, indices, offsets, pad_to: int | None = None):
    """NetTransformer.forward up to the pooled 512-d embedding, with vectorised padding to a fixed
    length (`pad_to`, else the next BUCKETS size) so MPS compiles few kernel shapes. Bags longer
    than the pad length are cut (none in these data at 3,072). Dropout follows model.training."""
    import torch
    indices = indices % model.num_embeddings
    ends = torch.cat([offsets[1:], torch.tensor([indices.shape[0]], device=offsets.device)])
    lengths = ends - offsets
    max_len = pad_to or bucket_len(int(lengths.max().item()))
    lengths = lengths.clamp(max=max_len)
    ar = torch.arange(max_len, device=indices.device)
    mask = ar.unsqueeze(0) < lengths.unsqueeze(1)
    pos = offsets.unsqueeze(1) + ar.unsqueeze(0)
    padded = torch.where(mask, indices[pos.clamp(max=indices.shape[0] - 1)], torch.zeros_like(pos))
    emb = model.embedding(padded)
    emb = model.transformer(emb, src_key_padding_mask=~mask)
    cnt = mask.sum(1).clamp(min=1).unsqueeze(-1).float()
    pooled = (emb * mask.unsqueeze(-1)).sum(1) / cnt
    return model.embedding_dropout(pooled)


def heads(model, emb):
    return (model.player_priority_head(emb), model.binary_head(emb), model.value_head(emb).squeeze(-1))


def device():
    import torch
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def embed_all(model, rows: np.ndarray, ptr: np.ndarray, batch: int = 64, log=None) -> np.ndarray:
    """Pooled trunk embeddings (float16 [N, 512]) for every bag, length-sorted batches."""
    import torch
    dev = device()
    model.to(dev).eval()
    n = len(ptr) - 1
    out = np.zeros((n, 512), np.float16)
    t0 = time.monotonic()
    with torch.no_grad():
        for j, sel in enumerate(length_batches(np.diff(ptr), batch)):
            idx, off = _batch(rows, ptr, sel, dev)
            out[sel] = trunk(model, idx, off).float().cpu().numpy().astype(np.float16)
            if log and j % 200 == 0:
                log(f"embed batch {j} {time.monotonic() - t0:.0f} s")
    if dev.type == "mps":
        torch.mps.empty_cache()
    return out


def head_outputs(model, emb: np.ndarray, batch: int = 4096) -> dict:
    """Priority logits, binary logits, value from pooled embeddings (CPU)."""
    import torch
    m = model.to("cpu").eval()
    pa, bi, v = [], [], []
    with torch.no_grad():
        for k in range(0, len(emb), batch):
            e = torch.as_tensor(emb[k:k + batch], dtype=torch.float32)
            a, b_, c = heads(m, e)
            pa.append(a.numpy())
            bi.append(b_.numpy())
            v.append(c.numpy())
    return {"pa": np.concatenate(pa), "bin": np.concatenate(bi), "v": np.concatenate(v)}


def dense_masks(indptr: np.ndarray, idx: np.ndarray, sel: np.ndarray, width: int = A_DIM) -> np.ndarray:
    m = np.zeros((len(sel), width), dtype=bool)
    for j, r in enumerate(sel):
        m[j, idx[indptr[r]:indptr[r + 1]]] = True
    return m


def train_heads(model, emb_tr, t_tr, emb_va, t_va, *, epochs: int = 30, lr: float = 1e-3, value_weight: float = 0.1,
                batch: int = 256, seed: int = 0, n_train: int | None = None, log=None) -> tuple[Any, dict]:
    """Fine-tune the priority and value heads (initialised from `model`'s) on frozen trunk
    embeddings: set NLL over the masked softmax + value_weight x MSE(v, z). Early stopping on val
    set-NLL. Returns (a model copy with the tuned heads, history)."""
    import copy
    import torch
    torch.manual_seed(seed)
    m = copy.deepcopy(model).to("cpu")
    params = list(m.player_priority_head.parameters()) + list(m.value_head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng(seed)
    pol_tr = np.flatnonzero(t_tr["meta/label_status"] != 2)
    all_tr = np.arange(len(t_tr["z"]))
    if n_train is not None:
        pol_tr = np.sort(rng.choice(pol_tr, min(n_train, len(pol_tr)), replace=False))
        all_tr = pol_tr
    Ltr = torch.as_tensor(dense_masks(t_tr["legal_indptr"], t_tr["legal_idx"], all_tr))
    Str = torch.as_tensor(dense_masks(t_tr["set_indptr"], t_tr["set_idx"], all_tr))
    Etr = torch.as_tensor(emb_tr[all_tr], dtype=torch.float32)
    Ztr = torch.as_tensor(t_tr["z"][all_tr])
    Ptr = torch.as_tensor(t_tr["meta/label_status"][all_tr] != 2)
    va = np.flatnonzero(t_va["meta/label_status"] != 2)
    Lva = torch.as_tensor(dense_masks(t_va["legal_indptr"], t_va["legal_idx"], va))
    Sva = torch.as_tensor(dense_masks(t_va["set_indptr"], t_va["set_idx"], va))
    Eva = torch.as_tensor(emb_va[va], dtype=torch.float32)
    best, best_state, hist, bad = float("inf"), None, [], 0
    for ep in range(epochs):
        m.train()
        perm = torch.as_tensor(rng.permutation(len(all_tr)))
        for k in range(0, len(perm), batch):
            j = perm[k:k + batch]
            pa, _, v = heads(m, Etr[j])
            lp = masked_log_softmax(pa, Ltr[j])
            nll = set_nll(lp, Str[j] & Ltr[j])
            pmask = Ptr[j]
            loss = (nll[pmask].mean() if pmask.any() else 0.0) + value_weight * ((v - Ztr[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            pa, _, _ = heads(m, Eva)
            vnll = float(set_nll(masked_log_softmax(pa, Lva), Sva & Lva).mean())
        hist.append({"epoch": ep, "val_set_nll": round(vnll, 4)})
        if log:
            log(f"heads epoch {ep}: val set-NLL {vnll:.4f}")
        if vnll < best - 1e-4:
            best, best_state, bad = vnll, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= 3:
                break
    m.load_state_dict(best_state)
    return m, {"history": hist, "best_val_set_nll": best, "n_train": int(len(all_tr))}


def train_full(model, rows_tr, ptr_tr, t_tr, rows_va, ptr_va, t_va, *, budget_s: float, lr: float,
               value_weight: float = 0.1, batch: int = 32, warmup: int = 200, n_train: int | None = None,
               token_dropout: float = 0.3,
               eval_every_s: float = 150.0, val_rows: int = 2000, seed: int = 0, log=None) -> tuple[Any, dict]:
    """Train the whole NetTransformer (token dropout 0.3 as in MageZero) on the human rows for a
    wall-clock budget: Adam, linear warm-up, set NLL on the masked priority softmax + value_weight
    x MSE(v, z). Early stopping on a fixed val subset (set NLL), best state kept."""
    import copy
    import torch
    torch.manual_seed(seed)
    dev = device()
    m = model.to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    pool = np.arange(len(t_tr["z"]))
    if n_train is not None:
        pool = np.sort(rng.choice(pool, min(n_train, len(pool)), replace=False))
    va = np.flatnonzero(t_va["meta/label_status"] != 2)
    va = np.sort(rng.choice(va, min(val_rows, len(va)), replace=False))
    t0 = time.monotonic()
    last_eval, step, seen = t0, 0, 0
    best, best_state, hist = float("inf"), None, []

    def val_nll():
        m.eval()
        tot = 0.0
        with torch.no_grad():
            for bsel in length_batches(np.diff(ptr_va)[va], 64):
                sel = va[bsel]
                idx, off = _batch(rows_va, ptr_va, sel, dev)
                pa, _, _ = heads(m, trunk(m, idx, off))
                L = torch.as_tensor(dense_masks(t_va["legal_indptr"], t_va["legal_idx"], sel), device=dev)
                S = torch.as_tensor(dense_masks(t_va["set_indptr"], t_va["set_idx"], sel), device=dev)
                tot += float(set_nll(masked_log_softmax(pa.float(), L), S & L).sum())
        m.train()
        if dev.type == "mps":
            torch.mps.empty_cache()
        return tot / len(va)

    v0 = val_nll()
    hist.append({"step": 0, "seen": 0, "val_set_nll": round(v0, 4), "sec": 0})
    best, best_state = v0, copy.deepcopy({k: v.detach().cpu() for k, v in m.state_dict().items()})
    if log:
        log(f"full: start val set-NLL {v0:.4f}, train pool {len(pool)}")
    m.train()
    # length-bucketed batches: few distinct padded shapes (MPS), little padding
    lens = (np.diff(ptr_tr)[pool] * (1 - token_dropout)).astype(int)
    bkt = np.array([bucket_len(int(x)) for x in lens])
    groups = {b_: pool[bkt == b_] for b_ in np.unique(bkt)}
    gkeys = [k for k in groups if len(groups[k]) >= batch] or list(groups)
    gw = np.array([len(groups[k]) for k in gkeys], np.float64)
    gw /= gw.sum()
    while time.monotonic() - t0 < budget_s:
        key = gkeys[rng.choice(len(gkeys), p=gw)]
        grp = groups[key]
        sel = rng.choice(grp, min(batch_for(int(key), batch), len(grp)), replace=False)
        idx, off = _batch(rows_tr, ptr_tr, sel, dev, token_dropout, rng)
        pa, _, v = heads(m, trunk(m, idx, off))
        L = torch.as_tensor(dense_masks(t_tr["legal_indptr"], t_tr["legal_idx"], sel), device=dev)
        S = torch.as_tensor(dense_masks(t_tr["set_indptr"], t_tr["set_idx"], sel), device=dev)
        P = torch.as_tensor(t_tr["meta/label_status"][sel] != 2, device=dev)
        z = torch.as_tensor(t_tr["z"][sel], device=dev)
        nll = set_nll(masked_log_softmax(pa.float(), L), S & L)
        loss = (nll[P].mean() if bool(P.any()) else pa.sum() * 0) + value_weight * ((v.float() - z) ** 2).mean()
        for gparam in opt.param_groups:
            gparam["lr"] = lr * min(1.0, (step + 1) / warmup)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        seen += len(sel)             # the token budget shrinks long-bag batches below `batch`
        if time.monotonic() - last_eval > eval_every_s:
            last_eval = time.monotonic()
            vn = val_nll()
            hist.append({"step": step, "seen": seen, "val_set_nll": round(vn, 4),
                         "sec": round(time.monotonic() - t0)})
            if log:
                log(f"full: step {step} seen {seen} val set-NLL {vn:.4f} ({time.monotonic() - t0:.0f} s)")
            if vn < best:
                best, best_state = vn, copy.deepcopy({k: v.detach().cpu() for k, v in m.state_dict().items()})
    vn = val_nll()
    hist.append({"step": step, "seen": seen, "val_set_nll": round(vn, 4), "sec": round(time.monotonic() - t0)})
    if vn < best:
        best, best_state = vn, copy.deepcopy({k: v.detach().cpu() for k, v in m.state_dict().items()})
    m.load_state_dict(best_state)
    m.eval()
    return m, {"history": hist, "best_val_set_nll": best, "steps": step, "samples_seen": seen,
               "n_train_pool": int(len(pool)), "budget_s": budget_s, "lr": lr, "value_weight": value_weight,
               "batch": batch, "token_dropout": token_dropout}


# ================================================================================================
# metrics
# ================================================================================================

def legal_scores(t: dict, logits: np.ndarray | None = None, kind: str = "model", mv_of=None,
                 lab_idx: list | None = None) -> list[np.ndarray]:
    """Per row, a score per legal (unique-index) option, in legal_idx order."""
    out = []
    for j in range(len(t["legal_indptr"]) - 1):
        li = t["legal_idx"][t["legal_indptr"][j]:t["legal_indptr"][j + 1]]
        if kind == "model":
            out.append(logits[j, li].astype(np.float64))
        elif kind == "uniform":
            out.append(np.zeros(len(li)))
        elif kind == "heuristic":
            labs = t["labels"][j]
            # the first label with each index (labels sharing an index are one option)
            first = {}
            for lab, i in zip(labs, lab_idx[j]):
                first.setdefault(i, lab)
            out.append(heuristic_scores([first.get(i, "") for i in li], mv_of))
    return out


def policy_rows(t: dict, scores: list[np.ndarray], probabilistic: bool = True) -> dict:
    """Per-row top-1 / top-3 in S (expected over random tie-breaks) and set NLL (softmax of the
    scores over the legal options) for rows with a trainable label."""
    top1, top3, nll, keep = [], [], [], []
    for j, sc in enumerate(scores):
        if t["meta/label_status"][j] == 2:
            continue
        li = t["legal_idx"][t["legal_indptr"][j]:t["legal_indptr"][j + 1]]
        S = t["set_idx"][t["set_indptr"][j]:t["set_indptr"][j + 1]]
        keep.append(j)
        top1.append(topk_in_set(sc, np.arange(len(li)), np.flatnonzero(np.isin(li, S)), 1))
        top3.append(topk_in_set(sc, np.arange(len(li)), np.flatnonzero(np.isin(li, S)), 3))
        if probabilistic:
            x = sc - sc.max()
            p = np.exp(x) / np.exp(x).sum()
            nll.append(-math.log(max(p[np.isin(li, S)].sum(), 1e-12)))
    out = {"rows": np.asarray(keep), "top1": np.asarray(top1), "top3": np.asarray(top3)}
    if probabilistic:
        out["nll"] = np.asarray(nll)
    return out


def cluster_ci(values: np.ndarray, clusters: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean with a 95% cluster-bootstrap CI (clusters = games: decisions of one game are
    correlated)."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    u, inv = np.unique(clusters, return_inverse=True)
    s = np.bincount(inv, weights=values, minlength=len(u))
    c = np.bincount(inv, minlength=len(u)).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(n_boot, len(u)))
    means = s[idx].sum(1) / c[idx].sum(1)
    return {"n": int(len(values)), "games": int(len(u)), "mean": float(values.mean()),
            "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5))}


def value_metrics(v: np.ndarray, won: np.ndarray, clusters: np.ndarray, n_boot: int = 500) -> dict:
    """P(win) = (1 + v) / 2: log-loss, Brier, AUC, accuracy, with cluster-bootstrap CIs."""
    from scipy.stats import rankdata
    p = np.clip((1 + np.asarray(v, np.float64)) / 2, 1e-4, 1 - 1e-4)
    y = np.asarray(won, np.float64)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    br = (p - y) ** 2
    acc = ((p >= 0.5) == (y > 0.5)).astype(np.float64)

    def auc(pp, yy):
        pos = yy > 0.5
        if pos.all() or (~pos).all():
            return float("nan")
        r = rankdata(pp)
        return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))
    u, inv = np.unique(clusters, return_inverse=True)
    rng = np.random.default_rng(1)
    groups = [np.flatnonzero(inv == k) for k in range(len(u))] if len(u) < 50000 else None
    aucs = []
    if groups is not None:
        for _ in range(min(n_boot, 200)):
            pick = rng.integers(0, len(u), len(u))
            ix = np.concatenate([groups[k] for k in pick])
            aucs.append(auc(p[ix], y[ix]))
    out = {"log_loss": cluster_ci(ll, clusters, n_boot), "brier": cluster_ci(br, clusters, n_boot),
           "accuracy": cluster_ci(acc, clusters, n_boot),
           "auc": {"n": int(len(p)), "mean": auc(p, y),
                   "lo": float(np.nanpercentile(aucs, 2.5)) if aucs else None,
                   "hi": float(np.nanpercentile(aucs, 97.5)) if aucs else None},
           "mean_p": float(p.mean()), "base_rate": float(y.mean())}
    return out


# ================================================================================================
# replayed-turn decisions
# ================================================================================================

def attack_baseline(text: str, ctx: dict, ids) -> dict:
    """'attack when the attacker's power >= the best untapped blocker's toughness' (approximate):
    printed power of the attacker (unknown -> 1); B's creatures untapped at the turn-start spec
    with printed toughness (tokens and unknown cards skipped); a flyer is only blockable by
    Flying/Reach creatures; no untapped (able) blocker -> attack."""
    name = text[len("attack with: "):].rstrip("?").strip() if text.startswith("attack with: ") else ""
    info = ids.info(name)
    power = info.power if info.power is not None else 1
    fly = "Flying" in info.keywords
    bl = [b for b in ctx.get("blockers", []) if b["fly"] or not fly]
    best_t = max((b["t"] for b in bl), default=None)
    return {"power": power, "best_t": best_t, "attack": best_t is None or power >= best_t}


def replay_tables(rp: list[dict], ids) -> dict:
    """Decisions of reproduced replayed turns -> per-kind tables: 'attack' (CHOOSE_USE 'attack with',
    exact) and 'priority' (PRIORITY decisions with their recorded-plays set; the chosen play is
    imputed_order, a Pass once nothing is left is exact)."""
    att = defaultdict(list)
    pri = defaultdict(list)
    for t in rp:
        if not t.get("reproduced"):
            continue
        base = {k: t.get(k) for k in ("row", "turn", "split", "won", "wr_bucket", "n_games_bucket", "on_play", "tier")}
        for d in t.get("decisions") or []:
            f = d.get("features")
            if f is None or len(f) == 0:
                continue
            if d["type"] == "CHOOSE_USE" and (d.get("text") or "").startswith("attack with: ") \
                    and d.get("label_kind") == "exact" and d.get("chosen") in ("yes", "no"):
                ab = attack_baseline(d["text"], t.get("attack_ctx") or {}, ids)
                att["features"].append(f)
                att["y"].append(1 if d["chosen"] == "yes" else 0)
                att["heur"].append(int(ab["attack"]))
                att["power"].append(ab["power"])
                for k, v in base.items():
                    att["meta/" + k].append(v)
            elif d["type"] == "PRIORITY" and d.get("label_kind") in ("exact", "imputed_order"):
                legal, lidx = d["legal"], d["legal_idx"]
                S = list(d.get("set") or [])
                if not S and d.get("chosen"):
                    S = [d["chosen"]]
                li, lab2i = _uniq_legal(legal, lidx)
                si = sorted({lab2i[s] for s in S if s in lab2i})
                if len(li) < 2 or not si:
                    continue
                pri["features"].append(f)
                pri["legal"].append(li)
                pri["labels"].append(legal)
                pri["lab_idx"].append(lidx)
                pri["S"].append(si)
                pri["kind"].append(d["label_kind"])
                pri["step"].append(d.get("step"))
                for k, v in base.items():
                    pri["meta/" + k].append(v)
    return {"attack": dict(att), "priority": dict(pri)}


def csr(features: list) -> tuple[np.ndarray, np.ndarray]:
    ptr = np.zeros(len(features) + 1, np.int64)
    ptr[1:] = np.cumsum([len(f) for f in features])
    return (np.concatenate(features).astype(np.int32) if features else np.zeros(0, np.int32)), ptr


def priority_table_from_replay(p: dict, sel: np.ndarray) -> dict:
    """A replay priority table in the ts_table shape (for the shared metrics)."""
    feats = [p["features"][i] for i in sel]
    ind, off = csr(feats)
    l_ptr, s_ptr = [0], [0]
    L, S = [], []
    for i in sel:
        L.extend(p["legal"][i])
        l_ptr.append(len(L))
        S.extend(p["S"][i])
        s_ptr.append(len(S))
    t = {"indices": ind, "offsets": off, "legal_indptr": np.asarray(l_ptr), "legal_idx": np.asarray(L, np.int32),
         "set_indptr": np.asarray(s_ptr), "set_idx": np.asarray(S, np.int32),
         "meta/label_status": np.ones(len(sel), np.int32),
         "meta/row": np.asarray([p["meta/row"][i] for i in sel]),
         "meta/turn": np.asarray([p["meta/turn"][i] for i in sel]),
         "labels": [p["labels"][i] for i in sel], "lab_idx": [p["lab_idx"][i] for i in sel],
         "kind": [p["kind"][i] for i in sel]}
    return t


# ================================================================================================
# decision statistics (part 2)
# ================================================================================================

def _wilson(k: int, n: int) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "p": None, "lo": None, "hi": None}
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": int(k), "n": int(n), "p": round(p, 4), "lo": round(c - h, 4), "hi": round(c + h, 4)}


def decision_stats(ts: list[dict]) -> dict:
    """Status of every turn-start request, and for the reached decisions: options, |S|, Pass
    labels, unreachable labels, by tier and turn (Wilson 95% CIs)."""
    st = Counter(r.get("status") for r in ts)
    ok = [r for r in ts if r.get("status") == "ok"]

    def block(rs):
        n = len(rs)
        nontriv = sum(1 for r in rs if sum(1 for x in r["legal"] if x != "Pass") >= 2)
        ls = Counter(r["label_status"] for r in rs)
        acted = [r for r in rs if r["label_status"] != "pass"]
        return {"n": n, "nontrivial_ge2_nonpass": _wilson(nontriv, n),
                "pass_label": _wilson(ls["pass"], n), "unreachable": _wilson(ls["unreachable"], n),
                "unreachable_of_acted": _wilson(ls["unreachable"], len(acted)),
                "mean_legal_options": round(float(np.mean([len(r["legal"]) for r in rs])), 3) if rs else None,
                "S_size": dict(sorted(Counter(len(r["S"]) for r in rs if r["label_status"] == "set").items())),
                "mean_S_size_set": round(float(np.mean([len(r["S"]) for r in rs if r["label_status"] == "set"] or [0])), 3)}
    out = {"requests": len(ts), "status": dict(st), "reached": _wilson(len(ok), len(ts)), "all": block(ok),
           "legal_options": dict(sorted(Counter(min(len(r["legal"]), 8) for r in ok).items())),
           "by_tier": {t: block([r for r in ok if r.get("tier") == t]) for t in TIERS},
           "by_turn": {str(k): block([r for r in ok if min(r["turn"], 10) == k]) for k in range(1, 11)},
           "by_split": {s: len([r for r in ok if r["split"] == i]) for i, s in enumerate(SPLITS)},
           "attacked_share": _wilson(sum(1 for r in ok if r.get("attacked")), len(ok)),
           "acted_but_pass_legal_only_when_attacked": None}
    return out


# ================================================================================================
# pipeline stages (python -m draftzero.gameplay.imitation <stage>)
# ================================================================================================

CACHE = OUT_DIR / "cache"


def _save_table(t: dict, path: Path, action_type: int = 0) -> None:
    import h5py
    write_h5(t, path, action_type)
    with h5py.File(path, "a") as f:
        f.create_dataset("legal_labels_json", data=[json.dumps(x) for x in t["labels"]],
                         dtype=h5py.string_dtype())
        f.create_dataset("legal_label_idx_json", data=[json.dumps(x) for x in t["lab_idx"]],
                         dtype=h5py.string_dtype())
        if "kind" in t:
            f.create_dataset("replay_label_kind", data=[str(x) for x in t["kind"]], dtype=h5py.string_dtype())


def load_table(path: Path) -> dict:
    import h5py
    t = {}
    with h5py.File(path, "r") as f:
        for k in ("indices", "offsets", "legal_indptr", "legal_idx", "set_indptr", "set_idx", "z"):
            t[k] = f[k][:]
        for k in f["meta"]:
            t["meta/" + k] = f["meta"][k][:]
        t["labels"] = [json.loads(x) for x in f["legal_labels_json"].asstr()[:]]
        t["lab_idx"] = [json.loads(x) for x in f["legal_label_idx_json"].asstr()[:]]
        if "replay_label_kind" in f:
            t["kind"] = list(f["replay_label_kind"].asstr()[:])
    return t


def stage_tables(log=print) -> dict:
    """Shards -> decision statistics + HDF5 tables (turn-start per split; replayed attack and
    priority decisions per split)."""
    from draftzero.gameplay.ids import Ids
    import h5py
    ids = Ids.load()
    sh = OUT_DIR / "shards"
    ts = load_shard(sh / "ts.pkl")
    out = {"decision_stats": decision_stats(ts)}
    for i, s in enumerate(SPLITS):
        rs = [r for r in ts if r.get("split") == i and r.get("status") == "ok"]
        t = ts_table(rs)
        t["lab_idx"] = [r["legal_idx"] for r in rs]
        _save_table(t, OUT_DIR / "h5" / f"turnstart_{s}.h5")
        log(f"turnstart_{s}: {len(rs)} rows")
    del ts
    rp = load_shard(sh / "rp.pkl")
    rep = Counter()
    by_tier = defaultdict(Counter)
    kinds = Counter()
    for t in rp:
        rep["turns"] += 1
        rep["reproduced"] += bool(t.get("reproduced"))
        by_tier[t.get("tier")]["n"] += 1
        by_tier[t.get("tier")]["ok"] += bool(t.get("reproduced"))
        if t.get("reproduced"):
            for d in t.get("decisions") or []:
                kinds[f"{d['type']}:{d.get('label_kind')}"] += 1
    out["replay"] = {"turns": rep["turns"], "reproduced": _wilson(rep["reproduced"], rep["turns"]),
                     "by_tier": {k: _wilson(v["ok"], v["n"]) for k, v in sorted(by_tier.items(), key=lambda kv: str(kv[0]))},
                     "decision_kinds": dict(kinds.most_common())}
    rt = replay_tables(rp, ids)
    del rp
    att = rt["attack"]
    sp = np.asarray(att["meta/split"])
    for i, s in enumerate(SPLITS):
        sel = np.flatnonzero(sp == i)
        ind, off = csr([att["features"][j] for j in sel])
        path = OUT_DIR / "h5" / f"replay_attack_{s}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("indices", data=ind, compression="gzip", compression_opts=1)
            f.create_dataset("offsets", data=off)
            row = np.zeros((len(sel), A_DIM + 4), np.float32)
            y = np.asarray(att["y"])[sel]
            row[np.arange(len(sel)), y] = 1.0                         # binary head: slot 0 no, 1 yes
            won = np.asarray(att["meta/won"])[sel].astype(bool)
            row[:, A_DIM] = np.where(won, 1.0, -1.0)
            row[:, A_DIM + 2] = 1.0
            row[:, A_DIM + 3] = ACTION_TYPE["CHOOSE_USE"]
            f.create_dataset("row", data=row, compression="gzip", compression_opts=1)
            f.create_dataset("y", data=y)
            f.create_dataset("heur", data=np.asarray(att["heur"])[sel])
            f.create_dataset("power", data=np.asarray(att["power"])[sel])
            for k in ("row", "turn", "won", "n_games_bucket", "on_play"):
                f.create_dataset("meta/" + k, data=np.asarray([att["meta/" + k][j] or 0 for j in sel], np.int32))
            f.create_dataset("meta/wr_bucket", data=np.asarray(
                [att["meta/wr_bucket"][j] if att["meta/wr_bucket"][j] is not None else np.nan for j in sel], np.float32))
            f.attrs["layout"] = "LabeledStateWriter (CHOOSE_USE rows) + y/heur (draftzero.gameplay.imitation)"
        log(f"replay_attack_{s}: {len(sel)} rows")
    pri = rt["priority"]
    sp = np.asarray(pri["meta/split"])
    for i, s in enumerate(SPLITS):
        sel = np.flatnonzero(sp == i)
        t = priority_table_from_replay(pri, sel)
        t["z"] = np.where(np.asarray([pri["meta/won"][j] for j in sel], bool), 1.0, -1.0).astype(np.float32)
        t["S"] = [[] for _ in sel]
        for k in ("won", "n_games_bucket", "tier", "on_play"):
            vals = [pri["meta/" + k][j] for j in sel]
            if k == "tier":
                vals = [TIERS.index(v) if v in TIERS else -1 for v in vals]
            t["meta/" + k] = np.asarray([v if v is not None else -1 for v in vals], np.int32)
        t["meta/wr_bucket"] = np.asarray([pri["meta/wr_bucket"][j] if pri["meta/wr_bucket"][j] is not None
                                          else np.nan for j in sel], np.float32)
        _save_table(t, OUT_DIR / "h5" / f"replay_priority_{s}.h5")
        log(f"replay_priority_{s}: {len(sel)} rows")
    (OUT_DIR / "stats.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def subset_table(t: dict, sel: np.ndarray) -> dict:
    """Rows `sel` of a table (CSR fields re-packed, per-row fields sliced)."""
    sel = np.asarray(sel)
    out = {}
    for base, ptr_key in (("indices", "offsets"), ("legal_idx", "legal_indptr"), ("set_idx", "set_indptr")):
        ptr = t[ptr_key]
        parts = [t[base][ptr[i]:ptr[i + 1]] for i in sel]
        out[base] = np.concatenate(parts) if parts else t[base][:0]
        out[ptr_key] = np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype(np.int64)
    for k, v in t.items():
        if k in out:
            continue
        if isinstance(v, np.ndarray) and len(v) == len(t["offsets"]) - 1:
            out[k] = v[sel]
        elif isinstance(v, list) and len(v) == len(t["offsets"]) - 1:
            out[k] = [v[i] for i in sel]
    return out


def load_rows(path: Path, sel: np.ndarray, block: int = 4000) -> dict:
    """Rows `sel` (sorted) of an HDF5 table, streamed block by block so the whole feature array is
    never in memory (the shared 16 GB machine was swapping). Only what training needs: features,
    legal/set CSR, z, label status."""
    import h5py
    sel = np.asarray(sel)
    feats, legal, sets = [], [], []
    with h5py.File(path, "r") as f:
        off, lp, sp = f["offsets"][:], f["legal_indptr"][:], f["set_indptr"][:]
        li_all, si_all = f["legal_idx"][:], f["set_idx"][:]
        ls = f["meta/label_status"][:][sel]
        z = f["z"][:][sel]
        row = f["meta/row"][:][sel]
        n = len(off) - 1
        for a in range(0, n, block):
            b = min(n, a + block)
            pick = sel[(sel >= a) & (sel < b)]
            if not len(pick):
                continue
            chunk = f["indices"][off[a]:off[b]]
            for r in pick:
                feats.append(chunk[off[r] - off[a]:off[r + 1] - off[a]])
        for r in sel:
            legal.append(li_all[lp[r]:lp[r + 1]])
            sets.append(si_all[sp[r]:sp[r + 1]])
    ind, ptr = csr(feats)
    l_idx, l_ptr = csr(legal)
    s_idx, s_ptr = csr(sets)
    return {"indices": ind, "offsets": ptr, "legal_idx": l_idx, "legal_indptr": l_ptr, "set_idx": s_idx,
            "set_indptr": s_ptr, "z": z, "meta/label_status": ls, "meta/row": row}


def load_attack(path: Path) -> dict:
    import h5py
    with h5py.File(path, "r") as f:
        t = {k: f[k][:] for k in ("indices", "offsets", "y", "heur", "power")}
        for k in f["meta"]:
            t["meta/" + k] = f["meta"][k][:]
    return t


def predict(model, vocab, t: dict, log=None) -> dict:
    """gen33-architecture outputs for a table: pooled embedding, priority logits, binary logits, value."""
    rows, ptr = map_features(vocab, t["indices"], t["offsets"])
    emb = embed_all(model, rows, ptr, log=log)
    ho = head_outputs(model, emb)
    return {"emb": emb, "pa": ho["pa"].astype(np.float16), "bin": ho["bin"], "v": ho["v"],
            "mapped_share": float(len(rows) / max(1, len(t["indices"])))}


def _savez(name: str, **arrs) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / f"{name}.npz", **arrs)


def stage_gen33(names=("turnstart_test", "turnstart_val", "replay_priority_test"), log=print) -> None:
    model, vocab = load_gen33()
    for name in names:
        t = load_table(OUT_DIR / "h5" / f"{name}.h5")
        t0 = time.monotonic()
        p = predict(model, vocab, t)
        _savez(f"gen33_{name}", **p)
        log(f"gen33 {name}: {len(t['offsets']) - 1} rows in {time.monotonic() - t0:.0f} s, "
            f"mapped share {p['mapped_share']:.3f}")


def stage_gen33_attack(log=print) -> None:
    model, vocab = load_gen33()
    for s in SPLITS:
        t = load_attack(OUT_DIR / "h5" / f"replay_attack_{s}.h5")
        t0 = time.monotonic()
        p = predict(model, vocab, t)
        _savez(f"gen33_attack_{s}", **p)
        log(f"gen33 attack {s}: {len(t['y'])} rows in {time.monotonic() - t0:.0f} s")


def stage_gen33_train(n: int = 40000, seed: int = 0, log=print) -> None:
    """gen33 trunk embeddings of a random subset of the train split (for heads-only fine-tuning)."""
    model, vocab = load_gen33()
    t = load_table(OUT_DIR / "h5" / "turnstart_train.h5")
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(len(t["z"]), min(n, len(t["z"])), replace=False))
    sub = subset_table(t, sel)
    t0 = time.monotonic()
    p = predict(model, vocab, sub)
    _savez("gen33_turnstart_train_subset", sel=sel, **p)
    log(f"gen33 train subset: {len(sel)} rows in {time.monotonic() - t0:.0f} s")


def _train_vocab(t: dict, min_count: int = 10):
    from magezero.vocab import FeatureVocab
    counts = np.bincount(t["indices"].astype(np.int64), minlength=2_000_000)
    return FeatureVocab(np.flatnonzero(counts >= min_count), feature_hash_bins=2_000_000)


def _train_subset(n_train: int | None, seed: int = 0) -> tuple[dict, np.ndarray]:
    """The train rows used by the full-network runs: the random subset whose gen33 embeddings the
    heads-only runs use (gen33_turnstart_train_subset.npz), or a random n_train of it."""
    sel = np.load(CACHE / "gen33_turnstart_train_subset.npz")["sel"]
    if n_train is not None and n_train < len(sel):
        sel = np.sort(np.random.default_rng(seed).choice(sel, n_train, replace=False))
    return load_rows(OUT_DIR / "h5" / "turnstart_train.h5", sel), sel


def _scratch_vocab(t: dict, sample: int = 10000, k: int = 3, seed: int = 0):
    """MageZero's ignore-list rule (vocab.kept_feature_ids: drop ids seen in <= k states, keep one id
    per identical occurrence pattern) on a random sample of the train rows. k=3 on 10k rows (MageZero
    uses k=10 on its much larger replay buffer). Cuts the ~1,300 raw ids of a state to about gen33's
    length, which the attention cost needs."""
    from magezero.vocab import FeatureVocab, kept_feature_ids
    n = len(t["offsets"]) - 1
    pick = np.sort(np.random.default_rng(seed).choice(n, min(sample, n), replace=False))
    sub = subset_table({k_: t[k_] for k_ in ("indices", "offsets", "legal_idx", "legal_indptr", "set_idx",
                                               "set_indptr")}, pick)
    return FeatureVocab(kept_feature_ids(sub["indices"], sub["offsets"], k=k), feature_hash_bins=2_000_000)


def stage_scratch(n_train: int | None = None, budget_s: float = 600, lr: float = 3e-4, value_weight: float = 0.1,
                  seed: int = 0, tag: str | None = None, test_rows: int | None = None, eval_every_s: float = 180,
                  log=print) -> dict:
    """(d): the gen33 architecture (NetTransformer, 2 layers, d=512) from scratch on human train rows
    (the heads-only subset, or n_train of it), vocab by MageZero's ignore rule, rows initialised as
    MageZero does (vocab.initial_rows); wall-clock budget."""
    _torch_env()
    _require_exp1_engine()
    import torch
    from magezero.model import NetTransformer
    from magezero.vocab import initial_rows
    tr, _ = _train_subset(n_train, seed)
    tag = tag or f"scratch_{len(tr['z'])}"
    va = load_table(OUT_DIR / "h5" / "turnstart_val.h5")
    vocab = _scratch_vocab(tr)
    torch.manual_seed(seed)
    model = NetTransformer(num_embeddings=len(vocab), policy_size_A=A_DIM)
    with torch.no_grad():
        model.embedding.weight.copy_(torch.as_tensor(initial_rows(vocab.ids, 512)))
    rtr, ptr_tr = map_features(vocab, tr["indices"], tr["offsets"])
    rva, ptr_va = map_features(vocab, va["indices"], va["offsets"])
    log(f"{tag}: vocab {len(vocab)} rows, mean mapped length {np.diff(ptr_tr).mean():.0f}")
    model, info = train_full(model, rtr.astype(np.int32), ptr_tr, tr, rva.astype(np.int32), ptr_va, va,
                             budget_s=budget_s, lr=lr, value_weight=value_weight, seed=seed,
                             eval_every_s=eval_every_s, log=log)
    info.update(vocab_rows=len(vocab), n_train_rows=int(len(tr["z"])),
                mean_tokens=float(np.diff(ptr_tr).mean()))
    del tr, rtr
    _predict_tests(model, vocab, tag, test_rows)
    (CACHE / f"{tag}.json").write_text(json.dumps(info, indent=1))
    log(f"{tag}: {info['steps']} steps, {info['samples_seen']} samples, best val set-NLL {info['best_val_set_nll']:.4f}")
    return info


def _predict_tests(model, vocab, tag: str, test_rows: int | None) -> None:
    if test_rows:            # learning-curve runs: a fixed random subset of the test split only
        te = load_table(OUT_DIR / "h5" / "turnstart_test.h5")
        sel = np.sort(np.random.default_rng(12345).choice(len(te["z"]), min(test_rows, len(te["z"])), replace=False))
        p = predict(model, vocab, subset_table(te, sel))
        _savez(f"{tag}_turnstart_testsubset", pa=p["pa"], v=p["v"], sel=sel)
    else:
        for name in ("turnstart_test", "replay_priority_test"):
            te = load_table(OUT_DIR / "h5" / f"{name}.h5")
            p = predict(model, vocab, te)
            _savez(f"{tag}_{name}", pa=p["pa"], v=p["v"])


def stage_finetune(budget_s: float = 600, lr: float = 1e-4, value_weight: float = 0.1, seed: int = 0,
                   n_train: int | None = None, tag: str = "finetune", test_rows: int | None = None,
                   log=print) -> dict:
    """(e): gen33 fine-tuned end to end on the human train rows (the heads-only subset). OOV
    policy: train features that gen33's vocab lacks and that MageZero's ignore rule keeps (same rule
    as the scratch vocab) are appended (FeatureVocab.extend, ascending id order) with rows
    initialised by MageZero's rule for new features (vocab.initial_rows: N(0,1) keyed by the
    feature id; gen33's trained rows have sd 1.00, so the scale matches); every other unknown id is
    dropped, as MageZero's inference server drops unknown ids."""
    _torch_env()
    from magezero.vocab import initial_rows
    model, vocab = load_gen33()
    tr, _ = _train_subset(n_train, seed)
    va = load_table(OUT_DIR / "h5" / "turnstart_val.h5")
    old = len(vocab)
    added = vocab.extend(_scratch_vocab(tr).ids)
    model.resize_embedding(len(vocab), init_rows=initial_rows(vocab.ids[old:], 512))
    rtr, ptr_tr = map_features(vocab, tr["indices"], tr["offsets"])
    rva, ptr_va = map_features(vocab, va["indices"], va["offsets"])
    log(f"{tag}: vocab {old} + {added} rows, mean mapped length {np.diff(ptr_tr).mean():.0f}")
    model, info = train_full(model, rtr.astype(np.int32), ptr_tr, tr, rva.astype(np.int32), ptr_va, va,
                             budget_s=budget_s, lr=lr, value_weight=value_weight, seed=seed, warmup=50,
                             eval_every_s=180, log=log)
    info.update(vocab_rows_gen33=old, vocab_rows_added=int(added), n_train_rows=int(len(tr["z"])),
                mean_tokens=float(np.diff(ptr_tr).mean()))
    del tr, rtr
    _predict_tests(model, vocab, tag, test_rows)
    (CACHE / f"{tag}.json").write_text(json.dumps(info, indent=1))
    log(f"{tag}: {info['steps']} steps, best val set-NLL {info['best_val_set_nll']:.4f}, +{added} vocab rows")
    return info


def train_binary_head(model, E_tr, y_tr, E_va, y_va, *, epochs: int = 40, lr: float = 1e-3, batch: int = 256,
                      seed: int = 0) -> tuple[Any, dict]:
    """The binary (CHOOSE_USE) head re-trained on frozen gen33 trunk embeddings of human attack
    decisions (cross-entropy, early stopping on val)."""
    import copy
    import torch
    torch.manual_seed(seed)
    m = copy.deepcopy(model.binary_head).to("cpu")
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Et, yt = torch.as_tensor(E_tr, dtype=torch.float32), torch.as_tensor(y_tr, dtype=torch.long)
    Ev, yv = torch.as_tensor(E_va, dtype=torch.float32), torch.as_tensor(y_va, dtype=torch.long)
    rng = np.random.default_rng(seed)
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(epochs):
        m.train()
        perm = torch.as_tensor(rng.permutation(len(yt)))
        for k in range(0, len(perm), batch):
            j = perm[k:k + batch]
            loss = torch.nn.functional.cross_entropy(m(Et[j]), yt[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(torch.nn.functional.cross_entropy(m(Ev), yv))
        hist.append(round(vl, 4))
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= 4:
                break
    m.load_state_dict(best_state)
    return m, {"val_ce": hist, "best_val_ce": best}


def stage_heads(sizes=(5000, 15000, None), log=print) -> dict:
    """(e, heads only): gen33's priority + value heads re-trained on its frozen trunk embeddings of
    the human train subset, at several train sizes; and the binary head on attack decisions."""
    _torch_env()
    import torch
    model, _ = load_gen33()
    tr_full = load_table(OUT_DIR / "h5" / "turnstart_train.h5")
    d = np.load(CACHE / "gen33_turnstart_train_subset.npz")
    tr = subset_table(tr_full, d["sel"])
    E_tr = d["emb"].astype(np.float32)
    va = load_table(OUT_DIR / "h5" / "turnstart_val.h5")
    E_va = np.load(CACHE / "gen33_turnstart_val.npz")["emb"].astype(np.float32)
    E_te = np.load(CACHE / "gen33_turnstart_test.npz")["emb"].astype(np.float32)
    E_rp = np.load(CACHE / "gen33_replay_priority_test.npz")["emb"].astype(np.float32)
    info = {}
    for n in sizes:
        m, inf = train_heads(model, E_tr, tr, E_va, va, n_train=n, epochs=30, log=None)
        tag = f"heads_{n or len(E_tr)}"
        ho = head_outputs(m, E_te)
        ho2 = head_outputs(m, E_rp)
        _savez(f"{tag}_turnstart_test", pa=ho["pa"].astype(np.float16), v=ho["v"])
        _savez(f"{tag}_replay_priority_test", pa=ho2["pa"].astype(np.float16), v=ho2["v"])
        info[tag] = inf
        log(f"{tag}: best val set-NLL {inf['best_val_set_nll']:.4f} ({inf['n_train']} rows)")
    # attack: binary head on frozen trunk embeddings
    A = {s: np.load(CACHE / f"gen33_attack_{s}.npz") for s in SPLITS}
    Y = {s: load_attack(OUT_DIR / "h5" / f"replay_attack_{s}.h5")["y"] for s in SPLITS}
    mb, inf = train_binary_head(model, A["train"]["emb"], Y["train"], A["val"]["emb"], Y["val"])
    with torch.no_grad():
        logits = mb(torch.as_tensor(A["test"]["emb"], dtype=torch.float32)).numpy()
    _savez("binary_head_attack_test", bin=logits)
    info["binary_head"] = {**inf, "n_train": int(len(Y["train"]))}
    (CACHE / "heads.json").write_text(json.dumps(info, indent=1))
    return info


def concat_tables(a: dict, b: dict) -> dict:
    """Row-concatenation of two tables (the CSR fields and the per-row fields both hold)."""
    out = {}
    for base, ptr in (("indices", "offsets"), ("legal_idx", "legal_indptr"), ("set_idx", "set_indptr")):
        out[base] = np.concatenate([a[base], b[base]])
        out[ptr] = np.concatenate([a[ptr], b[ptr][1:] + a[ptr][-1]])
    na, nb = len(a["offsets"]) - 1, len(b["offsets"]) - 1
    for k in a:
        if k in out or k not in b:
            continue
        va, vb = a[k], b[k]
        if isinstance(va, np.ndarray) and len(va) == na and len(vb) == nb:
            out[k] = np.concatenate([va, vb.astype(va.dtype)])
        elif isinstance(va, list) and len(va) == na and len(vb) == nb:
            out[k] = va + vb
    return out


def stage_replay_heads(n: int = 20000, seed: int = 0, log=print) -> dict:
    """Heads trained on the replayed-turn PRIORITY decisions (every decision of the turn, the exact
    'Pass once nothing is left' rows included), alone and mixed with the turn-start rows: gen33
    trunk embeddings of a train subset (GPU), then heads-only training (CPU)."""
    model, vocab = load_gen33()
    tr = load_table(OUT_DIR / "h5" / "replay_priority_train.h5")
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(len(tr["z"]), min(n, len(tr["z"])), replace=False))
    tr = subset_table(tr, sel)
    va = load_table(OUT_DIR / "h5" / "replay_priority_val.h5")
    t0 = time.monotonic()
    p_tr = predict(model, vocab, tr)
    p_va = predict(model, vocab, va)
    log(f"replay priority embeddings: {len(sel)} + {len(va['z'])} rows in {time.monotonic() - t0:.0f} s")
    E_te = np.load(CACHE / "gen33_turnstart_test.npz")["emb"].astype(np.float32)
    E_rp = np.load(CACHE / "gen33_replay_priority_test.npz")["emb"].astype(np.float32)
    ts_full = load_table(OUT_DIR / "h5" / "turnstart_train.h5")
    d = np.load(CACHE / "gen33_turnstart_train_subset.npz")
    ts_tr = subset_table(ts_full, d["sel"])
    ts_va = load_table(OUT_DIR / "h5" / "turnstart_val.h5")
    E_tsva = np.load(CACHE / "gen33_turnstart_val.npz")["emb"].astype(np.float32)
    info = {}
    runs = {"heads_replay": (tr, p_tr["emb"], va, p_va["emb"]),
            "heads_mix": (concat_tables(ts_tr, tr), np.concatenate([d["emb"], p_tr["emb"]]),
                          concat_tables(ts_va, va), np.concatenate([E_tsva.astype(np.float16), p_va["emb"]]))}
    for tag, (t_tr, e_tr, t_va, e_va) in runs.items():
        m, inf = train_heads(model, e_tr.astype(np.float32), t_tr, e_va.astype(np.float32), t_va, epochs=30)
        ho, ho2 = head_outputs(m, E_te), head_outputs(m, E_rp)
        _savez(f"{tag}_turnstart_test", pa=ho["pa"].astype(np.float16), v=ho["v"])
        _savez(f"{tag}_replay_priority_test", pa=ho2["pa"].astype(np.float16), v=ho2["v"])
        info[tag] = inf
        log(f"{tag}: best val set-NLL {inf['best_val_set_nll']:.4f} ({inf['n_train']} rows)")
    (CACHE / "heads_replay.json").write_text(json.dumps(info, indent=1))
    return info


def stage_gen33_extras(log=print) -> None:
    """gen33 on the extras (perfectInfo=true encodings) and the matching perfectInfo=false rows."""
    model, vocab = load_gen33()
    with open(OUT_DIR / "shards" / "extras.pkl", "rb") as f:
        ex = pickle.load(f)
    ex = [r for r in ex if r.get("status") == "ok"]
    for r in ex:
        r["lab_idx"] = r["legal_idx"]
    t = ts_table(ex)
    t["lab_idx"] = [r["legal_idx"] for r in ex]
    t["variant"] = [r["variant"] for r in ex]
    t["pair"] = [bool(r.get("pair")) for r in ex]
    p = predict(model, vocab, t)
    _savez("gen33_extras", pa=p["pa"], v=p["v"], row=t["meta/row"], turn=t["meta/turn"],
           variant=np.asarray(t["variant"]), pair=np.asarray(t["pair"]))
    with open(CACHE / "extras_table.pkl", "wb") as f:
        pickle.dump({k: v for k, v in t.items() if k not in ("indices",)}, f)
    log(f"gen33 extras: {len(ex)} rows")


# ================================================================================================
# report
# ================================================================================================

WR_BANDS = ((0.0, 0.50), (0.50, 0.54), (0.54, 0.58), (0.58, 1.01))
TURN_BANDS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 99))
PRIOR_TEMP, PRIOR_BONUS = 1.5, 0.1   # MCTSNode.setPriors defaults (magezero_mcts.md 4.2)


def wr_band(w: float) -> str:
    """The WR_BANDS label of a 17lands win-rate bucket. The buckets are multiples of 0.02 stored as
    float32 (0.58 reads back as 0.5799999833). NumPy 2 compares a float32 scalar with a Python float
    in float32 (NEP 50), so the band is right there, but as a Python float (or under NumPy 1) the
    0.58 bucket falls into the band below: round first so the result does not depend on the type."""
    w = round(float(w), 2)
    return next(f"{a:.2f}-{b:.2f}" for a, b in WR_BANDS if a <= w < b)


def search_prior(logits: np.ndarray, is_pass: np.ndarray, temp: float = PRIOR_TEMP,
                 bonus: float = PRIOR_BONUS) -> np.ndarray:
    """The priors MageZero's search derives from a head's logits over one decision's legal children
    (MCTSNode.setPriors): softmax(logit / priorTemp) over the children, plus priorBonus for every
    non-Pass (non-mana) child; renormalised here so it can be scored as a distribution."""
    x = np.asarray(logits, np.float64) / temp
    p = np.exp(x - x.max())
    p = p / p.sum() + bonus * ~np.asarray(is_pass, bool)
    return p / p.sum()


def auc_score(p: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC (Mann-Whitney, ties at mid-rank); nan when only one class is present."""
    from scipy.stats import rankdata
    y = np.asarray(y) > 0.5
    if y.all() or (~y).all():
        return float("nan")
    r = rankdata(p)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def cluster_auc_ci(p: np.ndarray, y: np.ndarray, clusters: np.ndarray, n_boot: int = 500, seed: int = 0) -> dict:
    """AUC with a 95% cluster-bootstrap CI (clusters = games)."""
    u, inv = np.unique(clusters, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(u) + 1))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(u), len(u))
        ix = np.concatenate([order[starts[k]:starts[k + 1]] for k in pick])
        bs.append(auc_score(p[ix], y[ix]))
    return {"n": int(len(p)), "games": int(len(u)), "mean": auc_score(p, y),
            "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5))}


def _ci(x, clusters) -> dict:
    return cluster_ci(np.asarray(x, np.float64), np.asarray(clusters))


def with_pass_if_attacked(t: dict) -> dict:
    """The coach's label variant (coach.human_17lands): S plus Pass when the user attacked that turn
    (some of its plays may have come after combat) and Pass is legal."""
    pass_idx = 0
    s_idx, s_ptr = [], [0]
    for r in range(len(t["z"])):
        S = list(t["set_idx"][t["set_indptr"][r]:t["set_indptr"][r + 1]])
        li = t["legal_idx"][t["legal_indptr"][r]:t["legal_indptr"][r + 1]]
        if S and t["meta/attacked"][r] and pass_idx in li and pass_idx not in S:
            S.append(pass_idx)
        s_idx.extend(S)
        s_ptr.append(len(s_idx))
    return {**t, "set_idx": np.asarray(s_idx, np.int32), "set_indptr": np.asarray(s_ptr, np.int64)}


def s_category_shares(t: dict) -> dict:
    c = Counter()
    for r in range(len(t["z"])):
        if t["meta/label_status"][r] == 2:
            c["unreachable"] += 1
            continue
        cats = sorted({_category(_label_of(t, r, i)) for i in t["set_idx"][t["set_indptr"][r]:t["set_indptr"][r + 1]]})
        c["+".join(cats)] += 1
    n = sum(c.values())
    return {k: {"n": v, "share": round(v / n, 4)} for k, v in c.most_common()}


def _label_of(t: dict, r: int, idx: int) -> str:
    """The first legal label of row r with vocab index idx."""
    for lab, i in zip(t["labels"][r], t["lab_idx"][r]):
        if i == idx:
            return lab
    return ""


def _category(label: str) -> str:
    if label == "Pass":
        return "pass"
    if label.startswith("Play "):
        return "land"
    if label.startswith("Cast "):
        return "cast"
    if label.startswith("Flashback"):
        return "flashback"
    return "activation"


def _policy_block(t: dict, scores: list, prob: bool, sel_rows=None) -> dict:
    pr = policy_rows(t, scores, probabilistic=prob)
    rows = pr["rows"]
    games = t["meta/row"][rows]
    keep = np.ones(len(rows), bool) if sel_rows is None else np.isin(rows, sel_rows)
    out = {"top1": _ci(pr["top1"][keep], games[keep]), "top3": _ci(pr["top3"][keep], games[keep])}
    if prob:
        out["set_nll"] = _ci(pr["nll"][keep], games[keep])
    return out, pr


def _card_rarity() -> dict:
    import csv
    out = {}
    try:
        with open(REPO / "data" / "17lands" / "cards.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("expansion") in ("FDN", "SPG") and r.get("name"):
                    out.setdefault(r["name"], r.get("rarity"))
    except OSError:
        pass
    return out


def report(log=print) -> dict:
    """Every metric of the experiment -> data/gameplay/imitation/results.json and results.md."""
    from draftzero.gameplay.ids import Ids
    ids = Ids.load()

    def mv_of(name):
        return cost_mv(ids.info(name).mana_cost)
    res: dict = {"date": time.strftime("%Y-%m-%d")}
    stats = json.loads((OUT_DIR / "stats.json").read_text())
    res["dataset"] = {"build": json.loads((OUT_DIR / "shards" / "build_stats.json").read_text()), **stats}
    te = load_table(OUT_DIR / "h5" / "turnstart_test.h5")
    games = te["meta/row"]
    models = {}
    for tag in ["gen33"] + [p.stem[:-len("_turnstart_test")] for p in sorted(CACHE.glob("*_turnstart_test.npz"))
                            if not p.stem.startswith("gen33")]:
        f = CACHE / f"{tag}_turnstart_test.npz"
        if f.exists():
            models[tag] = np.load(f)
    scores = {"uniform": (legal_scores(te, kind="uniform"), True),
              "heuristic": (legal_scores(te, kind="heuristic", mv_of=mv_of, lab_idx=te["lab_idx"]), False)}
    for tag, d in models.items():
        scores[tag] = (legal_scores(te, d["pa"].astype(np.float32), kind="model"), True)
    nontriv = np.flatnonzero(te["meta/n_nonpass"] >= 2)
    ge4 = np.flatnonzero(te["meta/n_legal"] >= 4)
    no_land = np.flatnonzero([not any(_label_of(te, r, i).startswith("Play ")
                                      for i in te["set_idx"][te["set_indptr"][r]:te["set_indptr"][r + 1]])
                              and any(lab.startswith("Cast ") for lab in te["labels"][r])
                              for r in range(len(te["z"]))])
    te_pp = with_pass_if_attacked(te)
    res["S_category_shares"] = s_category_shares(te)
    pol, prs = {}, {}
    for name, (sc, prob) in scores.items():
        pol[name], prs[name] = _policy_block(te, sc, prob)
        pol[name]["nontrivial"] = _policy_block(te, sc, prob, nontriv)[0]
        pol[name]["no_land_in_S_cast_legal"] = _policy_block(te, sc, prob, no_land)[0]
        pol[name]["S_plus_pass_if_attacked"] = _policy_block(te_pp, sc, prob)[0]
        pol[name]["ge4_legal_options"] = _policy_block(te, sc, prob, ge4)[0]   # top-3 is not automatic
        log(f"policy {name}: top1 {pol[name]['top1']['mean']:.4f} top3 {pol[name]['top3']['mean']:.4f}")
    res["policy_test"] = pol
    # learning-curve runs evaluated on a fixed test subset, with the full-test models on the same rows
    lc = {}
    for f in sorted(CACHE.glob("*_turnstart_testsubset.npz")):
        d = np.load(f)
        sel = d["sel"]
        sub = subset_table(te, sel)
        tag = f.stem[:-len("_turnstart_testsubset")]
        lc[tag] = _policy_block(sub, legal_scores(sub, d["pa"].astype(np.float32), kind="model"), True)[0]
        info = CACHE / f"{tag}.json"
        if info.exists():
            lc[tag]["training"] = {k: v for k, v in json.loads(info.read_text()).items() if k != "history"}
        for m, dd in models.items():
            if m not in lc:
                lc[m] = _policy_block(sub, legal_scores(sub, dd["pa"][sel].astype(np.float32), kind="model"), True)[0]
    if lc:
        res["learning_curve_test_subset"] = lc
    res["policy_test_n"] = {"rows_with_label": int(len(prs["uniform"]["rows"])),
                            "unreachable_excluded": int((te["meta/label_status"] == 2).sum()),
                            "games": int(len(np.unique(games))), "nontrivial_rows": int(len(nontriv))}
    # by tier / by turn / Maia-style skill (n_games >= 100) for the main models
    heads_tags = sorted((m for m in prs if re.fullmatch(r"heads_\d+", m)), key=lambda m: int(m.split("_")[1]))
    scratch_tags = sorted((m for m in prs if re.fullmatch(r"scratch_\d+", m)), key=lambda m: int(m.split("_")[1]))
    main = [m for m in ("uniform", "heuristic", "gen33", *heads_tags[-1:], "heads_mix", "finetune", *scratch_tags[-1:])
            if m in prs]

    def breakdown(key_fn, keys):
        out = {}
        for m in main:
            pr = prs[m]
            rows = pr["rows"]
            kv = np.asarray([key_fn(r) for r in rows], dtype=object)
            out[m] = {str(k): _ci(pr["top1"][kv == k], games[rows][kv == k]) for k in keys}
        return out
    res["top1_by_tier"] = breakdown(lambda r: TIERS[te["meta/tier"][r]] if te["meta/tier"][r] >= 0 else "?", TIERS)
    res["top1_by_turn"] = breakdown(lambda r: next(f"{a}-{b}" if b < 99 else f"{a}+" for a, b in TURN_BANDS
                                                   if a <= te["meta/turn"][r] <= b), [f"{a}-{b}" if b < 99 else f"{a}+" for a, b in TURN_BANDS])

    def band(r):
        if te["meta/n_games_bucket"][r] < 100 or np.isnan(te["meta/wr_bucket"][r]):
            return "excluded"
        return wr_band(te["meta/wr_bucket"][r])
    res["top1_by_skill_ngames_ge100"] = breakdown(band, [f"{a:.2f}-{b:.2f}" for a, b in WR_BANDS])
    # skill gradient: strongest band minus weakest, bootstrapped over games within each band
    grad = {}
    lo_b, hi_b = f"{WR_BANDS[0][0]:.2f}-{WR_BANDS[0][1]:.2f}", f"{WR_BANDS[-1][0]:.2f}-{WR_BANDS[-1][1]:.2f}"
    for m in main:
        pr = prs[m]
        kv = np.asarray([band(r) for r in pr["rows"]], dtype=object)
        a, b = pr["top1"][kv == lo_b], pr["top1"][kv == hi_b]
        ga, gb = games[pr["rows"]][kv == lo_b], games[pr["rows"]][kv == hi_b]
        ua, ia = np.unique(ga, return_inverse=True)
        ub, ib = np.unique(gb, return_inverse=True)
        sa, ca, sb, cb = np.bincount(ia, a), np.bincount(ia), np.bincount(ib, b), np.bincount(ib)
        rng = np.random.default_rng(0)
        ds = [sb[y_].sum() / cb[y_].sum() - sa[x_].sum() / ca[x_].sum()
              for x_, y_ in ((rng.integers(0, len(ua), len(ua)), rng.integers(0, len(ub), len(ub))) for _ in range(1000))]
        grad[m] = {"mean": float(b.mean() - a.mean()), "lo": float(np.percentile(ds, 2.5)),
                   "hi": float(np.percentile(ds, 97.5)), "n_low": int(len(a)), "n_high": int(len(b))}
    res["top1_skill_gradient_top_minus_bottom_band"] = grad
    # paired differences between models on the same test rows (cluster bootstrap over games)
    rows_ = prs["uniform"]["rows"]
    pdiff = {}
    for a, b in (("heads_15000", "heads_5000"), (heads_tags[-1] if heads_tags else "", "heads_15000"),
                 (heads_tags[-1] if heads_tags else "", "heuristic"), ("heads_mix", heads_tags[-1] if heads_tags else ""),
                 ("heuristic", "gen33"), ("finetune", "gen33")):
        if a in prs and b in prs and a != b:
            pdiff[f"{a} - {b}"] = {"top1": _ci(prs[a]["top1"] - prs[b]["top1"], games[rows_])}
            if "nll" in prs[a] and "nll" in prs[b]:
                pdiff[f"{a} - {b}"]["set_nll"] = _ci(prs[a]["nll"] - prs[b]["nll"], games[rows_])
    res["paired_differences_test"] = pdiff
    # gen33 as MageZero's search would use it: softmax(logit / priorTemp) over the legal children
    # + priorBonus for non-Pass children (MCTSNode.setPriors); experiment #1 had this prior OFF
    if "gen33" in models:
        pa = models["gen33"]["pa"].astype(np.float32)
        t1, nl = [], []
        for r in rows_:
            li = te["legal_idx"][te["legal_indptr"][r]:te["legal_indptr"][r + 1]]
            S = te["set_idx"][te["set_indptr"][r]:te["set_indptr"][r + 1]]
            p = search_prior(pa[r, li], li == 0)
            good = np.isin(li, S)
            t1.append(topk_in_set(p, np.arange(len(li)), np.flatnonzero(good), 1))
            nl.append(-math.log(max(p[good].sum(), 1e-12)))
        res["gen33_as_search_prior"] = {"prior_temp": PRIOR_TEMP, "prior_bonus": PRIOR_BONUS,
                                        "top1": _ci(t1, games[rows_]), "set_nll": _ci(nl, games[rows_])}
    set_cat = {}
    for r in range(len(te["z"])):
        cats = sorted({_category(_label_of(te, r, i)) for i in te["set_idx"][te["set_indptr"][r]:te["set_indptr"][r + 1]]})
        set_cat[r] = "+".join(cats) if cats in (["land"], ["cast"], ["cast", "land"], ["pass"]) else "other"
    res["top1_by_set_category"] = breakdown(lambda r: set_cat[r], ["land", "cast+land", "cast", "pass", "other"])
    # disagreement analysis: gen33 vs humans, by the category of the human's set and of gen33's pick
    dis = {}
    if "gen33" in prs:
        rar = _card_rarity()
        pa = models["gen33"]["pa"].astype(np.float32)
        cat_hits, pick_cat, miss_cards, hit_cards, pS_by = defaultdict(list), Counter(), Counter(), Counter(), defaultdict(list)
        rar_hits = defaultdict(list)
        for k, r in enumerate(prs["gen33"]["rows"]):
            li = te["legal_idx"][te["legal_indptr"][r]:te["legal_indptr"][r + 1]]
            S = te["set_idx"][te["set_indptr"][r]:te["set_indptr"][r + 1]]
            first = {}
            for lab, i in zip(te["labels"][r], te["lab_idx"][r]):
                first.setdefault(i, lab)
            labs = [first.get(i, "") for i in li]
            sc = pa[r, li]
            p = np.exp(sc - sc.max())
            p /= p.sum()
            top = labs[int(np.argmax(sc))]
            scats = sorted({_category(first.get(i, "")) for i in S})
            key = "+".join(scats)
            hit = prs["gen33"]["top1"][k]
            cat_hits[key].append(hit)
            pS_by[key].append(float(p[np.isin(li, S)].sum()))
            pick_cat[(key, _category(top))] += 1
            for i in S:
                lab = first.get(i, "")
                if lab.startswith("Cast "):
                    (hit_cards if hit >= 0.5 else miss_cards)[lab[5:]] += 1
                    rar_hits[rar.get(lab[5:], "?")].append(float(p[list(li).index(i)]))
        # the unmasked 1024-way softmax: the space MageZero's trainer fits (KL over all A slots, no
        # legal mask). Search does NOT use it: setPriors renormalises over the legal children.
        z = pa - pa.max(1, keepdims=True)
        full = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        lm = np.asarray([full[r, te["legal_idx"][te["legal_indptr"][r]:te["legal_indptr"][r + 1]]].sum()
                         for r in range(len(te["z"]))])
        dis["unmasked_mass_on_legal"] = _ci(lm, games)
        dis["unmasked_mass_on_pass"] = _ci(full[:, 0], games)
        dis["top1_by_human_set_category"] = {k: {"n": len(v), "top1": round(float(np.mean(v)), 4),
                                                 "mean_p_S": round(float(np.mean(pS_by[k])), 4)}
                                             for k, v in sorted(cat_hits.items(), key=lambda kv: -len(kv[1])) if len(v) >= 30}
        dis["gen33_pick_category_given_human_set_category"] = {f"{a} -> {b}": c for (a, b), c in pick_cat.most_common(20)}
        dis["most_missed_casts"] = [{"card": c, "missed": n, "hit": hit_cards[c], "rarity": rar.get(c)}
                                    for c, n in miss_cards.most_common(25)]
        dis["p_of_human_cast_by_rarity"] = {k: {"n": len(v), "mean_p": round(float(np.mean(v)), 4)}
                                            for k, v in rar_hits.items() if len(v) >= 20}
        # a precombat-main Pass goes to combat, and 17lands does not say whether the casts came
        # before or after it: split gen33's Pass picks by whether the user attacked that turn
        pp = defaultdict(lambda: ([], []))
        for k, r in enumerate(prs["gen33"]["rows"]):
            li = te["legal_idx"][te["legal_indptr"][r]:te["legal_indptr"][r + 1]]
            key = "+".join(sorted({_category(_label_of(te, r, i))
                                   for i in te["set_idx"][te["set_indptr"][r]:te["set_indptr"][r + 1]]}))
            if key in ("cast", "cast+land", "land"):
                att = "attacked" if te["meta/attacked"][r] else "no_attack"
                pp[(key, att)][0].append(float(li[int(np.argmax(pa[r, li]))] == 0))
                pp[(key, att)][1].append(games[r])
        dis["pick_pass_by_set_category_and_attack"] = {f"{a} / {b}": _ci(v[0], v[1]) for (a, b), v in sorted(pp.items())}
    res["gen33_disagreement"] = dis
    # value
    val = {}
    won = te["meta/won"]
    base = float(load_table(OUT_DIR / "h5" / "turnstart_train.h5")["meta/won"].mean())
    vals = {"constant_train_base_rate": np.full(len(won), 2 * base - 1)}
    for tag, d in models.items():
        vals[tag] = d["v"].astype(np.float64)
    for tag, v in vals.items():
        val[tag] = {"all": value_metrics(v, won, games)}
        for a, b in TURN_BANDS:
            sel = (te["meta/turn"] >= a) & (te["meta/turn"] <= b)
            val[tag][f"turn {a}-{b}" if b < 99 else f"turn {a}+"] = {
                k: x for k, x in value_metrics(v[sel], won[sel], games[sel], n_boot=200).items()}
    res["value_test"] = val
    # perfect-information variants (extras)
    ex_path = CACHE / "gen33_extras.npz"
    if ex_path.exists():
        res["gen33_perfect_info"] = _extras_report(te, models.get("gen33"))
    # replayed turns: attack and priority
    res["replay"] = _replay_report(ids, mv_of, log)
    for f in sorted(CACHE.glob("*.json")):
        f = f.name
        if (CACHE / f).exists():
            res.setdefault("training", {})[f[:-5]] = json.loads((CACHE / f).read_text())
    (OUT_DIR / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (OUT_DIR / "results.md").write_text(render_results(res))
    return res


def _extras_report(te: dict, g33) -> dict:
    with open(CACHE / "extras_table.pkl", "rb") as f:
        et = pickle.load(f)
    d = np.load(CACHE / "gen33_extras.npz")
    pa = d["pa"].astype(np.float32)
    variant = np.asarray(et["variant"])
    key = list(zip(et["meta/row"].tolist(), et["meta/turn"].tolist()))
    # perfectInfo=false rows: the main test predictions, plus the partner rows' own
    base = {}
    te_key = {(int(r), int(n)): j for j, (r, n) in enumerate(zip(te["meta/row"], te["meta/turn"]))}

    def row_metrics(li, S, sc):
        x = sc - sc.max()
        p = np.exp(x) / np.exp(x).sum()
        top1 = topk_in_set(sc, np.arange(len(li)), np.flatnonzero(np.isin(li, S)), 1)
        return top1, -math.log(max(p[np.isin(li, S)].sum(), 1e-12))
    for j, (k, v) in enumerate(zip(key, variant)):
        if v == "pi_false":
            base[k] = ("extras", j)
    for k in set(key):
        if k not in base and k in te_key:
            base[k] = ("test", te_key[k])
    per = defaultdict(dict)
    for j, (k, v) in enumerate(zip(key, variant)):
        if v == "pi_false" or et["meta/label_status"][j] == 2:
            continue
        li = et["legal_idx"][et["legal_indptr"][j]:et["legal_indptr"][j + 1]]
        S = et["set_idx"][et["set_indptr"][j]:et["set_indptr"][j + 1]]
        per[k][v] = (li, S, pa[j, li], j)
    out = {}
    for group, variants in (("belief_rows", ("belief_pi_true",)), ("pair_rows", ("belief_pi_true", "true_pi_true"))):
        recs = defaultdict(list)
        n_mismatch = 0
        for k, vs in per.items():
            if not all(v in vs for v in variants) or k not in base:
                continue
            if group == "belief_rows" and "true_pi_true" in vs:
                continue
            src, j0 = base[k]
            t0 = te if src == "test" else et
            li0 = t0["legal_idx"][t0["legal_indptr"][j0]:t0["legal_indptr"][j0 + 1]]
            if any(sorted(vs[v][0].tolist()) != sorted(li0.tolist()) for v in variants):
                n_mismatch += 1
                continue
            S0 = t0["set_idx"][t0["set_indptr"][j0]:t0["set_indptr"][j0 + 1]]
            if len(S0) == 0:
                continue
            sc0 = (g33["pa"][j0].astype(np.float32) if src == "test" else pa[j0])[li0]
            recs["pi_false"].append(row_metrics(li0, S0, sc0) + (k[0],))
            for v in variants:
                li, S, sc, _ = vs[v]
                recs[v].append(row_metrics(li, S, sc) + (k[0],))
        out[group] = {"legal_set_mismatch_dropped": n_mismatch}
        for v, rs in recs.items():
            a = np.asarray(rs, dtype=np.float64)
            out[group][v] = {"top1": _ci(a[:, 0], a[:, 2]), "set_nll": _ci(a[:, 1], a[:, 2])}
        # paired differences against perfectInfo=false
        for v in variants:
            if v in recs and recs["pi_false"]:
                a0 = np.asarray(recs["pi_false"], np.float64)
                a1 = np.asarray(recs[v], np.float64)
                out[group][f"{v}_minus_pi_false"] = {"top1": _ci(a1[:, 0] - a0[:, 0], a0[:, 2]),
                                                     "set_nll": _ci(a1[:, 1] - a0[:, 1], a0[:, 2])}
    return out


def _replay_report(ids, mv_of, log) -> dict:
    out = {}
    at = load_attack(OUT_DIR / "h5" / "replay_attack_test.h5")
    y = at["y"].astype(np.float64)
    g = at["meta/row"]
    yes_rate_train = float(load_attack(OUT_DIR / "h5" / "replay_attack_train.h5")["y"].mean())
    preds = {"always_attack": np.ones(len(y)), "never_attack": np.zeros(len(y)),
             "power_ge_best_blocker_toughness": at["heur"].astype(np.float64)}
    probs = {"constant_train_rate": np.full(len(y), yes_rate_train)}
    if (CACHE / "gen33_attack_test.npz").exists():
        b = np.load(CACHE / "gen33_attack_test.npz")["bin"].astype(np.float64)
        probs["gen33_binary_head"] = 1 / (1 + np.exp(-(b[:, 1] - b[:, 0])))
    if (CACHE / "binary_head_attack_test.npz").exists():
        b = np.load(CACHE / "binary_head_attack_test.npz")["bin"].astype(np.float64)
        probs["human_binary_head_frozen_gen33_trunk"] = 1 / (1 + np.exp(-(b[:, 1] - b[:, 0])))
    att = {"n": int(len(y)), "games": int(len(np.unique(g))), "yes_rate": _ci(y, g), "yes_rate_train": yes_rate_train}
    correct = {}
    for k, p in preds.items():
        correct[k] = (p == y).astype(float)
        att[k] = {"accuracy": _ci(correct[k], g)}
    att["power_ge_best_blocker_toughness"]["auc_as_0_1_score"] = cluster_auc_ci(preds["power_ge_best_blocker_toughness"], y, g)
    for k, p in probs.items():
        pc = np.clip(p, 1e-4, 1 - 1e-4)
        ll = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
        pos = y > 0.5
        correct[k] = ((pc >= 0.5) == pos).astype(float)
        auc = cluster_auc_ci(pc, y, g) if 0 < pos.sum() < len(y) else None
        att[k] = {"accuracy": _ci(correct[k], g), "log_loss": _ci(ll, g),
                  "brier": _ci((pc - y) ** 2, g), "auc": auc, "mean_p_yes": float(pc.mean())}
    # paired accuracy differences on the same decisions (cluster bootstrap over games)
    pairs_ = (("human_binary_head_frozen_gen33_trunk", "power_ge_best_blocker_toughness"),
              ("human_binary_head_frozen_gen33_trunk", "gen33_binary_head"),
              ("power_ge_best_blocker_toughness", "gen33_binary_head"))
    att["paired_accuracy_differences"] = {f"{a} - {b}": _ci(correct[a] - correct[b], g)
                                          for a, b in pairs_ if a in correct and b in correct}
    out["attack_test"] = att
    rp = load_table(OUT_DIR / "h5" / "replay_priority_test.h5")
    kinds = np.asarray(rp.get("kind", ["?"] * len(rp["z"])))
    pri = {"n": int(len(rp["z"])), "kinds": dict(Counter(kinds.tolist()))}
    scs = {"uniform": (legal_scores(rp, kind="uniform"), True),
           "heuristic": (legal_scores(rp, kind="heuristic", mv_of=mv_of, lab_idx=rp["lab_idx"]), False)}
    for p in sorted(CACHE.glob("*_replay_priority_test.npz")):
        tag = p.stem[:-len("_replay_priority_test")]
        scs[tag] = (legal_scores(rp, np.load(p)["pa"].astype(np.float32), kind="model"), True)
    for name, (sc, prob) in scs.items():
        blk, pr = _policy_block(rp, sc, prob)
        for kd in ("exact", "imputed_order"):
            sel = np.flatnonzero(kinds == kd)
            blk[kd] = _policy_block(rp, sc, prob, sel)[0]
        pri[name] = blk
    out["priority_test"] = pri
    return out


def _fmt(c: dict | None, pct: bool = True, d: int = 1) -> str:
    if not c or c.get("mean") is None:
        return "n/a"
    f = (lambda x: f"{100 * x:.{d}f}") if pct else (lambda x: f"{x:.3f}")
    return f"{f(c['mean'])} [{f(c['lo'])}, {f(c['hi'])}]"


PROTOCOL_AB = """## Protocol: the search A/B on a pod (human priors on vs off)

Question: does a human-trained prior make MageZero's search play better at a fixed budget (E3), and
does the human-trained value help (E4)? Offline accuracy does not answer it: a prior that matches
humans can still steer a 96-simulation search away from lines the value head prefers.

1. **Networks.** N0 = gen33 as served in experiment #1. N1 = gen33 fine-tuned on the human train
   split (priority + value heads, the run in this report or a longer one on the pod), with a binary
   head trained on human attack decisions. Both are served by the unmodified mz-engine server.
2. **Encoding first.** N1 was trained on perfectInfo=false rows, N0 on perfectInfo=true. Fix the
   `hiddenInfo` config key (critique WP0) and give each network the encoding it was trained with;
   do not mix them inside one arm. If N1 has to play with perfectInfo=true, fine-tune it on
   belief-determinized perfectInfo=true re-encodings instead (the extras stage measures that gap).
3. **Arms** (same network weights within a comparison, only `priors` differ, so the effect is the
   prior, not the value head):
   - A0: N1, priors {binary} only (experiment #1's setting; control);
   - A1: N1, priors {priority, binary} (the human priority prior on);
   - A2: N1, priors {priority, target, binary};
   - B0 vs A0 (N0 vs N1, both {binary}): the value / trunk effect of the human fine-tune (E4).
   Temperature 1, the same PUCT constant, the same priorTemp (1.5) and priorBonus (0.1: search
   adds it to every non-Pass child, and it moves gen33's offline top-1 from 44% to 52%, see
   "gen33 as MageZero's search would use it"), budgets 96 (experiment #1) and 300.
4. **Opponent and decks.** Every arm plays the same fixed reference (N0 with {binary}, budget 96)
   on the same paired deck list: each deck pair twice with seats swapped, identical game seeds
   across arms (XMage seeding as in the bridge: RandomUtil and GameState.localRandom per game).
5. **Size.** >= 400 games per arm (200 deck pairs x 2 seats): the SE of one arm's win rate near 50%
   is 2.5 points, so the SE of an unpaired A1 - A0 difference is 3.5 points: about 7 points reaches
   significance at 95% and about 10 points has 80% power (pairing by deck pair and seed lowers
   both); 40 games (+-15 points) is too coarse (docs/003 3.6). Pre-register: primary = win rate vs
   the reference, A1 - A0, paired by deck pair (a paired bootstrap over deck pairs; McNemar on
   seat-swapped pairs as a check).
6. **Secondary.** Game length, spells cast per turn, Pass share at non-trivial priority decisions,
   attack rate, root visit entropy, seconds per decision (a prior costs no extra forward pass);
   E0's human-likeness statistics per arm; and the offline agreement of the searched move with the
   human test set at budget 300 with priors on vs off (coach op, remote evaluator: about 60 s per
   decision on CPU, so on the pod's GPU).
7. **Stop / go.** Adopt the human prior for experiment #2 self-play only if A1 >= A0 with the 95% CI
   of the difference above -2 points (it must not hurt) and it lowers the Pass-when-a-cast-was-legal
   rate; otherwise keep priors off and use the human rows only as a policy-loss warm start. With a
   3.5-point SE that CI bound needs an observed gain of about +5 points at 400 games per arm, so
   either pair the arms (same deck pairs and seeds) or plan ~1,000 games per arm if a smaller
   non-inferiority margin is wanted.
"""


def render_results(res: dict) -> str:
    """results.md: the tables (numbers from results.json)."""
    L = ["# Imitation experiment (E1 + offline E3/E4): results", "",
         f"Generated {res.get('date')} by `python -m draftzero.gameplay.imitation report`. 95% CIs are cluster "
         "bootstraps over games (decisions of one game are correlated); rates in %.", ""]
    ds = res["dataset"]
    b = ds["build"]
    L += ["## Dataset", "",
          f"- Main pass: every {b['every']}th row of the FDN Premier Draft replay file from row {b['start']}: "
          f"{b['games']} games, {b['ts']} turn-start requests ({b['ts_ok']} reached the labelled decision), "
          f"{b['rp']} replayed turns ({b['rp_ok']} reproduced), {b['seconds']:.0f} s with {b['workers']} bridge JVMs "
          f"({b['decisions_per_s']:.0f} turn-start decisions/s end to end, replays included).",
          f"- Splits (by event, mirrored-pair partners' events joined): {ds['decision_stats']['by_split']}", ""]
    s = ds["decision_stats"]
    a = s["all"]
    L += ["## Decision statistics (turn-start decisions reached)", "",
          f"- Request outcomes: {s['status']}",
          f"- >= 2 legal non-Pass options (non-trivial): {a['nontrivial_ge2_nonpass']['p']:.3f} "
          f"[{a['nontrivial_ge2_nonpass']['lo']:.3f}, {a['nontrivial_ge2_nonpass']['hi']:.3f}] of {a['n']}",
          f"- Pass label (no land/cast/activation all turn): {a['pass_label']['p']:.3f}; unreachable: "
          f"{a['unreachable']['p']:.3f} ({a['unreachable_of_acted']['p']:.3f} of the turns with an action)",
          f"- |S| for set labels: {a['S_size']} (mean {a['mean_S_size_set']})",
          f"- legal options (capped at 8): {s['legal_options']}", "",
          "| tier | n | non-trivial | Pass label | unreachable | mean legal | mean abs S |", "|---|---|---|---|---|---|---|"]
    for k, v in list(s["by_tier"].items()) + [("turn " + k, v) for k, v in s["by_turn"].items()]:
        if v["n"]:
            L.append(f"| {k} | {v['n']} | {v['nontrivial_ge2_nonpass']['p']:.3f} | {v['pass_label']['p']:.3f} | "
                     f"{v['unreachable']['p']:.3f} | {v['mean_legal_options']} | {v['mean_S_size_set']} |")
    L += ["", "## Priority policy on the test split (turn start, masked softmax)", "",
          f"Rows: {res['policy_test_n']}", "",
          "| model | top-1 in S | top-3 in S | set NLL | top-1 non-trivial | set NLL non-trivial | top-1 no land in S | set NLL no land in S |",
          "|---|---|---|---|---|---|---|---|"]
    for m, v in res["policy_test"].items():
        L.append(f"| {m} | {_fmt(v['top1'])} | {_fmt(v['top3'])} | {_fmt(v.get('set_nll'), False)} | "
                 f"{_fmt(v['nontrivial']['top1'])} | {_fmt(v['nontrivial'].get('set_nll'), False)} | "
                 f"{_fmt(v['no_land_in_S_cast_legal']['top1'])} | {_fmt(v['no_land_in_S_cast_legal'].get('set_nll'), False)} |")
    L += ["", "Label variant S + {Pass if the user attacked} (the coach's policy):", "",
          "| model | top-1 in S | set NLL |", "|---|---|---|"]
    for m, v in res["policy_test"].items():
        L.append(f"| {m} | {_fmt(v['S_plus_pass_if_attacked']['top1'])} | "
                 f"{_fmt(v['S_plus_pass_if_attacked'].get('set_nll'), False)} |")
    L += ["", "Rows with >= 4 unique legal options (with 2-3 options top-3 is automatic):", "",
          "| model | n | top-1 in S | top-3 in S | set NLL |", "|---|---|---|---|---|"]
    for m, v in res["policy_test"].items():
        g4 = v.get("ge4_legal_options")
        if g4:
            L.append(f"| {m} | {g4['top1']['n']} | {_fmt(g4['top1'])} | {_fmt(g4['top3'])} | {_fmt(g4.get('set_nll'), False)} |")
    sp = res.get("gen33_as_search_prior")
    if sp:
        L += ["", f"gen33 as MageZero's search would use it (softmax(logit / {sp['prior_temp']}) over the legal "
              f"children + {sp['prior_bonus']} for non-Pass children, renormalised; experiment #1 had this prior off): "
              f"top-1 {_fmt(sp['top1'])}, set NLL {_fmt(sp['set_nll'], False)} (n={sp['top1']['n']})."]
    pdt = res.get("paired_differences_test") or {}
    if pdt:
        L += ["", "Paired differences on the same test rows (cluster bootstrap over games; top-1 in points):", "",
              "| comparison | top-1 | set NLL |", "|---|---|---|"]
        L += [f"| {k} | {_fmt(v['top1'])} | {_fmt(v.get('set_nll'), False)} |" for k, v in pdt.items()]
    L += ["", "Human set categories on the test split: " +
          "; ".join(f"{k} {100 * v['share']:.1f}%" for k, v in res.get("S_category_shares", {}).items())]
    lc = res.get("learning_curve_test_subset")
    if lc:
        L += ["", "### Learning curve (fixed random test subset)", "",
              "Samples seen: the runs in this report counted steps x 32, but the token budget shrinks "
              "long-bag batches (about 29-31 rows per step on these data), so the true counts are 4-9% lower "
              "(train_full now counts rows).", "",
              "| model | top-1 in S | set NLL | training |", "|---|---|---|---|"]
        for m, v in lc.items():
            tr_ = v.get("training") or {}
            L.append(f"| {m} | {_fmt(v['top1'])} | {_fmt(v.get('set_nll'), False)} | "
                     + (f"{tr_.get('n_train_rows')} rows, {tr_.get('samples_seen')} samples seen, best val NLL "
                        f"{tr_.get('best_val_set_nll', float('nan')):.3f}" if tr_ else "") + " |")
    for title, key in (("Top-1 by tier", "top1_by_tier"), ("Top-1 by user turn", "top1_by_turn"),
                       ("Top-1 by the category of the human's set", "top1_by_set_category"),
                       ("Top-1 by player skill (17lands win-rate bucket, users with >= 100 games)", "top1_by_skill_ngames_ge100")):
        blk = res.get(key) or {}
        if not blk:
            continue
        cols = list(next(iter(blk.values())).keys())
        L += ["", f"### {title}", "", "| model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for m, v in blk.items():
            L.append(f"| {m} | " + " | ".join(f"{_fmt(v[c])} (n={v[c]['n']})" for c in cols) + " |")
        if key == "top1_by_skill_ngames_ge100":
            L += ["", "Win-rate buckets are rounded to 0.01 before banding (they are stored as float32). Strongest band "
                  "minus weakest, top-1 points: " + "; ".join(
                      f"{m} {_fmt(v)}" for m, v in (res.get("top1_skill_gradient_top_minus_bottom_band") or {}).items())]
    dis = res.get("gen33_disagreement") or {}
    if dis:
        L += ["", "## Where gen33 disagrees with humans", "", "| human set category | n | gen33 top-1 | gen33 mean p(S) |",
              "|---|---|---|---|"]
        for k, v in dis["top1_by_human_set_category"].items():
            L.append(f"| {k} | {v['n']} | {100 * v['top1']:.1f} | {v['mean_p_S']:.3f} |")
        L += ["", f"gen33's unmasked 1024-way softmax (the space its trainer fits; search renormalises over the "
              f"legal children) puts {_fmt(dis['unmasked_mass_on_legal'])}% of its mass on the legal "
              f"options (Pass alone: {_fmt(dis['unmasked_mass_on_pass'])}%).", ""]
        ppa = dis.get("pick_pass_by_set_category_and_attack") or {}
        if ppa:
            L += ["gen33 picks Pass (by the human's set category / whether the user attacked that turn; a "
                  "precombat Pass goes to combat): " + "; ".join(f"{k} {_fmt(v)} (n={v['n']})" for k, v in ppa.items()), ""]
        L += ["", "gen33's pick by the human's set category (top 20): " +
              "; ".join(f"{k}: {v}" for k, v in dis["gen33_pick_category_given_human_set_category"].items()), "",
              "gen33 probability of a human's cast by rarity: " +
              "; ".join(f"{k}: {v['mean_p']:.3f} (n={v['n']})" for k, v in dis["p_of_human_cast_by_rarity"].items()), "",
              "Most-missed human casts (gen33 top-1 not in S): " +
              ", ".join(f"{x['card']} ({x['rarity']}, {x['missed']} missed / {x['hit']} hit)" for x in dis["most_missed_casts"][:15])]
    L += ["", "## Value head on the test split (P(win) = (1 + v) / 2)", "",
          "| model | log-loss | Brier | AUC | accuracy | mean P | base rate |", "|---|---|---|---|---|---|---|"]
    for m, v in res["value_test"].items():
        x = v["all"]
        au = x["auc"]
        L.append(f"| {m} | {_fmt(x['log_loss'], False)} | {_fmt(x['brier'], False)} | "
                 f"{au['mean']:.3f} [{au['lo'] if au['lo'] is None else round(au['lo'], 3)}, "
                 f"{au['hi'] if au['hi'] is None else round(au['hi'], 3)}] | {_fmt(x['accuracy'])} | "
                 f"{x['mean_p']:.3f} | {x['base_rate']:.3f} |")
    L += ["", "AUC by user turn:", "", "| model | " + " | ".join(k for k in next(iter(res["value_test"].values())) if k != "all") + " |"]
    L.append("|---" * (len(next(iter(res["value_test"].values()))) ) + "|")
    for m, v in res["value_test"].items():
        L.append(f"| {m} | " + " | ".join(f"{v[k]['auc']['mean']:.3f} (n={v[k]['auc']['n']})" for k in v if k != "all") + " |")
    pi = res.get("gen33_perfect_info")
    if pi:
        L += ["", "## gen33 under its training encoding (perfectInfo=true), test subsets", ""]
        for grp, v in pi.items():
            L.append(f"**{grp}** (legal-set mismatches dropped: {v.get('legal_set_mismatch_dropped')})")
            L.append("")
            L.append("| encoding | top-1 in S | set NLL |")
            L.append("|---|---|---|")
            for k, x in v.items():
                if isinstance(x, dict) and "top1" in x:
                    L.append(f"| {k} | {_fmt(x['top1'])} | {_fmt(x['set_nll'], False)} |")
            L.append("")
    rp = res.get("replay") or {}
    if rp:
        at = rp["attack_test"]
        L += ["", "## Replayed turns: attack decisions (CHOOSE_USE, exact labels), test split", "",
              f"n = {at['n']} decisions in {at['games']} games; human attack rate {_fmt(at['yes_rate'])}", "",
              "| predictor | accuracy | log-loss | Brier | AUC |", "|---|---|---|---|---|"]
        for k, v in at.items():
            if isinstance(v, dict) and "accuracy" in v:
                au = v.get("auc") if isinstance(v.get("auc"), dict) else v.get("auc_as_0_1_score")
                L.append(f"| {k} | {_fmt(v['accuracy'])} | {_fmt(v.get('log_loss'), False)} | "
                         f"{_fmt(v.get('brier'), False)} | {_fmt(au, False) if isinstance(au, dict) else 'n/a'} |")
        pdA = at.get("paired_accuracy_differences") or {}
        if pdA:
            L += ["", "Paired accuracy differences on the same decisions (points):", "",
                  "| comparison | difference |", "|---|---|"]
            L += [f"| {k} | {_fmt(v)} |" for k, v in pdA.items()]
        L += ["", "AUC of the power/toughness rule is for its 0/1 output. Replayed-turn rows come only from "
              "reproduced turns (T0 97%, T3 78%), so they over-represent simple turns."]
        pr = rp["priority_test"]
        L += ["", "## Replayed turns: PRIORITY decisions with set labels, test split", "",
              f"n = {pr['n']} ({pr['kinds']})", "",
              "| model | top-1 in S | set NLL | top-1 exact rows | top-1 imputed_order rows |", "|---|---|---|---|---|"]
        for k, v in pr.items():
            if isinstance(v, dict) and "top1" in v:
                L.append(f"| {k} | {_fmt(v['top1'])} | {_fmt(v.get('set_nll'), False)} | "
                         f"{_fmt(v['exact']['top1'])} | {_fmt(v['imputed_order']['top1'])} |")
    interp = OUT_DIR / "interpretation.md"
    if interp.exists():         # the written reading of these numbers (kept apart so a re-run keeps it)
        L += ["", interp.read_text().rstrip()]
    L += ["", PROTOCOL_AB]
    tr = res.get("training") or {}
    if tr:
        L += ["", "## Training runs", "", "```", json.dumps(tr, indent=1, default=float)[:6000], "```"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m draftzero.gameplay.imitation", description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=("splits", "build", "tables", "extras", "gen33", "gen33_attack", "gen33_train",
                                      "gen33_extras", "scratch", "finetune", "heads", "replay_heads", "report"))
    ap.add_argument("--every", type=int, default=48)
    ap.add_argument("--start", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--n-train", type=int)
    ap.add_argument("--budget", type=float, default=900)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--runtime-root")
    ap.add_argument("--test-rows", type=int)
    a = ap.parse_args(argv)
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)   # noqa: E731
    rt = Path(a.runtime_root) if a.runtime_root else None
    if a.stage == "splits":
        row_splits(rebuild=True)
    elif a.stage == "build":
        log(build(every=a.every, start=a.start, workers=a.workers, limit=a.limit, runtime_root=rt, log=log))
    elif a.stage == "tables":
        stage_tables(log)
    elif a.stage == "extras":
        ts = load_shard(OUT_DIR / "shards" / "ts.pkl")
        log(build_extras(ts, runtime_root=rt, log=log))
    elif a.stage == "gen33":
        stage_gen33(log=log)
    elif a.stage == "gen33_attack":
        stage_gen33_attack(log)
    elif a.stage == "gen33_train":
        stage_gen33_train(a.n_train or 40000, log=log)
    elif a.stage == "gen33_extras":
        stage_gen33_extras(log)
    elif a.stage == "scratch":
        stage_scratch(a.n_train, a.budget, lr=a.lr or 3e-4, test_rows=a.test_rows, log=log)
    elif a.stage == "finetune":
        stage_finetune(a.budget, lr=a.lr or 1e-4, n_train=a.n_train, test_rows=a.test_rows, log=log)
    elif a.stage == "heads":
        log(stage_heads(log=log))
    elif a.stage == "replay_heads":
        log(stage_replay_heads(a.n_train or 20000, log=log))
    elif a.stage == "report":
        report(log=log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
