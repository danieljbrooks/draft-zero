"""Build docs/012's frozen decision set, sb-v1: held-out decisions by top 17lands players.

    python tools/search_bench/items.py build [--out data/search_bench/sb-v1] [--every 36] [--workers 4]
    python tools/search_bench/items.py stats [--out ...]

Top players: user_game_win_rate_bucket >= 0.60 and at least 100 games (docs/008's "top"). Held
out: no row of any imitation table (rows 5, 53, 101, ...: docs/008 §7.3 and experiment #2b used
the same 16,483 games) nor its mirrored partner. At most two decisions per game (a mirrored pair
counts as one game), turns 3-12, fidelity tiers T0/T1 only.

Four decision types, which the gameplay pipeline can rebuild today (docs/012 §2.2's mid-turn and
target types need replay states and are left for later):

  spell   the first main-phase priority after the human's land drop, in a turn in which the human
          cast a spell or activated an ability. Label: the turn's casts and activations that are
          legal here, plus Pass when the human attacked (the coach's convention: those plays may
          have come after combat). label_strict drops that Pass.
  hold    the same decision in a turn in which the human cast nothing and activated nothing,
          with a spell castable. Label: Pass.
  attack  the first "attack with X?" question, in a turn in which the human cast nothing (so the
          pre-combat state is the turn start plus its land). Exact label. Kept when the defender
          has an untapped creature or an untapped land.
  block   the first "what does X block?" question in the opponent's next turn. Exact label
          (a unique pairing).

Each item carries its worlds (docs/012 §2.4): `real`, the position with the opponent's hand and
deck filled from the belief model with the item's own seed (what clairvoyant MCTS searches), and
`worlds`, 8 more belief samples with the search seed (PIMC with 1 world uses the first, PIMC with
4 the first four, IS-MCTS all eight). The belief model leaves out the game's drafts (and a
mirrored partner's). Draft ids are never written.

Output (git-ignored; mirrored-pair rows are in it): <out>/items.jsonl.gz and <out>/build.json.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from draftzero.gameplay import coach as co  # noqa: E402
from draftzero.gameplay import fingerprints as fp  # noqa: E402
from draftzero.gameplay import pairs as pm  # noqa: E402
from draftzero.gameplay import reconstruct as rc  # noqa: E402
from draftzero.gameplay import replay  # noqa: E402
from draftzero.gameplay.bridge import BridgePool  # noqa: E402
from draftzero.gameplay.ids import Ids  # noqa: E402

VERSION = "sb-v1"
QUOTA = {"test": {"spell": 375, "hold": 125, "attack": 300, "block": 200},
         "dev": {"spell": 110, "hold": 40, "attack": 90, "block": 60}}
TURNS = range(3, 13)
DEV_SHARE = 0.24
N_ROWS = 791_159
IMITATION = set(range(5, N_ROWS, 48))
REAL_SEED = 900_000     # + row: the item's own belief sample and build seed
SEARCH_SEED = 7         # the search's belief worlds (nested: seed 7's first k samples)
N_WORLDS = 8
SEED = 20260928


def held_out_rows(pmap: dict) -> set[int]:
    return IMITATION | {pmap[r] for r in IMITATION if r in pmap}


def is_top(g) -> bool:
    return "top" in fp.skill_groups(g.meta)


class Rejected(Exception):
    pass


def _legal(d: dict) -> list[str]:
    return [o["label"] for o in d.get("legal") or []]


def defender_open(dump: dict, seat: str) -> bool:
    """The defending seat has an untapped creature or an untapped land (a real attack question)."""
    for p in ((dump or {}).get("players") or {}).get(seat, {}).get("battlefield") or []:
        x = p.get("x") or {}
        if x.get("canBlock") or (x.get("power") is not None and not p.get("tapped")):
            return True
        name = x.get("name") or p.get("name") or ""
        if not p.get("tapped") and name in ("Plains", "Island", "Swamp", "Mountain", "Forest"):
            return True
    return False


def make_item(g, partner, n: int, kind: str, bridge, ids) -> dict:
    """One decision of `kind` in user turn n of game g, or raise Rejected(reason)."""
    block = kind == "block"
    try:
        spec = (rc.state_after_user_turn(g, n, "declare_attackers", ids=ids, labels=True) if block
                else rc.state_at_user_turn(g, n, "eot_rollover", ids=ids, labels=True))
    except ValueError as e:  # e.g. the opponent did not attack after user turn n
        raise Rejected("spec: " + str(e).split(": ", 1)[-1][:60])
    fid, low = co.fidelity(spec)
    if low:
        raise Rejected("fidelity")
    labels = spec.labels or {}
    acts = [x for x in (labels.get("casts") or []) + (labels.get("activations") or []) if x.get("key")]
    lands = [x for x in labels.get("lands") or [] if x.get("key")]
    attacked = any((labels.get("attacks") or {}).values())
    if kind == "spell" and not acts:
        raise Rejected("no cast")
    if kind in ("hold", "attack") and acts:
        raise Rejected("cast")
    if kind == "attack" and not (labels.get("attacks") or {}):
        raise Rejected("no attack question recorded")
    slot = g.user_slot(n) if block else g.prev_slot(n)
    if slot is None:
        raise Rejected("no slot")
    seen = Counter(rc.analyze(g, ids).states[slot.seq].revealed)
    exclude = rc.holdout_drafts(g, partner)
    real, info = co.plan_determinization(spec, k=1, seed=REAL_SEED + g.row_index, exclude_drafts=exclude, seen=seen)
    if real is None:
        raise Rejected("belief: " + str(info.get("fallback") or info.get("method")))
    worlds, _ = co.plan_determinization(spec, k=N_WORLDS, seed=SEARCH_SEED, exclude_drafts=exclude, seen=seen)
    opts = dict(labels.get("bridge") or {})
    if kind == "attack":
        opts["decideFrom"] = {"turn": opts.get("decideFrom", {}).get("turn", n), "step": "DECLARE_ATTACKERS"}
    if kind != "block" and lands:
        opts["preLand"] = lands[0]["key"]
    real_d = real[0].to_dict()
    try:
        b = bridge.request("build", real_d, seed=REAL_SEED + g.row_index, dumpDecisionState=True, **opts)
    except Exception as e:  # noqa: BLE001
        raise Rejected("build: " + str(e).splitlines()[0][:120])
    d = b.get("decision")
    if not d:
        raise Rejected("no decision: " + str(b.get("noDecision"))[:80])
    where = d.get("where") or {}
    t, text, step = d.get("type"), d.get("text") or "", where.get("step")
    df = opts.get("decideFrom") or {}
    if where.get("turn") != df.get("turn"):
        raise Rejected("wrong turn")
    if kind in ("spell", "hold") and not (t == "PRIORITY" and step == "PRECOMBAT_MAIN"):
        raise Rejected(f"reached {t} at {step}")
    if kind == "attack" and not (t == "CHOOSE_USE" and text.startswith("attack with: ")):
        raise Rejected(f"reached {t} at {step}")
    if kind == "block" and not (t == "CHOOSE_TARGET" and step == "DECLARE_BLOCKERS"):
        raise Rejected(f"reached {t} at {step}")
    legal = _legal(d)
    if len(set(legal)) < 2:
        raise Rejected("trivial")
    decision = {"type": t, "text": text, "turn": where.get("turn"), "step": step}
    names = co.alias_names(b.get("dump"))
    ha = co._as_human(co.human_17lands(labels), decision, names)
    if ha is None or not ha.labels:
        raise Rejected("label: " + str(getattr(ha, "reason", None))[:80])
    members = [m for m in (co.match_label(a, legal, t) for a in ha.labels) if m]
    members = list(dict.fromkeys(members))
    strict = list(members)
    if kind == "spell":
        strict = [m for m in members if m != "Pass"]
        if not strict:
            raise Rejected("no human cast legal here")
        if attacked and "Pass" in legal and "Pass" not in members:
            members.append("Pass")
    elif kind == "hold":
        if members != ["Pass"]:
            raise Rejected("hold label")
        if not any(lab.startswith("Cast ") for lab in legal):
            raise Rejected("nothing castable")
    elif kind == "attack":
        if not defender_open(b.get("dump"), "B"):
            raise Rejected("defender closed")
    if not members:
        raise Rejected("human unmatched")
    return {
        "type": kind, "row": g.row_index, "turn": n, "on_play": g.on_play,
        "win_rate_bucket": g.meta.get("user_game_win_rate_bucket"), "rank": g.meta.get("rank"),
        "mirrored": partner is not None, "tier": fid["tier"],
        "decision": decision, "legal": legal, "n_options": len(set(legal)),
        "label": members, "label_strict": strict, "label_note": ha.note,
        "request": {k: v for k, v in opts.items() if k in ("decisionPlayer", "decideFrom", "preLand")},
        "build_seed": REAL_SEED + g.row_index,
        "real": real_d, "worlds": [w.to_dict() for w in worlds or []],
        "state": b.get("decisionState"),
        "warnings": (b.get("warnings") or [])[:5],
    }


def cmd_build(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pmap = pm.partner_map(pm.load_pairs())
    held = held_out_rows(pmap)
    ids = Ids.load("FDN")
    rows_seen = 0

    def keep(g) -> bool:
        return is_top(g) and g.row_index not in held and pmap.get(g.row_index) not in held

    games = list(replay.iter_games(every=a.every, start=a.start, predicate=keep))
    print(f"{len(games)} top, held-out games from every {a.every}th row ({time.time() - t0:.0f}s)", flush=True)
    rng = random.Random(SEED)
    rng.shuffle(games)
    # a mirrored pair is one game: keep the first of its two rows
    used, uniq = set(), []
    for g in games:
        if g.row_index in used:
            continue
        used.add(g.row_index)
        if g.row_index in pmap:
            used.add(pmap[g.row_index])
        uniq.append(g)
    partners = replay.read_games(sorted({pmap[g.row_index] for g in uniq if g.row_index in pmap}))
    print(f"{len(uniq)} games, {len(partners)} mirrored partners read ({time.time() - t0:.0f}s)", flush=True)

    split_of = {g.row_index: ("dev" if rng.random() < DEV_SHARE else "test") for g in uniq}
    counts = {s: Counter() for s in QUOTA}
    reasons = Counter()
    items = []
    lock = threading.Lock()
    pool = BridgePool(a.workers, prefix="sb_items")

    def need(split, kind) -> bool:
        return counts[split][kind] < QUOTA[split][kind]

    def done() -> bool:
        return all(not need(s, k) for s in QUOTA for k in QUOTA[s])

    def work(g):
        split = split_of[g.row_index]
        partner = partners.get(pmap.get(g.row_index)) if g.row_index in pmap else None
        turns = [n for n in g.decision_turns() if n in TURNS]
        grng = random.Random(SEED ^ g.row_index)
        cands = [(n, k) for n in turns for k in ("spell", "hold", "attack", "block")]
        grng.shuffle(cands)
        # the rarer types first, so a game's two slots go to them when it has one
        rank = {"attack": 0, "hold": 1, "block": 2, "spell": 3}
        cands.sort(key=lambda c: rank[c[1]])
        got, used_turns = 0, set()
        for n, kind in cands:
            if got >= 2 or done():
                return
            with lock:
                if not need(split, kind) or (n, kind == "block") in used_turns:
                    continue
            try:
                it = make_item(g, partner, n, kind, pool, ids)
            except Rejected as e:
                with lock:
                    reasons[f"{kind}: {str(e)[:60]}"] += 1
                continue
            except Exception as e:  # noqa: BLE001 - one bad game must not stop the build
                with lock:
                    reasons[f"{kind}: error {type(e).__name__}"] += 1
                continue
            with lock:
                if not need(split, kind):
                    continue
                counts[split][kind] += 1
                it["split"] = split
                items.append(it)
                got += 1
                used_turns.add((n, kind == "block"))
                if len(items) % 50 == 0:
                    print(f"  {len(items)} items  test {dict(counts['test'])}  dev {dict(counts['dev'])}"
                          f"  ({time.time() - t0:.0f}s)", flush=True)

    try:
        with ThreadPoolExecutor(a.workers) as ex:
            for _ in ex.map(work, uniq):
                if done():
                    break
    finally:
        pool.close()
    # stable ids: by split, type, row, turn
    items.sort(key=lambda x: (x["split"], x["type"], x["row"], x["turn"]))
    for i, it in enumerate(items):
        it["id"] = f"{VERSION}-{it['split']}-{i:04d}"
    with gzip.open(out / "items.jsonl.gz", "wt") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    meta = {"version": VERSION, "counts": {s: dict(c) for s, c in counts.items()}, "quota": QUOTA,
            "games_scanned": len(uniq), "every": a.every, "start": a.start, "seed": SEED,
            "real_seed": REAL_SEED, "search_seed": SEARCH_SEED, "n_worlds": N_WORLDS,
            "rejections": dict(reasons.most_common()), "seconds": round(time.time() - t0)}
    (out / "build.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("counts", "games_scanned", "seconds")}), flush=True)
    return 0


def load_items(path) -> list[dict]:
    with gzip.open(Path(path) / "items.jsonl.gz", "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_stats(a) -> int:
    items = load_items(a.out)
    c = Counter((it["split"], it["type"]) for it in items)
    print(dict(c))
    by = defaultdict(list)
    for it in items:
        by[it["type"]].append(it["n_options"])
    for k, v in by.items():
        print(k, "options: mean", round(sum(v) / len(v), 2), "chance", round(sum(1 / x for x in v) / len(v), 3))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default=str(REPO / "data/search_bench" / VERSION))
    b.add_argument("--every", type=int, default=36)
    b.add_argument("--start", type=int, default=17)
    b.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("stats")
    s.add_argument("--out", default=str(REPO / "data/search_bench" / VERSION))
    a = ap.parse_args(argv)
    return {"build": cmd_build, "stats": cmd_stats}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
