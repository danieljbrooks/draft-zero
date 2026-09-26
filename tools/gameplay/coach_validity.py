#!/usr/bin/env python
"""
coach_validity.py — experiment E5 (critique.md §5; §4 questions 8 and 9): is the MCTS coach better
than the student, and how much do its verdicts depend on hidden information and on the evaluator?

Parts (each a `run` setting; results under data/gameplay/coach_validity/, gitignored):

  1. skill       Network coach (gen33 through the mz-engine inference server, `--port`) and offline
                 coach on the SAME 17lands decisions: users with >= 100 games (the win-rate bucket
                 includes the game itself, which dominates it for users with fewer; critique N3),
                 stratified over coach.WR_BANDS (round-robin order, so a run cut short by its time
                 cap stays balanced), one user turn 3-8 per game, tier T0/T1 and not low fidelity, a
                 PRIORITY decision at the turn's first main phase with >= 2 legal non-Pass options and
                 the human's answer (the 17lands turn's actions, set-valued) matched. Mirrored-pair
                 rows are left to part 2. Metrics by skill band: best-of-set regret (win-probability
                 units), agreement (the engine's best option is in the human's set) against the chance
                 rate |set| / |legal|, and Spearman's rho of the win-rate bucket against the loss.
  2. hindsight   Mirrored-pair rows (both drafts held out of the belief pool): coached with belief
                 determinization (what a coach could know) and with exact_spec (the opponent's true
                 deck and hand); same best option, Spearman of per-option mean Q, human's regret.
  3. stability   K=16 offline searches of one decision (belief determinizations, seed 0, so the
                 first 8 are the K=8 run's); subsets of K=2/4/8 resampled: how often their best
                 option is K=16's, plus split-half (two disjoint K=8) agreement.
  4. cost        seconds per decision per setting (from the verdict rows).

Settings (`run --setting`):
  net_bin          remote gen33, priors binary only (as exp #1), K=8 x 300, skill decisions
  net_pri          remote gen33, priors priority + binary, K=8 x 300, skill decisions
  off              offline heuristic, K=8 x 300 (combat 1,000; thin faults re-searched at 3,000)
  pair_belief_off  / pair_exact_off   offline, K=8 x 300, pair decisions
  pair_belief_net  / pair_exact_net   remote gen33 binary, K=8 x 300, pair decisions
  stab16           offline K=16 x 300, raw per-determinization Q, skill decisions

Usage:
  python tools/gameplay/coach_validity.py sample [--every 173 --per-band 110 --pairs 120]
  python tools/gameplay/coach_validity.py run --setting net_bin --workers 2 --port 50093 --minutes 85
  python tools/gameplay/coach_validity.py run --setting off --workers 2 --limit 400
  python tools/gameplay/coach_validity.py analyze
  python tools/gameplay/coach_validity.py clean        # delete the cached games (they hold draft ids)

Privacy: row indices, turns and skill fields only are written, and only under data/gameplay/.
Draft ids live in memory and in the game cache (data/gameplay/coach_validity/cache/, deleted by
`clean`); partner rows of mirrored pairs are never written.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from draftzero.gameplay import bridge as br, coach, reconstruct as rc, replay  # noqa: E402
from draftzero.gameplay import pairs as pm  # noqa: E402
from draftzero.gameplay.ids import Ids  # noqa: E402

OUT = REPO / "data" / "gameplay" / "coach_validity"
CACHE = OUT / "cache"
SAMPLE = OUT / "sample.json"
TURNS = range(3, 9)
N_ROWS = 791_159
DEFAULT_RUNTIME = Path(os.environ.get("CV_RUNTIME", str(br.RUNTIME_ROOT)))   # bridge worker dirs

SETTINGS = {
    "net_bin": dict(part="skill", k=8, budget=300, evaluator="remote", priors=["binary"]),
    "net_pri": dict(part="skill", k=8, budget=300, evaluator="remote", priors=["priority", "binary"]),
    "off": dict(part="skill", k=8, budget=300, evaluator="offline"),
    "pair_belief_off": dict(part="pair", k=8, budget=300, evaluator="offline", hindsight=False),
    "pair_exact_off": dict(part="pair", k=8, budget=300, evaluator="offline", hindsight=True),
    "pair_belief_net": dict(part="pair", k=8, budget=300, evaluator="remote", priors=["binary"], hindsight=False),
    "pair_exact_net": dict(part="pair", k=8, budget=300, evaluator="remote", priors=["binary"], hindsight=True),
    "stab16": dict(part="skill", k=16, budget=300, evaluator="offline", raw=True),
}


def band(w) -> str:
    return next((f"{a:.2f}-{b:.2f}" for a, b in coach.WR_BANDS if w is not None and a <= w < b), "none")


BANDS = [band(a) for a, _ in coach.WR_BANDS]


# ------------------------------------------------------------------------------------------------
# sampling
# ------------------------------------------------------------------------------------------------

def prescreen(spec, b) -> tuple[dict | None, str]:
    """Build the spec once (as coach_decision does) and check that the decision is the labelled
    PRIORITY decision with >= 2 legal non-Pass options and a matched human answer."""
    tier_ok = spec.provenance.tier in ("T0", "T1") and not coach.fidelity(spec)[1]
    if not tier_ok:
        return None, "tier"
    opts = spec.labels.get("bridge") or br.turn_start_options(spec, "A")
    try:
        r = b.request("build", spec.to_dict(), seed=0, **opts)
    except br.BridgeError:
        return None, "build_error"
    d = r.get("decision")
    if not d:
        return None, "no_decision"
    where = d.get("where") or {}
    dec = {"player": d.get("player"), "type": d.get("type"), "text": d.get("text"), "turn": where.get("turn"),
           "phase": where.get("phase"), "step": where.get("step"), "passedBefore": where.get("passedBefore")}
    legal = [o["label"] for o in d.get("legal") or []]
    if dec["type"] != "PRIORITY":
        return None, "not_priority"
    if coach.expect_17lands(spec)(dec):
        return None, "decision_mismatch"
    nonpass = [x for x in legal if x != "Pass"]
    if len(nonpass) < 2:
        return None, "trivial"
    ha = coach.human_17lands(spec.labels)(dec, {})
    members = [m for m in (coach.match_label(a, legal, dec["type"]) for a in ha.labels) if m]
    members = list(dict.fromkeys(members))
    if not members:
        return None, "human_unmatched"
    return {"n_legal": len(legal), "n_nonpass": len(nonpass), "n_members": len(members),
            "set_covers_all": set(members) >= set(legal), "tier": spec.provenance.tier,
            "build_warnings": len(r.get("warnings") or [])}, "ok"


def cmd_sample(a) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    partners = pm.partner_map(pm.load_pairs())
    stride = list(range(a.start, N_ROWS, a.every))
    want = set(stride) | {partners[r] for r in stride if r in partners}
    print(f"reading {len(want)} rows ({len(stride)} stride rows, every {a.every})", flush=True)
    games = replay.read_games(sorted(want))
    print(f"read {len(games)} games in {time.monotonic() - t0:.0f} s", flush=True)
    ids = Ids.load("FDN")
    rng = random.Random(a.seed)
    stats: dict = {"stride_rows": len(stride), "reasons": Counter(), "few_games": 0, "pair_rows": 0}
    by_band: dict[str, list] = defaultdict(list)
    pair_rows = []
    for r in stride:
        g = games.get(r)
        if g is None:
            continue
        if r in partners:
            stats["pair_rows"] += 1
            if partners[r] in games:
                pair_rows.append(r)
            continue
        if (g.meta.get("user_n_games_bucket") or 0) < a.min_games:
            stats["few_games"] += 1
            continue
        by_band[band(g.meta.get("user_game_win_rate_bucket"))].append(r)
    stats["candidates_by_band"] = {k: len(v) for k, v in sorted(by_band.items())}
    print("candidates by band", stats["candidates_by_band"], "pair rows", len(pair_rows), flush=True)

    def pick(g, partner=None) -> dict | None:
        turns = [n for n in g.decision_turns() if n in TURNS]
        rng.shuffle(turns)
        for n in turns:
            try:
                spec = rc.state_at_user_turn(g, n, "eot_rollover", ids=ids, labels=True)
            except Exception:                                          # noqa: BLE001
                stats["reasons"]["spec_error"] += 1
                continue
            info, why = prescreen(spec, b)
            stats["reasons"][why] += 1
            if info is None:
                continue
            if partner is not None:
                try:
                    ex = rc.exact_spec(g, partner, n, "eot_rollover", ids=ids, labels=True)
                except Exception:                                      # noqa: BLE001
                    stats["reasons"]["exact_error"] += 1
                    continue
                if ex.is_partial() or ex.players["B"].decklistSource != "exact":
                    stats["reasons"]["exact_partial"] += 1
                    continue
                exinfo, why = prescreen(ex, b)
                stats["reasons"]["exact_" + why] += 1
                if exinfo is None:
                    continue
                info["exact_tier"] = exinfo["tier"]
                info["exact_n_legal"] = exinfo["n_legal"]
            m = g.meta
            return {"row": g.row_index, "turn": n, "band": band(m.get("user_game_win_rate_bucket")),
                    "win_rate_bucket": m.get("user_game_win_rate_bucket"), "n_games_bucket": m.get("user_n_games_bucket"),
                    "rank": m.get("rank"), "won": bool(m.get("won")), "on_play": bool(m.get("on_play")), **info}
        return None

    skill: dict[str, list] = {}
    pairs_out = []
    with br.Bridge("cvsample", heap="2500m", auto_build=False, runtime_root=Path(a.runtime)) as b:
        for bd in BANDS:
            rows = by_band.get(bd, [])
            rng.shuffle(rows)
            got = []
            for r in rows:
                if len(got) >= a.per_band:
                    break
                d = pick(games[r])
                if d:
                    got.append(d)
            skill[bd] = got
            print(f"band {bd}: {len(got)} decisions from {len(rows)} candidate games", flush=True)
        rng.shuffle(pair_rows)
        seen_pairs = set()
        for r in pair_rows:
            if len(pairs_out) >= a.pairs:
                break
            key = frozenset((r, partners[r]))
            if key in seen_pairs:                     # both halves of one game in the stride: keep one
                continue
            seen_pairs.add(key)
            d = pick(games[r], games[partners[r]])
            if d:
                pairs_out.append(d)
        print(f"pairs: {len(pairs_out)} decisions", flush=True)
    order = []
    for i in range(max(len(v) for v in skill.values())):
        for bd in BANDS:
            if i < len(skill[bd]):
                order.append(skill[bd][i])
    for i, d in enumerate(order):
        d["id"] = f"s{i:04d}"
    for i, d in enumerate(pairs_out):
        d["id"] = f"p{i:04d}"
    stats["reasons"] = dict(stats["reasons"])
    stats["seconds"] = round(time.monotonic() - t0, 1)
    SAMPLE.write_text(json.dumps({"settings": vars(a), "stats": stats, "skill": order, "pairs": pairs_out}, indent=1))
    keep = {d["row"] for d in order} | {d["row"] for d in pairs_out} | {partners[d["row"]] for d in pairs_out}
    with open(CACHE / "games.pkl", "wb") as f:
        pickle.dump({r: games[r] for r in keep}, f)
    print(f"wrote {SAMPLE} ({len(order)} skill, {len(pairs_out)} pair decisions) in {stats['seconds']} s")
    return 0


# ------------------------------------------------------------------------------------------------
# runs
# ------------------------------------------------------------------------------------------------

_LOCK = threading.Lock()


def _locked_opponent_model():
    """coach.opponent_model evicts from a module dict: serialize it for worker threads."""
    orig = coach.opponent_model
    lock = threading.Lock()

    def f(*args, **kw):
        with lock:
            return orig(*args, **kw)
    coach.opponent_model = f


def verdict_row(d: dict, v, setting: str) -> dict:
    h = v.human or {}
    matched = list(h.get("matched") or [])
    legal = (v.decision or {}).get("legal") or []
    opts = [{"label": o.label, "q": o.mean_q, "sd": o.sd_q, "n": o.n, "share": o.visit_share, "rank": o.rank,
             "argmax": o.argmax_share} for o in v.options]
    return {"id": d["id"], "setting": setting, "row": d["row"], "turn": d["turn"], "band": d["band"],
            "win_rate_bucket": d["win_rate_bucket"], "n_games_bucket": d["n_games_bucket"], "rank": d["rank"],
            "won": d["won"], "classification": v.classification, "reason": v.reason,
            "wp_loss": h.get("wp_loss"), "wp_loss_mean_of_set": h.get("wp_loss_mean_of_set"),
            "regret": h.get("regret"), "q_human": h.get("q_human"), "q_best": h.get("q_best"),
            "visits_per_det": h.get("visits_per_det"), "matched": matched, "n_legal": len(legal),
            "legal": legal, "best": v.best, "agree": (v.best in matched) if v.best and matched else None,
            "options": opts, "flags": v.flags, "caution": v.caution, "tier": (v.fidelity or {}).get("tier"),
            "det_method": (v.determinization or {}).get("method"), "fallback": (v.determinization or {}).get("fallback"),
            "held_out_drafts": (v.determinization or {}).get("held_out_drafts"),
            "first_pass": (v.noise or {}).get("first_pass"), "near_tie": (v.noise or {}).get("near_tie"),
            "se_paired": (v.noise or {}).get("se_paired"), "budget": (v.settings or {}).get("budget"),
            "sims_per_s": (v.value or {}).get("sims_per_s"), "root_value": (v.value or {}).get("root_value_mean"),
            "seconds": v.seconds, "warnings": len(v.warnings or [])}


def raw_row(d: dict, spec, b, k: int, budget: int, model_exclude, seen) -> dict:
    """Part 3: K belief determinizations searched offline; per-determinization Q kept."""
    t0 = time.monotonic()
    specs, info = coach.plan_determinization(spec, k=k, seed=0, exclude_drafts=model_exclude, seen=seen)
    opts = spec.labels.get("bridge") or br.turn_start_options(spec, "A")
    common = dict(seed=0, evaluator={"type": "offline"}, budget=budget, **opts)
    if specs is not None:
        r = b.request("coach", None, specs=[s.to_dict() for s in specs], **common)
    else:
        r = b.request("coach", spec.to_dict(), determinizations=k, resample=info.get("resample"), **common)
    dec = r.get("decision") or {}
    legal = [o["label"] for o in dec.get("legal") or []]
    ha = coach.human_17lands(spec.labels)({"type": dec.get("type")}, {})
    matched = list(dict.fromkeys(m for m in (coach.match_label(x, legal, dec.get("type")) for x in ha.labels) if m))
    per_det = []
    for det in r.get("determinizations") or []:
        acc: dict = {}
        for c in det.get("children") or []:
            if c.get("Q") is None or not c.get("N"):
                continue
            s = acc.setdefault(c["label"], [0.0, 0])
            s[0] += c["N"] * c["Q"]
            s[1] += c["N"]
        per_det.append({lab: [s / n, n] for lab, (s, n) in acc.items()})
    return {"id": d["id"], "setting": "stab16", "row": d["row"], "turn": d["turn"], "band": d["band"],
            "legal": legal, "matched": matched, "best": r.get("best"), "consistent": r.get("consistent"),
            "det_method": info.get("method"), "per_det": per_det, "seconds": round(time.monotonic() - t0, 2)}


def cmd_run(a) -> int:
    cfg = dict(SETTINGS[a.setting])
    sample = json.loads(SAMPLE.read_text())
    todo = sample["skill"] if cfg["part"] == "skill" else sample["pairs"]
    if a.ids:
        want = set(a.ids.split(","))
        todo = [d for d in todo if d["id"] in want]
    if a.offset:
        todo = todo[a.offset:]
    if a.limit:
        todo = todo[:a.limit]
    out = OUT / f"verdicts_{a.setting}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text().splitlines() if line.strip()}
    todo = [d for d in todo if d["id"] not in done]
    print(f"{a.setting}: {len(todo)} decisions to coach ({len(done)} done already)", flush=True)
    if not todo:
        return 0
    with open(CACHE / "games.pkl", "rb") as f:
        games = pickle.load(f)
    partners = pm.partner_map(pm.load_pairs())
    ids = Ids.load("FDN")
    coach.opponent_model()                                     # warm the caches before threads start
    coach._pool_cards()
    _locked_opponent_model()
    ports = [int(x) for x in str(a.port).split(",") if x.strip()]
    deadline = time.monotonic() + a.minutes * 60 if a.minutes else None
    q = list(reversed(todo))
    qlock = threading.Lock()
    t_start = time.monotonic()
    count = Counter()

    def worker(i: int) -> None:
        port = ports[i % len(ports)]          # one inference server per worker, e.g. MPS and CPU
        ev = coach.remote(port) if cfg["evaluator"] == "remote" else "offline"
        with br.Bridge(f"cv{a.setting[:8]}{i}", heap=a.heap, auto_build=False, runtime_root=Path(a.runtime)) as b:
            while True:
                if deadline and time.monotonic() > deadline:
                    return
                with qlock:
                    if not q:
                        return
                    d = q.pop()
                coach._revive(b)
                g = games[d["row"]]
                partner = games.get(partners.get(d["row"])) if cfg["part"] == "pair" else None
                try:
                    if cfg.get("raw"):
                        spec = rc.state_at_user_turn(g, d["turn"], "eot_rollover", ids=ids, labels=True)
                        slot = g.prev_slot(d["turn"])
                        seen = Counter(rc.analyze(g, ids).states[slot.seq].revealed) if slot is not None else Counter()
                        rec = raw_row(d, spec, b, cfg["k"], cfg["budget"], rc.holdout_drafts(g, partner), seen)
                    else:
                        kw = dict(k=cfg["k"], budget=cfg["budget"], seed=0, evaluator=ev,
                                  priors=cfg.get("priors"), ids=ids)
                        if cfg["part"] == "pair":
                            v = coach.coach_17lands(d["row"], d["turn"], bridge=b, game=g, partner=partner,
                                                    hindsight=cfg["hindsight"], **kw)
                        else:
                            v = coach.coach_17lands(d["row"], d["turn"], bridge=b, game=g, pairs=partners, **kw)
                        rec = verdict_row(d, v, a.setting)
                        if cfg["evaluator"] == "remote":
                            rec["server_port"] = port
                except Exception as e:                                  # noqa: BLE001
                    rec = {"id": d["id"], "setting": a.setting, "row": d["row"], "turn": d["turn"],
                           "classification": "error", "reason": f"{type(e).__name__}: {str(e)[:200]}"}
                with _LOCK:
                    with open(out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    count[rec.get("classification", "raw")] += 1
                    n = sum(count.values())
                    el = time.monotonic() - t_start
                    print(f"[{a.setting}] {n}/{len(todo)} {d['id']} {rec.get('classification', '')} "
                          f"{rec.get('seconds')} s  sims/s={rec.get('sims_per_s')}  elapsed {el / 60:.1f} min",
                          flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(worker, range(a.workers)))
    print(f"{a.setting}: {dict(count)} in {(time.monotonic() - t_start) / 60:.1f} min", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------
# analysis
# ------------------------------------------------------------------------------------------------

def load(setting: str) -> list[dict]:
    p = OUT / f"verdicts_{setting}.jsonl"
    if not p.exists():
        return []
    rows = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r                   # a re-run of an id replaces the earlier row
    return list(rows.values())


def wilson(k: int, n: int, z: float = 1.96) -> list | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def boot_mean(xs, n_boot: int = 2000, seed: int = 0) -> list | None:
    xs = np.asarray([x for x in xs if x is not None], float)
    if len(xs) < 2:
        return None
    rng = np.random.default_rng(seed)
    m = xs[rng.integers(0, len(xs), (n_boot, len(xs)))].mean(1)
    return [round(float(np.quantile(m, 0.025)), 5), round(float(np.quantile(m, 0.975)), 5)]


def rank(v):
    from scipy.stats import rankdata
    return rankdata(v)


def spearman(x, y, min_n: int = 4) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < min_n or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def boot_spearman(x, y, n_boot: int = 2000, seed: int = 0) -> list | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 10:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        s = spearman(x[i], y[i])
        if s is not None:
            vals.append(s)
    return [round(float(np.quantile(vals, 0.025)), 4), round(float(np.quantile(vals, 0.975)), 4)] if vals else None


GRADED = coach.GRADED
FAULTS = coach.FAULTS


def skill_summary(rows: list[dict]) -> dict:
    g = [r for r in rows if r.get("classification") in GRADED and r.get("wp_loss") is not None]
    out = {"n_coached": len(rows), "n_graded": len(g), "classes": dict(Counter(r.get("classification") for r in rows))}

    def block(xs):
        agree = [r for r in xs if r.get("agree") is not None]
        k = sum(bool(r["agree"]) for r in agree)
        faults = sum(r["classification"] in FAULTS for r in xs)
        chance = [len(r["matched"]) / r["n_legal"] for r in xs if r.get("n_legal")]
        return {"n": len(xs),
                "mean_wp_loss": round(float(np.mean([r["wp_loss"] for r in xs])), 5) if xs else None,
                "mean_wp_loss_ci": boot_mean([r["wp_loss"] for r in xs]),
                "mean_wp_loss_mean_of_set": round(float(np.mean([r["wp_loss_mean_of_set"] for r in xs])), 5) if xs else None,
                "agreement": round(k / len(agree), 4) if agree else None, "agreement_ci": wilson(k, len(agree)),
                "chance_agreement": round(float(np.mean(chance)), 4) if chance else None,
                "share_faults": round(faults / len(xs), 4) if xs else None, "share_faults_ci": wilson(faults, len(xs)),
                "share_faults_no_caution": round(sum(r["classification"] in FAULTS and not r["caution"] for r in xs)
                                                 / len(xs), 4) if xs else None,
                "mean_set_size": round(float(np.mean([len(r["matched"]) for r in xs])), 3) if xs else None,
                "mean_legal": round(float(np.mean([r["n_legal"] for r in xs])), 3) if xs else None}

    out["all"] = block(g)
    out["by_band"] = {bd: block([r for r in g if r["band"] == bd]) for bd in BANDS}
    out["by_result"] = {k: block([r for r in g if bool(r["won"]) == (k == "won")]) for k in ("won", "lost")}
    out["by_rank"] = {k: block([r for r in g if (r.get("rank") or "none") == k])
                      for k in sorted({r.get("rank") or "none" for r in g})}
    x = [r["win_rate_bucket"] for r in g]
    y = [r["wp_loss"] for r in g]
    out["spearman_wr_vs_loss"] = spearman(x, y)
    out["spearman_wr_vs_loss_ci"] = boot_spearman(x, y)
    ym = [r["wp_loss_mean_of_set"] for r in g]
    out["spearman_wr_vs_loss_mean_of_set"] = spearman(x, ym)
    out["spearman_wr_vs_loss_mean_of_set_ci"] = boot_spearman(x, ym)
    ya = [float(bool(r["agree"])) for r in g]
    out["spearman_wr_vs_agree"] = spearman(x, ya)
    out["spearman_wr_vs_agree_ci"] = boot_spearman(x, ya)
    # the same controlling for chance: agreement minus |set|/|legal|
    yx = [float(bool(r["agree"])) - len(r["matched"]) / r["n_legal"] for r in g]
    out["spearman_wr_vs_excess_agree"] = spearman(x, yx)
    out["spearman_wr_vs_excess_agree_ci"] = boot_spearman(x, yx)
    # informative subset: the human's set does not cover every legal option
    inf = [r for r in g if not set(r["legal"]) <= set(r["matched"])]
    out["informative_subset"] = {"n": len(inf),
                                 "spearman_wr_vs_loss": spearman([r["win_rate_bucket"] for r in inf],
                                                                 [r["wp_loss"] for r in inf]),
                                 "spearman_wr_vs_loss_ci": boot_spearman([r["win_rate_bucket"] for r in inf],
                                                                         [r["wp_loss"] for r in inf])}
    # won vs lost: a valid coach's regret should be higher in lost games (point-biserial as Spearman)
    out["spearman_won_vs_loss"] = spearman([float(r["won"]) for r in g], y)
    out["spearman_won_vs_loss_ci"] = boot_spearman([float(r["won"]) for r in g], y)
    q = [r for r in g]
    spreads = [max(o["q"] for o in r["options"] if o["q"] is not None) - min(o["q"] for o in r["options"] if o["q"] is not None)
               for r in q if any(o["q"] is not None for o in r["options"])]
    out["q_spread_median"] = round(float(np.median(spreads)), 4) if spreads else None
    out["near_tie_share"] = round(sum("near_tie" in r["flags"] for r in g) / len(g), 4) if g else None
    out["thin_search_share"] = round(sum("thin_search" in r["flags"] for r in g) / len(g), 4) if g else None
    out["unstable_share"] = round(sum("unstable" in r["flags"] for r in g) / len(g), 4) if g else None
    out["flags"] = dict(Counter(f for r in rows for f in r.get("flags") or []).most_common())
    out["det_methods"] = dict(Counter(r.get("det_method") for r in rows))
    secs = [r["seconds"] for r in rows if r.get("seconds") is not None]
    sps = [r["sims_per_s"] for r in rows if r.get("sims_per_s")]
    out["seconds"] = {"median": round(float(np.median(secs)), 2), "p90": round(float(np.quantile(secs, 0.9)), 2),
                      "mean": round(float(np.mean(secs)), 2), "total": round(float(np.sum(secs)), 1)} if secs else None
    out["sims_per_s_median"] = float(np.median(sps)) if sps else None
    return out


def option_q(r: dict) -> dict:
    return {o["label"]: o["q"] for o in r.get("options") or [] if o.get("q") is not None}


def paired(a_rows: list[dict], b_rows: list[dict], name_a: str, name_b: str) -> dict:
    """Two settings on the same decisions: same best, rank agreement of per-option Q, regrets."""
    A = {r["id"]: r for r in a_rows if r.get("classification") in GRADED}
    B = {r["id"]: r for r in b_rows if r.get("classification") in GRADED}
    common = sorted(set(A) & set(B))
    same, rhos, la, lb, agree_a, agree_b, cls_same = 0, [], [], [], [], [], 0
    for i in common:
        a, b = A[i], B[i]
        same += a["best"] == b["best"]
        qa, qb = option_q(a), option_q(b)
        labs = sorted(set(qa) & set(qb))
        if len(labs) >= 3:
            # min_n=3: the default (4) silently dropped every 3-option decision (verification fix)
            s = spearman([qa[x] for x in labs], [qb[x] for x in labs], min_n=3)
            if s is not None:
                rhos.append(s)
        la.append(a["wp_loss"])
        lb.append(b["wp_loss"])
        agree_a.append(bool(a["agree"]))
        agree_b.append(bool(b["agree"]))
        cls_same += a["classification"] == b["classification"]
    n = len(common)
    out = {"a": name_a, "b": name_b, "n": n}
    if not n:
        return out
    out.update(same_best=same, same_best_share=round(same / n, 4), same_best_ci=wilson(same, n),
               same_class_share=round(cls_same / n, 4),
               option_q_spearman_median=round(float(np.median(rhos)), 4) if rhos else None,
               option_q_spearman_mean=round(float(np.mean(rhos)), 4) if rhos else None,
               option_q_spearman_mean_ci=boot_mean(rhos), option_q_spearman_n=len(rhos),
               mean_wp_loss_a=round(float(np.mean(la)), 5), mean_wp_loss_a_ci=boot_mean(la),
               mean_wp_loss_b=round(float(np.mean(lb)), 5), mean_wp_loss_b_ci=boot_mean(lb),
               mean_wp_loss_diff_ci=boot_mean([x - y for x, y in zip(la, lb)]),
               loss_spearman=spearman(la, lb), loss_spearman_ci=boot_spearman(la, lb),
               agreement_a=round(sum(agree_a) / n, 4), agreement_a_ci=wilson(sum(agree_a), n),
               agreement_b=round(sum(agree_b) / n, 4), agreement_b_ci=wilson(sum(agree_b), n),
               agreement_diff_ci=boot_mean([float(x) - float(y) for x, y in zip(agree_a, agree_b)]),
               both_agree=sum(x and y for x, y in zip(agree_a, agree_b)),
               human_is_best_both_or_neither=round(sum(x == y for x, y in zip(agree_a, agree_b)) / n, 4))
    return out


def stability(rows: list[dict], ks=(1, 2, 4, 8), n_sub: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {"n": 0, "by_k": {}}

    def best_of(per_det, idx):
        acc = defaultdict(list)
        share = defaultdict(float)
        for i in idx:
            d = per_det[i]
            tot = sum(n for _, n in d.values()) or 1
            for lab, (q, n) in d.items():
                acc[lab].append(q)
                share[lab] += n / tot
        if not acc:
            return None
        return max(acc, key=lambda lab: (np.mean(acc[lab]), share[lab]))

    good = [r for r in rows if r.get("per_det") and len(r["per_det"]) >= 16]
    out["n"] = len(good)
    full_best = {r["id"]: best_of(r["per_det"], range(16)) for r in good}
    for k in ks:
        hits, hum = [], []
        for r in good:
            ref = full_best[r["id"]]
            h16 = ref in r["matched"]
            for _ in range(n_sub if k < 16 else 1):
                idx = rng.choice(16, k, replace=False)
                b = best_of(r["per_det"], idx)
                hits.append(b == ref)
                hum.append((b in r["matched"]) == h16)
        # decision-level: share of subsets matching, averaged over decisions
        per_dec = np.asarray(hits, float).reshape(len(good), -1).mean(1) if good else np.array([])
        out["by_k"][str(k)] = {"match_k16": round(float(np.mean(hits)), 4) if hits else None,
                               "match_k16_ci": boot_mean(per_dec.tolist()),
                               "agreement_verdict_same_as_k16": round(float(np.mean(hum)), 4) if hum else None}
    # split-half: two disjoint K=8 halves
    sh = []
    for r in good:
        for _ in range(n_sub):
            p = rng.permutation(16)
            sh.append(best_of(r["per_det"], p[:8]) == best_of(r["per_det"], p[8:]))
    per = np.asarray(sh, float).reshape(len(good), -1).mean(1) if good else np.array([])
    out["split_half_k8_same_best"] = round(float(np.mean(sh)), 4) if sh else None
    out["split_half_k8_ci"] = boot_mean(per.tolist())
    # how often is the K=16 best the argmax of a single determinization (per-det agreement)
    out["seconds_median"] = float(np.median([r["seconds"] for r in good])) if good else None
    out["k16_best_in_human_set"] = round(sum(full_best[r["id"]] in r["matched"] for r in good) / len(good), 4) if good else None
    return out


def cmd_analyze(a) -> int:
    sample = json.loads(SAMPLE.read_text())
    res: dict = {"sample": {"stats": sample["stats"], "n_skill": len(sample["skill"]), "n_pairs": len(sample["pairs"]),
                            "skill_by_band": dict(Counter(d["band"] for d in sample["skill"])),
                            "settings": sample["settings"]}}
    rows = {s: load(s) for s in SETTINGS}
    res["skill"] = {s: skill_summary(rows[s]) for s in ("net_bin", "net_pri", "off") if rows[s]}
    # offline restricted to the decisions the network run reached (paired)
    net_ids = {r["id"] for r in rows["net_bin"]}
    if rows["off"] and net_ids:
        res["skill"]["off_on_net_decisions"] = skill_summary([r for r in rows["off"] if r["id"] in net_ids])
    res["paired"] = {}
    if rows["net_bin"] and rows["off"]:
        res["paired"]["net_bin_vs_off"] = paired(rows["net_bin"], rows["off"], "net_bin", "off")
    if rows["net_pri"] and rows["net_bin"]:
        res["paired"]["net_pri_vs_net_bin"] = paired(rows["net_pri"], rows["net_bin"], "net_pri", "net_bin")
    if rows["net_pri"] and rows["off"]:
        res["paired"]["net_pri_vs_off"] = paired(rows["net_pri"], rows["off"], "net_pri", "off")
    res["hindsight"] = {}
    for ev in ("off", "net"):
        a_, b_ = rows[f"pair_belief_{ev}"], rows[f"pair_exact_{ev}"]
        if a_ and b_:
            p = paired(a_, b_, f"pair_belief_{ev}", f"pair_exact_{ev}")
            p["belief_classes"] = dict(Counter(r["classification"] for r in a_))
            p["exact_classes"] = dict(Counter(r["classification"] for r in b_))
            p["belief_det_methods"] = dict(Counter(r.get("det_method") for r in a_))
            p["exact_det_methods"] = dict(Counter(r.get("det_method") for r in b_))
            p["seconds_belief_median"] = float(np.median([r["seconds"] for r in a_ if r.get("seconds")]))
            p["seconds_exact_median"] = float(np.median([r["seconds"] for r in b_ if r.get("seconds")]))
            res["hindsight"][ev] = p
    if rows["pair_belief_off"] and rows["pair_belief_net"]:
        res["hindsight"]["belief_net_vs_off"] = paired(rows["pair_belief_net"], rows["pair_belief_off"],
                                                       "pair_belief_net", "pair_belief_off")
    if rows["stab16"]:
        res["stability"] = stability(rows["stab16"])
    res["cost"] = {}
    for s, rs in rows.items():
        secs = [r["seconds"] for r in rs if r.get("seconds") is not None]
        if secs:
            sps = [r["sims_per_s"] for r in rs if r.get("sims_per_s")]
            res["cost"][s] = {"n": len(secs), "median_s": round(float(np.median(secs)), 2),
                              "p90_s": round(float(np.quantile(secs, 0.9)), 2),
                              "mean_s": round(float(np.mean(secs)), 2),
                              "sims_per_s_median": float(np.median(sps)) if sps else None,
                              "errors": sum(r.get("classification") == "error" for r in rs)}
    extra = OUT / "extra.json"
    if extra.exists():
        res["extra"] = json.loads(extra.read_text())
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (OUT / "results.md").write_text(render(res))
    print(json.dumps(res, indent=1, default=float)[:3000])
    return 0


def _f(x, pct: bool = False, d: int = 3) -> str:
    if x is None:
        return "–"
    return f"{100 * x:.{max(d - 2, 0)}f}%" if pct else f"{x:.{d}f}"


def _ci(c, pct: bool = False, d: int = 3) -> str:
    return "–" if not c else f"[{_f(c[0], pct, d)}, {_f(c[1], pct, d)}]"


SETTING_TEXT = {
    "net_bin": "network (gen33, binary prior only, as exp #1), K=8 x 300",
    "net_pri": "network (gen33, priority + binary priors), K=8 x 300",
    "off": "offline heuristic, K=8 x 300 (thin faults re-searched at 3,000)",
    "off_on_net_decisions": "offline, restricted to the decisions the network run coached",
}


def render(res: dict) -> str:
    L = ["# Coaching validity (E5): results", "",
         "Generated by `tools/gameplay/coach_validity.py analyze` from the verdict rows in this directory.",
         "Loss = the human's best-of-set regret in win-probability units ((Q_best - Q_human) / 2, nominal).",
         "Agreement = the engine's best option is in the human's 17lands action set; chance = mean |set| / |legal|.",
         "95% CIs: bootstrap over decisions (means, Spearman), Wilson (rates).", ""]
    interp = OUT / "interpretation.md"
    if interp.exists():
        L += [interp.read_text().rstrip(), ""]
    s = res["sample"]
    L += ["## Sample", "",
          f"- Skill decisions sampled: {s['n_skill']} (per band {s['skill_by_band']}); pair decisions: {s['n_pairs']}.",
          f"- Stride rows {s['stats']['stride_rows']} (every {s['settings']['every']}th row from {s['settings']['start']}); "
          f"users with < {s['settings']['min_games']} games skipped: {s['stats']['few_games']}; mirrored-pair rows "
          f"(part 2 only): {s['stats']['pair_rows']}.",
          f"- Prescreen outcomes per tried turn: {s['stats']['reasons']}.", ""]
    L += ["## 1. Skill signal", ""]
    for name, sm in res.get("skill", {}).items():
        L += [f"### {name}: {SETTING_TEXT.get(name, name)}", "",
              f"n coached {sm['n_coached']}, graded {sm['n_graded']}; classes {sm['classes']}.", "",
              "| group | n | mean loss [95% CI] | agreement [95% CI] | chance | faults (inaccuracy+) [95% CI] | faults w/o caution |",
              "|---|---|---|---|---|---|---|"]
        rows = [("all", sm["all"])] + [(f"win rate {k}", v) for k, v in sm["by_band"].items()] + \
               [(k, v) for k, v in sm["by_result"].items()]
        for lab, b in rows:
            if not b["n"]:
                continue
            L.append(f"| {lab} | {b['n']} | {_f(b['mean_wp_loss'], True, 4)} {_ci(b['mean_wp_loss_ci'], True, 4)} | "
                     f"{_f(b['agreement'], True)} {_ci(b['agreement_ci'], True)} | {_f(b['chance_agreement'], True)} | "
                     f"{_f(b['share_faults'], True)} {_ci(b['share_faults_ci'], True)} | {_f(b['share_faults_no_caution'], True)} |")
        L += ["",
              f"- Spearman rho(win-rate bucket, loss) = {_f(sm['spearman_wr_vs_loss'])} {_ci(sm['spearman_wr_vs_loss_ci'])} "
              f"(a valid coach gives a NEGATIVE rho); mean-of-set loss: {_f(sm['spearman_wr_vs_loss_mean_of_set'])} "
              f"{_ci(sm['spearman_wr_vs_loss_mean_of_set_ci'])}.",
              f"- rho(win-rate bucket, agreement) = {_f(sm['spearman_wr_vs_agree'])} {_ci(sm['spearman_wr_vs_agree_ci'])}; "
              f"agreement minus chance: {_f(sm['spearman_wr_vs_excess_agree'])} {_ci(sm['spearman_wr_vs_excess_agree_ci'])} "
              f"(valid coach: POSITIVE).",
              f"- Informative subset (the human's set does not cover every legal option), n={sm['informative_subset']['n']}: "
              f"rho(bucket, loss) = {_f(sm['informative_subset']['spearman_wr_vs_loss'])} "
              f"{_ci(sm['informative_subset']['spearman_wr_vs_loss_ci'])}.",
              f"- rho(won, loss) = {_f(sm['spearman_won_vs_loss'])} {_ci(sm['spearman_won_vs_loss_ci'])} "
              f"(valid coach: negative, mistakes lose games).",
              f"- Median Q spread (best - worst option): {_f(sm['q_spread_median'], d=4)}; near-tie flag "
              f"{_f(sm['near_tie_share'], True)}, unstable {_f(sm['unstable_share'], True)}, thin_search "
              f"{_f(sm['thin_search_share'], True)}.",
              f"- Seconds per decision: {sm['seconds']}; median simulations/s {sm['sims_per_s_median']}.", ""]
    if res.get("paired"):
        L += ["### Paired comparisons on the same decisions", "",
              "| a vs b | n | same best [95% CI] | same class | per-option Q Spearman, mean [95% CI] | loss a | loss b | "
              "loss diff a-b [95% CI] | rho(loss a, loss b) [95% CI] | agreement a | agreement b | agreement diff a-b [95% CI] |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for p in res["paired"].values():
            if not p.get("n"):
                continue
            L.append(f"| {p['a']} vs {p['b']} | {p['n']} | {_f(p['same_best_share'], True)} {_ci(p['same_best_ci'], True)} | "
                     f"{_f(p['same_class_share'], True)} | {_f(p['option_q_spearman_mean'])} {_ci(p['option_q_spearman_mean_ci'])} "
                     f"(n={p['option_q_spearman_n']}) | {_f(p['mean_wp_loss_a'], True, 4)} | {_f(p['mean_wp_loss_b'], True, 4)} | "
                     f"{_ci(p['mean_wp_loss_diff_ci'], True, 4)} | {_f(p['loss_spearman'])} {_ci(p['loss_spearman_ci'])} | "
                     f"{_f(p['agreement_a'], True)} | {_f(p['agreement_b'], True)} | {_ci(p.get('agreement_diff_ci'), True)} |")
        L.append("")
    if res.get("hindsight"):
        L += ["## 2. Hindsight: belief determinization (PIMC) vs the true opponent hand and deck", "",
              "Mirrored-pair rows, both drafts held out of the belief pool. a = belief, b = exact_spec (true hand).", "",
              "| evaluator | n | same best [95% CI] | same class | per-option Q Spearman, mean [95% CI] | human loss belief | "
              "human loss exact | diff [95% CI] | rho(loss) [95% CI] | agreement belief | agreement exact |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for ev, p in res["hindsight"].items():
            if not p.get("n"):
                continue
            L.append(f"| {ev} | {p['n']} | {_f(p['same_best_share'], True)} {_ci(p['same_best_ci'], True)} | "
                     f"{_f(p['same_class_share'], True)} | {_f(p['option_q_spearman_mean'])} {_ci(p['option_q_spearman_mean_ci'])} "
                     f"(n={p['option_q_spearman_n']}) | {_f(p['mean_wp_loss_a'], True, 4)} | {_f(p['mean_wp_loss_b'], True, 4)} | "
                     f"{_ci(p['mean_wp_loss_diff_ci'], True, 4)} | {_f(p['loss_spearman'])} {_ci(p['loss_spearman_ci'])} | "
                     f"{_f(p['agreement_a'], True)} | {_f(p['agreement_b'], True)} |")
        for ev, p in res["hindsight"].items():
            if "belief_classes" in p:
                L.append(f"- {ev}: classes belief {p['belief_classes']}, exact {p['exact_classes']}; determinization "
                         f"belief {p['belief_det_methods']}, exact {p['exact_det_methods']}; median seconds "
                         f"{p['seconds_belief_median']:.1f} / {p['seconds_exact_median']:.1f}.")
        L.append("")
    st = res.get("stability")
    if st:
        L += ["## 3. Stability: how many determinizations", "",
              f"{st['n']} decisions searched offline with K=16 x 300 (belief determinizations, seed 0). For each K, 200 random "
              "subsets of the 16 per decision; 'match' = the subset's best option (highest mean Q) equals K=16's.", "",
              "| K | best matches K=16 [95% CI over decisions] | human-agreement verdict same as K=16 |", "|---|---|---|"]
        for k, v in st["by_k"].items():
            L.append(f"| {k} | {_f(v['match_k16'], True)} {_ci(v['match_k16_ci'], True)} | "
                     f"{_f(v['agreement_verdict_same_as_k16'], True)} |")
        L += ["", "A K-subset shares K of K=16's determinizations, so the match rates above are inflated self-agreement "
                  "(K=8 shares half); the split-half figure below is the fair K=8 test-retest reliability.",
              f"- Split-half: two disjoint K=8 halves pick the same best in {_f(st['split_half_k8_same_best'], True)} "
                  f"{_ci(st['split_half_k8_ci'], True)}.",
              f"- Median seconds per K=16 decision: {st['seconds_median']}.", ""]
    if res.get("cost"):
        L += ["## 4. Cost on this Mac (M1 Pro, shared; 2 bridge JVMs; gen33 on MPS, plus a CPU server for one "
              "JVM after the first 48 network decisions)", "",
              "| setting | n | median s | p90 s | mean s | median sims/s (per search) | errors |", "|---|---|---|---|---|---|---|"]
        for k, v in res["cost"].items():
            L.append(f"| {k} | {v['n']} | {v['median_s']} | {v['p90_s']} | {v['mean_s']} | {v['sims_per_s_median']} | {v['errors']} |")
        L.append("")
    if res.get("extra"):
        L += ["## Extra measurements", "", "```json", json.dumps(res["extra"], indent=1), "```", ""]
    return "\n".join(L) + "\n"


def mana_efficiency(g, n: int) -> float | None:
    """Share of user turn n's lands whose mana the user spent (capped at 1): a crude, engine-free
    skill proxy (it also carries luck: flood, screw, hand quality)."""
    t = g.user_slot(n)
    if t is None:
        return None
    lands = len(t.L("eot_user_lands_in_play"))
    if lands == 0:
        return None
    return min(t.num("user_mana_spent") / lands, 1.0)


def cmd_proxy(a) -> int:
    """Power reference without the engine: how strongly does a crude per-turn skill proxy (mana
    efficiency) track the win-rate bucket, at the coach's n and at a larger n? Also: does the
    coach's loss track it? Writes extra.json (merged into results by analyze)."""
    sample = json.loads(SAMPLE.read_text())
    partners = pm.partner_map(pm.load_pairs())
    rng = random.Random(1)
    st = sample["settings"]
    t0 = time.monotonic()
    big, games = [], {}
    want = {d["row"] for d in sample["skill"]}
    for g in replay.iter_games(every=st["every"], start=st["start"]):
        if g.row_index in want:
            games[g.row_index] = g
        if g.row_index in partners or (g.meta.get("user_n_games_bucket") or 0) < st["min_games"]:
            continue
        turns = [n for n in g.decision_turns() if n in TURNS]
        if not turns:
            continue
        e = mana_efficiency(g, rng.choice(turns))
        if e is not None:
            big.append((g.meta.get("user_game_win_rate_bucket"), e, bool(g.meta.get("won"))))
    small = []
    for d in sample["skill"]:
        e = mana_efficiency(games[d["row"]], d["turn"])
        if e is not None:
            small.append((d["win_rate_bucket"], e, d["won"], d["id"]))
    out = {"what": "Spearman of the win-rate bucket (and of the game result) with mana efficiency at one user turn "
                   "(3-8) per game: an engine-free skill proxy, for power reference",
           "stride_games": {"n": len(big), "rho_wr": spearman([x[0] for x in big], [x[1] for x in big]),
                            "rho_wr_ci": boot_spearman([x[0] for x in big], [x[1] for x in big]),
                            "rho_won": spearman([float(x[2]) for x in big], [x[1] for x in big]),
                            "rho_won_ci": boot_spearman([float(x[2]) for x in big], [x[1] for x in big])},
           "coach_decisions": {"n": len(small), "rho_wr": spearman([x[0] for x in small], [x[1] for x in small]),
                               "rho_wr_ci": boot_spearman([x[0] for x in small], [x[1] for x in small])}}
    eff = {x[3]: x[1] for x in small}
    for s in ("net_bin", "off", "net_pri"):
        rows = [r for r in load(s) if r.get("classification") in GRADED and r["id"] in eff]
        if len(rows) >= 10:
            xs = [eff[r["id"]] for r in rows]
            ys = [r["wp_loss"] for r in rows]
            out[f"loss_vs_efficiency_{s}"] = {"n": len(rows), "rho": spearman(xs, ys), "rho_ci": boot_spearman(xs, ys),
                                              "note": "a valid coach charges more loss on turns with unspent mana: negative rho"}
            sub = [r for r in rows if eff[r["id"]] is not None]
            out[f"coach_n_{s}_rho_wr_efficiency"] = spearman([r["win_rate_bucket"] for r in sub], [eff[r["id"]] for r in sub])
    out["seconds"] = round(time.monotonic() - t0, 1)
    prev = json.loads((OUT / "extra.json").read_text()) if (OUT / "extra.json").exists() else {}
    prev = {k: v for k, v in prev.items() if not k.startswith(("loss_vs_efficiency_", "coach_n_"))}
    (OUT / "extra.json").write_text(json.dumps({**prev, **out}, indent=1))     # keeps e.g. "throughput"
    print(json.dumps(out, indent=1))
    return 0


def cmd_clean(a) -> int:
    p = CACHE / "games.pkl"
    if p.exists():
        p.unlink()
    if CACHE.exists() and not any(CACHE.iterdir()):
        CACHE.rmdir()
    print("game cache deleted")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--every", type=int, default=173)
    s.add_argument("--start", type=int, default=11)
    s.add_argument("--per-band", type=int, default=110)
    s.add_argument("--pairs", type=int, default=120)
    s.add_argument("--min-games", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--runtime", default=str(DEFAULT_RUNTIME))
    r = sub.add_parser("run")
    r.add_argument("--setting", required=True, choices=sorted(SETTINGS))
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--port", default="50093", help="inference server port(s), comma-separated: worker i uses port i mod n")
    r.add_argument("--minutes", type=float, default=0)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--offset", type=int, default=0)
    r.add_argument("--ids", default="")
    r.add_argument("--heap", default="2500m")
    r.add_argument("--runtime", default=str(DEFAULT_RUNTIME))
    sub.add_parser("analyze")
    sub.add_parser("proxy")
    sub.add_parser("clean")
    a = ap.parse_args(argv)
    return {"sample": cmd_sample, "run": cmd_run, "analyze": cmd_analyze, "proxy": cmd_proxy,
            "clean": cmd_clean}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
