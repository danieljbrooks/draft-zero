"""Offline gate: does a value target that reaches the seat's next own turn fix the value's overshoot late in its own
turn and on the opponent's turn, at no cost to the policy or to the value against results? (draft-zero-report
docs/038 §7 step 4)

E1's value target is the search's root value (learner.value_blend 1.0), whose leaves lie under a turn ahead, so it
never sees the opponent's turn, where held mana and tricks pay off. The treatment trains the value partly on the
seat's root value at its next own turn (nets.train --value-next; KataGo's short-term value targets, MuZero's n-step
bootstrap). From one of E1's versions (default the last, v36) the arms fine-tune on E1's newest training window,
with the same steps and seed:

  control     --value-blend 1.0                     (E1's recipe)
  treatment   --value-blend 0.5 --value-next 0.5    (the proposal's)
  next-only   --value-blend 0   --value-next 1.0    (opt-in with --arms: the whole target from the next own turn)

The split is replay_ratio.py's: the run's shards in the learner's ingest order, the newest 15% the test set (no arm
trains on it), the window the newest learner.window_positions positions before it (the run's effective config:
config.yml plus events.jsonl). A version's schedule is learner.epochs (E1's 2) passes of the window, as the learner
runs it (--epochs, or --steps N). Each arm's best.pt (its holdout pick, which the learner would publish) is scored on
the test set, beside the starting network and the search's own root value:

- policy CE (visit targets), value log loss, AUC and Brier score against the game results, as nets.train evaluates;
- drift = mean(value - next_rv) in points (x100), next_rv being the training target's: the seat's root value at its
  next own turn (nets.data.next_own_turn), or the game's result when the game ends first, over records with a known
  result. A calibrated value has zero mean drift from it (by optional stopping), and the records whose game ends
  before the seat's next turn are kept, since they overshoot most. With a 95% interval clustered by game, by window
  class (whose turn it is and the step: dense[19] and the one-hot dense[2:14]) and by whether the seat holds a
  castable instant-speed card: a card in its hand (group 2) with the encoder's castable bit (raw[63]: not a land,
  mana value at most its floating plus available mana; colours and targets are not checked) and instant-speed bit
  (raw[65]: an instant, or a card with Flash). Positive drift is overshoot (docs/038 quotes next minus now, so its
  signs flip). A second table, "while the game continues", is the same over the records that have a next own turn,
  against its root value.

Each arm beside the control passes the gate when its drift over GATE_WINDOWS while holding is within +-1 point, its
policy CE is at most 0.005 above the control's and its value log loss is not above the control's. Writes
OUT/results.json and OUT/summary.md. A finished arm (OUT/<arm>/summary.json) is reused by a rerun with the same
settings and refused with other settings. Packs come from --packs folders (the learner's cache, or replay_ratio.py's
OUT/packs) when they hold a shard whole, else are built in OUT/packs with dzgorge pack.

    python value_next_gate.py STORE OUT [--init PT] [--packs DIR ...] [--epochs E | --steps N] [--max-shards N]
                              [--threads 2] [--arms control,treatment[,next-only]]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_ratio import pack_dirs, stream                     # noqa: E402  (the same split as replay_ratio.py)

import draftzero                                               # noqa: E402
from draftzero import config as C                              # noqa: E402
from draftzero.loop import control as ctl                      # noqa: E402
from draftzero.loop.store import LocalStore                    # noqa: E402
from draftzero.nets import data as D                           # noqa: E402

ARMS = {"control": (1.0, 0.0), "treatment": (0.5, 0.5), "next-only": (0.0, 1.0)}   # (value_blend, value_next)
STEPS = ("untap", "upkeep", "draw", "main1", "begin-combat", "declare-attackers", "declare-blockers", "combat-damage",
         "end-combat", "main2", "end", "cleanup")            # dense[2:14], gorge policynet stepNames
OWN = {s: s for s in ("upkeep", "draw", "main1", "begin-combat", "declare-attackers", "declare-blockers", "main2",
                      "end")}
OPP = {"upkeep": "upkeep", "draw": "draw", "main1": "main", "main2": "main", "begin-combat": "combat",
       "declare-attackers": "combat", "declare-blockers": "combat", "combat-damage": "combat", "end-combat": "combat",
       "end": "end", "cleanup": "end"}
CLASSES = [f"own {s}" for s in ("upkeep", "draw", "main1", "begin-combat", "declare-attackers", "declare-blockers",
                                "main2", "end", "other")] + \
          [f"opp {s}" for s in ("upkeep", "draw", "main", "combat", "end", "other")]
# the windows where docs/038 §7 found the root value overshooting while the seat holds a castable instant
GATE_WINDOWS = ("own begin-combat", "own declare-blockers", "own main2", "own end", "opp upkeep", "opp draw")
CASTABLE, INSTANT, HAND = 63, 65, 2                          # card raw bits (gorge entCastable, entInstant); group
GATE = {"drift_points": 1.0, "policy_ce": 0.005, "drift_share": 0.5}     # drift_share: an informational check
METRICS = ("policy_ce", "policy_ce_uniform", "value_logloss", "value_auc", "value_brier")


# ------------------------------------------------------------------------------------------------ the test set

def window_class(dense) -> np.ndarray:
    """[n] index into CLASSES from the dense features: whose turn (dense[19]) and the step (dense[2:14])."""
    dense = np.asarray(dense)
    own = dense[:, D.ACTIVE] > 0.5
    oh = dense[:, 2:14]
    step = np.where(oh.max(1) > 0, oh.argmax(1), -1)
    own_ix = {s: CLASSES.index(f"own {OWN.get(s, 'other')}") for s in STEPS}
    opp_ix = {s: CLASSES.index(f"opp {OPP.get(s, 'other')}") for s in STEPS}
    own_lut = np.array([own_ix[s] for s in STEPS] + [CLASSES.index("own other")])
    opp_lut = np.array([opp_ix[s] for s in STEPS] + [CLASSES.index("opp other")])
    return np.where(own, own_lut[step], opp_lut[step])       # step -1 (none) indexes the last entry: other


def holding(ds: D.ShardSet) -> np.ndarray:
    """[n] bool: the seat holds a castable instant-speed card (hand, raw[CASTABLE] and raw[INSTANT])."""
    out = []
    for p in ds.packs:
        off = np.asarray(p["card_off"], np.int64)
        raw = p["card_raw"]
        hit = (np.asarray(p["card_group"]) == HAND) & (np.asarray(raw[:, CASTABLE]) > 0.5) & \
              (np.asarray(raw[:, INSTANT]) > 0.5)
        c = np.concatenate([[0], np.cumsum(hit, dtype=np.int64)])
        out.append(c[off[1:]] > c[off[:-1]])
    return np.concatenate(out) if out else np.zeros(0, bool)


def clustered(x: np.ndarray, groups: np.ndarray) -> dict:
    """The mean of x with a 95% interval from a standard error clustered by group (records of one game are not
    independent): var = G / (G - 1) * sum over groups of (sum of x - mean)^2 / n^2."""
    n = len(x)
    if n == 0:
        return {"n": 0, "games": 0, "mean": None, "lo": None, "hi": None}
    m = float(x.mean())
    _, g = np.unique(groups, return_inverse=True)
    G = int(g.max()) + 1
    s = np.bincount(g, weights=x - m, minlength=G)
    se = math.sqrt(G / max(G - 1, 1) * float((s ** 2).sum())) / n if G > 1 else float("nan")
    return {"n": n, "games": G, "mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se}


def test_set(dirs: list[str]) -> dict:
    """The test records' per-record arrays: game keys, outcome, root value, next (the next own turn's root value, -1
    none), nrv (the training target's next_rv: next, else the outcome), class, holding."""
    ds = D.ShardSet(dirs)
    nxt = ds.next_own_turn()
    rv = ds.array("root_value").astype(np.float64)
    oc = ds.array("outcome").astype(np.float64)
    dense = np.concatenate([np.asarray(p["dense"][:, :D.ACTIVE + 1]) for p in ds.packs])
    nx = np.where(nxt >= 0, rv[np.maximum(nxt, 0)], -1.0)
    return {"ds": ds, "game": ds.game_keys(), "outcome": oc, "rv": rv, "next": nx,
            "nrv": np.where(nxt >= 0, nx, oc), "cls": window_class(dense), "hold": holding(ds)}


def drift_table(value: np.ndarray, t: dict, against: str = "target") -> dict:
    """Drift in points, 100 x (value - reference), by class and holding, over the gate windows while holding and over
    all. against "target": the training target's next_rv, over records with a known result (each cell also gives
    the share whose game ends before the seat's next own turn, whose reference is the result); "continues": the
    next own turn's root value, over the records that have one."""
    if against == "target":
        ref, keep = t["nrv"], t["outcome"] >= 0
    else:
        ref, keep = t["next"], t["next"] >= 0
    d = (value - ref) * 100
    ends = t["next"] < 0

    def cell(m):
        r = clustered(d[m], t["game"][m])
        if against == "target":
            r["ends_first"] = float(ends[m].mean()) if m.any() else None
        return r
    out = {"all": cell(keep)}
    for i, c in enumerate(CLASSES):
        for h in (True, False):
            out[f"{c} | {'holding' if h else 'not holding'}"] = cell(keep & (t["cls"] == i) & (t["hold"] == h))
    out["gate windows | holding"] = cell(keep & gate_mask(t))
    return out


def gate_mask(t: dict) -> np.ndarray:
    """[n] bool: the record is at one of GATE_WINDOWS and the seat holds a castable instant-speed card."""
    return t["hold"] & np.isin(t["cls"], [CLASSES.index(c) for c in GATE_WINDOWS])


# ------------------------------------------------------------------------------------------------ scoring

def score(pt: Path | None, t: dict, threads: int, batch: int = 1024) -> dict:
    """A checkpoint on the test set: nets.train.evaluate's policy and value metrics (value against the results), the
    per-record value and policy CE (for paired differences) and the drift table. pt None: the search's root value
    (value metrics and drift only)."""
    if pt is None:
        p = np.clip(t["rv"], 1e-6, 1 - 1e-6)
        return {**value_metrics(np.log(p / (1 - p)), t), "drift": drift_table(t["rv"], t),
                "drift_continues": drift_table(t["rv"], t, "continues")}
    import torch
    from draftzero.nets import models as M
    from draftzero.nets import train
    torch.set_num_threads(threads)
    model, _ = M.load(pt)
    ds = t["ds"]
    args = SimpleNamespace(policy_target="visits", visits_temp=1.0, cq_c_visit=50.0, cq_c_scale=0.1, value_blend=0.0,
                           value_next=0.0, value_weight=1.0)
    vlogit = np.zeros(ds.n)
    ce = np.full(ds.n, np.nan)
    unif = np.full(ds.n, np.nan)
    idx = np.arange(ds.n, dtype=np.int64)
    bl = D.batches(idx, batch, shuffle=False)
    with torch.inference_mode():
        for sel, b in zip(bl, D.Loader(ds, bl, prefetch=4, workers=1)):
            _, _, _, x = train.losses(model, b, args, weighted=False)
            vlogit[sel] = x["vlogit"].float().numpy()
            pm = x["pm"].numpy()
            k = train.seg_sum(torch.ones_like(b.cand_visits), b.cand_state, b.B)
            ce[sel[pm]] = x["ce_state"].numpy()[pm]
            unif[sel[pm]] = torch.log(k.clamp(min=1)).numpy()[pm]
    pol = ~np.isnan(ce)
    v = 1 / (1 + np.exp(-vlogit))
    out = {"n_policy": int(pol.sum()), "policy_ce": float(ce[pol].mean()), "policy_ce_uniform": float(unif[pol].mean()),
           **value_metrics(vlogit, t), "drift": drift_table(v, t), "drift_continues": drift_table(v, t, "continues")}
    out["_ce"], out["_vl"] = ce, vlogit
    return out


def value_metrics(vlogit: np.ndarray, t: dict) -> dict:
    """nets.train.evaluate's value metrics against the outcome (records with a known one)."""
    from draftzero.nets.train import auc
    vt = t["outcome"]
    m = vt >= 0
    l, y = vlogit[m], vt[m]
    p = 1 / (1 + np.exp(-l))
    dec = (y == 0) | (y == 1)
    return {"n_value": int(m.sum()), "value_logloss": float(np.mean(np.logaddexp(0, l) - y * l)),
            "value_brier": float(np.mean((p - y) ** 2)), "value_auc": auc(l[dec], y[dec])}


def paired(a: dict, b: dict, t: dict) -> dict:
    """Arm b minus the control a, per record and clustered by game: policy CE, value log loss against results, and
    the drift at the gate windows while holding (points; the same records as the drift table's)."""
    pol = ~np.isnan(a["_ce"]) & ~np.isnan(b["_ce"])
    m = t["outcome"] >= 0
    y = t["outcome"]
    g = m & gate_mask(t)

    def vl(l):
        return np.logaddexp(0, l) - y * l

    def p(l):
        return 1 / (1 + np.exp(-l))
    return {"policy_ce": clustered((b["_ce"] - a["_ce"])[pol], t["game"][pol]),
            "value_logloss": clustered((vl(b["_vl"]) - vl(a["_vl"]))[m], t["game"][m]),
            "gate_drift": clustered(((p(b["_vl"]) - p(a["_vl"])) * 100)[g], t["game"][g])}


# ------------------------------------------------------------------------------------------------ data and training

def find_pack(e: dict, caches: list[Path], store: LocalStore, out: Path, gorge_box: list, cfg) -> list[str]:
    """The pack dirs of one ingested shard: a cache's copy when it holds the shard whole (its records add up to the
    shard's positions), else OUT/packs, built with dzgorge pack (game ids prefixed as the learner does)."""
    name = f"{e['actor']}-{e['seq']:06d}"
    for c in caches:
        d = c / name
        if d.is_dir() and any(d.rglob("meta.json")):
            dirs = pack_dirs(d)
            if sum(json.loads((Path(x) / "meta.json").read_text())["n"] for x in dirs) == e["positions"]:
                return dirs
    dst = out / "packs" / name
    if not (dst / ".done").exists():
        shutil.rmtree(dst, ignore_errors=True)
        corpus = out / "corpus" / f"{name}.jsonl.gz"
        if store.get(f"{e['shard']}/corpus.jsonl.gz", corpus) is None:
            raise FileNotFoundError(f"{e['shard']}/corpus.jsonl.gz")
        if not gorge_box:
            from draftzero.engine.gorge import Gorge
            gorge_box.append(Gorge.from_config(cfg))
        gorge_box[0].pack([corpus], dst, game_prefix=f"{e['actor']}-{e['seq']}-")
        corpus.unlink(missing_ok=True)
        (dst / ".done").write_text("")
    return pack_dirs(dst)


def train_arm(name: str, blend: float, nxt: float, init: Path, window: list[str], cfg, seed: int, out: Path, a,
              dry: bool = False) -> dict | None:
    """nets.train from init on the window, with the learner's settings (as Learner.train_args); best.pt and the
    summary. A finished arm is read back when it was trained with the same arguments, window and checkpoint pick,
    and refused otherwise (a smoke run's arm is not scored as the full gate's). dry: only that check (None: to
    train)."""
    d = out / name
    lr = cfg.learner
    argv = [sys.executable, "-m", "draftzero.nets.train", "--init", str(init), "--train", *window,
            "--out", str(d / "train"), "--batch", str(int(lr.batch)), "--lr", repr(float(lr.lr)),
            "--warmup", str(int(lr.warmup)), "--policy-target", lr.policy_target, "--value-blend", repr(blend),
            *(["--value-next", repr(nxt)] if nxt > 0 else []), "--device", "cpu", "--in-ram",
            "--eval-max", str(a.eval_max), "--seed", str(seed), "--workers", str(max(1, min(2, a.threads)))]
    argv += ["--epochs", "1000000", "--max-steps", str(a.steps)] if a.steps else ["--epochs", repr(float(a.epochs))]
    i = argv.index("--train") + 1
    key = {"argv": argv[3:i] + [f"<{len(window)} pack dirs>"] + argv[i + len(window):],
           "window": hashlib.sha1("\n".join(window).encode()).hexdigest(), "checkpoint_arg": a.checkpoint}
    done = d / "summary.json"
    if done.exists():
        res = json.loads(done.read_text())
        diff = [k for k in key if res.get(k) != key[k]]
        if not diff:
            return res
        raise SystemExit(f"{name}: {d} holds an arm trained with other settings (differing: {', '.join(diff)}); "
                         "use another OUT, or remove that folder to retrain")
    if dry:
        return None
    src = str(Path(draftzero.__file__).resolve().parents[1])       # the subprocess trains with this checkout's code
    env = {**os.environ, "OMP_NUM_THREADS": str(a.threads), "MKL_NUM_THREADS": str(a.threads),
           "PYTHONPATH": os.pathsep.join([src] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])}
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    t0 = time.time()
    with open(d / "train.log", "w") as log:
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=log, text=True, env=env)
    if p.returncode:
        raise SystemExit(f"{name}: training failed (see {d / 'train.log'})")
    summary = json.loads(p.stdout.strip().splitlines()[-1])
    pick = "best" if a.checkpoint == "best" and (d / "train" / "best.pt").exists() else "last"
    res = {"arm": name, "value_blend": blend, "value_next": nxt, "steps": summary["steps"],
           "best_step": summary.get("best_step"), "checkpoint": pick, "pt": str(d / "train" / f"{pick}.pt"),
           "train_s": round(time.time() - t0, 1), "holdout": {k: (summary.get("best") or {}).get("holdout", {}).get(k)
                                                               for k in ("total", *METRICS)}, **key}
    done.write_text(json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------------------------------------ report

def fmt(r: dict | None, digits: int = 2) -> str:
    """A clustered mean as '+m (lo to hi)'."""
    if not r or r.get("mean") is None:
        return "–"
    m, lo, hi = r["mean"], r["lo"], r["hi"]
    if not math.isfinite(lo):
        return f"{m:+.{digits}f}"
    return f"{m:+.{digits}f} ({lo:+.{digits}f} to {hi:+.{digits}f})"


def summary_md(res: dict) -> str:
    rows = res["rows"]
    names = [k for k in ("search", "init", *ARMS) if k in rows]
    s = res["setup"]
    L = ["# Value-next gate", "",
         f"*{res['time']} · init `{s['init']}` · window {s['window_shards']} shards, {s['window_positions']:,} "
         f"positions · test {s['test_shards']} shards, {s['test_records']:,} records ({s['test_with_next']:,} with a "
         f"next own turn) · {s['schedule']} · seed {s['seed']}*", ""]
    if s.get("smoke"):
        L += ["**A smoke run** (--max-shards): a few shards, so its numbers and verdict mean nothing.", ""]
    if s.get("init_saw_test"):
        L += [f"The starting network trained on {s['init_saw_test']} of the {s['test_shards']} test shards "
              "(its own window), so the test set is not unseen by it; every arm shares that.", ""]
    if not res["gate"]:
        L += ["**Gate: needs the control and another arm.**", ""]
    for arm, g in res["gate"].items():
        bl, nx = ARMS[arm]
        L += [f"**Gate, {arm} (value_blend {bl:g}, value_next {nx:g}): {'pass' if g['pass'] else 'fail'}.**", "",
              f"| Check | {arm} | control | Want | Pass |", "|---|---|---|---|---|"]
        for c in g["checks"]:
            ok = ("yes" if c["pass"] else "no") + ("" if c.get("gating", True) else " (not gating)")
            L.append(f"| {c['name']} | {c['treatment']} | {c['control']} | {c['want']} | {ok} |")
        L.append("")
    gs = rows["search"]["drift"]["gate windows | holding"]["mean"]
    gi = rows["init"]["drift"]["gate windows | holding"]["mean"] if "init" in rows else None
    if gs is not None and gi is not None:
        L += [f"For scale, at the gate windows while holding: the search's root value drifts {gs:+.2f} points and "
              f"the starting network {gi:+.2f}, so the network sits {gi - gs:+.2f} above the root value it was "
              f"trained on. A network that fitted value_blend b (the rest value_next) exactly would drift about "
              f"b x {gs:+.2f}; one that also kept the starting network's bias would add {gi - gs:+.2f}.", ""]
    L += ["## Test-set metrics (against game results)", "",
          "| | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for k in METRICS:
        L.append(f"| {k} | " + " | ".join(f"{rows[n][k]:.4f}" if isinstance(rows[n].get(k), float) else "–"
                                          for n in names) + " |")
    for arm, p in res.get("paired", {}).items():
        L += ["", f"{arm} minus control, per record and clustered by game: policy CE {fmt(p['policy_ce'], 4)}, "
              f"value log loss {fmt(p['value_logloss'], 4)}, drift at the gate windows while holding "
              f"{fmt(p['gate_drift'])} points."]
    L += ["", "## Drift from the value target's next_rv (points, 95% interval)", "",
          "Value minus next_rv: the seat's root value at its next own turn, or the game's result when the game ends "
          "first (a calibrated value drifts 0 from it, by optional stopping), over records with a known result. "
          "Positive = overshoot. *search* is the root value itself; *ends first* is the share of the cell's records "
          "whose game ends before the seat's next own turn.", "",
          "| Window | Holding | n | ends first | " + " | ".join(names) + " |",
          "|---|---|---|---" + "|---" * len(names) + "|"]
    L += drift_rows(rows, names, "drift")
    L += ["", "## Drift while the game continues (points, 95% interval)", "",
          "Value minus the root value at the seat's next own turn, over the records that have one (the records "
          "whose game ends first left out).", "",
          "| Window | Holding | n | " + " | ".join(names) + " |", "|---|---|---" + "|---" * len(names) + "|"]
    L += drift_rows(rows, names, "drift_continues")
    return "\n".join(L) + "\n"


def drift_rows(rows: dict, names: list[str], kind: str) -> list[str]:
    """The drift table's lines: the gate windows, all records, then each class and holding with records."""
    out = []
    for key in ["gate windows | holding", "all"] + [f"{c} | {h}" for c in CLASSES for h in ("holding", "not holding")]:
        cell = rows[names[0]][kind].get(key)
        if not cell or not cell["n"]:
            continue
        w, _, h = key.partition(" | ")
        ends = f" {cell['ends_first']:.0%} |" if "ends_first" in cell else ""
        out.append(f"| {w} | {h or 'any'} | {cell['n']:,} |{ends} "
                   + " | ".join(fmt(rows[n][kind][key]) for n in names) + " |")
    return out


def gate(rows: dict) -> dict:
    """Each arm beside the control (none without the control): {arm: {"pass", "checks"}}."""
    co = rows.get("control")
    return {k: gate_arm(rows[k], co) for k in ARMS if k != "control" and k in rows} if co else {}


def gate_arm(tr: dict, co: dict) -> dict:
    """The gate's checks for one arm against the control. The drift share (the arm's drift at most half the
    control's) is reported, not gated on: an alternative criterion for Dan."""
    dt, dc = tr["drift"]["gate windows | holding"], co["drift"]["gate windows | holding"]
    lim, share = GATE["drift_points"], GATE["drift_share"]
    both = dt["mean"] is not None and dc["mean"] is not None
    checks = [
        {"name": "drift at the gate windows, holding (points)", "treatment": fmt(dt), "control": fmt(dc),
         "want": f"within ±{lim:g}", "pass": dt["mean"] is not None and abs(dt["mean"]) <= lim},
        {"name": "policy CE", "treatment": f"{tr['policy_ce']:.4f}", "control": f"{co['policy_ce']:.4f}",
         "want": f"at most control + {GATE['policy_ce']:g}",
         "pass": tr["policy_ce"] <= co["policy_ce"] + GATE["policy_ce"]},
        {"name": "value log loss against results", "treatment": f"{tr['value_logloss']:.4f}",
         "control": f"{co['value_logloss']:.4f}", "want": "at most control's",
         "pass": tr["value_logloss"] <= co["value_logloss"]},
        {"name": "that drift's size against the control's", "control": "1",
         "treatment": f"{abs(dt['mean']) / abs(dc['mean']):.2f}" if both and dc["mean"] else "–",
         "want": f"at most {share:g}", "pass": both and abs(dt["mean"]) <= share * abs(dc["mean"]), "gating": False},
    ]
    return {"pass": all(c["pass"] for c in checks if c.get("gating", True)), "checks": checks}


# ------------------------------------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store", help="a finished run's store folder (config.yml, learner/ingested.json, shards/)")
    ap.add_argument("out")
    ap.add_argument("--init", default=None, help="the starting model.pt (default: the store's newest version)")
    ap.add_argument("--packs", action="append", default=[],
                    help="a folder of packs named <actor>-<seq> (the learner's cache, replay_ratio.py's OUT/packs); "
                         "repeatable")
    ap.add_argument("--arms", default="control,treatment", help=f"of {', '.join(ARMS)}")
    ap.add_argument("--epochs", type=float, default=None, help="passes over the window (default: learner.epochs)")
    ap.add_argument("--steps", type=int, default=None, help="training steps instead of --epochs")
    ap.add_argument("--max-shards", type=int, default=None, help="at most N window shards and N test shards (smoke)")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--eval-max", type=int, default=30000, help="holdout records per nets.train evaluation")
    ap.add_argument("--checkpoint", choices=("best", "last"), default="best",
                    help="best: the holdout pick, as published")
    ap.add_argument("--seed", type=int, default=None, help="default: the learner's seed for the next version")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    arm_names = [x.strip() for x in a.arms.split(",") if x.strip()]
    if not arm_names or set(arm_names) - set(ARMS):
        ap.error(f"--arms: of {', '.join(ARMS)}")
    store, out = LocalStore(a.store), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = ctl.effective(C.loads((Path(a.store) / "config.yml").read_text()), store)
    a.epochs = float(cfg.learner.epochs) if a.epochs is None else a.epochs
    latest = int(store.read_json(ctl.latest_path())["version"])
    init = Path(a.init) if a.init else Path(a.store) / ctl.version_file(latest, "model.pt")
    init_v = int(init.parent.name[1:]) if init.parent.name[:1] == "v" and init.parent.name[1:].isdigit() else latest
    seed = a.seed if a.seed is not None else (int(cfg.seed) * 1_000_003 + init_v + 1) % 2**31

    entries = stream(store)
    cut = int(len(entries) * (1 - a.test_frac))
    test_e = entries[cut:][:a.max_shards] if a.max_shards else entries[cut:]
    win, n = [], 0
    for e in reversed(entries[:cut]):
        if n >= int(cfg.learner.window_positions) or (a.max_shards and len(win) >= a.max_shards):
            break
        win.append(e)
        n += e["positions"]
    win = win[::-1]
    caches, box = [Path(p) for p in a.packs], []
    dirs = {e["shard"]: find_pack(e, caches, store, out, box, cfg) for e in win + test_e}
    window = [d for e in win for d in dirs[e["shard"]]]
    imeta = json.loads((init.parent / "meta.json").read_text()) if (init.parent / "meta.json").exists() else {}
    setup = {"store": str(a.store), "init": str(init), "init_version": init_v, "seed": seed,
             "window_shards": len(win), "window_positions": n, "test_shards": len(test_e),
             "test_positions": sum(e["positions"] for e in test_e), "shards": len(entries), "cut": cut,
             "window_first": win[0]["shard"] if win else None, "window_last": win[-1]["shard"] if win else None,
             "test_first": test_e[0]["shard"] if test_e else None,
             "init_saw_test": len({e["shard"] for e in test_e} & set(imeta.get("shards", []))),
             "schedule": f"{a.steps} steps" if a.steps else f"{a.epochs:g} epochs of the window",
             "learner": {k: ctl.get(cfg, f"learner.{k}") for k in ("batch", "lr", "warmup", "policy_target",
                                                                     "window_positions", "epochs",
                                                                     "steps_per_position")},
             "gate_windows": list(GATE_WINDOWS), "threads": a.threads, "checkpoint": a.checkpoint,
             "smoke": bool(a.max_shards)}
    print(json.dumps(setup), flush=True)
    for name in arm_names:                                     # refuse a stale OUT before the scoring
        train_arm(name, *ARMS[name], init, window, cfg, seed, out, a, dry=True)

    t = test_set([d for e in test_e for d in dirs[e["shard"]]])
    setup.update(test_records=int(t["ds"].n), test_with_next=int((t["next"] >= 0).sum()),
                 test_holding=float(t["hold"].mean()) if t["ds"].n else None)
    rows = {"search": score(None, t, a.threads), "init": score(init, t, a.threads)}
    arms = {}
    for name in arm_names:
        blend, nxt = ARMS[name]
        arms[name] = train_arm(name, blend, nxt, init, window, cfg, seed, out, a)
        rows[name] = score(Path(arms[name]["pt"]), t, a.threads)
        print(json.dumps({"arm": name, **{k: rows[name].get(k) for k in METRICS},
                          "gate_drift": rows[name]["drift"]["gate windows | holding"]}), flush=True)
    pr = {k: paired(rows["control"], rows[k], t) for k in arms if k != "control"} if "control" in rows else {}
    res = {"time": time.strftime("%Y-%m-%d %H:%M"), "setup": setup, "arms": arms,
           "rows": {k: {x: y for x, y in v.items() if not x.startswith("_")} for k, v in rows.items()},
           "gate": gate(rows), "paired": pr}
    (out / "results.json").write_text(json.dumps(res, indent=1))
    (out / "summary.md").write_text(summary_md(res))
    print(json.dumps({"gate": {k: g["pass"] for k, g in res["gate"].items()}, "results": str(out / "results.json")}),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
