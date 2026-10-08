"""docs/028's offline arms: train one recipe on a fixed set of self-play games and score it as it trains.

Every arm reads the same games (batch files from draftzero.selfplay.records.pack) and the same held-out games, so
arms differ only in their recipe. Each arm is the spec's `base` plus its own overrides:

    base:
      start: runs/gnn/full_r1/best_policy_calibrated.pt.gz   # the network the games came from (and the KL anchor)
      td_lambda: 0.99          # the value target: MageZero's TD labels (1.0: the game result alone)
      heldout_share: 0.1       # games kept out (hashed by deck pair, the same for every arm)
      epochs: 2                # passes over the training rows (or steps:)
      eval_every: 400          # steps between checks
      lr_schedule: cosine      # constant | cosine (to lr_min_frac of lr over the run)
      lr_min_frac: 0.1
      select: policy_ce        # the checkpoint kept as best: policy_ce | value_logloss | final
      human_val: {config: configs/gnn_full_r1.yml, tables_dir: data/imitation_graph/slim, every: 2}
      trainer: {...}           # GnnTrainer's settings (draftzero.selfplay.config's trainer keys)
      phases:                  # optional: run in phases, each a share of the steps with its own trainer overrides
        - {share: 0.25, td_lambda: 1.0, trainer: {policy_weight: 0, train_only: [value_head]}}
        - {share: 0.75}
    arms:
      lam95: {td_lambda: 0.95}

    python tools/selfplay_transition/sp_train.py --spec configs/sp_arms.yml --arm lam95 \
        --batches runs/sp/batches/v0 --out runs/sp/arms/lam95

Writes <out>/metrics.jsonl (one line per check: training losses, held-out self-play checks, human validation),
summary.json, best.pt.gz and final.pt.gz. Rows are cached per batch file under <batches>/rows/.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from draftzero.selfplay import config as sc  # noqa: E402
from draftzero.selfplay import tables  # noqa: E402

BASE = {
    "start": None, "td_lambda": 0.99, "heldout_share": 0.1, "epochs": 2.0, "steps": None, "eval_every": 400,
    "lr_schedule": "cosine", "lr_min_frac": 0.1, "select": "policy_ce", "human_val": None, "trainer": {},
    "phases": None, "max_games": None,
}


def merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def arm_config(spec: dict, arm: str) -> dict:
    arms = spec.get("arms") or {}
    if arm not in arms:
        raise SystemExit(f"no arm {arm!r} in the spec (arms: {sorted(arms)})")
    cfg = merge(merge(BASE, spec.get("base") or {}), arms[arm] or {})
    unknown = set(cfg) - set(BASE)
    if unknown:
        raise SystemExit(f"unknown settings {sorted(unknown)}")
    return cfg


def retd(rows: dict, lam: float) -> np.ndarray:
    """Each row's value target recomputed at `lam` from the rows' own root values (tables.td_targets per seat-game;
    a game without a result keeps NaN)."""
    g, q, z = rows["game"], rows["q_root"].astype(np.float64), rows["z"]
    out = np.full(len(g), np.nan, np.float32)
    if not len(g):
        return out
    start = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1, len(g)]
    for a, b in zip(start[:-1], start[1:]):
        if np.isfinite(z[a]):
            out[a:b] = tables.td_targets(q[a:b], float(z[a]), lam)
    return out


def load_rows(batch_dirs: list[Path], heldout_share: float, max_games: int | None, log) -> dict:
    files = sorted(f for d in batch_dirs for f in (Path(d).glob("*.jsonl.gz") if Path(d).is_dir() else [Path(d)]))
    if not files:
        raise SystemExit(f"no batch files in {batch_dirs}")
    parts = []
    for f in files:
        cache = f.parent / "rows" / f"{f.name.removesuffix('.jsonl.gz')}-h{heldout_share:g}.npz"
        if cache.exists() and cache.stat().st_mtime >= f.stat().st_mtime:
            parts.append(tables.load(cache))
        else:
            r = tables.graph_rows(f, lam=0.99, heldout_share=heldout_share)
            tables.save(r, cache)
            parts.append(r)
        log(f"sp_train: {f.name}: {tables.n_rows(parts[-1])} rows")
    rows = tables.concat(parts)
    if max_games:
        keep_games = np.unique(rows["game"])[:int(max_games)]
        rows = tables.select(rows, np.isin(rows["game"], keep_games))
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--batches", type=Path, nargs="+", required=True, help="batch files or directories of them")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    import torch
    from draftzero.gameplay import graph_supervised as gs
    from draftzero.gameplay import supervised as sv
    from draftzero.selfplay import gnn_train

    spec = yaml.safe_load(a.spec.read_text())
    cfg = arm_config(spec, a.arm)
    a.out.mkdir(parents=True, exist_ok=True)
    logf = open(a.out / "train.log", "a")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    (a.out / "config.json").write_text(json.dumps({"arm": a.arm, **cfg, "batches": [str(b) for b in a.batches]},
                                                  indent=1))
    tcfg = sc.load(None, start=str(cfg["start"]), trainer=merge(cfg["trainer"], {"device": a.device}))["trainer"]
    start = sc.path(cfg["start"])

    rows = load_rows(a.batches, float(cfg["heldout_share"]), cfg["max_games"], log)
    rows["z_td"] = retd(rows, float(cfg["td_lambda"]))
    held = tables.select(rows, rows["heldout"])
    n_games = len(np.unique(rows["game"]))
    log(f"sp_train: {a.arm}: {tables.n_rows(rows)} rows from {n_games} seat-games, "
        f"{tables.n_rows(held)} held out; td_lambda {cfg['td_lambda']}")

    tr = gnn_train.GnnTrainer(start, start, tcfg, log=log)
    n = tr.prepare(rows)
    bs = int(tcfg["batch_rows"])
    total = int(cfg["steps"]) if cfg["steps"] else max(1, math.ceil(float(cfg["epochs"]) * n / bs))
    phases = cfg["phases"] or [{"share": 1.0}]
    shares = np.asarray([float(p.get("share", 1.0)) for p in phases])
    phase_steps = np.maximum(1, np.round(total * shares / shares.sum()).astype(int))
    log(f"sp_train: {n} training rows, batch {bs}, {int(phase_steps.sum())} steps in {len(phases)} phase(s)")

    hv = None
    hcfg = cfg["human_val"]
    if hcfg:
        gcfg = gs.resolve_config(sv.load_config_file(sc.path(hcfg["config"])),
                                 {"tables_dir": str(sc.path(hcfg["tables_dir"])), "device": a.device})
        hv = (gcfg, gs.load_data(gcfg, vocabs=(tr.vocab, tr.edge_vocab), splits=("val",), log=log).val)
    hv_every = int((hcfg or {}).get("every", 1))

    def human_metrics() -> dict:
        res = gs.evaluate(tr.model, hv[1], hv[0], tr.dev, tr.dtype)
        keep = ("policy/set_nll", "policy/top1_nonpass", "policy/pass_top1", "binary/acc", "opp_block/top1",
                "replay_target/top1", "value/auc", "value/logloss", "value/ece")
        flat = sv._jsonable(res)
        return {k: flat.get(k) for k in keep if k in flat}

    best_key = cfg["select"]
    best, best_val = None, None
    mf = open(a.out / "metrics.jsonl", "a")
    checks = 0

    def check(phase: int, train_out: dict | None) -> None:
        nonlocal best, best_val, checks
        rec = {"step": tr.step, "phase": phase, "epoch": round(tr.step * bs / max(1, n), 3), "time": time.time()}
        if train_out:
            rec["train"] = train_out
        rec["heldout"] = tr.evaluate(held, heldout_only=False)
        if hv is not None and (checks % hv_every == 0):
            rec["human"] = human_metrics()
        checks += 1
        mf.write(json.dumps(rec) + "\n")
        mf.flush()
        h = rec["heldout"]
        log(f"sp_train: step {tr.step} (epoch {rec['epoch']}): held-out ce {h.get('ce')} agree {h.get('agree_top1')} "
            f"kl {h.get('kl_start')} ent {h.get('entropy')} value auc {h.get('value_auc')} ll {h.get('value_logloss')}"
            + (f"; human nll {rec['human'].get('policy/set_nll')}" if "human" in rec else ""))
        score = {"policy_ce": h.get("ce"), "value_logloss": h.get("value_logloss")}.get(best_key)
        if best_key != "final" and tr.step > 0 and score is not None and (best_val is None or score < best_val):
            best_val, best = score, tr.step
            tr.save_weights(a.out / "best.pt.gz", {"arm": a.arm, "step": tr.step, best_key: score})

    check(0, None)
    t0 = time.time()
    for pi, (ph, ps) in enumerate(zip(phases, phase_steps)):
        tr.cfg = merge(tcfg, ph.get("trainer") or {})
        if ph.get("td_lambda") is not None:      # this phase's value target (e.g. the result first, then TD)
            rows["z_td"] = retd(rows, float(ph["td_lambda"]))
            tr.prepare(rows)
            log(f"sp_train: phase {pi}: td_lambda {ph['td_lambda']}")
        if "train_only" in (ph.get("trainer") or {}) or pi > 0:
            tr.set_trainable(tr.cfg.get("train_only"))
        done = 0
        sched = cfg["lr_schedule"]
        lo = float(cfg["lr_min_frac"])
        tr.lr_fn = (lambda i, d0=0, P=int(ps): 1.0) if sched == "constant" else None
        while done < ps:
            k = int(min(cfg["eval_every"], ps - done))
            if sched == "cosine":
                tr.lr_fn = (lambda i, d0=done, P=int(ps): lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * (d0 + i) / P)))
            out = tr.run_steps(k)
            done += k
            check(pi, out)
    tr.save_weights(a.out / "final.pt.gz", {"arm": a.arm, "step": tr.step})
    summary = {"arm": a.arm, "steps": tr.step, "rows": n, "seat_games": n_games, "best_step": best,
               "best_" + best_key: best_val, "minutes": round((time.time() - t0) / 60, 1)}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"sp_train: done: {summary}")


if __name__ == "__main__":
    main()
