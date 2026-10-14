"""Offline A/B of the trainer's opt-in KataGo extras (research memo B, D5, D8) on E1's packed shards.

The same split as games_vs_positions.py: the learner's pack dirs sorted by mtime (learner ingest order), the newest
15% are the test set, the older 85% the training set (its crc32 game holdout picks the model). The shards are first
copied into OUT/data (the live learner prunes its packs), so later runs and seeds see the same data. Arms, each from
scratch with identical settings (configs/base/player.yml's width-128 MLP, 4 epochs, lr 1e-3, visit targets, value
target = root value, the same seed per seed index):

  baseline         nothing extra
  surprise         --surprise-weight 0.5
  surprise_value   --surprise-weight 0.5 --value-surprise 0.1
  ema              --ema-decay 0.999 --export-ema

Each run's model is the one its holdout picks (for ema, raw weights and moving average compete, as --export-ema
does); its test-set metrics are reported. OUT/results.json holds a row per arm and seed; a finished run is not rerun.

    python trainer_ab.py PACKS_DIR OUT_DIR [--seeds 2] [--epochs 4] [--threads 2] [--cpus 4,5] [--arms a,b]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from draftzero.nets import packfmt as pf

ARMS = {
    "baseline": [],
    "surprise": ["--surprise-weight", "0.5"],
    "surprise_value": ["--surprise-weight", "0.5", "--value-surprise", "0.1"],
    "ema": ["--ema-decay", "0.999", "--export-ema"],
}
METRICS = ("total", "policy_ce", "policy_ce_uniform", "top1_visits", "value_logloss", "value_auc", "value_brier")


def snapshot(packs: Path, data: Path) -> dict:
    """Copy the pack dirs (oldest first by mtime) into data/all/NNNNN; link the split into data/train and
    data/test. A shard the learner deletes mid-copy is dropped. Returns the split record (also data/split.json)."""
    rec = data / "split.json"
    if rec.exists():
        return json.loads(rec.read_text())
    dirs = sorted((m.parent for m in packs.glob("*/*/meta.json")), key=lambda p: p.stat().st_mtime)
    tmp = data / "all.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    kept = []
    for d in dirs:
        dst = tmp / f"{len(kept):05d}"
        try:
            mtime = d.stat().st_mtime
            shutil.copytree(d, dst)
            n = int(json.loads((dst / "meta.json").read_text())["n"])
            pack, _ = pf.read_shard(dst)                 # every array present and consistent with meta.json
            if len(pack["outcome"]) != n or len(pack["cand_off"]) != n + 1:
                raise ValueError(f"{len(pack['outcome'])} records, meta says {n}")
        except (OSError, ValueError, KeyError) as e:
            print(f"skip {d}: {e}", flush=True)
            shutil.rmtree(dst, ignore_errors=True)
            continue
        kept.append({"src": str(d), "mtime": mtime, "n": n, "dir": dst.name})
    shutil.rmtree(data / "all", ignore_errors=True)
    tmp.rename(data / "all")
    cut = int(len(kept) * 0.85)
    for name, part in (("train", kept[:cut]), ("test", kept[cut:])):
        (data / name).mkdir(exist_ok=True)
        for k in part:
            link = data / name / k["dir"]
            if not link.exists():
                link.symlink_to(Path("..") / "all" / k["dir"])
    out = {"packs": str(packs), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "shards": kept,
           "train": [k["dir"] for k in kept[:cut]], "test": [k["dir"] for k in kept[cut:]],
           "train_positions": sum(k["n"] for k in kept[:cut]), "test_positions": sum(k["n"] for k in kept[cut:])}
    rec.write_text(json.dumps(out, indent=1))
    return out


def player_mlp(repo: Path) -> dict:
    from draftzero import config
    return dict(config.load(repo / "configs" / "base" / "player.yml").player.config)


def select(curves: Path, ema: bool) -> dict:
    """The evaluation the holdout picks: (weights, step, holdout total, test metrics). With ema, the moving average
    is a candidate at each evaluation too (the raw weights win ties), as --export-ema decides best.pt."""
    evals = [e for e in (json.loads(x) for x in curves.read_text().splitlines()) if e.get("type") == "eval"]
    cands = []
    for e in evals:
        cands.append((e["eval"]["holdout"]["total"], 0, "raw", e["step"], e["eval"]))
        if ema and e.get("ema"):
            cands.append((e["ema"]["holdout"]["total"], 1, "ema", e["step"], e["ema"]))
    best = min((c for c in cands if c[0] is not None), key=lambda c: (c[0], c[3], c[1]))
    return {"weights": best[2], "step": best[3], "holdout_total": best[0], "test": best[4]["test"]}


def run_arm(arm: str, seed: int, a, data: Path, cfg: dict, out: Path) -> dict:
    d = out / arm / f"s{seed}"
    row_path = d / "row.json"
    if row_path.exists():
        return json.loads(row_path.read_text())
    shutil.rmtree(d / "run", ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "draftzero.nets.train", "--arch", "mlp", "--config", json.dumps(cfg),
           "--train", str(data / "train"), "--val", f"test={data / 'test'}", "--out", str(d / "run"),
           "--epochs", str(a.epochs), "--lr", "1e-3", "--policy-target", "visits", "--value-blend", "1",
           "--device", "cpu", "--in-ram", "--workers", "1", "--seed", str(seed), *ARMS[arm]]
    if a.cpus:
        cmd = ["taskset", "-c", a.cpus, *cmd]
    env = {**os.environ, "OMP_NUM_THREADS": str(a.threads), "MKL_NUM_THREADS": str(a.threads)}
    t0 = time.time()
    with open(d / "train.log", "w") as log:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=log, text=True, env=env)
    if r.returncode:
        raise SystemExit(f"train failed for {arm} seed {seed}: see {d / 'train.log'}")
    summary = json.loads(r.stdout.strip().splitlines()[-1])
    sel = select(d / "run" / "curves.jsonl", ema="--ema-decay" in ARMS[arm])
    t = sel["test"]
    row = {"arm": arm, "seed": seed, "weights": sel["weights"], "step": sel["step"], "steps": summary["steps"],
           "holdout_total": sel["holdout_total"], "wall_s": round(time.time() - t0, 1),
           **{k: t.get(k) for k in METRICS}}
    if "--ema-decay" in ARMS[arm]:      # the same run judged on its raw weights alone and on its average alone
        raw = select(d / "run" / "curves.jsonl", ema=False)
        evals = [e for e in (json.loads(x) for x in (d / "run" / "curves.jsonl").read_text().splitlines())
                 if e.get("type") == "eval" and e.get("ema")]
        e = min(evals, key=lambda e: e["ema"]["holdout"]["total"])
        row["raw_only"] = {"step": raw["step"], **{k: raw["test"].get(k) for k in METRICS}}
        row["ema_only"] = {"step": e["step"], **{k: e["ema"]["test"].get(k) for k in METRICS}}
    row_path.write_text(json.dumps(row, indent=1))
    print(json.dumps(row), flush=True)
    return row


def breakdown(pt: Path, test: Path, threads: int) -> dict:
    """The checkpoint on the test set, split where the extras should matter: policy CE and top-1 on the 10% of
    states whose policy surprise KL(visits || prior) is highest (a property of the data, the same states for every
    model), on priority (casting) decisions and on the rest; the value against the root value it was trained on
    (log loss and squared error)."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from draftzero.nets import data as D, models as M, targets as T, train
    torch.set_num_threads(threads)
    model, _ = M.load(pt)
    acc = {k: [] for k in ("ce", "top1", "kl", "prio", "vl", "rv")}
    with torch.inference_mode():
        for d in sorted(test.iterdir()):
            pack, meta = pf.read_shard(d)
            kinds = np.array(meta.get("kinds") or [str(i) for i in range(256)])
            for s in range(0, int(meta["n"]), 1024):
                idx = np.arange(s, min(s + 1024, int(meta["n"])))
                b = D.collate(pack, idx)
                scores, vlogit = model(b)
                z, lp = M.candidate_logits(scores, b)
                tgt = T.visits_target(b)
                pm = T.policy_mask(b)
                ce = train.seg_sum(-(tgt * lp), b.cand_state, b.B)
                hit = train.seg_argmax(z, b.cand_state, b.B) == train.seg_argmax(b.cand_visits.float(), b.cand_state, b.B)
                kl = T.policy_surprise(b, tgt)
                acc["ce"].append(ce[pm].numpy())
                acc["top1"].append(hit[pm].numpy())
                acc["kl"].append(kl[pm].numpy())
                acc["prio"].append(kinds[np.asarray(pack["kind"])[idx]][pm.numpy()] == "priority")
                acc["vl"].append(vlogit.float().numpy())
                acc["rv"].append(b.root_value.float().numpy())
    a = {k: np.concatenate(v) for k, v in acc.items()}
    top = a["kl"] >= np.quantile(a["kl"], 0.9)
    vl, rv = torch.tensor(a["vl"], dtype=torch.float64), torch.tensor(a["rv"], dtype=torch.float64)
    def mean(x):
        return float(x.mean()) if len(x) else None
    return {"policy_ce": mean(a["ce"]), "policy_ce_top10_surprise": mean(a["ce"][top]),
            "top1_top10_surprise": mean(a["top1"][top]), "policy_ce_rest": mean(a["ce"][~top]),
            "policy_ce_priority": mean(a["ce"][a["prio"]]), "policy_ce_not_priority": mean(a["ce"][~a["prio"]]),
            "value_logloss_root": float(F.binary_cross_entropy_with_logits(vl, rv)),
            "value_mse_root": float(((torch.sigmoid(vl) - rv) ** 2).mean()), "n": int(len(a["ce"]))}


def table(rows: list[dict]) -> str:
    keys = ("total", "policy_ce", "top1_visits", "value_logloss", "value_auc", "value_brier")
    lines = ["| arm | seed | picked | " + " | ".join(keys) + " |", "|---|---|---|" + "---|" * len(keys)]
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        for r in rs:
            lines.append(f"| {arm} | {r['seed']} | {r['weights']}@{r['step']} | "
                         + " | ".join(f"{r[k]:.4f}" for k in keys) + " |")
        lines.append(f"| {arm} | mean | | " + " | ".join(f"{sum(r[k] for r in rs) / len(rs):.4f}" for k in keys) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("packs")
    ap.add_argument("out")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cpus", default=None, help="taskset CPU list for the training processes (e.g. 4,5)")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--breakdown", action="store_true",
                    help="after the runs, score each run's best.pt on the test set by surprise, kind and root value")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2] / "wt-trainer"),
                    help="checkout whose configs/base/player.yml gives the MLP config")
    a = ap.parse_args(argv)
    out = Path(a.out)
    split = snapshot(Path(a.packs), out / "data")
    print(f"shards {len(split['shards'])} (train {len(split['train'])}, test {len(split['test'])}); "
          f"train {split['train_positions']} positions, test {split['test_positions']}", flush=True)
    cfg = player_mlp(Path(a.repo))
    rows = []

    def save():
        (out / "results.json").write_text(json.dumps({"split": {k: v for k, v in split.items() if k != "shards"},
                                                      "config": cfg, "epochs": a.epochs, "rows": rows}, indent=1))
    for seed in range(a.seeds):
        for arm in a.arms.split(","):
            rows.append(run_arm(arm, seed, a, out / "data", cfg, out))
            save()
    print(table(rows), flush=True)
    if a.breakdown:
        if a.cpus:
            os.sched_setaffinity(0, {int(c) for c in a.cpus.split(",")})
        for r in rows:
            r["breakdown"] = breakdown(out / r["arm"] / f"s{r['seed']}" / "run" / "best.pt", out / "data" / "test",
                                       a.threads)
            print(json.dumps({"arm": r["arm"], "seed": r["seed"], **r["breakdown"]}), flush=True)
            save()


if __name__ == "__main__":
    main()
