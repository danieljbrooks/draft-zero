"""Offline: how many training steps per new position suit the loop's data stream? (docs/034 §8, item 7)

E1 trained each position about 27 times over its life (2 epochs of a 300,000-position window per version, ~20,000
new positions a version); KataGo trains 3-4. E1's value stopped improving at version 13 while its policy kept
learning, which too much replay would explain. This replays a finished run's shards, in the learner's ingest order,
through the learner's own schedule at several replay ratios R (`learner.steps_per_position`), and with E1's own
schedule, "e2" (`learner.epochs: 2`: 2 epochs of the window a version, so R grows from ~2 to ~30 as it fills):

- a version every 20,000 new positions (`learner.min_new_positions`) on the newest 300,000 (`window_positions`);
- ceil(R x new positions / 512) steps a version, warm-started from the previous version (random init at version 1,
  the run's `player.config`), with the learner's other settings (lr, warmup, visit targets, value blend, in RAM);
- the version is the checkpoint its holdout picks (best.pt), as the learner publishes.

The newest 15% of the run's shards (by ingest order) never enter the stream: they are the test set, scored at every
evaluation, so OUT/results.json has each version's test metrics for each R. Packs come from the learner's cache when
it still holds them, else are rebuilt from the store's corpus with dzgorge pack.

    python replay_ratio.py STORE OUT [--ratios 2,4,8,e2] [--threads 2] [--packs CACHE] [--max-versions N]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from draftzero import config as C
from draftzero.loop import control as ctl
from draftzero.loop.store import LocalStore

KEYS = ("total", "policy_ce", "policy_ce_uniform", "top1_visits", "value_logloss", "value_auc", "value_brier")


def stream(store: LocalStore) -> list[dict]:
    """The run's shards in the learner's async ingest order (version, time, seq, actor), empty ones dropped."""
    ing = store.read_json("learner/ingested.json") or []
    key = lambda e: (e.get("version") or 0, str(e.get("time") or ""), e["seq"], e.get("actor") or "")  # noqa: E731
    return sorted((e for e in ing if e.get("positions", 0) > 0), key=key)


def packs_for(e: dict, store: LocalStore, out: Path, cache: Path | None, gorge) -> Path:
    name = f"{e['actor']}-{e['seq']:06d}"
    dst = out / "packs" / name
    if (dst / ".done").exists():
        return dst
    shutil.rmtree(dst, ignore_errors=True)
    if cache is not None and (cache / name).is_dir() and any((cache / name).rglob("meta.json")):
        shutil.copytree(cache / name, dst)
    else:
        corpus = out / "corpus" / f"{name}.jsonl.gz"
        if store.get(f"{e['shard']}/corpus.jsonl.gz", corpus) is None:
            raise FileNotFoundError(f"{e['shard']}/corpus.jsonl.gz")
        gorge.pack([corpus], dst, game_prefix=f"{e['actor']}-{e['seq']}-")
        corpus.unlink(missing_ok=True)
    (dst / ".done").write_text("")
    return dst


def pack_dirs(d: Path) -> list[str]:
    return sorted(str(m.parent) for m in d.rglob("meta.json"))


def test_metrics(curves: Path, step) -> dict:
    """The test set's metrics at the evaluation the holdout picked (the last one when it picked none)."""
    evals = [e for e in (json.loads(x) for x in curves.read_text().splitlines()) if e.get("type") == "eval"]
    pick = [e for e in evals if e.get("step") == step] or evals[-1:]
    ev = pick[0]["eval"] if pick else {}
    return {"step": pick[0].get("step") if pick else None,
            "holdout": {k: ev.get("holdout", {}).get(k) for k in KEYS},
            "test": {k: ev.get("test", {}).get(k) for k in KEYS}}


def run_ratio(spec: str, entries: list[dict], dirs: dict, cfg, out: Path, a) -> list[dict]:
    """spec: a replay ratio ("4": steps_per_position 4) or "e<E>" (E epochs of the window a version)."""
    epochs = float(spec[1:]) if spec.startswith("e") else None
    r = None if epochs else float(spec)
    blend = float(cfg.learner.value_blend) if a.value_blend is None else a.value_blend
    net = {**ctl.as_dict(cfg.player.config), **json.loads(a.player_config or "{}")}
    rd = out / ((f"E{epochs:g}" if epochs else f"R{r:g}") + ("" if a.value_blend is None else f"-vb{blend:g}")
                + ("" if not a.player_config else "-" + "-".join(f"{k}{v}" for k, v in json.loads(a.player_config).items())))
    rd.mkdir(parents=True, exist_ok=True)
    rows_path = rd / "rows.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {row["version"] for row in rows}
    batch, cap, every = int(cfg.learner.batch), int(cfg.learner.window_positions), int(cfg.learner.min_new_positions)
    seen, new, version, prev = [], 0, 0, None
    for e in entries:
        seen.append(e)
        new += e["positions"]
        if new < every:
            continue
        version += 1
        if a.max_versions and version > a.max_versions:
            break
        vd = rd / f"v{version:04d}"
        win, n = [], 0
        for x in reversed(seen):
            win.append(x)
            n += x["positions"]
            if n >= cap:
                break
        steps = max(1, math.ceil((epochs * n if epochs else r * new) / batch))
        if version not in done:
            argv = [sys.executable, "-m", "draftzero.nets.train", "--out", str(vd / "train"),
                    "--train", *[p for x in reversed(win) for p in dirs[x["shard"]]],
                    "--val", f"test={out / 'test'}", "--batch", str(batch), "--lr", repr(float(cfg.learner.lr)),
                    "--warmup", str(int(cfg.learner.warmup)), "--policy-target", cfg.learner.policy_target,
                    "--value-blend", repr(blend), "--device", "cpu", "--in-ram",
                    "--eval-max", str(a.eval_max), "--seed", str(1_000_003 + version), "--epochs", "1000000",
                    "--max-steps", str(steps)]
            argv += ["--init", str(prev)] if prev else ["--arch", cfg.player.arch,
                                                        "--config", json.dumps(net)]
            env = {**os.environ, "OMP_NUM_THREADS": str(a.threads), "MKL_NUM_THREADS": str(a.threads)}
            t0 = time.time()
            shutil.rmtree(vd, ignore_errors=True)
            vd.mkdir(parents=True)
            with open(vd / "train.log", "w") as log:
                p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=log, text=True, env=env)
            if p.returncode:
                raise SystemExit(f"{spec} version {version}: training failed (see {vd / 'train.log'})")
            summary = json.loads(p.stdout.strip().splitlines()[-1])
            row = {"ratio": spec, "value_blend": blend, "version": version, "new_positions": new,
                   "window_positions": n,
                   "window_shards": len(win), "steps": steps, "wall_s": round(time.time() - t0, 1),
                   **test_metrics(vd / "train" / "curves.jsonl", summary.get("best_step"))}
            rows.append(row)
            rows_path.write_text(json.dumps(rows, indent=1))
            print(json.dumps({k: row[k] for k in ("ratio", "version", "steps", "wall_s")} | {"test": row["test"]}),
                  flush=True)
        best = vd / "train" / "best.pt"
        prev = best if best.exists() else vd / "train" / "last.pt"
        new = 0
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store", help="a finished run's store folder (learner/ingested.json and shards/)")
    ap.add_argument("out")
    ap.add_argument("--ratios", default="2,4,8,e2")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--packs", default=None, help="the learner's pack cache (reused when it holds a shard)")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--eval-max", type=int, default=30000, help="records per evaluation set")
    ap.add_argument("--max-versions", type=int, default=0, help="stop each ratio after this many versions (0: all)")
    ap.add_argument("--player-config", default=None,
                    help='network overrides for the random init, e.g. \'{"d": 256, "ff": 512}\' (default: the run\'s)')
    ap.add_argument("--value-blend", type=float, default=None,
                    help="value target (1-x) outcome + x root value (default: the run's learner.value_blend)")
    a = ap.parse_args(argv)
    store, out = LocalStore(a.store), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = C.loads((Path(a.store) / "config.yml").read_text())
    from draftzero.engine.gorge import Gorge
    gorge = Gorge.from_config(cfg)
    entries = stream(store)
    cut = int(len(entries) * (1 - a.test_frac))
    cache = Path(a.packs) if a.packs else None
    dirs = {}
    for i, e in enumerate(entries):
        dirs[e["shard"]] = pack_dirs(packs_for(e, store, out, cache, gorge))
        if i % 50 == 0:
            print(f"packs {i + 1}/{len(entries)}", flush=True)
    test = out / "test"
    test.mkdir(exist_ok=True)
    for e in entries[cut:]:
        for d in dirs[e["shard"]]:
            link = test / Path(d).relative_to(out / "packs").as_posix().replace("/", "__")
            if not link.exists():
                link.symlink_to(d)
    split = {"shards": len(entries), "stream": cut, "test": len(entries) - cut,
             "stream_positions": sum(e["positions"] for e in entries[:cut]),
             "test_positions": sum(e["positions"] for e in entries[cut:])}
    print(json.dumps(split), flush=True)
    results = {"split": split, "rows": []}
    for spec in a.ratios.split(","):
        results["rows"] += run_ratio(spec.strip(), entries[:cut], dirs, cfg, out, a)
        tag = ("" if a.value_blend is None else f"-vb{a.value_blend:g}") + ("" if not a.player_config else "-net")
        (out / f"results{tag}.json").write_text(
            json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
