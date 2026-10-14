"""Offline test (research memo D1): is learning limited by games or by positions?

From E1's packed shards: the newest ~15% of shards (by learner ingest order) are the test set. From the rest:
  A = every position of a random quarter of the games
  B = a random quarter of all positions (from every game)
trimmed to the same number of positions. Both train the same width-128 MLP from scratch with the same settings;
compare held-out policy cross-entropy and value metrics on the test set. If B clearly wins, games (diversity) are the
limit, and cheaper games (playout-cap randomization) should pay.

    python games_vs_positions.py PACKS_DIR OUT_DIR [--seeds 2] [--epochs 4] [--threads 2]
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import zlib
from pathlib import Path

from draftzero.nets import packfmt as pf


def load_records(dirs):
    recs = []
    for d in dirs:
        pack, meta = pf.read_shard(d, in_ram=True)
        games = meta["games"]
        n = int(meta["n"])
        for i in range(n):
            r = pf.get_record(pack, i)
            r["_game"] = games[int(pack["game"][i])] if "game" in pack else f"{d}:{i}"
            recs.append(r)
    return recs


def write(recs, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    games = sorted({r["_game"] for r in recs})
    gi = {g: i for i, g in enumerate(games)}
    for r in recs:
        r["game"] = gi[r.pop("_game")]
    pf.write_shard(out / "00000", pf.pack_records(recs), games, source="games_vs_positions")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("packs")
    ap.add_argument("out")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    out = Path(a.out)
    dirs = sorted((m.parent for m in Path(a.packs).glob("*/*/meta.json")), key=lambda p: p.stat().st_mtime)
    cut = int(len(dirs) * 0.85)
    train_dirs, test_dirs = dirs[:cut], dirs[cut:]
    test = load_records(test_dirs)
    write(test, out / "test")
    pool = load_records(train_dirs)
    games = sorted({r["_game"] for r in pool})
    print(f"shards {len(dirs)} (train {len(train_dirs)}, test {len(test_dirs)}); pool {len(pool)} positions, "
          f"{len(games)} games; test {len(test)} positions", flush=True)
    results = []
    for seed in range(a.seeds):
        rng = random.Random(1000 + seed)
        pick = set(rng.sample(games, len(games) // 4))
        A = [dict(r) for r in pool if r["_game"] in pick]
        B = [dict(r) for r in rng.sample(pool, len(pool) // 4)]
        n = min(len(A), len(B))
        A, B = A[:n], B[:n]
        for name, recs in (("A_games", A), ("B_positions", B)):
            d = out / f"s{seed}" / name
            write(recs, d / "data")
            cfg = json.dumps({"d": 128, "ff": 256, "layers": 1, "d_emb": 32, "card_d": 64, "head_hidden": 128})
            cmd = [sys.executable, "-m", "draftzero.nets.train", "--arch", "mlp", "--config", cfg,
                   "--train", str(d / "data"), "--val", f"test={out / 'test'}", "--out", str(d / "run"),
                   "--epochs", str(a.epochs), "--lr", "1e-3", "--policy-target", "visits", "--value-blend", "1",
                   "--device", "cpu", "--in-ram", "--seed", str(seed)]
            env = {"OMP_NUM_THREADS": str(a.threads), "TORCH_THREADS": str(a.threads)}
            r = subprocess.run(cmd, capture_output=True, text=True, env={**__import__("os").environ, **env})
            if r.returncode:
                print(r.stderr[-2000:], flush=True)
                raise SystemExit(f"train failed for {name} seed {seed}")
            evals = [e for e in (json.loads(l) for l in (d / "run" / "curves.jsonl").read_text().splitlines())
                     if e.get("type") == "eval"]
            # model selection on the run's own holdout (as best.pt), never on the test set
            sel = min(evals, key=lambda e: e["eval"].get("holdout", e["eval"]["test"])["total"])
            t = sel["eval"]["test"]
            row = {"seed": seed, "arm": name, "positions": n, "games": len({r_["game"] for r_ in recs}),
                   "step": sel.get("step"),
                   **{k: t.get(k) for k in ("total", "policy_ce", "policy_ce_uniform", "top1_visits", "value_logloss",
                                            "value_auc", "value_brier")}}
            print(json.dumps(row), flush=True)
            results.append(row)
    (out / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
