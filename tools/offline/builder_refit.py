"""Offline: refit a builder run's deck ensemble on chosen versions' games, fresh or warm-started, and play its best
decks against best23@1's on the run's held-out evaluation pools exactly as its builder evaluator does (the same
pools, deals, pilot and simulations), so scores pair by deal with the run's own versions (docs/034 §10).

It reads the run's store and writes only under OUT/NAME. Run it with the run's own code on the path. --eval-ens
replays a published ensemble instead (E3's version 22, shard 0: the same 40 games as the evaluator's).

    python builder_refit.py STORE OUT NAME --versions LO-HI [--warm V] [--eval-ens PATH] [--pools-cache DIR]
                            [--shards 0,1,2,3,4] [--workers 2] [--threads 2]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from draftzero import config as C
from draftzero.loop import builder as B
from draftzero.loop import control as ctl
from draftzero.build import model as M
from draftzero.build import pools as P
from draftzero.build import selfplay as S



def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store"); ap.add_argument("out"); ap.add_argument("name")
    ap.add_argument("--pools-cache", default=None, help="a folder holding the evaluation pools (default: OUT/pools)")
    ap.add_argument("--versions", default=None); ap.add_argument("--warm", type=int, default=None)
    ap.add_argument("--eval-ens", default=None); ap.add_argument("--shards", default="0,1,2,3,4")
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    STORE, OUT = Path(a.store), Path(a.out)
    POOLS = Path(a.pools_cache) if a.pools_cache else OUT / "pools"
    cfg = C.loads((STORE / "config.yml").read_text(), {"builder.threads": a.threads})
    out = OUT / a.name
    out.mkdir(parents=True, exist_ok=True)
    report = {"name": a.name, "versions": a.versions, "warm": a.warm}

    if a.eval_ens:
        ens_path = Path(a.eval_ens)
    else:
        lo, hi = map(int, a.versions.split("-"))
        ing = json.loads((STORE / B.INGESTED).read_text())
        win = sorted((e for e in ing if e["decided"] > 0 and lo <= (e.get("version") or 0) <= hi),
                     key=lambda e: (e.get("version") or 0, str(e.get("time") or ""), e["seq"], e.get("actor") or ""))
        dirs = [STORE / e["shard"] for e in win]
        decks = S.load_decks([d / "decks.jsonl" for d in dirs])
        games = S.load_games([d / "games.jsonl" for d in dirs])
        init = M.Ensemble.load(STORE / f"versions/builder/v{a.warm:04d}/ens.pt") if a.warm is not None else None
        t0 = time.time()
        ens, rep = B.default_fit(decks, games, init=init, cfg=cfg,
                                 log=lambda s: print(s, flush=True))
        share = float(ctl.get(cfg, "builder.land_support") or 0)
        if share > 0:
            ens.meta["lands"] = M.land_support(games, decks, share)
        ens_path = out / "ens.pt"
        ens.save(ens_path)
        report.update(shards=len(win), games=len(games), decks=len(decks), fit_s=round(time.time() - t0, 1),
                      lands=ens.meta.get("lands"),
                      heldout={k: rep.get(k) for k in ("games", "logloss", "base_logloss", "acc")})
        print(json.dumps(report), flush=True)

    from draftzero.engine.gorge import Gorge
    gorge = Gorge.from_config(cfg)
    e = ctl.as_dict(ctl.get(cfg, "builder.eval"))
    seed, pairs, sims, npools = int(e["seed"]), int(e["pairs"]), int(e["sims"]), int(e["pools"])
    pools = P.cached(gorge, POOLS, **B.pool_kw(cfg, held_out=True))
    assert len(pools) == npools
    builder = B._load_builder(ens_path, a.threads)
    spec = B.pilot_spec(cfg, sims, B.fixed_pilot(cfg))
    games_all = []
    for sid in map(int, a.shards.split(",")):
        first, n = sid * B.EVAL_SHARD_POOLS, min(B.EVAL_SHARD_POOLS, npools - sid * B.EVAL_SHARD_POOLS)
        sd = out / f"eval-{sid:04d}"
        shutil.rmtree(sd, ignore_errors=True); sd.mkdir(parents=True)
        lines = []
        for k in range(first, first + n):
            pool = pools[k]
            mine = builder.best(P.pool_vector(pool), np.random.default_rng([seed, k]))
            lines.append(B.deck_line(f"{B.TAG}{k}", mine, pool=pool["pool_id"], kind="b"))
            lines.append(B.deck_line(f"{B.OPP}{k}", P.deck_vector(pool["best23"]["cards"]), pool=pool["pool_id"],
                                     kind="best23"))
        (sd / "decks.jsonl").write_text("".join(x + "\n" for x in lines))
        (sd / "pairs.tsv").write_text("".join(f"{x}\t{y}\t{s}\n" for x, y, s in B.match_lines(first, n, pairs, seed)))
        t0 = time.time()
        res = gorge.play(spec, spec, pairs=0, seed=seed, out_dir=sd / "play", workers=a.workers,
                         deckfile=sd / "decks.jsonl",
                         pairs_file=sd / "pairs.tsv", single=True, belief=B.pilot_belief(cfg), timeout_s=3600)
        rows = [{**g, "shard": sid} for g in res.games]
        (sd / "games.jsonl").write_text("".join(json.dumps(g, sort_keys=True) + "\n" for g in rows))
        games_all += rows
        print(f"shard {sid}: {len(rows)} games in {time.time() - t0:.0f} s", flush=True)
    s = B.score_match(games_all)
    report["eval"] = s
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: s[k] for k in ("score", "lo", "hi", "n_games")} | {"name": a.name}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
