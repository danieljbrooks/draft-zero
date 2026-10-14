"""
Sealed pools from gorge's draftlab, for training and evaluating the deck builder without 17lands data.

`dzgorge sealed-pools` (the gorge fork, cmd/dzgorge/sealed.go) opens `packs` boosters of a set per pool with
draftlab.SealedPool; pool i of base seed S comes from rand.NewPCG(S, 0xD25E01<<32|i), so a pool is a pure function of
(set, collation, S, i, packs) and every machine that asks for the same pools gets the same ones. One JSON line a pool:

    {"pool_id": "FDN-1-000012", "index": 12, "set": "FDN", "collation": "play-arena", "packs": 6, "base_seed": 1,
     "seed": <the pool's own seed>, "cards": {name: count, ...},              84 cards for six FDN Play Boosters
     "best23": {"builder": "best23@1", "cards": [{"name", "count"}, ...]}}     with -best23: gorge's 40-card deck

best23 ranks cards by 17lands' games-in-hand win rates (limited/pickbot), so its decks are an evaluation opponent,
never training data.

    cached(gorge, cache_dir, set="FDN", n=5000, seed=1)   the pools, generated once into cache_dir
    pool_vector(pool)                                       its count vector over cards.load() (basics set aside)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

COLLATIONS = ("arena", "play")
TIMEOUT_S = 1800.0


def argv(binary: Path, cards: Path, out: Path, *, set: str = "FDN", n: int, seed: int, packs: int = 6,
         collation: str = "arena", best23: bool = False, first: int = 0) -> list[str]:
    """The `dzgorge sealed-pools` command line."""
    if collation not in COLLATIONS:
        raise ValueError(f"collation {collation!r}: want one of {COLLATIONS}")
    a = [str(binary), "sealed-pools", "-cards", str(cards), "-set", set, "-n", str(int(n)), "-first",
         str(int(first)), "-seed", str(int(seed)), "-packs", str(int(packs)), "-collation", collation, "-out", str(out)]
    return a + (["-best23"] if best23 else [])


def generate(gorge, out: Path, *, timeout_s: float = TIMEOUT_S, **kw) -> Path:
    """Run `dzgorge sealed-pools` (gorge: a draftzero.engine.Gorge, for its binary and card corpus) into `out`."""
    from draftzero.engine.gorge import GorgeError, run_killable
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    err = out.with_name(out.name + ".err")
    with open(err, "wb") as f:
        rc, killed, _ = run_killable(argv(gorge.binary, gorge.cards, out, **kw), timeout_s=timeout_s, stderr=f)
    msg = err.read_text(errors="replace")
    err.unlink(missing_ok=True)
    if rc != 0 or killed or not out.exists():
        hint = " (a dzgorge without sealed-pools? it needs the gorge fork's draftzero-sealed commit)" \
            if "usage:" in msg else ""
        raise GorgeError(f"dzgorge sealed-pools {'timed out' if killed else f'exited {rc}'}{hint}: {msg[-2000:]}")
    return out


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def cache_name(*, set: str = "FDN", n: int, seed: int, packs: int = 6, collation: str = "arena",
               best23: bool = False) -> str:
    return f"{set}-{collation}-p{packs}-s{seed}-n{n}" + ("-best23" if best23 else "") + ".jsonl"


def cached(gorge, cache_dir: Path, *, make=None, **kw) -> list[dict]:
    """The pools `kw` names (argv's keywords), from cache_dir, generating them first if missing (written under a
    temporary name and renamed, so a killed run leaves no partial cache). `make(out, **kw)` replaces generate."""
    path = Path(cache_dir) / cache_name(**kw)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        (make or (lambda out, **k: generate(gorge, out, **k)))(tmp, **kw)
        os.replace(tmp, path)
    pools = read(path)
    if len(pools) != kw["n"]:
        raise ValueError(f"{path}: {len(pools)} pools, want {kw['n']} (delete it to regenerate)")
    return pools


def pool_vector(pool: dict) -> np.ndarray:
    """A pool's count vector over cards.load(), basic lands set aside (they are unlimited). A card outside the
    table is an error: the deck network has no row for it."""
    from . import cards as cardsmod
    C = cardsmod.load()
    missing = sorted(c for c in pool["cards"] if c not in C.index)
    if missing:
        raise KeyError(f"pool {pool.get('pool_id')}: cards outside the deck network's table: {missing}")
    v = C.vec(pool["cards"])
    v[[C.index[b] for b in cardsmod.BASICS]] = 0
    return v


def deck_vector(cards: list[dict]) -> np.ndarray:
    """A [{"name", "count"}] deck (best23's) as a count vector, basics included."""
    from . import cards as cardsmod
    return cardsmod.load().vec({c["name"]: c["count"] for c in cards})
