"""pools.py — the deck pool: a `pool.tsv` (one row per deck) and a directory of gorge deck files beside it.

A pool is what dzgorge's `-decks DIR -pool FILE -split S` reads (and `-belief-decks/-belief-pool` for a
`:closedlist` seat's belief): `DIR/<deck>.json` per deck, and a TSV with a header row whose first three columns
are `deck  split  colors` (dzgorge reads only those; `main_colors` and `cards` are for people and plots).

    rows = pools.read("data/gorge/pool.tsv")
    pools.splits(rows)                              # {"train": [...28,366 names], "eval": [...3,150]}
    pools.sample(rows, 40, split="train")           # 40 names, the same on every machine and Python
    pools.subset(rows, "data/gorge/decks", names, "assets/sample")   # copy those decks + write their pool.tsv

    python -m draftzero.data.pools info   [--pool data/gorge/pool.tsv]
    python -m draftzero.data.pools sample --split train --n 40 [--seed draftzero-sample-v1]
    python -m draftzero.data.pools subset --names assets/sample/train.txt assets/sample/eval.txt --out assets/sample

`assets/sample` (80 decks: 40 train, 40 eval) is such a subset of the full FDN pool, so a fresh clone can run the
smoke config without building the 31,516-deck pool (`python -m draftzero.data.decks`).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from draftzero.data.paths import ASSETS, GORGE_DECKS, GORGE_POOL

HEADER = ("deck", "split", "colors", "main_colors", "cards")
SAMPLE_SEED = "draftzero-sample-v1"      # assets/sample/train.txt = sample(train, 40, seed=SAMPLE_SEED)
WUBRG = "WUBRG"
NAME_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")   # a deck name is also a file stem: dzgorge opens DIR/<name>.json


@dataclass(frozen=True)
class PoolRow:
    deck: str
    split: str
    colors: str = ""
    main_colors: str = ""
    cards: int = 0

    def line(self) -> str:
        return "\t".join((self.deck, self.split, self.colors, self.main_colors, str(self.cards)))


Rows = Sequence[PoolRow]


def read(pool: str | Path) -> list[PoolRow]:
    """Rows of a pool.tsv in file order (the order dzgorge loads a belief corpus in). Extra columns are ignored;
    missing trailing ones default."""
    lines = Path(pool).read_text().splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    if header[:2] != ["deck", "split"]:
        raise ValueError(f"{pool}: not a pool.tsv (header {header[:3]}, want {list(HEADER[:3])})")
    out = []
    for line in lines[1:]:
        if not line.strip():
            continue
        v = dict(zip(header, line.split("\t")))
        out.append(PoolRow(v["deck"], v.get("split", ""), v.get("colors", ""), v.get("main_colors", ""),
                           int(v.get("cards") or 0)))
    return out


def write(pool: str | Path, rows: Iterable[PoolRow]) -> Path:
    """Write a pool.tsv (header + rows, in the order given). Refuses names that cannot be file stems."""
    pool = Path(pool)
    pool.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    for r in rows:
        check_name(r.deck)
    pool.write_text("\t".join(HEADER) + "\n" + "".join(r.line() + "\n" for r in rows))
    return pool


def check_name(name: str) -> str:
    if not NAME_RE.match(name or ""):
        raise ValueError(f"deck name {name!r} is not a plain file stem ([A-Za-z0-9_.+-])")
    return name


def _rows(pool: str | Path | Rows) -> list[PoolRow]:
    return read(pool) if isinstance(pool, (str, Path)) else list(pool)


def splits(pool: str | Path | Rows) -> dict[str, list[str]]:
    """Deck names by split, each in pool order."""
    out: dict[str, list[str]] = {}
    for r in _rows(pool):
        out.setdefault(r.split, []).append(r.deck)
    return out


def names(pool: str | Path | Rows, split: str | None = None) -> list[str]:
    """Deck names of one split (None or "all": every deck), in pool order."""
    return [r.deck for r in _rows(pool) if split in (None, "all") or r.split == split]


def sample(pool: str | Path | Rows, n: int, *, split: str | None = None, seed: str | int = SAMPLE_SEED) -> list[str]:
    """`n` deck names drawn without replacement, deterministically: `random.Random(seed)` over the sorted names, so
    the draw depends only on the seed and the set of names (not on file order, machine or Python version).
    Returned sorted. n larger than the split is an error."""
    pop = sorted(names(pool, split))
    if n > len(pop):
        raise ValueError(f"asked for {n} decks of split {split!r}, the pool has {len(pop)}")
    return sorted(random.Random(seed).sample(pop, n))


def deck_path(decks_dir: str | Path, name: str) -> Path:
    return Path(decks_dir) / f"{check_name(name)}.json"


def load_deck(decks_dir: str | Path, name: str) -> dict:
    return json.loads(deck_path(decks_dir, name).read_text())


def deck_cards(deck: Mapping) -> dict[str, int]:
    """{card name: count} of a gorge deck file ({name, cards: [{name, count}]}), or of a plain {card: count} map
    (string values, such as a "colors" tag beside the counts, are not cards)."""
    out: dict[str, int] = {}
    if "cards" in deck and isinstance(deck["cards"], list):
        for e in deck["cards"]:
            out[e["name"]] = out.get(e["name"], 0) + int(e.get("count", 1))
        return out
    for k, v in deck.items():
        if not isinstance(v, str):
            out[str(k)] = int(v)
    return out


def gorge_deck(name: str, cards: Mapping[str, int], *, notes: str | None = None) -> dict:
    """A gorge deck file's JSON (gorge `deck.File`: bare names and counts, sorted by name)."""
    d = {"name": check_name(name), "format": "limited"}
    if notes:
        d["notes"] = notes
    d["cards"] = [{"name": c, "count": int(n)} for c, n in sorted(cards.items()) if int(n) > 0]
    return d


@lru_cache(maxsize=1)
def card_colors() -> dict[str, str]:
    """Card name -> the colours of its mana cost (WUBRG order), from assets/cards/FDN_cards.tsv. Lands and cards
    outside FDN are absent. Hybrid and Phyrexian symbols count every colour they name."""
    f = ASSETS / "cards" / "FDN_cards.tsv"
    out: dict[str, str] = {}
    if not f.exists():
        return out
    lines = [l for l in f.read_text().splitlines() if not l.startswith("#")]
    header = lines[0].split("\t")
    for line in lines[1:]:
        v = dict(zip(header, line.split("\t")))
        cost = v.get("mana_cost", "")
        if cost:
            out[v["name"]] = "".join(c for c in WUBRG if c in cost)
    return out


def deck_colors(cards: Mapping[str, int], *, main_min: int = 3) -> tuple[str, str]:
    """(colors, main_colors) of a deck from its spells' mana costs: every colour present, and the colours at least
    `main_min` spells carry (the rest are a splash). Cards the FDN table does not know count for nothing."""
    table = card_colors()
    count = {c: 0 for c in WUBRG}
    for name, n in cards.items():
        for c in table.get(name, ""):
            count[c] += int(n)
    colors = "".join(c for c in WUBRG if count[c] > 0)
    main = "".join(c for c in WUBRG if count[c] >= main_min)
    return colors, main


def subset(pool: str | Path | Rows, decks_dir: str | Path, deck_names: Iterable[str], out_dir: str | Path,
           *, decks_subdir: str = "decks") -> tuple[Path, Path]:
    """Copy the named decks (byte for byte) to `out_dir/decks/` and write their rows, in pool order, to
    `out_dir/pool.tsv`. Returns (decks dir, pool path). Unknown names are an error."""
    rows = _rows(pool)
    want = set(deck_names)
    by_name = {r.deck: r for r in rows}
    missing = sorted(want - by_name.keys())
    if missing:
        raise KeyError(f"{len(missing)} decks not in the pool, e.g. {missing[:3]}")
    out_dir = Path(out_dir)
    dst = out_dir / decks_subdir
    dst.mkdir(parents=True, exist_ok=True)
    keep = [r for r in rows if r.deck in want]
    for r in keep:
        shutil.copyfile(deck_path(decks_dir, r.deck), deck_path(dst, r.deck))
    return dst, write(out_dir / "pool.tsv", keep)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m draftzero.data.pools", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("info", "sample", "subset"):
        p = sub.add_parser(cmd)
        p.add_argument("--pool", default=str(GORGE_POOL))
        if cmd == "sample":
            p.add_argument("--n", type=int, required=True)
            p.add_argument("--split", default=None)
            p.add_argument("--seed", default=SAMPLE_SEED)
        if cmd == "subset":
            p.add_argument("--decks", default=str(GORGE_DECKS))
            p.add_argument("--names", nargs="+", required=True, help="files with one deck name a line")
            p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = read(a.pool)
    if a.cmd == "info":
        for split, ns in sorted(splits(rows).items()):
            print(f"{split}\t{len(ns)}")
    elif a.cmd == "sample":
        print("\n".join(sample(rows, a.n, split=a.split, seed=a.seed)))
    elif a.cmd == "subset":
        want = [n for f in a.names for n in Path(f).read_text().split()]
        d, p = subset(rows, a.decks, want, a.out)
        print(f"wrote {len(want)} decks to {d} and {p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
