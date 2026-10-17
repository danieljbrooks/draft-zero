"""belief.py — the corpus of deck lists a `:closedlist` search deals the opponent's hidden cards from (v0).

DraftZero's opponent-deck model, version 0, is gorge's own closed-list world plus a corpus this module writes. With
`:closedlist` an az seat knows its own list but not the opponent's; for every simulation it draws one list uniformly
from the corpus lists whose colours cover the opponent's seen colours, removes the cards already seen, and deals the
rest onto the opponent's unknown hand and library (gorge `searchprobe.Belief.closedDeal`; notes in
dz-notes/gorge-main-closedlist.md). dzgorge takes the corpus as `-belief-decks DIR -belief-pool FILE -belief-split S`;
`Gorge.play(..., belief=Belief(decks, pool, split))` passes those.

    b = write_belief({"d1": deck_json, "d2": {"Llanowar Elves": 2, ...}}, "runs/x/belief")
    gorge.play(a, b_spec, ..., belief=b)

    per = leave_one_out(decks, "runs/x/belief-loo")       # {deck: Belief without that deck}, one shared decks dir
    corpus_hash(b)                                       # record it in the run manifest

What to know when choosing a corpus:

- **Tabula rasa.** gorge's default corpus is the pool's TRAIN split, i.e. 17lands human decks. A run that must not
  learn from human data writes its own decks here (the builder's decks, or the run's actor decks) instead.
- **Leave one out.** The corpus can hold the opponent's real list: on train/train games dzgorge allows it, and with a
  small population (a sealed pod) colour covering can shrink the candidates to that one list, which is open lists
  again. Exclude the game's own decks (`write_belief(..., exclude=...)`), or give each opponent deck a corpus without
  it (`leave_one_out`). One belief is loaded per dzgorge process and shared by both seats, so a per-opponent corpus
  means one `Gorge.play` per opponent deck (`-pairs-file` / `-deckfile` runs).
- **Order matters.** dzgorge loads the lists in pool.tsv order and draws by index, so the same decks in another order
  give other worlds. Rows are written sorted by name; `corpus_hash` covers the order and every deck file.
- **Refresh by restarting.** The belief is read once per process: actors pick up a new corpus on their next chunk.
- **Never mix targets.** Open-list (`redeal`) and closed-list (`redeal-closed`) visit records are different targets;
  a record's `world` field says which.

The learned model that replaces this (pick- or pool-conditioned, behind `Belief.closedDeal`) is later work.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

from draftzero.data import pools

if TYPE_CHECKING:   # the engine package owns Belief; imported lazily so this module loads without it
    from draftzero.engine.gorge import Belief

DECKS_DIR = "decks"
POOL = "pool.tsv"


def _belief(decks: Path, pool: Path, split: str) -> "Belief":
    from draftzero.engine.gorge import Belief
    return Belief(decks=Path(decks), pool=Path(pool), split=split)


def _write_decks(decks: Mapping[str, Mapping], deck_dir: Path) -> list[pools.PoolRow]:
    """Each deck as deck_dir/<name>.json (a gorge deck file); returns their pool rows sorted by name (split unset)."""
    deck_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in sorted(decks):
        d = decks[name]
        cards = pools.deck_cards(d)
        if not cards:
            raise ValueError(f"deck {name!r} has no cards")
        pools.deck_path(deck_dir, name).write_text(json.dumps(pools.gorge_deck(name, cards), indent=0) + "\n")
        colors, main = pools.deck_colors(cards)
        colors = d.get("colors", colors) if isinstance(d.get("colors"), str) else colors
        main = d.get("main_colors", main) if isinstance(d.get("main_colors"), str) else main
        rows.append(pools.PoolRow(name, "", colors, main, sum(cards.values())))
    return rows


def write_belief(decks: Mapping[str, Mapping], out_dir: str | Path, *, exclude: Iterable[str] = (),
                 split: str = "train") -> "Belief":
    """Write a belief corpus: out_dir/decks/<name>.json for every deck and out_dir/pool.tsv listing them (sorted by
    name, all in `split`), minus the names in `exclude` (whose files are not written either).

    `decks` maps a deck name (a file stem: letters, digits, `_.+-`) to a gorge deck file's JSON
    ({name, cards: [{name, count}]}) or a plain {card name: count}. A deck's optional "colors"/"main_colors"
    strings fill those pool columns; else they come from the FDN card table. Returns the engine's Belief."""
    out_dir = Path(out_dir)
    drop = set(exclude)
    keep = {k: v for k, v in decks.items() if k not in drop}
    if not keep:
        raise ValueError("a belief needs at least one deck (all were excluded)")
    rows = [pools.PoolRow(r.deck, split, r.colors, r.main_colors, r.cards)
            for r in _write_decks(keep, out_dir / DECKS_DIR)]
    pool = pools.write(out_dir / POOL, rows)
    return _belief(out_dir / DECKS_DIR, pool, split)


def leave_one_out(decks: Mapping[str, Mapping], out_dir: str | Path, *, names: Iterable[str] | None = None,
                  split: str = "train") -> dict[str, "Belief"]:
    """One corpus per held-out deck, for games whose opponent plays that deck: {name: Belief over every other deck}.

    The deck files are written once (out_dir/decks/) and each corpus is only a pool file, out_dir/loo/<name>.tsv,
    so N held-out decks cost N small TSVs, not N copies. `names` (default: every deck) picks which to hold out."""
    out_dir = Path(out_dir)
    if len(decks) < 2:
        raise ValueError("leave-one-out needs at least two decks")
    rows = _write_decks(decks, out_dir / DECKS_DIR)
    held = list(decks) if names is None else list(names)
    unknown = sorted(set(held) - set(decks))
    if unknown:
        raise KeyError(f"not in the corpus: {unknown[:3]}")
    out = {}
    for name in sorted(held):
        pool = pools.write(out_dir / "loo" / f"{pools.check_name(name)}.tsv",
                           [pools.PoolRow(r.deck, split, r.colors, r.main_colors, r.cards) for r in rows if r.deck != name])
        out[name] = _belief(out_dir / DECKS_DIR, pool, split)
    return out


def corpus_hash(belief=None, *, decks: str | Path | None = None, pool: str | Path | None = None,
                split: str | None = None) -> str:
    """sha256 (16 hex) of what a corpus deals from: the rows of `split` in pool order and each one's deck file.
    Pass a Belief, or decks/pool/split. Decks the pool lists but the directory lacks hash as missing (dzgorge
    skips them too)."""
    if belief is not None:
        decks, pool, split = belief.decks, belief.pool, belief.split
    if decks is None or pool is None:
        raise ValueError("corpus_hash needs a Belief or decks= and pool=")
    split = split or "train"
    h = hashlib.sha256()
    for r in pools.read(pool):
        if split != "all" and r.split != split:
            continue
        p = Path(decks) / f"{r.deck}.json"
        h.update(r.deck.encode() + b"\0")
        h.update(p.read_bytes() if p.exists() else b"<missing>")
        h.update(b"\0")
    return h.hexdigest()[:16]
