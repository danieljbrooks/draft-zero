"""Card statistics from dzgorge games against 17lands: games-in-hand (GIH) win rates and colour-pair records.

17lands' GIH win rate of a card is the win rate of the players who had it in hand: kept in the opening hand or
drawn. A dzgorge games.jsonl row carries the same thing per seat (`in_hand[seat]`, gorge fork play.go), so the
simulated games give a GIH win rate per card, and its Spearman rank correlation with 17lands' FDN Premier Draft
GIH WR (assets/reference/FDN_gih.json, made by `draftzero.eval.reference`) measures format knowledge without
saturating (docs/019 §4.4, docs/026). Only games with a winner count (draws and stalls are left out), each seat
once; a card drawn twice in a game counts once.

    rep = gih_report(games, min_games=30)       {"commons": {cards, spearman, ...}, "all": {...}, "colors": {...}}
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

from draftzero.data.paths import ASSETS, REPO
from draftzero.eval.stats import spearman

REFERENCE = ASSETS / "reference" / "FDN_gih.json"
RARITY = ASSETS / "cards" / "FDN_card_ids.tsv"
POOL = REPO / "data" / "gorge" / "pool.tsv"
BASICS = frozenset({"Plains", "Island", "Swamp", "Mountain", "Forest"})
MIN_COMMON_GAMES = 30       # exp #1's threshold for its commons correlation (docs/003 §6.4)


def load_reference(path: Path | None = REFERENCE) -> dict:
    """17lands' reference {"cards": {name: {gih_games, gih_wr}}, "colors": {pair: {games, wr}}}; empty if missing."""
    return json.loads(Path(path).read_text()) if path and Path(path).exists() else {"cards": {}, "colors": {}}


def load_rarity(path: Path = RARITY) -> dict[str, str]:
    """card name -> rarity, from 17lands' cards.csv (assets/cards/FDN_card_ids.tsv). A name printed in several
    sets takes its FDN rarity. Empty if the file is missing."""
    out: dict[str, str] = {}
    if not Path(path).exists():
        return out
    with open(path) as f:
        for r in csv.DictReader((l for l in f if l.strip() and not l.startswith("#")), delimiter="\t"):
            if r.get("expansion") == "FDN" or r["name"] not in out:
                out[r["name"]] = r.get("rarity", "")
    return out


def deck_colors(pool: Path = POOL) -> dict[str, str]:
    """deck -> main colours, from a pool.tsv (columns deck, split, colors, main_colors, ...); empty if missing."""
    if not Path(pool).exists():
        return {}
    with open(pool) as f:
        return {r["deck"]: r.get("main_colors") or "" for r in csv.DictReader(f, delimiter="\t")}


def seats(games: Iterable[dict], side: Optional[str] = None) -> Iterator[tuple[int, str, list, bool]]:
    """(seat, deck, in_hand, won) for each seat of each game with a winner; `side` "A"/"B" keeps one policy's."""
    for g in games:
        if g.get("result") not in ("A", "B"):
            continue
        for s in (0, 1):
            if side and g["side"][s] != side:
                continue
            yield s, g["decks"][s], (g.get("in_hand") or [None, None])[s] or [], g["winner"] == s


def card_gih(games: Iterable[dict], side: Optional[str] = None) -> dict[str, list[int]]:
    """card -> [games in hand, wins]."""
    acc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for _, _, hand, won in seats(games, side):
        for c in set(hand):
            acc[c][0] += 1
            acc[c][1] += int(won)
    return dict(acc)


def color_records(games: Iterable[dict], colors: dict[str, str], side: Optional[str] = None) -> dict[str, list[int]]:
    """main colours -> [games, wins], each seat's deck looked up in `colors` (deck_colors); unknown decks "?"."""
    acc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for _, deck, _, won in seats(games, side):
        c = colors.get(deck) or "?"
        acc[c][0] += 1
        acc[c][1] += int(won)
    return dict(acc)


def gih_rows(acc: dict, reference: dict, min_games: int, only: Optional[set] = None,
             exclude: Iterable[str] = BASICS) -> list[tuple[str, int, float, float]]:
    """(card, games, simulated GIH WR, 17lands GIH WR) for cards with `min_games`+ games that 17lands rates,
    best simulated first."""
    ref = reference.get("cards", {})
    exclude = set(exclude)
    rows = [(c, n, w / n, ref[c]["gih_wr"]) for c, (n, w) in acc.items()
            if n >= min_games and c in ref and c not in exclude and (only is None or c in only)]
    return sorted(rows, key=lambda r: (-r[2], r[0]))


def gih_correlation(acc: dict, reference: dict, min_games: int, only: Optional[set] = None,
                    exclude: Iterable[str] = BASICS) -> tuple[Optional[float], int]:
    """(Spearman rho of simulated vs 17lands GIH WR, cards compared); rho None for fewer than 3 cards."""
    rows = gih_rows(acc, reference, min_games, only, exclude)
    return spearman([r[2] for r in rows], [r[3] for r in rows]), len(rows)


def gih_report(games: list[dict], min_games: int = MIN_COMMON_GAMES, side: Optional[str] = None, *,
               reference: dict | None = None, rarity: dict | None = None, colors: dict | None = None) -> dict:
    """GIH correlations over commons and over every non-basic card, the spread of the simulated and 17lands win
    rates, and the two-colour decks' records against 17lands'. Inputs default to the repo's assets and
    data/gorge/pool.tsv."""
    reference = load_reference() if reference is None else reference
    rarity = load_rarity() if rarity is None else rarity
    colors = deck_colors() if colors is None else colors
    acc = card_gih(games, side)
    out: dict = {"games": len(games), "player_games": sum(1 for _ in seats(games, side)), "min_games": min_games}
    for name, only in (("commons", {c for c, r in rarity.items() if r == "common"}), ("all", None)):
        rows = gih_rows(acc, reference, min_games, only)
        sim, hum = np.array([r[2] for r in rows]), np.array([r[3] for r in rows])
        out[name] = {"cards": len(rows), "spearman": spearman(sim, hum),
                     "sd_sim": float(sim.std()) if len(rows) else None,
                     "sd_17lands": float(hum.std()) if len(rows) else None,
                     "rows": rows}
    by = {c: v for c, v in color_records(games, colors, side).items() if len(c) == 2}
    refc = reference.get("colors", {})
    pairs = sorted(c for c in by if c in refc)
    out["colors"] = {"rows": {c: {"games": by[c][0], "wr": by[c][1] / by[c][0], "ref_wr": refc[c]["wr"]} for c in pairs},
                     "spearman": spearman([by[c][1] / by[c][0] for c in pairs], [refc[c]["wr"] for c in pairs])}
    return out
