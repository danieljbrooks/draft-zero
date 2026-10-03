"""The noise ceiling of the games-in-hand comparison (docs/004's method, docs/018): how well 17lands' own games
correlate with 17lands' full-data GIH win rates when only N player-games are counted. Sample N player-games from
the public FDN Premier Draft game data, recompute each card's GIH win rate (opening hand or drawn), keep the
cards with at least --min-games games in hand, and take the Spearman correlation against the full data's; repeat.
A simulation's correlation at N player-games is judged against this.

    python tools/imitation_scale/gih_ceiling.py --n 2880 18950 --rarity common all [--reps 60]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "data" / "17lands" / "game_data_public.FDN.PremierDraft.csv.gz"
sys.path.insert(0, str(REPO / "src"))


def _gih():
    spec = importlib.util.spec_from_file_location("gih", Path(__file__).with_name("gih.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_hands(csv_path: Path, cards: set[str] | None = None):
    """(won [P], in hand [P, C] bool, card names) for every player-game in the 17lands game data (standard library
    csv, no pandas: ~2 minutes for the 791k FDN player-games)."""
    import csv
    import gzip
    with gzip.open(csv_path, "rt", newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        col = {h: i for i, h in enumerate(head)}
        names = [h[len("drawn_"):] for h in head if h.startswith("drawn_")]
        if cards is not None:
            names = [n for n in names if n in cards]
        iw = col["won"]
        io = np.array([col[f"opening_hand_{n}"] for n in names])
        idr = np.array([col[f"drawn_{n}"] for n in names])
        won, rows = [], []
        for r in rd:
            won.append(r[iw] in ("True", "true", "1"))
            rows.append(np.fromiter(((r[a] not in ("0", "")) or (r[b] not in ("0", "")) for a, b in zip(io, idr)),
                                    dtype=bool, count=len(names)))
    return np.array(won), np.vstack(rows), names


def ceiling(won, hand, names, ref: dict, n: int, min_games: int, reps: int, seed: int = 0) -> list[float]:
    gih = _gih()
    rng = np.random.default_rng(seed)
    full = np.array([ref.get(c, {}).get("gih_wr", np.nan) for c in names])
    out = []
    for _ in range(reps):
        idx = rng.choice(len(won), n, replace=False)
        h, w = hand[idx], won[idx]
        g = h.sum(0)
        wins = (h & w[:, None]).sum(0)
        keep = (g >= min_games) & np.isfinite(full)
        out.append(gih.spearman(wins[keep] / g[keep], full[keep]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", required=True, help="player-games per sample")
    ap.add_argument("--rarity", nargs="+", default=["all"], help="common, uncommon, rare+mythic, all (non-basic)")
    ap.add_argument("--min-games", type=int, default=30)
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--csv", type=Path, default=CSV)
    a = ap.parse_args(argv)
    from draftzero.stats import load_rarity
    rar = load_rarity()
    basics = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
    ref = json.loads((REPO / "assets" / "reference" / "FDN_gih.json").read_text())["cards"]
    won, hand, names = load_hands(a.csv, {c for c in ref if c not in basics})
    print(f"{len(won)} player-games, {len(names)} non-basic cards, min {a.min_games} games in hand, {a.reps} samples")
    for group in a.rarity:
        want = None if group == "all" else set(group.split("+"))
        cols = [i for i, c in enumerate(names) if want is None or rar.get(c) in want]
        for n in a.n:
            r = np.array(ceiling(won, hand[:, cols], [names[i] for i in cols], ref, n, a.min_games, a.reps))
            print(f"  {group:12s} n {n:6d}: median {np.median(r):.3f}  [5-95%: {np.percentile(r, 5):.3f}, {np.percentile(r, 95):.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
