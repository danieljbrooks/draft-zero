"""The games-in-hand win rate (17lands' GIH WR) of simulated games, against 17lands' (docs/018,
policy-only evaluation). A card's GIH WR is the win rate of the seats that saw it in hand (the
opening hand or a draw: play.py's seats.inHand), over the games with a winner. The agreement with
17lands is the Spearman correlation over the cards with at least --min-games games in hand, basic
lands left out (assets/reference/FDN_gih.json: 17lands' FDN Premier Draft games).

Judge the correlation against the noise ceiling at the same number of player-games (two a game):
17lands' own data, subsampled, reaches about 0.35 at 1,000 player-games, 0.49 at 2,880, 0.75 at
10,000 and 0.94 at 50,000 (computed 2026-09-25, commons with 30+ games in hand).

    python tools/imitation_scale/gih.py runs/exp4/policy_eval/*/games.jsonl [--min-games 30] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "assets" / "reference" / "FDN_gih.json"
BASICS = {"Plains", "Island", "Swamp", "Mountain", "Forest"}


def load_games(paths: list[Path]) -> list[dict]:
    games = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                g = json.loads(line)
                if not g.get("error"):
                    games.append(g)
    return games


def gih_counts(games: list[dict]) -> tuple[dict[str, list[int]], int]:
    """card -> [games in hand, wins] over the seats of games with a winner; and the player-games counted.
    A card counts once per seat however many copies were seen."""
    out: dict[str, list[int]] = {}
    seats = 0
    for g in games:
        if g.get("winner") not in ("A", "B"):
            continue
        for seat in ("A", "B"):
            hand = ((g.get("seats") or {}).get(seat) or {}).get("inHand")
            if hand is None:
                continue
            seats += 1
            won = int(g["winner"] == seat)
            for card in hand:
                c = out.setdefault(card, [0, 0])
                c[0] += 1
                c[1] += won
    return out, seats


def ranks(x: np.ndarray) -> np.ndarray:
    """Ranks from 1, ties sharing their mean rank."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b) -> float:
    ra, rb = ranks(np.asarray(a, float)), ranks(np.asarray(b, float))
    if len(ra) < 3 or ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def compare(counts: dict[str, list[int]], reference: dict[str, dict], min_games: int) -> dict:
    rows = []
    for card, (n, w) in sorted(counts.items()):
        ref = reference.get(card)
        if card in BASICS or ref is None or n < min_games:
            continue
        rows.append({"card": card, "games": n, "gih_wr": w / n, "ref_gih_wr": ref["gih_wr"], "ref_games": ref["gih_games"]})
    rho = spearman([r["gih_wr"] for r in rows], [r["ref_gih_wr"] for r in rows]) if rows else float("nan")
    return {"cards": len(rows), "spearman": rho, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", nargs="+", type=Path, help="play.py games.jsonl files")
    ap.add_argument("--min-games", type=int, default=30)
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    games = load_games(a.games)
    counts, seats = gih_counts(games)
    ref = json.loads(a.reference.read_text())["cards"]
    res = compare(counts, ref, a.min_games)
    print(f"{len(games)} games, {seats} player-games with a winner, {len(counts)} cards seen in hand")
    print(f"{res['cards']} cards with {a.min_games}+ games in hand (basics out): Spearman {res['spearman']:.3f} against 17lands")
    rows = sorted(res["rows"], key=lambda r: r["gih_wr"] - r["ref_gih_wr"])
    for title, part in (("most under 17lands", rows[:10]), ("most over 17lands", rows[-10:][::-1])):
        print(f"\n{title}:")
        for r in part:
            print(f"  {r['card']:32s} sim {r['gih_wr']:.3f} ({r['games']:5d})  17lands {r['ref_gih_wr']:.3f}")
    if a.json:
        a.json.write_text(json.dumps({"games": len(games), "player_games": seats, "min_games": a.min_games, **res}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
