"""17lands-style card statistics of simulated games, next to 17lands' own (docs/018, C3). For every non-basic card, over
the player-games with a winner: the seats whose deck had it (games played, GP), that saw it in hand (the opening hand or
a draw, GIH) or did not (games not seen, GNS), their win rates, and the improvement when drawn (IWD = GIH WR - GNS WR).
The same measures from 17lands' public FDN Premier Draft game data, which also splits the opening hand (OH) from cards
drawn later (GD); the simulation's records (play.py's seats.inHand) do not. Writes one CSV row a card (all rarities)
and prints the Spearman correlations with 17lands by rarity, over the cards with --min-games games in hand.

    python tools/imitation_scale/card_stats.py runs/exp4/games/mlp-selfplay-t0/games.jsonl --out card_stats.csv \\
        [--also "Transformer=runs/exp4/games/c1-selfplay-t0-s0/games.jsonl,runs/exp4/games/c1-selfplay-t0-s1/games.jsonl"]

17lands' measures take ~2 minutes to count from the CSV and are cached next to it (FDN_card_stats.json).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
GAMES_CSV = REPO / "data" / "17lands" / "game_data_public.FDN.PremierDraft.csv.gz"
CARDS_CSV = REPO / "data" / "17lands" / "cards.csv"
DECKS = REPO / "data" / "deckgen" / "FDN_PremierDraft_wr60" / "top_player_FDN_decks"
BASICS = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
RARITY_GROUPS = (("common", {"common"}), ("uncommon", {"uncommon"}), ("rare + mythic", {"rare", "mythic"}),
                 ("all", {"common", "uncommon", "rare", "mythic"}))


def _gih():
    spec = importlib.util.spec_from_file_location("gih", Path(__file__).with_name("gih.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def deck_cards(path: Path) -> set[str]:
    """The non-basic cards of a .dck file's main deck ("N [SET:num] Name" lines; sideboard lines start with SB:)."""
    out = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("NAME:", "SB:", "#")) or "]" not in line:
            continue
        name = line.split("]", 1)[1].strip()
        if name not in BASICS:
            out.add(name)
    return out


def sim_counts(paths: list[Path], decks: Path) -> tuple[dict[str, np.ndarray], int]:
    """card -> [GP, GP wins, GIH, GIH wins, GNS, GNS wins] over the seats of the games with a winner; player-games."""
    counts: dict[str, np.ndarray] = {}
    cache: dict[str, set[str]] = {}
    seats = 0
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            g = json.loads(line)
            if g.get("error") or g.get("winner") not in ("A", "B"):
                continue
            for seat, deck in (("A", g["deck1"]), ("B", g["deck2"])):
                if deck not in cache:
                    cache[deck] = deck_cards(decks / f"{deck}.dck")
                won = int(g["winner"] == seat)
                hand = {c for c, n in (g["seats"][seat].get("inHand") or {}).items() if n and c not in BASICS}
                seats += 1
                for c in cache[deck] | hand:
                    v = counts.setdefault(c, np.zeros(6, np.int64))
                    v[0] += 1; v[1] += won
                    if c in hand:
                        v[2] += 1; v[3] += won
                    else:
                        v[4] += 1; v[5] += won
    return counts, seats


def human_counts(path: Path = GAMES_CSV) -> dict[str, list[int]]:
    """card -> [GP, GP wins, OH, OH wins, GD, GD wins, GIH, GIH wins, GNS, GNS wins] over 17lands' player-games
    (GD: drawn, not in the opening hand), cached in FDN_card_stats.json next to the CSV."""
    cache = path.with_name("FDN_card_stats.json")
    if cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
        return json.loads(cache.read_text())
    with gzip.open(path, "rt", newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        col = {h: i for i, h in enumerate(head)}
        names = [h[len("deck_"):] for h in head if h.startswith("deck_") and f"drawn_{h[len('deck_'):]}" in col]
        iw = col["won"]
        idk = [col[f"deck_{n}"] for n in names]
        ioh = [col[f"opening_hand_{n}"] for n in names]
        idr = [col[f"drawn_{n}"] for n in names]
        acc = np.zeros((len(names), 10), np.int64)
        for r in rd:
            won = r[iw] in ("True", "true", "1")
            deck = np.fromiter((r[i] not in ("0", "") for i in idk), bool, len(names))
            oh = np.fromiter((r[i] not in ("0", "") for i in ioh), bool, len(names))
            dr = np.fromiter((r[i] not in ("0", "") for i in idr), bool, len(names))
            gd, gih = dr & ~oh, oh | dr
            gns = deck & ~gih
            for k, m in enumerate((deck, oh, gd, gih, gns)):
                acc[m, 2 * k] += 1
                if won:
                    acc[m, 2 * k + 1] += 1
    out = {n: acc[i].tolist() for i, n in enumerate(names)}
    cache.write_text(json.dumps(out))
    return out


def card_info(path: Path = CARDS_CSV) -> dict[str, tuple[str, str]]:
    """name -> (rarity, colour identity): the rarity the other C3 tools use (draftzero.stats.load_rarity), the colour
    from 17lands' cards.csv, FDN's printing first."""
    from draftzero.stats import load_rarity
    rarity = load_rarity()
    color: dict[str, str] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["expansion"] == "FDN" or r["name"] not in color:
                color[r["name"]] = r["color_identity"]
    return {n: (rarity.get(n, ""), color.get(n, "")) for n in set(rarity) | set(color)}


def wr(w, n):
    return w / n if n else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--decks", type=Path, default=DECKS)
    ap.add_argument("--also", action="append", default=[], metavar="LABEL=GAMES[,GAMES]",
                    help="another simulation's GIH WR as a column, e.g. the transformer's self-play")
    ap.add_argument("--min-games", type=int, default=30)
    a = ap.parse_args(argv)
    gih = _gih()
    sim, seats = sim_counts(a.games, a.decks)
    hum = human_counts()
    info = card_info()
    also = []
    for spec in a.also:
        label, paths = spec.split("=", 1)
        c, n = sim_counts([Path(x) for x in paths.split(",")], a.decks)
        also.append((label, c, n))
    names = sorted(n for n in set(sim) | set(hum) if n not in BASICS)
    rows = []
    for n in names:
        s = sim.get(n, np.zeros(6, np.int64))
        h = hum.get(n, [0] * 10)
        rar, color = info.get(n, ("", ""))
        row = {"Name": n, "Color": color, "Rarity": rar,
               "# GP": int(s[0]), "GP WR": wr(s[1], s[0]), "# GIH": int(s[2]), "GIH WR": wr(s[3], s[2]),
               "# GNS": int(s[4]), "GNS WR": wr(s[5], s[4]), "IWD": wr(s[3], s[2]) - wr(s[5], s[4]),
               "17lands # GP": h[0], "17lands GP WR": wr(h[1], h[0]), "17lands OH WR": wr(h[3], h[2]),
               "17lands GD WR": wr(h[5], h[4]), "17lands # GIH": h[6], "17lands GIH WR": wr(h[7], h[6]),
               "17lands GNS WR": wr(h[9], h[8]), "17lands IWD": wr(h[7], h[6]) - wr(h[9], h[8])}
        for label, c, _ in also:
            v = c.get(n, np.zeros(6, np.int64))
            row[f"{label} # GIH"] = int(v[2])
            row[f"{label} GIH WR"] = wr(v[3], v[2])
        rows.append(row)
    ok = [r for r in rows if r["# GIH"] >= a.min_games and r["17lands # GIH"] > 0]
    for key in ("GIH WR", "17lands GIH WR"):         # ranks among the cards with enough games, 1 = best
        order = sorted(ok, key=lambda r: -r[key])
        for i, r in enumerate(order, 1):
            r[f"{key} rank"] = i
    rows.sort(key=lambda r: (-(r["GIH WR"] if r["# GIH"] >= a.min_games else -1.0), r["Name"]))
    cols = list(rows[0].keys()) + ["GIH WR rank", "17lands GIH WR rank"]
    cols = list(dict.fromkeys(cols))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) and v == v else "" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"{seats:,} player-games with a winner; {len(rows)} non-basic cards ({len(ok)} with {a.min_games}+ games "
          f"in hand) -> {a.out}")
    print(f"{'Spearman with 17lands':<16} {'cards':>5} {'GIH WR':>7} {'GP WR':>7} {'GNS WR':>7} {'IWD':>7}" +
          "".join(f" {label + ' GIH':>16}" for label, _, _ in also))
    for gname, rs in RARITY_GROUPS:
        sel = [r for r in ok if r["Rarity"] in rs]
        line = f"{gname:<16} {len(sel):>5}"
        for k in ("GIH WR", "GP WR", "GNS WR", "IWD"):
            line += f" {gih.spearman([r[k] for r in sel], [r['17lands ' + k] for r in sel]):>7.3f}"
        for label, _, _ in also:
            sel2 = [r for r in sel if r[f"{label} # GIH"] >= a.min_games]
            line += f" {gih.spearman([r[label + ' GIH WR'] for r in sel2], [r['17lands GIH WR'] for r in sel2]):>16.3f}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
