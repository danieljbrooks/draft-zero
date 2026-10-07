"""17lands-style card statistics of dzk self-play games, against 17lands, with a noise ceiling matched card by card.

For each non-basic card over the player-games with a winner: games played (GP: the seat's deck had it), games in hand
(GIH: opening hand or drawn, `seats.X.inHand`), games not seen (GNS), their win rates and IWD = GIH WR - GNS WR; the
same from 17lands' FDN Premier Draft data (draft-zero's card_stats.py cache); Spearman correlations by rarity.

The noise ceiling answers "how well would 17lands' own games agree with 17lands, at this sample size?". The playable
pool is 38 cards, each in far more of our games than of 17lands', so matching the number of player-games (as
gih_ceiling.py does) would understate our sample. Instead, for every card, draw as many of 17lands' games-in-hand
player-games as the simulation has for that card, recompute its GIH WR, and correlate with 17lands' full-data
GIH WR; repeat (--reps) and report the median and the 5-95% range.

    python3 mtgkernel/py/gih_report.py runs/mtgkernel/gih/mcts100/games.jsonl \\
        --decks-dir mtgkernel/decks/gen_v1 --decks-dir assets/sample/decks --label "mcts:100 self-play" \\
        --out-dir runs/mtgkernel/gih/mcts100 [--main-repo ../draft-zero]

17lands' data is git-ignored, so it is read from the main checkout (--main-repo, default ../draft-zero).
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASICS = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
GROUPS = (("common", {"common"}), ("uncommon", {"uncommon"}), ("rare+mythic", {"rare", "mythic"}),
          ("all", {"common", "uncommon", "rare", "mythic"}))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gih = _load("gih", REPO / "tools" / "imitation_scale" / "gih.py")


def deck_cards(path):
    out = set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("NAME:", "SB:", "#")):
            continue
        name = line.split("]", 1)[1].strip() if "]" in line else line.split(" ", 1)[1].strip()
        if name not in BASICS:
            out.add(name)
    return out


def load_records(paths):
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    g = json.loads(line)
                    if not g.get("error") and g.get("winner") in ("A", "B"):
                        yield g


def sim_counts(paths, deck_dirs):
    """card -> [GP, GP wins, GIH, GIH wins, GNS, GNS wins, GIH expected wins]; player-games with a winner; decided
    games. "Expected wins" sums, over the card's games in hand, its seat's deck win rate in this run (the deck's
    win rate over all its seats), so GIH WR minus expected/GIH is the card's deck-adjusted GIH WR: how much better
    the deck does when it sees the card than it does on average. Two passes over the records."""
    cache = {}

    def cards(deck):
        if deck not in cache:
            found = [d / f"{deck}.dck" for d in deck_dirs if (d / f"{deck}.dck").exists()]
            if not found:
                raise SystemExit(f"deck {deck} not found in {deck_dirs}")
            cache[deck] = deck_cards(found[0])
        return cache[deck]

    deck_wl = {}
    for g in load_records(paths):
        for seat, deck in (("A", g["deck1"]), ("B", g["deck2"])):
            v = deck_wl.setdefault(deck, [0, 0])
            v[0] += 1
            v[1] += int(g["winner"] == seat)
    deck_wr = {d: w / n for d, (n, w) in deck_wl.items()}
    counts, seats, games = {}, 0, 0
    for g in load_records(paths):
        games += 1
        for seat, deck in (("A", g["deck1"]), ("B", g["deck2"])):
            won = int(g["winner"] == seat)
            hand = {c for c, n in (g["seats"][seat].get("inHand") or {}).items() if n and c not in BASICS}
            seats += 1
            for c in cards(deck) | hand:
                v = counts.setdefault(c, np.zeros(7, np.float64))
                v[0] += 1
                v[1] += won
                k = 2 if c in hand else 4
                v[k] += 1
                v[k + 1] += won
                if c in hand:
                    v[6] += deck_wr[deck]
    return counts, seats, games


def wr(w, n):
    return w / n if n else float("nan")


def ceiling_17lands(csv_path, cards, sim_gih, ref, reps, seed, cache):
    """Per-card matched ceiling: Spearman of 17lands GIH WR recomputed from sim_gih[c] of its own GIH player-games."""
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        won, hand, names = z["won"], z["hand"], list(z["names"])
    else:
        gc = _load("gih_ceiling", REPO / "tools" / "imitation_scale" / "gih_ceiling.py")
        won, hand, names = gc.load_hands(csv_path, set(cards))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, won=won, hand=hand, names=np.array(names))
    col = {n: i for i, n in enumerate(names)}
    use = [c for c in cards if c in col and c in ref]
    rows = {c: np.flatnonzero(hand[:, col[c]]) for c in use}
    full = np.array([ref[c]["gih_wr"] for c in use])
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        est = []
        for c in use:
            idx = rows[c]
            k = min(int(sim_gih[c]), len(idx))
            pick = rng.choice(idx, k, replace=False)
            est.append(won[pick].mean())
        out.append(gih.spearman(est, full))
    return np.array(out), use


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", nargs="+", type=Path)
    ap.add_argument("--decks-dir", action="append", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-games", type=int, default=30)
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--main-repo", type=Path, default=REPO.parent / "draft-zero")
    a = ap.parse_args(argv)
    data17 = a.main_repo / "data" / "17lands"
    cs = _load("card_stats", REPO / "tools" / "imitation_scale" / "card_stats.py")
    hum = cs.human_counts(data17 / "game_data_public.FDN.PremierDraft.csv.gz")
    ref = json.loads((REPO / "assets" / "reference" / "FDN_gih.json").read_text())["cards"]
    sys.path.insert(0, str(REPO / "src"))
    from draftzero.stats import load_rarity
    rarity = load_rarity()

    sim, seats, games = sim_counts(a.games, a.decks_dir)
    rows = []
    for n in sorted(c for c in sim if c not in BASICS):
        s, h = sim[n], hum.get(n, [0] * 10)
        rows.append({"card": n, "rarity": rarity.get(n, ""), "gp": int(s[0]), "gp_wr": wr(s[1], s[0]),
                     "gih": int(s[2]), "gih_wr": wr(s[3], s[2]), "gns": int(s[4]), "gns_wr": wr(s[5], s[4]),
                     "iwd": wr(s[3], s[2]) - wr(s[5], s[4]), "gih_adj": wr(s[3] - s[6], s[2]),
                     "l17_gih": h[6], "l17_gih_wr": wr(h[7], h[6]), "l17_gp_wr": wr(h[1], h[0]),
                     "l17_iwd": wr(h[7], h[6]) - wr(h[9], h[8])})
    ok = [r for r in rows if r["gih"] >= a.min_games and r["l17_gih"] > 0]
    for key, rk in (("gih_wr", "rank"), ("l17_gih_wr", "l17_rank")):
        for i, r in enumerate(sorted(ok, key=lambda r: -r[key]), 1):
            r[rk] = i
    sim_gih = {r["card"]: r["gih"] for r in ok}
    ceil, used = ceiling_17lands(data17 / "game_data_public.FDN.PremierDraft.csv.gz", sorted(sim_gih), sim_gih, ref,
                                 a.reps, a.seed, a.out_dir.parent / "_cache" / "l17_hands_supported.npz")
    groups = {}
    for gname, rs in GROUPS:
        sel = [r for r in ok if r["rarity"] in rs]
        g = {"cards": len(sel)}
        for k, k17 in (("gih_wr", "l17_gih_wr"), ("gih_adj", "l17_gih_wr"), ("gp_wr", "l17_gp_wr"), ("iwd", "l17_iwd")):
            g[f"spearman_{k}"] = gih.spearman([r[k] for r in sel], [r[k17] for r in sel]) if len(sel) >= 3 else None
        idx = [i for i, c in enumerate(used) if rarity.get(c, "") in rs]
        if len(idx) >= 3:
            sub, _ = ceiling_17lands(data17 / "game_data_public.FDN.PremierDraft.csv.gz", [used[i] for i in idx],
                                     sim_gih, ref, a.reps, a.seed, a.out_dir.parent / "_cache" / "l17_hands_supported.npz")
            g["ceiling_gih_median"] = float(np.median(sub))
            g["ceiling_gih_p5_p95"] = [float(np.percentile(sub, 5)), float(np.percentile(sub, 95))]
        groups[gname] = g
    a.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"label": a.label, "games": [str(p) for p in a.games], "decided_games": games, "player_games": seats,
               "min_games": a.min_games, "reps": a.reps, "groups": groups,
               "median_gih_per_card": float(np.median([r["gih"] for r in ok])) if ok else 0}
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    import csv
    cols = ["card", "rarity", "gp", "gp_wr", "gih", "gih_wr", "gih_adj", "gns", "gns_wr", "iwd", "rank",
            "l17_gih", "l17_gih_wr", "l17_gp_wr", "l17_iwd", "l17_rank"]
    with open(a.out_dir / "card_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r.get("rank", 999)):
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"{a.label}: {games:,} decided games, {seats:,} player-games; {len(ok)} cards with {a.min_games}+ GIH "
          f"(median {summary['median_gih_per_card']:.0f} per card)")
    print(f"{'group':<12} {'cards':>5} {'GIH rho':>8} {'ceiling [5-95%]':>22} {'adjGIH':>8} {'IWD rho':>8} {'GP rho':>8}")
    for gname, g in groups.items():
        c = (f"{g['ceiling_gih_median']:.3f} [{g['ceiling_gih_p5_p95'][0]:.2f}, {g['ceiling_gih_p5_p95'][1]:.2f}]"
             if "ceiling_gih_median" in g else "-")
        f = lambda x: f"{x:.3f}" if isinstance(x, float) and x == x else "-"
        print(f"{gname:<12} {g['cards']:>5} {f(g['spearman_gih_wr']):>8} {c:>22} {f(g['spearman_gih_adj']):>8} {f(g['spearman_iwd']):>8} "
              f"{f(g['spearman_gp_wr']):>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
