"""
stats.py — format-level statistics for the FDN generalist (see docs/).

The XMage fork logs one GAME_SUMMARY line per game (decks, who went first, winner, turns, and every
card each player drew, opening hand included). From those:

  per-card GIH win rate   win rate of a player in games where they drew the card (17lands' "game in
                          hand"), from self-play games, where both players are the same network, and
                          its Spearman rank correlation with 17lands' human GIH WR
  deck records            W/L of every deck played, aggregated by the deck's main colors

  python -m draftzero.stats runs/<run_id>     prints the current tables
"""
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

from magezero.metrics import wilson

SUMMARY_TAG = "GAME_SUMMARY "


def parse_summaries(log_path: Path) -> list[dict]:
    """GAME_SUMMARY lines of one JVM log, in completion order."""
    out = []
    with open(log_path, errors="replace") as f:
        for line in f:
            i = line.find(SUMMARY_TAG)
            if i < 0:
                continue
            body = line[i + len(SUMMARY_TAG):].rsplit(" =>[", 1)[0]
            try:
                out.append(json.loads(body))
            except json.JSONDecodeError:
                pass
    return out


def load_deck_meta(tsv: Path) -> dict[str, dict]:
    """deck name (the .dck stem) -> {split, colors, main_colors, player_wr_bucket}"""
    rows = Path(tsv).read_text().splitlines()
    header = rows[0].split("\t")
    meta = {}
    for r in rows[1:]:
        vals = dict(zip(header, r.split("\t")))
        meta[vals.pop("deck")] = vals
    return meta


def load_reference(path: Path) -> dict:
    return json.loads(Path(path).read_text()) if path and Path(path).exists() else {"cards": {}, "colors": {}}


def spearman(xs: list[float], ys: list[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (sx * sy) if sx and sy else None


def _sides(g: dict):
    """(side, deck, drawn, won) for each player whose result counts. Draws count as losses for both."""
    for s in ("a", "b") if g.get("agent_side") == "both" else ("a",):
        yield s, g[f"deck_{s}"], g.get(f"drawn_{s}") or [], g.get("winner") == s.upper()


def card_gih(games: list[dict]) -> dict[str, list[int]]:
    """card -> [games in hand, wins]; a card drawn twice in one game counts once."""
    acc = defaultdict(lambda: [0, 0])
    for g in games:
        for _, _, drawn, won in _sides(g):
            for c in set(drawn):
                acc[c][0] += 1
                acc[c][1] += won
    return acc


def gih_correlation(acc: dict, reference: dict, min_games: int) -> tuple[Optional[float], int]:
    ref = reference.get("cards", {})
    pairs = [(w / n, ref[c]["gih_wr"]) for c, (n, w) in acc.items() if n >= min_games and c in ref]
    return spearman([p[0] for p in pairs], [p[1] for p in pairs]), len(pairs)


def color_records(games: list[dict], meta: dict) -> dict[str, list[int]]:
    acc = defaultdict(lambda: [0, 0])
    for g in games:
        for _, deck, _, won in _sides(g):
            c = meta.get(deck, {}).get("main_colors") or "?"
            acc[c][0] += 1
            acc[c][1] += won
    return acc


def deck_records(games: list[dict]) -> dict[str, list[int]]:
    acc = defaultdict(lambda: [0, 0])
    for g in games:
        for _, deck, _, won in _sides(g):
            acc[deck][0] += 1
            acc[deck][1] += won
    return acc


def selfplay_games(games: list[dict], gens: Optional[tuple[int, int]] = None) -> list[dict]:
    """Games where both players were the same agent (network self-play, or gen 0's heuristic self-play)."""
    return [g for g in games if g.get("agent_side") == "both"
            and (gens is None or gens[0] <= g["gen"] <= gens[1])]


def summarize(games: list[dict], meta: dict, reference: dict, window_gens: int = 10,
              min_card_games: int = 15) -> dict:
    """Everything the dashboard shows about format knowledge, from games.jsonl rows."""
    gens = sorted({g["gen"] for g in games})
    last = gens[-1] if gens else 0
    net = selfplay_games(games, (1, last))
    recent = selfplay_games(games, (max(1, last - window_gens + 1), last))
    heuristic = selfplay_games(games, (0, 0))

    # rank correlation per gen: heuristic gen 0 alone, then network self-play cumulative from gen 1
    rho_series = []
    cum = defaultdict(lambda: [0, 0])
    for gen in gens:
        gen_games = selfplay_games(games, (gen, gen))
        if gen == 0:
            rho, n = gih_correlation(card_gih(gen_games), reference, min_card_games)
            rho_series.append({"gen": 0, "series": "heuristic search (gen 0)", "v": rho, "cards": n})
            continue
        for c, (n_, w) in card_gih(gen_games).items():
            cum[c][0] += n_
            cum[c][1] += w
        rho, n = gih_correlation(cum, reference, min_card_games)
        rho_series.append({"gen": gen, "series": "network self-play, all gens", "v": rho, "cards": n})

    ref_cards = reference.get("cards", {})
    acc_all, acc_recent, acc_heur = card_gih(net), card_gih(recent), card_gih(heuristic)
    cards = []
    for c in sorted(set(acc_all) | set(acc_heur) | set(ref_cards)):
        n, w = acc_all.get(c, (0, 0))
        nr, wr_ = acc_recent.get(c, (0, 0))
        nh, wh = acc_heur.get(c, (0, 0))
        ref = ref_cards.get(c, {})
        cards.append({"card": c, "games": n, "wr": w / n if n else None,
                      "recent_games": nr, "recent_wr": wr_ / nr if nr else None,
                      "heuristic_games": nh, "heuristic_wr": wh / nh if nh else None,
                      "ref_wr": ref.get("gih_wr"), "ref_games": ref.get("gih_games")})

    ref_colors = reference.get("colors", {})
    col_all, col_recent, col_heur = color_records(net, meta), color_records(recent, meta), color_records(heuristic, meta)
    colors = []
    for c in sorted(set(col_all) | set(col_heur), key=lambda c: -(col_all.get(c, [0])[0] + col_heur.get(c, [0])[0])):
        n, w = col_all.get(c, (0, 0))
        p, lo, hi = wilson(w, n)
        nr, wr_ = col_recent.get(c, (0, 0))
        nh, wh = col_heur.get(c, (0, 0))
        colors.append({"colors": c, "games": n, "wins": w, "wr": p, "lo": lo, "hi": hi,
                       "recent_games": nr, "recent_wr": wr_ / nr if nr else None,
                       "heuristic_games": nh, "heuristic_wr": wh / nh if nh else None,
                       "ref_wr": ref_colors.get(c, {}).get("wr")})

    rho_all, n_all = gih_correlation(acc_all, reference, min_card_games)
    rho_recent, n_recent = gih_correlation(acc_recent, reference, min_card_games)
    rho_heur, n_heur = gih_correlation(acc_heur, reference, min_card_games)
    return {
        "window_gens": window_gens, "min_card_games": min_card_games,
        "selfplay_games": {"network": len(net), "recent": len(recent), "heuristic": len(heuristic)},
        "gih_rho": {"network": rho_all, "network_cards": n_all, "recent": rho_recent, "recent_cards": n_recent,
                    "heuristic": rho_heur, "heuristic_cards": n_heur},
        "rho_series": rho_series,
        "cards": cards,
        "colors": colors,
        "decks_seen": len(deck_records(games)),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--meta", default="assets/decks.tsv")
    ap.add_argument("--reference", default="assets/reference/FDN_gih.json")
    args = ap.parse_args()
    rows = [json.loads(l) for l in (Path(args.run_dir) / "games.jsonl").read_text().splitlines() if l.strip()]
    s = summarize(rows, load_deck_meta(Path(args.meta)), load_reference(Path(args.reference)))
    print(json.dumps({k: s[k] for k in ("selfplay_games", "gih_rho", "decks_seen")}, indent=1))
    for c in s["colors"][:12]:
        print(f"{c['colors']:6s} {c['games']:5d} games  wr {c['wr'] if c['wr'] is None else round(c['wr'], 3)}  "
              f"17lands {c['ref_wr'] if c['ref_wr'] is None else round(c['ref_wr'], 3)}")
