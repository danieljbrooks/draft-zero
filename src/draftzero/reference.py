"""
Per-card GIH ("game in hand") win rate from a 17lands public game-data file, as 17lands defines it:
the win rate of games where the card was in the opening hand or drawn, over all players.

  python -m draftzero.reference data/17lands/game_data_public.FDN.PremierDraft.csv.gz \
      assets/reference/FDN_gih.json

Used to check whether the agent's own per-card GIH win rates rank cards like human play does.
Also writes the win rate of each main color combination, for the deck-color table.
"""
import csv
import gzip
import json
import os
import sys
from collections import Counter


def main(src: str, out: str) -> None:
    csv.field_size_limit(sys.maxsize)
    games, wins = Counter(), Counter()
    color_games, color_wins = Counter(), Counter()
    rows = 0
    with gzip.open(src, "rt", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        won = header.index("won")
        main_colors = header.index("main_colors")
        cards = {}
        for i, h in enumerate(header):
            for prefix in ("opening_hand_", "drawn_"):
                if h.startswith(prefix):
                    cards.setdefault(h[len(prefix):], []).append(i)
        for row in reader:
            rows += 1
            w = row[won] == "True"
            color_games[row[main_colors]] += 1
            color_wins[row[main_colors]] += w
            for name, cols in cards.items():
                if any(row[c] not in ("", "0") for c in cols):
                    games[name] += 1
                    wins[name] += w
    ref = {name: {"gih_games": games[name], "gih_wr": wins[name] / games[name]}
           for name in sorted(games) if games[name]}
    colors = {c: {"games": color_games[c], "wr": color_wins[c] / color_games[c]}
              for c in sorted(color_games) if color_games[c] >= 100}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"source": os.path.basename(src), "games": rows, "cards": ref, "colors": colors},
              open(out, "w"), indent=1)
    top = sorted(ref.items(), key=lambda kv: -kv[1]["gih_wr"])
    print(f"{rows:,} games, {len(ref)} cards -> {out}")
    print("top:", ", ".join(f"{n} {v['gih_wr']:.3f}" for n, v in top[:5]))
    print("bottom:", ", ".join(f"{n} {v['gih_wr']:.3f}" for n, v in top[-5:]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
