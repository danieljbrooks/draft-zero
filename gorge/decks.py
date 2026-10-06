"""
Write DraftZero's FDN deck pool as gorge deck files.

gorge reads a deck as bare {name, count} JSON (its `deck` package). This converts the decks that
`tools/extract_decks.py` pulls from 17lands into that format, one file per deck, under the same
stems the XMage pool uses (`FDN_top_00001_WB`), so the train/eval split in `assets/decks.tsv`
applies unchanged.

  python tools/extract_decks.py --set FDN --format PremierDraft --min-winrate 0.60
  python gorge/decks.py            # -> data/gorge/decks/*.json and data/gorge/pool.tsv

Deck data: 17lands public datasets, CC BY 4.0.
"""
import argparse
import csv
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def stem(i: int, d: dict, set_code: str) -> str:
    # The same stem tools/extract_decks.py gives the .dck file (write_dck): the deck's 1-based
    # position in the sorted decks.jsonl, then main+splash colours.
    colors = d["main_colors"] + (f"+{d['splash_colors']}" if d["splash_colors"] else "")
    return f"{set_code}_top_{i:05d}_{(colors or 'C').replace('+', 'p')}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decks", default=os.path.join(REPO, "data/deckgen/FDN_PremierDraft_wr60/decks.jsonl"))
    ap.add_argument("--split", default=os.path.join(REPO, "assets/decks.tsv"))
    ap.add_argument("--out", default=os.path.join(REPO, "data/gorge"))
    ap.add_argument("--set", default="FDN")
    args = ap.parse_args()

    split = {r["deck"]: r for r in csv.DictReader(open(args.split), delimiter="\t")}
    out_dir = os.path.join(args.out, "decks")
    os.makedirs(out_dir, exist_ok=True)
    rows, missing = [], 0
    with open(args.decks) as f:
        for i, line in enumerate(f, 1):
            d = json.loads(line)
            s = stem(i, d, args.set)
            if s not in split:
                missing += 1
                continue
            deck = {
                "name": s,
                "format": "limited",
                "notes": f"17lands {args.set} PremierDraft deck {d['draft_id']}:{d['build_index']}, "
                         f"player win-rate bucket {d['user_game_win_rate_bucket']:.2f} (CC BY 4.0)",
                "cards": [{"name": c, "count": n} for c, n in sorted(d["cards"].items())],
            }
            with open(os.path.join(out_dir, s + ".json"), "w") as g:
                json.dump(deck, g, indent=0)
            rows.append((s, split[s]["split"], split[s]["colors"], split[s]["main_colors"], sum(d["cards"].values())))
    with open(os.path.join(args.out, "pool.tsv"), "w") as f:
        f.write("deck\tsplit\tcolors\tmain_colors\tcards\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    print(f"wrote {len(rows):,} decks to {out_dir} ({sum(r[1] == 'train' for r in rows):,} train, "
          f"{sum(r[1] == 'eval' for r in rows):,} eval); {missing} not in {args.split}", file=sys.stderr)
    if len(rows) != len(split):
        sys.exit(f"expected {len(split):,} decks from {args.split}, wrote {len(rows):,}")


if __name__ == "__main__":
    main()
