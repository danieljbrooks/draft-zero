"""
Write DraftZero's FDN deck pool as gorge deck files.

gorge reads a deck as bare {name, count} JSON (its `deck` package). This converts the decks that
`tools/extract_decks.py` pulls from 17lands into that format, one file per deck, under the same
stems the v0.1.0 XMage pool used (`FDN_top_00001_WB`), so the train/eval split in `assets/decks.tsv`
applies unchanged, and writes the pool.tsv (deck, split, colors, main_colors, cards) dzgorge's -pool reads.

  python tools/extract_decks.py --set FDN --format PremierDraft --min-winrate 0.60
  python -m draftzero.data.decks   # -> data/gorge/decks/*.json and data/gorge/pool.tsv

In code: `convert(decks_jsonl, split_tsv, out_dir)`. A deck in decks.jsonl that `assets/decks.tsv` does not list is
skipped and counted; a listed deck that decks.jsonl lacks is an error (the split would silently shrink).

Deck data: 17lands public datasets, CC BY 4.0.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from draftzero.data import pools
from draftzero.data.paths import REPO

DEFAULT_DECKS = REPO / "data/deckgen/FDN_PremierDraft_wr60/decks.jsonl"
DEFAULT_SPLIT = REPO / "assets/decks.tsv"
DEFAULT_OUT = REPO / "data/gorge"


def stem(i: int, d: dict, set_code: str) -> str:
    # The stem v0.1.0's XMage .dck files had (assets/decks.tsv): the deck's 1-based position in the
    # sorted decks.jsonl (tools/extract_decks.py), then main+splash colours.
    colors = d["main_colors"] + (f"+{d['splash_colors']}" if d["splash_colors"] else "")
    return f"{set_code}_top_{i:05d}_{(colors or 'C').replace('+', 'p')}"


def gorge_deck(s: str, d: dict, set_code: str) -> dict:
    """One decks.jsonl row as a gorge deck file, with its 17lands provenance in `notes`."""
    return pools.gorge_deck(
        s, d["cards"],
        notes=f"17lands {set_code} PremierDraft deck {d['draft_id']}:{d['build_index']}, "
              f"player win-rate bucket {d['user_game_win_rate_bucket']:.2f} (CC BY 4.0)")


def read_split(split_tsv: str | Path) -> dict[str, dict]:
    """assets/decks.tsv by deck stem (columns deck, split, colors, main_colors, player_wr_bucket)."""
    with open(split_tsv) as f:
        return {r["deck"]: r for r in csv.DictReader(f, delimiter="\t")}


def convert(decks_jsonl: str | Path, split_tsv: str | Path, out_dir: str | Path, set_code: str = "FDN",
            *, strict: bool = True) -> tuple[list[pools.PoolRow], int]:
    """Write out_dir/decks/<stem>.json for every deck of decks_jsonl that split_tsv lists, and out_dir/pool.tsv
    (in decks.jsonl order). Returns (pool rows, decks skipped). strict: every split_tsv deck must be written."""
    split = read_split(split_tsv)
    out_dir = Path(out_dir)
    deck_dir = out_dir / "decks"
    deck_dir.mkdir(parents=True, exist_ok=True)
    rows, missing = [], 0
    with open(decks_jsonl) as f:
        for i, line in enumerate(f, 1):
            d = json.loads(line)
            s = stem(i, d, set_code)
            if s not in split:
                missing += 1
                continue
            with open(pools.deck_path(deck_dir, s), "w") as g:
                json.dump(gorge_deck(s, d, set_code), g, indent=0)
            meta = split[s]
            rows.append(pools.PoolRow(s, meta["split"], meta["colors"], meta["main_colors"], sum(d["cards"].values())))
    pools.write(out_dir / "pool.tsv", rows)
    if strict and len(rows) != len(split):
        raise SystemExit(f"expected {len(split):,} decks from {split_tsv}, wrote {len(rows):,}")
    return rows, missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decks", default=str(DEFAULT_DECKS))
    ap.add_argument("--split", default=str(DEFAULT_SPLIT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--set", default="FDN")
    args = ap.parse_args(argv)
    rows, missing = convert(args.decks, args.split, args.out, args.set)
    n = {s: sum(r.split == s for r in rows) for s in ("train", "eval")}
    print(f"wrote {len(rows):,} decks to {Path(args.out) / 'decks'} ({n['train']:,} train, {n['eval']:,} eval); "
          f"{missing} not in {args.split}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
