"""
Split a deck set (from tools/extract_decks.py --dck) into train / held-out eval deck pools.

  python -m draftzero.pools --decks data/decks --out data/pools/

The split is by draft_id (every build of a draft lands on the same side), using a stable hash, so
it is reproducible and independent of file order. Writes:
  train.txt / eval.txt       one absolute .dck path per line (the XMage deck_pool format)
  decks.tsv                  deck file -> split, colors, player bucket (for the stats)
"""
import argparse
import hashlib
import json
import os


def is_eval(draft_id: str, eval_frac: float) -> bool:
    h = int(hashlib.sha256(draft_id.encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < eval_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decks", required=True, help="deckgen output dir containing decks.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-frac", type=float, default=0.10)
    args = ap.parse_args()

    deck_dir = next(os.path.join(args.decks, d) for d in os.listdir(args.decks) if d.startswith("top_player_"))
    os.makedirs(args.out, exist_ok=True)
    train, evals, rows = [], [], []
    for line in open(os.path.join(args.decks, "decks.jsonl")):
        d = json.loads(line)
        if "dck_file" not in d:
            continue
        path = os.path.abspath(os.path.join(deck_dir, d["dck_file"]))
        split = "eval" if is_eval(d["draft_id"], args.eval_frac) else "train"
        (evals if split == "eval" else train).append(path)
        colors = d["main_colors"] + (f"+{d['splash_colors']}" if d["splash_colors"] else "")
        rows.append((d["dck_file"][:-4], split, colors, d["main_colors"], str(d["user_game_win_rate_bucket"])))
    for name, paths in (("train.txt", train), ("eval.txt", evals)):
        with open(os.path.join(args.out, name), "w") as f:
            f.write("\n".join(paths) + "\n")
    with open(os.path.join(args.out, "decks.tsv"), "w") as f:
        f.write("deck\tsplit\tcolors\tmain_colors\tplayer_wr_bucket\n")
        f.writelines("\t".join(r) + "\n" for r in rows)
    print(f"{len(train):,} train decks, {len(evals):,} eval decks -> {args.out}")


if __name__ == "__main__":
    main()
