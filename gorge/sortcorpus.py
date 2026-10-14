"""
Put a visit corpus's records in game order (stable within a game), so a network trained on it doesn't depend on
the order the games finished in. dzgorge writes in game order itself; this is for corpora written before it did.

  python gorge/sortcorpus.py data/gorge/runs/az/gen0.visits.jsonl.gz data/gorge/runs/az/gen0.visits.sorted.jsonl.gz
"""
import gzip
import re
import sys
from collections import defaultdict

GAME = re.compile(rb'"game_id":"([^"]*)"')


def main(src: str, dst: str) -> None:
    games = defaultdict(list)
    n = 0
    with gzip.open(src, "rb") as f:  # reads every concatenated gzip member
        for line in f:
            m = GAME.search(line)
            if not m:
                sys.exit(f"{src}: a record without game_id")
            games[m.group(1)].append(line)
            n += 1
    with gzip.open(dst, "wb", compresslevel=6) as g:
        for gid in sorted(games):  # p<pair:07d>g<leg>: lexicographic is game order
            g.writelines(games[gid])
    print(f"{n} records from {len(games)} games -> {dst}", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:3])
