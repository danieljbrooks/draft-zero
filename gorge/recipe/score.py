"""Side A's score in paired games: games.jsonl files from `dzgorge play` -> score, SE across deck pairs, n.

    python gorge/recipe/score.py RUN.jsonl [...] [--json]
"""
import json
import math
import sys
from collections import defaultdict


def score(paths):
    pairs = defaultdict(list)
    for p in paths:
        for line in open(p):
            if line.strip():
                g = json.loads(line)
                r = g.get("result")
                pairs[(p, g["pair"])].append(1.0 if r == "A" else 0.0 if r == "B" else 0.5)
    full = [sum(v) / len(v) for v in pairs.values() if len(v) == 2]
    n = len(full)
    if not n:
        return {"score": None, "se": None, "pairs": 0, "games": 0}
    m = sum(full) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in full) / max(n - 1, 1))
    return {"score": m, "se": sd / math.sqrt(n), "pairs": n, "games": 2 * n}


if __name__ == "__main__":
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    s = score(files)
    if "--json" in sys.argv:
        print(json.dumps(s))
    elif s["pairs"]:
        print(f"{100 * s['score']:.1f}% ± {100 * s['se']:.1f} of {s['games']} games ({s['pairs']} pairs)")
    else:
        print("no complete pairs")
