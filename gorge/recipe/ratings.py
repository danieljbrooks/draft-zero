"""docs/031's rating ladder: one Bradley-Terry fit over every paired-game match on the eval deals, so each bot gets a
rating on one scale (Elo points; gorge's heuristic bot is 0) without saturating against any one opponent.

    python gorge/recipe/ratings.py DIR [...] [--min-games 200] [--json OUT]

Reads every games file (`dzgorge play -out`) under the directories; a player is its policy spec with the network's path
replaced by a short name (B7, E13, start, ...) and the search settings every bot shares removed. Self-play files
(both sides the same spec) are skipped. Standard errors come from the fit's curvature and treat games as independent
(paired games are correlated, so they are somewhat too small).
"""
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SHARED = ("autopay", "oppnodes", "oppfull", "morekinds", "topk=6")


def netname(path):
    for pat, fmt in ((r"r1nets/(\w+)/", "{0}"), (r"loops/([A-Z])/gen0/", "start"), (r"loops/([A-Z])/gen(\d+)/", "{0}{1}"),
                     (r"e2/visits-result/", "start"), (r"e1/(w\d+-g\d+k)/", "e1-{0}"), (r"e2/([\w-]+)/", "e2-{0}"),
                     (r"e5/(\w+)/", "e5-{0}"), (r"e6/(w\d+)/", "e6-{0}")):
        m = re.search(pat, path)
        if m:
            return fmt.format(*m.groups())
    return Path(path).parent.name


def label(spec):
    if spec == "bot":
        return "bot"
    parts = spec.split(":")
    kind, keys = parts[0], []
    for p in parts[1:]:
        if p in SHARED:
            continue
        if p.startswith("gonet="):
            keys.append("net=" + netname(p[6:]))
        else:
            keys.append(p)
    return ":".join([kind] + keys)


def load(dirs):
    games = []
    for d in dirs:
        for f in Path(d).rglob("*.jsonl"):
            if ".chunk" in f.name or "audit.jsonl" in f.name or "visits" in f.name:
                continue
            with open(f) as fh:
                first = fh.readline()
                if '"pols"' not in first:
                    continue
                fh.seek(0)
                for line in fh:
                    try:
                        g = json.loads(line)
                    except json.JSONDecodeError:   # a killed run's last, partial line
                        continue
                    if g.get("result") == "error":
                        continue
                    pa, pb = g["pols"][g["side"].index("A")], g["pols"][g["side"].index("B")]
                    if pa == pb:
                        break
                    s = {"A": 1.0, "B": 0.0}.get(g["result"], 0.5)
                    games.append((label(pa), label(pb), s))
    return games


def fit(games, min_games):
    count = defaultdict(int)
    for a, b, _ in games:
        count[a] += 1
        count[b] += 1
    keep = {p for p, c in count.items() if c >= min_games}
    games = [g for g in games if g[0] in keep and g[1] in keep]
    players = sorted(keep)
    ix = {p: i for i, p in enumerate(players)}
    A = np.array([ix[a] for a, _, _ in games])
    B = np.array([ix[b] for _, b, _ in games])
    S = np.array([s for _, _, s in games])
    n = len(players)
    r = np.zeros(n)
    anchor = ix.get("bot", 0)
    for _ in range(100):  # Newton's method on the log-likelihood, anchor fixed at 0, a tiny ridge
        p = 1 / (1 + np.exp(-(r[A] - r[B])))
        g = np.zeros(n)
        np.add.at(g, A, S - p)
        np.add.at(g, B, -(S - p))
        w = p * (1 - p)
        H = np.zeros((n, n))
        np.add.at(H, (A, A), -w)
        np.add.at(H, (B, B), -w)
        np.add.at(H, (A, B), w)
        np.add.at(H, (B, A), w)
        H -= 1e-6 * np.eye(n)
        free = [i for i in range(n) if i != anchor]
        step = np.linalg.solve(H[np.ix_(free, free)], g[free])
        r[free] -= step
        if np.abs(step).max() < 1e-9:
            break
    cov = np.linalg.inv(-H[np.ix_(free, free)])
    se = np.zeros(n)
    se[free] = np.sqrt(np.diag(cov))
    k = 400 / math.log(10)
    return [(players[i], k * r[i], k * se[i], count[players[i]]) for i in range(n)], len(games)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--min-games", type=int, default=200)
    ap.add_argument("--json")
    a = ap.parse_args()
    rows, n = fit(load(a.dirs), a.min_games)
    rows.sort(key=lambda x: -x[1])
    print(f"{n:,} games; Elo points, gorge's bot = 0")
    for p, r, se, c in rows:
        print(f"{r:7.0f} ± {se:3.0f}  {c:6d} games  {p}")
    if a.json:
        Path(a.json).write_text(json.dumps([{"player": p, "elo": r, "se": se, "games": c} for p, r, se, c in rows], indent=1))


if __name__ == "__main__":
    main()
