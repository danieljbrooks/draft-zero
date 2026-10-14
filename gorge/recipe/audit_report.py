"""docs/031's audit: how often a bot's choice matches a deeper search's, and how much win chance it gives up.

    python gorge/recipe/audit_report.py AUDIT.jsonl [...] [--min-visits 20] [--json]

Each line of `dzgorge play -audit-out` holds the bot's search record and the oracle's at one decision (same options,
so a candidate -- a set of option positions -- names the same answer in both). For each decision:
  agree   the bot's choice is the oracle's most-visited candidate
  loss    the oracle's estimate (q) of its own choice minus its estimate of the bot's choice, when the oracle visited
          the bot's choice at least --min-visits times (else the estimate is too noisy and the decision is "unscored");
          0 when they agree. In units of win chance.
Per decision kind, and overall: decisions, agreement, mean loss, the share of decisions losing over 0.05 and 0.10
("mistakes" and "blunders"), and the loss summed per game (a game's expected win chance given away, by the oracle).
"""
import argparse
import json
import sys
from collections import defaultdict


def load(paths):
    for p in paths:
        for line in open(p):
            if line.strip():
                yield json.loads(line)


def analyse(rows, min_visits):
    by = defaultdict(lambda: {"n": 0, "agree": 0, "scored": 0, "loss": 0.0, "mistakes": 0, "blunders": 0})
    games = defaultdict(float)
    for r in rows:
        a, o = r["agent"], r["oracle"]
        if not o["cands"] or not a["cands"]:
            continue
        pick = a["cands"][a["choice"]]
        best = max(range(len(o["visits"])), key=lambda i: o["visits"][i])
        idx = next((i for i, c in enumerate(o["cands"]) if c == pick), None)
        for k in (r["kind"], "all"):
            s = by[k]
            s["n"] += 1
            if idx == best:
                s["agree"] += 1
                s["scored"] += 1
                continue
            if idx is None or o["visits"][idx] < min_visits:
                continue
            loss = max(0.0, o["q"][best] - o["q"][idx])
            s["scored"] += 1
            s["loss"] += loss
            s["mistakes"] += loss > 0.05
            s["blunders"] += loss > 0.10
            if k == "all":
                games[(r["pair"], r["leg"])] += loss
    out = {}
    for k, s in by.items():
        sc = max(s["scored"], 1)
        out[k] = {"decisions": s["n"], "agree": s["agree"] / max(s["n"], 1), "scored": s["scored"] / max(s["n"], 1),
                  "mean_loss": s["loss"] / sc, "mistakes": s["mistakes"] / sc, "blunders": s["blunders"] / sc}
    out["all"]["games"] = len(games) or None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--min-visits", type=int, default=20)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    out = analyse(load(a.files), a.min_visits)
    if a.json:
        print(json.dumps(out))
        return
    print(f"{'kind':12} {'decisions':>9} {'agree':>6} {'scored':>6} {'mean loss':>9} {'>0.05':>6} {'>0.10':>6}")
    for k in sorted(out, key=lambda k: (k == "all", -out[k]["decisions"])):
        s = out[k]
        print(f"{k:12} {s['decisions']:9d} {s['agree']:6.1%} {s['scored']:6.1%} {s['mean_loss']:9.4f} "
              f"{s['mistakes']:6.1%} {s['blunders']:6.1%}")


if __name__ == "__main__":
    main()
