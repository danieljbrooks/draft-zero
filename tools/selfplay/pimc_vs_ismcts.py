"""PIMC with one belief world against IS-MCTS, on the same sb-v2 decisions and machine (docs/021 §2.5).

    python tools/selfplay/pimc_vs_ismcts.py runs/compute_bench/pimc-vs-is-3090/sb [--items data/search_bench/sb-v2]

For each budget, on the decisions both methods finished:
  cost       wall seconds a decision (what a worker spends: building the worlds, then the search), the search
             alone, building the worlds, engine steps and network evaluations a simulation; the IS-MCTS/PIMC ratio
  agreement  the balanced score (docs/016 §8.3: cast-or-pass, attack and block balanced accuracies, a constant
             answer scores 0.50) of each, and PIMC minus IS-MCTS with a 95% interval from resampling games
             (a game's decisions go together); A_set as docs/016 reported it; how often the two choose the same move
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "search_bench"))

from analyze import balanced_score, load_items, load_rows, paired  # noqa: E402
from diagnose import act_split  # noqa: E402


def balanced(items: dict, R: dict) -> float | None:
    s = act_split(items, R)
    vals = [s[g]["bal_acc"] for g in ("cast", "attack", "block") if s[g]["bal_acc"] is not None]
    return sum(vals) / len(vals) if vals else None


def paired_balanced(items: dict, A: dict, B: dict, n: int = 1000, seed: int = 0) -> tuple[float, list[float]]:
    """balanced(A) - balanced(B) on their common decisions, with a game-bootstrap 95% interval."""
    common = [i for i in A if i in B and i in items]
    d0 = balanced(items, {i: A[i] for i in common}) - balanced(items, {i: B[i] for i in common})
    by_game = defaultdict(list)
    for i in common:
        by_game[items[i]["row"]].append(i)
    games = list(by_game)
    rng = random.Random(seed)
    ds = []
    for _ in range(n):
        ids = [i for g in (rng.choice(games) for _ in games) for i in by_game[g]]
        a, b = balanced(items, {i: A[i] for i in ids}), balanced(items, {i: B[i] for i in ids})
        if a is not None and b is not None:
            ds.append(a - b)
    ds.sort()
    return d0, [ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds)) - 1]]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+", help="run directories (each with decisions/*.jsonl)")
    ap.add_argument("--items", type=Path, default=REPO / "data" / "search_bench" / "sb-v2")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    items = load_items(a.items)
    rows, _ = load_rows(a.runs)
    by = defaultdict(dict)                                  # (method, budget) -> item -> row
    run_of = {}
    for r in rows:
        by[(r["method"], r["budget"])][r["item_id"]] = r
        run_of[(r["method"], r["budget"])] = r["run_id"]
    out = []
    for b in sorted({k[1] for k in by}):
        P, I = by.get(("pimc1", b), {}), by.get(("ismcts", b), {})
        common = sorted(i for i in P if i in I and i in items)
        if not common:
            continue
        res = {"budget": b, "decisions": len(common)}
        for m, R in (("pimc1", P), ("ismcts", I)):
            rs = [R[i] for i in common]
            sims = sum(r["stats"]["sims"] for r in rs)
            res[m] = {
                "wall_s": round(mean(r["wall_s"] for r in rs), 3),
                "search_s": round(mean(r["timing_ms"]["search"] / 1000 for r in rs), 3),
                "build_s": round(mean(r["timing_ms"].get("build", 0) / 1000 for r in rs), 3),
                "ms_per_sim": round(1000 * sum(r["wall_s"] for r in rs) / sims, 2),
                "engine_steps_per_sim": round(sum(r["stats"]["engineSteps"] for r in rs) / sims, 2),
                "evals_per_sim": round(sum(r["stats"]["evals"] for r in rs) / sims, 3),
                "balanced": balanced_score(items, {i: R[i] for i in common}, n_boot=200)[:2],
            }
        res["ismcts_over_pimc"] = {k: round(res["ismcts"][k] / res["pimc1"][k], 2)
                                   for k in ("wall_s", "search_s") if res["pimc1"][k]}
        d, ci = paired_balanced(items, P, I)
        res["balanced_pimc_minus_ismcts"] = {"diff": round(d, 4), "ci": [round(ci[0], 4), round(ci[1], 4)]}
        res["A_set_pimc_minus_ismcts"] = paired(rows, items, run_of[("pimc1", b)], run_of[("ismcts", b)])
        res["same_move"] = round(mean(float(P[i].get("best") == I[i].get("best")) for i in common), 3)
        out.append(res)
        p, s = res["pimc1"], res["ismcts"]
        print(f"budget {b}: {len(common)} decisions")
        print(f"  wall s/decision   PIMC {p['wall_s']:.2f}  IS-MCTS {s['wall_s']:.2f}  ratio {res['ismcts_over_pimc'].get('wall_s')}")
        print(f"  search s/decision PIMC {p['search_s']:.2f}  IS-MCTS {s['search_s']:.2f}  ratio {res['ismcts_over_pimc'].get('search_s')}")
        print(f"  build s/decision  PIMC {p['build_s']:.2f}  IS-MCTS {s['build_s']:.2f}")
        print(f"  engine steps/sim  PIMC {p['engine_steps_per_sim']}  IS-MCTS {s['engine_steps_per_sim']};"
              f" evals/sim PIMC {p['evals_per_sim']}  IS-MCTS {s['evals_per_sim']}")
        print(f"  balanced          PIMC {p['balanced']}  IS-MCTS {s['balanced']};"
              f" PIMC - IS {res['balanced_pimc_minus_ismcts']}")
        print(f"  A_set PIMC - IS   {res['A_set_pimc_minus_ismcts']['diff']} {res['A_set_pimc_minus_ismcts']['ci']};"
              f" same move {res['same_move']}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
