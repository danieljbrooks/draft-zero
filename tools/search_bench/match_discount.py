"""Per-action and per-turn discounts matched to 0.95 per ply (docs/012 §2.1, E2b): from a run's
trees, plies per action = edges / edges out of priority decisions, plies per turn = edges / turns
crossed (both over every backed-up simulation); the matched discount is 0.95 ** that.

    python tools/search_bench/match_discount.py <run dir> <run id>   ->  "<d_action> <d_turn>"
"""
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in open(Path(sys.argv[1]) / "decisions" / f"{sys.argv[2]}.jsonl")]
st = [r["stats"] for r in rows if r.get("stats")]
e = sum(s["edgeVisits"] for s in st)
p = sum(s["priorityEdgeVisits"] for s in st)
t = sum(s["turnEdgeSum"] for s in st)
print(f"{0.95 ** (e / p):.4f} {0.95 ** (e / t):.4f}")
print(f"plies/action {e / p:.3f}, plies/turn {e / t:.2f}, from {len(st)} searches", file=sys.stderr)
