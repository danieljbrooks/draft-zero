"""E1's table: each arm's best checkpoint (by its training holdout), scored on the eval decks' games."""
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/gorge/runs/r31/e1")
print(f"{'arm':14} {'epochs':>6} {'val logloss':>11} {'AUC':>6} {'policy CE':>9} {'(uniform)':>9}")
for d in sorted(root.iterdir(), key=lambda d: [int(x) for x in re.findall(r"\d+", d.name)]):
    f = d / "curves.jsonl"
    if not f.exists():
        continue
    ev = [r for r in map(json.loads, open(f)) if r.get("type") == "eval"]
    if not ev:
        continue
    b = min(ev, key=lambda r: r["eval"]["holdout"]["total"])
    e = b["eval"]["evaldecks"]
    done = "" if (d / "best.dzgw").exists() else " (training)"
    print(f"{d.name:14} {b['epoch']:6.2f} {e['value_logloss']:11.4f} {e['value_auc']:6.3f} {e['policy_ce']:9.4f} "
          f"{e['policy_ce_uniform']:9.4f}{done}")
