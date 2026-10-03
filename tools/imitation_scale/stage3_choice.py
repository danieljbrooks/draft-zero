"""Stage 3's two open settings (docs/018), decided from the 30% checks by rules fixed before their
results (2026-10-03, 00:20 UTC), and written as `--set` overrides for configs/exp4_train.yml (whose
own values are the defaults: x3 on the opponent's turn and the player's later stops, value weight 0.5).

  The passivity fix's reach: x3 on all three priority tables (s30-l1-actall3-td99) only if it beats
  the default (s30-l1-actor3-td99) on the player's own later stops (replay_priority set NLL) by more
  than 0.003; the turn-start overshoot (Pass on top 0.021 against the humans' 0.051) is the price.

  The value-loss weight: 0.2 (s30-l1-actor3-vw02) only if it lowers the policy's set NLL by at least
  0.004 with non-Pass top-1 no worse, and costs the value at most 0.005 AUC and 0.008 log-loss.

    python tools/imitation_scale/stage3_choice.py --runs runs/exp4 --out runs/exp4/main/choice.json
    # prints one --set override per line (none: the config as it is)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT = "sweep_s30/runs/s30-l1-actor3-td99"
ALL3 = "sweep_s30/runs/s30-l1-actall3-td99"
VW02 = "sweep_s30b/runs/s30-l1-actor3-vw02"


def final_eval(run: Path) -> dict:
    rows = [json.loads(line) for line in (run / "evals.jsonl").read_text().splitlines() if line.strip()]
    if not (run / "summary.json").exists():
        raise SystemExit(f"{run} has not finished")
    return rows[-1]


def choose(d: dict, a: dict, v: dict) -> dict:
    """d: the default's final validation measures, a: x3 on all tables', v: value weight 0.2's."""
    sets, why = [], []
    gain = d["replay_priority/set_nll"] - a["replay_priority/set_nll"]
    if gain > 0.003:
        sets.append("act_weights={opp_priority: 3.0, replay_priority: 3.0, turnstart: 3.0}")
        why.append(f"x3 on all three tables: the player's later stops' set NLL {gain:.4f} lower than the default's")
    else:
        why.append(f"x3 on the opponent's turn and later stops (the default): all three tables gain only {gain:.4f} there")
    dn = d["policy/set_nll"] - v["policy/set_nll"]
    ok = (dn >= 0.004 and v["policy/top1_nonpass"] >= d["policy/top1_nonpass"]
          and d["value/auc"] - v["value/auc"] <= 0.005 and v["value/logloss"] - d["value/logloss"] <= 0.008)
    if ok:
        sets.append("value_weight=0.2")
    why.append(f"value weight {'0.2' if ok else '0.5 (the default)'}: set NLL {dn:+.4f} lower, non-Pass "
               f"{v['policy/top1_nonpass'] - d['policy/top1_nonpass']:+.4f}, value AUC {v['value/auc'] - d['value/auc']:+.4f}, "
               f"log-loss {v['value/logloss'] - d['value/logloss']:+.4f}")
    return {"sets": sets, "why": why}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs/exp4"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    res = choose(final_eval(a.runs / DEFAULT), final_eval(a.runs / ALL3), final_eval(a.runs / VW02))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    for line in res["sets"]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
