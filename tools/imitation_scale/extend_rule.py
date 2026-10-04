"""Whether to extend the full-data MLP by another epoch (docs/018, "Extending run 2"; rules fixed before the results).

Each run directory is one epoch, in order (run 2 first); its last validation row (evals.jsonl) is its result. Walking
the epochs, the best so far is replaced only by an epoch that lowers the validation set NLL by at least 0.001, and an
epoch whose value log-loss is more than 0.01 above the first epoch's must also gain 0.002. The first epoch that fails
stops the extension, as does reaching --max-epochs. Prints "continue BEST" or "stop BEST" (BEST: the best run's
directory) and the table.

    python tools/imitation_scale/extend_rule.py runs/exp4/mlp_1ep runs/exp4/mlp_ext2 ... [--max-epochs 20]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MIN_GAIN = 0.001          # set NLL an epoch must gain over the best so far
VALUE_SLACK = 0.01        # value log-loss allowed above the first epoch's ...
VALUE_GAIN = 0.002        # ... unless the policy gains this much


def last_eval(run: Path) -> dict:
    rows = [json.loads(x) for x in (Path(run) / "evals.jsonl").read_text().splitlines() if x.strip()]
    return rows[-1]


def decide(results: list[dict], max_epochs: int) -> tuple[str, int, list[str]]:
    """results: one {"policy/set_nll", "value/logloss"} per epoch, in order. Returns (decision, best index, notes)."""
    best, base_vll, notes = 0, results[0]["value/logloss"], []
    for i, r in enumerate(results[1:], 1):
        gain = results[best]["policy/set_nll"] - r["policy/set_nll"]
        if gain < MIN_GAIN:
            notes.append(f"epoch {i + 1}: gain {gain:+.4f} < {MIN_GAIN}")
            return "stop", best, notes
        if r["value/logloss"] > base_vll + VALUE_SLACK and gain < VALUE_GAIN:
            notes.append(f"epoch {i + 1}: value log-loss {r['value/logloss']:.4f} > {base_vll + VALUE_SLACK:.4f} "
                         f"with a gain of only {gain:+.4f}")
            return "stop", best, notes
        notes.append(f"epoch {i + 1}: gain {gain:+.4f}")
        best = i
    if len(results) >= max_epochs:
        notes.append(f"{len(results)} epochs: the cap")
        return "stop", best, notes
    return "continue", best, notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--max-epochs", type=int, default=20)
    a = ap.parse_args(argv)
    results = [last_eval(r) for r in a.runs]
    decision, best, notes = decide(results, a.max_epochs)
    print(f"{decision} {a.runs[best]}")
    for i, (run, r) in enumerate(zip(a.runs, results)):
        print(f"  epoch {i + 1} {run}: set NLL {r['policy/set_nll']:.4f}, non-Pass {r['policy/top1_nonpass']:.4f}, "
              f"value log-loss {r['value/logloss']:.4f}, AUC {r['value/auc']:.4f}", file=sys.stderr)
    for n in notes:
        print("  " + n, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
