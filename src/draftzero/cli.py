"""cli.py — the `dz` entry point.

    dz version                    the installed DraftZero version
    dz provenance                 what code is in play here: DraftZero's commit, the gorge pin, python, torch

The v0.2.0 commands (run, actor, learner, evaluator, controller, plan, status, stop, report, models, play, eval,
engine, pod; contracts §7) are added as their modules land.
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="dz", description="DraftZero")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version", help="print the installed DraftZero version")
    sub.add_parser("provenance", help="print the versions that define a run")
    a = ap.parse_args(argv)

    if a.cmd == "version":
        from draftzero import __version__
        print(__version__)
        return 0
    if a.cmd == "provenance":
        from draftzero import provenance
        print(json.dumps(provenance.collect(), indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
