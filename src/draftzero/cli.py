"""cli.py — the `dz` entry point.

    dz version                    the installed DraftZero version
    dz provenance                 what code is in play here (draftzero.manifest.collect without a config): DraftZero's
                                  commit, the gorge pin and build, card lock, python, torch, the machine

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
        from draftzero import manifest
        print(json.dumps(manifest.collect(None), indent=2, default=str))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
