"""cli.py — `dz` entry point.

    dz status                     what the local run is doing
    dz pools normalize            rewrite pool files to machine-independent deck stems
    dz dashboard runs/<id>        render the format-knowledge page
    dz workers rank               in-stock RunPod GPUs ranked by vCPU per dollar
    dz provenance                 what code is in play here
"""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dz", description="DraftZero")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("provenance", help="print the versions that define a run")

    p_pools = sub.add_parser("pools", help="deck pool maintenance")
    p_pools.add_argument("action", choices=["build", "normalize", "check"])
    p_pools.add_argument("--meta", default="assets/decks.tsv")
    p_pools.add_argument("--dir", default="data/pools")
    p_pools.add_argument("--deck-root", default="data/decks")

    p_w = sub.add_parser("workers", help="worker helpers")
    p_w.add_argument("action", choices=["rank"])
    p_w.add_argument("--min-vcpu", type=int, default=8)

    p_dash = sub.add_parser("dashboard", help="render the format-knowledge page for a run")
    p_dash.add_argument("run_dir")
    p_dash.add_argument("--meta")
    p_dash.add_argument("--reference")

    sub.add_parser("status", help="local run status")

    a = ap.parse_args(argv)

    if a.cmd == "provenance":
        from draftzero import provenance
        print(json.dumps(provenance.collect(), indent=2))
        return 0

    if a.cmd == "pools":
        from draftzero import paths
        files = [Path(a.dir) / f for f in ("train.txt", "eval.txt")]
        if a.action == "build":
            counts = paths.build_pools(Path(a.meta), Path(a.dir))
            print(f"built {a.dir}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
            return 0
        if a.action == "normalize":
            for f in files:
                if f.exists():
                    print(f"{f}: {paths.normalize_pool_file(f)} lines rewritten to stems")
            return 0
        bad = 0
        for f in files:
            if not f.exists():
                continue
            stems = paths.read_pool(f)
            missing = [s for s in stems[:200] if not paths.deck_path(s, Path(a.deck_root)).exists()]
            print(f"{f}: {len(stems)} decks, {len(missing)} of first 200 missing under {a.deck_root}")
            bad += len(missing)
        return 1 if bad else 0

    if a.cmd == "workers":
        from draftzero.workers.runpod import rank_offers
        rows = rank_offers(min_vcpu=a.min_vcpu)
        print(f"{'GPU':<22}{'$/hr':>7}{'vCPU':>6}{'RAM':>6}{'vCPU/$':>8}  datacenters")
        for r in rows[:12]:
            print(f"{r['name']:<22}{r['price']:>7}{r['vcpu']:>6}{str(r['ram'])+'G':>6}"
                  f"{r['vcpu_per_dollar']:>8}  {','.join(r['datacenters'][:3])}")
        return 0

    if a.cmd == "dashboard":
        from draftzero import dashboard
        out = dashboard.render(Path(a.run_dir),
                               meta_path=Path(a.meta) if a.meta else None,
                               reference_path=Path(a.reference) if a.reference else None)
        if out is None:
            print("no games in that run yet; nothing rendered", file=sys.stderr)
            return 1
        print(out)
        return 0

    if a.cmd == "status":
        from draftzero.workers import make_worker
        print(json.dumps(make_worker("local").status(), indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
