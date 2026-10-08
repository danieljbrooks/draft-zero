"""docs/028: a network between two others: (1 - alpha) x A + alpha x B, weight by weight (WiSE-FT, Wortsman et al.
2022). A free dial between the start network (A) and a self-play network (B), scored with games, no retraining.

    python tools/selfplay_transition/interp.py --a $M0 --b runs/sp/arms/cq30/final.pt.gz --alpha 0.5 --out mid.pt.gz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--alpha", type=float, required=True, help="0: A, 1: B")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from draftzero.gameplay import graph_supervised as gs
    ma, va, ea, _ = gs.load_checkpoint(a.a)
    mb, vb, eb, _ = gs.load_checkpoint(a.b)
    if not (np.array_equal(np.asarray(va.ids), np.asarray(vb.ids)) and np.array_equal(np.asarray(ea.ids), np.asarray(eb.ids))):
        raise SystemExit("the two networks have different vocabularies")
    sa, sb = ma.state_dict(), mb.state_dict()
    out = {k: (sa[k].float() * (1 - a.alpha) + sb[k].float() * a.alpha).to(sa[k].dtype) if sa[k].is_floating_point()
           else sa[k] for k in sa}
    ma.load_state_dict(out)
    gs.save_weights(a.out, ma, va, ea, {"interp": {"a": a.a, "b": a.b, "alpha": a.alpha}})
    print(a.out)


if __name__ == "__main__":
    main()
