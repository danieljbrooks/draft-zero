"""How a network does on rarely seen decisions and states (docs/018, the MLP follow-ups). On the validation tables
(up to --rows a table, the same rows for every network):

  by the human's move: each non-Pass decision is bucketed by how often its human move's action slot is chosen in
      the training split (the most frequent slot of the human's set, so a decision counts as rare only when every
      acceptable move is rare); top-1 among the non-Pass options and the set NLL over all legal options, per bucket.
      Priority tables use the player-priority head, target tables (spell targets, blocks) the target head.
  by the state: each decision is bucketed by the share of its raw features outside the reference vocabulary
      (--ref-vocab, default stage 3's: features seen in more than 10 states of all training games), in quartiles
      of that share over the evaluated rows (the same edges for every network).

    python tools/imitation_scale/rare_eval.py CKPT [CKPT ...] --out runs/exp4/rare_eval [--rows 20000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
PRIORITY = ("turnstart", "replay_priority", "opp_priority")
TARGET = ("replay_target", "opp_block")
PASS = 0
MOVE_BUCKETS = ((0, 1_000, "<1k"), (1_000, 10_000, "1k-10k"), (10_000, 100_000, "10k-100k"), (100_000, 10 ** 12, "100k+"))


def slot_counts(tables_dir: Path, names) -> np.ndarray:
    """Training-split count of each action slot in the human's label sets (non-Pass), over the given tables."""
    c = np.zeros(1024, np.int64)
    for t in names:
        with h5py.File(tables_dir / f"{t}_train.h5", "r") as f:
            s = f["set_idx"][:]
        c += np.bincount(s[s != PASS] % 1024, minlength=1024)
    return c


def load_rows(tables_dir: Path, t: str, n: int, seed: int = 0):
    f = h5py.File(tables_dir / f"{t}_val.h5", "r")
    off = f["offsets"][:]
    k = len(off) - 1
    sel = np.sort(np.random.default_rng(seed).choice(k, min(n, k), replace=False))
    li, lp = f["legal_idx"][:], f["legal_indptr"][:]
    si, sp = f["set_idx"][:], f["set_indptr"][:]
    rows = []
    for i in sel:
        rows.append((f["indices"][off[i]:off[i + 1]], li[lp[i]:lp[i + 1]], si[sp[i]:sp[i + 1]]))
    return rows


@torch.no_grad()
def logits_for(model, vocab, states, head: str, dev, batch: int = 128):
    out = []
    for lo in range(0, len(states), batch):
        idx, off = [], []
        for ids in states[lo:lo + batch]:
            r, _ = vocab.map_bags(list(ids), [0])
            r = np.asarray(r, np.int64)
            if len(r) == 0:
                r = np.zeros(1, np.int64)
            off.append(sum(len(x) for x in idx))
            idx.append(r)
        o = model(torch.as_tensor(np.concatenate(idx), device=dev), torch.as_tensor(np.asarray(off), device=dev))
        out.append(o[0 if head == "priority" else 2].float().cpu().numpy())
    return np.concatenate(out)


def score(lg: np.ndarray, legal: np.ndarray, hum: np.ndarray):
    """(top-1 among non-Pass options or None, set NLL over all legal options or None)."""
    legal_u = np.unique(legal % 1024)
    hum = np.unique(hum % 1024)
    if not np.isin(hum, legal_u).any():
        return None, None
    z = lg[legal_u]
    p = np.exp(z - z.max()); p /= p.sum()
    nll = float(-np.log(max(p[np.isin(legal_u, hum)].sum(), 1e-12)))
    np_legal = legal_u[legal_u != PASS]
    np_hum = hum[hum != PASS]
    if len(np_hum) == 0 or len(np_legal) < 2:
        return None, nll
    best = np_legal[int(np.argmax(lg[np_legal]))]
    return float(best in set(np_hum.tolist())), nll


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tables-dir", type=Path, default=REPO / "data/imitation_scale/h5")
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--ref-vocab", default=str(REPO / "runs/exp4/main/best_policy.pt.gz"))
    a = ap.parse_args(argv)
    from draftzero.gameplay import supervised as sv
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a.out.mkdir(parents=True, exist_ok=True)
    counts = {"priority": slot_counts(a.tables_dir, PRIORITY), "target": slot_counts(a.tables_dir, TARGET)}
    _, ref_vocab, _ = sv.load_any_checkpoint(a.ref_vocab)
    data = {t: load_rows(a.tables_dir, t, a.rows) for t in PRIORITY + TARGET}
    unk = {t: np.array([float((ref_vocab.lookup(np.asarray(ids, np.int64)) < 0).mean()) if len(ids) else 0.0
                        for ids, _, _ in rows]) for t, rows in data.items()}
    q = np.quantile(np.concatenate(list(unk.values())), [0.25, 0.5, 0.75])
    STATE_BUCKETS = ((0.0, q[0], f"Q1 <{q[0]:.0%}"), (q[0], q[1], f"Q2 {q[0]:.0%}-{q[1]:.0%}"),
                     (q[1], q[2], f"Q3 {q[1]:.0%}-{q[2]:.0%}"), (q[2], 1.01, f"Q4 >{q[2]:.0%}"))
    for ck in a.checkpoints:
        model, vocab, meta = sv.load_any_checkpoint(ck, device=dev)
        rec = {"checkpoint": ck, "by_move": {}, "by_state": {}, "tables": {}}
        allm = {b[2]: [[], []] for b in MOVE_BUCKETS}
        alls = {b[2]: [[], []] for b in STATE_BUCKETS}
        for t, rows in data.items():
            head = "priority" if t in PRIORITY else "target"
            lg = logits_for(model, vocab, [r[0] for r in rows], head, dev)
            tt = [[], []]
            for j, (_, legal, hum) in enumerate(rows):
                top1, nll = score(lg[j], legal, hum)
                np_hum = np.unique(hum % 1024); np_hum = np_hum[np_hum != PASS]
                freq = int(counts[head][np_hum].max()) if len(np_hum) else None
                if top1 is not None:
                    tt[0].append(top1)
                    for lo, hi, name in MOVE_BUCKETS:
                        if lo <= freq < hi:
                            allm[name][0].append(top1); allm[name][1].append(nll)
                    for lo, hi, name in STATE_BUCKETS:
                        if lo <= unk[t][j] < hi:
                            alls[name][0].append(top1)
                if nll is not None:
                    tt[1].append(nll)
                    for lo, hi, name in STATE_BUCKETS:
                        if lo <= unk[t][j] < hi:
                            alls[name][1].append(nll)
            rec["tables"][t] = {"top1_nonpass": float(np.mean(tt[0])) if tt[0] else None, "n_nonpass": len(tt[0]),
                                "set_nll": float(np.mean(tt[1])) if tt[1] else None, "n": len(tt[1])}
        for name, (t1, nl) in allm.items():
            rec["by_move"][name] = {"n": len(t1), "top1_nonpass": float(np.mean(t1)) if t1 else None,
                                    "set_nll": float(np.mean(nl)) if nl else None}
        for name, (t1, nl) in alls.items():
            rec["by_state"][name] = {"n_nonpass": len(t1), "top1_nonpass": float(np.mean(t1)) if t1 else None,
                                     "n": len(nl), "set_nll": float(np.mean(nl)) if nl else None}
        name = Path(ck).parent.name + "-" + Path(ck).name.split(".")[0]
        (a.out / f"{name}.json").write_text(json.dumps(rec, indent=1))
        bm = "  ".join(f"{k} {v['top1_nonpass']:.3f} (n {v['n']})" for k, v in rec["by_move"].items() if v["top1_nonpass"] is not None)
        bs = "  ".join(f"{k} {v['top1_nonpass']:.3f}" for k, v in rec["by_state"].items() if v["top1_nonpass"] is not None)
        print(f"{name}\n  by move frequency (non-Pass top-1): {bm}\n  by unknown-feature share: {bs}", flush=True)
        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
