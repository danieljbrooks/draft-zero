"""Train a dzg network on searched decisions.

  python -m dzg.train --arch mlp --train DIR [DIR ...] --out OUT [--val eval=DIR] [--epochs 3]

Writes OUT/config.json, OUT/curves.jsonl (one line per log interval and per evaluation), OUT/best.pt
(lowest holdout policy CE + value log loss) and OUT/last.pt, and prints a final summary JSON line.
The holdout is the training directories' games with crc32(game id) % 20 == 0.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import data as D
from . import models as M
from . import targets as T

DEFAULT_LR = {"mlp": 1e-3, "transformer": 3e-4, "gnn": 3e-4}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="dzg.train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=M.ARCHS, help="default mlp, or --init's arch")
    ap.add_argument("--config", default=None, help="architecture overrides: a JSON object, e.g. '{\"d\": 256}', or a JSON file")
    ap.add_argument("--train", nargs="+", required=True, help="shard directories (or dzgorge pack outputs)")
    ap.add_argument("--val", action="append", default=[], metavar="NAME=DIR", help="extra validation set (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=None, help="default 1e-3 mlp, 3e-4 transformer and gnn")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--min-lr-frac", type=float, default=0.1, help="cosine decays to this fraction of --lr")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--amp", choices=("bf16", "off"), default="bf16", help="bf16 autocast (CUDA only)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--policy-target", choices=("visits", "cq"), default="visits")
    ap.add_argument("--cq-c-visit", type=float, default=50.0)
    ap.add_argument("--cq-c-scale", type=float, default=0.1)
    ap.add_argument("--visits-temp", type=float, default=1.0)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--value-blend", type=float, default=0.0, help="value target (1-x) outcome + x root_value")
    ap.add_argument("--eval-every", type=int, default=None, help="steps (default: 4 times an epoch)")
    ap.add_argument("--eval-max", type=int, default=100000, help="records per evaluation set (fixed subsample)")
    ap.add_argument("--eval-batch", type=int, default=1024)
    ap.add_argument("--log-every", type=int, default=50, help="steps between training log lines")
    ap.add_argument("--patience", type=int, default=0, help="stop after K evaluations without a holdout improvement (0 = never)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="warm start from a checkpoint (its arch and config)")
    ap.add_argument("--in-ram", action="store_true", help="load the shards into RAM (merged into one pack, so a batch "
                    "is one gather however many shards it spans) instead of memory-mapping them")
    ap.add_argument("--limit-records", type=int, default=None, help="use only the first N training records")
    ap.add_argument("--workers", type=int, default=2, help="collate threads (more than 2 rarely helps: the GIL)")
    ap.add_argument("--loader-procs", type=int, default=0,
                    help="collate in N worker processes instead of threads (scales with cores; batches pass "
                         "through /dev/shm)")
    return ap.parse_args(argv)


# ------------------------------------------------------------------------------------------------
# losses and metrics
# ------------------------------------------------------------------------------------------------

def seg_sum(x, seg, n):
    return x.new_zeros(n).index_add(0, seg, x)


def seg_argmax(x, seg, n):
    """Global index of the first maximum of x within each segment (len(x) for an empty segment).
    The index reduction runs in float32 (exact below 2^24 candidates per batch): MPS has no int64 atomic min."""
    big = x.shape[0]
    x = x.float()
    m = x.new_full((n,), float("-inf")).scatter_reduce(0, seg, x, "amax")
    idx = torch.arange(big, device=x.device, dtype=torch.float32)
    hit = torch.where(x == m[seg], idx, torch.full_like(idx, big))
    return x.new_full((n,), float(big)).scatter_reduce(0, seg, hit, "amin").long()


def losses(model, b, args):
    """(total, policy CE, value log loss, extras) for one batch."""
    scores, vlogit = model(b)
    z, lp = M.candidate_logits(scores, b)
    tgt = T.policy_target(b, args.policy_target, args.visits_temp, args.cq_c_visit, args.cq_c_scale)
    pm = T.policy_mask(b)
    ce_state = seg_sum(-(tgt * lp), b.cand_state, b.B)
    npol = pm.sum()
    pce = torch.where(pm, ce_state, torch.zeros_like(ce_state)).sum() / npol.clamp(min=1)
    vt, vm = T.value_target(b, args.value_blend)
    vl = F.binary_cross_entropy_with_logits(vlogit.float(), vt, reduction="none")
    nval = vm.sum()
    vloss = torch.where(vm, vl, torch.zeros_like(vl)).sum() / nval.clamp(min=1)
    total = pce + args.value_weight * vloss
    return total, pce, vloss, {"scores": scores, "vlogit": vlogit, "z": z, "lp": lp, "tgt": tgt, "pm": pm,
                               "ce_state": ce_state, "npol": npol, "nval": nval}


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC AUC (ties get average ranks); nan without both classes."""
    pos = labels == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s), np.float64)
    _, start, count = np.unique(s, return_index=True, return_counts=True)
    avg = start + (count + 1) / 2.0
    ranks[order] = np.repeat(avg, count)
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


@torch.inference_mode()
def evaluate(model, ds: D.ShardSet, idx: np.ndarray, args, device) -> dict:
    model.eval()
    acc = {k: 0.0 for k in ("ce", "uniform", "entropy", "top1_visits", "top1_bot", "npol")}
    vlogits, vtargets = [], []
    loader = D.Loader(ds, D.batches(idx, args.eval_batch, shuffle=False), prefetch=4, workers=args.workers,
                      pin=device.type == "cuda", procs=args.loader_procs)
    for b in loader:
        b = b.to(device)
        with M.autocast(device, args.amp):
            _, _, _, x = losses(model, b, args)
        pm = x["pm"]
        acc["npol"] += float(pm.sum())
        acc["ce"] += float(x["ce_state"][pm].sum())
        k = seg_sum(torch.ones_like(b.cand_visits), b.cand_state, b.B)
        acc["uniform"] += float(torch.log(k.clamp(min=1))[pm].sum())
        t = x["tgt"]
        ent = seg_sum(-(t * torch.log(t.clamp(min=1e-30))), b.cand_state, b.B)
        acc["entropy"] += float(ent[pm].sum())
        am = seg_argmax(x["z"], b.cand_state, b.B)
        av = seg_argmax(b.cand_visits.float(), b.cand_state, b.B)
        first = (k.cumsum(0) - k).long()   # each state's candidate 0 (the bot's answer)
        acc["top1_visits"] += float(((am == av) & pm).sum())
        acc["top1_bot"] += float(((am == first) & pm).sum())
        vt, vm = T.value_target(b, 0.0)   # value metrics against the game outcome
        vlogits.append(x["vlogit"].float()[vm].cpu().numpy())
        vtargets.append(vt[vm].cpu().numpy())
    model.train()
    n = max(acc["npol"], 1)
    out = {"n_policy": int(acc["npol"]), "policy_ce": acc["ce"] / n, "policy_ce_uniform": acc["uniform"] / n,
           "target_entropy": acc["entropy"] / n, "top1_visits": acc["top1_visits"] / n, "top1_bot": acc["top1_bot"] / n}
    vl = np.concatenate(vlogits) if vlogits else np.zeros(0, np.float32)
    vt = np.concatenate(vtargets) if vtargets else np.zeros(0, np.float32)
    out["n_value"] = int(len(vt))
    if len(vt):
        vl64, vt64 = vl.astype(np.float64), vt.astype(np.float64)
        p = 1 / (1 + np.exp(-vl64))
        out["value_logloss"] = float(np.mean(np.logaddexp(0, vl64) - vt64 * vl64))
        out["value_brier"] = float(np.mean((p - vt64) ** 2))
        base = float(np.clip(vt64.mean(), 1e-6, 1 - 1e-6))
        out["value_base_rate"] = base
        out["value_base_logloss"] = float(-np.mean(vt64 * np.log(base) + (1 - vt64) * np.log(1 - base)))
        dec = (vt64 == 0) | (vt64 == 1)
        out["value_auc"] = auc(vl64[dec], vt64[dec])
    else:
        out.update(value_logloss=float("nan"), value_brier=float("nan"), value_base_rate=float("nan"),
                   value_base_logloss=float("nan"), value_auc=float("nan"))
    out["total"] = out["policy_ce"] + (out["value_logloss"] if out["n_value"] else 0.0)
    return out


# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------

def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, Path):
        return str(o)
    return o


def _optimizer(model: nn.Module, lr: float, wd: float, device):
    emb = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    decay = [p for p in model.parameters() if p.ndim >= 2 and id(p) not in emb]
    other = [p for p in model.parameters() if not (p.ndim >= 2 and id(p) not in emb)]
    groups = [{"params": decay, "weight_decay": wd}, {"params": other, "weight_decay": 0.0}]
    kw = {"fused": True} if device.type == "cuda" else {}
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), **kw)


def _lr_at(step: int, peak: float, warmup: int, total: int, min_frac: float) -> float:
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    span = max(1, total - warmup)
    t = min(1.0, max(0.0, (step - warmup) / span))
    return peak * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def _subsample(idx: np.ndarray, k: int | None, seed: int) -> np.ndarray:
    if k is None or len(idx) <= k:
        return idx
    return np.sort(np.random.default_rng(seed).choice(idx, k, replace=False))


def main(argv=None) -> dict:
    args = parse_args(argv)
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = M.pick_device(args.device)
    overrides = {}
    if args.config:   # inline JSON, or a path to a JSON file
        overrides = json.loads(args.config if args.config.lstrip().startswith("{") else Path(args.config).read_text())

    # data
    ds = D.ShardSet(args.train, in_ram=args.in_ram)
    tr_idx, ho_idx = ds.split(limit=args.limit_records)
    ho_idx = _subsample(ho_idx, args.eval_max, args.seed + 1)
    vals = []
    for spec in args.val:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--val {spec!r}: expected NAME=DIR")
        vds = D.ShardSet([path], in_ram=args.in_ram)
        vals.append((name, vds, _subsample(np.arange(len(vds), dtype=np.int64), args.eval_max, args.seed + 2)))
    if len(tr_idx) == 0:
        raise SystemExit("no training records")

    # model
    init_meta = None
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=True)
        if args.arch and args.arch != ck["arch"]:
            raise SystemExit(f"--arch {args.arch} but --init is a {ck['arch']} checkpoint")
        arch = ck["arch"]
        model = M.build(arch, {**ck["config"], **overrides})
        model.load_state_dict(ck["state_dict"])
        init_meta = {"path": str(args.init), "meta": ck.get("meta", {})}
    else:
        arch = args.arch or "mlp"
        model = M.build(arch, overrides)
    model.to(device).train()
    params = model.parameter_counts()
    lr = args.lr if args.lr is not None else DEFAULT_LR[arch]
    opt = _optimizer(model, lr, args.wd, device)

    steps_per_epoch = max(1, len(tr_idx) // args.batch) if len(tr_idx) >= args.batch else 1
    total_steps = max(1, int(math.ceil(args.epochs * steps_per_epoch)))
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    eval_every = args.eval_every or max(1, steps_per_epoch // 4)

    config = {"args": vars(args), "arch": arch, "model_config": model.config, "params": params, "lr": lr,
              "device": str(device), "n_train": int(len(tr_idx)), "n_holdout": int(len(ho_idx)),
              "val": {n: int(len(i)) for n, _, i in vals}, "shards": [str(d) for d in ds.dirs],
              "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "eval_every": eval_every,
              "init": init_meta, "torch": torch.__version__}
    (out / "config.json").write_text(json.dumps(_jsonable(config), indent=2))
    curves = open(out / "curves.jsonl", "w")

    def log(rec):
        curves.write(json.dumps(_jsonable(rec)) + "\n")
        curves.flush()

    print(f"dzg.train: {arch} {params['total']:,} params on {device}; {len(tr_idx):,} train, {len(ho_idx):,} holdout "
          f"records; {total_steps} steps ({steps_per_epoch}/epoch), eval every {eval_every}", file=sys.stderr, flush=True)

    best = {"total": float("inf"), "step": None, "metrics": None}
    since_best = 0
    step, samples, epoch = 0, 0, 0
    stop_reason = "steps"
    win = {"n": 0, "pce": 0.0, "vl": 0.0, "samples": 0, "t": time.time(), "npol": 0.0, "nval": 0.0}

    def do_eval():
        nonlocal since_best
        model_meta = {"step": step, "epoch": step / steps_per_epoch, "samples": samples}
        res = {}
        if len(ho_idx):
            res["holdout"] = evaluate(model, ds, ho_idx, args, device)
        for name, vds, vidx in vals:
            res[name] = evaluate(model, vds, vidx, args, device)
        log({"type": "eval", **model_meta, "wall": time.time() - t0, "eval": res})
        # early stopping and best.pt follow the holdout (or, without one, the first --val set)
        key_set = "holdout" if "holdout" in res else (vals[0][0] if vals else None)
        key = res[key_set]["total"] if key_set else float("nan")
        meta = {**model_meta, "metrics": res, "args": vars(args), "wall": time.time() - t0}
        improved = math.isfinite(key) and key < best["total"]
        if improved or key_set is None or best["step"] is None:
            if improved:
                best["total"] = key
            best["step"], best["metrics"] = step, res
            since_best = 0
            M.save(model, out / "best.pt", _jsonable(meta))
        else:
            since_best += 1
        M.save(model, out / "last.pt", _jsonable(meta))
        h = res.get("holdout", {})
        print(f"  eval step {step}: holdout policy CE {h.get('policy_ce', float('nan')):.4f} "
              f"(uniform {h.get('policy_ce_uniform', float('nan')):.4f}), value log loss "
              f"{h.get('value_logloss', float('nan')):.4f} (base {h.get('value_base_logloss', float('nan')):.4f}), "
              f"AUC {h.get('value_auc', float('nan')):.3f}", file=sys.stderr, flush=True)
        return args.patience > 0 and since_best >= args.patience

    last_eval = -1
    eval_time = 0.0
    loop_start = time.time()
    done = False
    while not done:
        bl = D.batches(tr_idx, args.batch, shuffle=True, seed=args.seed * 1000 + epoch, drop_last=len(tr_idx) >= args.batch)
        for b in D.Loader(ds, bl, prefetch=6, workers=args.workers, pin=device.type == "cuda",
                          procs=args.loader_procs):
            b = b.to(device)
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, lr, args.warmup, total_steps, args.min_lr_frac)
            with M.autocast(device, args.amp):
                total, pce, vloss, x = losses(model, b, args)
            opt.zero_grad(set_to_none=True)
            total.backward()
            if args.clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            step += 1
            samples += b.B
            win["n"] += 1
            win["samples"] += b.B
            # accumulate on the device: no host sync per step
            win["pce"] = win["pce"] + pce.detach() * x["npol"]
            win["vl"] = win["vl"] + vloss.detach() * x["nval"]
            win["npol"] = win["npol"] + x["npol"]
            win["nval"] = win["nval"] + x["nval"]
            if step % args.log_every == 0 or step == total_steps:
                now = time.time()
                npol, nval = float(win["npol"]), float(win["nval"])
                log({"type": "train", "step": step, "epoch": step / steps_per_epoch, "samples": samples,
                     "wall": now - t0, "samples_per_s": win["samples"] / max(now - win["t"], 1e-9),
                     "policy_ce": float(win["pce"]) / max(npol, 1), "value_logloss": float(win["vl"]) / max(nval, 1),
                     "lr": opt.param_groups[0]["lr"]})
                win = {"n": 0, "pce": 0.0, "vl": 0.0, "samples": 0, "t": now, "npol": 0.0, "nval": 0.0}
            if step % eval_every == 0 or step == total_steps:
                last_eval = step
                te = time.time()
                stop = do_eval()
                eval_time += time.time() - te
                win["t"] += time.time() - te      # the window's samples/s excludes evaluation
                if stop:
                    stop_reason, done = "patience", True
                    break
            if step >= total_steps:
                done = True
                break
        epoch += 1
    if last_eval != step:
        te = time.time()
        do_eval()
        eval_time += time.time() - te
    train_time = time.time() - loop_start - eval_time
    curves.close()
    wall = time.time() - t0
    summary = {"arch": arch, "params": params["total"], "device": str(device), "steps": step, "samples": samples,
               "epochs": step / steps_per_epoch, "wall_s": wall, "train_s": train_time,
               "train_samples_per_s": samples / max(train_time, 1e-9),
               "stopped": stop_reason, "best_step": best["step"], "best": best["metrics"], "out": str(out)}
    print(json.dumps(_jsonable(summary)), flush=True)
    return summary


if __name__ == "__main__":
    main()
