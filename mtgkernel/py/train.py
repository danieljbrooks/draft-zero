"""Train DzkNet on dzk-data-v1 decisions: soft cross-entropy of the policy against the search's root visit distribution
(or a one-hot), plus value_weight * MSE(v, value target). AdamW, cosine learning rate with warmup, gradient clipping;
a split by game for validation. Exports the selected epoch (--select) to the Rust format.

    uv run --no-project --python 3.12 --with torch --with numpy python mtgkernel/py/train.py \\
        --data runs/x/g0/data --out runs/x/g0/train --epochs 4 [--init runs/x/g-1/train/ckpt_best.pt]

Writes OUT/ckpt_best.pt (the selected epoch), OUT/ckpt_last.pt (the last epoch, with the optimizer state: a rerun
with the same arguments resumes from it), OUT/net.dzkn (the selected epoch), OUT/metrics.json, OUT/train_log.jsonl.

Value target (--value-target, default rootq): the game's result z overfits within an epoch or two on a few thousand
games (the value head memorizes results and, through the shared trunk, hurts the policy); the search's root value
at the decision (rootq) does not, and gave the strongest greedy policy (988 paired games against mcts:100: 0.360
rootq, 0.310 mix:0.5, 0.245 for the z network the old best-by-total-loss rule exported). mix:L = L z + (1-L) rootq.

Checkpoint selection (--select, default policy): the epoch with the lowest validation policy cross-entropy (ties to
the later epoch); `loss` = the summed validation loss (the old rule, which kept an early epoch when the value head
overfit), `last` = the last epoch.

Validation metrics: policy cross-entropy and KL against the (policy-weighted) target, top-1 agreement with the
played move and with the target's top move (strict argmax, ties to the lower index, as the Rust bot), value MSE
against the value target and against z, value AUC and log-loss vs the result (z = +-1 decisions).
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402
import model as M  # noqa: E402


def is_val(game_key, frac):
    h = int(hashlib.md5(game_key.encode()).hexdigest()[:8], 16)
    return (h % 10000) < frac * 10000


def auc(scores, labels):
    """Area under the ROC curve (ties averaged)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def first_argmax(x, seg, n_seg):
    """Per segment, the global index of its largest entry (ties to the lower index, like the Rust argmax)."""
    m = x.new_full((n_seg,), float("-inf")).scatter_reduce(0, seg, x, "amax", include_self=True)
    pos = torch.arange(len(x), device=x.device)
    cand = torch.where(x >= m[seg], pos, torch.full_like(pos, len(x)))
    return torch.full((n_seg,), len(x), device=x.device, dtype=pos.dtype).scatter_reduce(
        0, seg, cand, "amin", include_self=True)


def losses(net, b, value_weight):
    """(loss, per-decision policy cross-entropy, log-probabilities, logits, values). The policy term is weighted by
    b["pw"] (0 drops a decision's policy target), the value term by b["vw"] against b["vt"]."""
    logit, v = net(b)
    B = b["global"].shape[0]
    logp = M.segment_log_softmax(logit, b["act_dec"], B)
    ce = -(logit.new_zeros(B).index_add_(0, b["act_dec"], b["target"] * logp))
    pw = b["pw"]
    pol = (ce * pw).sum() / pw.sum().clamp(min=1.0)
    vw = b["vw"]
    vmse = ((v - b["vt"]) ** 2 * vw).sum() / vw.sum().clamp(min=1.0)
    return pol + value_weight * vmse, ce, logp, logit, v


@torch.no_grad()
def evaluate(net, P, idx, batch, value_weight, device):
    net.eval()
    tot = {"n": 0, "np": 0.0, "ce": 0.0, "ent": 0.0, "nll_chosen": 0.0, "top1": 0, "top1_target": 0,
           "uniform_ce": 0.0, "loss": 0.0}
    vs, zs, oks, vts, vws = [], [], [], [], []
    for b in P.batches(idx, batch, device=device):
        loss, ce, logp, logit, v = losses(net, b, value_weight)
        B = ce.shape[0]
        t, pw = b["target"], b["pw"]
        ent = -(logit.new_zeros(B).index_add_(0, b["act_dec"], torch.where(t > 0, t * torch.log(t.clamp(min=1e-12)), 0)))
        n_act = torch.bincount(b["act_dec"], minlength=B).float()
        tot["n"] += B
        tot["np"] += pw.sum().item()
        tot["ce"] += (ce * pw).sum().item()
        tot["ent"] += (ent * pw).sum().item()
        tot["uniform_ce"] += (torch.log(n_act) * pw).sum().item()
        tot["nll_chosen"] += -logp[b["chosen"]].sum().item()
        # (on the CPU: integer scatter-min is not available on every device, e.g. MPS)
        seg = b["act_dec"].cpu()
        top = first_argmax(logit.cpu(), seg, B)
        tot["top1"] += (top == b["chosen"].cpu()).sum().item()
        tot["top1_target"] += (top == first_argmax(t.cpu(), seg, B)).sum().item()
        tot["loss"] += loss.item() * B
        vs.append(v.cpu().numpy())
        zs.append(b["z"].cpu().numpy())
        oks.append(b["value_ok"].cpu().numpy() > 0)
        vts.append(b["vt"].cpu().numpy())
        vws.append(b["vw"].cpu().numpy() > 0)
    net.train()
    n, npw = max(tot["n"], 1), max(tot["np"], 1.0)
    v, z, ok = np.concatenate(vs), np.concatenate(zs), np.concatenate(oks)
    vt, vw = np.concatenate(vts), np.concatenate(vws)
    res = {
        "decisions": tot["n"],
        "policy_decisions": int(tot["np"]),
        "loss": tot["loss"] / n,
        "policy_ce": tot["ce"] / npw,
        "policy_kl": (tot["ce"] - tot["ent"]) / npw,
        "uniform_ce": tot["uniform_ce"] / npw,
        "policy_nll_chosen": tot["nll_chosen"] / n,
        "policy_top1": tot["top1"] / n,
        "policy_top1_target": tot["top1_target"] / n,
    }
    if vw.any():
        res["value_mse_target"] = float(((v[vw] - vt[vw]) ** 2).mean())
    if ok.any():
        res["value_mse"] = float(((v[ok] - z[ok]) ** 2).mean())
        res["value_mse_baseline"] = float((z[ok] ** 2).mean())
        dec = ok & (z != 0)
        if dec.any():
            p = np.clip((1 + v[dec]) / 2, 1e-6, 1 - 1e-6)
            y = (z[dec] > 0).astype(float)
            res["value_logloss"] = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
            res["value_auc"] = auc(v[dec], y > 0)
            res["value_decisions"] = int(dec.sum())
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="data directories (dzk play --data-out) or shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", help="warm start from this checkpoint (.pt)")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.05, help="fraction of steps")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--value-target", default="rootq", help="z | rootq | mix:L (L z + (1-L) root value)")
    ap.add_argument("--select", default="policy", choices=["policy", "loss", "last"],
                    help="exported epoch: lowest validation policy cross-entropy, lowest validation loss, or the last")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--sources", default="search", help="comma list of: " + ",".join(D.SOURCES))
    ap.add_argument("--max-actions", type=int, default=512, help="skip decisions with more actions")
    ap.add_argument("--flat-k-frac", type=float, default=0.25,
                    help="no policy target for mcts:N decisions with more than this fraction of N actions (0: keep)")
    ap.add_argument("--target", default="visits",
                    help="policy target: visits (root visit distribution), top (one-hot of the most-visited move), "
                         "chosen (one-hot of the played move, exploration samples included) or sharp:T")
    ap.add_argument("--feat-dtype", default="float16", choices=["float16", "float32"],
                    help="in-memory storage of the features (batches are float32 either way)")
    ap.add_argument("--E", type=int, default=32)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="stop training after this long, then evaluate and export as usual (0 = no limit)")
    ap.add_argument("--no-resume", action="store_true", help="ignore OUT/ckpt_last.pt")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    P, stats = D.load_packed(a.data, sources=a.sources.split(","), max_actions=a.max_actions,
                             flat_k_frac=a.flat_k_frac, feat_dtype=np.dtype(a.feat_dtype))
    load_s = time.time() - t0
    P.set_policy_target(a.target)
    P.set_value_target(a.value_target)
    keys = [g.get("game_key", str(i)) for i, g in enumerate(P.games)]
    val_game = np.array([is_val(k, a.val_frac) for k in keys], bool)
    in_val = val_game[P.game] if P.n else np.zeros(0, bool)
    train = np.flatnonzero(~in_val)
    val = np.flatnonzero(in_val)
    print(f"loaded {stats} in {load_s:.1f} s ({P.nbytes() / 1e6:.0f} MB): {len(train)} train / {len(val)} val "
          f"decisions ({int((~val_game).sum())} / {int(val_game.sum())} games)", flush=True)
    if not len(train):
        sys.exit("no training decisions")
    if a.init:
        net, _ = M.load_checkpoint(a.init)
        print(f"warm start from {a.init}", flush=True)
    else:
        net = M.DzkNet(E=a.E, H=a.H, S=a.S)
    net.to(a.device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = math.ceil(len(train) / a.batch)
    total = max(1, int(round(a.epochs * steps_per_epoch)))
    warm = max(1, int(a.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    # what a resumed run must share with the run it resumes
    config = {k: getattr(a, k) for k in ("data", "init", "epochs", "batch", "lr", "wd", "warmup", "clip", "value_weight",
                                         "value_target", "val_frac", "sources", "max_actions", "flat_k_frac", "target",
                                         "E", "H", "S", "seed", "select")}
    config.update(train_decisions=int(len(train)), total_steps=total)
    best, best_epoch, history = float("inf"), None, []
    step, epoch = 0, 0
    last_path = os.path.join(a.out, "ckpt_last.pt")
    resumed = False
    if not a.no_resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        st = ck.get("train_state")
        if st and st.get("config") == json.loads(json.dumps(config)) and st["step"] < total:
            net.load_state_dict(ck["state_dict"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            step, epoch, best, best_epoch, history = st["step"], st["epoch"], st["best"], st["best_epoch"], st["history"]
            resumed = True
            print(f"resumed from {last_path}: epoch {epoch}, step {step} of {total}", flush=True)
        elif st and st.get("config") == json.loads(json.dumps(config)):
            print(f"{last_path} already finished all {total} steps; exporting", flush=True)
            step, epoch, best, best_epoch, history = st["step"], st["epoch"], st["best"], st["best_epoch"], st["history"]
            resumed = True
        else:
            print(f"{last_path} is from another configuration: starting over", flush=True)
    log = open(os.path.join(a.out, "train_log.jsonl"), "a")
    t_train = time.time()
    stopped_early = False
    while step < total and not stopped_early:
        run = {"loss": 0.0, "n": 0}
        gn = torch.tensor(0.0)
        for b in P.batches(train, a.batch, shuffle=True, seed=a.seed * 1000 + epoch, device=a.device):
            loss, ce, *_ = losses(net, b, a.value_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)
            opt.step()
            sched.step()
            step += 1
            run["loss"] += loss.item() * ce.shape[0]
            run["n"] += ce.shape[0]
            if step >= total:
                break
            if a.max_minutes and time.time() - t_train > a.max_minutes * 60:
                stopped_early = True
                break
        epoch += 1
        ev = evaluate(net, P, val, 512, a.value_weight, a.device) if len(val) else {}
        rec = {"epoch": epoch, "step": step, "epochs_done": step / steps_per_epoch, "lr": sched.get_last_lr()[0],
               "train_loss": run["loss"] / max(run["n"], 1), "grad_norm_last": float(gn),
               "seconds": round(time.time() - t_train, 1), "val": ev}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        history.append(rec)
        if a.select == "last" or not ev:
            score = -epoch
        elif a.select == "policy":
            score = ev["policy_ce"]
        else:
            score = ev["loss"]
        meta = {"epoch": epoch, "step": step, "val": ev, "data": a.data, "sources": a.sources,
                "value_target": a.value_target, "target": a.target}
        if score <= best:
            best, best_epoch = score, epoch
            M.save_checkpoint(os.path.join(a.out, "ckpt_best.pt"), net, {"meta": meta})
        state = {"config": config, "step": step, "epoch": epoch, "best": best, "best_epoch": best_epoch,
                 "history": history, "opt": opt.state_dict(), "sched": sched.state_dict()}
        M.save_checkpoint(last_path + ".tmp", net, {"meta": meta, "train_state": state})
        os.replace(last_path + ".tmp", last_path)
    best_net, ck = M.load_checkpoint(os.path.join(a.out, "ckpt_best.pt"))
    digest = M.export_dzkn(best_net, os.path.join(a.out, "net.dzkn"),
                           {"epoch": best_epoch, "val": ck["meta"]["val"], "data": a.data, "init": a.init,
                            "select": a.select, "value_target": a.value_target, "target": a.target})
    train_s = time.time() - t_train
    out = {"best_epoch": best_epoch, "select": a.select, "selection_score": best, "epochs_run": epoch,
           "stopped_early": stopped_early, "resumed": resumed, "net": os.path.join(a.out, "net.dzkn"),
           "net_sha256": digest, "load_seconds": round(load_s, 1), "train_seconds": round(train_s, 1),
           "data_megabytes": round(P.nbytes() / 1e6, 1), "steps": step, "total_steps": total,
           "train_decisions": int(len(train)), "val_decisions": int(len(val)), "data_stats": stats,
           "args": vars(a), "history": history, "torch": torch.__version__, "threads": a.threads,
           "params": sum(p.numel() for p in best_net.parameters())}
    json.dump(out, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("best_epoch", "select", "selection_score", "stopped_early", "train_seconds",
                                          "net_sha256", "params")}))


if __name__ == "__main__":
    main()
