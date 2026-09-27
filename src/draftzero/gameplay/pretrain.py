"""
pretrain.py — a MageZero v0.2 network pretrained on human 17lands decisions, as the starting
point of experiment #2's run 2 (the imitation A/B; ROADMAP "Run 2 handoff").

The study's from-scratch path (imitation.new_net, trunk/heads, set NLL over the masked priority
softmax), extended to every head the loop's search can read, all trained together:

  priority   turn-start human action SETS (turnstart_train): -log sum_{a in S} p(a) over the legal
             options, as docs/008 §7.3 scored it
  binary     human attack decisions (replay_attack_train): cross-entropy on yes / no, slot 1 = yes,
             as MageZero's CHOOSE_USE rows
  value      the game's result z = +1 / -1 on both tables (MSE, weight `value_weight`)

The feature vocab is MageZero's own rule (vocab.kept_feature_ids, k = 10 over the train rows) with
v0.2's hash width, and the checkpoint is MageZero's format (model_state_dict + feature_vocab), so
the v0.2 inference server serves it and `train.py --checkpoint` continues from it (appending the
self-play features to the vocab).

    python -m draftzero.gameplay.pretrain train --out models/imitation/pretrained.pt.gz --budget 2400
    python -m draftzero.gameplay.pretrain agreement --checkpoint models/imitation/pretrained.pt.gz
    python -m draftzero.gameplay.pretrain agreement --checkpoint <gen N> --rows 3000 --json

`agreement` is the human-agreement rate: top-1 in S on held-out turn-start decisions (docs/008's
headline metric, 73% for gen 33's trunk with retrained heads), plus the value head's AUC on
game results and the attack head's accuracy. The loop logs it every generation for run 2.

17lands public data is CC BY 4.0 (https://www.17lands.com/public_datasets).
"""
from __future__ import annotations

import argparse
import copy
import gzip
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from draftzero.gameplay import imitation as im

GLOBAL_MAX = 2 ** 31 - 1          # MageZero v0.2's feature id range (model.GLOBAL_MAX)
H5 = im.OUT_DIR / "h5"


# ================================================================================================
# data
# ================================================================================================

def load_all(path: Path) -> dict:
    """Every row of a turn-start / replay-priority table (features, legal / set CSR, z, status)."""
    import h5py
    with h5py.File(path, "r") as f:
        n = len(f["offsets"]) - 1
    return im.load_rows(path, np.arange(n))


def build_vocab(indices: np.ndarray, offsets: np.ndarray, k: int = 10):
    """MageZero's ignore-list rule (the one train.py applies to self-play data) with v0.2's encoding."""
    from magezero.vocab import FeatureVocab, kept_feature_ids
    return FeatureVocab(kept_feature_ids(indices.astype(np.int64), offsets.astype(np.int64), k=k),
                        feature_hash_bins=GLOBAL_MAX)


def _mapped(vocab, t: dict) -> tuple[np.ndarray, np.ndarray]:
    rows, ptr = im.map_features(vocab, t["indices"], t["offsets"])
    return rows.astype(np.int32), ptr


# ================================================================================================
# training
# ================================================================================================

def save_checkpoint(model, vocab, path: Path, info: dict) -> None:
    """MageZero's checkpoint format: what server.init and train.py --checkpoint read."""
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    torch.save({"epoch": 0,
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "feature_vocab": vocab.state_dict(),
                "pretrain": info}, buf)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb", compresslevel=1) as f:
        f.write(buf.getvalue())
    os.replace(tmp, path)


def train(out: Path, *, budget_s: float = 2400, lr: float = 3e-4, value_weight: float = 0.5,
          attack_weight: float = 1.0, batch: int = 64, warmup: int = 300, token_dropout: float = 0.3,
          eval_every_s: float = 240, val_rows: int = 3000, vocab_k: int = 10, n_train: int | None = None,
          seed: int = 0, log=print) -> dict:
    """Train the whole NetTransformer from scratch on the human train split for a wall-clock
    budget. Each step is a turn-start batch (priority set NLL + value) or, in proportion to the
    tables' sizes, an attack batch (binary CE + value). Early stopping on the val split's
    set NLL + value_weight x value MSE; the best state is saved."""
    im._torch_env()
    import torch
    from magezero.vocab import initial_rows
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = im.device()
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    t0 = time.monotonic()
    tr = load_all(H5 / "turnstart_train.h5")
    va = load_all(H5 / "turnstart_val.h5")
    at_tr = im.load_attack(H5 / "replay_attack_train.h5")
    at_va = im.load_attack(H5 / "replay_attack_val.h5")
    if n_train is not None:                     # smoke tests: a random subset of the train rows
        sel = np.sort(rng.choice(len(tr["z"]), min(n_train, len(tr["z"])), replace=False))
        tr = im.subset_table(tr, sel)
    at_z = lambda t: np.where(t["meta/won"].astype(bool), 1.0, -1.0).astype(np.float32)
    at_tr["z"], at_va["z"] = at_z(at_tr), at_z(at_va)
    log(f"pretrain: {len(tr['z'])} turn-start and {len(at_tr['y'])} attack train rows "
        f"({time.monotonic() - t0:.0f} s to load)")

    vocab = build_vocab(tr["indices"], tr["offsets"], k=vocab_k)
    rtr, ptr_tr = _mapped(vocab, tr)
    rva, ptr_va = _mapped(vocab, va)
    rat, ptr_at = _mapped(vocab, at_tr)
    rav, ptr_av = _mapped(vocab, at_va)
    log(f"pretrain: vocab {len(vocab)} rows (k={vocab_k}), mean mapped length {np.diff(ptr_tr).mean():.0f} "
        f"(raw {np.diff(tr['offsets']).mean():.0f})")

    model = im.new_net(len(vocab))
    with torch.no_grad():
        model.embedding.weight.copy_(torch.as_tensor(initial_rows(vocab.ids, 512)))
    m = model.to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")

    va_sel = np.flatnonzero(va["meta/label_status"] != 2)
    va_sel = np.sort(rng.choice(va_sel, min(val_rows, len(va_sel)), replace=False))
    av_sel = np.arange(len(at_va["y"]))

    def groups_of(ptr, pool):
        lens = (np.diff(ptr)[pool] * (1 - token_dropout)).astype(int)
        bkt = np.array([im.bucket_len(int(x)) for x in lens])
        g = {b: pool[bkt == b] for b in np.unique(bkt)}
        keys = [k for k in g if len(g[k]) >= batch] or list(g)
        w = np.array([len(g[k]) for k in keys], np.float64)
        return g, keys, w / w.sum()

    ts_groups = groups_of(ptr_tr, np.arange(len(tr["z"])))
    at_groups = groups_of(ptr_at, np.arange(len(at_tr["y"])))
    p_attack = len(at_tr["y"]) / (len(at_tr["y"]) + len(tr["z"]))

    def draw(groups):
        g, keys, w = groups
        key = keys[rng.choice(len(keys), p=w)]
        return rng.choice(g[key], min(im.batch_for(int(key), batch), len(g[key])), replace=False)

    def evaluate() -> dict:
        m.eval()
        nll = vse = 0.0
        acc = ave = 0.0
        with torch.no_grad(), torch.autocast(**amp):
            for b in im.length_batches(np.diff(ptr_va)[va_sel], 64):
                sel = va_sel[b]
                idx, off = im._batch(rva, ptr_va, sel, dev)
                pa, _, v = im.heads(m, im.trunk(m, idx, off))
                L = torch.as_tensor(im.dense_masks(va["legal_indptr"], va["legal_idx"], sel), device=dev)
                S = torch.as_tensor(im.dense_masks(va["set_indptr"], va["set_idx"], sel), device=dev)
                nll += float(im.set_nll(im.masked_log_softmax(pa.float(), L), S & L).sum())
                vse += float(((v.float() - torch.as_tensor(va["z"][sel], device=dev)) ** 2).sum())
            for b in im.length_batches(np.diff(ptr_av)[av_sel], 64):
                sel = av_sel[b]
                idx, off = im._batch(rav, ptr_av, sel, dev)
                _, bl, v = im.heads(m, im.trunk(m, idx, off))
                acc += float((bl.float().argmax(-1).cpu().numpy() == at_va["y"][sel]).sum())
                ave += float(((v.float() - torch.as_tensor(at_va["z"][sel], device=dev)) ** 2).sum())
        m.train()
        r = {"val_set_nll": nll / len(va_sel), "val_value_mse": vse / len(va_sel),
             "val_attack_acc": acc / len(av_sel), "val_attack_value_mse": ave / len(av_sel)}
        r["val_loss"] = r["val_set_nll"] + value_weight * r["val_value_mse"]
        return {k: round(v, 4) for k, v in r.items()}

    def snapshot():
        return copy.deepcopy({k: v.detach().cpu() for k, v in m.state_dict().items()})

    t0 = time.monotonic()
    ev = evaluate()
    hist = [{"step": 0, "seen": 0, "sec": 0, **ev}]
    best, best_state = ev["val_loss"], snapshot()
    log(f"pretrain: start {ev}")
    m.train()
    step = seen = 0
    last_eval = t0
    while time.monotonic() - t0 < budget_s:
        attack = rng.random() < p_attack
        with torch.autocast(**amp):
            if attack:
                sel = draw(at_groups)
                idx, off = im._batch(rat, ptr_at, sel, dev, token_dropout, rng)
                _, bl, v = im.heads(m, im.trunk(m, idx, off))
                y = torch.as_tensor(at_tr["y"][sel], dtype=torch.long, device=dev)
                z = torch.as_tensor(at_tr["z"][sel], device=dev)
                loss = attack_weight * torch.nn.functional.cross_entropy(bl.float(), y) \
                    + value_weight * ((v.float() - z) ** 2).mean()
            else:
                sel = draw(ts_groups)
                idx, off = im._batch(rtr, ptr_tr, sel, dev, token_dropout, rng)
                pa, _, v = im.heads(m, im.trunk(m, idx, off))
                L = torch.as_tensor(im.dense_masks(tr["legal_indptr"], tr["legal_idx"], sel), device=dev)
                S = torch.as_tensor(im.dense_masks(tr["set_indptr"], tr["set_idx"], sel), device=dev)
                P = torch.as_tensor(tr["meta/label_status"][sel] != 2, device=dev)
                z = torch.as_tensor(tr["z"][sel], device=dev)
                nll = im.set_nll(im.masked_log_softmax(pa.float(), L), S & L)
                loss = (nll[P].mean() if bool(P.any()) else pa.sum() * 0) \
                    + value_weight * ((v.float() - z) ** 2).mean()
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / warmup)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        seen += len(sel)
        if time.monotonic() - last_eval > eval_every_s:
            last_eval = time.monotonic()
            ev = evaluate()
            hist.append({"step": step, "seen": seen, "sec": round(last_eval - t0), **ev})
            log(f"pretrain: step {step} seen {seen} {ev} ({last_eval - t0:.0f} s)")
            if ev["val_loss"] < best:
                best, best_state = ev["val_loss"], snapshot()
                save_checkpoint(model_from(best_state, len(vocab)), vocab, out,
                                {"partial": True, "step": step})
    ev = evaluate()
    hist.append({"step": step, "seen": seen, "sec": round(time.monotonic() - t0), **ev})
    if ev["val_loss"] < best:
        best, best_state = ev["val_loss"], snapshot()
    info = {"history": hist, "best_val_loss": best, "steps": step, "samples_seen": seen,
            "n_train_turnstart": int(len(tr["z"])), "n_train_attack": int(len(at_tr["y"])),
            "vocab_rows": len(vocab), "vocab_k": vocab_k, "budget_s": budget_s, "lr": lr,
            "value_weight": value_weight, "attack_weight": attack_weight, "batch": batch,
            "token_dropout": token_dropout, "seed": seed}
    save_checkpoint(model_from(best_state, len(vocab)), vocab, out, info)
    log(f"pretrain: {step} steps, {seen} samples, best val loss {best:.4f} -> {out}")
    return info


def model_from(state: dict, rows: int):
    m = im.new_net(rows)
    m.load_state_dict(state)
    return m.eval()


# ================================================================================================
# human agreement
# ================================================================================================

def agreement(checkpoint: Path, *, table: str = "turnstart_test", rows: int | None = None,
              attack: bool = True, seed: int = 12345, keep: Path | None = None,
              attack_keep: Path | None = None) -> dict:
    """How closely a checkpoint matches held-out human decisions: priority top-1 / top-3 in S and
    set NLL (docs/008 §7.3's scoring, ties broken in expectation), the same on decisions with 4+
    legal options, the value head's AUC against game results, and the attack head's accuracy.
    `rows` scores a fixed random subset (the same rows every call), for cheap per-generation use.
    `keep` / `attack_keep` (.npy row positions) restrict the tables first, e.g. to the test rows
    whose mirrored game (pairs.py) is not in the train or val split."""
    im._torch_env()
    model, vocab = im.load_checkpoint(Path(checkpoint))
    t = im.load_table(H5 / f"{table}.h5")
    if keep is not None:
        t = im.subset_table(t, np.load(keep))
    if rows is not None and rows < len(t["z"]):
        sel = np.sort(np.random.default_rng(seed).choice(len(t["z"]), rows, replace=False))
        t = im.subset_table(t, sel)
    p = im.predict(model, vocab, t)
    pr = im.policy_rows(t, im.legal_scores(t, p["pa"].astype(np.float32)))
    n_legal = np.diff(t["legal_indptr"])[pr["rows"]]
    won = t["z"] > 0
    out = {"checkpoint": str(checkpoint), "table": table, "rows": int(len(pr["rows"])),
           "top1": float(pr["top1"].mean()), "top3": float(pr["top3"].mean()),
           "set_nll": float(pr["nll"].mean()),
           "top1_4plus": float(pr["top1"][n_legal >= 4].mean()) if (n_legal >= 4).any() else None,
           "value_auc": float(im.auc_score(p["v"].astype(np.float64), won.astype(np.float64))),
           "mapped_share": p["mapped_share"]}
    if attack:
        a = im.load_attack(H5 / "replay_attack_test.h5")
        if attack_keep is not None:
            sel = np.load(attack_keep)
            ind, off = _sub_csr(a, sel)
            a = {**{k: v[sel] for k, v in a.items() if k not in ("indices", "offsets")},
                 "indices": ind, "offsets": off}
        if rows is not None and rows < len(a["y"]):
            sel = np.sort(np.random.default_rng(seed).choice(len(a["y"]), rows, replace=False))
            ind, off = _sub_csr(a, sel)
            a = {**{k: v[sel] for k, v in a.items() if k not in ("indices", "offsets")},
                 "indices": ind, "offsets": off}
        pa = im.predict(model, vocab, a)
        prob = 1 / (1 + np.exp(-(pa["bin"][:, 1] - pa["bin"][:, 0])))
        out.update(attack_rows=int(len(a["y"])),
                   attack_acc=float(((prob > 0.5).astype(int) == a["y"]).mean()),
                   attack_auc=float(im.auc_score(prob, a["y"].astype(np.float64))))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def _sub_csr(t: dict, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parts = [t["indices"][t["offsets"][i]:t["offsets"][i + 1]] for i in sel]
    return np.concatenate(parts), np.concatenate([[0], np.cumsum([len(x) for x in parts])]).astype(np.int64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m draftzero.gameplay.pretrain", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--budget", type=float, default=2400, help="wall-clock seconds")
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--value-weight", type=float, default=0.5)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--eval-every", type=float, default=240)
    t.add_argument("--val-rows", type=int, default=3000)
    t.add_argument("--n-train", type=int, default=None, help="a random subset of the train rows (smoke tests)")
    t.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("agreement")
    a.add_argument("--checkpoint", type=Path, required=True)
    a.add_argument("--table", default="turnstart_test")
    a.add_argument("--rows", type=int, default=None)
    a.add_argument("--no-attack", action="store_true")
    a.add_argument("--keep", type=Path, default=None, help="table row positions to score (.npy)")
    a.add_argument("--attack-keep", type=Path, default=None, help="attack table row positions (.npy)")
    a.add_argument("--json", action="store_true", help="print one JSON line (for the loop)")
    args = ap.parse_args(argv)
    if args.cmd == "train":
        info = train(args.out, budget_s=args.budget, lr=args.lr, value_weight=args.value_weight,
                     batch=args.batch, eval_every_s=args.eval_every, val_rows=args.val_rows,
                     n_train=args.n_train, seed=args.seed)
        args.out.with_suffix("").with_suffix(".json").write_text(json.dumps(info, indent=1))
    else:
        r = agreement(args.checkpoint, table=args.table, rows=args.rows, attack=not args.no_attack,
                      keep=args.keep, attack_keep=args.attack_keep)
        print(("HUMAN_AGREEMENT " + json.dumps(r)) if args.json else json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
