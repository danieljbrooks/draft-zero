"""Where a training step of the MLP goes, on the CPU or the GPU (docs/020), and what a sparse feature table would buy.

The trainer (draftzero.gameplay.supervised) keeps the feature-embedding table dense: every step zeroes and fills a
gradient the size of the whole table (62,231 x 1,024 for the leading MLP, 64M of its 80.5M parameters) and AdamW
updates every row, though a batch touches a few thousand. On a GPU that is cheap; on a CPU it is memory traffic.
This times the trainer's own step on real batches (the trainer's sampler over real tables) in three parts, forward
+ loss, backward, and clip + optimizer, for:

  dense   the trainer as it is (EmbeddingBag max + mean, AdamW over everything, the table at 30x the rate)
  fused   the same with AdamW's fused kernel (one flag; torch has it for the CPU too)
  frozen  the table not trained at all: the speed limit of any sparse update of it (AdamW on the rest only)
  sparse  the same network and loss, but the table read with a sparse-gradient gather (max and mean pooled with
          segment_reduce) and updated by SparseAdam (only the rows a batch touches; lazy moments), the rest by
          AdamW; the clip covers the dense parameters only. Not in the trainer: what such a change would buy.

    python tools/compute_bench/train_step_bench.py --checkpoint <mlp.pt.gz> --config configs/compute_bench_human.yml \\
        --tables-dir data/compute_bench/h5_human --threads 8,16 --steps 30 --json out.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from draftzero.gameplay import supervised as sv  # noqa: E402


def encode_sparse(m, idx, off):
    """BagMLPNet.encode with the table read by a sparse-gradient gather: max + mean pooled per state."""
    idx = idx % m.embedding.num_embeddings
    tok = F.embedding(idx, m.embedding.weight, sparse=True)
    lens = torch.diff(torch.cat([off, torch.tensor([idx.numel()], device=off.device)]))
    x = torch.segment_reduce(tok, "max", lengths=lens)
    if m.add_mean:
        x = x + torch.segment_reduce(tok, "mean", lengths=lens)
    if m.emb_proj is not None:
        x = m.emb_proj(x)
    for b in m.blocks:
        x = b(x)
    return m.embedding_dropout(m.norm(x))


def loss_of(tr, b, emb):
    """The trainer's loss for priority, target and binary tables (Trainer.step), from the state vectors."""
    cfg, dev, m = tr.cfg, tr.dev, tr.model
    t = tr.data.train[b["ti"]]
    w = b["w"].to(dev)
    if t.kind in sv.POLICY_KINDS:
        L, S = b["L"].to(dev), b["S"].to(dev)
        head = m.target_head if t.kind == "target" else m.player_priority_head
        nll = sv.policy_nll(head(emb), L, S)
        valid = (w > 0) & (S & L).any(1)
        lp = torch.where(valid, nll * w, torch.zeros_like(nll)).sum() / valid.sum().clamp(min=1)
    elif t.kind == "binary":
        ce = F.cross_entropy(m.binary_head(emb), b["y"].to(dev), reduction="none")
        lp = (ce * w).sum() / (w > 0).sum().clamp(min=1)
    else:
        raise ValueError(f"table kind {t.kind} is not timed here")
    vx = sv.value_logit(m, emb)
    tgt = tr.value_targets()[b["gidx"].to(dev)]
    vm = b["vmask"].to(dev) & torch.isfinite(tgt)
    tgt = torch.where(vm, tgt, torch.zeros_like(tgt))
    per = sv.value_bce_from_logit(vx, tgt)
    lv = torch.where(vm, per, torch.zeros_like(per)).sum() / max(1.0, b["veligible"] * b["vfrac"])
    return lp + cfg["value_weight"] * lv


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def time_variant(tr, batches, variant: str, steps: int, warm: int = 5) -> dict:
    m, dev, cfg = tr.model, tr.dev, tr.cfg
    lr = cfg["lr"]
    emb = [m.embedding.weight]
    rest = [p for n, p in m.named_parameters() if not n.endswith("embedding.weight")]
    m.embedding.weight.requires_grad_(variant != "frozen")
    if variant in ("dense", "fused"):
        opts = [torch.optim.AdamW([{"params": rest, "lr": lr}, {"params": emb, "lr": lr * cfg["emb_lr_mult"]}],
                                  lr=lr, weight_decay=cfg["weight_decay"], **({"fused": True} if variant == "fused" else {}))]
        clip = rest + emb
        enc = lambda i, o: m.encode(i, o)  # noqa: E731
    elif variant == "frozen":
        opts = [torch.optim.AdamW(rest, lr=lr, weight_decay=cfg["weight_decay"])]
        clip = rest
        enc = lambda i, o: m.encode(i, o)  # noqa: E731
    else:
        opts = [torch.optim.AdamW(rest, lr=lr, weight_decay=cfg["weight_decay"]),
                torch.optim.SparseAdam(emb, lr=lr * cfg["emb_lr_mult"])]
        clip = rest
        enc = lambda i, o: encode_sparse(m, i, o)  # noqa: E731
    m.train()
    t_f, t_b, t_o, rows, uniq = [], [], [], [], []
    for k in range(warm + steps):
        b = batches[k % len(batches)]
        idx, off = b["idx"].to(dev), b["off"].to(dev)
        sync(dev)
        t0 = time.perf_counter()
        with sv._autocast(dev, tr.dtype):
            e = enc(idx, off)
        with sv._fp32(dev, tr.dtype):
            loss = loss_of(tr, b, e.float())
        sync(dev)
        t1 = time.perf_counter()
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        sync(dev)
        t2 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(clip, cfg["grad_clip"])
        for o in opts:
            o.step()
        sync(dev)
        t3 = time.perf_counter()
        if k >= warm:
            t_f.append(t1 - t0)
            t_b.append(t2 - t1)
            t_o.append(t3 - t2)
            rows.append(b["n"])
            uniq.append(int(torch.unique(idx % m.embedding.num_embeddings).numel()))
    m.embedding.weight.requires_grad_(True)
    step = [a + b_ + c for a, b_, c in zip(t_f, t_b, t_o)]
    return {"variant": variant, "steps": steps, "rows_per_step": statistics.fmean(rows),
            "unique_rows_per_step": round(statistics.fmean(uniq)),
            "ms_forward": round(1000 * statistics.median(t_f), 1), "ms_backward": round(1000 * statistics.median(t_b), 1),
            "ms_optimizer": round(1000 * statistics.median(t_o), 1), "ms_step": round(1000 * statistics.median(step), 1),
            "samples_per_s": round(sum(rows) / sum(step), 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=str(REPO / "configs/compute_bench_human.yml"))
    ap.add_argument("--tables-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", default="8", help="comma-separated torch thread counts (CPU)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--variants", default="dense,fused,frozen,sparse")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="trainer config overrides")
    ap.add_argument("--out", default="/tmp/train_step_bench")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    cfg = sv.resolve_config(sv.load_config_file(Path(a.config)), sv.parse_overrides(a.set),
                            {"tables_dir": a.tables_dir, "init_checkpoint": a.checkpoint, "device": a.device})
    import shutil
    shutil.rmtree(a.out, ignore_errors=True)
    tr = sv.Trainer(cfg, Path(a.out), log=lambda *_: None)
    batches = [tr.sampler.make_batch(*tr.sampler.next()) for _ in range(40)]
    state0 = {k: v.detach().clone() for k, v in tr.model.state_dict().items()}
    res = []
    for th in [int(x) for x in a.threads.split(",")]:
        if tr.dev.type == "cpu":
            torch.set_num_threads(th)
        for v in a.variants.split(","):
            tr.model.load_state_dict(state0)
            r = {"device": str(tr.dev), "threads": th if tr.dev.type == "cpu" else None,
                 **time_variant(tr, batches, v, a.steps)}
            print(json.dumps(r), flush=True)
            res.append(r)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
