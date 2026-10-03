"""docs/018 (the MLP's bottlenecks). The learned part of the feature embeddings: W_trained - W_init (the init is reproducible: magezero.vocab.initial_rows
keyed by feature id, times emb_init_std). Is it low-rank? And how far has each row moved, by how often it appears?"""
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from draftzero.gameplay import supervised as sv
from magezero.vocab import initial_rows
# usage: python tools/imitation_scale/emb_drift.py CHECKPOINT EMB_INIT_STD  (run from the repo root; data/imitation_scale/h5)
CK, STD = sys.argv[1], float(sys.argv[2])
dev = torch.device("cuda")
m, vocab, meta = sv.load_any_checkpoint(CK, device=dev)
W = m.embedding.weight.detach()
W0 = torch.as_tensor(initial_rows(vocab.ids, W.shape[1]), device=dev) * STD
D = W - W0
# frequency of each row in a sample of validation states
cnt = torch.zeros(W.shape[0], device=dev)
rng = np.random.default_rng(0)
for t in ["turnstart", "replay_priority", "opp_priority", "replay_attack", "replay_target", "opp_block"]:
    f = h5py.File(f"data/imitation_scale/h5/{t}_val.h5", "r"); off = f["offsets"][:]
    for i in rng.choice(len(off) - 1, 1000, replace=False):
        rows, _ = vocab.map_bags(list(f["indices"][off[i]:off[i + 1]]), [0])
        cnt[torch.as_tensor(np.asarray(rows, np.int64), device=dev)] += 1
def eff(X, name):
    lam = torch.linalg.svdvals(X.float()) ** 2; lam = lam / lam.sum(); c = torch.cumsum(lam, 0)
    k = lambda q: int((c < q).sum().item()) + 1
    print(f"  {name:44s} participation ratio {(1/(lam**2).sum()).item():7.1f}  k50 {k(.5):4d}  k90 {k(.9):4d}  k99 {k(.99):5d}")
print(f"init std {W0.std().item():.4f}; trained std {W.std().item():.4f}; learned change std {D.std().item():.4f}")
rel = D.norm(dim=1) / W0.norm(dim=1)
for lo, hi, name in ((0, 1, "rows not seen in the sample"), (1, 10, "seen 1-9 times in 6,000 states"), (10, 100, "seen 10-99 times"), (100, 1000, "seen 100-999 times"), (1000, 1e9, "seen 1,000+ times (in most states)")):
    sel = (cnt >= lo) & (cnt < hi)
    if sel.any():
        print(f"  {name:40s} rows {int(sel.sum()):6d}  |change| / |init| median {rel[sel].median().item():.2f}")
used = cnt > 0
print("spectrum of the learned change (rows these states use, weighted by frequency):")
eff(D[used] * cnt[used].sqrt()[:, None], "W_trained - W_init")
eff(W0[used] * cnt[used].sqrt()[:, None], "W_init (reference: isotropic)")
eff(W[used] * cnt[used].sqrt()[:, None], "W_trained")
eff(D, "W_trained - W_init, all rows unweighted")
