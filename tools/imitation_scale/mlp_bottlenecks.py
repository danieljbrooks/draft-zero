"""docs/018 (the MLP's bottlenecks). Where does the MLP squeeze information? On ~6,000 validation states (1,000 per decision table), follow a trained
MLP layer by layer: tokens in, max-pool winners, effective dimensions (PCA participation ratio, components for
90/99% of variance) of the pooled vector, each block's output, and each head's hidden layer; and the singular
spectrum of the feature-embedding table (is the per-feature information low-rank?)."""
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from draftzero.gameplay import supervised as sv
# usage: python tools/imitation_scale/mlp_bottlenecks.py MLP_CHECKPOINT [STATES_A_TABLE]  (from the repo root)
CK = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
dev = torch.device("cuda")
model, vocab, meta = sv.load_any_checkpoint(CK, device=dev)
m = model.eval()
tables = ["turnstart", "replay_priority", "opp_priority", "replay_attack", "replay_target", "opp_block"]
rng = np.random.default_rng(0)
bags = []
for t in tables:
    f = h5py.File(f"data/imitation_scale/h5/{t}_val.h5", "r")
    off = f["offsets"][:]; n = len(off) - 1
    for i in np.sort(rng.choice(n, min(N, n), replace=False)):
        ids = f["indices"][off[i]:off[i + 1]]
        rows, _ = vocab.map_bags(list(ids), [0])
        bags.append((t, len(ids), np.asarray(rows, np.int64)))
print(f"{len(bags)} states; raw tokens a state: mean {np.mean([b[1] for b in bags]):.0f}, known to the vocab: mean {np.mean([len(b[2]) for b in bags]):.0f}, distinct known: mean {np.mean([len(np.unique(b[2])) for b in bags]):.0f}")

def eff(X, name):
    X = X - X.mean(0, keepdims=True)
    lam = torch.linalg.svdvals(X.float()) ** 2
    lam = lam / lam.sum()
    c = torch.cumsum(lam, 0)
    pr = 1.0 / (lam ** 2).sum()
    k = lambda q: int((c < q).sum().item()) + 1
    print(f"  {name:34s} dims {X.shape[1]:5d}  participation ratio {pr.item():7.1f}  k50 {k(0.5):4d}  k90 {k(0.9):4d}  k99 {k(0.99):5d}")
    return lam

W = m.embedding.weight.detach()
pooled, winners, cover, nwin = [], [], [], []
with torch.no_grad():
    for t, _, rows in bags:
        r = torch.as_tensor(rows, device=dev)
        E = W[r]                                   # [n_tok, e]
        v, a = E.max(0)                            # max-pool: value and winning token per channel
        pooled.append(v); winners.append(r[a])
        u = torch.unique(r[a]).numel(); nwin.append(u); cover.append(u / torch.unique(r).numel())
    P = torch.stack(pooled)                        # [S, e]
    WIN = torch.stack(winners)                     # [S, e] winning embedding row per channel
    # check against the model's own EmbeddingBag
    r0 = torch.as_tensor(bags[0][2], device=dev)
    assert torch.allclose(m.embedding(r0, torch.zeros(1, dtype=torch.long, device=dev))[0], P[0], atol=1e-5)
    print(f"max pooling: distinct tokens that win >= 1 channel: mean {np.mean(nwin):.0f} a state "
          f"({100*np.mean(cover):.0f}% of the state's distinct tokens; median {100*np.median(cover):.0f}%, 10th pct {100*np.percentile(cover,10):.0f}%)")
    # channel behaviour across states: how often the same row wins a channel
    top_share = []
    for c in range(WIN.shape[1]):
        _, cnt = torch.unique(WIN[:, c], return_counts=True)
        top_share.append(cnt.max().item() / WIN.shape[0])
    top_share = np.array(top_share)
    print(f"channels whose single most frequent winner wins in >90% of states: {int((top_share > 0.9).sum())} of {len(top_share)}; >50%: {int((top_share > 0.5).sum())}")
    std = P.std(0)
    print(f"pooled channel std across states: median {std.median().item():.4f}, channels with std < 1% of the median: {int((std < 0.01 * std.median()).sum())}")
    print("effective dimensions (PCA over the states):")
    eff(P, "pooled vector (after max)")
    x = P if m.emb_proj is None else m.emb_proj(P)
    if m.emb_proj is not None:
        eff(x, "after the projection")
    for j, b in enumerate(m.blocks):
        x = b(x); eff(x, f"after block {j + 1}")
    s = m.norm(x); eff(s, "state vector (final LayerNorm)")
    for hn in ("player_priority_head", "opponent_priority_head", "target_head", "binary_head", "value_head"):
        h = getattr(m, hn)
        hid = torch.relu(h[0](s))
        dead = int((hid.max(0).values <= 0).sum())
        eff(hid, f"{hn} hidden (dead units {dead}/{hid.shape[1]})")
    # the embedding table's spectrum: all rows, and the rows these states use weighted by how often they appear
    print("feature-embedding table, singular spectrum (energy = sigma^2):")
    eff(W, f"all {W.shape[0]} rows")
    used, cnt = torch.unique(torch.cat([torch.as_tensor(b[2], device=dev) for b in bags]), return_counts=True)
    eff(W[used] * cnt.float().sqrt()[:, None], f"{used.numel()} rows used here, freq-weighted")
    noise = torch.randn_like(W[used]) * W[used].std()
    eff(noise * cnt.float().sqrt()[:, None], "  reference: random rows, same scale")
