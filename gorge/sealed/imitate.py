"""
A reference builder trained on human decks (not the self-play model): the same DeckNet and builder, but s is fit so
that, within a pool, the deck a human built ranks above nearby alternatives (the human deck with 1-4 random swaps, and
random builds from the same pool). A listwise softmax loss over (human deck, K alternatives) per train pool.

It shows how close the shared network and builder can get to human decks when shown them: a ceiling for the
agreement measures, and a human-taught opponent for the win-rate measure.

  python -m sealed.imitate OUT.pt
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from . import decks as D
from . import model as M

C = D.C


def make_sets(pools, k, rng):
    X = []
    for p in pools:
        h, pool = p["ref_v"], p["pool_v"]
        alts = [D.perturb(h, pool, rng, int(rng.integers(1, 5))) for _ in range(k - k // 4)]
        alts += [D.random_build(pool, rng) for _ in range(k // 4)]
        X.append(np.stack([h] + alts))
    return torch.tensor(np.array(X))


def fit(pools, val, k=16, epochs=30, lr=3e-3, seed=0, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = M.DeckNet()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    Xv = make_sets(val, k, np.random.default_rng(99))
    best, state = 1e9, None
    for ep in range(epochs):
        X = make_sets(pools, k, rng)          # fresh alternatives each epoch
        m.train()
        for i in range(0, len(X), 256):
            x = X[i:i + 256]
            s = m(x.reshape(-1, len(C))).reshape(x.shape[0], x.shape[1])
            loss = F.cross_entropy(s, torch.zeros(len(x), dtype=torch.long)) + 1e-3 * m.reg()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            s = m(Xv.reshape(-1, len(C))).reshape(Xv.shape[0], Xv.shape[1])
            vl = F.cross_entropy(s, torch.zeros(len(Xv), dtype=torch.long)).item()
            top1 = (s.argmax(1) == 0).float().mean().item()
        log(f"epoch {ep}: val loss {vl:.4f}, human deck ranked first {top1:.3f}")
        if vl < best:
            best, state = vl, {kk: v.clone() for kk, v in m.state_dict().items()}
    m.load_state_dict(state); m.eval()
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--members", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    train = D.load_pools("sealed", "train")
    val = D.load_pools("sealed", "val")[:300]
    ms = [fit(train, val, seed=s, log=lambda x: print(x, file=sys.stderr, flush=True)) for s in range(a.members)]
    M.Ensemble(ms).save(a.out)


if __name__ == "__main__":
    main()
