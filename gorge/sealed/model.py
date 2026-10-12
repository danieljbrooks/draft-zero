"""
The deck value network: a deck's strength s(deck), learned from game results alone.

A game between decks a and b is a Bradley-Terry comparison: P(a wins) = sigmoid(s(a) - s(b) + beta * play), where play
is +1 when a's player went first. s has three parts, each a function of the deck's card counts n (286 cards):

1. **Card quality:** sum_c n_c q_c, one learned number per card (computed from its printed facts plus a learned
   per-card correction), like a self-play games-in-hand win rate.
2. **Structure:** a small MLP over deck aggregates built from *learnable* card roles. Each role starts at the printed
   card and may move: how much a card counts as a land, its effective mana value (a distribution over 0..7+), whether
   it is a creature, and which colours of mana it makes. Aggregates: the curve of the non-land cards, the land count,
   the creature count, coloured sources and coloured pips per colour, their ratios, and the deck size. So the network
   can learn what a good curve and land count are in this format, and that a misleading card (a 6-drop with
   landcycling) partly counts as a land.
3. **Context:** a DeepSets term, an MLP over the mean of the deck's card vectors, for synergies that 1-2 miss.

No human deck or 17lands statistic enters: card inputs are printed facts (cards.py) and the labels are game results.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cards as cardsmod

C = cardsmod.load()
NB = 8   # mana-value bins 0..7+


class DeckNet(nn.Module):
    def __init__(self, d=32, hidden=64):
        super().__init__()
        X = torch.tensor(C.X)
        self.register_buffer("X", X)
        n = len(C)
        self.card = nn.Sequential(nn.Linear(X.shape[1], d), nn.ReLU(), nn.Linear(d, d))
        self.emb = nn.Parameter(torch.zeros(n, d))
        self.q_lin = nn.Linear(d, 1)
        self.q_card = nn.Parameter(torch.zeros(n))
        # learnable roles, initialised from the printed card
        land0 = torch.tensor(np.where(C.is_land, 4.0, -4.0), dtype=torch.float32)
        mvb = torch.tensor(np.minimum(C.mv, NB - 1).astype(int))
        mv0 = F.one_hot(mvb, NB).float() * 6.0
        crea0 = torch.tensor(np.where(C.is_creature, 4.0, -4.0), dtype=torch.float32)
        src0 = torch.tensor(np.where(C.makes > 0, 4.0, -4.0), dtype=torch.float32)
        for k, v in (("land0", land0), ("mv0", mv0), ("crea0", crea0), ("src0", src0)):
            self.register_buffer(k, v)
        self.register_buffer("pips", torch.tensor(C.pips, dtype=torch.float32))
        self.d_land = nn.Parameter(torch.zeros(n))
        self.d_mv = nn.Parameter(torch.zeros(n, NB))
        self.d_crea = nn.Parameter(torch.zeros(n))
        self.d_src = nn.Parameter(torch.zeros(n, 5))
        nf = NB + 1 + 1 + 5 + 5 + 5 + 5 + 3
        self.struct = nn.Sequential(nn.Linear(nf, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                    nn.Linear(hidden, 1))
        self.ctx = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.beta = nn.Parameter(torch.zeros(()))

    def roles(self):
        land = torch.sigmoid(self.land0 + self.d_land)
        mv = torch.softmax(self.mv0 + self.d_mv, -1)
        crea = torch.sigmoid(self.crea0 + self.d_crea)
        src = torch.sigmoid(self.src0 + self.d_src)
        return land, mv, crea, src

    def card_vecs(self):
        return self.card(self.X) + self.emb

    def quality(self):
        """Per-card quality q_c (the additive part)."""
        return self.q_lin(self.card_vecs()).squeeze(-1) + self.q_card

    def features(self, n):
        land, mv, crea, src = self.roles()
        L = n @ land
        H = (n * (1 - land)) @ mv                       # curve of the non-land part
        K = n @ crea
        S = n @ src                                      # coloured sources
        R = n @ self.pips                                # coloured pips
        N = n.sum(-1, keepdim=True)
        ratio = R / (S + 1.0)
        need = torch.tanh(R / 2.0)                       # which colours the deck asks for
        short = need * F.relu(4.0 - S) / 4.0             # a colour asked for with fewer than ~4 sources
        ncol = need.sum(-1, keepdim=True)
        return torch.cat([H / 10, L[:, None] / 17, K[:, None] / 15, S / 8, R / 10, ratio, short,
                          ncol / 2, (N - 40) / 2, (L[:, None] - 17) ** 2 / 4], -1)

    def forward(self, n, parts=False):
        """s(deck) for a batch of count vectors n (B x 286)."""
        q = self.quality()
        v = self.card_vecs()
        s_q = n @ q
        s_s = self.struct(self.features(n)).squeeze(-1)
        s_c = self.ctx((n @ v) / n.sum(-1, keepdim=True).clamp(min=1)).squeeze(-1)
        if parts:
            return s_q, s_s, s_c
        return s_q + s_s + s_c

    def reg(self):
        """Penalty keeping learned corrections small unless the games ask for them."""
        return (self.emb.pow(2).mean() + self.q_card.pow(2).mean() + 0.1 * self.d_land.pow(2).mean()
                + 0.1 * self.d_mv.pow(2).mean() + 0.1 * self.d_crea.pow(2).mean() + 0.1 * self.d_src.pow(2).mean())


def games_tensors(games, decks):
    """games: dicts with decks [a, b], winner (seat), starter; decks: name -> count vector.
    -> (A, B, play, y) tensors; draws, stalls and skipped games are dropped."""
    A, B, P, Y = [], [], [], []
    for g in games:
        w = g.get("winner", -1)
        if w not in (0, 1):
            continue
        a, b = g["decks"]
        if a not in decks or b not in decks:
            continue
        A.append(decks[a]); B.append(decks[b])
        P.append(1.0 if g.get("starter", 0) == 0 else -1.0)
        Y.append(1.0 if w == 0 else 0.0)
    return (torch.tensor(np.array(A)), torch.tensor(np.array(B)), torch.tensor(P), torch.tensor(Y))


def loss_fn(m, A, B, P, Y, reg=1e-3):
    logit = m(A) - m(B) + m.beta * P
    return F.binary_cross_entropy_with_logits(logit, Y), reg * m.reg()


def train(games, decks, val_games=None, seed=0, epochs=40, lr=3e-3, wd=1e-4, batch=1024, reg=1e-3, bootstrap=False,
          init=None, patience=4, log=None):
    """Fit one DeckNet. With val_games, early-stop on their log-loss; bootstrap resamples the games (ensembles)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    A, B, P, Y = games_tensors(games, decks)
    if bootstrap:
        idx = torch.tensor(rng.integers(0, len(Y), len(Y)))
        A, B, P, Y = A[idx], B[idx], P[idx], Y[idx]
    m = DeckNet()
    if init is not None:
        m.load_state_dict(init.state_dict())
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    V = games_tensors(val_games, decks) if val_games else None
    best, best_state, bad = math.inf, None, 0
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Y))
        tot = 0.0
        for i in range(0, len(Y), batch):
            j = perm[i:i + batch]
            l, r = loss_fn(m, A[j], B[j], P[j], Y[j], reg)
            opt.zero_grad(); (l + r).backward(); opt.step()
            tot += l.item() * len(j)
        if V is not None:
            m.eval()
            with torch.no_grad():
                vl = loss_fn(m, *V, reg)[0].item()
            if log:
                log(f"  epoch {ep}: train {tot / len(Y):.4f} val {vl:.4f}")
            if vl < best - 1e-4:
                best, best_state, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        m.load_state_dict(best_state)
    m.eval()
    m.val_loss = best
    return m


class Ensemble:
    def __init__(self, members):
        self.members = members

    @torch.no_grad()
    def scores(self, n):
        """(K x B) scores of a batch of count vectors (numpy or tensor)."""
        n = torch.as_tensor(np.asarray(n, dtype=np.float32))
        return torch.stack([m(n) for m in self.members]).numpy()

    def save(self, path):
        torch.save([m.state_dict() for m in self.members], path)

    @classmethod
    def load(cls, path):
        ms = []
        for sd in torch.load(path):
            m = DeckNet(); m.load_state_dict(sd); m.eval(); ms.append(m)
        return cls(ms)
