"""Training targets from a batch's search statistics.

Policy targets are per candidate (a distribution within each state):
  visits  visits^(1/T), normalised
  cq      Gumbel MuZero's completed-Q improved policy: softmax(log prior + sigma(normalised completed q))
Value target: the outcome, optionally blended with the search's root value.
States with fewer than 2 candidates or no visits get no policy loss; outcome -1 gets no value loss.
"""
from __future__ import annotations

import torch

from .models import segment_logsumexp


def _seg_sum(x: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    return x.new_zeros(n).index_add(0, seg, x)


def _seg_reduce(x: torch.Tensor, seg: torch.Tensor, n: int, how: str, fill: float) -> torch.Tensor:
    return x.new_full((n,), fill).scatter_reduce(0, seg, x, how, include_self=False)


def _seg_softmax(logits: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    return (logits - segment_logsumexp(logits, seg, n)[seg]).exp()


def policy_mask(b) -> torch.Tensor:
    """[B] bool: states with >= 2 candidates and some visits."""
    ncand = _seg_sum(torch.ones_like(b.cand_visits), b.cand_state, b.B)
    total = _seg_sum(b.cand_visits.float(), b.cand_state, b.B)
    return (ncand >= 2) & (total > 0)


def visits_target(b, temp: float = 1.0) -> torch.Tensor:
    n = b.cand_visits.float().clamp(min=0)
    if temp != 1.0:
        n = torch.where(n > 0, n.pow(1.0 / temp), n)
    tot = _seg_sum(n, b.cand_state, b.B)
    return n / tot.clamp(min=1e-12)[b.cand_state]


def cq_target(b, c_visit: float = 50.0, c_scale: float = 0.1) -> torch.Tensor:
    """softmax(log(prior + 1e-8) + (c_visit + max N) * c_scale * qnorm) per state, where qnorm is the
    per-state min-max normalisation of the completed q (q for visited candidates, else v_mix; a
    constant completed q normalises to 0.5) and
    v_mix = (root_value + sum N / (sum over visited of prior) * sum over visited of prior * q) / (1 + sum N)."""
    seg, B = b.cand_state, b.B
    N = b.cand_visits.float().clamp(min=0)
    P = b.cand_prior.float().clamp(min=0)
    Q = b.cand_q.float()
    visited = N > 0
    sumN = _seg_sum(N, seg, B)
    p_vis = _seg_sum(torch.where(visited, P, torch.zeros_like(P)), seg, B)
    pq_vis = _seg_sum(torch.where(visited, P * Q, torch.zeros_like(P)), seg, B)
    rv = b.root_value.float()
    weighted = torch.where(p_vis > 0, sumN / p_vis.clamp(min=1e-12) * pq_vis, torch.zeros_like(p_vis))
    v_mix = (rv + weighted) / (1 + sumN)
    cq = torch.where(visited, Q, v_mix[seg])
    lo = _seg_reduce(cq, seg, B, "amin", 0.0)
    hi = _seg_reduce(cq, seg, B, "amax", 0.0)
    span = (hi - lo)[seg]
    qn = torch.where(span > 0, (cq - lo[seg]) / span.clamp(min=1e-12), torch.full_like(cq, 0.5))
    maxN = _seg_reduce(N, seg, B, "amax", 0.0)
    sigma = (c_visit + maxN[seg]) * c_scale * qn
    return _seg_softmax(torch.log(P + 1e-8) + sigma, seg, B)


def policy_target(b, kind: str = "visits", temp: float = 1.0, c_visit: float = 50.0, c_scale: float = 0.1):
    if kind == "visits":
        return visits_target(b, temp)
    if kind == "cq":
        return cq_target(b, c_visit, c_scale)
    raise ValueError(f"unknown policy target {kind!r} (visits, cq)")


def value_target(b, blend: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """(target [B], mask [B]): (1 - blend) * outcome + blend * root_value, masked where outcome < 0."""
    z = b.outcome.float()
    mask = z >= 0
    t = z if blend == 0 else (1 - blend) * z + blend * b.root_value.float()
    return torch.where(mask, t, torch.zeros_like(t)), mask
