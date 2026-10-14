"""Small statistics shared by the evaluation code: Wilson score intervals and Spearman's rank correlation.

The XMage-era parts (GAME_SUMMARY log parsing, the per-generation dashboard tables) went with XMage; card
statistics on dzgorge games are in `draftzero.eval.gih`, paired-game scores in `draftzero.eval.paired`.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


def wilson(wins: float, n: int, z: float = 1.96) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """(rate, low, high): the win rate and its Wilson score interval; (None, None, None) for no games."""
    if n == 0:
        return None, None, None
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def ranks(x: Sequence[float]) -> np.ndarray:
    """1-based ranks, ties sharing their average rank."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman's rho (Pearson on average ranks); None for fewer than 3 points or a constant side."""
    if len(xs) != len(ys):
        raise ValueError(f"spearman: {len(xs)} vs {len(ys)} values")
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])
