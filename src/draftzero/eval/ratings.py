"""Bradley-Terry ratings: one fit over every evaluated match, so each player gets a rating on one scale (Elo
points, the anchor = 0) without saturating against any one opponent (docs/031's ladder; v0.1.0
gorge/recipe/ratings.py).

    fit_bt([("v4", "sbtactical", 1.0), ("v4", "sbtactical", 0.5), ...])        -> {"v4": (elo, se), ...}
    fit_bt(summary_triples([{"a": "v4", "b": "prev", "version": 4, "score": .56, "n_games": 200}]))
    fit_bt(game_triples(games_rows))                                           players = shortened policy specs

A result is (a, b, s) or (a, b, s, weight): a's points against b (1 win, 0 loss, 0.5 draw; any value in
[0, 1]) in `weight` games (default 1). A match's mean score in n games is the same likelihood as its n games, so
summaries fit exactly like the games they summarise. The fit is Newton's method on the log-likelihood with a
small ridge (which keeps a perfect record finite), the anchor fixed at 0. Standard errors come from the
curvature and treat games as independent (the two games of a pair are not), so read them as approximate; a
player not connected to the anchor by any chain of matches gets a huge one.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np

ELO = 400 / math.log(10)
SHARED = ("autopay", "oppnodes", "oppfull", "morekinds", "topk=6")   # gorge/recipe/ratings.py's


def fit_bt(results: Iterable[tuple], *, anchor: str = "sbtactical", ridge: float = 1e-6,
           min_games: float = 0, max_iter: int = 200) -> dict[str, tuple[float, float]]:
    """{player: (elo, se)} from (a, b, score[, weight]) results. The anchor is pinned at (0, 0); when it never
    played, the alphabetically first player is. Players with fewer than `min_games` (weighted) games are left
    out, with their results. Self-matches (a == b) are ignored."""
    acc: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for r in results:
        a, b, s = r[0], r[1], float(r[2])
        w = float(r[3]) if len(r) > 3 else 1.0
        if a == b or w <= 0:
            continue
        if not 0.0 <= s <= 1.0:
            raise ValueError(f"score {s} for {a} vs {b} is outside [0, 1]")
        acc[(a, b)][0] += w
        acc[(a, b)][1] += w * s
    count: dict[str, float] = defaultdict(float)
    for (a, b), (w, _) in acc.items():
        count[a] += w
        count[b] += w
    keep = {p for p, c in count.items() if c >= min_games}
    acc = {k: v for k, v in acc.items() if k[0] in keep and k[1] in keep}
    players = sorted({p for k in acc for p in k})
    if not players:
        return {}
    ix = {p: i for i, p in enumerate(players)}
    A = np.array([ix[a] for a, _ in acc], dtype=int)
    B = np.array([ix[b] for _, b in acc], dtype=int)
    W = np.array([v[0] for v in acc.values()])
    S = np.array([v[1] for v in acc.values()])     # weighted points
    n = len(players)
    pin = ix.get(anchor, 0)
    free = np.array([i for i in range(n) if i != pin], dtype=int)
    r = np.zeros(n)
    for _ in range(max_iter):
        g, H = _grad_hess(r, A, B, W, S, ridge)
        if not len(free):
            break
        step = np.linalg.solve(H[np.ix_(free, free)], g[free])
        step = np.clip(step, -2.0, 2.0)           # damped Newton: no overshoot from a flat start
        r[free] -= step
        if np.abs(step).max() < 1e-10:
            break
    g, H = _grad_hess(r, A, B, W, S, ridge)
    se = np.zeros(n)
    if len(free):
        cov = np.linalg.inv(-H[np.ix_(free, free)])
        se[free] = np.sqrt(np.clip(np.diag(cov), 0, None))
    return {p: (float(ELO * r[i]), float(ELO * se[i])) for p, i in ix.items()}


def _grad_hess(r, A, B, W, S, ridge):
    """Gradient and Hessian of the penalised log-likelihood sum(S log p + (W - S) log(1 - p)) - ridge |r|^2 / 2."""
    n = len(r)
    p = 1 / (1 + np.exp(-(r[A] - r[B])))
    d = S - W * p
    g = -ridge * r
    np.add.at(g, A, d)
    np.add.at(g, B, -d)
    h = W * p * (1 - p)
    H = -ridge * np.eye(n)
    np.add.at(H, (A, A), -h)
    np.add.at(H, (B, B), -h)
    np.add.at(H, (A, B), h)
    np.add.at(H, (B, A), h)
    return g, H


def table(results: Iterable[tuple], **kw) -> list[dict]:
    """fit_bt as rows, best first: [{player, elo, se, games}]."""
    results = list(results)
    games: dict[str, float] = defaultdict(float)
    for r in results:
        if r[0] != r[1]:
            w = float(r[3]) if len(r) > 3 else 1.0
            games[r[0]] += w
            games[r[1]] += w
    fit = fit_bt(results, **kw)
    return sorted(({"player": p, "elo": e, "se": s, "games": games[p]} for p, (e, s) in fit.items()),
                  key=lambda x: -x["elo"])


# --------------------------------------------------------------------------------------------- building results

def version_name(v: int | str) -> str:
    """The rating name of a player version: 4, "4", "v4", "v0004" -> "v4"."""
    m = re.fullmatch(r"v?0*(\d+)", str(v))
    return f"v{int(m[1])}" if m else str(v)


def opponent_name(opp: str, version: int | None = None) -> str:
    """An evaluator opponent as a rating name: "prev" -> v{version-1}, "v0003" -> "v3", others unchanged
    ("sbtactical", "bot", "nonet", "file:/x.dzgw")."""
    if opp == "prev":
        if version is None:
            raise ValueError("opponent 'prev' needs the version")
        return version_name(int(version) - 1)
    return version_name(opp) if re.fullmatch(r"v\d+", opp) else opp


def summary_triples(summaries: Iterable[Mapping]) -> list[tuple[str, str, float, float]]:
    """(a, b, score, games) results from paired summaries. Each summary is a `score_pairs` dict plus the two
    players: "a" (or "player", or "version" alone -> v{version}) and "b" (or "opponent"; "prev" is resolved
    with "version"). Games = n_games (else 2 * n_pairs, else n). Summaries without a score are skipped."""
    out = []
    for s in summaries:
        if s.get("score") is None:
            continue
        version = s.get("version")
        a = s.get("a") or s.get("player") or (version_name(version) if version is not None else None)
        b = s.get("b") or s.get("opponent")
        if a is None or b is None:
            raise ValueError(f"summary without players: {dict(s)}")
        a, b = version_name(a) if re.fullmatch(r"v?\d+", str(a)) else str(a), opponent_name(str(b), version)
        n = s.get("n_games") or 2 * (s.get("n_pairs") or 0) or s.get("n") or 0
        if n:
            out.append((a, b, float(s["score"]), float(n)))
    return out


def label(spec: str, shared: Iterable[str] = SHARED) -> str:
    """A policy spec as a player name: the keys every gauntlet player shares dropped, a network named by its
    directory ("…/versions/player/v0004/model.dzgw" -> "net=v0004")."""
    parts = spec.split(":")
    keys = []
    for p in parts[1:]:
        if p in shared:
            continue
        keys.append("net=" + Path(p[6:]).parent.name if p.startswith("gonet=") else p)
    return ":".join([parts[0]] + keys)


def game_triples(games: Iterable[dict], name: Callable[[str], str] | Mapping[str, str] | None = None
                 ) -> list[tuple[str, str, float]]:
    """(A's player, B's player, A's points) per game of `dzgorge play` rows; errors and self-play skipped.
    `name` maps a policy spec to a player name (a function or a dict; default `label`)."""
    if name is None:
        name = label
    nm = name.get if isinstance(name, Mapping) else name
    out = []
    for g in games:
        r = g.get("result")
        if r not in ("A", "B", "draw", "stall"):
            continue
        side, pols = list(g["side"]), g["pols"]
        pa, pb = pols[side.index("A")], pols[side.index("B")]
        if pa == pb:
            continue
        a, b = nm(pa) or pa, nm(pb) or pb
        out.append((a, b, {"A": 1.0, "B": 0.0}.get(r, 0.5)))
    return out
