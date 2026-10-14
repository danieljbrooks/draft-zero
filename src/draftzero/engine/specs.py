"""engine/specs.py — gorge policy specs as data.

dzgorge names a seat's policy with one colon string (cmd/dzgorge/policy.go in the fork), e.g.
`az:sims=100:explore:autopay:oppnodes:oppfull:morekinds:closedlist:cpuct=0.25:topk=6:gonet=/x/v3.dzgw`.
PolicySpec holds the parts DraftZero sets as typed fields and renders them in one fixed order:

    kind [sims] [explore] [keys...] [closedlist] [cpuct] [topk] [pcr [cheap] [pcrk]] [gonet] [extra...]

`parse` is its inverse: parse(render(x)) == x for every spec that constructs, so a spec can live in a config, a
manifest or a game record as a string. Construction normalises what the string cannot tell apart: playout-cap
randomization off (pcr None or 0) drops cheap and pcrk, and a no-search kind (bot, sbtactical, random) drops the
search fields and the default keys, so PolicySpec(kind="sbtactical").render() == "sbtactical".

Bare flags the fork knows (FLAG_KEYS) go in `keys`; anything else the fork accepts (kinds=..., worlds=4, fpu=0.1,
ksims=..., skippass=2, net=..., remote=...) goes in `extra`, verbatim and in order. The fork reads a spec left to
right and a later key wins (morekinds after kinds=priority means every kind; skippass after skippass=3 means 1),
so a bare flag written after an extra entry stays in `extra`, in its place: extra may hold bare flags, just not
first. gonet= takes no ':' (the fork splits the spec on colons); remote= takes the rest of the spec, so it must be
the last extra.
"""
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass

FULL = ("autopay", "oppnodes", "oppfull", "morekinds")
KINDS = ("az", "prior", "bot", "sbtactical", "random")
SEARCH_KINDS = ("az", "prior")
# The searched decision kinds (the fork's :kinds= names), keys of pcrk.
KIND_NAMES = ("priority", "attackers", "blockers", "target", "modes", "optional", "choose", "multitarget")
# Bare flags that parse into `keys` (the rest of the fork's bare flags are fields: explore, closedlist).
FLAG_KEYS = ("autopay", "oppnodes", "oppfull", "morekinds", "nonoise", "skippass", "mull", "keep7")
# Keys with a field of their own: never in keys or extra.
FIELD_KEYS = ("sims", "explore", "closedlist", "cpuct", "topk", "gonet", "pcr", "cheap", "pcrk")


def _int(name: str, v, lo: int) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool) or not float(v).is_integer():
        raise ValueError(f"PolicySpec.{name}={v!r}: want an integer")
    if int(v) < lo:
        raise ValueError(f"PolicySpec.{name}={v!r}: want >= {lo}")
    return int(v)


def _float(name: str, v, lo: float, hi: float) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValueError(f"PolicySpec.{name}={v!r}: want a number")
    f = float(v)
    if not (math.isfinite(f) and lo <= f <= hi):
        raise ValueError(f"PolicySpec.{name}={v!r}: want a number in [{lo}, {hi}]")
    return f


def _num(x: float) -> str:
    """A float as the shortest string that reads back exactly (Go's ParseFloat takes it)."""
    return repr(float(x))


def _pcrk(v) -> tuple[tuple[str, float], ...]:
    items = v.items() if isinstance(v, Mapping) else v
    out, seen = [], set()
    for k, p in items:
        if k not in KIND_NAMES:
            raise ValueError(f"PolicySpec.pcrk: unknown searched kind {k!r}, want one of {', '.join(KIND_NAMES)}")
        if k in seen:
            raise ValueError(f"PolicySpec.pcrk: kind {k!r} listed twice")
        seen.add(k)
        out.append((k, _float(f"pcrk[{k}]", p, 0.0, 1.0)))
    return tuple(out)


def _parse_pcrk(v: str) -> tuple[tuple[str, float], ...]:
    """The fork's pcrk= list: kind.P or kind=P entries, comma-separated."""
    out = []
    for kv in v.split(","):
        k, sep, p = kv.partition("=")
        if not sep:
            k, sep, p = kv.partition(".")
        if not sep or not k:
            raise ValueError(f"pcrk entry {kv!r}: want kind.P")
        out.append((k, float(p)))
    return tuple(out)


@dataclass(frozen=True)
class PolicySpec:
    """One seat's policy (see the module docstring). Fields left None are not rendered (the fork's defaults:
    sims 100, cpuct 1.5, no topk, cheap 40)."""
    kind: str = "az"            # az | prior | bot | sbtactical | random
    sims: int | None = 100
    cpuct: float | None = None
    topk: int | None = None
    explore: bool = False       # self-play only: sample by visits on turns 1-4, root noise
    closedlist: bool = False    # the opponent's list is drawn from the belief corpus (play -belief-*)
    gonet: str | None = None    # path to .dzgw (no ':' allowed)
    pcr: float | None = None    # playout-cap randomization: P(full search, recorded); None/0 = off
    cheap: int | None = None    # cheap-search sims (only rendered when pcr > 0; fork default 40)
    pcrk: tuple[tuple[str, float], ...] = ()  # per-kind full-search probs (KIND_NAMES); a Mapping is accepted
    keys: tuple[str, ...] = FULL  # bare flags (FLAG_KEYS) rendered in order
    extra: tuple[str, ...] = ()   # raw "k=v" or "flag" the fields do not cover, rendered last in order

    def __post_init__(self):
        put = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731 (frozen dataclass)
        if self.kind not in KINDS:
            raise ValueError(f"PolicySpec.kind={self.kind!r}: want one of {', '.join(KINDS)}")
        put("sims", _int("sims", self.sims, 1))
        put("topk", _int("topk", self.topk, 1))
        put("cpuct", _float("cpuct", self.cpuct, 0.0, math.inf))
        put("explore", bool(self.explore))
        put("closedlist", bool(self.closedlist))
        pcr = _float("pcr", self.pcr, 0.0, 1.0)
        if not pcr:     # off: nothing of playout-cap randomization is rendered, so nothing of it is kept
            put("pcr", None)
            put("cheap", None)
            put("pcrk", ())
        else:
            put("pcr", pcr)
            put("cheap", _int("cheap", self.cheap, 1))
            put("pcrk", _pcrk(self.pcrk))
        if self.gonet is not None:
            g = os.fspath(self.gonet)
            if not g or ":" in g:
                raise ValueError(f"PolicySpec.gonet={g!r}: want a non-empty path without ':' (the spec separator)")
            put("gonet", g)
        keys = tuple(self.keys)
        for k in keys:
            if k not in FLAG_KEYS:
                raise ValueError(f"PolicySpec.keys: {k!r} is not a bare key ({', '.join(FLAG_KEYS)}); "
                                 "put other keys in extra")
        put("keys", keys)
        extra = tuple(self.extra)
        for i, x in enumerate(extra):
            k = x.partition("=")[0]
            if not x or k in FIELD_KEYS or (x in FLAG_KEYS and i == 0):
                raise ValueError(f"PolicySpec.extra: {x!r} has a field or keys entry of its own (a bare key "
                                 "goes in extra only after another extra entry)")
            if ":" in x and not (k == "remote" and i == len(extra) - 1):
                raise ValueError(f"PolicySpec.extra: {x!r} has a ':' (only a last remote=... may)")
        put("extra", extra)
        if self.kind not in SEARCH_KINDS:
            # bot, sbtactical and random search nothing: clear the search fields and the default keys
            # (bot:autopay, bot:mull and the like stay expressible through explicit keys).
            for k, v in (("sims", None), ("explore", False), ("closedlist", False), ("cpuct", None), ("topk", None),
                         ("gonet", None), ("pcr", None), ("cheap", None), ("pcrk", ())):
                put(k, v)
            if keys == FULL:
                put("keys", ())
        elif self.kind == "prior" and (self.closedlist or self.pcr):
            raise ValueError("PolicySpec: closedlist and pcr are az keys (the fork refuses them on prior)")

    def render(self) -> str:
        """The dzgorge spec string, e.g. "az:sims=100:explore:autopay:oppnodes:oppfull:morekinds:closedlist:
        cpuct=0.25:topk=6:gonet=/x/v3.dzgw"."""
        parts = [self.kind]
        if self.sims is not None:
            parts.append(f"sims={self.sims}")
        if self.explore:
            parts.append("explore")
        parts.extend(self.keys)
        if self.closedlist:
            parts.append("closedlist")
        if self.cpuct is not None:
            parts.append(f"cpuct={_num(self.cpuct)}")
        if self.topk is not None:
            parts.append(f"topk={self.topk}")
        if self.pcr:
            parts.append(f"pcr={_num(self.pcr)}")
            if self.cheap is not None:
                parts.append(f"cheap={self.cheap}")
            if self.pcrk:
                parts.append("pcrk=" + ",".join(f"{k}.{_num(p)}" for k, p in self.pcrk))
        if self.gonet is not None:
            parts.append(f"gonet={self.gonet}")
        parts.extend(self.extra)
        return ":".join(parts)

    __str__ = render

    @staticmethod
    def parse(s: str) -> "PolicySpec":
        """The inverse of render, and a reader for any spec the fork accepts: unknown keys land in extra, and so
        does a bare key after them (see the module docstring), so the rendered spec means what `s` meant."""
        s = s.strip()
        rest = ""
        if (i := s.find(":remote=")) >= 0:     # remote=unix:/path and remote=tcp:host:port take the rest
            s, rest = s[:i], s[i + 1:]
        kind, *tokens = s.split(":")
        f: dict = {"kind": kind, "sims": None, "keys": [], "extra": []}
        for t in tokens:
            k, eq, v = t.partition("=")
            try:
                if not t:
                    raise ValueError("empty key")
                elif eq and k == "sims":
                    f["sims"] = int(v)
                elif t in ("explore", "closedlist"):
                    f[t] = True
                elif eq and k in ("cpuct", "pcr"):
                    f[k] = float(v)
                elif eq and k == "topk":
                    f["topk"] = int(v) or None          # the fork's topk=0 is no top-k cut, i.e. none
                elif eq and k == "cheap":
                    f["cheap"] = int(v)
                elif eq and k == "gonet":
                    f["gonet"] = v
                elif eq and k == "pcrk":
                    f["pcrk"] = _parse_pcrk(v)
                elif not eq and t in FLAG_KEYS and not f["extra"]:
                    f["keys"].append(t)
                else:                                   # a bare key after an extra entry stays in its place
                    f["extra"].append(t)
            except ValueError as e:
                raise ValueError(f"policy spec {s!r}: {t!r}: {e}") from None
        if rest:
            f["extra"].append(rest)
        return PolicySpec(**f)
