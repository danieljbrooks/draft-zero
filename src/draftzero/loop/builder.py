"""The sealed deck builder on the actor-learner loop (docs/033 work plan step 6; the method is docs/032's, the code
draftzero.build). On only with builder.learn; with it off nothing here runs and the player loop is unchanged.

Three roles, beside the player's, sharing the run's store and control.json:

  builder actor      many. Each chunk: draws builder.pools_per_chunk train pools from the run's draftlab pools
                     (`dzgorge sealed-pools`, cached per machine: build.pools), builds decks from each with the
                     current builder version (build.kinds: b best, t Thompson, p perturbed, l land-shifted, r random;
                     an untrained version 0 builds four random deck-shaped decks a pool), pairs decks of different
                     pools (build.selfplay.pair_up: each pairing dealt twice, the seats swapped, one seed) and plays
                     them with Gorge.play(deckfile, pairs_file, single=True). The pilot is builder.pilot (a fixed
                     .dzgw) when set, else the current player version's .dzgw when player.learn, else the search
                     without a network, at builder.sims (null: actor.sims; 0: gorge's heuristic bot). It uploads
                     builder/shards/<actor>/<seq>/ {decks.jsonl, games.jsonl, meta.json last} with its heartbeat and
                     metrics in one put_many, and idles once the store holds builder.games_per_version decided games
                     of the current builder version (docs/032's generation).
  builder learner    one (builder/learner/lease.json). Publishes version 0 (builder.init's ens.pt, else untrained:
                     meta only); ingests shards; once the newest version v has games_per_version decided games, fits
                     the ensemble (build.train.fit_games: builder.members bootstrap members, Bradley-Terry on the
                     results) on the window (builder.window: the newest `size` versions' games, or the newest shards
                     holding `size` games), warm-started from v's ensemble at builder.warm_lr when builder.warm_start,
                     bounds the land count by build.model.land_support (builder.land_support), and publishes
                     versions/builder/v{v+1:04d}/{ens.pt, meta.json}, then versions/builder/latest.json.
  builder evaluator  a few. Plays the evaluation shards the controller assigns (control["builder_evals"]): version v's
                     best decks against best23@1's on builder.eval.pools held-out draftlab pools (base seed
                     builder.eval.seed, never the train seed), pools 2i and 2i+1 crossed, each crossing dealt
                     builder.eval.pairs times with the seats swapped, piloted as the actors' games are (the player
                     version current when the job opened) at builder.eval.sims. evals/<job>/<shard>.games.jsonl;
                     the last shard's evaluator scores the job (score_match: v's points, 1 a win, 0.5 a draw or
                     stall, the mean over pool pairs with its standard error) into evals/<job>/summary.json and an
                     eval.score metric (opponent best23, model builder).

The controller (draftzero.loop.controller) moves control["version"]["builder"] to each new builder version, opens
one evaluation job per version v >= 1 with v % builder.eval.every_versions == 0, assigns its shards to live builder
evaluators, and counts builder versions against budget.max_versions.

    python -m draftzero.loop.cli builder-actor|builder-learner|builder-evaluator --config C --store S [--machine M]
"""
from __future__ import annotations

import collections
import contextlib
import dataclasses
import json
import math
import os
import shutil
import time
import uuid
from pathlib import Path

from draftzero.loop import control as ctl
from draftzero.loop.actor import Actor, belief, policy_spec
from draftzero.loop.evaluator import Evaluator
from draftzero.loop.learner import LeaseHeld

COMPONENT = "builder"
SHARDS = "builder/shards/"
LEASE = "builder/learner/lease.json"
INGESTED = "builder/learner/ingested.json"
LEARNER_STATUS = "builder/learner/status.json"
EVALS = "builder_evals"                          # control.json's list of builder evaluation jobs
EVAL_SHARD_POOLS = 20                            # pools a builder evaluation shard plays (even: pools pair up)
TAG, OPP = "m", "h"                              # deck-name prefixes in an evaluation: the builder's, the opponent's


# ------------------------------------------------------------------------------------------------ helpers

def enabled(cfg) -> bool:
    return bool(ctl.get(cfg, "builder.learn", False))


def player_learns(cfg) -> bool:
    return bool(ctl.get(cfg, "player.learn", True))


def shard_dir(actor: str, seq: int) -> str:
    return f"{SHARDS}{actor}/{int(seq):06d}"


def parse_shard(path: str) -> tuple[str, int] | None:
    """(actor, seq) of a builder shard folder or of any file in one."""
    parts = path.strip("/").split("/")
    if len(parts) >= 4 and parts[:2] == ["builder", "shards"] and parts[3].isdigit():
        return parts[2], int(parts[3])
    return None


def job_id(version: int, opponent: str = "best23") -> str:
    return f"builder-v{int(version):04d}-vs-{opponent}"


def pool_kw(cfg, *, held_out: bool = False) -> dict:
    """build.pools.cached's keywords for the run's train pools, or (held_out) its evaluation pools with best23."""
    p = ctl.as_dict(ctl.get(cfg, "builder.pools"))
    kw = {"set": p.get("set", "FDN"), "packs": int(p.get("packs", 6)), "collation": p.get("collation", "arena")}
    if held_out:
        e = ctl.as_dict(ctl.get(cfg, "builder.eval"))
        return {**kw, "n": int(e.get("pools", 200)), "seed": int(e.get("seed", 2)), "best23": True}
    return {**kw, "n": int(p.get("count", 5000)), "seed": int(p.get("seed", 1)), "best23": False}


def pilot_sims(cfg, play: dict | None = None) -> int:
    """The actors' pilot simulations: builder.sims, else actor.sims (control["play"]'s, which a plan may change)."""
    s = ctl.get(cfg, "builder.sims")
    if s is None:
        s = (play or {}).get("sims") or ctl.get(cfg, "actor.sims", 100)
    return int(s)


def pilot_spec(cfg, sims: int, gonet: Path | None):
    """The pilot both seats play: gorge's heuristic bot at 0 simulations, else the run's search (search.keys,
    cpuct; topk and gonet with a network) without exploration, closed-list only with builder.closedlist."""
    if int(sims) <= 0:
        return "bot"
    spec = policy_spec(cfg, sims=int(sims), explore=False, gonet=gonet)
    return dataclasses.replace(spec, closedlist=bool(ctl.get(cfg, "builder.closedlist", False)))


def fixed_pilot(cfg) -> Path | None:
    p = ctl.get(cfg, "builder.pilot")
    if not p:
        return None
    from draftzero.engine.gorge import resolve_path
    path = resolve_path(p)
    if not path.exists():
        raise FileNotFoundError(f"builder.pilot {p}: no such file")
    return path


def pilot_belief(cfg):
    return belief(cfg) if ctl.get(cfg, "builder.closedlist", False) else None


def deck_line(name: str, deck, **extra) -> str:
    """A -deckfile line (a gorge deck file) with `extra` fields (dzgorge reads name and cards only)."""
    from draftzero.build import decks as D
    return json.dumps({**json.loads(D.gorge_deck(name, deck)), **extra}, sort_keys=True)


def deal_seed(seed: int, pool: int, rep: int) -> int:
    """The deal of an evaluation crossing: a function of the job's seed, the builder deck's pool and the repeat,
    so every version meets the same deals."""
    import numpy as np
    return int(np.random.default_rng([int(seed), int(pool), int(rep)]).integers(1, 2**62))


def match_lines(first: int, pools: int, pairs: int, seed: int) -> list[tuple[str, str, int]]:
    """An evaluation shard's games (deck A in seat 0, deck B, seed): pools first..first+pools-1 in pairs (i, i+1);
    the builder's deck from each against the opponent's from the other, `pairs` deals each, both seat orders."""
    out = []
    for i in range(first, first + pools - 1, 2):
        for a, b in ((i, i + 1), (i + 1, i)):
            for r in range(pairs):
                s = deal_seed(seed, a, r)
                out += [(f"{TAG}{a}", f"{OPP}{b}", s), (f"{OPP}{b}", f"{TAG}{a}", s)]
    return out


def score_match(games, tag: str = TAG) -> dict:
    """The builder's score in an evaluation's games: per game 1 for a win by the builder's deck, 0 a loss, 0.5 a
    draw or stall (errors and skipped games are dropped); the score is the mean over pool pairs (the builder deck's
    pool // 2) of their means, with the standard error over pool pairs, as docs/032's win rate (which counted only
    decided games: `decided_winrate`)."""
    groups: dict[int, list[float]] = collections.defaultdict(list)
    n = collections.Counter()
    for g in games:
        decks = g.get("decks") or []
        if len(decks) != 2:
            continue
        mine = 0 if str(decks[0]).startswith(tag) else 1 if str(decks[1]).startswith(tag) else None
        if mine is None:
            continue
        w, r = g.get("winner", -1), g.get("result")
        if w in (0, 1):
            pts = 1.0 if w == mine else 0.0
            n["wins" if pts else "losses"] += 1
        elif r in ("draw", "stall"):
            pts = 0.5
            n["draws" if r == "draw" else "stalls"] += 1
        else:
            n["errors"] += 1
            continue
        groups[int(str(decks[mine])[len(tag):]) // 2].append(pts)
    means = [sum(v) / len(v) for v in groups.values()]
    k = len(means)
    out = {"score": None, "se": None, "lo": None, "hi": None, "n_pairs": k, "n_games": sum(map(len, groups.values())),
           "wins": n["wins"], "losses": n["losses"], "draws": n["draws"], "stalls": n["stalls"], "errors": n["errors"],
           "decided_winrate": n["wins"] / (n["wins"] + n["losses"]) if n["wins"] + n["losses"] else None}
    if k:
        m = sum(means) / k
        se = math.sqrt(sum((x - m) ** 2 for x in means) / max(k - 1, 1)) / math.sqrt(k)
        out.update(score=m, se=se, lo=max(0.0, m - 1.96 * se), hi=min(1.0, m + 1.96 * se))
    return out


def make_job(version: int, cfg, *, player_version: int | None = None, now: float | None = None) -> dict:
    """An evaluation job for builder version `version` (the controller opens it; evaluators play its shards).
    player_version: the pilot's player version (None: builder.pilot or the search without a network)."""
    e = ctl.as_dict(ctl.get(cfg, "builder.eval"))
    pools = int(e["pools"])
    shards = [{"id": i, "first_pool": lo, "pools": min(EVAL_SHARD_POOLS, pools - lo), "machine": None,
               "assigned": None, "done": False} for i, lo in enumerate(range(0, pools, EVAL_SHARD_POOLS))]
    return {"job": job_id(version, e["opponent"]), "component": COMPONENT, "version": int(version),
            "opponent": e["opponent"], "pools": pools, "pairs": int(e["pairs"]), "sims": int(e["sims"]),
            "seed": int(e["seed"]), "player_version": player_version, "created": ctl.now_iso(now), "shards": shards}


class ShardIndex:
    """Every complete builder shard's meta.json (complete once its meta.json is in), read once each."""

    def __init__(self, store, known: dict[str, dict] | None = None):
        self.store = store
        self.metas: dict[str, dict] = dict(known or {})

    def refresh(self) -> list[dict]:
        new = []
        for p in self.store.list(SHARDS):
            if not p.endswith("/meta.json"):
                continue
            d = p[: -len("/meta.json")]
            if d in self.metas or parse_shard(d) is None:
                continue
            m = self.store.read_json(p)
            if m is None:
                continue
            m["path"] = d
            self.metas[d] = m
            new.append(m)
        return sorted(new, key=lambda m: (m.get("version") or 0, m.get("seq", 0), m.get("actor", "")))

    def decided(self, version: int | None = None) -> int:
        return sum(int(m.get("decided", 0)) for m in self.metas.values()
                   if version is None or m.get("version") == version)


def _load_builder(path: Path | None, threads: int | None = None):
    """A build.kinds.Builder for an ens.pt (None: untrained); threads: torch's, for building with it."""
    from draftzero.build.kinds import Builder
    if threads:
        import torch
        torch.set_num_threads(int(threads))
    if path is None:
        return Builder(None)
    from draftzero.build.model import Ensemble
    return Builder(Ensemble.load(path))


def _fetch_builder(store, version: int, root: Path) -> tuple[dict, Path | None]:
    meta, files = ctl.fetch_version(store, version, root / f"v{version:04d}", ("ens.pt",), component=COMPONENT)
    return meta, files.get("ens.pt")


class _Wait(Exception):
    """Nothing to play yet; the message says why."""


# ------------------------------------------------------------------------------------------------ actor

class BuilderActor(Actor):
    """A builder actor (module docstring). Actor's loop, idling, heartbeats and upload spool, with builder chunks."""

    ROLE = "builder-actor"

    def __init__(self, store, cfg, machine: str, work_dir: Path, *, gorge=None, log=print, poll_s: float = 30.0,
                 make_pools=None):
        super().__init__(store, cfg, machine, work_dir, gorge=gorge, log=log, poll_s=poll_s)
        self.base = cfg
        self.index = ShardIndex(store)
        self.make_pools = make_pools                     # tests: replaces `dzgorge sealed-pools`
        self.builders: dict[int, object] = {}
        self._pools: list[dict] | None = None
        self.decided = 0

    @property
    def pools(self) -> list[dict]:
        if self._pools is None:
            from draftzero.build import pools
            self._pools = pools.cached(self.gorge, self.wd / "pools", make=self.make_pools, **pool_kw(self.cfg))
            self.log(f"builder actor {self.machine}: {len(self._pools)} train pools")
        return self._pools

    def step(self) -> str:
        """One pass: "stop", "wait", "paused", "played" or "failed"."""
        c = ctl.read(self.store)
        if c is None:
            return self._idle("waiting", "no control.json yet (is the controller running?)")
        if c.get("stop"):
            self._idle("stopped", "the run is stopping", force=True)
            return "stop"
        if c.get("paused"):
            return self._idle("paused", "the run is paused", ret="paused")
        bv = (c.get("version") or {}).get(COMPONENT)
        if bv is None:
            return self._idle("waiting", "waiting for the builder learner to publish version 0")
        want = int(ctl.get(self.base, "builder.games_per_version"))           # builder.* is fixed for a run
        self.index.refresh()
        left = want - self.index.decided(bv)
        if left <= 0:
            return self._idle("waiting", f"builder version {bv} has its {want} decided games")
        self.cfg = ctl.effective(self.base, self.store)                       # live fields: actor.sims, ...
        play ={**ctl.as_dict(ctl.get(self.cfg, "actor")), **(c.get("play") or {})}
        sims = pilot_sims(self.cfg, play)
        try:
            gonet, pv = self.pilot(c, sims)
            builder = self.ensure_builder(bv)
        except _Wait as e:
            return self._idle("waiting", str(e))
        except FileNotFoundError as e:                   # a version control.json names that this store lacks yet
            return self._idle("waiting", f"{e}; retrying at each poll")
        n = min(int(ctl.get(self.cfg, "builder.games_per_chunk")), left + left % 2)
        spec = pilot_spec(self.cfg, sims, gonet)
        return "played" if self.play_decks(bv, pv, builder, spec, n, sims) else "failed"

    def pilot(self, c: dict, sims: int) -> tuple[Path | None, int | None]:
        """(the pilot's .dzgw or None, the player version it is or None)."""
        if sims <= 0:
            return None, None
        fixed = fixed_pilot(self.cfg)
        if fixed is not None:
            return fixed, None
        if not player_learns(self.cfg):
            return None, None
        v = ctl.player_version(c)
        if v is None:
            raise _Wait("waiting for the player learner to publish version 0 (the pilot)")
        return self.ensure_version(v), v

    def ensure_builder(self, bv: int):
        if bv not in self.builders:
            root = self.wd / "builder-versions"
            meta, ens = _fetch_builder(self.store, bv, root)
            self.builders = {bv: _load_builder(ens, ctl.get(self.cfg, "builder.threads"))}
            for d in root.glob("v*"):
                if d.name != f"v{bv:04d}":
                    shutil.rmtree(d, ignore_errors=True)
            self.log(f"builder actor {self.machine}: building with builder version {bv}"
                     + ("" if meta.get("net") else " (untrained: random deck-shaped decks)"))
        return self.builders[bv]

    def next_seq(self) -> int:
        if self.seq is None:
            self.seq = max(ctl.next_seq(self.store, f"{SHARDS}{self.machine}/"),
                           max((int(p.name) + 1 for p in (self.wd / "spool").glob("[0-9]*")), default=1))
        else:
            self.seq += 1
        return self.seq

    def build_decks(self, builder, seed: int, tag: str) -> tuple[dict, dict, list[str], list[str]]:
        """({name: deck}, {name: pool id}, deck lines, pool ids) of one chunk: builder.pools_per_chunk pools drawn
        with the chunk's rng, each built into its kinds."""
        import numpy as np
        from draftzero.build import pools as P
        rng = np.random.default_rng(seed)
        pools = self.pools
        k = min(int(ctl.get(self.cfg, "builder.pools_per_chunk")), len(pools))
        pick = sorted(int(i) for i in rng.choice(len(pools), k, replace=False))
        kinds, share = str(ctl.get(self.cfg, "builder.kinds")), float(ctl.get(self.cfg, "builder.random_share"))
        decks, pool_of, lines = {}, {}, []
        for i in pick:
            pool = pools[i]
            for kind, d in builder.decks(P.pool_vector(pool), kinds, rng, share).items():
                name = f"{tag}.{i}.{kind}"
                decks[name], pool_of[name] = d, pool["pool_id"]
                lines.append(deck_line(name, d, pool=pool["pool_id"], kind=kind))
        return decks, pool_of, lines, [pools[i]["pool_id"] for i in pick]

    def play_decks(self, bv: int, pv: int | None, builder, spec, games: int, sims: int) -> bool:
        import numpy as np
        from draftzero.build.selfplay import pair_up
        seq = self.next_seq()
        seed = ctl.chunk_seed(int(ctl.get(self.cfg, "seed", 1)), f"builder-{self.machine}", seq)
        out = self.wd / "chunks" / f"{seq:06d}"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        t0 = time.time()
        decks, pool_of, lines, pool_ids = self.build_decks(builder, seed, f"b{bv:04d}-{self.machine}-{seq:06d}")
        build_s = time.time() - t0
        pairs = pair_up(list(decks), pool_of, games, np.random.default_rng([seed, 1]))
        (out / "decks.jsonl").write_text("".join(x + "\n" for x in lines))
        (out / "pairs.tsv").write_text("".join(f"{a}\t{b}\t{s}\n" for a, b, s in pairs))
        t1 = time.time()
        with ctl.heartbeating(self.store, lambda: self._heartbeat("playing"), self.HEARTBEAT_EVERY_S, self.log):
            res = self.gorge.play(spec, spec, pairs=0, seed=seed, out_dir=out / "play",
                                  workers=int(ctl.get(self.cfg, "builder.workers", 2)), deckfile=out / "decks.jsonl",
                                  pairs_file=out / "pairs.tsv", single=True, belief=pilot_belief(self.cfg),
                                  timeout_s=float(ctl.get(self.cfg, "builder.chunk_secs") or 3600))
        wall = time.time() - t1
        rows = [{**g, "line": g.get("pair")} for g in res.games]
        decided = sum(1 for g in rows if g.get("winner", -1) in (0, 1))
        results = collections.Counter(str(g.get("result")) for g in rows)
        self.last = {"version": bv, "seq": seq, "games": len(rows), "decided": decided, "wall_s": round(wall, 1),
                     "build_s": round(build_s, 1), "games_per_hour": round(len(rows) / max(wall, 1e-9) * 3600, 1)}
        mpath = out / "metrics.jsonl"
        mlog = ctl.metrics_log(mpath, self.cfg, self.ROLE, self.machine)
        mlog.log("throughput.games_per_hour", len(rows) / max(wall, 1e-9) * 3600, version=bv, step=seq,
                 n=len(rows), model=COMPONENT)
        mlog.log("data.builder_games", len(rows), version=bv, step=seq, decided=decided)
        mlog.log("throughput.builder_build_s", build_s, version=bv, step=seq, n=len(decks))
        sd = shard_dir(self.machine, seq)
        files: dict = {}
        if rows:
            files[f"{sd}/decks.jsonl"] = out / "decks.jsonl"
            files[f"{sd}/games.jsonl"] = "".join(json.dumps(g, sort_keys=True) + "\n" for g in rows).encode()
        files[ctl.metrics_path(self.ROLE, self.machine, seq)] = mpath
        self.games += len(rows)
        self.decided += decided
        self.chunks += 1
        self.failed += 0 if rows else 1
        hb_path, hb = self._heartbeat("playing")
        files[hb_path] = hb
        if rows:
            meta = {"actor": self.machine, "seq": seq, "seed": seed, "version": bv, "player_version": pv,
                    "games": len(rows), "decided": decided, "decks": len(decks), "pools": pool_ids,
                    "kinds": str(ctl.get(self.cfg, "builder.kinds")), "sims": int(sims),
                    "spec": spec.render() if hasattr(spec, "render") else str(spec), "time": ctl.now_iso(),
                    "wall_s": round(wall, 2), "build_s": round(build_s, 2), "results": dict(results),
                    "skipped": list(res.skipped or []), "run": ctl.get(self.cfg, "run")}
            files[f"{sd}/meta.json"] = json.dumps(meta, indent=1, sort_keys=True).encode()
        self._upload(files, spool_as=seq)
        self.state, self.hb_time = "playing", time.time()
        shutil.rmtree(out, ignore_errors=True)
        self.log(f"builder actor {self.machine}: shard {seq} (builder v{bv}"
                 + (f", player v{pv}" if pv is not None else "") + f"): {len(decks)} decks from {len(pool_ids)} "
                 f"pools in {build_s:.0f} s, {len(rows)} games ({decided} decided) in {wall:.0f} s"
                 + (f", {len(res.skipped)} skipped" if res.skipped else ""))
        return bool(rows)

    def _heartbeat(self, state: str) -> tuple[str, bytes]:
        return ctl.heartbeat(self.machine, self.ROLE, ctl.get(self.cfg, "run"), state=state,
                             version=max(self.builders, default=None), games=self.games,
                             decided=getattr(self, "decided", 0), chunks=self.chunks, failed_chunks=self.failed,
                             last=self.last, games_per_hour=self.last.get("games_per_hour"))


# ------------------------------------------------------------------------------------------------ learner

def default_fit(decks, games, *, init, cfg, log):
    """build.train.fit_games with the builder section's settings (torch threads: builder.threads)."""
    import torch
    from draftzero.build import train as T
    torch.set_num_threads(int(ctl.get(cfg, "builder.threads", 4)))
    lr = float(ctl.get(cfg, "builder.warm_lr" if init is not None else "builder.lr"))
    return T.fit_games(decks, games, int(ctl.get(cfg, "builder.members")), int(ctl.get(cfg, "builder.epochs")),
                       init=init, log=log, draws=bool(ctl.get(cfg, "builder.draws")),
                       castable=bool(ctl.get(cfg, "builder.castable")), lr=lr)


class BuilderLearner:
    """The builder learner (module docstring): one per run, holding builder/learner/lease.json."""

    ROLE = "builder-learner"
    LEASE_S = 1800.0
    STATUS_EVERY_S = 60.0
    FILE_TRIES = 5              # a shard whose meta is in but its decks or games aren't (a bucket batch isn't atomic)

    def __init__(self, store, cfg, machine: str, work_dir: Path, *, fit=None, log=print, poll_s: float = 30.0):
        self.store, self.base, self.cfg, self.log, self.poll_s = store, cfg, cfg, log, poll_s
        self.machine = ctl.check_name(machine)
        self.wd = Path(work_dir).resolve()
        self.wd.mkdir(parents=True, exist_ok=True)
        self._fit = fit or default_fit
        self.holder = f"{machine}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.lease_renewed = float("-inf")
        self.ingested: list[dict] = []
        self.tries: dict[str, int] = {}
        self.index: ShardIndex | None = None
        self.versions_made = 0
        self.last_status, self.state = 0.0, None

    def c(self, dotted: str, default=None):
        return ctl.get(self.cfg, dotted, default)

    # ---------------------------------------------------------------------------- lease
    def lease(self, now: float | None = None, *, renew_only: bool = False) -> None:
        now = time.time() if now is None else now
        if now - self.lease_renewed < self.LEASE_S / 3:
            return
        cur = self.store.read_json(LEASE)
        if cur and cur.get("holder") != self.holder:
            if renew_only or cur.get("expires", 0) > now:
                raise LeaseHeld(f"builder learner {cur['holder']} holds the lease "
                                f"(until {time.ctime(cur.get('expires', 0))})")
            self._forget()
        self.store.write_json(LEASE, {"holder": self.holder, "machine": self.machine, "expires": now + self.LEASE_S,
                                      "time": now}, f"builder lease: {self.holder}")
        self.lease_renewed = now

    def _hold(self, parent: int | None = None) -> None:
        """Before a write the lease guards: LeaseHeld unless this learner still holds it and no other learner
        published after `parent`."""
        self.lease_renewed = float("-inf")
        self.lease(renew_only=True)
        cur = self.latest() if parent is not None else None
        if cur is not None and int(cur["version"]) != parent:
            raise LeaseHeld(f"builder version {cur['version']} was published by another learner")

    def _forget(self) -> None:
        self.index, self.ingested, self.tries = None, [], {}
        self.lease_renewed = float("-inf")

    def release(self) -> None:
        cur = self.store.read_json(LEASE)
        if cur and cur.get("holder") == self.holder:
            self.store.write_json(LEASE, {**cur, "expires": time.time(), "released": True}, "builder lease released")

    # ---------------------------------------------------------------------------- versions
    def latest(self) -> dict | None:
        return self.store.read_json(ctl.latest_path(COMPONENT))

    def bootstrap(self) -> dict:
        """latest.json, publishing version 0 first on a fresh store: builder.init's ensemble, else untrained."""
        latest = self.latest()
        if latest is not None:
            return latest
        init = self.c("builder.init")
        meta = {"version": 0, "parent": None, "net": False, "sha256": {}, "shards": [], "games": 0, "decided": 0,
                "metrics": {}, "time": ctl.now_iso(), "source": init}
        files: dict = {}
        if init:
            from draftzero.engine.gorge import resolve_path
            src = resolve_path(init)
            if not src.exists():
                raise FileNotFoundError(f"builder.init {init}: no such file")
            d = self.wd / "versions" / "v0000"
            d.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, d / "ens.pt")
            _load_builder(d / "ens.pt")                 # refuse a file the deck network can't load
            files[ctl.version_file(0, "ens.pt", COMPONENT)] = d / "ens.pt"
            meta.update(net=True, sha256={"pt": ctl.sha256(d / "ens.pt")})
        latest = {"version": 0, "path": ctl.version_dir(0, COMPONENT)}
        files[ctl.version_file(0, "meta.json", COMPONENT)] = json.dumps(meta, indent=1, sort_keys=True).encode()
        files[ctl.latest_path(COMPONENT)] = json.dumps(latest, indent=1, sort_keys=True).encode()
        self._hold()
        self.store.put_many(files, "builder version 0")
        self.log(f"builder learner: published version 0 ({'from ' + str(init) if init else 'untrained'})")
        return latest

    # ---------------------------------------------------------------------------- ingest
    def load_state(self) -> None:
        self.ingested = list(self.store.read_json(INGESTED) or [])
        self.index = ShardIndex(self.store, {e["shard"]: {**e, "path": e["shard"]} for e in self.ingested})
        if self.ingested:
            self.log(f"builder learner: {len(self.ingested)} shards ingested earlier, "
                     f"{sum(e['decided'] for e in self.ingested)} decided games")

    def local(self, e: dict) -> Path:
        """The shard's local folder with its decks.jsonl and games.jsonl (FileNotFoundError if the store lacks one)."""
        d = self.wd / "shards" / f"{e['actor']}-{int(e['seq']):06d}"
        for name in ("decks.jsonl", "games.jsonl"):
            if not (d / name).exists() and self.store.get(f"{e['shard']}/{name}", d / name) is None:
                raise FileNotFoundError(f"{e['shard']}/{name}")
        return d

    def ingest(self) -> list[dict]:
        if self.index is None:
            self.load_state()
        waiting = [m for m in self.index.metas.values() if m["path"] in self.tries]
        new = []
        for m in waiting + self.index.refresh():
            e = {"shard": m["path"], "actor": m.get("actor"), "seq": int(m.get("seq", 0)), "version": m.get("version"),
                 "player_version": m.get("player_version"), "games": int(m.get("games", 0)),
                 "decided": int(m.get("decided", 0)), "time": m.get("time"), "used": None}
            try:
                self.local(e)
            except FileNotFoundError:
                n = self.tries[m["path"]] = self.tries.get(m["path"], 0) + 1
                if n < self.FILE_TRIES:
                    continue
                self.log(f"builder learner: {m['path']} lacks its decks or games after {n} tries; ingested empty")
                e.update(missing=True, games=0, decided=0)
            self.tries.pop(m["path"], None)
            self.ingested.append(e)
            new.append(e)
        if new:
            self.log(f"builder learner: ingested {len(new)} shards ({sum(e['decided'] for e in new)} decided games)")
        return new

    # ---------------------------------------------------------------------------- when and what to train
    def due(self, latest: dict) -> str | None:
        v = int(latest["version"])
        mx = self.c("budget.max_versions")
        if mx is not None and v >= int(mx):
            return None
        have = sum(e["decided"] for e in self.ingested if e.get("version") == v)
        want = int(self.c("builder.games_per_version"))
        return f"{have} decided games of builder version {v}" if have >= want else None

    def window(self, v: int) -> list[dict]:
        """The shards the next version trains on (oldest first): the games of versions v-size+1..v (versions), or
        the newest shards holding `size` decided games (games)."""
        w = ctl.as_dict(self.c("builder.window"))
        have = [e for e in self.ingested if e["decided"] > 0]
        key = lambda e: (e.get("version") or 0, str(e.get("time") or ""), e["seq"], e.get("actor") or "")  # noqa: E731
        if w.get("kind", "versions") == "versions":
            return sorted((e for e in have if v - int(w.get("size", 4)) < (e.get("version") or 0) <= v), key=key)
        out, n = [], 0
        for e in sorted(have, key=key, reverse=True):
            out.append(e)
            n += e["decided"]
            if n >= int(w.get("size", 1)):
                break
        return out[::-1]

    def train_and_publish(self, latest: dict, reason: str = "") -> dict:
        from draftzero.build import model as M
        from draftzero.build import selfplay as S
        parent = int(latest["version"])
        v = parent + 1
        win = self.window(parent)
        dirs = [self.local(e) for e in win]
        decks = S.load_decks([d / "decks.jsonl" for d in dirs])
        games = S.load_games([d / "games.jsonl" for d in dirs])
        init = None
        if self.c("builder.warm_start", True):
            _, ens_path = _fetch_builder(self.store, parent, self.wd / "versions")
            if ens_path is not None:
                init = M.Ensemble.load(ens_path)
        n_new = sum(e["decided"] for e in self.ingested if e.get("used") is None)
        self.log(f"builder learner: training version {v} ({reason}): {len(win)} shards, {len(games)} games, "
                 f"{len(decks)} decks; " + (f"warm start from v{parent}" if init is not None else "fresh members"))
        self.status("training", force=True, training=v)
        t0 = time.time()
        ens, rep = self._fit(decks, games, init=init, cfg=self.cfg, log=lambda s: self.log(f"builder learner: {s}"))
        train_s = time.time() - t0
        share = float(self.c("builder.land_support") or 0)
        if share > 0:
            ens.meta["lands"] = M.land_support(games, decks, share)
        ens.meta.update(version=v, parent=parent)
        out = self.wd / "train" / f"v{v:04d}"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        ens.save(out / "ens.pt")
        for e in self.ingested:
            if e.get("used") is None:
                e["used"] = v
        metrics = {k: rep.get(k) for k in ("games", "logloss", "base_logloss", "acc")}
        meta = {"version": v, "parent": parent, "net": True, "sha256": {"pt": ctl.sha256(out / "ens.pt")},
                "shards": [e["shard"] for e in win], "games": len(games),
                "decided": sum(e["decided"] for e in win), "new_decided": n_new, "decks": len(decks),
                "members": len(ens.members), "lands": ens.meta.get("lands"), "castable": bool(self.c("builder.castable")),
                "init": f"v{parent:04d}" if init is not None else None, "metrics": metrics, "time": ctl.now_iso(),
                "train_s": round(train_s, 1),
                "player_versions": sorted({e.get("player_version") for e in win if e.get("player_version") is not None})}
        mpath = out / "metrics.jsonl"
        mlog = ctl.metrics_log(mpath, self.cfg, self.ROLE, self.machine)
        for k, name in (("logloss", "learn.builder_logloss"), ("base_logloss", "learn.builder_base_logloss"),
                        ("acc", "learn.builder_acc")):
            if rep.get(k) is not None:
                mlog.log(name, float(rep[k]), version=v, n=rep.get("games"), model=COMPONENT)
        mlog.log("data.builder_window_games", len(games), version=v, model=COMPONENT)
        mlog.log("learn.builder_train_s", train_s, version=v, model=COMPONENT)
        files: dict = {ctl.version_file(v, "ens.pt", COMPONENT): out / "ens.pt"}
        if mpath.exists():
            files[ctl.metrics_path(self.ROLE, self.machine, v)] = mpath
        files[ctl.version_file(v, "meta.json", COMPONENT)] = json.dumps(meta, indent=1, sort_keys=True).encode()
        files[INGESTED] = json.dumps(self.ingested, indent=1, sort_keys=True).encode()
        files[ctl.latest_path(COMPONENT)] = json.dumps({"version": v, "path": ctl.version_dir(v, COMPONENT)},
                                                       indent=1).encode()
        self._hold(parent)
        self.store.put_many(files, f"builder version {v}")
        self.versions_made += 1
        mine = self.wd / "versions" / f"v{v:04d}"         # the next warm start, without downloading it again
        mine.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / "ens.pt", mine / "ens.pt")
        self.prune(win, keep=v)
        held = ", ".join(f"{k} {rep[k]:.4f}" for k in ("logloss", "base_logloss", "acc")
                         if isinstance(rep.get(k), (int, float)))
        self.log(f"builder learner: published version {v} in {train_s:.0f} s: "
                 + (f"held-out {held} over {rep.get('games')} games" if held else "no held-out games")
                 + f"; lands {ens.meta.get('lands')}")
        return meta

    def prune(self, win: list[dict], keep: int) -> None:
        """Delete local shards outside the window and old training and version folders (the store keeps all)."""
        inside = {f"{e['actor']}-{int(e['seq']):06d}" for e in win}
        for d in (self.wd / "shards").glob("*"):
            if d.name not in inside:
                shutil.rmtree(d, ignore_errors=True)
        for sub in ("train", "versions"):
            for d in (self.wd / sub).glob("v*"):
                if d.name != f"v{keep:04d}":
                    shutil.rmtree(d, ignore_errors=True)

    # ---------------------------------------------------------------------------- status and the loop
    def status(self, state: str, force: bool = False, done: bool = False, **extra) -> None:
        now = time.time()
        if not force and state == self.state and now - self.last_status < self.STATUS_EVERY_S:
            return
        self.state, self.last_status = state, now
        st = {"holder": self.holder, "machine": self.machine, "state": state, "done": done, "time": now,
              "iso": ctl.now_iso(now), "shards": len(self.ingested), "games": sum(e["games"] for e in self.ingested),
              "decided": sum(e["decided"] for e in self.ingested), "versions_made": self.versions_made, **extra}
        hb_path, hb = ctl.heartbeat(self.machine, self.ROLE, self.c("run"), state=state)
        self.store.put_many({LEARNER_STATUS: json.dumps(st, indent=1, sort_keys=True).encode(), hb_path: hb},
                            f"builder learner: {state}")

    def step(self) -> str:
        """One pass: "stop", "published" or "wait"; LeaseHeld when another builder learner holds the lease."""
        try:
            return self._step()
        except LeaseHeld:
            self._forget()
            raise

    def _step(self) -> str:
        self.lease()
        c = ctl.read(self.store)
        if c and c.get("stop"):
            self.status("stopped", force=True, done=True)
            return "stop"
        self.cfg = ctl.effective(self.base, self.store)
        latest = self.bootstrap()
        new = self.ingest()
        reason = self.due(latest)
        if reason:
            self.train_and_publish(latest, reason)
            self.status("waiting", force=True)
            return "published"
        if new:
            self._hold()
            self.store.write_json(INGESTED, self.ingested, f"builder: ingested {len(self.ingested)} shards")
        mx = self.c("budget.max_versions")
        self.status("done" if mx is not None and int(latest["version"]) >= int(mx) else "waiting", force=bool(new))
        return "wait"

    def run(self, max_versions: int | None = None) -> int:
        made = 0
        try:
            while True:
                state = self.step()
                if state == "stop":
                    return 0
                if state == "published":
                    made += 1
                    if max_versions is not None and made >= max_versions:
                        return 0
                    continue
                time.sleep(self.poll_s)
        finally:
            with contextlib.suppress(Exception):
                self.release()


# ------------------------------------------------------------------------------------------------ evaluator

class BuilderEvaluator(Evaluator):
    """A builder evaluator (module docstring). Evaluator's loop, idling and heartbeats, on control["builder_evals"]."""

    ROLE = "builder-evaluator"

    def __init__(self, store, cfg, machine: str, work_dir: Path, *, gorge=None, log=print, poll_s: float = 30.0,
                 make_pools=None):
        super().__init__(store, cfg, machine, work_dir, gorge=gorge, log=log, poll_s=poll_s)
        self.make_pools = make_pools
        self._pools: dict[tuple, list[dict]] = {}
        self.builders: dict[int, object] = {}

    def step(self) -> str:
        c = ctl.read(self.store)
        if c is None:
            return self._idle("waiting")
        if c.get("stop"):
            self._idle("stopped", force=True)
            return "stop"
        for job in c.get(EVALS) or []:
            if job["job"] in self.summarised:
                continue
            present = self._present(job)
            if len(present) == len(job["shards"]):
                self.summarise(job)
                continue
            for sh in job["shards"]:
                if sh.get("machine") == self.machine and sh["id"] not in present:
                    self.cfg = ctl.effective(self.base, self.store)
                    self.play_shard(job, sh)
                    if len(self._present(job)) == len(job["shards"]):
                        self.summarise(job)
                    return "played"
        return self._idle("waiting")

    def _heartbeat(self, state: str) -> tuple[str, bytes]:
        return ctl.heartbeat(self.machine, self.ROLE, ctl.get(self.cfg, "run"), state=state,
                             shards=self.shards_played)

    def eval_pools(self, job: dict) -> list[dict]:
        kw = {**pool_kw(self.cfg, held_out=True), "n": int(job["pools"]), "seed": int(job["seed"])}
        key = tuple(sorted(kw.items()))
        if key not in self._pools:
            from draftzero.build import pools
            self._pools[key] = pools.cached(self.gorge, self.wd / "pools", make=self.make_pools, **kw)
        return self._pools[key]

    def ensure_builder(self, bv: int):
        if bv not in self.builders:
            _, ens = _fetch_builder(self.store, bv, self.wd / "builder-versions")
            self.builders = {bv: _load_builder(ens, ctl.get(self.cfg, "builder.threads"))}
        return self.builders[bv]

    def pilot_net(self, job: dict) -> Path | None:
        if int(job["sims"]) <= 0:
            return None
        fixed = fixed_pilot(self.cfg)
        if fixed is not None:
            return fixed
        pv = job.get("player_version")
        if pv is None:
            return None
        _, files = ctl.fetch_version(self.store, int(pv), self.wd / "versions" / f"v{int(pv):04d}")
        return files.get("model.dzgw")

    def write_match(self, job: dict, sh: dict, out: Path) -> tuple[Path, Path]:
        """The shard's -deckfile (m<k>: the version's best deck from pool k; h<k>: the opponent's) and -pairs-file."""
        import numpy as np
        from draftzero.build import pools as P
        pools = self.eval_pools(job)
        builder = self.ensure_builder(int(job["version"]))
        first, n = int(sh["first_pool"]), int(sh["pools"])
        lines = []
        for k in range(first, first + n):
            pool = pools[k]
            if job["opponent"] != "best23" or not pool.get("best23"):
                raise ValueError(f"evaluation pool {pool.get('pool_id')} has no {job['opponent']} deck")
            mine = builder.best(P.pool_vector(pool), np.random.default_rng([int(job["seed"]), k]))
            lines.append(deck_line(f"{TAG}{k}", mine, pool=pool["pool_id"], kind="b"))
            lines.append(deck_line(f"{OPP}{k}", P.deck_vector(pool["best23"]["cards"]), pool=pool["pool_id"],
                                   kind=job["opponent"]))
        out.mkdir(parents=True, exist_ok=True)
        (out / "decks.jsonl").write_text("".join(x + "\n" for x in lines))
        (out / "pairs.tsv").write_text("".join(f"{a}\t{b}\t{s}\n" for a, b, s in
                                               match_lines(first, n, int(job["pairs"]), int(job["seed"]))))
        return out / "decks.jsonl", out / "pairs.tsv"

    def play_shard(self, job: dict, sh: dict) -> None:
        out = self.wd / "evals" / job["job"] / f"{sh['id']:04d}"
        shutil.rmtree(out, ignore_errors=True)
        deckfile, pairs_file = self.write_match(job, sh, out)
        spec = pilot_spec(self.cfg, int(job["sims"]), self.pilot_net(job))
        t0 = time.time()
        with ctl.heartbeating(self.store, lambda: self._heartbeat("playing"), self.HEARTBEAT_EVERY_S, self.log):
            res = self.gorge.play(spec, spec, pairs=0, seed=int(job["seed"]), out_dir=out / "play",
                                  workers=int(ctl.get(self.cfg, "builder.eval.workers", 2)), deckfile=deckfile,
                                  pairs_file=pairs_file, single=True, belief=pilot_belief(self.cfg),
                                  timeout_s=float(ctl.get(self.cfg, "builder.chunk_secs") or 3600))
        wall = time.time() - t0
        rows = [{**g, "job": job["job"], "shard": sh["id"], "line": g.get("pair")} for g in res.games]
        self.shards_played += 1
        hb_path, hb = self._heartbeat("playing")
        self.store.put_many({hb_path: hb, ctl.eval_games_path(job["job"], sh["id"]):
                             "".join(json.dumps(g, sort_keys=True) + "\n" for g in rows).encode()},
                            f"builder evaluator {self.machine}: {job['job']} shard {sh['id']}")
        self.state, self.hb_time = "playing", time.time()
        self.done.add((job["job"], sh["id"]))
        shutil.rmtree(out, ignore_errors=True)
        self.log(f"builder evaluator {self.machine}: {job['job']} shard {sh['id']}: {len(rows)} games in {wall:.0f} s"
                 + (f", skipped {list(res.skipped)}" if res.skipped else ""))

    def summarise(self, job: dict) -> dict | None:
        if self.store.exists(ctl.eval_summary_path(job["job"])):
            self.summarised.add(job["job"])
            return None
        games = []
        for sh in job["shards"]:
            b = self.store.read_bytes(ctl.eval_games_path(job["job"], sh["id"]))
            games += [json.loads(x) for x in (b or b"").decode().splitlines() if x.strip()]
        s = score_match(games)
        summary = {"job": job["job"], "component": COMPONENT, "version": job["version"], "opponent": job["opponent"],
                   "pools": job["pools"], "pairs": job["pairs"], "sims": job["sims"], "seed": job["seed"],
                   "player_version": job.get("player_version"), "shards": len(job["shards"]), "time": ctl.now_iso(),
                   "by": self.machine, **s}
        if self.mseq is None:
            self.mseq = ctl.next_seq(self.store, f"metrics/{self.ROLE}/{self.machine}/")
        mpath = self.wd / "metrics" / f"{self.mseq:06d}.jsonl"
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.unlink(missing_ok=True)
        mlog = ctl.metrics_log(mpath, self.cfg, self.ROLE, self.machine)
        if s.get("score") is not None:
            mlog.log("eval.score", float(s["score"]), version=job["version"], n=s.get("n_pairs"), lo=s.get("lo"),
                     hi=s.get("hi"), opponent=job["opponent"], job=job["job"], sims=job["sims"], model=COMPONENT,
                     n_games=s.get("n_games"), n_pairs=s.get("n_pairs"), player_version=job.get("player_version"))
        files: dict = {ctl.eval_summary_path(job["job"]): json.dumps(summary, indent=1, sort_keys=True).encode()}
        if mpath.exists():
            files = {ctl.metrics_path(self.ROLE, self.machine, self.mseq): mpath, **files}
        self.store.put_many(files, f"builder evaluator {self.machine}: {job['job']} scored")
        self.mseq += 1
        self.summarised.add(job["job"])
        self.log(f"builder evaluator {self.machine}: {job['job']}: score {s.get('score')} (se {s.get('se')}) over "
                 f"{s.get('n_pairs')} pool pairs, {s.get('n_games')} games")
        return summary
