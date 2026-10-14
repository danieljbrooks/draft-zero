"""The actor (contracts §4): many, on any machine. Each loop it

  1. reads control.json: stops on "stop", idles while "paused" or before the learner publishes version 0;
  2. in sync mode, idles once the store holds sync.games_per_version games of the current version (a chunk is
     cut to what is missing, so one actor plays exactly that many);
  3. fetches the current player version's .dzgw (checked against its SHA-256); version 0 without player.init
     has no network: the search runs on its own priors, without topk. A version control.json names but the
     store lacks (a relay host that has control.json before the version) is waited for, not an error;
  4. plays one chunk of self-play games, the same spec on both seats: Gorge.play(spec, spec, pairs=chunk_games,
     record=True, record_every, timeout=chunk_secs) on the train split, with exploration and :closedlist;
  5. uploads the shard (corpus.jsonl.gz, games.jsonl, meta.json last), its heartbeat and its metrics in ONE
     put_many (one Hub API call on a bucket). A failed upload is spooled to <work-dir>/spool and retried with the
     next one.

A chunk's seed is int(sha256(f"{run_seed}|{machine}|{seq}")) % 2**31, seq the actor's shard number (1, 2, ...,
continuing from the store after a restart). The live settings (sims, record_every, chunk_games, pcr, decided,
forced, ...) come from control["play"], so a plan change reaches actors without a restart.

    python -m draftzero.loop.cli actor --config C --store S --machine pod1 [--work-dir DIR]
"""
from __future__ import annotations

import collections
import json
import shutil
import time
from pathlib import Path

from draftzero.loop import control as ctl


def policy_spec(cfg, *, sims: int, explore: bool, gonet: Path | None, pcr: dict | None = None,
                decided: dict | None = None, forced: dict | None = None):
    """The az PolicySpec of this run's search: keys, cpuct and :closedlist from `search`; topk and gonet only with
    a network; playout-cap randomization when pcr["prob"] > 0, the decided-game search when decided["high"] > 0 and
    forced playouts when forced["k"] > 0 (the actor's settings: self-play only, so the evaluator passes none)."""
    from draftzero.engine.specs import PolicySpec
    kw = {}
    if pcr and float(pcr.get("prob") or 0) > 0:
        kw = {"pcr": float(pcr["prob"]), "cheap": int(pcr.get("cheap_sims") or 40),
              "pcrk": tuple(sorted((str(k), float(v)) for k, v in (pcr.get("by_kind") or {}).items()))}
    if decided and float(decided.get("high") or 0) > 0:
        w = decided.get("weight")
        kw.update(decided=float(decided["high"]), decidedn=int(decided.get("n") or 3),
                  decidedsims=int(decided.get("sims") or 40), decidedw=0.1 if w is None else float(w))
    if forced and float(forced.get("k") or 0) > 0:
        kw["forced"] = float(forced["k"])
    return PolicySpec(kind="az", sims=int(sims), cpuct=ctl.get(cfg, "search.cpuct"),
                      topk=ctl.get(cfg, "search.topk") if gonet else None, explore=bool(explore),
                      closedlist=bool(ctl.get(cfg, "search.closedlist", False)),
                      gonet=str(gonet) if gonet else None, keys=tuple(ctl.get(cfg, "search.keys") or ()), **kw)


def belief(cfg):
    """The :closedlist belief corpus: None (gorge's default, the game decks and pool) unless search.belief names
    other decks, another pool or another split."""
    b = ctl.as_dict(ctl.get(cfg, "search.belief"))
    if not b.get("decks") and not b.get("pool") and b.get("split", "train") == "train":
        return None
    from draftzero.engine.gorge import Belief, resolve_path     # relative paths as Gorge.from_config reads data.*
    return Belief(resolve_path(b.get("decks") or ctl.get(cfg, "data.decks")),
                  resolve_path(b.get("pool") or ctl.get(cfg, "data.pool")), split=b.get("split") or "train")


def idle_sleep(poll_s: float, idle: int) -> float:
    """Seconds to wait after `idle` idle passes in a row: poll_s, doubling up to 4 x poll_s (Hub calls add up when
    many machines wait together, e.g. for a sync-mode version)."""
    return poll_s * 2 ** min(idle, 2)


def check_playable(cfg) -> None:
    """ValueError for a config this loop cannot play, when a role starts (not once the run is up): an arch other
    than mlp, or search and actor settings that make no PolicySpec."""
    arch = ctl.get(cfg, "player.arch", "mlp")
    if arch != "mlp":
        raise ValueError(f"player.arch {arch}: only an mlp runs inside gorge (.dzgw); a {arch} needs a network "
                         "server (dzg serve, remote=), which the v0.2.0 loop doesn't start")
    play = ctl.as_dict(ctl.get(cfg, "actor"))
    policy_spec(cfg, sims=int(play.get("sims") or 1), explore=bool(play.get("explore")), gonet=Path("check.dzgw"),
                pcr=ctl.as_dict(play.get("pcr")), decided=ctl.as_dict(play.get("decided")),
                forced=ctl.as_dict(play.get("forced")))


class Actor:
    HEARTBEAT_EVERY_S = 300.0        # while idle or playing; a played chunk always carries one

    def __init__(self, store, cfg, machine: str, work_dir: Path, *, gorge=None, log=print, poll_s: float = 30.0):
        self.store, self.cfg, self.log, self.poll_s = store, cfg, log, poll_s
        self.machine = ctl.check_name(machine)
        check_playable(cfg)
        self.wd = Path(work_dir).resolve()
        if ":" in str(self.wd):
            raise ValueError(f"work dir {self.wd}: gorge policy strings can't hold ':' (the .dzgw path goes in one)")
        self.wd.mkdir(parents=True, exist_ok=True)
        self._gorge = gorge
        self.seq: int | None = None
        self.index = ctl.ShardIndex(store)
        self.versions: dict[int, Path | None] = {}
        self.games = self.records = self.chunks = self.failed = 0
        self.last = {}                 # the last chunk's numbers, for the heartbeat
        self.state, self.why, self.hb_time = None, None, 0.0

    @property
    def gorge(self):
        if self._gorge is None:
            self._gorge = ctl.make_gorge(self.cfg)
        return self._gorge

    # ---------------------------------------------------------------------------- the loop
    def run(self, max_chunks: int | None = None) -> int:
        played = idle = 0
        while True:
            state = self.step()
            if state == "stop":
                return 0
            if state == "played":
                played, idle = played + 1, 0
                if max_chunks is not None and played >= max_chunks:
                    return 0
            elif state == "failed":
                time.sleep(min(self.poll_s, 60.0))
            else:
                time.sleep(idle_sleep(self.poll_s, idle))
                idle += 1

    def step(self) -> str:
        """One pass: "stop", "wait" (no version, or sync mode has its games), "paused", "played" or "failed"."""
        c = ctl.read(self.store)
        if c is None:
            return self._idle("waiting", "no control.json yet (is the controller running?)")
        if c.get("stop"):
            self._idle("stopped", "the run is stopping", force=True)
            return "stop"
        if c.get("paused"):
            return self._idle("paused", "the run is paused", ret="paused")
        if not ctl.get(self.cfg, "player.learn", True):
            return self._idle("waiting", "player.learn is false: this run trains no player (start builder actors)")
        v = ctl.player_version(c)
        if v is None:
            return self._idle("waiting", "waiting for the learner to publish version 0")
        play = {**ctl.as_dict(ctl.get(self.cfg, "actor")), **(c.get("play") or {})}
        n = int(play["chunk_games"])
        if play.get("games_per_version"):
            self.index.refresh()
            left = int(play["games_per_version"]) - self.index.games(v)
            if left <= 0:
                return self._idle("waiting", f"sync: version {v} has its {play['games_per_version']} games")
            n = min(n, left)
        try:
            gonet = self.ensure_version(v)
        except FileNotFoundError as e:              # e.g. a relay copied control.json before the version itself
            return self._idle("waiting", f"version {v} is not in this store yet ({e}); retrying at each poll")
        spec = policy_spec(self.cfg, sims=play["sims"], explore=play.get("explore", True), gonet=gonet,
                           pcr=play.get("pcr"), decided=ctl.as_dict(play.get("decided")),
                           forced=ctl.as_dict(play.get("forced")))
        return "played" if self.play_chunk(v, spec, n, play) else "failed"

    def _idle(self, state: str, why: str, force: bool = False, ret: str = "wait") -> str:
        """Heartbeat (on a change of state, else every HEARTBEAT_EVERY_S); `why` is logged once while it holds."""
        now = time.time()
        if (state, why) != (self.state, self.why):
            self.log(f"actor {self.machine}: {why}")
        if force or state != self.state or now - self.hb_time >= self.HEARTBEAT_EVERY_S:
            path, hb = self._heartbeat(state)
            self._upload({path: hb})
        self.state, self.why = state, why
        return ret

    # ---------------------------------------------------------------------------- versions
    def ensure_version(self, v: int) -> Path | None:
        """The local .dzgw of version v (None: no network), fetched once; older local versions are deleted."""
        if v in self.versions:
            return self.versions[v]
        root = self.wd / "versions"
        meta, files = ctl.fetch_version(self.store, v, root / f"v{v:04d}")
        self.versions = {v: files.get("model.dzgw")}
        for d in root.glob("v*"):
            if d.name != f"v{v:04d}":
                shutil.rmtree(d, ignore_errors=True)
        self.log(f"actor {self.machine}: playing version {v}" + ("" if meta.get("net") else " (no network)"))
        return self.versions[v]

    # ---------------------------------------------------------------------------- a chunk
    def next_seq(self) -> int:
        if self.seq is None:
            self.seq = max(ctl.next_seq(self.store, f"shards/{self.machine}/"),
                           max((int(p.name) + 1 for p in (self.wd / "spool").glob("[0-9]*")), default=1))
        else:
            self.seq += 1
        return self.seq

    def play_chunk(self, version: int, spec, games: int, play: dict) -> bool:
        seq = self.next_seq()
        seed = ctl.chunk_seed(int(ctl.get(self.cfg, "seed", 1)), self.machine, seq)
        out = self.wd / "chunks" / f"{seq:06d}"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        t0 = time.time()
        with ctl.heartbeating(self.store, lambda: self._heartbeat("playing"), self.HEARTBEAT_EVERY_S, self.log):
            res = self.gorge.play(spec, spec, pairs=games, seed=seed, out_dir=out,
                                  split=ctl.get(self.cfg, "data.train_split", "train"),
                                  workers=int(play.get("workers", 2)), record=True,
                                  record_every=int(play.get("record_every", 1)), belief=belief(self.cfg),
                                  timeout_s=float(play["chunk_secs"]) if play.get("chunk_secs") else None)
        wall = time.time() - t0
        rows = list(res.games)
        records = sum(int(g.get("visits") or 0) for g in rows)
        results = collections.Counter(str(g.get("result")) for g in rows)
        summary = res.summary or {}
        self.last = {"version": version, "seq": seq, "games": len(rows), "records": records, "wall_s": round(wall, 1),
                     "games_per_hour": round(len(rows) / max(wall, 1e-9) * 3600, 1)}
        mlog_path = out / "metrics.jsonl"
        mlog = ctl.metrics_log(mlog_path, self.cfg, "actor", self.machine)
        mlog.log("throughput.games_per_hour", len(rows) / max(wall, 1e-9) * 3600, version=version, step=seq,
                 n=len(rows))
        mlog.log("throughput.positions_per_hour", records / max(wall, 1e-9) * 3600, version=version, step=seq,
                 n=records)
        if summary.get("az_ms_per_searched") is not None:
            mlog.log("throughput.ms_per_searched", float(summary["az_ms_per_searched"]), version=version, step=seq,
                     n=summary.get("az_decisions_searched"))
        mlog.log("data.games", len(rows), version=version, step=seq)
        mlog.log("data.records", records, version=version, step=seq)
        if getattr(spec, "decided", None) and summary.get("visit_records_weighted") is not None:
            # the decided-game searches' records, at weight decidedw (weight.npy in the learner's packs)
            mlog.log("data.records_weighted", int(summary["visit_records_weighted"]), version=version, step=seq,
                     n=records)
        if getattr(spec, "forced", None) and summary.get("az_pruned_visit_share") is not None:
            mlog.log("search.pruned_visit_share", float(summary["az_pruned_visit_share"]), version=version,
                     step=seq, n=summary.get("az_forced_searched"))
        files: dict[str, Path | bytes] = {}
        if rows:
            sd = ctl.shard_dir(self.machine, seq)
            if records and res.corpus_path and Path(res.corpus_path).exists():
                files[f"{sd}/corpus.jsonl.gz"] = Path(res.corpus_path)
            files[f"{sd}/games.jsonl"] = "".join(json.dumps(g, sort_keys=True) + "\n" for g in rows).encode()
        if mlog_path.exists():
            files[ctl.metrics_path("actor", self.machine, seq)] = mlog_path
        self.games += len(rows)
        self.records += records
        self.chunks += 1
        self.failed += 0 if rows else 1
        hb_path, hb = self._heartbeat("playing")
        files[hb_path] = hb
        if rows:
            meta = {"actor": self.machine, "seq": seq, "seed": seed, "version": version, "games": len(rows),
                    "records": records, "sims": int(play["sims"]), "spec": spec.render(), "time": ctl.now_iso(),
                    "wall_s": round(wall, 2), "results": dict(results), "skipped": list(res.skipped or []),
                    "record_every": int(play.get("record_every", 1)), "run": ctl.get(self.cfg, "run"),
                    "pcr": (play.get("pcr") or {}).get("prob") or 0.0,
                    "decided": ctl.as_dict(play.get("decided")).get("high") or 0.0,
                    "forced": ctl.as_dict(play.get("forced")).get("k") or 0.0}
            files[f"{ctl.shard_dir(self.machine, seq)}/meta.json"] = json.dumps(meta, indent=1, sort_keys=True).encode()
        self._upload(files, spool_as=seq)
        self.state, self.hb_time = "playing", time.time()
        shutil.rmtree(out, ignore_errors=True)
        if rows:
            self.log(f"actor {self.machine}: shard {seq} (v{version}): {len(rows)} games, {records} records, "
                     f"{wall:.0f} s" + (f", {len(res.skipped)} pairs skipped" if res.skipped else ""))
        else:
            self.log(f"actor {self.machine}: chunk {seq} (v{version}) produced no complete game "
                     f"(skipped pairs {list(res.skipped or [])}); nothing uploaded")
        return bool(rows)

    def _heartbeat(self, state: str) -> tuple[str, bytes]:
        return ctl.heartbeat(self.machine, "actor", ctl.get(self.cfg, "run"), state=state,
                             version=max(self.versions, default=None), games=self.games, records=self.records,
                             chunks=self.chunks, failed_chunks=self.failed, last=self.last,
                             games_per_hour=self.last.get("games_per_hour"))

    # ---------------------------------------------------------------------------- upload, with a spool
    def _upload(self, files: dict, spool_as: int | None = None) -> bool:
        """One put_many of the spooled files (oldest first) and `files`; on failure spool `files` (not heartbeats)."""
        spooled = sorted((self.wd / "spool").glob("[0-9]*"))
        batch: dict = {}
        for d in spooled:
            for path, name in json.loads((d / "files.json").read_text()).items():
                batch[path] = d / name
        batch.update(files)
        try:
            self.store.put_many(batch, f"actor {self.machine}")
        except Exception as e:                          # noqa: BLE001 - keep playing; retry with the next upload
            self.log(f"actor {self.machine}: upload failed ({type(e).__name__}: {e}); spooled for the next one")
            if spool_as is not None:
                self._spool(spool_as, files)
            return False
        for d in spooled:
            shutil.rmtree(d, ignore_errors=True)
        self.hb_time = time.time()
        return True

    def _spool(self, seq: int, files: dict) -> None:
        d = self.wd / "spool" / f"{seq:06d}"
        d.mkdir(parents=True, exist_ok=True)
        index = {}
        for i, (path, src) in enumerate(files.items()):
            if path.startswith("heartbeats/"):
                continue
            name = f"{i:03d}-{Path(path).name}"
            if isinstance(src, (bytes, bytearray)):
                (d / name).write_bytes(src)
            else:
                shutil.copyfile(src, d / name)
            index[path] = name
        (d / "files.json").write_text(json.dumps(index))
