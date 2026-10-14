"""The evaluator (contracts §4): a few, on any machine. The controller opens a job for every version v with
v % eval.every_versions == 0 and every opponent in eval.opponents, split into shards of SHARD_PAIRS deck pairs,
and assigns the shards to live evaluators (control["evals"]). Each loop an evaluator

  1. reads control.json (stops on "stop") and picks the first shard assigned to it whose games aren't in the
     store yet;
  2. plays it: Gorge.play(version v's spec, the opponent's spec, pairs, first_pair, seed=eval.seed,
     split=data.eval_split, record=False), no exploration, both at eval.sims, :closedlist as in self-play;
  3. uploads evals/<job>/<shard>.games.jsonl with its heartbeat in one put_many;
  4. once every shard of the job is in, scores the pairs (draftzero.eval.paired.score_pairs, side A = version v),
     runs the sequential test against 50% (informational: jobs always play every pair), and writes
     evals/<job>/summary.json and the eval.score metric. The controller then closes the job.

Opponents: prev (version v-1), v<N>, sbtactical, bot, random, nonet (the same search without a network, i.e.
version 0's policy at eval.sims), file:/path/model.dzgw, spec:<raw gorge policy string>. Every version meets the
same deals: the seed is eval.seed and pairs are deck pairs of the eval split.

    python -m draftzero.loop.cli evaluator --config C --store S [--machine M] [--work-dir DIR]
"""
from __future__ import annotations

import json
import math
import re
import shutil
import time
from pathlib import Path

from draftzero.loop import control as ctl
from draftzero.loop.actor import check_playable, idle_sleep, policy_spec

SHARD_PAIRS = 10
BUILTIN = ("sbtactical", "bot", "random")


def job_id(version: int, opponent: str) -> str:
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", opponent)
    label = re.sub(r"-{2,}", "-", label).strip("-")
    return f"v{int(version):04d}-vs-{label[:48]}"


def opponent_valid(version: int, opponent: str) -> bool:
    """Whether the opponent means something for this version (v0 has no prev, and is itself nonet)."""
    if opponent == "prev":
        return version >= 1
    if opponent == "nonet":
        return version >= 1
    m = re.fullmatch(r"v(\d+)", opponent)
    if m:
        return int(m.group(1)) != version and int(m.group(1)) <= version
    return opponent in BUILTIN or opponent.startswith(("file:", "spec:"))


def make_job(version: int, opponent: str, cfg, now: float | None = None) -> dict:
    pairs = int(ctl.get(cfg, "eval.pairs"))
    shards = [{"id": i, "first_pair": lo, "pairs": min(SHARD_PAIRS, pairs - lo), "machine": None, "assigned": None,
               "done": False} for i, lo in enumerate(range(0, pairs, SHARD_PAIRS))]
    return {"job": job_id(version, opponent), "version": int(version), "opponent": opponent, "pairs": pairs,
            "sims": int(ctl.get(cfg, "eval.sims")), "seed": int(ctl.get(cfg, "eval.seed")),
            "created": ctl.now_iso(now), "shards": shards}


def score(games: list[dict]) -> dict:
    """The paired score of side A (the version) and the sequential test of H1 55% vs H0 50% on its games."""
    from draftzero.eval import paired
    s = dict(paired.score_pairs(games, hero_side="A"))
    s["sprt"] = paired.sprt(s.get("wins", 0), s.get("draws", 0) + s.get("stalls", 0), s.get("losses", 0))
    se = s.get("se")
    if s.get("score") is not None and se is not None and math.isfinite(se):
        s["lo"], s["hi"] = max(0.0, s["score"] - 1.96 * se), min(1.0, s["score"] + 1.96 * se)
    return s


class Evaluator:
    HEARTBEAT_EVERY_S = 300.0

    def __init__(self, store, cfg, machine: str, work_dir: Path, *, gorge=None, log=print, poll_s: float = 30.0):
        self.store, self.base, self.cfg, self.log, self.poll_s = store, cfg, cfg, log, poll_s
        self.machine = ctl.check_name(machine)
        check_playable(cfg)
        self.wd = Path(work_dir).resolve()
        if ":" in str(self.wd):
            raise ValueError(f"work dir {self.wd}: gorge policy strings can't hold ':'")
        self.wd.mkdir(parents=True, exist_ok=True)
        self._gorge = gorge
        self.done: set[tuple[str, int]] = set()      # shards whose games are in the store
        self.summarised: set[str] = set()
        self.shards_played = 0
        self.mseq: int | None = None
        self.state, self.hb_time = None, 0.0

    @property
    def gorge(self):
        if self._gorge is None:
            self._gorge = ctl.make_gorge(self.cfg)
        return self._gorge

    def run(self, max_shards: int | None = None) -> int:
        played = idle = 0
        while True:
            state = self.step()
            if state == "stop":
                return 0
            if state == "played":
                played, idle = played + 1, 0
                if max_shards is not None and played >= max_shards:
                    return 0
            else:
                time.sleep(idle_sleep(self.poll_s, idle))
                idle += 1

    def step(self) -> str:
        """One pass: "stop", "wait" (nothing assigned) or "played"."""
        c = ctl.read(self.store)
        if c is None:
            return self._idle("waiting")
        if c.get("stop"):
            self._idle("stopped", force=True)
            return "stop"
        for job in c.get("evals") or []:
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

    def _present(self, job: dict) -> set[int]:
        known = {i for j, i in self.done if j == job["job"]}
        if len(known) < len(job["shards"]):
            for p in self.store.list(f"evals/{job['job']}/"):
                name = p.rsplit("/", 1)[-1]
                if name.endswith(".games.jsonl") and name.split(".")[0].isdigit():
                    self.done.add((job["job"], int(name.split(".")[0])))
            known = {i for j, i in self.done if j == job["job"]}
        return known

    def _idle(self, state: str, force: bool = False) -> str:
        now = time.time()
        if force or state != self.state or now - self.hb_time >= self.HEARTBEAT_EVERY_S:
            path, hb = self._heartbeat(state)
            self.store.put_many({path: hb}, f"evaluator {self.machine}")
            self.hb_time = now
        self.state = state
        return "wait"

    def _heartbeat(self, state: str) -> tuple[str, bytes]:
        return ctl.heartbeat(self.machine, "evaluator", ctl.get(self.cfg, "run"), state=state,
                             shards=self.shards_played)

    # ---------------------------------------------------------------------------- specs
    def spec(self, version: int, sims: int):
        """Version `version`'s evaluation spec: no exploration, no playout caps; its network if it has one."""
        root = self.wd / "versions" / f"v{version:04d}"
        _, files = ctl.fetch_version(self.store, version, root)
        return policy_spec(self.cfg, sims=sims, explore=False, gonet=files.get("model.dzgw"))

    def opponent_spec(self, version: int, opponent: str, sims: int):
        if opponent == "prev":
            return self.spec(version - 1, sims)
        if re.fullmatch(r"v\d+", opponent):
            return self.spec(int(opponent[1:]), sims)
        if opponent == "nonet":
            return policy_spec(self.cfg, sims=sims, explore=False, gonet=None)
        if opponent in BUILTIN:
            return opponent
        if opponent.startswith("file:"):
            p = Path(opponent[len("file:"):]).resolve()
            if not p.exists():
                raise FileNotFoundError(f"eval opponent {opponent}: no such file")
            return policy_spec(self.cfg, sims=sims, explore=False, gonet=p)
        if opponent.startswith("spec:"):
            return opponent[len("spec:"):]
        raise ValueError(f"eval opponent {opponent!r}: want prev, v<N>, nonet, sbtactical, bot, random, file:<path> "
                         "or spec:<policy>")

    # ---------------------------------------------------------------------------- a shard
    def play_shard(self, job: dict, sh: dict) -> None:
        v, opp, sims = job["version"], job["opponent"], int(job["sims"])
        a, b = self.spec(v, sims), self.opponent_spec(v, opp, sims)
        out = self.wd / "evals" / job["job"] / f"{sh['id']:04d}"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        from draftzero.loop.actor import belief
        t0 = time.time()
        res = self.gorge.play(a, b, pairs=int(sh["pairs"]), seed=int(job["seed"]), out_dir=out,
                              split=ctl.get(self.cfg, "data.eval_split", "eval"), first_pair=int(sh["first_pair"]),
                              workers=int(ctl.get(self.cfg, "eval.workers", 2)), record=False, belief=belief(self.cfg),
                              timeout_s=float(ctl.get(self.cfg, "actor.chunk_secs") or 3600))
        wall = time.time() - t0
        rows = [{**g, "job": job["job"], "shard": sh["id"]} for g in res.games]
        self.shards_played += 1
        hb_path, hb = self._heartbeat("playing")
        self.store.put_many({hb_path: hb, ctl.eval_games_path(job["job"], sh["id"]):
                             "".join(json.dumps(g, sort_keys=True) + "\n" for g in rows).encode()},
                            f"evaluator {self.machine}: {job['job']} shard {sh['id']}")
        self.state, self.hb_time = "playing", time.time()
        self.done.add((job["job"], sh["id"]))
        shutil.rmtree(out, ignore_errors=True)
        self.log(f"evaluator {self.machine}: {job['job']} shard {sh['id']}: {len(rows)} games in {wall:.0f} s"
                 + (f", pairs skipped {list(res.skipped)}" if res.skipped else ""))

    def summarise(self, job: dict) -> dict | None:
        """Score a job whose shards are all in; write summary.json and eval.score (once; a race writes the same)."""
        if self.store.exists(ctl.eval_summary_path(job["job"])):
            self.summarised.add(job["job"])
            return None
        games = []
        for sh in job["shards"]:
            b = self.store.read_bytes(ctl.eval_games_path(job["job"], sh["id"]))
            games += [json.loads(x) for x in (b or b"").decode().splitlines() if x.strip()]
        s = score(games)
        summary = {"job": job["job"], "version": job["version"], "opponent": job["opponent"], "pairs": job["pairs"],
                   "sims": job["sims"], "seed": job["seed"], "split": ctl.get(self.cfg, "data.eval_split", "eval"),
                   "shards": len(job["shards"]), "time": ctl.now_iso(), "by": self.machine, **s}
        if self.mseq is None:
            self.mseq = ctl.next_seq(self.store, f"metrics/evaluator/{self.machine}/")
        mpath = self.wd / "metrics" / f"{self.mseq:06d}.jsonl"
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.unlink(missing_ok=True)
        mlog = ctl.metrics_log(mpath, self.cfg, "evaluator", self.machine)
        if s.get("score") is not None:
            mlog.log("eval.score", float(s["score"]), version=job["version"], n=s.get("n_pairs"), lo=s.get("lo"),
                     hi=s.get("hi"), opponent=job["opponent"], job=job["job"], sims=job["sims"])
        files: dict = {ctl.eval_summary_path(job["job"]): json.dumps(summary, indent=1, sort_keys=True).encode()}
        if mpath.exists():
            files = {ctl.metrics_path("evaluator", self.machine, self.mseq): mpath, **files}
        self.store.put_many(files, f"evaluator {self.machine}: {job['job']} scored")
        self.mseq += 1
        self.summarised.add(job["job"])
        self.log(f"evaluator {self.machine}: {job['job']}: score {s.get('score')} (se {s.get('se')}) over "
                 f"{s.get('n_pairs')} pairs; SPRT {s.get('sprt') or 'undecided'}")
        return summary
