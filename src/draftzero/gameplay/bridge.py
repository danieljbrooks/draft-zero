"""
bridge.py — Python client for the Java XMage bridge (java/mzbridge).

The bridge turns a StateSpec v1 (see `statespec`) into a live XMage game and answers questions
about the first decision the chosen player faces there:

  build    build the state; return a StateSpec-shaped dump of the built game (for round-trip
           checks) and the decision {player, type, text, legal: [{label, idx}]}
  encode   the same decision encoded by MageZero's StateEncoder (sorted feature ids), no search
  coach    rate every legal option with MageZero's MCTS over K determinizations of the hidden
           cards (offline heuristic or a MageZero inference server), plus the human action's rank
           and regret
  ping     liveness

Each `Bridge` owns one long-lived JVM (`java/mzbridge/run.sh <name>`, ~1 s to open the card
database) that reads JSON lines on stdin and answers one JSON line per request on stdout. Every
worker runs in data/mzbridge/runtime/<name>/ with its own copy of xmage/db, because H2 cannot be
shared between JVMs. `BridgePool(n)` runs n workers for parallel requests. The jar is rebuilt
automatically when it is missing or older than the Java sources. The protocol, options and
gotchas are documented in java/mzbridge/README.md.

    python -m draftzero.gameplay.bridge ping
    python -m draftzero.gameplay.bridge build java/mzbridge/specs/main_phase.json
    python -m draftzero.gameplay.bridge encode java/mzbridge/specs/block.json --decision-player A
    python -m draftzero.gameplay.bridge coach java/mzbridge/specs/main_phase.json -K 4 --budget 300 \\
        --human-action "Play Plains"
    python -m draftzero.gameplay.bridge encode eot_rollover --turn-start   # A's decision in its next turn
    python -m draftzero.gameplay.bridge build-jar
    python -m draftzero.gameplay.bridge bench              # timings and memory on this machine

    from draftzero.gameplay.bridge import Bridge
    with Bridge("w0") as b:
        r = b.request("coach", spec, determinizations=4, budget=300, seed=1)
        r["aggregate"][0]["label"], r["human"]

The decision player defaults to the spec's priority / active player. A 17lands `eot_rollover`
spec is the OPPONENT's end step, so pass `**turn_start_options(spec)` (decision player A, window
from A's next main phase) and check `decision.where` (turn, step, passedBefore) before using a
decision as the label of a given human action: the bridge moves on past trivial windows.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import weakref
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[3]
BRIDGE_DIR = REPO / "java" / "mzbridge"
JAR = BRIDGE_DIR / "build" / "mzbridge.jar"
RUN_SH = BRIDGE_DIR / "run.sh"
BUILD_SH = BRIDGE_DIR / "build.sh"
SPECS_DIR = BRIDGE_DIR / "specs"
RUNTIME_ROOT = REPO / "data" / "mzbridge" / "runtime"
XMAGE_DIR = REPO / "xmage"
VOCAB = REPO / "assets" / "vocab" / "FDN_SPG.tsv"

_build_lock = threading.Lock()


class BridgeError(RuntimeError):
    """A request the worker rejected (bad spec, engine error) or a worker that died."""

    def __init__(self, message: str, error: dict | None = None, response: dict | None = None):
        super().__init__(message)
        self.error = error or {}
        self.response = response or {}

    @property
    def problems(self) -> list[str]:
        return list(self.error.get("problems") or [])


class BridgeUnavailable(BridgeError):
    """Java, the XMage build or the vocabulary is missing: the bridge cannot run here."""


# ------------------------------------------------------------------------------------------------
# environment and build

def environment_problems(xmage_dir: Path | None = None, vocab: Path | None = None) -> list[str]:
    """Everything that stops a worker from starting on this machine. Empty = ready."""
    xmage_dir = Path(xmage_dir or os.environ.get("MZ_XMAGE_DIR") or XMAGE_DIR)
    vocab = Path(vocab or os.environ.get("MZ_ACTION_VOCAB") or VOCAB)
    out = []
    java = os.environ.get("JAVA", "java")
    if not shutil.which(java):
        out.append(f"'{java}' is not on PATH: install a JDK 17+ (the bridge was developed on Temurin 26)")
    elif not JAR.exists() and not shutil.which(os.environ.get("JAVAC", "javac")):
        out.append("javac is not on PATH and java/mzbridge/build/mzbridge.jar has not been built")
    if not (xmage_dir / "lib").is_dir():
        out.append(f"no XMage build at {xmage_dir} (expected lib/*.jar; set MZ_XMAGE_DIR)")
    elif not (xmage_dir / "db" / "cards.h2.mv.db").is_file():
        out.append(f"no card database at {xmage_dir / 'db' / 'cards.h2.mv.db'}")
    if not vocab.is_file():
        out.append(f"no action vocabulary at {vocab} (set MZ_ACTION_VOCAB)")
    return out


def jar_is_stale() -> bool:
    """True when the jar is missing or older than any Java source or the build script."""
    if not JAR.exists():
        return True
    built = JAR.stat().st_mtime
    sources = list((BRIDGE_DIR / "src").rglob("*.java")) + [BUILD_SH]
    return any(p.stat().st_mtime > built for p in sources)


def build_jar(force: bool = False, xmage_dir: Path | None = None) -> Path:
    """Compile java/mzbridge (about 2 s). No-op when the jar is up to date unless force."""
    with _build_lock:
        if not force and not jar_is_stale():
            return JAR
        env = dict(os.environ)
        if xmage_dir:
            env["MZ_XMAGE_DIR"] = str(xmage_dir)
        r = subprocess.run(["sh", str(BUILD_SH)], capture_output=True, text=True, env=env)
        if r.returncode != 0 or not JAR.exists():
            raise BridgeUnavailable(f"java/mzbridge/build.sh failed (exit {r.returncode}):\n{r.stderr[-3000:]}")
        return JAR


# ------------------------------------------------------------------------------------------------
# one worker

def spec_dict(spec) -> dict:
    """A StateSpec, a dict, or a path to a JSON file -> the JSON object the worker reads."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        return spec
    if isinstance(spec, (str, Path)):
        with open(spec) as f:
            return json.load(f)
    if hasattr(spec, "to_dict"):
        return spec.to_dict()
    raise TypeError(f"spec must be a StateSpec, dict or path, not {type(spec).__name__}")


class Bridge:
    """One bridge worker (a JVM). Not thread-safe: use one Bridge per thread, or a BridgePool."""

    def __init__(self, name: str = "w0", *, heap: str = "3g", log_level: str = "warn", auto_build: bool = True,
                 timeout: float = 900.0, start: bool = True, xmage_dir: Path | None = None,
                 vocab: Path | None = None, runtime_root: Path | None = None, log_file: bool = True,
                 java_opts: str = ""):
        self.name = name
        self.java_opts = java_opts
        self.heap = heap
        self.log_level = log_level
        self.auto_build = auto_build
        self.timeout = timeout
        self.xmage_dir = Path(xmage_dir) if xmage_dir else None
        self.vocab = Path(vocab) if vocab else None
        self.runtime_root = Path(runtime_root) if runtime_root else RUNTIME_ROOT
        self.log_file = log_file
        self.proc: subprocess.Popen | None = None
        self.ready: dict = {}
        self.startup_s: float | None = None
        self._ids = itertools.count(1)
        self._lines: queue.Queue = queue.Queue()
        self._stderr: deque = deque(maxlen=200)
        self._finalizer = None
        if start:
            self.start()

    # -- lifecycle ---------------------------------------------------------------------------------

    def start(self) -> "Bridge":
        if self.proc and self.proc.poll() is None:
            return self
        problems = environment_problems(self.xmage_dir, self.vocab)
        if problems:
            raise BridgeUnavailable("mzbridge cannot start: " + "; ".join(problems))
        if self.auto_build:
            build_jar(xmage_dir=self.xmage_dir)
        elif not JAR.exists():
            raise BridgeUnavailable(f"{JAR} is missing; run `python -m draftzero.gameplay.bridge build-jar`")
        env = dict(os.environ)
        env["MZB_HEAP"] = self.heap
        env["MZB_LOG_LEVEL"] = self.log_level
        env["MZB_RUNTIME_ROOT"] = str(self.runtime_root)
        if self.java_opts:
            env["MZB_JAVA_OPTS"] = self.java_opts
        if self.xmage_dir:
            env["MZ_XMAGE_DIR"] = str(self.xmage_dir)
        if self.vocab:
            env["MZ_ACTION_VOCAB"] = str(self.vocab)
        t0 = time.monotonic()
        self._lines = queue.Queue()      # fresh pipes: a restarted worker must not see the old sentinel
        self._stderr.clear()
        self.proc = subprocess.Popen(["sh", str(RUN_SH), self.name], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1, env=env)
        log = None
        if self.log_file:
            (self.runtime_root / self.name).mkdir(parents=True, exist_ok=True)
            log = open(self.runtime_root / self.name / "worker.log", "a")
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, args=(log,), daemon=True).start()
        self._finalizer = weakref.finalize(self, _kill, self.proc)
        first = self._next_line(timeout=120.0, what="the worker's ready line")
        if not first.get("ok"):
            self.close()
            raise BridgeUnavailable(f"mzbridge worker '{self.name}' failed to start: {first.get('error')}")
        self.ready = first
        self.startup_s = time.monotonic() - t0
        return self

    def close(self, timeout: float = 15.0) -> None:
        p = self.proc
        if p is None:
            return
        if p.poll() is None:
            try:
                p.stdin.write(json.dumps({"id": 0, "op": "quit"}) + "\n")
                p.stdin.flush()
                p.wait(timeout=timeout)
            except (OSError, subprocess.TimeoutExpired, ValueError):
                p.kill()
                p.wait()
        if self._finalizer:
            self._finalizer.detach()
        pid_file = self.runtime_root / self.name / "worker.pid"
        try:
            # only our own: a start refused because the name is taken must not unlock the other worker
            if pid_file.read_text().strip() == str(p.pid):
                pid_file.unlink()
        except (OSError, ValueError):
            pass
        self.proc = None

    def __enter__(self) -> "Bridge":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None

    def rss_mb(self) -> float | None:
        """Resident memory of the worker JVM (MB), from ps."""
        if not self.pid:
            return None
        r = subprocess.run(["ps", "-o", "rss=", "-p", str(self.pid)], capture_output=True, text=True)
        return int(r.stdout.strip()) / 1024 if r.stdout.strip() else None

    def stderr_tail(self, n: int = 40) -> str:
        return "\n".join(list(self._stderr)[-n:])

    # -- requests ----------------------------------------------------------------------------------

    def request(self, op: str, spec=None, *, timeout: float | None = None, **options) -> dict:
        """Send one request and wait for its response. Raises BridgeError when ok is false."""
        if self.proc is None or self.proc.poll() is not None:
            raise BridgeError(f"mzbridge worker '{self.name}' is not running\n{self.stderr_tail()}")
        rid = next(self._ids)
        msg: dict[str, Any] = {"id": rid, "op": op, "options": {k: v for k, v in options.items() if v is not None}}
        if spec is not None:
            msg["spec"] = spec_dict(spec)
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise BridgeError(f"mzbridge worker '{self.name}' died: {e}\n{self.stderr_tail()}") from e
        deadline = time.monotonic() + (timeout or self.timeout)
        while True:
            try:
                resp = self._next_line(timeout=max(0.1, deadline - time.monotonic()), what=f"response to {op} #{rid}")
            except BridgeError:
                # a request that outlives its timeout is a hung engine: do not leave a JVM spinning
                # on a shared machine, and do not let its late answer queue ahead of the next
                # request. start() (or a BridgePool) brings a fresh worker up.
                self.close(timeout=0.5)
                raise
            if resp.get("id") == rid:
                break
        if not resp.get("ok"):
            err = resp.get("error") or {}
            probs = err.get("problems")
            detail = "\n  - " + "\n  - ".join(probs) if probs else ""
            raise BridgeError(f"{op} failed: {err.get('type')}: {err.get('message')}{detail}", err, resp)
        return resp

    def ping(self) -> dict:
        return self.request("ping")

    def build(self, spec, **options) -> dict:
        return self.request("build", spec, **options)

    def encode(self, spec, **options) -> dict:
        return self.request("encode", spec, **options)

    def coach(self, spec, **options) -> dict:
        return self.request("coach", spec, **options)

    # -- plumbing ----------------------------------------------------------------------------------

    def _pump_stdout(self) -> None:
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _pump_stderr(self, log) -> None:
        for line in self.proc.stderr:
            self._stderr.append(line.rstrip("\n"))
            if log:
                log.write(line)
                log.flush()
        if log:
            log.close()

    def _next_line(self, timeout: float, what: str) -> dict:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            raise BridgeError(f"timed out after {timeout:.0f} s waiting for {what} from '{self.name}'\n"
                              f"{self.stderr_tail()}") from None
        if line is None:
            code = self.proc.wait() if self.proc else None
            raise BridgeError(f"mzbridge worker '{self.name}' exited (code {code}) while waiting for {what}\n"
                              f"{self.stderr_tail()}")
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            raise BridgeError(f"worker '{self.name}' wrote a non-JSON line: {line[:200]!r}") from None


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()


# ------------------------------------------------------------------------------------------------
# a pool of workers

class BridgePool:
    """n workers (runtime dirs <prefix>0..<prefix>n-1) for parallel requests. Thread-safe."""

    def __init__(self, n: int, prefix: str = "pool", **bridge_kw):
        if n < 1:
            raise ValueError("n >= 1")
        if bridge_kw.get("auto_build", True):
            build_jar(xmage_dir=bridge_kw.get("xmage_dir"))
        self.workers = [Bridge(f"{prefix}{i}", start=False, **bridge_kw) for i in range(n)]
        try:
            with ThreadPoolExecutor(n) as ex:  # each worker opens its own database copy (~1 s)
                list(ex.map(lambda w: w.start(), self.workers))
        except BaseException:
            self.close()  # no JVM left running behind a pool that failed to start
            raise
        self._free: queue.Queue = queue.Queue()
        for w in self.workers:
            self._free.put(w)

    def request(self, op: str, spec=None, **options) -> dict:
        w = self._free.get()
        try:
            if w.proc is None or w.proc.poll() is not None:
                w.start()  # died or was stopped after a timeout: a fresh JVM on the same runtime dir
            return w.request(op, spec, **options)
        finally:
            self._free.put(w)

    def map(self, op: str, specs: Iterable, *, options_for=None, return_exceptions: bool = False, **options) -> list:
        """Run op on every spec in parallel; results in input order. options_for(spec) -> dict adds
        per-spec options. With return_exceptions a failed request yields its BridgeError instead
        of raising."""
        def one(spec):
            try:
                return self.request(op, spec, **{**options, **(options_for(spec) if options_for else {})})
            except BridgeError as e:
                if return_exceptions:
                    return e
                raise
        with ThreadPoolExecutor(len(self.workers)) as ex:
            return list(ex.map(one, list(specs)))

    def close(self) -> None:
        for w in self.workers:
            w.close()

    def __enter__(self) -> "BridgePool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ------------------------------------------------------------------------------------------------
# checks

def _perm_key(p: dict, seat: str, attackers: set) -> tuple:
    # the dump echoes a spec's token/set and always adds the resolved tokenClass, so token wins
    if p.get("token"):
        ident = f"token:{p['token']}/{p.get('set')}"
    elif p.get("tokenClass"):
        ident = "class:" + p["tokenClass"].rsplit(".", 1)[-1]
    else:
        ident = p.get("name")
    alias = f"{seat}:{p['id']}" if p.get("id") else None
    counters = tuple(sorted((k, v) for k, v in (p.get("counters") or {}).items() if v))
    tapped = None if alias in attackers else bool(p.get("tapped", False))
    return (ident, tapped, bool(p.get("sick", False)), int(p.get("damage", 0)), counters,
            p.get("attachTo"), bool(p.get("faceDown", False)))


def diff_dump(spec, dump: dict) -> list[str]:
    """Differences between a spec and the dump of the game built from it (empty = round trip ok).

    Compares what a spec pins down: position, the player on the play, the priority holder and
    passed players of a PRIORITY_HELD entry, life, lands played, hand (known cards; the hidden
    ones by count), graveyard order, exile, library top and size, mana pool, battlefield
    (identity, tapped, sick, damage, counters, attachment), stack, attackers and blockers.
    Attackers' tapped state comes from the engine and is not compared.
    """
    s = spec_dict(spec)
    out = []
    for k in ("turn", "activePlayer", "phase", "step"):
        if s.get(k, _DEFAULTS.get(k)) != dump.get(k):
            out.append(f"{k}: spec {s.get(k, _DEFAULTS.get(k))!r} != built {dump.get(k)!r}")
    # who is on the play decides turn order and who skipped the first draw; derived from the turn's
    # parity when the spec does not say (statespec: the player on the play takes the odd turns)
    turn, active = s.get("turn", 1), s.get("activePlayer", "A")
    starting = s.get("startingPlayer") or (active if turn % 2 == 1 else ("B" if active == "A" else "A"))
    if starting != dump.get("startingPlayer"):
        out.append(f"startingPlayer: spec {starting!r} != built {dump.get('startingPlayer')!r}")
    if s.get("enterMode") == "PRIORITY_HELD":
        if s.get("priorityPlayer") != dump.get("priorityPlayer"):
            out.append(f"priorityPlayer: spec {s.get('priorityPlayer')!r} != built {dump.get('priorityPlayer')!r}")
        if sorted(s.get("passedPlayers", [])) != sorted(dump.get("passedPlayers", [])):
            out.append(f"passedPlayers: spec {s.get('passedPlayers', [])} != built {dump.get('passedPlayers')}")
    attackers = {a["attacker"] for a in s.get("attackers", [])}
    for seat in ("A", "B"):
        sp, dp = s["players"][seat], dump["players"][seat]
        for k, default in (("life", 20), ("landsPlayed", 0)):
            if sp.get(k, default) != dp.get(k):
                out.append(f"{seat}.{k}: spec {sp.get(k, default)} != built {dp.get(k)}")
        known, hidden = sp.get("hand", []), sp.get("handUnknown", 0)
        built_hand = Counter(dp.get("hand", []))
        if len(dp.get("hand", [])) != len(known) + hidden:
            out.append(f"{seat}.hand: {len(dp.get('hand', []))} cards built, spec {len(known)} known + {hidden} hidden")
        missing = Counter(known) - built_hand
        if missing:
            out.append(f"{seat}.hand: known cards missing {dict(missing)}")
        if sp.get("graveyard", []) != dp.get("graveyard", []):
            out.append(f"{seat}.graveyard: spec {sp.get('graveyard', [])} != built {dp.get('graveyard', [])}")
        if Counter(sp.get("exile", [])) != Counter(dp.get("exile", [])):
            out.append(f"{seat}.exile: spec {sp.get('exile', [])} != built {dp.get('exile', [])}")
        top = sp.get("libraryTop", [])
        if dp.get("libraryTop", [])[:len(top)] != top:
            out.append(f"{seat}.libraryTop: spec {top} != built {dp.get('libraryTop')}")
        if sp.get("librarySize") is not None and sp["librarySize"] != dp.get("librarySize"):
            out.append(f"{seat}.librarySize: spec {sp['librarySize']} != built {dp.get('librarySize')}")
        if sorted(sp.get("manaPool") or "") != sorted(dp.get("manaPool") or ""):
            out.append(f"{seat}.manaPool: spec {sp.get('manaPool')!r} != built {dp.get('manaPool')!r}")
        want = Counter()
        for p in sp.get("battlefield", []):
            want[_perm_key(p, seat, attackers)] += p.get("count", 1)
        have = Counter(_perm_key(p, seat, attackers) for p in dp.get("battlefield", []))
        if want != have:
            out.append(f"{seat}.battlefield: missing {dict(want - have)}, extra {dict(have - want)}")
    stack = [(x["controller"], x["card"], tuple(x.get("targets", []))) for x in s.get("stack", [])]
    built_stack = [(x["controller"], x["card"], tuple(x.get("targets", []))) for x in dump.get("stack", [])]
    if stack != built_stack:
        out.append(f"stack: spec {stack} != built {built_stack}")
    atk = {(a["attacker"], a["defender"]) for a in s.get("attackers", [])}
    if atk != {(a["attacker"], a["defender"]) for a in dump.get("attackers", [])}:
        out.append(f"attackers: spec {sorted(atk)} != built {dump.get('attackers')}")
    blk = {(b["blocker"], b["attacker"]) for b in s.get("blockers", [])}
    if blk != {(b["blocker"], b["attacker"]) for b in dump.get("blockers", [])}:
        out.append(f"blockers: spec {sorted(blk)} != built {dump.get('blockers')}")
    return out


_DEFAULTS = {"turn": 1, "activePlayer": "A", "phase": "PRECOMBAT_MAIN", "step": "PRECOMBAT_MAIN"}


def golden_specs() -> dict[str, Path]:
    """The golden scenarios shipped with the bridge (java/mzbridge/specs/*.json), by stem."""
    return {p.stem: p for p in sorted(SPECS_DIR.glob("*.json"))}


def turn_start_options(spec, seat: str = "A", step: str = "PRECOMBAT_MAIN") -> dict:
    """Request options that ask for `seat`'s decision in its own coming turn.

    Without options the decision player is the spec's priority or active player. For an
    end-of-turn snapshot of the other seat's turn (17lands `eot_rollover`: B's END_TURN, meaning
    "the start of A's turn") that is the wrong seat, and even for A the first decision is often
    an instant in B's end step or A's upkeep, i.e. before the draw. This opens the window at `step`
    of the next turn instead (the critique's label policy uses the first main-phase decision)."""
    s = spec_dict(spec)
    opts: dict[str, Any] = {"decisionPlayer": seat}
    if s.get("step") in ("END_TURN", "CLEANUP") and s.get("activePlayer", "A") != seat:
        opts["decideFrom"] = {"turn": s.get("turn", 1) + 1, "step": step}
    return opts


def bench(name: str = "bench0", seeds: int = 30, budget: int = 300, determinizations: int = 4, **bridge_kw) -> dict:
    """Time a worker on the golden specs: start, build, build + decision capture, encode, coach
    (determinizations x budget, offline), and resident memory. Numbers in ms unless named _s."""
    def q(xs):
        xs = sorted(xs)
        return {"median": xs[len(xs) // 2], "p90": xs[int(0.9 * (len(xs) - 1))], "n": len(xs)}

    specs = {n: spec_dict(p) for n, p in golden_specs().items()}
    dec = {n: s.get("labels", {}).get("decisionPlayer") for n, s in specs.items()}
    out: dict[str, Any] = {"load_avg": os.getloadavg()[0] if hasattr(os, "getloadavg") else None}
    with Bridge(name, **bridge_kw) as b:
        out["start_s"] = round(b.startup_s, 2)
        out["db_open_ms"] = b.ready.get("db_ms")
        out["rss_mb_start"] = round(b.rss_mb() or 0)
        build, capture, encode, enc_only = [], [], [], []
        for k in range(seeds):
            for n, s in specs.items():
                build.append(b.request("build", s, seed=k, advance=False)["timing_ms"]["build"])
                capture.append(b.request("build", s, seed=k, decisionPlayer=dec[n])["timing_ms"]["total"])
                r = b.request("encode", s, seed=k, decisionPlayer=dec[n])
                encode.append(r["timing_ms"]["total"])
                enc_only.append(r["timing_ms"]["encode"])
        skip = len(specs) if seeds > 1 else 0  # the first round pays for class loading
        out.update(build_ms=q(build[skip:]), build_capture_ms=q(capture[skip:]), encode_request_ms=q(encode[skip:]),
                   encode_only_ms=q(enc_only[skip:]), rss_mb_after_encode=round(b.rss_mb() or 0))
        coach = {}
        for n, s in specs.items():
            t = time.monotonic()
            r = b.request("coach", s, seed=1, decisionPlayer=dec[n], determinizations=determinizations, budget=budget)
            coach[n] = {"s": round(time.monotonic() - t, 2), "sims_per_s": r["timing_ms"]["simsPerSec"]}
        out["coach"] = coach
        out["coach_s"] = q([c["s"] for c in coach.values()])
        out["rss_mb_after_coach"] = round(b.rss_mb() or 0)
    return out


# ------------------------------------------------------------------------------------------------
# CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m draftzero.gameplay.bridge", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-jar", help="compile java/mzbridge")
    sub.add_parser("ping", help="start a worker and ping it")
    bp = sub.add_parser("bench", help="time build / encode / coach on the golden specs")
    bp.add_argument("--seeds", type=int, default=30)
    bp.add_argument("--budget", type=int, default=300)
    bp.add_argument("-K", "--determinizations", type=int, default=4)
    for cmd in ("build", "encode", "coach"):
        p = sub.add_parser(cmd)
        p.add_argument("spec", help="StateSpec v1 JSON file, or the name of a golden spec (e.g. main_phase)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--decision-player", choices=["A", "B"])
        p.add_argument("--decide-from", metavar="TURN:STEP", help="open the decision window there, e.g. 7:PRECOMBAT_MAIN")
        p.add_argument("--turn-start", action="store_true",
                       help="decide for A (or --decision-player) from its next main phase (see turn_start_options)")
        p.add_argument("--lenient", action="store_true", help="warn instead of failing on card accounting")
        if cmd == "build":
            p.add_argument("--dump-decision-state", action="store_true")
            p.add_argument("--dump-library", action="store_true")
        if cmd == "encode":
            p.add_argument("--perfect-info", action="store_true", help="encode the opponent's hand too")
        if cmd == "coach":
            p.add_argument("-K", "--determinizations", type=int, default=4)
            p.add_argument("--budget", type=int, default=300)
            p.add_argument("--evaluator", choices=["offline", "remote"], default="offline")
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=50052)
            p.add_argument("--priors", default="", help="comma list of priority,target,binary,opponent (remote only)")
            p.add_argument("--resample", default=None, help="seats whose hand is re-drawn, e.g. B or A,B")
            p.add_argument("--human-action")
            p.add_argument("--timeout-sec", type=float, default=120.0)
    for p in sub.choices.values():
        p.add_argument("--name", default="cli0", help="worker / runtime dir name")
        p.add_argument("--out", help="write the JSON response here instead of stdout")
    a = ap.parse_args(argv)

    if a.cmd == "build-jar":
        print(build_jar(force=True))
        return 0
    problems = environment_problems()
    if problems:
        print("mzbridge unavailable:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    if a.cmd == "bench":
        print(json.dumps(bench(a.name, a.seeds, a.budget, a.determinizations), indent=1))
        return 0
    with Bridge(a.name) as b:
        if a.cmd == "ping":
            r = {"ready": b.ready, "ping": b.ping(), "startup_s": round(b.startup_s, 2), "rss_mb": b.rss_mb()}
        else:
            spec = a.spec
            if not Path(spec).exists() and spec in golden_specs():
                spec = golden_specs()[spec]
            opts: dict[str, Any] = {"seed": a.seed, "decisionPlayer": a.decision_player, "lenient": a.lenient or None}
            if a.turn_start:
                opts.update(turn_start_options(spec, a.decision_player or "A"))
            if a.decide_from:
                turn, _, step = a.decide_from.partition(":")
                opts["decideFrom"] = {"turn": int(turn), "step": step or None}
            if a.cmd == "build":
                opts.update(dumpDecisionState=a.dump_decision_state or None, dumpLibrary=a.dump_library or None)
            elif a.cmd == "encode":
                opts.update(perfectInfo=a.perfect_info)
            else:
                priors = {k: True for k in a.priors.split(",") if k}
                opts.update(determinizations=a.determinizations, budget=a.budget, humanAction=a.human_action,
                            timeoutSec=a.timeout_sec,
                            evaluator={"type": a.evaluator, "host": a.host, "port": a.port},
                            priors=priors or None,
                            resample=a.resample.split(",") if a.resample else None)
            try:
                r = b.request(a.cmd, spec, **opts)
            except BridgeError as e:
                print(str(e), file=sys.stderr)
                return 1
    text = json.dumps(r, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
