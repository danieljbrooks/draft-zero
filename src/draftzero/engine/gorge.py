"""engine/gorge.py — run the pinned gorge fork's `dzgorge` driver from Python.

`Gorge` owns everything the driver needs besides the two policies: the checkout (bin/dzgorge, .cards), the deck pool,
and the engine settings every game of a run shares (census, turn and decision caps, mulligans, the topk option cap).

play() is the resilient part (docs/031's gorge/recipe/lib.sh `_chunk` + trim.py, in Python). A game that never ends
(a decision with a huge option list was seen to hang one for hours) would hold every later game, since dzgorge
writes games in order. So a run is hard-killed after timeout_s (its whole process group: SIGTERM, then SIGKILL after
10 s; Go ignores SIGALRM), its complete games are kept, the first missing pair is replayed alone once with a shorter
limit and, if that hangs too, recorded in `skipped`; play then continues from the next pair. Games are a pure
function of (seed, pair index), so the resumed games are the ones an uninterrupted run would have played.

Files in out_dir: games.jsonl (dzgorge's game lines; `game` renumbered as one uninterrupted run would number it),
visits.jsonl.gz (with record=True: one gzip member per game), games.jsonl.summary.json, play.log (dzgorge's stderr).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import time
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from draftzero.engine.specs import PolicySpec

if TYPE_CHECKING:   # draftzero.config is the core agent's module; only its shape is used here
    from draftzero.config import RunConfig

log = logging.getLogger(__name__)

KILL_GRACE_S = 10.0         # SIGTERM -> SIGKILL
PROBE_MIN_S = 300.0         # a replayed pair gets max(PROBE_MIN_S, timeout_s / 4)
COVERAGE_TIMEOUT_S = 900.0


class GorgeError(RuntimeError):
    """dzgorge failed for a reason other than a hung game (a bad spec, a missing file, ...)."""


@dataclass
class Belief:
    """A :closedlist seat's belief corpus: deck JSONs + pool.tsv, and the pool split its decks come from."""
    decks: Path
    pool: Path
    split: str = "train"


@dataclass
class PlayResult:
    games: list[dict]           # parsed games.jsonl rows (complete games only)
    summary: dict               # dzgorge -out summary json (or {} if missing); recomputed when the call needed several runs
    games_path: Path
    corpus_path: Path | None    # visits.jsonl.gz when record=True
    skipped: list[int]          # pair indices skipped after a hang
    runs: list[dict] = field(default_factory=list)  # one per dzgorge run: first_pair, pairs, returncode, killed, wall_s, games
    wall_s: float = 0.0


# ------------------------------------------------------------------------------------------------
# processes
# ------------------------------------------------------------------------------------------------

def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _kill_group(proc: subprocess.Popen, grace: float) -> None:
    """SIGTERM the process group, SIGKILL it after `grace` seconds if the leader has not exited."""
    _signal_group(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        _signal_group(proc.pid, signal.SIGKILL)
        proc.wait()
    _signal_group(proc.pid, signal.SIGKILL)     # anything of the group that outlived its leader


def run_killable(argv: Sequence[str], *, timeout_s: float | None, stdout=subprocess.DEVNULL,
                 stderr=subprocess.DEVNULL) -> tuple[int, bool, float]:
    """(returncode, killed, wall_s) of argv run in its own process group, hard-killed after timeout_s."""
    t0 = time.monotonic()
    proc = subprocess.Popen([str(a) for a in argv], stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                            start_new_session=True)
    killed = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        killed = True
        _kill_group(proc, KILL_GRACE_S)
    except BaseException:
        _kill_group(proc, KILL_GRACE_S)
        raise
    return proc.returncode, killed, time.monotonic() - t0


# ------------------------------------------------------------------------------------------------
# trimming a run's output to complete games
# ------------------------------------------------------------------------------------------------

_GAME_ID = re.compile(rb'"game_id":\s*"p(\d+)g(\d+)"')
_GAME_FIELD = re.compile(rb'^\{"game":-?\d+')


def gzip_members(data: bytes) -> list[tuple[int, int, tuple[int, int] | None]]:
    """(start, end, (pair, leg)) of each complete gzip member of a corpus; a truncated last member is left out."""
    out, pos, mv = [], 0, memoryview(data)
    while pos < len(data):
        d, fed, head = zlib.decompressobj(31), pos, b""
        try:
            while not d.eof and fed < len(data):
                piece = mv[fed:fed + (1 << 20)]
                fed += len(piece)
                o = d.decompress(piece)
                if len(head) < 4096:
                    head += o[:4096]
        except zlib.error:
            break
        if not d.eof:
            break
        end = fed - len(d.unused_data)
        m = _GAME_ID.search(head)
        out.append((pos, end, (int(m[1]), int(m[2])) if m else None))
        pos = end
    return out


def trim_run(raw_games: bytes, raw_corpus: bytes | None, first_pair: int, legs: int
             ) -> tuple[list[bytes], list[dict], list[bytes], int]:
    """(lines, rows, members, complete pairs) of one run's output, kept to its leading complete pairs.

    A complete line ends with a newline. A game is complete when its line is, and (with a corpus) so is its gzip
    member if it recorded anything (`visits` > 0): dzgorge writes the line, then the member. Games come in game
    order, `legs` per pair; a pair is kept only with all its legs, so a replay never duplicates a leg."""
    lines = raw_games.split(b"\n")[:-1]
    members = {}
    if raw_corpus is not None:
        for s, e, key in gzip_members(raw_corpus):
            if key is not None:
                members[key] = raw_corpus[s:e]
    rows, ok = [], 0
    for i, line in enumerate(lines):
        try:
            r = json.loads(line)
        except ValueError:
            break
        key = (int(r.get("pair", -1)), int(r.get("leg", -1)))
        if key != (first_pair + i // legs, i % legs):
            break
        if raw_corpus is not None and r.get("visits", 0) > 0 and key not in members:
            break
        rows.append(r)
        ok += 1
    pairs = ok // legs
    n = pairs * legs
    kept_members = [members[(r["pair"], r["leg"])] for r in rows[:n] if (r["pair"], r["leg"]) in members]
    return lines[:n], rows[:n], kept_members, pairs


def _count_pairs_file(path: Path) -> int:
    """Lines of a -pairs-file dzgorge reads (blank and # lines are not pairs)."""
    return sum(1 for l in Path(path).read_text().splitlines() if l.strip() and not l.strip().startswith("#"))


def _spec_str(p: PolicySpec | str) -> str:
    return p.render() if isinstance(p, PolicySpec) else str(p)


def _closed(spec: str) -> bool:
    return "closedlist" in spec.split(":remote=")[0].split(":")[1:]


def _summarize(rows: list[dict], runs: list[dict], skipped: list[int]) -> dict:
    """A summary over several runs (after a kill): the counts from the kept games, the runs' own summaries."""
    res = [r.get("result") for r in rows]
    wall = sum(r["wall_s"] for r in runs)
    return {"games": len(rows), "a_wins": res.count("A"), "b_wins": res.count("B"), "draws": res.count("draw"),
            "stalls": res.count("stall"), "errors": res.count("error"),
            "visit_records": sum(int(r.get("visits", 0)) for r in rows), "wall_s": wall,
            "games_per_hour": len(rows) / wall * 3600 if wall > 0 else 0.0, "partial": True, "skipped": list(skipped),
            "runs": [r.get("summary", {}) for r in runs]}


# ------------------------------------------------------------------------------------------------
# config helpers
# ------------------------------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(os.environ.get("DZ_REPO") or Path(__file__).resolve().parents[3])


def resolve_path(p: str | Path) -> Path:
    """A config path: absolute as is; relative to the working directory if it exists there, else to the repo."""
    q = Path(p).expanduser()
    if q.is_absolute():
        return q
    if q.exists():
        return q.absolute()
    return _repo_root() / q


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


_SHA_CACHE: dict[tuple[str, int, int], str] = {}


def sha256_file(path: Path) -> str | None:
    try:
        st = Path(path).stat()
    except OSError:
        return None
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


# ------------------------------------------------------------------------------------------------
# Gorge
# ------------------------------------------------------------------------------------------------

class Gorge:
    """The dzgorge driver of one gorge checkout, with one run's engine settings (contracts §2)."""

    def __init__(self, root: Path, decks: Path, pool: Path, *, binary: Path | None = None, cards: Path | None = None,
                 census: bool = True, max_turns: int = 100, max_intents: int = 20000, mulligans: int = 2,
                 mull_heuristic: bool = True, topk_max_options: int = 64):
        self.root = Path(root).absolute()
        self.decks = Path(decks).absolute()
        self.pool = Path(pool).absolute()
        self.binary = Path(binary).absolute() if binary else self.root / "bin" / "dzgorge"
        self.cards = Path(cards).absolute() if cards else self.root / ".cards"
        self.census = bool(census)
        self.max_turns, self.max_intents, self.mulligans = int(max_turns), int(max_intents), int(mulligans)
        self.mull_heuristic = bool(mull_heuristic)
        self.topk_max_options = int(topk_max_options or 0)

    def __repr__(self) -> str:
        return f"Gorge(root={str(self.root)!r}, decks={str(self.decks)!r}, pool={str(self.pool)!r}, census={self.census})"

    @classmethod
    def from_config(cls, cfg: "RunConfig") -> "Gorge":
        """From a RunConfig's engine and data sections (relative paths: see resolve_path)."""
        e, d = _get(cfg, "engine"), _get(cfg, "data")
        name = _get(e, "name", "gorge")
        if name != "gorge":
            raise ValueError(f"engine.name={name!r}: only gorge is supported")
        return cls(resolve_path(_get(e, "root")), resolve_path(_get(d, "decks")), resolve_path(_get(d, "pool")),
                   census=_get(e, "census", True), max_turns=_get(e, "max_turns", 100),
                   max_intents=_get(e, "max_intents", 20000), mulligans=_get(e, "mulligans", 2),
                   mull_heuristic=_get(e, "mull_heuristic", True), topk_max_options=_get(e, "topk_max_options", 64))

    # -- provenance ---------------------------------------------------------------------------

    def version(self) -> dict:
        """{repo, ref (fork commit), dirty, binary_sha256, build, cards_lock, census, root, binary,
        binary_matches_build}. ref/dirty come from git in root, else from the build record."""
        build = _read_json(Path(str(self.binary) + ".build.json"))
        ref = _git(self.root, "rev-parse", "HEAD")
        status = _git(self.root, "status", "--porcelain", "--untracked-files=no") if ref else None
        sha = sha256_file(self.binary)
        repo = build.get("repo") or _git(self.root, "remote", "get-url", "fork") or _git(self.root, "remote", "get-url", "origin")
        return {"repo": repo, "ref": ref or build.get("ref"), "dirty": bool(status) if ref else build.get("dirty"),
                "binary_sha256": sha, "build": build, "cards_lock": _read_json(self.cards / "cards.lock"),
                "census": self.census, "root": str(self.root), "binary": str(self.binary),
                "binary_matches_build": bool(sha) and sha == build.get("binary_sha256")}

    def coverage(self) -> dict:
        """`dzgorge coverage` over the pool: {cards, decks, unsupported_cards, failed_decks, supported_decks,
        unsupported: {card: {decks, missing}}, supported: [deck names playing only supported cards]}."""
        import tempfile
        with tempfile.TemporaryDirectory(prefix="dzcov-") as tmp:
            out, txt = Path(tmp) / "coverage.out", Path(tmp) / "supported.txt"
            with open(out, "wb") as so, open(Path(tmp) / "coverage.err", "wb") as se:
                rc, killed, _ = run_killable([self.binary, "coverage", "-cards", self.cards, "-decks", self.decks,
                                              "-pool", self.pool, "-out", txt], timeout_s=COVERAGE_TIMEOUT_S,
                                             stdout=so, stderr=se)
            text = out.read_text(errors="replace")
            if rc != 0 or killed:
                err = (Path(tmp) / "coverage.err").read_text(errors="replace")
                raise GorgeError(f"dzgorge coverage {'timed out' if killed else f'exited {rc}'}: {err[-2000:]}")
            supported = txt.read_text().split() if txt.exists() else []
        res: dict = {"unsupported": {}, "supported": supported}
        for line in text.splitlines():
            if m := re.match(r"unsupported: (.+?)\s+in\s+(\d+) decks\s+missing \[(.*)\]\s*$", line):
                res["unsupported"][m[1]] = {"decks": int(m[2]), "missing": m[3]}
            elif m := re.match(r"(\d+) distinct cards in (\d+) decks; (\d+) unsupported; (\d+) decks failed to "
                               r"resolve; (\d+) of \d+ decks play only supported cards", line):
                res.update(cards=int(m[1]), decks=int(m[2]), unsupported_cards=int(m[3]), failed_decks=int(m[4]),
                           supported_decks=int(m[5]))
        if "decks" not in res:
            raise GorgeError(f"dzgorge coverage: no summary line in its output: {text[-2000:]}")
        return res

    # -- play ---------------------------------------------------------------------------------

    def play_argv(self, a: PolicySpec | str, b: PolicySpec | str, *, pairs: int, seed: int, out: Path,
                  split: str = "train", first_pair: int = 0, workers: int = 2, corpus: Path | None = None,
                  record_every: int = 1, belief: Belief | None = None, deckfile: Path | None = None,
                  pairs_file: Path | None = None, single: bool = False, extra_flags: Sequence[str] = ()) -> list[str]:
        """The `dzgorge play` command line of one run (corpus set = record)."""
        sa, sb = _spec_str(a), _spec_str(b)
        argv = [str(self.binary), "play", "-cards", str(self.cards), "-decks", str(self.decks), "-pool", str(self.pool),
                "-split", split, "-seed", str(int(seed)), "-first-pair", str(int(first_pair)), "-pairs", str(int(pairs)),
                "-workers", str(int(workers)), "-a", sa, "-b", sb, "-out", str(out),
                "-max-turns", str(self.max_turns), "-max-intents", str(self.max_intents),
                "-mulligans", str(self.mulligans), "-progress", "0"]
        if self.mull_heuristic:
            argv.append("-mull-heuristic")
        if self.topk_max_options > 0:
            argv += ["-topk-max-options", str(self.topk_max_options)]
        if not self.census:
            argv.append("-no-paid-mana-census")
        if corpus is not None:
            argv += ["-record-features", "entity", "-record-every", str(int(record_every)), "-corpus", str(corpus)]
        if belief is None and (_closed(sa) or _closed(sb)):
            belief = Belief(self.decks, self.pool, "train")     # dzgorge's own default, made explicit
        if belief is not None:
            if belief.split == split and split != "train":
                raise ValueError(f"belief split {belief.split!r} is the games' own split: the belief would hold "
                                 "the opponents' real decks (dzgorge refuses it)")
            argv += ["-belief-decks", str(belief.decks), "-belief-pool", str(belief.pool), "-belief-split", belief.split]
        if deckfile is not None:
            argv += ["-deckfile", str(deckfile)]
        if pairs_file is not None:
            argv += ["-pairs-file", str(pairs_file)]
        if single:
            argv.append("-single")
        return argv + [str(x) for x in extra_flags]

    def play(self, a: PolicySpec | str, b: PolicySpec | str, *, pairs: int, seed: int, out_dir: Path,
             split: str = "train", first_pair: int = 0, workers: int = 2, record: bool = False, record_every: int = 1,
             belief: Belief | None = None, timeout_s: float | None = None, deckfile: Path | None = None,
             pairs_file: Path | None = None, single: bool = False, extra_flags: Sequence[str] = (),
             probe_timeout_s: float | None = None) -> PlayResult:
        """Play pairs [first_pair, first_pair + pairs) of (seed, split) into out_dir (see the module docstring).

        Each pair is played twice, the policies swapping seats, unless a == b (as strings) or single: then once,
        so with a == b `pairs` counts GAMES. With pairs_file, pairs <= 0 means every line from first_pair.
        timeout_s limits each dzgorge run (None: no limit); a replayed pair gets probe_timeout_s, default
        max(300, timeout_s / 4). Raises GorgeError when dzgorge fails without playing (bad spec, missing file)."""
        sa, sb = _spec_str(a), _spec_str(b)
        legs = 1 if (sa == sb or single) else 2
        if pairs_file is not None:
            n_lines = _count_pairs_file(pairs_file)
            if pairs <= 0 or first_pair + pairs > n_lines:
                pairs = n_lines - first_pair
        if pairs <= 0:
            raise ValueError(f"pairs={pairs}: nothing to play")
        if probe_timeout_s is None and timeout_s is not None:
            probe_timeout_s = max(PROBE_MIN_S, timeout_s / 4)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        games_path, corpus_path = out_dir / "games.jsonl", (out_dir / "visits.jsonl.gz" if record else None)
        chunk, chunk_gz, chunk_err = out_dir / ".run.jsonl", out_dir / ".run.jsonl.gz", out_dir / ".run.err"
        summary_path = Path(str(chunk) + ".summary.json")
        kw = dict(split=split, workers=workers, record_every=record_every, belief=belief, deckfile=deckfile,
                  pairs_file=pairs_file, single=single, extra_flags=extra_flags)
        rows: list[dict] = []
        runs: list[dict] = []
        skipped: list[int] = []
        t0 = time.monotonic()

        def run(first: int, n: int, limit: float | None) -> tuple[int, dict]:
            """One dzgorge run of pairs [first, first + n); appends its complete pairs. -> (pairs kept, run record)"""
            for p in (chunk, chunk_gz, summary_path, chunk_err):
                p.unlink(missing_ok=True)
            argv = self.play_argv(sa, sb, pairs=n, seed=seed, out=chunk, first_pair=first,
                                  corpus=chunk_gz if record else None, **kw)
            with open(chunk_err, "wb") as err:
                rc, killed, wall = run_killable(argv, timeout_s=limit, stderr=err)
            stderr = chunk_err.read_bytes()
            with open(out_dir / "play.log", "ab") as f:
                f.write(f"# dzgorge play -first-pair {first} -pairs {n} (limit {limit} s): ".encode()
                        + (b"killed" if killed else f"exit {rc}".encode()) + f" after {wall:.1f} s\n".encode() + stderr)
            raw = chunk.read_bytes() if chunk.exists() else b""
            raw_gz = (chunk_gz.read_bytes() if chunk_gz.exists() else b"") if record else None
            lines, kept, members, k = trim_run(raw, raw_gz, first, legs)
            with open(games_path, "ab") as f:
                for line, r in zip(lines, kept):
                    g = (r["pair"] - first_pair) * legs + r["leg"]
                    if r.get("game") != g:
                        r["game"] = g
                        line = _GAME_FIELD.sub(b'{"game":%d' % g, line, count=1)
                    f.write(line + b"\n")
            if record:
                with open(corpus_path, "ab") as f:
                    for m in members:
                        f.write(m)
            rows.extend(kept)
            rec = {"first_pair": first, "pairs": n, "timeout_s": limit, "returncode": rc, "killed": killed,
                   "wall_s": wall, "games": len(kept), "summary": _read_json(summary_path)}
            runs.append(rec)
            # A crash (a panic, a Go fatal error, a signal from outside) is handled like a hang; an exit with an
            # error and no game is a refusal (bad spec, missing file) that every replay would repeat.
            crashed = rc < 0 or b"panic:" in stderr or b"fatal error:" in stderr
            if not killed and rc > 0 and not kept and not crashed:
                raise GorgeError(f"dzgorge play exited {rc} without a game: "
                                 f"{stderr.decode(errors='replace')[-2000:]}\n  argv: {' '.join(argv)}")
            return k, rec

        try:
            games_path.write_bytes(b"")
            if record:
                corpus_path.write_bytes(b"")
            s, end = first_pair, first_pair + pairs
            while s < end:
                k, rec = run(s, end - s, timeout_s)
                if s + k >= end:
                    break
                m = s + k   # the first missing pair: killed (hung) or crashed there
                log.warning("dzgorge play %s at pair %d (%d of %d pairs done); replaying it alone",
                            "killed" if rec["killed"] else f"exited {rec['returncode']}", m, k, end - s)
                k1, _ = run(m, 1, probe_timeout_s)
                if k1 < 1:
                    log.warning("pair %d (seed %d) did not finish alone either; skipped", m, seed)
                    skipped.append(m)
                s = m + 1
        finally:
            for p in (chunk, chunk_gz, summary_path, chunk_err):
                p.unlink(missing_ok=True)
        wall = time.monotonic() - t0
        if len(runs) == 1 and not runs[0]["killed"] and runs[0]["returncode"] == 0 and runs[0]["summary"]:
            summary = runs[0]["summary"]
        else:
            summary = _summarize(rows, runs, skipped)
        Path(str(games_path) + ".summary.json").write_text(json.dumps(summary, indent=1))
        for r in runs:
            r.pop("summary", None)
        return PlayResult(games=rows, summary=summary, games_path=games_path, corpus_path=corpus_path,
                          skipped=skipped, runs=runs, wall_s=wall)

    # -- pack ---------------------------------------------------------------------------------

    def pack(self, corpus: Sequence[Path], out_dir: Path, *, shard: int = 200_000, game_prefix: str | None = None,
             shrink: bool = True) -> list[Path]:
        """`dzgorge pack` the visit corpora into shards out_dir/00000, ... (nets/SPEC.md §2), then shrink card_raw
        to float16 and prefix the game ids. Built in a sibling temporary directory and renamed into place, so
        out_dir either holds the whole pack or does not exist. out_dir must not exist or be empty.
        Empty corpora are skipped; no record at all gives an empty out_dir and []."""
        out_dir = Path(out_dir)
        if out_dir.exists() and any(out_dir.iterdir()):
            raise FileExistsError(f"{out_dir}: not empty")
        srcs = [Path(c) for c in corpus if Path(c).exists() and Path(c).stat().st_size > 0]
        for c in srcs:
            if "," in str(c):
                raise ValueError(f"{c}: dzgorge pack -in is comma-separated; the path has a comma")
        tmp = out_dir.with_name(f".{out_dir.name}.tmp-{os.getpid()}")
        if tmp.exists():
            shutil.rmtree(tmp)
        try:
            if srcs:
                tmp.mkdir(parents=True)
                errf = tmp.with_name(tmp.name + ".err")
                with open(errf, "wb") as err:
                    rc, _, _ = run_killable([self.binary, "pack", "-in", ",".join(str(c) for c in srcs), "-out", tmp,
                                             "-shard", str(int(shard))], timeout_s=None, stderr=err)
                msg = errf.read_text(errors="replace")
                errf.unlink(missing_ok=True)
                if rc != 0:
                    raise GorgeError(f"dzgorge pack exited {rc}: {msg[-2000:]}")
                if "stopped at a bad record" in msg:
                    log.warning("dzgorge pack: %s", msg.strip())
                for d in sorted(p for p in tmp.iterdir() if (p / "meta.json").exists()):
                    if shrink:
                        shrink_card_raw(d)
                    if game_prefix:
                        prefix_games(d, game_prefix)
            else:
                tmp.mkdir(parents=True)
            if out_dir.exists():
                out_dir.rmdir()
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp, out_dir)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
        return sorted(p for p in out_dir.iterdir() if (p / "meta.json").exists())


def shrink_card_raw(shard_dir: Path) -> bool:
    """Store a shard's card_raw (about 95% of its bytes) as float16, in place (v0.1.0 gorge/recipe/shrink.py).

    The values are small multiples of 0.05-0.25; float16 keeps them within 2.5e-4 relative. The loaders
    (nets.packfmt.read_shard, nets.data.to_batch) cast card_raw to float32 when they build a batch.
    Returns False when it already was float16."""
    import numpy as np
    f = Path(shard_dir) / "card_raw.npy"
    a = np.load(f, mmap_mode="r")
    if a.dtype == np.float16:
        return False
    tmp = f.with_name("card_raw.tmp.npy")
    np.save(tmp, np.asarray(a, np.float16))
    del a
    os.replace(tmp, f)
    return True


def prefix_games(shard_dir: Path, prefix: str) -> None:
    """Rewrite a shard's meta.json game ids to prefix + id, so that the holdout (crc32(game id) % 20 == 0)
    differs across actors and chunks, whose ids (p<pair>g<leg>) repeat."""
    f = Path(shard_dir) / "meta.json"
    meta = json.loads(f.read_text())
    meta["games"] = [f"{prefix}{g}" for g in meta.get("games", [])]
    tmp = f.with_name("meta.tmp.json")
    tmp.write_text(json.dumps(meta, indent=1))
    os.replace(tmp, f)


__all__ = ["Belief", "Gorge", "GorgeError", "PlayResult", "gzip_members", "prefix_games", "resolve_path",
           "run_killable", "shrink_card_raw", "trim_run"]
