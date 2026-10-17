"""manifest.py — what produced a run (interface 9, contracts §3): written once, at the start, as manifest.json.

Two runs that disagree are unexplainable unless their code, engine, cards, config and machines were written down
at the time. collect(cfg) gathers them:

    format        "draftzero-manifest/1"
    run, seed     from the config
    created       UTC ISO 8601
    draftzero     {version, commit, branch, dirty}: the package and the checkout it runs from (git; None if unknown)
    engine_pin    {repo, ref}: engine/gorge/GORGE_REPO and GORGE_REF of that checkout (what build.sh would build)
    config        the whole config (draftzero.config.to_dict), and config_hash
    engine        Gorge.version() when a Gorge is given, else the same record read from <engine.root>:
                  {repo, ref, dirty, binary_sha256 (of bin/dzgorge as it is now), build (bin/dzgorge.build.json),
                   cards_lock (.cards/cards.lock), census}
    belief        with search.closedlist: the corpus the search deals the opponent's hidden list from,
                  {decks, pool, split, corpus_hash} (draftzero.oppdeck.belief.corpus_hash; corpus_hash None and an
                  "error" when the pool can't be read here); None without closedlist
    python, torch, numpy, platform
    machine       {host, cpu_quota (cgroup cores, None if uncapped), cpus (visible), mem_gb (cgroup cap, else RAM)}

Nothing here fails for a missing git, binary or package: the field is None (git runs as a subprocess; a host without
git still gets the commit from .git when it can be read). `python -m draftzero.manifest [CONFIG]` prints one.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FORMAT = "draftzero-manifest/1"
PATH = "manifest.json"
_HERE = Path(__file__).resolve().parents[2]


def _checkout() -> Path | None:
    """The source checkout this package runs from, or None for an installed copy."""
    root = Path(os.environ.get("DZ_REPO") or _HERE)
    return root if (root / "pyproject.toml").is_file() and (root / "src" / "draftzero").is_dir() else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ------------------------------------------------------------------------------------------------ code

def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=20,
                             env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _git_dir(repo: Path) -> Path | None:
    g = repo / ".git"
    if g.is_dir():
        return g
    if g.is_file():                                   # a worktree: "gitdir: <path>"
        line = g.read_text().strip()
        if line.startswith("gitdir:"):
            p = Path(line[len("gitdir:"):].strip())
            p = p if p.is_absolute() else (repo / p)
            return p if p.is_dir() else None
    return None


def _head_from_files(repo: Path) -> tuple[str | None, str | None]:
    """(commit, branch) read from .git without git; (None, None) when it cannot be read."""
    try:
        gd = _git_dir(repo)
        if gd is None:
            return None, None
        head = (gd / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head or None, None
        ref = head[4:].strip()
        common = gd
        if (gd / "commondir").is_file():
            c = Path((gd / "commondir").read_text().strip())
            common = c if c.is_absolute() else (gd / c).resolve()
        branch = ref.removeprefix("refs/heads/")
        for base in (gd, common):
            if (base / ref).is_file():
                return (base / ref).read_text().strip() or None, branch
        packed = common / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0], branch
        return None, branch
    except OSError:
        return None, None


def git_state(repo: Path | None) -> dict:
    """{commit, branch, dirty} of a checkout; branch None when detached, dirty None when unknown."""
    if repo is None:
        return {"commit": None, "branch": None, "dirty": None}
    commit = _git(repo, "rev-parse", "HEAD")
    if commit is None:
        commit, branch = _head_from_files(repo)
        return {"commit": commit, "branch": branch, "dirty": None}
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git(repo, "status", "--porcelain")
    return {"commit": commit, "branch": None if branch in (None, "HEAD") else branch,
            "dirty": None if status is None else bool(status)}


def engine_pin(repo: Path | None) -> dict:
    """{repo, ref}: the gorge pin this checkout builds against (engine/gorge/GORGE_REPO, GORGE_REF)."""
    out: dict = {"repo": None, "ref": None}
    for key, name in (("repo", "GORGE_REPO"), ("ref", "GORGE_REF")):
        f = (repo / "engine" / "gorge" / name) if repo else None
        if f is not None and f.is_file():
            out[key] = f.read_text().strip() or None
    return out


# ------------------------------------------------------------------------------------------------ engine

def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def engine_version(root: str | Path, census: bool = True) -> dict:
    """Gorge.version()'s record read straight from a gorge checkout: {repo, ref, dirty, binary_sha256, build,
    cards_lock, census}. Missing files give None fields."""
    root = Path(root)
    build = _json(root / "bin" / "dzgorge.build.json")
    build = build if isinstance(build, dict) else None
    lock = _json(root / ".cards" / "cards.lock")
    if lock is None and build:
        lock = build.get("cards_lock")
    b = build or {}
    return {"repo": b.get("repo"), "ref": b.get("ref"), "dirty": b.get("dirty"),
            "binary_sha256": _sha256(root / "bin" / "dzgorge"), "build": build, "cards_lock": lock,
            "census": bool(census)}


def _engine(cfg, gorge) -> dict:
    if gorge is not None:
        try:
            return dict(gorge.version())
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
    if cfg is None:
        from draftzero.config import EngineConfig
        e = EngineConfig()
    else:
        e = cfg.engine
    return engine_version(e.root, census=e.census)


def belief(cfg) -> dict | None:
    """The :closedlist belief corpus of a run: {decks, pool, split, corpus_hash}, the paths resolved as the engine
    resolves them (search.belief, else the run's data decks and pool); None when search.closedlist is off."""
    if cfg is None or not cfg.search.closedlist:
        return None
    from draftzero.engine.gorge import resolve_path
    b = cfg.search.belief
    decks, pool = resolve_path(b.decks or cfg.data.decks), resolve_path(b.pool or cfg.data.pool)
    rec: dict = {"decks": str(decks), "pool": str(pool), "split": b.split or "train", "corpus_hash": None}
    try:
        from draftzero.oppdeck.belief import corpus_hash
        rec["corpus_hash"] = corpus_hash(decks=decks, pool=pool, split=rec["split"])
    except Exception as e:                               # noqa: BLE001 - a manifest never fails for a missing file
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


# ------------------------------------------------------------------------------------------------ machine

def _dist_version(name: str) -> str | None:
    mod = sys.modules.get(name)
    if mod is not None and getattr(mod, "__version__", None):
        return str(mod.__version__)
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _mem_total_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024 ** 2
    except (OSError, ValueError, IndexError):
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except (ValueError, OSError, AttributeError):
        return None


def machine() -> dict:
    """{host, cpu_quota, cpus, mem_gb}: the cgroup's CPU quota in cores (None if uncapped), the visible CPUs, and the
    memory cap (else the machine's RAM) in GB."""
    from draftzero.resources import cpu_quota, mem_limit_gb
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count()
    mem = mem_limit_gb() or _mem_total_gb()
    q = cpu_quota()
    return {"host": socket.gethostname(), "cpu_quota": None if q is None else round(q, 3), "cpus": cpus,
            "mem_gb": None if mem is None else round(mem, 2)}


# ------------------------------------------------------------------------------------------------ the manifest

def collect(cfg, *, gorge=None, repo: Path | None = None) -> dict:
    """The manifest for a run of `cfg` (a draftzero.config.RunConfig; None gives the code/engine/machine part
    only, as `dz manifest` prints). `gorge`: a draftzero.engine.Gorge whose version() is the engine record (the
    controller passes one when the engine is built on its machine); without one the record is read from
    cfg.engine.root. `repo`: the DraftZero checkout (default: this one)."""
    from draftzero import __version__
    repo = Path(repo) if repo is not None else _checkout()
    config = config_hash = None
    if cfg is not None:
        from draftzero.config import config_hash as _hash
        from draftzero.config import to_dict
        config, config_hash = to_dict(cfg), _hash(cfg)
    return {
        "format": FORMAT,
        "run": None if cfg is None else cfg.run,
        "created": _now(),
        "draftzero": {"version": __version__, **git_state(repo)},
        "engine_pin": engine_pin(repo),
        "config": config,
        "config_hash": config_hash,
        "engine": _engine(cfg, gorge),
        "belief": belief(cfg),
        "python": platform.python_version(),
        "torch": _dist_version("torch"),
        "numpy": _dist_version("numpy"),
        "platform": platform.platform(),
        "machine": machine(),
        "seed": None if cfg is None else cfg.seed,
    }


def write(store, manifest: dict) -> None:
    """Write `manifest` to the store's manifest.json. `store` is a Store (anything with write_json), a local folder,
    or a store spec that draftzero.loop.store.open_store opens."""
    if hasattr(store, "write_json"):
        store.write_json(PATH, manifest)
        return
    s = str(store)
    if "://" in s:
        from draftzero.loop.store import open_store
        open_store(s).write_json(PATH, manifest)
        return
    p = Path(s) / PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{PATH}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    os.replace(tmp, p)


def read(store) -> dict | None:
    """A store's manifest.json, or None if it has none."""
    if hasattr(store, "read_json"):
        return store.read_json(PATH)
    s = str(store)
    if "://" in s:
        from draftzero.loop.store import open_store
        return open_store(s).read_json(PATH)
    return _json(Path(s) / PATH)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    cfg = None
    if args:
        from draftzero.config import load
        cfg = load(args[0])
    print(json.dumps(collect(cfg), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
