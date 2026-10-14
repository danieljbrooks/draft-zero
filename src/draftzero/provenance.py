"""provenance.py — record exactly which code produced a run.

Two moving parts decide what a run does: DraftZero and the gorge engine it pins
(engine/gorge/GORGE_REPO and GORGE_REF, built by engine/gorge/build.sh). Two runs that disagree
are unexplainable unless both are written down at the time. draftzero.manifest (contracts §3)
replaces this module with the run manifest, which adds the built binary and its card corpus.
"""
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _git(repo: Path, *args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _repo_state(repo: Path) -> dict:
    sha = _git(repo, "rev-parse", "HEAD")
    if not sha:
        return {}
    dirty = bool(_git(repo, "status", "--porcelain"))
    return {"sha": sha, "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"), "dirty": dirty}


def _package(name: str) -> dict:
    """Version and, when installed from a checkout, the git state of a dependency."""
    info: dict = {}
    try:
        from importlib.metadata import version
        info["version"] = version(name)
    except Exception:
        pass
    try:
        # find_spec locates the package WITHOUT executing it: importing torch here just to
        # read a version string can hard-fail on an unrelated NumPy ABI mismatch.
        import importlib.util
        spec = importlib.util.find_spec(name)
        if not spec or not spec.origin:
            return info
        src = Path(spec.origin).resolve().parent
        info["path"] = str(src)
        # walk up looking for the checkout root (src/<pkg> layout puts it two levels up)
        for parent in (src, *src.parents[:3]):
            st = _repo_state(parent)
            if st:
                info.update(st)
                break
    except Exception:
        pass
    return info


def _engine_pin(repo: Path) -> dict:
    """The gorge pin this checkout builds against (engine/gorge/GORGE_REPO and GORGE_REF)."""
    out: dict = {}
    for key, name in (("repo", "GORGE_REPO"), ("ref", "GORGE_REF")):
        f = repo / "engine" / "gorge" / name
        if f.exists():
            out[key] = f.read_text().strip()
    return out


def collect(root: Optional[Path] = None) -> dict:
    root = Path(root or os.getcwd())
    here = Path(__file__).resolve().parent
    return {
        "draftzero": {**_package("draftzero"), **_repo_state(here.parents[1])},
        "gorge": _engine_pin(here.parents[1]),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _package("torch").get("version"),
        "cwd": str(root),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(collect(), indent=2))
