"""provenance.py — record exactly which code produced a run.

Three moving parts decide what a run does: DraftZero, the pinned MageZero, and the XMage
build. Two runs that disagree are unexplainable unless all three are written down at the
time, so this goes into run.json before the first game.
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


def _xmage(root: Path) -> dict:
    """Identify the XMage build: an explicit VERSION file if present, else the jar set."""
    xmage = root / "xmage"
    if not xmage.exists():
        return {}
    out: dict = {"path": str(xmage.resolve())}
    for marker in ("VERSION", "BUILD", "build.txt"):
        f = xmage / marker
        if f.exists():
            out["version"] = f.read_text().strip()[:200]
            break
    lib = xmage / "lib"
    if lib.exists():
        jars = sorted(p.name for p in lib.glob("mage-magezero*.jar"))
        if jars:
            out["magezero_jar"] = jars[0]
        out["jar_count"] = len(list(lib.glob("*.jar")))
    return out


def collect(root: Optional[Path] = None) -> dict:
    root = Path(root or os.getcwd())
    here = Path(__file__).resolve().parent
    return {
        "draftzero": {**_package("draftzero"), **_repo_state(here.parents[1])},
        "magezero": _package("magezero"),
        "xmage": _xmage(root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _package("torch").get("version"),
        "cwd": str(root),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(collect(), indent=2))
