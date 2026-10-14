"""Smoke: the deck builder on the loop, in local mode on the real engine (`pytest -m smoke`, at most ~10 minutes).

`python -m draftzero.loop.cli local --config configs/smoke-sealed.yml` (player.learn false: the pilot is the
16-simulation search without a network) against a fresh folder store. It must stop by itself at the config's budget
(two trained builder versions) inside --max-hours, having published builder versions 0-2, played builder shards on
draftlab pools from `dzgorge sealed-pools`, and evaluated versions 1 and 2 against best23's decks.

The engine is a dzgorge with sealed-pools: DZ_SEALED_DZGORGE (default /work/bin-sealed/dzgorge, built from the gorge
fork's draftzero-sealed branch), run from a temporary engine root holding bin/dzgorge and .cards (symlinks; the
cards from DZ_GORGE_ROOT, default /work/gorge-fork). Skips when either is missing or the binary has no sealed-pools.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "smoke-sealed.yml"
DZGORGE = Path(os.environ.get("DZ_SEALED_DZGORGE", "/work/bin-sealed/dzgorge"))
CARDS = Path(os.environ.get("DZ_GORGE_ROOT", "/work/gorge-fork")) / ".cards"
MAX_HOURS = 0.16                                         # 9.6 minutes, then local mode stops the run itself


def _skip_reason() -> str | None:
    if not DZGORGE.exists():
        return f"no {DZGORGE} (build the gorge fork's draftzero-sealed branch there, or set DZ_SEALED_DZGORGE)"
    if not (CARDS / "cardsfolder").is_dir():
        return f"no card corpus at {CARDS} (set DZ_GORGE_ROOT)"
    p = subprocess.run([str(DZGORGE), "sealed-pools", "-h"], capture_output=True, text=True, timeout=60)
    if "-best23" not in p.stdout + p.stderr:
        return f"{DZGORGE} has no sealed-pools command"
    try:
        import torch  # noqa: F401
    except ImportError:
        return "no torch"
    return None


def _tail(path: Path, n: int = 40) -> str:
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:]) if path.exists() else "(no log)"


def test_sealed_loop_smoke(tmp_path):
    why = _skip_reason()
    if why:
        pytest.skip(why)
    root = tmp_path / "engine"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "dzgorge").symlink_to(DZGORGE)
    if Path(str(DZGORGE) + ".build.json").exists():
        (root / "bin" / "dzgorge.build.json").symlink_to(Path(str(DZGORGE) + ".build.json"))
    (root / ".cards").symlink_to(CARDS)
    store, work = tmp_path / "store", tmp_path / "work"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONUNBUFFERED="1")
    from draftzero import config
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(config.dump(config.load(CONFIG, {"engine.root": str(root)})))
    cli = [sys.executable, "-m", "draftzero.loop.cli"]
    p = subprocess.run(cli + ["local", "--config", str(cfg_path), "--store", str(store), "--work-dir", str(work),
                              "--builder-actors", "1", "--max-hours", str(MAX_HOURS), "--poll", "5"],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=MAX_HOURS * 3600 + 120)
    logs = "\n\n".join(f"== {f.name}\n{_tail(f)}" for f in sorted((work / "logs").glob("*.log")))
    assert p.returncode == 0, f"local mode exited {p.returncode} (2 = the smoke budget didn't fit in " \
                              f"{MAX_HOURS} h)\n{p.stdout[-3000:]}\n{logs}"
    assert sorted(f.stem for f in (work / "logs").glob("*.log")) == \
        ["builder-actor-0", "builder-evaluator", "builder-learner", "controller"]

    control = json.loads((store / "control.json").read_text())
    assert control["stop"] and control["stop_reason"] == "budget.max_versions 2", control
    assert control["version"] == {"player": None, "builder": 2} and control["builder_evals"] == []
    assert not (store / "versions" / "player").exists() and not (store / "shards").exists()
    for v in (1, 2):
        meta = json.loads((store / f"versions/builder/v{v:04d}/meta.json").read_text())
        assert meta["net"] and meta["parent"] == v - 1 and meta["decided"] > 0, meta
        assert (store / f"versions/builder/v{v:04d}/ens.pt").exists()
    metas = [json.loads(m.read_text()) for m in store.glob("builder/shards/*/*/meta.json")]
    assert {m["version"] for m in metas} == {0, 1} and sum(m["decided"] for m in metas) >= 32, metas
    assert all(m["spec"].startswith("az:sims=16") and "gonet" not in m["spec"] for m in metas)
    decks = [json.loads(x) for d in store.glob("builder/shards/*/*/decks.jsonl") for x in d.read_text().splitlines()]
    assert all(sum(c["count"] for c in d["cards"]) == 40 for d in decks)
    assert {d["kind"] for d in decks} >= {"r0", "b", "t", "p", "l"}
    summaries = {s["version"]: s for s in (json.loads(f.read_text()) for f in store.glob("evals/*/summary.json"))}
    assert sorted(summaries) == [1, 2], logs
    assert all(s["opponent"] == "best23" and s["n_games"] > 0 and s["score"] is not None for s in summaries.values())
    for role in ("builder-actor", "builder-learner", "builder-evaluator"):
        assert list(store.glob(f"metrics/{role}/*/*.jsonl")), f"no {role} metrics"
    assert "**Builder:**" in (store / "status.md").read_text()
