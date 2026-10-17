"""Smoke: the whole loop in local mode on the real engine (`pytest -m smoke`, at most ~10 minutes).

`python -m draftzero.loop.cli local --config configs/smoke.yml` with one actor, against a fresh folder store, with a
plan change queued before the start. It must stop by itself (the smoke config's budget) inside --max-hours, having
published version 0 and at least one trained version, ingested real shards, evaluated, and applied the change.
Skips when dzgorge isn't built (engine/gorge/build.sh) or the deck data isn't linked under data/gorge.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "smoke.yml"
DZGORGE = Path("/work/gorge-fork/bin/dzgorge")
MAX_HOURS = 0.16                                         # 9.6 minutes, then local mode stops the run itself


def _skip_reason() -> str | None:
    if not CONFIG.exists():
        return f"{CONFIG} is missing"
    for mod in ("draftzero.config", "draftzero.engine.gorge", "draftzero.eval.paired", "draftzero.manifest"):
        try:
            importlib.import_module(mod)
        except ImportError as e:
            return f"{mod} is not importable ({e})"
    from draftzero import config
    cfg = config.load(CONFIG)
    root = Path(cfg.engine.root)
    if not (root / "bin" / "dzgorge").exists() and not DZGORGE.exists():
        return f"no dzgorge at {root}/bin (run engine/gorge/build.sh)"
    for p in (cfg.data.decks, cfg.data.pool):
        if not (REPO / p).exists() and not Path(p).exists():
            return f"{p} is missing (link the gorge deck pool under data/gorge)"
    return None


def _tail(path: Path, n: int = 40) -> str:
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:]) if path.exists() else "(no log)"


def test_local_loop_smoke(tmp_path):
    why = _skip_reason()
    if why:
        pytest.skip(why)
    store, work = tmp_path / "store", tmp_path / "work"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONUNBUFFERED="1")
    cli = [sys.executable, "-m", "draftzero.loop.cli"]
    subprocess.run(cli + ["plan", str(store), "set", "learner.lr=2e-4", "--by", "smoke", "--before-start"], cwd=REPO,
                   env=env, check=True, timeout=60)
    p = subprocess.run(cli + ["local", "--config", str(CONFIG), "--store", str(store), "--work-dir", str(work),
                              "--actors", "1", "--max-hours", str(MAX_HOURS), "--poll", "5"],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=MAX_HOURS * 3600 + 120)
    logs = "\n\n".join(f"== {f.name}\n{_tail(f)}" for f in sorted((work / "logs").glob("*.log")))
    assert p.returncode == 0, f"local mode exited {p.returncode} (2 = the smoke budget didn't fit in " \
                              f"{MAX_HOURS} h)\n{p.stdout[-3000:]}\n{logs}"

    control = json.loads((store / "control.json").read_text())
    assert control["format"] == "dz-control/1" and control["stop"], control
    assert (store / "config.yml").exists() and (store / "manifest.json").exists() and (store / "status.md").exists()
    latest = json.loads((store / "versions/player/latest.json").read_text())
    assert latest["version"] >= 1, logs
    v0 = json.loads((store / "versions/player/v0000/meta.json").read_text())
    v1 = json.loads((store / "versions/player/v0001/meta.json").read_text())
    assert v1["net"] and v1["parent"] == 0 and v1["positions"] > 0 and v1["shards"]
    assert (store / "versions/player/v0001/model.dzgw").exists() and (store / "versions/player/v0001/model.pt").exists()
    assert v0["net"] == bool(v0.get("source"))
    metas = [json.loads(m.read_text()) for m in store.glob("shards/*/*/meta.json")]
    assert metas and sum(m["games"] for m in metas) > 0 and sum(m["records"] for m in metas) > 0
    assert all((store / "shards" / m["actor"] / f"{m['seq']:06d}" / "games.jsonl").exists() for m in metas)
    summaries = [json.loads(s.read_text()) for s in store.glob("evals/*/summary.json")]
    assert summaries and all(s["n_pairs"] > 0 for s in summaries), logs
    events = [json.loads(x) for x in (store / "events.jsonl").read_text().splitlines()]
    assert any(e["path"] == "learner.lr" and e["new"] == 2e-4 for e in events), events
    for role in ("actor", "learner", "evaluator"):
        assert list(store.glob(f"metrics/{role}/*/*.jsonl")), f"no {role} metrics"
