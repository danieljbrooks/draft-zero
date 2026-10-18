"""Interface 8, the run-config schema: every file in configs/ loads and round-trips, and the schema's keys and live
paths are pinned here."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from draftzero import config as C

REPO = Path(__file__).resolve().parents[2]
CONFIGS = sorted((REPO / "configs").rglob("*.yml"))
BASE = REPO / "configs" / "base" / "player.yml"

# changing these is changing interface 8: old configs must keep running, or fail loudly
SCHEMA_KEYS = [
    "run", "code", "seed", "mode", "budget.max_versions", "budget.max_hours", "budget.max_games",
    "engine.name", "engine.root", "engine.census", "engine.max_turns", "engine.max_intents", "engine.mulligans",
    "engine.mull_heuristic", "engine.topk_max_options",
    "data.decks", "data.pool", "data.train_split", "data.eval_split",
    "player.arch", "player.config", "player.init", "player.learn",
    "search.keys", "search.topk", "search.cpuct", "search.closedlist", "search.belief.decks", "search.belief.pool",
    "search.belief.split",
    "actor.sims", "actor.explore", "actor.record_every", "actor.chunk_games", "actor.chunk_secs", "actor.workers",
    "actor.pcr.prob", "actor.pcr.cheap_sims", "actor.pcr.by_kind", "actor.decided.high", "actor.decided.n",
    "actor.decided.sims", "actor.decided.weight", "actor.forced.k",
    "learner.device", "learner.min_new_positions", "learner.window_positions", "learner.epochs",
    "learner.steps_per_position", "learner.batch", "learner.lr", "learner.warmup", "learner.policy_target",
    "learner.value_blend", "learner.value_next", "learner.warm_start", "learner.threads", "learner.surprise_weight",
    "learner.value_surprise", "learner.ema_decay", "learner.window.kind", "learner.window.c", "learner.window.alpha",
    "learner.window.beta",
    "eval.every_versions", "eval.sims", "eval.pairs", "eval.seed", "eval.workers", "eval.opponents",
    "store", "tracking.trackio", "tracking.project", "sync.games_per_version", "paused",
    "builder.learn", "builder.kinds", "builder.random_share", "builder.members", "builder.pools_per_chunk",
    "builder.games_per_chunk", "builder.games_per_version", "builder.sims", "builder.pilot", "builder.closedlist",
    "builder.workers", "builder.chunk_secs", "builder.init", "builder.window.kind", "builder.window.size",
    "builder.warm_start", "builder.epochs", "builder.lr", "builder.warm_lr", "builder.draws", "builder.castable",
    "builder.land_support", "builder.threads", "builder.pools.kind", "builder.pools.set", "builder.pools.seed",
    "builder.pools.packs", "builder.pools.collation", "builder.pools.count", "builder.eval.every_versions",
    "builder.eval.pools", "builder.eval.pairs", "builder.eval.seed", "builder.eval.sims", "builder.eval.opponent",
    "builder.eval.workers",
]
LIVE = {
    "actor.sims", "actor.pcr.prob", "actor.pcr.cheap_sims", "actor.pcr.by_kind", "actor.record_every",
    "actor.chunk_games", "actor.decided.high", "actor.decided.n", "actor.decided.sims", "actor.decided.weight",
    "actor.forced.k", "learner.lr", "learner.window_positions", "learner.min_new_positions",
    "learner.steps_per_position", "learner.epochs", "eval.every_versions", "eval.pairs", "eval.sims",
    "budget.max_versions", "budget.max_hours", "budget.max_games", "paused",
}


def _leaves(d: dict, prefix: str = "") -> set[str]:
    out = set()
    for k, v in d.items():
        if isinstance(v, dict) and k not in ("config", "by_kind"):
            out |= _leaves(v, f"{prefix}{k}.")
        else:
            out.add(prefix + k)
    return out


def test_schema_keys_and_live_paths_are_pinned():
    assert list(C.flatten(C.from_dict({"run": "t"}))) == SCHEMA_KEYS
    assert set(C.LIVE) == LIVE


def test_there_are_configs():
    names = {p.relative_to(REPO / "configs").as_posix() for p in CONFIGS}
    assert {"base/player.yml", "smoke.yml", "smoke-distributed.yml", "runs/e1-baseline-mlp.yml",
            "runs/e2-p0-control.yml", "runs/e2-p1-pcr-200-40.yml", "runs/e2-p3-pcr-500-100.yml"} <= names


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.relative_to(REPO / "configs").as_posix())
def test_every_config_loads_and_round_trips(path, tmp_path):
    cfg = C.load(path)
    out = tmp_path / "config.yml"
    out.write_text(C.dump(cfg))
    again = C.load(out)
    assert again == cfg and C.config_hash(again) == C.config_hash(cfg)
    assert C.config_hash(C.loads(C.dump(again))) == C.config_hash(cfg)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.relative_to(REPO / "configs").as_posix())
def test_every_player_config_builds(path):
    pytest.importorskip("torch")
    from draftzero.nets.models import full_config
    cfg = C.load(path)
    full_config(cfg.player.arch, cfg.player.config)


def test_base_states_every_key():
    raw = yaml.safe_load(BASE.read_text())
    assert "extends" not in raw and _leaves(raw) == set(SCHEMA_KEYS)


def test_base_is_loop_f():
    c = C.load(BASE)
    assert (c.actor.sims, c.search.cpuct, c.actor.record_every, c.player.arch, c.player.init) == (200, 0.25, 1, "mlp", None)
    assert (c.learner.lr, c.learner.epochs, c.learner.value_blend, c.learner.warm_start) == (1e-4, 2.0, 1.0, True)
    assert (c.eval.sims, c.eval.pairs, c.eval.opponents) == (100, 100, ["prev", "sbtactical"])
    assert c.actor.pcr.prob == 0 and c.search.closedlist
    assert c.actor.decided.high == 0 and c.actor.forced.k == 0         # self-play extras off by default
    assert c.learner.value_next == 0.0                                 # so is the value target's next-turn share


def test_run_ids_are_unique_and_match_file_names():
    runs = {}
    for p in CONFIGS:
        c = C.load(p)
        assert c.run not in runs, f"{p} and {runs.get(c.run)} share run id {c.run}"
        runs[c.run] = p
        if p.parent.name == "runs":
            assert c.run == p.stem and c.store.endswith(c.run)


def test_e2_arms_differ_from_base_only_in_actor():
    base = C.load(BASE)
    arms = [p for p in CONFIGS if p.name.startswith("e2-")]
    assert len(arms) >= 3
    for p in arms:
        changed = set(C.diff(base, C.load(p))) - {"run", "store"}
        assert all(k.startswith("actor.") for k in changed), (p.name, changed)


def test_smoke_is_small():
    c = C.load(REPO / "configs" / "smoke.yml")
    assert c.run == "smoke" and c.store == "runs/smoke" and c.budget.max_versions == 2
    assert c.actor.sims <= 16 and c.actor.chunk_games <= 4 and c.eval.pairs <= 4 and c.learner.min_new_positions <= 200
