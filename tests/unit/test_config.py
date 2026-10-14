"""draftzero.config: the run-config schema (interface 8), extends, conversion, checks, plan changes and the hash."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from draftzero import config as C


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def cfg() -> C.RunConfig:
    return C.from_dict({"run": "t"})


# ------------------------------------------------------------------------------------------------ defaults, schema

def test_defaults_are_the_contract_values(cfg):
    d = C.to_dict(cfg)
    assert d["run"] == "t" and d["code"] == "dev" and d["seed"] == 1 and d["mode"] == "async"
    assert d["budget"] == {"max_versions": None, "max_hours": None, "max_games": None}
    assert d["engine"] == {"name": "gorge", "root": "/work/gorge-fork", "census": True, "max_turns": 100,
                           "max_intents": 20000, "mulligans": 2, "mull_heuristic": True, "topk_max_options": 64}
    assert d["data"] == {"decks": "data/gorge/decks", "pool": "data/gorge/pool.tsv", "train_split": "train",
                         "eval_split": "eval"}
    assert d["player"] == {"arch": "mlp", "init": None, "learn": True,
                           "config": {"d": 128, "ff": 256, "layers": 1, "d_emb": 32, "card_d": 64, "head_hidden": 128}}
    assert d["search"] == {"keys": ["autopay", "oppnodes", "oppfull", "morekinds"], "topk": 6, "cpuct": 0.25,
                           "closedlist": True, "belief": {"decks": None, "pool": None, "split": "train"}}
    assert d["actor"] == {"sims": 100, "explore": True, "record_every": 1, "chunk_games": 50, "chunk_secs": 3600,
                          "workers": 2, "pcr": {"prob": 0.0, "cheap_sims": 40, "by_kind": {}}}
    assert d["learner"] == {"device": "cpu", "min_new_positions": 20000, "window_positions": 300000, "epochs": 2.0,
                            "steps_per_position": None, "batch": 512, "lr": 1e-4, "warmup": 100,
                            "policy_target": "visits", "value_blend": 1.0, "warm_start": True, "threads": 4,
                            "surprise_weight": 0.0, "value_surprise": 0.0, "ema_decay": 0.0,
                            "window": {"kind": "fixed", "c": 150000, "alpha": 0.75, "beta": 0.4}}
    assert d["eval"] == {"every_versions": 1, "sims": 100, "pairs": 100, "seed": 3131, "workers": 2,
                         "opponents": ["prev", "sbtactical"]}
    assert d["store"] == "runs/smoke" and d["tracking"] == {"trackio": False, "project": "draftzero"}
    assert d["sync"] == {"games_per_version": 200} and d["paused"] is False
    assert d["builder"] == {
        "learn": False, "kinds": "btplr", "random_share": 0.15, "members": 5, "pools_per_chunk": 50,
        "games_per_chunk": 400, "games_per_version": 10000, "sims": None, "pilot": None, "closedlist": False,
        "workers": 2, "chunk_secs": 3600, "init": None, "window": {"kind": "versions", "size": 4}, "warm_start": True,
        "epochs": 40, "lr": 3e-3, "warm_lr": 1e-3, "draws": False, "castable": False, "land_support": 0.02,
        "threads": 4, "pools": {"kind": "draftlab", "set": "FDN", "seed": 1, "packs": 6, "collation": "arena",
                                "count": 5000},
        "eval": {"every_versions": 1, "pools": 200, "pairs": 1, "seed": 2, "sims": 100, "opponent": "best23",
                 "workers": 2}}


@pytest.mark.parametrize("over, path", [
    ({"builder": {"kinds": "bx"}}, "builder.kinds"),
    ({"builder": {"kinds": ""}}, "builder.kinds"),
    ({"builder": {"games_per_chunk": 15}}, "builder.games_per_chunk"),
    ({"builder": {"pools_per_chunk": 1}}, "builder.pools_per_chunk"),
    ({"builder": {"sims": -1}}, "builder.sims"),
    ({"builder": {"pilot": "/x/model.pt"}}, "builder.pilot"),
    ({"builder": {"pilot": "/x:y/model.dzgw"}}, "builder.pilot"),
    ({"builder": {"window": {"kind": "hours"}}}, "builder.window.kind"),
    ({"builder": {"land_support": 0.5}}, "builder.land_support"),
    ({"builder": {"pools": {"set": "BLB"}}}, "builder.pools.set"),
    ({"builder": {"pools": {"collation": "draft"}}}, "builder.pools.collation"),
    ({"builder": {"pools": {"count": 10}}}, "builder.pools.count"),
    ({"builder": {"eval": {"pools": 3}}}, "builder.eval.pools"),
    ({"builder": {"eval": {"seed": 1}}}, "builder.eval.seed"),
    ({"builder": {"eval": {"opponent": "random"}}}, "builder.eval.opponent"),
    ({"player": {"learn": False}}, "player.learn"),
])
def test_builder_checks(over, path):
    with pytest.raises(C.ConfigError) as e:
        C.from_dict({"run": "t", **over})
    assert e.value.path == path


def test_builder_only_run_loads_and_its_machine_paths_are_machine():
    cfg = C.from_dict({"run": "t", "player": {"learn": False},
                       "builder": {"learn": True, "pilot": "/nets/b7.dzgw", "kinds": "bt"}})
    assert not cfg.player.learn and cfg.builder.learn and cfg.builder.pilot == "/nets/b7.dzgw"
    assert {"builder.pilot", "builder.init"} <= C.MACHINE
    assert not C.is_live("builder.learn") and not C.is_live("builder.games_per_chunk")


def test_defaults_are_not_shared_between_configs():
    a, b = C.from_dict({"run": "a"}), C.from_dict({"run": "b"})
    a.eval.opponents.append("bot")
    a.player.config["d"] = 7
    assert b.eval.opponents == ["prev", "sbtactical"] and b.player.config["d"] == 128
    assert C.PLAYER_MLP["d"] == 128


def test_run_is_required():
    with pytest.raises(C.ConfigError, match="missing required key run") as e:
        C.from_dict({})
    assert e.value.path == "run"


@pytest.mark.parametrize("d, path", [
    ({"run": "t", "actr": {}}, "actr"),
    ({"run": "t", "actor": {"simz": 3}}, "actor.simz"),
    ({"run": "t", "actor": {"pcr": {"cheap": 3}}}, "actor.pcr.cheap"),
    ({"run": "t", "search": {"belief": {"deck": "x"}}}, "search.belief.deck"),
])
def test_unknown_keys_name_the_dotted_path(d, path):
    with pytest.raises(C.ConfigError, match=f"unknown key {path}") as e:
        C.from_dict(d)
    assert e.value.path == path


def test_unknown_key_suggests_the_near_one():
    with pytest.raises(C.ConfigError, match=r"did you mean actor\.sims\?"):
        C.from_dict({"run": "t", "actor": {"simz": 3}})


# ------------------------------------------------------------------------------------------------ conversion

def test_values_are_converted_to_the_field_type():
    c = C.from_dict({"run": "t", "learner": {"lr": "1e-4", "epochs": 1, "window_positions": "3e5",
                                             "min_new_positions": 2000.0},
                     "actor": {"explore": "false", "pcr": {"prob": "0.25", "by_kind": {"priority": 1}}},
                     "player": {"init": "null"}, "budget": {"max_hours": 2}})
    assert c.learner.lr == 1e-4 and isinstance(c.learner.lr, float)
    assert c.learner.epochs == 1.0 and isinstance(c.learner.epochs, float)
    assert c.learner.window_positions == 300000 and isinstance(c.learner.window_positions, int)
    assert c.learner.min_new_positions == 2000 and isinstance(c.learner.min_new_positions, int)
    assert c.actor.explore is False and c.actor.pcr.prob == 0.25 and c.actor.pcr.by_kind == {"priority": 1.0}
    assert c.player.init is None and c.budget.max_hours == 2.0


def test_learner_extras_default_off_and_convert():
    c = C.from_dict({"run": "t"})
    assert (c.learner.surprise_weight, c.learner.value_surprise, c.learner.ema_decay) == (0.0, 0.0, 0.0)
    assert (c.learner.window.kind, c.learner.window.c, c.learner.window.alpha, c.learner.window.beta) == \
        ("fixed", 150000, 0.75, 0.4)
    c = C.from_dict({"run": "t", "learner": {"surprise_weight": "0.5", "value_surprise": 0.1, "ema_decay": "0.999",
                                             "window": {"kind": "katago", "c": "1.5e5"}}})
    assert (c.learner.surprise_weight, c.learner.value_surprise, c.learner.ema_decay) == (0.5, 0.1, 0.999)
    assert c.learner.window.kind == "katago" and c.learner.window.c == 150000 and c.learner.window.alpha == 0.75
    with pytest.raises(C.ConfigError, match=r"did you mean learner\.window\.kind\?"):
        C.from_dict({"run": "t", "learner": {"window": {"kinds": "katago"}}})


@pytest.mark.parametrize("d, path", [
    ({"actor": {"sims": 1.5}}, "actor.sims"),
    ({"actor": {"sims": True}}, "actor.sims"),
    ({"actor": {"sims": "many"}}, "actor.sims"),
    ({"actor": {"explore": 1}}, "actor.explore"),
    ({"learner": {"lr": "fast"}}, "learner.lr"),
    ({"learner": {"lr": float("nan")}}, "learner.lr"),
    ({"eval": {"opponents": "prev"}}, "eval.opponents"),
    ({"eval": {"opponents": ["prev", 3]}}, "eval.opponents[1]"),
    ({"store": 5}, "store"),
    ({"actor": 5}, "actor"),
])
def test_wrong_types_name_the_path(d, path):
    with pytest.raises(C.ConfigError) as e:
        C.from_dict({"run": "t", **d})
    assert e.value.path == path


@pytest.mark.parametrize("d, path", [
    ({"mode": "lockstep"}, "mode"),
    ({"run": "has space"}, "run"),
    ({"player": {"arch": "cnn"}}, "player.arch"),
    ({"actor": {"sims": 0}}, "actor.sims"),
    ({"actor": {"pcr": {"prob": 1.5}}}, "actor.pcr.prob"),
    ({"actor": {"pcr": {"prob": 0.5, "by_kind": {"combat": 0.5}}}}, "actor.pcr.by_kind.combat"),
    ({"actor": {"pcr": {"prob": 0.5, "by_kind": {"priority": 2}}}}, "actor.pcr.by_kind.priority"),
    ({"actor": {"pcr": {"by_kind": {"priority": 0.5}}}}, "actor.pcr.by_kind"),
    ({"actor": {"sims": 30, "pcr": {"prob": 0.5, "cheap_sims": 40}}}, "actor.pcr.cheap_sims"),
    ({"learner": {"window_positions": 10, "min_new_positions": 20}}, "learner.window_positions"),
    ({"learner": {"policy_target": "q"}}, "learner.policy_target"),
    ({"learner": {"value_blend": 1.5}}, "learner.value_blend"),
    ({"learner": {"surprise_weight": 1.5}}, "learner.surprise_weight"),
    ({"learner": {"value_surprise": -0.1}}, "learner.value_surprise"),
    ({"learner": {"ema_decay": 1.0}}, "learner.ema_decay"),
    ({"learner": {"window": {"kind": "sqrt"}}}, "learner.window.kind"),
    ({"learner": {"window": {"c": 0}}}, "learner.window.c"),
    ({"learner": {"window": {"alpha": 0}}}, "learner.window.alpha"),
    ({"learner": {"window": {"beta": 1.5}}}, "learner.window.beta"),
    ({"eval": {"opponents": ["prev", "gnubg"]}}, "eval.opponents[1]"),
    ({"budget": {"max_versions": 0}}, "budget.max_versions"),
    ({"store": "hf://danbrooks/repo"}, "store"),
    ({"store": "ssh://host"}, "store"),
    ({"search": {"keys": ["autopay:x"]}}, "search.keys[0]"),
    ({"search": {"keys": ["autopay", "worlds=4"]}}, "search.keys[1]"),        # PolicySpec takes FLAG_KEYS only
    ({"search": {"keys": ["explore"]}}, "search.keys[0]"),
    ({"search": {"keys": ["autopay", "oppnodez"]}}, "search.keys[1]"),
    ({"engine": {"name": "xmage"}}, "engine.name"),
    ({"player": {"init": "hf://danbrooks/repo"}}, "player.init"),
    ({"player": {"init": "hf://buckets/danbrooks/dz-e1/x.pt"}}, "player.init"),
])
def test_checks_name_the_path(d, path):
    with pytest.raises(C.ConfigError) as e:
        C.from_dict(C._merge({"run": "t"}, d))
    assert e.value.path == path


@pytest.mark.parametrize("init", [None, "runs/x/model.pt", "hf://danbrooks/draftzero-v0.1.0@v1/player/model.pt"])
def test_player_init(init):
    assert C.from_dict({"run": "t", "player": {"init": init}}).player.init == init


def test_player_config_is_checked_and_converted_at_load():
    """player.config keys are the arch's (draftzero.nets.models.DEFAULTS), each value of its default's type:
    YAML reads 5e-2 as text, a typo or another arch's key is a load error, not a crash at the first training."""
    from draftzero.nets import models as M
    c = C.loads("run: x\nplayer: {config: {d: '96', emb_std: 5e-2, dropout: 1e-1, policy_layers: 3.0}}")
    assert c.player.config == {"d": 96, "emb_std": 0.05, "dropout": 0.1, "policy_layers": 3}
    assert all(type(v) is type(M.DEFAULTS["mlp"][k]) for k, v in c.player.config.items())
    for bad, path in (({"head_hiden": 128}, "player.config.head_hiden"), ({"heads": 4}, "player.config.heads"),
                      ({"d": 1.5}, "player.config.d"), ({"emb_std": "wide"}, "player.config.emb_std")):
        with pytest.raises(C.ConfigError) as e:
            C.from_dict({"run": "t", "player": {"config": bad}})
        assert e.value.path == path
    with pytest.raises(C.ConfigError, match="did you mean player.config.head_hidden"):
        C.from_dict({"run": "t", "player": {"config": {"head_hiden": 128}}})
    # a null drops a key (the arch's default): how a run that extends base/player.yml changes player.arch
    base = {"run": "t", "extends": "base/player.yml"}
    t = C.from_dict(C._merge(base, {"player": {"arch": "transformer", "config": {"card_d": None, "heads": 4}}}))
    assert t.player.config == {"d": 128, "ff": 256, "layers": 1, "d_emb": 32, "head_hidden": 128, "heads": 4}
    M.full_config("transformer", t.player.config)
    with pytest.raises(C.ConfigError) as e:
        C.from_dict(C._merge(base, {"player": {"arch": "transformer", "config": {"card_d": None, "heads": 5}}}))
    assert e.value.path == "player.config.heads"                         # d 128 is not a multiple of 5
    with pytest.raises(C.ConfigError) as e:
        C.from_dict(C._merge(base, {"player": {"arch": "transformer", "config": {"heads": 4}}}))
    assert e.value.path == "player.config.card_d"                        # mlp-only


def test_arch_defaults_are_read_without_importing_torch(monkeypatch):
    import sys
    from draftzero.nets import models as M
    monkeypatch.delitem(sys.modules, "draftzero.nets.models")
    assert C._arch_defaults.__wrapped__() == M.DEFAULTS and "draftzero.nets.models" not in sys.modules


def test_search_keys_are_the_engines_bare_flags():
    from draftzero.engine.specs import FLAG_KEYS
    assert C.from_dict({"run": "t", "search": {"keys": list(FLAG_KEYS)}}).search.keys == list(FLAG_KEYS)
    with pytest.raises(C.ConfigError, match="nonoise"):                   # the message lists them
        C.from_dict({"run": "t", "search": {"keys": ["fpu=0.1"]}})


def test_cheap_sims_above_sims_is_fine_while_pcr_is_off():
    c = C.from_dict({"run": "t", "actor": {"sims": 16}})
    assert c.actor.pcr.cheap_sims == 40 and c.actor.pcr.prob == 0


@pytest.mark.parametrize("opp", ["prev", "v0", "v12", "sbtactical", "bot", "nonet", "file:/x/v3.dzgw"])
def test_opponents(opp):
    assert C.from_dict({"run": "t", "eval": {"opponents": [opp]}}).eval.opponents == [opp]


# ------------------------------------------------------------------------------------------------ files, extends

def test_extends_merges_deep_and_chains(tmp_path):
    write(tmp_path / "configs/base/player.yml",
          "run: base\nactor: {sims: 200, pcr: {prob: 0.5, cheap_sims: 40, by_kind: {priority: 0.6}}}\n"
          "player: {config: {d: 64, ff: 128}}\neval: {opponents: [prev, sbtactical]}\n")
    write(tmp_path / "configs/mid.yml", "extends: base/player.yml\nrun: mid\nactor: {pcr: {by_kind: {target: 0.3}}}\n")
    leaf = write(tmp_path / "configs/runs/leaf.yml",
                 "extends: mid.yml\nrun: leaf\nactor: {sims: 300}\nplayer: {config: {d: 96}}\neval: {opponents: [bot]}\n")
    c = C.load(leaf)
    assert c.run == "leaf" and c.actor.sims == 300
    assert c.actor.pcr.prob == 0.5 and c.actor.pcr.by_kind == {"priority": 0.6, "target": 0.3}   # dicts merge
    assert c.player.config == {"d": 96, "ff": 128}
    assert c.eval.opponents == ["bot"]                                                           # lists replace


def test_an_empty_mapping_clears_a_free_dict(tmp_path):
    """{} for actor.pcr.by_kind or player.config clears it (a child config, an override, a section plan change);
    a non-empty mapping merges key by key; an empty section is no change."""
    write(tmp_path / "configs/base/arm.yml",
          "run: arm\nactor: {pcr: {prob: 0.5, cheap_sims: 40, by_kind: {priority: 0.65, target: 0.3}}}\n")
    leaf = write(tmp_path / "configs/runs/uniform.yml",
                 "extends: base/arm.yml\nrun: uniform\nactor: {pcr: {by_kind: {}}}\nbudget: {}\n")
    c = C.load(leaf)
    assert c.actor.pcr.by_kind == {} and c.actor.pcr.prob == 0.5
    assert C.load(tmp_path / "configs/base/arm.yml", {"actor.pcr.by_kind": {}}).actor.pcr.by_kind == {}
    arm = C.load(tmp_path / "configs/base/arm.yml")
    assert C.apply_change(arm, "actor.pcr", {"prob": 0.4, "by_kind": {}}).actor.pcr.by_kind == {}
    assert C.apply_change(arm, "actor.pcr", "{by_kind: {}}").actor.pcr.prob == 0.5
    assert C.apply_change(arm, "actor.pcr", {"by_kind": {"blockers": 0.2}}).actor.pcr.by_kind == \
        {"priority": 0.65, "target": 0.3, "blockers": 0.2}
    assert C.from_dict({"run": "t", "player": {"config": {}}}).player.config == {}


def test_extends_list_and_absolute(tmp_path):
    a = write(tmp_path / "a.yml", "run: a\nseed: 5\nactor: {sims: 7}\n")
    write(tmp_path / "b.yml", "actor: {sims: 9, workers: 3}\n")
    c = C.load(write(tmp_path / "c.yml", f"extends: [{a}, b.yml]\nrun: c\n"))
    assert (c.seed, c.actor.sims, c.actor.workers) == (5, 9, 3)


def test_extends_errors(tmp_path):
    with pytest.raises(C.ConfigError, match="extends: nope.yml not found"):
        C.load(write(tmp_path / "x.yml", "extends: nope.yml\nrun: x\n"))
    write(tmp_path / "p.yml", "extends: q.yml\nrun: p\n")
    with pytest.raises(C.ConfigError, match="extends cycle"):
        C.load(write(tmp_path / "q.yml", "extends: p.yml\nrun: q\n"))


def test_unknown_key_in_a_base_names_the_base_file(tmp_path):
    base = write(tmp_path / "base.yml", "run: b\nlearner: {lrr: 1}\n")
    with pytest.raises(C.ConfigError, match=r"base\.yml: unknown key learner\.lrr"):
        C.load(write(tmp_path / "run.yml", "extends: base.yml\nrun: r\n"))
    assert base.exists()


def test_load_errors_name_the_file(tmp_path):
    p = write(tmp_path / "bad.yml", "run: x\nactor: {sims: 0}\n")
    with pytest.raises(C.ConfigError, match=r"bad\.yml: actor\.sims"):
        C.load(p)
    with pytest.raises(C.ConfigError, match="a config is a YAML mapping"):
        C.load(write(tmp_path / "list.yml", "- 1\n"))


def test_overrides_dotted_or_nested(tmp_path):
    p = write(tmp_path / "c.yml", "run: x\nactor: {sims: 50, workers: 3}\n")
    c = C.load(p, {"actor.sims": "16", "learner": {"lr": 3e-4}, "eval.opponents": ["prev"], "player.init": "a.pt"})
    assert (c.actor.sims, c.actor.workers, c.learner.lr, c.eval.opponents, c.player.init) == (16, 3, 3e-4, ["prev"], "a.pt")
    with pytest.raises(C.ConfigError, match="unknown key actor.simz"):
        C.load(p, {"actor.simz": 1})


def test_dump_round_trips_and_hash_ignores_spelling(tmp_path):
    a = C.load(write(tmp_path / "a.yml", "run: x\nlearner: {lr: 1e-4, epochs: 2}\n"))
    b = C.load(write(tmp_path / "b.yml", "run: x\nlearner: {lr: 0.0001, epochs: 2.0}\n"))
    assert C.config_hash(a) == C.config_hash(b)
    text = C.dump(a)
    assert "extends" not in text and text.startswith("# run config x: config_hash ")
    assert C.loads(text) == a and C.config_hash(C.loads(text)) == C.config_hash(a)
    p = write(tmp_path / "dumped.yml", text)
    assert C.load(p) == a


def test_config_hash_is_16_hex_of_canonical_json(cfg):
    h = C.config_hash(cfg)
    assert len(h) == 16 and int(h, 16) >= 0
    import hashlib
    canon = json.dumps(C.to_dict(cfg), sort_keys=True, separators=(",", ":"))
    assert h == hashlib.sha256(canon.encode()).hexdigest()[:16]
    assert C.config_hash(dataclasses.replace(cfg, seed=2)) != h


def test_loads_resolves_extends_against_base_dir(tmp_path):
    write(tmp_path / "base/player.yml", "run: b\nactor: {sims: 33}\n")
    c = C.loads("extends: base/player.yml\nrun: s\n", base_dir=tmp_path)
    assert c.run == "s" and c.actor.sims == 33


def test_flatten_get_diff(cfg):
    f = C.flatten(cfg)
    assert f["actor.pcr.prob"] == 0.0 and f["actor.pcr.by_kind"] == {} and f["paused"] is False
    assert "actor.pcr" not in f and "actor" not in f
    assert C.get(cfg, "actor.sims") == 100 and C.get(cfg, "actor.pcr") == {"prob": 0.0, "cheap_sims": 40, "by_kind": {}}
    with pytest.raises(C.ConfigError, match="unknown key actor.simz"):
        C.get(cfg, "actor.simz")
    other = C.apply_change(cfg, "actor.sims", 200)
    assert C.diff(cfg, other) == {"actor.sims": (100, 200)}


def test_resolve_path(tmp_path):
    assert C.resolve_path(tmp_path) == tmp_path
    assert C.resolve_path("data/gorge/decks").name == "decks" and C.resolve_path("x").is_absolute()


# ------------------------------------------------------------------------------------------------ plan changes

def test_live_set():
    assert {"actor.sims", "actor.pcr.prob", "actor.pcr.cheap_sims", "actor.pcr.by_kind", "actor.record_every",
            "actor.chunk_games", "learner.lr", "learner.window_positions", "learner.min_new_positions",
            "learner.steps_per_position", "learner.epochs", "eval.every_versions", "eval.pairs", "eval.sims",
            "budget.max_versions", "budget.max_hours", "budget.max_games", "paused"} == C.LIVE
    flat = C.flatten(C.from_dict({"run": "t"}))
    assert C.LIVE <= set(flat)                            # every live path is a real field
    assert C.is_live("actor.pcr.by_kind.target") and C.is_live("budget") and C.is_live("actor.pcr")
    assert not C.is_live("actor") and not C.is_live("seed") and not C.is_live("actor.explore")


def test_apply_change_returns_a_new_config(cfg):
    new = C.apply_change(cfg, "learner.lr", "3e-4")
    assert new.learner.lr == 3e-4 and cfg.learner.lr == 1e-4
    assert C.apply_change(cfg, "paused", "true").paused is True
    assert C.apply_change(cfg, "budget.max_hours", 12).budget.max_hours == 12.0
    assert C.apply_change(cfg, "learner.steps_per_position", "0.5").learner.steps_per_position == 0.5
    assert C.apply_change(new, "learner.steps_per_position", "null").learner.steps_per_position is None


def test_apply_change_sections_and_dict_entries(cfg):
    c = C.apply_change(cfg, "actor.pcr", {"prob": 0.25})
    assert (c.actor.pcr.prob, c.actor.pcr.cheap_sims) == (0.25, 40)              # a section merges
    c = C.apply_change(c, "actor.pcr.by_kind.priority", 0.65)
    c = C.apply_change(c, "actor.pcr.by_kind.target", "0.3")
    assert c.actor.pcr.by_kind == {"priority": 0.65, "target": 0.3}
    c = C.apply_change(c, "actor.pcr.by_kind.target", None)
    assert c.actor.pcr.by_kind == {"priority": 0.65}
    c = C.apply_change(c, "budget", "{max_versions: 3}")
    assert c.budget.max_versions == 3


@pytest.mark.parametrize("path, value, match", [
    ("seed", 2, "seed is fixed"),
    ("actor.explore", False, "actor.explore is fixed"),
    ("player.arch", "gnn", "player.arch is fixed"),
    ("actor", {"sims": 3}, "actor is fixed"),
    ("actor.simz", 3, "unknown key actor.simz"),
    ("actor.sims", "lots", "actor.sims: expected an integer"),
    ("actor.sims", 0, "actor.sims: >= 1"),
    ("actor.sims.deeper", 1, "unknown key actor.sims.deeper"),
    ("actor.pcr.by_kind.combat", 0.5, "actor.pcr.by_kind.combat"),
    ("learner.window_positions", 10, "learner.window_positions"),
    ("learner.surprise_weight", 0.5, "learner.surprise_weight is fixed"),
    ("learner.ema_decay", 0.999, "learner.ema_decay is fixed"),
    ("learner.window.kind", "katago", "learner.window.kind is fixed"),
])
def test_apply_change_rejects(cfg, path, value, match):
    with pytest.raises(C.ConfigError, match=match):
        C.apply_change(cfg, path, value)


def test_effective_replays_set_events_in_order(cfg):
    events = [
        {"time": "t1", "kind": "set", "path": "learner.lr", "old": 1e-4, "new": 3e-4, "by": "dan", "at_version": 2},
        {"time": "t2", "kind": "restart", "machine": "r1"},
        {"time": "t3", "kind": "set", "path": "actor.sims", "old": 100, "new": 50, "by": "dan", "at_version": 5},
        {"time": "t4", "kind": "set", "path": "learner.lr", "old": 3e-4, "new": 1e-3, "by": "dan", "at_version": 5},
        {"time": "t5", "path": "eval.pairs", "old": 100, "new": 200},
    ]
    assert C.effective(cfg, []) == cfg
    v1 = C.effective(cfg, events, upto_version=1)
    assert (v1.learner.lr, v1.actor.sims, v1.eval.pairs) == (1e-4, 100, 200)    # no at_version: always in effect
    v4 = C.effective(cfg, events, upto_version=4)
    assert (v4.learner.lr, v4.actor.sims) == (3e-4, 100)
    end = C.effective(cfg, events)
    assert (end.learner.lr, end.actor.sims, end.eval.pairs) == (1e-3, 50, 200)
    assert cfg.learner.lr == 1e-4                                                 # the launched config is untouched


def test_effective_replays_without_the_live_check(cfg):
    c = C.effective(cfg, [{"kind": "set", "path": "seed", "new": 9}])
    assert c.seed == 9
    with pytest.raises(C.ConfigError):
        C.effective(cfg, [{"kind": "set", "path": "actor.sims", "new": "x"}])


def test_module_main_prints_the_dump(tmp_path, capsys):
    p = write(tmp_path / "c.yml", "run: m\n")
    assert C.main([str(p), "actor.sims=12"]) == 0
    out = capsys.readouterr().out
    assert "sims: 12" in out and out.startswith("# run config m")
