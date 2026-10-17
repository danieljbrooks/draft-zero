"""The deck builder on the loop (draftzero.loop.builder), engine-free: synthetic draftlab pools (real FDN card names,
drawn at random) and a fake Gorge whose games a hidden deck strength decides, so the Bradley-Terry learner has
something to learn. Also: with builder.learn off, the player loop's control.json, local-mode roles and status are
what they were."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from draftzero.loop import builder as bld
from draftzero.loop import control as ctl
from draftzero.loop import local
from draftzero.loop.controller import Controller, status_md
from draftzero.loop.store import LocalStore

torch = pytest.importorskip("torch")

from draftzero.build import cards as cardsmod  # noqa: E402
from draftzero.build import decks as D  # noqa: E402
from draftzero.build import pools as P  # noqa: E402
from draftzero.build.kinds import Builder  # noqa: E402

C = cardsmod.load()
NONBASIC = [n for n in C.names if n not in cardsmod.BASICS]


# ------------------------------------------------------------------------------------------------ fakes

def fake_pools(out, *, set="FDN", n, seed, packs=6, collation="arena", best23=False, first=0):
    """`dzgorge sealed-pools`' output for synthetic pools: 84 random FDN cards each, best23 a random build."""
    lines = []
    for i in range(first, first + n):
        rng = random.Random(f"{seed}-{i}")
        cards: dict[str, int] = {}
        for c in rng.choices(NONBASIC, k=14 * packs):
            cards[c] = cards.get(c, 0) + 1
        p = {"pool_id": f"{set}-{seed}-{i:06d}", "index": i, "set": set, "collation": "play-arena", "packs": packs,
             "base_seed": seed, "seed": rng.getrandbits(63), "cards": cards}
        if best23:
            d = D.random_build(P.pool_vector(p), np.random.default_rng([seed, i]))
            p["best23"] = {"builder": "best23@1",
                           "cards": [{"name": k, "count": v} for k, v in sorted(C.dict(d).items())]}
        lines.append(json.dumps(p))
    Path(out).write_text("".join(x + "\n" for x in lines))
    return Path(out)


def strength(cards: list[dict]) -> float:
    """The fake engine's hidden deck strength: 17 lands and many creatures win."""
    v = C.vec({c["name"]: c["count"] for c in cards})
    return -abs(float(v[C.is_land].sum()) - 17) + 0.3 * float(v[C.is_creature].sum())


class FakeGorge:
    """Gorge.play for -deckfile/-pairs-file/-single games: each line once, deck A in seat 0, the stronger deck
    winning more often; every 10th game a stall."""

    binary, cards = Path("/nonexistent/dzgorge"), Path("/nonexistent/.cards")

    def __init__(self):
        self.plays = []

    def play(self, a, b, *, pairs, seed, out_dir, deckfile=None, pairs_file=None, single=False, workers=2,
             belief=None, timeout_s=None, **kw):
        assert single and deckfile and pairs_file and a == b
        self.plays.append({"spec": a, "seed": seed, "belief": belief, "workers": workers, "deckfile": deckfile})
        decks = {json.loads(x)["name"]: json.loads(x)["cards"] for x in Path(deckfile).read_text().splitlines()}
        rows = []
        for i, line in enumerate(Path(pairs_file).read_text().splitlines()):
            da, db, s = line.split("\t")
            rng = random.Random(int(s) * 2 + i)
            starter = rng.randrange(2)
            row = {"game": i, "pair": i, "leg": 0, "seed": int(s), "decks": [da, db], "starter": starter,
                   "in_hand": [[], []], "turns": 10}
            if i % 10 == 9:
                row.update(winner=-1, result="stall")
            else:
                p0 = 1 / (1 + math.exp(-(strength(decks[da]) - strength(decks[db]))))
                w = 0 if rng.random() < p0 else 1
                row.update(winner=w, result="A" if w == 0 else "B")
            rows.append(row)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(games=rows, summary={}, skipped=[], corpus_path=None)


def make_cfg(tmp_path, **sections):
    base = {"run": "b", "seed": 3, "store": str(tmp_path / "store"), "player": {"learn": False},
            "eval": {"opponents": []}, "budget": {"max_versions": 2},
            "builder": {"learn": True, "kinds": "bp", "members": 1, "pools_per_chunk": 4, "games_per_chunk": 12,
                        "games_per_version": 10, "sims": 4, "epochs": 3, "threads": 1, "workers": 1,
                        "pools": {"count": 12}, "eval": {"pools": 4, "pairs": 1, "sims": 4, "workers": 1}}}

    def deep(a, b):
        for k, v in b.items():
            a[k] = deep(dict(a.get(k) or {}), v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
        return a
    p = tmp_path / f"cfg-{len(list(tmp_path.glob('cfg-*.yml')))}.yml"
    p.write_text(yaml.safe_dump(deep(base, sections)))
    from draftzero import config
    return config.load(p)


class Run:
    """The controller and the builder's roles in one process on one LocalStore, stepped by hand."""

    def __init__(self, tmp_path, cfg):
        self.store, self.cfg, self.log = LocalStore(tmp_path / "store"), cfg, []
        self.gorge = FakeGorge()

        def manifest(s, c):
            s.write_json(ctl.MANIFEST, {"format": "draftzero-manifest/1", "run": ctl.get(c, "run")})
        say = self.log.append
        self.controller = Controller(self.store, cfg, log=say, status_every_s=0, hb_cache_s=0, write_manifest=manifest)
        self.learner = bld.BuilderLearner(self.store, cfg, "bl", tmp_path / "w" / "bl", log=say)
        self.actor = bld.BuilderActor(self.store, cfg, "ba", tmp_path / "w" / "ba", gorge=self.gorge, log=say,
                                      make_pools=fake_pools)
        self.evaluator = bld.BuilderEvaluator(self.store, cfg, "be", tmp_path / "w" / "be", gorge=self.gorge,
                                              log=say, make_pools=fake_pools)

    def control(self):
        return ctl.read(self.store)


# ------------------------------------------------------------------------------------------------ pieces

def test_paths_jobs_and_match_lines():
    assert bld.shard_dir("pod1", 3) == "builder/shards/pod1/000003"
    assert bld.parse_shard("builder/shards/pod1/000003/meta.json") == ("pod1", 3)
    assert bld.parse_shard("shards/pod1/000003/meta.json") is None
    assert ctl.parse_shard("builder/shards/pod1/000003") is None           # the player's learner never sees them
    assert bld.job_id(3) == "builder-v0003-vs-best23"
    lines = bld.match_lines(4, 4, 2, 9)
    assert len(lines) == 4 * 2 * 2                                          # 4 pools x 2 deals x 2 seat orders
    assert {(a, b) for a, b, _ in lines} == {("m4", "h5"), ("h5", "m4"), ("m5", "h4"), ("h4", "m5"),
                                             ("m6", "h7"), ("h7", "m6"), ("m7", "h6"), ("h6", "m7")}
    assert lines[0][2] == lines[1][2] != lines[2][2]                        # both seat orders share a deal
    assert bld.match_lines(4, 4, 2, 9) == lines and bld.deal_seed(9, 4, 0) == lines[0][2]


def test_score_match():
    g = lambda a, b, w, r=None: {"decks": [a, b], "winner": w, "result": r or ("A" if w == 0 else "B")}  # noqa
    games = [g("m0", "h1", 0), g("h1", "m0", 0), g("m1", "h0", 1), g("h0", "m1", -1, "stall"),
             g("m2", "h3", 0), g("h3", "m2", 1), g("m3", "h2", 0), g("h2", "m3", -1, "error")]
    s = bld.score_match(games)
    assert (s["wins"], s["losses"], s["stalls"], s["errors"], s["n_games"], s["n_pairs"]) == (4, 2, 1, 1, 7, 2)
    assert s["score"] == pytest.approx((0.375 + 1.0) / 2) and s["decided_winrate"] == pytest.approx(4 / 6)
    assert s["lo"] <= s["score"] <= s["hi"]
    assert bld.score_match([])["score"] is None


def test_pools_cache_and_vectors(tmp_path):
    calls = []

    def make(out, **kw):
        calls.append(kw)
        return fake_pools(out, **kw)
    a = P.cached(None, tmp_path, make=make, n=5, seed=4, best23=True)
    b = P.cached(None, tmp_path, make=make, n=5, seed=4, best23=True)
    assert a == b and len(calls) == 1 and (tmp_path / "FDN-arena-p6-s4-n5-best23.jsonl").exists()
    v = P.pool_vector(a[0])
    assert v.sum() == 84 and v[[C.index[x] for x in cardsmod.BASICS]].sum() == 0
    assert P.deck_vector(a[0]["best23"]["cards"]).sum() == 40
    with pytest.raises(KeyError, match="outside"):
        P.pool_vector({"pool_id": "x", "cards": {"Black Lotus": 1}})
    (tmp_path / "FDN-arena-p6-s4-n5-best23.jsonl").write_text(json.dumps(a[0]) + "\n")
    with pytest.raises(ValueError, match="want 5"):
        P.cached(None, tmp_path, make=make, n=5, seed=4, best23=True)
    argv = P.argv(Path("/g/bin/dzgorge"), Path("/g/.cards"), Path("o.jsonl"), n=3, seed=2, best23=True)
    assert argv[:2] == ["/g/bin/dzgorge", "sealed-pools"] and argv[-1] == "-best23" and "-collation" in argv


def test_kinds(tmp_path):
    pool = P.pool_vector(json.loads(fake_pools(tmp_path / "p.jsonl", n=1, seed=1).read_text()))
    rng = np.random.default_rng(0)
    untrained = Builder(None).decks(pool, "btplr", rng)
    assert sorted(untrained) == ["r0", "r1", "r2", "r3"]
    from draftzero.build.model import DeckNet, Ensemble
    torch.manual_seed(0)
    ens = Ensemble([DeckNet().eval(), DeckNet().eval()], {"lands": [16, 18]})
    out = Builder(ens).decks(pool, "btplrck", rng, random_share=0.5)
    assert sorted(out) == sorted("btplrck")
    for kind, d in out.items():
        assert d.sum() == 40, kind
        spells = d.copy()
        spells[[C.index[x] for x in cardsmod.BASICS]] = 0
        assert (spells <= pool).all(), kind
    assert 16 <= out["b"][C.is_land].sum() <= 18
    with pytest.raises(ValueError, match="deck kinds"):
        Builder(ens).decks(pool, "bx", rng)


def test_pilot_spec_and_sims(tmp_path):
    cfg = make_cfg(tmp_path)
    assert bld.pilot_spec(cfg, 0, None) == "bot"
    spec = bld.pilot_spec(cfg, 16, None)
    assert spec.sims == 16 and not spec.closedlist and not spec.explore and spec.gonet is None and spec.topk is None
    spec = bld.pilot_spec(make_cfg(tmp_path, builder={"closedlist": True}), 16, Path("/n/x.dzgw"))
    assert spec.closedlist and spec.gonet == "/n/x.dzgw" and spec.topk == 6
    assert bld.pilot_sims(make_cfg(tmp_path, builder={"sims": None}), {"sims": 33}) == 33
    assert bld.pilot_sims(cfg, {"sims": 33}) == 4


# ------------------------------------------------------------------------------------------------ the loop

def test_builder_loop_end_to_end(tmp_path):
    cfg = make_cfg(tmp_path)
    r = Run(tmp_path, cfg)
    c = r.controller.tick()
    assert c["version"] == {"player": None, "builder": None} and c["builder_evals"] == []
    assert r.actor.step() == "wait"                                         # no builder version yet
    assert r.learner.step() == "wait"                                       # publishes v0, untrained
    v0 = r.store.read_json("versions/builder/v0000/meta.json")
    assert v0["net"] is False and r.store.read_json("versions/builder/latest.json")["version"] == 0
    assert r.controller.tick()["version"]["builder"] == 0

    while r.actor.step() == "played":                                       # until v0 has 10 decided games
        pass
    metas = [r.store.read_json(p) for p in r.store.list("builder/shards/") if p.endswith("meta.json")]
    assert metas and sum(m["decided"] for m in metas) >= 10 and all(m["version"] == 0 for m in metas)
    decks = [json.loads(x) for x in r.store.read_bytes(f"{bld.shard_dir('ba', 1)}/decks.jsonl").decode().splitlines()]
    assert len(decks) == 4 * 4 and {d["kind"] for d in decks} == {"r0", "r1", "r2", "r3"}
    assert all(sum(e["count"] for e in d["cards"]) == 40 for d in decks)
    games = r.store.read_bytes(f"{bld.shard_dir('ba', 1)}/games.jsonl").decode().splitlines()
    assert len(games) == 10                                                 # cut to the version's 10 games
    assert r.gorge.plays[0]["spec"].sims == 4 and r.gorge.plays[0]["belief"] is None
    assert r.store.list("metrics/builder-actor/ba/") and r.store.read_json("heartbeats/ba.json")["role"] == \
        "builder-actor"
    assert r.actor.step() == "wait"                                         # the version has its games

    assert r.learner.step() == "published"
    v1 = r.store.read_json("versions/builder/v0001/meta.json")
    assert v1["net"] and v1["parent"] == 0 and v1["members"] == 1 and v1["decided"] >= 10 and v1["lands"]
    assert r.store.exists("versions/builder/v0001/ens.pt") and r.store.list("metrics/builder-learner/bl/")
    ing = r.store.read_json(bld.INGESTED)
    assert ing and all(e["used"] == 1 for e in ing)

    assert r.evaluator.step() == "wait"                                     # heartbeat: a live evaluator
    c = r.controller.tick()
    assert c["version"]["builder"] == 1 and [j["job"] for j in c["builder_evals"]] == ["builder-v0001-vs-best23"]
    job = c["builder_evals"][0]
    assert job["pools"] == 4 and job["player_version"] is None and all(s["machine"] == "be" for s in job["shards"])
    assert r.evaluator.step() == "played"
    summary = r.store.read_json("evals/builder-v0001-vs-best23/summary.json")
    assert summary["component"] == "builder" and summary["n_games"] > 0 and summary["score"] is not None
    lines = [json.loads(x) for p in r.store.list("metrics/builder-evaluator/") for x in
             r.store.read_bytes(p).decode().splitlines()]
    assert [(x["name"], x["opponent"], x["model"], x["version"]) for x in lines] == \
        [("eval.score", "best23", "builder", 1)]

    while r.actor.step() == "played":                                       # v1's decks: kinds b and p
        pass
    kinds = {json.loads(x)["kind"] for p in r.store.list("builder/shards/ba/") if p.endswith("decks.jsonl")
             for x in r.store.read_bytes(p).decode().splitlines()}
    assert {"b", "p"} <= kinds
    assert r.learner.step() == "published"
    v2 = r.store.read_json("versions/builder/v0002/meta.json")
    assert v2["init"] == "v0001" and set(v2["shards"]) >= {m["path"] for m in
                                                           r.actor.index.metas.values() if m["version"] == 1}
    c = r.controller.tick()                                                 # max_versions 2: drain, eval v2 first
    assert c["paused"] and not c["stop"] and c["builder_evals"][0]["version"] == 2
    assert r.actor.step() == "paused"
    assert r.evaluator.step() == "played"
    c = r.controller.tick()
    assert c["stop"] and c["stop_reason"] == "budget.max_versions 2"
    st = r.store.read_json(ctl.STATUS)
    assert st["builder"]["version"] == 2 and st["builder"]["latest"]["version"] == 2
    assert "**Builder:** builds with version 2" in status_md(st)
    assert {s["job"] for s in st["closed_evals"]} == {"builder-v0001-vs-best23", "builder-v0002-vs-best23"}
    assert r.learner.step() == "stop" and r.actor.step() == "stop"


def test_actor_pilot_follows_the_player_or_the_fixed_net(tmp_path):
    cfg = make_cfg(tmp_path, player={"learn": True})
    r = Run(tmp_path, cfg)
    r.controller.tick()
    r.learner.step()
    r.controller.tick()
    assert r.actor.step() == "wait" and "player learner" in r.log[-1]       # no player version 0 yet
    pv = {"version": 0, "parent": None, "net": False, "sha256": {}}
    r.store.put_many({ctl.version_file(0, "meta.json"): json.dumps(pv).encode(),
                      ctl.latest_path(): json.dumps({"version": 0, "path": ctl.version_dir(0)}).encode()})
    c = r.controller.tick()
    assert c["version"] == {"player": 0, "builder": 0}
    assert r.actor.step() == "played"
    meta = r.store.read_json(f"{bld.shard_dir('ba', 1)}/meta.json")
    assert meta["player_version"] == 0 and "gonet" not in meta["spec"]

    net = tmp_path / "fixed.dzgw"
    net.write_bytes(b"DZGW")
    cfg2 = make_cfg(tmp_path, player={"learn": True}, builder={"pilot": str(net)})
    a = bld.BuilderActor(r.store, cfg2, "ba2", tmp_path / "w" / "ba2", gorge=r.gorge, make_pools=fake_pools,
                         log=r.log.append)
    assert a.step() == "played"
    assert r.gorge.plays[-1]["spec"].gonet == str(net)
    assert r.store.read_json(f"{bld.shard_dir('ba2', 1)}/meta.json")["player_version"] is None


def test_builder_learner_never_republishes_version_zero_over_a_run_in_progress(tmp_path):
    cfg = make_cfg(tmp_path)
    st = LocalStore(tmp_path / "s")
    st.write_json(ctl.version_file(2, "meta.json", bld.COMPONENT), {"version": 2})
    lr = bld.BuilderLearner(st, cfg, "bl", tmp_path / "w")
    with pytest.raises(RuntimeError, match="refusing to publish version 0"):
        lr.bootstrap()
    assert st.read_json(ctl.latest_path(bld.COMPONENT)) is None


def test_learner_window_by_versions_and_by_games(tmp_path):
    cfg = make_cfg(tmp_path, builder={"window": {"kind": "versions", "size": 2}})
    lr = bld.BuilderLearner(LocalStore(tmp_path / "s"), cfg, "bl", tmp_path / "w")
    lr.ingested = [{"shard": f"s{i}", "actor": "a", "seq": i, "version": v, "decided": 10, "time": f"t{i}"}
                   for i, v in enumerate([0, 0, 1, 2, 2, 3])]
    assert [e["seq"] for e in lr.window(3)] == [3, 4, 5]
    lr.cfg = make_cfg(tmp_path, builder={"window": {"kind": "games", "size": 25}})
    assert [e["seq"] for e in lr.window(3)] == [3, 4, 5]
    lr.cfg = make_cfg(tmp_path, builder={"window": {"kind": "games", "size": 5}})
    assert [e["seq"] for e in lr.window(3)] == [5]


def test_builder_only_plan_changes_wait_for_builder_versions(tmp_path):
    cfg = make_cfg(tmp_path)
    r = Run(tmp_path, cfg)
    r.controller.tick()
    ctl.request_change(r.store, "actor.sims", 9, "test")
    r.controller.tick()
    assert ctl.read_events(r.store) == []                                   # no version boundary yet
    r.learner.step()                                                        # builder v0
    r.controller.tick()
    ev = ctl.read_events(r.store)
    assert [(e["path"], e["new"], e["at_version"]) for e in ev] == [("actor.sims", 9, 0)]


def test_a_builder_only_restart_records_its_config_changes_at_the_builder_version(tmp_path):
    """A builder-only run's plan changes count builder versions, so a restart's allowed changes do too."""
    cfg = make_cfg(tmp_path)
    r = Run(tmp_path, cfg)
    r.controller.tick()
    c = ctl.read(r.store)
    c["version"]["builder"] = 3
    ctl.write(r.store, c, "test: builder version 3")
    wide = make_cfg(tmp_path, builder={"window": {"kind": "games", "size": 50}, "warm_start": False})
    Controller(r.store, wide, log=lambda *_: None, write_manifest=lambda *a: None, allow_config_change=True).tick()
    ev = ctl.read_events(r.store)
    assert sorted((e["path"], e["new"], e["at_version"]) for e in ev) == [
        ("builder.warm_start", False, 3), ("builder.window.kind", "games", 3), ("builder.window.size", 50, 3)]


def test_builder_role_refuses_a_run_without_the_builder(tmp_path):
    from draftzero.loop import cli
    p = tmp_path / "c.yml"
    p.write_text(yaml.safe_dump({"run": "x", "store": str(tmp_path / "s")}))
    assert cli.main(["builder-actor", "--config", str(p), "--machine", "m", "--work-dir", str(tmp_path / "w")]) == 2


# ------------------------------------------------------------------------------------------------ builder off

def test_builder_off_leaves_the_player_loop_alone(tmp_path):
    from draftzero import config
    p = tmp_path / "c.yml"
    p.write_text(yaml.safe_dump({"run": "p", "store": str(tmp_path / "store")}))
    cfg = config.load(p)
    assert not cfg.builder.learn and cfg.player.learn
    store = LocalStore(tmp_path / "store")
    ctrl = Controller(store, cfg, log=lambda s: None, status_every_s=0, hb_cache_s=0,
                      write_manifest=lambda s, c: None)
    c = ctrl.tick()
    assert c["version"] == {"player": None} and "builder_evals" not in c
    assert not store.list("builder/") and not store.exists(ctl.latest_path(bld.COMPONENT))
    st = store.read_json(ctl.STATUS)
    assert "builder" not in st and "Builder" not in status_md(st)
    assert set(c) == set(ctl.new_control(cfg)) | {"started"}
    cmds = local.role_commands("c.yml", "s", tmp_path, "m", 2)
    assert list(cmds) == ["controller", "learner", "evaluator", "actor-0", "actor-1"]
    assert local.role_commands("c.yml", "s", tmp_path, "m", 2, player=True, builder=False, builder_actors=3) == cmds
    both = local.role_commands("c.yml", "s", tmp_path, "m", 1, builder=True, builder_actors=2)
    assert list(both) == ["controller", "learner", "evaluator", "actor-0", "builder-learner", "builder-evaluator",
                          "builder-actor-0", "builder-actor-1"]
    only = local.role_commands("c.yml", "s", tmp_path, "m", 1, player=False, builder=True)
    assert list(only) == ["controller", "builder-learner", "builder-evaluator", "builder-actor-0"]
    assert only["builder-actor-0"][3:5] == ["builder-actor", "--config"]


def test_player_ratings_leave_builder_evaluations_out():
    from draftzero.plots import standard as st
    line = {"name": "eval.score", "version": 1, "n_games": 100}
    player = [{**line, "value": 0.6, "opponent": "sbtactical"}]
    builder = [{**line, "value": 0.7, "opponent": "best23", "model": "builder"}]
    assert st.ratings_from_metrics(player + builder) == st.ratings_from_metrics(player)
    assert "best23" not in st.ratings_from_metrics(player + builder)


def test_a_player_actor_idles_when_the_player_does_not_learn(tmp_path):
    from draftzero.loop.actor import Actor
    cfg = make_cfg(tmp_path)
    r = Run(tmp_path, cfg)
    r.controller.tick()
    a = Actor(r.store, cfg, "pa", tmp_path / "w" / "pa", gorge=r.gorge, log=r.log.append)
    assert a.step() == "wait" and "player.learn is false" in r.log[-1] and not r.gorge.plays
