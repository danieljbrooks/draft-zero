"""The actor-learner loop (draftzero.loop): control.json and the store layout, the actor, learner, evaluator and
controller, plan changes, budgets, sync mode and the command line, engine-free.

gorge is a FakeGorge: play() writes games.jsonl rows and a stand-in corpus, pack() writes synthetic dzg-pack-1
shards (draftzero.nets.packfmt), so the real draftzero.nets.train can run on them. Until the v0.2.0 modules other
agents write are merged (draftzero.config, draftzero.engine, draftzero.eval.paired, draftzero.metrics.MetricsLog),
small stand-ins following contracts §2-3 and §5 replace whichever of them is missing; once they exist, the real
ones are used.
"""
from __future__ import annotations

import copy
import dataclasses
import gzip
import hashlib
import importlib
import json
import math
import random
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from draftzero.loop import control as ctl
from draftzero.loop.actor import Actor, policy_spec
from draftzero.loop.controller import Controller, status_md
from draftzero.loop.evaluator import Evaluator, job_id, make_job, opponent_valid
from draftzero.loop.learner import LeaseHeld, Learner
from draftzero.loop.store import LocalStore

# ------------------------------------------------------------------------------------------------ stand-ins

DEFAULTS = {
    "run": "smoke", "code": "dev", "seed": 1, "mode": "async", "paused": False,
    "budget": {"max_versions": None, "max_hours": None, "max_games": None},
    "engine": {"name": "gorge", "root": "/work/gorge-fork", "census": True, "max_turns": 100, "max_intents": 20000,
               "mulligans": 2, "mull_heuristic": True, "topk_max_options": 64},
    "data": {"decks": "data/gorge/decks", "pool": "data/gorge/pool.tsv", "train_split": "train", "eval_split": "eval"},
    "player": {"arch": "mlp",
               "config": {"d": 128, "ff": 256, "layers": 1, "d_emb": 32, "card_d": 64, "head_hidden": 128},
               "init": None},
    "search": {"keys": ["autopay", "oppnodes", "oppfull", "morekinds"], "topk": 6, "cpuct": 0.25, "closedlist": True,
               "belief": {"decks": None, "pool": None, "split": "train"}},
    "actor": {"sims": 100, "explore": True, "record_every": 1, "chunk_games": 50, "chunk_secs": 3600, "workers": 2,
              "pcr": {"prob": 0.0, "cheap_sims": 40, "by_kind": {}}},
    "learner": {"device": "cpu", "min_new_positions": 20000, "window_positions": 300000, "epochs": 2,
                "steps_per_position": None, "batch": 512, "lr": 1.0e-4, "warmup": 100, "policy_target": "visits",
                "value_blend": 1.0, "warm_start": True, "threads": 4},
    "eval": {"every_versions": 1, "sims": 100, "pairs": 100, "seed": 3131, "workers": 2,
             "opponents": ["prev", "sbtactical"]},
    "store": "runs/smoke", "tracking": {"trackio": False, "project": "draftzero"},
    "sync": {"games_per_version": 200},
}
LIVE = frozenset({"actor.sims", "actor.pcr.*", "actor.record_every", "actor.chunk_games", "learner.lr",
                  "learner.window_positions", "learner.min_new_positions", "learner.steps_per_position",
                  "learner.epochs", "eval.every_versions", "eval.pairs", "eval.sims", "budget.*", "paused"})
LEAF_DICTS = {"player.config", "actor.pcr.by_kind"}


def _fake_config():
    m = types.ModuleType("draftzero.config")

    class ConfigError(ValueError):
        pass

    def merge(base, over, where=""):
        out = copy.deepcopy(base)
        for k, v in (over or {}).items():
            if k not in base:
                raise ConfigError(f"unknown key {where}{k}")
            sub = f"{where}{k}"
            if isinstance(base[k], dict) and sub not in LEAF_DICTS and isinstance(v, dict):
                out[k] = merge(base[k], v, sub + ".")
            else:
                out[k] = v
        return out

    def build(d, where=""):
        return SimpleNamespace(**{k: build(v, f"{where}{k}.") if isinstance(v, dict) and f"{where}{k}" not in LEAF_DICTS
                                  else copy.deepcopy(v) for k, v in d.items()})

    def to_dict(cfg):
        return {k: to_dict(v) if isinstance(v, SimpleNamespace) else copy.deepcopy(v) for k, v in vars(cfg).items()}

    def load(path, overrides=None):
        return build(merge(DEFAULTS, yaml.safe_load(Path(path).read_text()) or {}))

    def live(path):
        return path in LIVE or any(p.endswith(".*") and path.startswith(p[:-1]) for p in LIVE)

    def apply_change(cfg, path, value):
        if not live(path):
            raise ConfigError(f"{path} is not a live field")
        d = to_dict(cfg)
        cur, keys = d, path.split(".")
        for k in keys[:-1]:
            cur = cur[k]
        if keys[-1] not in cur:
            raise ConfigError(f"unknown key {path}")
        old = cur[keys[-1]]
        if old is not None and value is not None and not isinstance(value, type(old)) \
                and not (isinstance(old, float) and isinstance(value, int)):
            raise ConfigError(f"{path}: want {type(old).__name__}, got {value!r}")
        cur[keys[-1]] = value
        return build(d)

    def effective(cfg, events, upto_version=None):
        for e in events:
            if e.get("kind") == "set" and (upto_version is None or (e.get("at_version") or 0) <= upto_version):
                cfg = apply_change(cfg, e["path"], e["new"])
        return cfg

    m.ConfigError, m.LIVE, m.load, m.to_dict = ConfigError, LIVE, load, to_dict
    m.apply_change, m.effective = apply_change, effective
    m.dump = lambda cfg: yaml.safe_dump(to_dict(cfg), sort_keys=False)
    m.config_hash = lambda cfg: hashlib.sha256(json.dumps(to_dict(cfg), sort_keys=True).encode()).hexdigest()[:16]
    return m


@dataclasses.dataclass(frozen=True)
class FakeSpec:
    kind: str = "az"
    sims: int | None = 100
    cpuct: float | None = None
    topk: int | None = None
    explore: bool = False
    closedlist: bool = False
    gonet: str | None = None
    pcr: float | None = None
    cheap: int | None = None
    pcrk: tuple = ()
    keys: tuple = ("autopay", "oppnodes", "oppfull", "morekinds")
    extra: tuple = ()

    def render(self) -> str:
        parts = [self.kind] + ([f"sims={self.sims}"] if self.sims else []) + (["explore"] if self.explore else [])
        parts += list(self.keys) + (["closedlist"] if self.closedlist else [])
        parts += [f"{k}={v}" for k, v in (("cpuct", self.cpuct), ("topk", self.topk), ("pcr", self.pcr),
                                          ("cheap", self.cheap), ("gonet", self.gonet)) if v is not None]
        return ":".join(parts)


def _fake_engine():
    eng = types.ModuleType("draftzero.engine")
    eng.__path__ = []
    specs = types.ModuleType("draftzero.engine.specs")
    specs.PolicySpec = FakeSpec
    gorge = types.ModuleType("draftzero.engine.gorge")

    @dataclasses.dataclass
    class Belief:
        decks: Path
        pool: Path
        split: str = "train"
    gorge.Belief = Belief
    return {"draftzero.engine": eng, "draftzero.engine.specs": specs, "draftzero.engine.gorge": gorge}


def _fake_paired():
    m = types.ModuleType("draftzero.eval.paired")

    def score_pairs(games, *, hero_side="A"):
        by, out = {}, {"wins": 0, "losses": 0, "draws": 0, "stalls": 0, "errors": 0}
        for g in games:
            r = g.get("result")
            if r == "error":
                out["errors"] += 1
                continue
            s = 1.0 if r == hero_side else 0.5 if r in ("draw", "stall") else 0.0
            out["wins" if s == 1 else "losses" if s == 0 else "draws" if r == "draw" else "stalls"] += 1
            by.setdefault(g["pair"], []).append(s)
        means = [sum(v) / len(v) for v in by.values() if len(v) == 2]
        n = len(means)
        mu = sum(means) / n if n else None
        sd = math.sqrt(sum((x - mu) ** 2 for x in means) / (n - 1)) if n > 1 else float("nan")
        return {**out, "score": mu, "se": sd / math.sqrt(n) if n > 1 else float("nan"), "n_pairs": n,
                "n_games": 2 * n}

    def sprt(w, d, l, p0=.5, p1=.55, alpha=.05, beta=.05):
        n = w + d + l
        if not n:
            return None
        m_ = (w + .5 * d) / n
        var = max((w * (1 - m_) ** 2 + d * (.5 - m_) ** 2 + l * m_ ** 2) / n, 1e-6)
        llr = n * (p1 - p0) * (2 * m_ - p0 - p1) / (2 * var)
        return "H1" if llr >= math.log((1 - beta) / alpha) else "H0" if llr <= math.log(beta / (1 - alpha)) else None
    m.score_pairs, m.sprt = score_pairs, sprt
    return m


class FakeMetricsLog:
    def __init__(self, path, *, run, component, machine=None):
        self.path, self.base = Path(path), {"run": run, "component": component, "machine": machine}

    def log(self, name, value, *, version=None, step=None, n=None, lo=None, hi=None, **tags):
        line = {"schema": "draftzero-metrics/1", "time": ctl.now_iso(), **self.base, "name": name, "value": value,
                "version": version, "step": step, "n": n, "lo": lo, "hi": hi, **tags}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(line) + "\n")
        return line


def _missing(name: str) -> bool:
    try:
        importlib.import_module(name)
        return False
    except ImportError:
        return True


@pytest.fixture(autouse=True)
def stand_ins(monkeypatch):
    if _missing("draftzero.config"):
        monkeypatch.setitem(sys.modules, "draftzero.config", _fake_config())
    if _missing("draftzero.engine.specs"):
        for k, v in _fake_engine().items():
            monkeypatch.setitem(sys.modules, k, v)
    if _missing("draftzero.eval.paired"):
        monkeypatch.setitem(sys.modules, "draftzero.eval.paired", _fake_paired())
    import draftzero.metrics
    if not hasattr(draftzero.metrics, "MetricsLog"):
        monkeypatch.setattr(draftzero.metrics, "MetricsLog", FakeMetricsLog, raising=False)


# ------------------------------------------------------------------------------------------------ fakes

class FakeGorge:
    """Gorge.play / Gorge.pack (contracts §2) without the engine. Self-play (a == b): `pairs` games, each with
    `records` recorded decisions; a match: two legs a pair, side A winning with probability `p_a`."""

    def __init__(self, records: int = 5, p_a: float = 0.6, fail_play: bool = False):
        self.records, self.p_a, self.fail_play = records, p_a, fail_play
        self.plays, self.packs = [], []

    def play(self, a, b, *, pairs, seed, out_dir, split="train", first_pair=0, workers=2, record=False,
             record_every=1, belief=None, timeout_s=None, **kw):
        self.plays.append({"a": a, "b": b, "pairs": pairs, "seed": seed, "split": split, "first_pair": first_pair,
                           "workers": workers, "record": record, "record_every": record_every, "belief": belief,
                           "timeout_s": timeout_s})
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed)
        games = []
        if self.fail_play:
            pass
        elif a == b:
            for i in range(pairs):
                games.append({"game": i, "pair": first_pair + i, "leg": 0, "seed": seed, "side": ["A", "A"],
                              "result": rng.choice(["A", "B"]), "visits": self.records if record else 0})
        else:
            for p in range(first_pair, first_pair + pairs):
                for leg in (0, 1):
                    games.append({"game": 2 * p + leg, "pair": p, "leg": leg, "seed": seed,
                                  "side": ["A", "B"] if leg == 0 else ["B", "A"],
                                  "result": "A" if rng.random() < self.p_a else "B", "visits": 0})
        gp = out_dir / "games.jsonl"
        gp.write_text("".join(json.dumps(g) + "\n" for g in games))
        corpus = None
        if record:
            corpus = out_dir / "visits.jsonl.gz"
            with gzip.open(corpus, "wt") as f:
                for g in games:
                    for k in range(g["visits"]):
                        f.write(json.dumps({"game": f"p{g['pair']:07d}g{g['leg']}", "k": k}) + "\n")
        n = sum(g["visits"] for g in games)
        return SimpleNamespace(games=games, summary={"az_ms_per_searched": 12.5, "az_decisions_searched": n},
                               games_path=gp, corpus_path=corpus, skipped=[])

    def pack(self, corpus, out_dir, *, shard=200_000, game_prefix=None, shrink=True):
        from draftzero.nets import packfmt
        n = sum(1 for c in corpus for _ in gzip.open(c, "rt"))
        self.packs.append({"corpus": [str(c) for c in corpus], "n": n, "game_prefix": game_prefix})
        dirs = packfmt.write_synthetic(out_dir, n, seed=len(self.packs), shard_size=shard, game_len=2)
        for d in dirs:
            meta = json.loads((d / "meta.json").read_text())
            meta["games"] = [f"{game_prefix}{g}" for g in meta["games"]]
            (d / "meta.json").write_text(json.dumps(meta))
        return dirs


class Recording(LocalStore):
    """A LocalStore that remembers every put_many (its paths, in order)."""

    def __init__(self, root):
        super().__init__(root)
        self.puts: list[list[str]] = []
        self.fail = 0

    def put_many(self, files, message=""):
        if self.fail:
            self.fail -= 1
            raise ConnectionError("store unreachable")
        self.puts.append(list(files))
        super().put_many(files, message)


def fake_train(calls: list):
    """nets.train.main's outputs (best.pt, curves.jsonl) without training; records argv."""
    def train(argv):
        calls.append(argv)
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "best.pt").write_bytes(f"weights {len(calls)}".encode())
        ev = {"holdout": {"n_policy": 40, "policy_ce": 1.2 - 0.1 * len(calls), "policy_ce_uniform": 1.6,
                          "top1_visits": 0.5, "n_value": 40, "value_logloss": 0.65, "value_auc": 0.7, "total": 1.85}}
        (out / "curves.jsonl").write_text(json.dumps({"type": "train", "step": 3, "policy_ce": 1.0}) + "\n"
                                          + json.dumps({"type": "eval", "step": 3, "eval": ev}) + "\n")
        return {"steps": 3}
    return train


def fake_export(pt: Path, dzgw: Path) -> None:
    dzgw.write_bytes(b"DZGW0001" + Path(pt).read_bytes())


def make_cfg(tmp_path, **sections):
    """A run config: the test defaults below, with `sections` merged in (nested dicts merge)."""
    base = {"run": "t", "seed": 7,
            "player": {"config": {"d": 32, "ff": 64, "layers": 1, "d_emb": 16, "card_d": 24, "head_hidden": 32}},
            "actor": {"sims": 8, "chunk_games": 4, "chunk_secs": 60, "workers": 1},
            "learner": {"min_new_positions": 30, "window_positions": 1000, "epochs": 1, "batch": 64, "threads": 1},
            "eval": {"pairs": 12, "sims": 8, "workers": 1}, "store": str(tmp_path / "store")}

    def deep(a, b):
        for k, v in b.items():
            a[k] = deep(dict(a.get(k) or {}), v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
        return a
    p = tmp_path / f"cfg-{len(list(tmp_path.glob('cfg-*.yml')))}.yml"
    p.write_text(yaml.safe_dump(deep(base, sections)))
    from draftzero import config
    return config.load(p)


class Run:
    """Every role in one process, on one LocalStore, stepped by hand."""

    def __init__(self, tmp_path, cfg, *, actors=("a1",), records=5, p_a=0.6, store=None):
        self.store = store or Recording(tmp_path / "store")
        self.cfg, self.log = cfg, []
        self.gorge = FakeGorge(records=records, p_a=p_a)
        say = self.log.append
        self.trains = []
        def manifest(s, c):
            s.write_json(ctl.MANIFEST, {"format": "draftzero-manifest/1", "run": ctl.get(c, "run")})
        self.controller = Controller(self.store, cfg, log=say, status_every_s=0, hb_cache_s=0, write_manifest=manifest)
        self.learner = Learner(self.store, cfg, "l1", tmp_path / "work" / "learner", gorge=self.gorge,
                               train=fake_train(self.trains), export=fake_export, log=say)
        self.actors = [Actor(self.store, cfg, a, tmp_path / "work" / f"actor-{a}", gorge=self.gorge, log=say)
                       for a in actors]
        self.evaluator = Evaluator(self.store, cfg, "e1", tmp_path / "work" / "evaluator", gorge=self.gorge, log=say)

    def control(self):
        return ctl.read(self.store)


# ------------------------------------------------------------------------------------------------ control and layout

def test_layout_paths_seeds_and_values():
    assert ctl.version_dir(3) == "versions/player/v0003" and ctl.latest_path() == "versions/player/latest.json"
    assert ctl.version_file(12, "model.dzgw") == "versions/player/v0012/model.dzgw"
    assert ctl.shard_dir("pod1", 3) == "shards/pod1/000003"
    assert ctl.parse_shard("shards/pod1/000003/meta.json") == ("pod1", 3) and ctl.parse_shard("evals/x/0") is None
    assert ctl.eval_games_path("v0001-vs-prev", 2) == "evals/v0001-vs-prev/0002.games.jsonl"
    assert ctl.metrics_path("actor", "pod1", 7) == "metrics/actor/pod1/000007.jsonl"
    want = int(hashlib.sha256(b"31|pod1|5").hexdigest(), 16) % 2**31
    assert ctl.chunk_seed(31, "pod1", 5) == want and ctl.chunk_seed(31, "pod2", 5) != want
    for bad in ("a/b", "", "../x", "a:b"):
        with pytest.raises(ValueError):
            ctl.check_name(bad)
    assert ctl.parse_value("3e-4") == 3e-4 and ctl.parse_value("true") is True and ctl.parse_value("null") is None
    assert ctl.parse_value("[1, 2]") == [1, 2] and ctl.parse_value("sbtactical") == "sbtactical"
    assert ctl.immediate("paused") and ctl.immediate("budget.max_hours") and not ctl.immediate("learner.lr")
    assert ctl.get({"a": SimpleNamespace(b={"c": 3})}, "a.b.c") == 3 and ctl.get({}, "x.y", 5) == 5


def test_new_control_schema(tmp_path):
    cfg = make_cfg(tmp_path, mode="sync", sync={"games_per_version": 10})
    c = ctl.new_control(cfg)
    assert c["format"] == "dz-control/1" and c["version"] == {"player": None} and c["evals"] == []
    assert c["stop"] is False and c["paused"] is False and c["updated"]
    assert c["play"]["sims"] == 8 and c["play"]["chunk_games"] == 4 and c["play"]["games_per_version"] == 10
    assert c["play"]["pcr"]["cheap_sims"] == 40
    s = LocalStore(tmp_path / "s")
    s.write_json(ctl.CONTROL, {**c, "format": "dz-control/0"})
    with pytest.raises(ValueError, match="format"):
        ctl.read(s)


# ------------------------------------------------------------------------------------------------ actor

def test_actor_plays_version_0_without_a_net_and_uploads_one_batch(tmp_path):
    r = Run(tmp_path, make_cfg(tmp_path))
    a = r.actors[0]
    assert a.step() == "wait"                                 # no control.json yet
    r.controller.tick()
    assert a.step() == "wait"                                 # no version yet
    assert r.learner.step() == "wait"                         # publishes version 0
    meta0 = r.store.read_json(ctl.version_file(0, "meta.json"))
    assert meta0["net"] is False and meta0["version"] == 0 and r.store.read_json(ctl.latest_path())["version"] == 0
    assert r.controller.tick()["version"]["player"] == 0
    r.store.puts.clear()
    assert a.step() == "played"
    assert len(r.store.puts) == 1                              # shard + heartbeat + metrics: ONE put_many
    put = r.store.puts[0]
    sd = "shards/a1/000001"
    assert {f"{sd}/corpus.jsonl.gz", f"{sd}/games.jsonl", f"{sd}/meta.json", "heartbeats/a1.json",
            "metrics/actor/a1/000001.jsonl"} == set(put)
    assert put[-1] == f"{sd}/meta.json"                        # the file that marks the shard complete goes last
    call = r.gorge.plays[-1]
    spec = call["a"]
    assert call["b"] == spec and call["record"] and call["pairs"] == 4 and call["split"] == "train"
    assert call["workers"] == 1 and call["timeout_s"] == 60 and call["belief"] is None
    assert call["seed"] == ctl.chunk_seed(7, "a1", 1)
    assert spec.kind == "az" and spec.sims == 8 and spec.explore and spec.closedlist and spec.cpuct == 0.25
    assert spec.gonet is None and spec.topk is None and spec.pcr is None    # version 0: no network, no topk
    assert spec.keys == ("autopay", "oppnodes", "oppfull", "morekinds")
    meta = r.store.read_json(f"{sd}/meta.json")
    assert {k: meta[k] for k in ("actor", "seq", "seed", "version", "games", "records", "sims")} == \
        {"actor": "a1", "seq": 1, "seed": call["seed"], "version": 0, "games": 4, "records": 20, "sims": 8}
    assert meta["spec"] == spec.render() and meta["time"] and meta["wall_s"] >= 0
    assert len(r.store.read_bytes(f"{sd}/games.jsonl").decode().splitlines()) == 4
    lines = [json.loads(x) for x in r.store.read_bytes("metrics/actor/a1/000001.jsonl").decode().splitlines()]
    assert {x["name"] for x in lines} >= {"throughput.games_per_hour", "throughput.positions_per_hour",
                                          "throughput.ms_per_searched"}
    hb = r.store.read_json("heartbeats/a1.json")
    assert hb["role"] == "actor" and hb["games"] == 4 and hb["version"] == 0
    assert a.step() == "played" and r.gorge.plays[-1]["seed"] == ctl.chunk_seed(7, "a1", 2)
    assert not list((a.wd / "chunks").iterdir())               # chunk folders are cleaned up
    # a restarted actor continues the sequence from the store
    a2 = Actor(r.store, r.cfg, "a1", tmp_path / "w2", gorge=r.gorge, log=lambda *_: None)
    assert a2.next_seq() == 3


def test_actor_spec_with_a_network_and_playout_caps(tmp_path):
    pcr = {"prob": 0.5, "cheap_sims": 4, "by_kind": {"priority": 0.65, "attackers": 0.5}}
    cfg = make_cfg(tmp_path, actor={"pcr": pcr})
    r = Run(tmp_path, cfg)
    r.controller.tick()
    r.learner.step()
    dz = tmp_path / "v1.dzgw"
    dz.write_bytes(b"DZGW0001net")
    r.store.put_many({ctl.version_file(1, "model.dzgw"): dz, ctl.version_file(1, "meta.json"): json.dumps(
        {"version": 1, "net": True, "sha256": {"dzgw": ctl.sha256(dz)}}).encode(),
        ctl.latest_path(): json.dumps({"version": 1, "path": ctl.version_dir(1)}).encode()})
    r.controller.tick()
    assert r.actors[0].step() == "played"
    spec = r.gorge.plays[-1]["a"]
    assert spec.gonet and Path(spec.gonet).read_bytes() == b"DZGW0001net" and ":" not in spec.gonet
    assert spec.topk == 6 and spec.pcr == 0.5 and spec.cheap == 4
    assert spec.pcrk == (("attackers", 0.5), ("priority", 0.65))
    # a corrupted download is refused
    bad = {"version": 2, "net": True, "sha256": {"dzgw": "0" * 64}}
    r.store.put(ctl.version_file(2, "meta.json"), json.dumps(bad).encode())
    r.store.put(ctl.version_file(2, "model.dzgw"), b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        r.actors[0].ensure_version(2)


def test_actor_stop_pause_and_spooled_uploads(tmp_path):
    r = Run(tmp_path, make_cfg(tmp_path))
    r.controller.tick()
    r.learner.step()
    r.controller.tick()
    a = r.actors[0]
    r.store.fail = 1                                           # the shard upload fails: spooled
    assert a.step() == "played"
    assert not r.store.list("shards/") and len(list((a.wd / "spool").iterdir())) == 1
    assert a.step() == "played"                                # the next upload carries both shards
    assert {ctl.parse_shard(p) for p in r.store.list("shards/")} == {("a1", 1), ("a1", 2)}
    assert not list((a.wd / "spool").iterdir())
    r.store.put(ctl.CONTROL, json.dumps({**r.control(), "paused": True}).encode())
    assert a.step() == "paused" and r.store.read_json("heartbeats/a1.json")["state"] == "paused"
    r.store.put(ctl.CONTROL, json.dumps({**r.control(), "paused": False, "stop": True}).encode())
    assert a.run() == 0 and r.store.read_json("heartbeats/a1.json")["state"] == "stopped"


def test_actor_waits_for_a_version_its_store_lacks(tmp_path):
    """control.json can name a version this host's store does not hold yet (a relay copied control.json before
    the version): the actor waits and retries on its next poll, saying so once, instead of dying."""
    r = Run(tmp_path, make_cfg(tmp_path))
    r.controller.tick()
    r.learner.step()
    r.controller.tick()
    a = r.actors[0]
    assert a.step() == "played"
    r.store.put(ctl.CONTROL, json.dumps({**r.control(), "version": {"player": 1}}).encode())
    r.log.clear()
    assert [a.step() for _ in range(3)] == ["wait"] * 3
    assert len([x for x in r.log if "version 1" in x]) == 1
    assert r.store.read_json("heartbeats/a1.json")["state"] == "waiting"
    meta1 = {"version": 1, "net": False, "sha256": {}}
    r.store.put_many({ctl.version_file(1, "meta.json"): json.dumps(meta1).encode()})
    assert a.step() == "played" and r.store.read_json("shards/a1/000002/meta.json")["version"] == 1


def test_actor_sync_mode_plays_exactly_games_per_version(tmp_path):
    r = Run(tmp_path, make_cfg(tmp_path, mode="sync", sync={"games_per_version": 10}))
    r.controller.tick()
    r.learner.step()
    r.controller.tick()
    a = r.actors[0]
    assert [a.step() for _ in range(4)] == ["played", "played", "played", "wait"]
    assert [p["pairs"] for p in r.gorge.plays] == [4, 4, 2]
    assert r.store.read_json("heartbeats/a1.json")["state"] == "waiting"


# ------------------------------------------------------------------------------------------------ learner

def test_learner_ingests_trains_and_publishes(tmp_path):
    r = Run(tmp_path, make_cfg(tmp_path))
    r.controller.tick()
    r.learner.step()                                           # v0, no network
    r.controller.tick()
    r.actors[0].step()                                         # 20 positions: not due (min 30)
    assert r.learner.step() == "wait" and r.learner.new_positions() == 20 and not r.trains
    assert r.gorge.packs[-1]["game_prefix"] == "a1-1-"
    assert all(g.startswith("a1-1-") for g in json.loads(
        next((r.learner.wd / "packs").glob("*/00000/meta.json")).read_text())["games"])
    assert r.store.read_json(ctl.INGESTED)[0]["positions"] == 20
    r.actors[0].step()
    r.store.puts.clear()
    assert r.learner.step() == "published"
    argv = r.trains[-1]
    assert argv[argv.index("--arch") + 1] == "mlp" and json.loads(argv[argv.index("--config") + 1])["d"] == 32
    assert "--in-ram" in argv and "--init" not in argv         # v0 has no network: v1 starts from random init
    assert argv[argv.index("--epochs") + 1] == "1.0" and argv[argv.index("--lr") + 1] == "0.0001"
    assert argv[argv.index("--device") + 1] == "cpu" and argv[argv.index("--batch") + 1] == "64"
    assert argv[argv.index("--policy-target") + 1] == "visits" and argv[argv.index("--value-blend") + 1] == "1.0"
    i = argv.index("--train") + 1
    assert len([x for x in argv[i:] if not x.startswith("--")][:2]) == 2
    put = next(p for p in r.store.puts if ctl.latest_path() in p)
    assert put[-1] == ctl.latest_path() and ctl.version_file(1, "model.pt") in put and ctl.INGESTED in put
    meta = r.store.read_json(ctl.version_file(1, "meta.json"))
    assert meta["parent"] == 0 and meta["net"] is True and meta["positions"] == 40 and meta["new_positions"] == 40
    assert meta["shards"] == ["shards/a1/000001", "shards/a1/000002"] and meta["games"] == 8
    for name, key in (("model.pt", "pt"), ("model.dzgw", "dzgw")):
        local = r.store.get(ctl.version_file(1, name), tmp_path / "chk" / name)
        assert ctl.sha256(local) == meta["sha256"][key]
    assert meta["metrics"]["policy_ce"] == pytest.approx(1.1) and meta["metrics"]["value_auc"] == 0.7
    lines = [json.loads(x) for x in r.store.read_bytes("metrics/learner/l1/000001.jsonl").decode().splitlines()]
    learn = {x["name"]: x for x in lines}
    assert learn["learn.policy_ce"]["version"] == 1 and learn["learn.policy_ce"]["n"] == 40
    assert {"learn.value_logloss", "learn.top1_visits", "learn.value_auc", "data.window_positions"} <= set(learn)
    assert all(e["used"] == 1 for e in r.store.read_json(ctl.INGESTED))
    assert r.store.read_json(ctl.latest_path()) == {"version": 1, "path": "versions/player/v0001"}
    # the next version warm-starts from v1, which the learner keeps without downloading it
    r.controller.tick()
    r.actors[0].step()
    r.actors[0].step()
    assert r.learner.step() == "published"
    argv = r.trains[-1]
    init = Path(argv[argv.index("--init") + 1])
    assert init.name == "model.pt" and init.parent.name == "v0001"
    meta2 = r.store.read_json(ctl.version_file(2, "meta.json"))
    assert meta2["new_positions"] == 40 and meta2["positions"] == 80 and meta2["init"] == "v0001"


def test_learner_window_and_steps_per_position(tmp_path):
    r = Run(tmp_path, make_cfg(tmp_path, learner={"window_positions": 40, "steps_per_position": 2.0}))
    r.controller.tick()
    r.learner.step()
    r.controller.tick()
    for _ in range(3):
        r.actors[0].step()
    r.learner.ingest()
    assert [e["shard"] for e in r.learner.window()] == ["shards/a1/000002", "shards/a1/000003"]
    assert r.learner.step() == "published"
    argv = r.trains[-1]
    assert argv[argv.index("--max-steps") + 1] == str(math.ceil(2.0 * 60 / 64))
    assert argv[argv.index("--epochs") + 1] == "1000000"
    assert r.store.read_json(ctl.version_file(1, "meta.json"))["shards"] == ["shards/a1/000002", "shards/a1/000003"]
    packs = {p.name for p in (r.learner.wd / "packs").iterdir()}
    assert packs == {"a1-000002", "a1-000003"}                 # the shard outside the window was pruned locally


def test_learner_lease(tmp_path):
    cfg = make_cfg(tmp_path)
    s = LocalStore(tmp_path / "st")
    s.write_json(ctl.LEASE, {"holder": "other", "expires": time.time() + 600})
    learner = Learner(s, cfg, "l1", tmp_path / "w", gorge=FakeGorge(), log=lambda *_: None)
    with pytest.raises(LeaseHeld):
        learner.step()
    s.write_json(ctl.LEASE, {"holder": "other", "expires": time.time() - 1})
    assert learner.step() == "wait" and s.read_json(ctl.LEASE)["holder"] == learner.holder
    learner.release()
    assert s.read_json(ctl.LEASE)["expires"] <= time.time()
    other = Learner(s, cfg, "l2", tmp_path / "w2", gorge=FakeGorge(), log=lambda *_: None)
    other.lease()                                              # a released lease is free at once


def test_a_learner_that_lost_its_lease_does_not_publish_and_forgets_its_state(tmp_path):
    """While learner A trains, its lease runs out and B takes it and publishes version 1: A does not publish over
    it; it forgets what it knew of the store, so once it holds the lease again it starts from B's ingested.json
    and does not train again on the data B used."""
    r = Run(tmp_path, make_cfg(tmp_path))
    r.controller.tick()
    r.learner.step()                                           # v0
    r.controller.tick()
    r.actors[0].step()
    r.actors[0].step()                                         # 40 new positions: due
    a = r.learner
    b = Learner(r.store, r.cfg, "l2", tmp_path / "work" / "l2", gorge=r.gorge, train=fake_train(r.trains),
                export=fake_export, log=r.log.append)
    inner = a._train

    def overtaken(argv):
        r.store.write_json(ctl.LEASE, {**r.store.read_json(ctl.LEASE), "expires": time.time() - 1})
        assert b.step() == "published"                         # B takes the lapsed lease and publishes v1
        return inner(argv)
    a._train = overtaken
    with pytest.raises(LeaseHeld):
        a.step()
    assert sum(ctl.version_file(1, "model.pt") in p for p in r.store.puts) == 1
    assert r.store.read_bytes(ctl.version_file(1, "model.pt")) == b"weights 1"      # B's (A trained second)
    assert a.index is None and a.ingested == [] and a.local == {}
    with pytest.raises(LeaseHeld):
        a.step()                                               # B holds the lease
    b.release()
    trained = len(r.trains)
    assert a.step() == "wait" and len(r.trains) == trained     # nothing new since B's v1
    assert all(e["used"] == 1 for e in r.store.read_json(ctl.INGESTED))


def test_a_learner_renews_its_lease_while_it_ingests(tmp_path):
    """Ingesting a backlog and re-packing the window can take longer than the lease: the step renews it from a
    thread throughout, not only while training."""
    r = Run(tmp_path, make_cfg(tmp_path))
    r.controller.tick()
    r.learner.step()
    r.controller.tick()
    r.actors[0].step()
    renewing = []
    pack = r.gorge.pack
    r.gorge.pack = lambda *a, **k: renewing.append(any(
        t.name == "learner-lease" and t.is_alive() for t in threading.enumerate())) or pack(*a, **k)
    r.learner.step()
    assert renewing == [True]


def test_learner_sync_mode_trains_once_per_version(tmp_path):
    r = Run(tmp_path, make_cfg(tmp_path, mode="sync", sync={"games_per_version": 10},
                               learner={"min_new_positions": 1}))
    r.controller.tick()
    r.learner.step()
    r.controller.tick()
    a = r.actors[0]
    a.step()
    a.step()
    assert r.learner.step() == "wait"                          # 8 of 10 games of version 0
    assert a.step() == "played" and r.learner.step() == "published"
    assert r.learner.step() == "wait"                          # once per version: waits for version 1's games
    r.controller.tick()
    assert [a.step() for _ in range(3)] == ["played"] * 3
    assert r.learner.step() == "published" and len(r.trains) == 2
    meta2 = r.store.read_json(ctl.version_file(2, "meta.json"))
    assert meta2["new_positions"] == 50


def test_learner_with_the_real_trainer_and_exporter(tmp_path):
    """nets.train and nets.export on synthetic packs: the learner's argv and outputs match the real trainer."""
    cfg = make_cfg(tmp_path, learner={"min_new_positions": 200, "batch": 32, "warmup": 2})
    r = Run(tmp_path, cfg, records=25)
    r.learner._train = None
    r.learner._export = __import__("draftzero.loop.learner", fromlist=["x"]).default_export
    r.controller.tick()
    r.learner.step()
    r.controller.tick()
    r.actors[0].step()
    r.actors[0].step()
    assert r.learner.step() == "published"
    meta = r.store.read_json(ctl.version_file(1, "meta.json"))
    assert meta["steps"] and {"policy_ce", "value_logloss"} <= set(meta["metrics"])
    from draftzero.nets import export
    header, _ = export.read(r.store.get(ctl.version_file(1, "model.dzgw"), tmp_path / "m.dzgw"))
    assert header["arch"] == "mlp" and header["config"]["d"] == 32


def test_published_metrics_are_the_published_checkpoints(tmp_path):
    """The learner publishes best.pt: its meta.json metrics, its learn.* lines and the finite-loss check are those
    of best.pt's evaluation (summary best_step), not of the last one (last.pt)."""
    def train(argv):
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "best.pt").write_bytes(b"best")
        (out / "last.pt").write_bytes(b"last")
        ev = lambda ce, total: {"holdout": {"n_policy": 40, "policy_ce": ce, "n_value": 40,     # noqa: E731
                                            "value_logloss": 0.6, "total": total}}
        (out / "curves.jsonl").write_text("".join(json.dumps(x) + "\n" for x in (
            {"type": "eval", "step": 3, "eval": ev(1.0, 1.6)}, {"type": "train", "step": 6, "policy_ce": 0.9},
            {"type": "eval", "step": 6, "eval": ev(1.3, float("nan"))})))
        return {"steps": 6, "best_step": 3}
    r = Run(tmp_path, make_cfg(tmp_path))
    r.learner._train = train
    r.controller.tick()
    r.learner.step()
    r.controller.tick()
    r.actors[0].step()
    r.actors[0].step()
    assert r.learner.step() == "published"                     # the last eval's loss is NaN; best.pt's isn't
    meta = r.store.read_json(ctl.version_file(1, "meta.json"))
    assert r.store.read_bytes(ctl.version_file(1, "model.pt")) == b"best"
    assert meta["metrics"]["policy_ce"] == 1.0 and meta["step"] == 3 and meta["steps"] == 6
    lines = [json.loads(x) for x in r.store.read_bytes("metrics/learner/l1/000001.jsonl").decode().splitlines()]
    ce = next(x for x in lines if x["name"] == "learn.policy_ce")
    assert (ce["value"], ce["step"]) == (1.0, 3)


def test_warm_start_keeps_the_checkpoints_own_config(tmp_path):
    """A player.init of another width than player.config: version 1 warm-starts from it with its own arch and
    config (player.arch/config are for a random init) instead of failing to load its weights."""
    from draftzero.loop.learner import default_export
    from draftzero.nets import models as M
    init = tmp_path / "init.pt"
    M.save(M.build("mlp", {"d": 48, "ff": 64, "layers": 1, "d_emb": 16, "card_d": 24, "head_hidden": 32}), init)
    cfg = make_cfg(tmp_path, player={"init": str(init)}, learner={"min_new_positions": 200, "batch": 32, "warmup": 2})
    r = Run(tmp_path, cfg, records=25)
    r.learner._train, r.learner._export = None, default_export
    r.controller.tick()
    r.learner.step()                                           # v0: the init checkpoint
    r.controller.tick()
    r.actors[0].step()
    r.actors[0].step()
    assert r.learner.step() == "published"
    argv = r.learner.train_args(tmp_path / "o", [], tmp_path / "init.pt", 2, 1)
    assert "--init" in argv and "--config" not in argv and "--arch" not in argv
    from draftzero.nets import export
    header, _ = export.read(r.store.get(ctl.version_file(1, "model.dzgw"), tmp_path / "m.dzgw"))
    assert header["config"]["d"] == 48


# ------------------------------------------------------------------------------------------------ evaluator

def test_eval_jobs_and_opponents(tmp_path):
    assert job_id(4, "prev") == "v0004-vs-prev" and job_id(4, "file:/x/best.dzgw") == \
        "v0004-vs-file-x-best.dzgw-" + hashlib.sha1(b"file:/x/best.dzgw").hexdigest()[:8]
    assert not opponent_valid(0, "prev") and not opponent_valid(0, "nonet") and opponent_valid(0, "sbtactical")
    assert opponent_valid(3, "v1") and not opponent_valid(3, "v3") and not opponent_valid(3, "v5")
    assert not opponent_valid(3, "nonsense") and not opponent_valid(3, "spec:bot") and not opponent_valid(3, "random")
    from draftzero.config import OPPONENT_RE                 # the evaluator plays what a run config may name
    for name in ("prev", "v1", "v3", "sbtactical", "bot", "nonet", "file:/x/m.dzgw", "file:/x/m.pt", "random",
                 "spec:bot", "gnubg", "v", "file:"):
        assert opponent_valid(5, name) == bool(OPPONENT_RE.match(name)), name
    job = make_job(4, "sbtactical", make_cfg(tmp_path, eval={"pairs": 25, "sims": 50, "seed": 9}), now=0)
    assert (job["job"], job["version"], job["opponent"], job["pairs"], job["sims"], job["seed"]) == \
        ("v0004-vs-sbtactical", 4, "sbtactical", 25, 50, 9)
    assert [(s["first_pair"], s["pairs"]) for s in job["shards"]] == [(0, 10), (10, 10), (20, 5)]
    spec = policy_spec(make_cfg(tmp_path, search={"closedlist": False, "keys": ["autopay"]}), sims=5, explore=False,
                       gonet=None)
    assert (spec.sims, spec.closedlist, spec.keys, spec.explore, spec.topk) == (5, False, ("autopay",), False, None)


def test_eval_job_ids_are_unique_per_opponent(tmp_path):
    """Opponents whose cleaned labels agree (long file: paths in one folder) get distinct job ids, a name listed
    twice opens one job, and every opponent is played and scored on its own."""
    ref = tmp_path / "workspace" / "draftzero" / "models" / "reference"
    ref.mkdir(parents=True)
    paths = [ref / "fdn-mlp-v12.dzgw", ref / "fdn-mlp-v13.dzgw"]
    for p in paths:
        p.write_bytes(b"DZGW0001" + p.name.encode())
    opps = [f"file:{p}" for p in paths]
    ids = [job_id(5, o) for o in opps]
    assert ids[0] != ids[1] and all(len(i) <= 64 for i in ids) and job_id(5, opps[0]) == ids[0]
    assert job_id(5, "prev") == "v0005-vs-prev" and job_id(5, "v3") == "v0005-vs-v3"
    r = Run(tmp_path, make_cfg(tmp_path, eval={"opponents": [*opps, "sbtactical", "sbtactical"], "pairs": 2}))
    r.controller.tick()
    r.learner.step()                                           # v0
    assert r.evaluator.step() == "wait"                        # its heartbeat
    c = r.controller.tick()
    jobs = [j["job"] for j in c["evals"]]
    assert len(jobs) == len(set(jobs)) == 3
    while r.evaluator.step() == "played":
        pass
    assert sorted(Path(p["b"].gonet).name for p in r.gorge.plays if not isinstance(p["b"], str)) == \
        ["fdn-mlp-v12.dzgw", "fdn-mlp-v13.dzgw"]
    assert r.controller.tick()["evals"] == []
    summaries = [r.store.read_json(p) for p in r.store.list("evals/") if p.endswith("summary.json")]
    assert sorted(s["opponent"] for s in summaries) == sorted([*opps, "sbtactical"])


def test_evaluator_plays_assigned_shards_and_scores_them(tmp_path):
    r = Run(tmp_path, make_cfg(tmp_path, eval={"opponents": ["prev", "sbtactical"]}), p_a=0.9)
    r.controller.tick()
    r.learner.step()                                           # v0
    c = r.controller.tick()
    assert [j["job"] for j in c["evals"]] == ["v0000-vs-sbtactical"]       # v0 has no prev
    r.actors[0].step()
    r.actors[0].step()
    r.learner.step()                                           # v1
    assert r.evaluator.step() == "wait"                        # its heartbeat; nothing assigned yet
    c = r.controller.tick()
    jobs = {j["job"]: j for j in c["evals"]}
    assert set(jobs) == {"v0000-vs-sbtactical", "v0001-vs-prev", "v0001-vs-sbtactical"}
    job = jobs["v0001-vs-prev"]
    assert [(s["first_pair"], s["pairs"]) for s in job["shards"]] == [(0, 10), (10, 2)]
    assert all(s["machine"] == "e1" for j in c["evals"] for s in j["shards"])
    while r.evaluator.step() == "played":
        pass
    plays = [p for p in r.gorge.plays if not p["record"]]
    assert len(plays) == 6 and {p["split"] for p in plays} == {"eval"} and {p["seed"] for p in plays} == {3131}
    assert [p["first_pair"] for p in plays[:2]] == [0, 10]
    vs_prev = [p for p in plays if not isinstance(p["b"], str)]
    assert len(vs_prev) == 2
    hero, opp = vs_prev[0]["a"], vs_prev[0]["b"]
    assert hero.gonet and hero.topk == 6 and not hero.explore and hero.sims == 8 and hero.pcr is None
    assert opp.gonet is None and opp.topk is None and not opp.explore   # prev = v0 = search without a network
    assert {p["b"] for p in plays if isinstance(p["b"], str)} == {"sbtactical"}
    s = r.store.read_json(ctl.eval_summary_path("v0001-vs-prev"))
    assert s["n_pairs"] == 12 and s["n_games"] == 24 and s["score"] > 0.7 and s["lo"] < s["score"] < s["hi"]
    assert s["version"] == 1 and s["opponent"] == "prev"
    games = r.store.read_bytes(ctl.eval_games_path("v0001-vs-prev", 1)).decode().splitlines()
    assert len(games) == 4 and json.loads(games[0])["job"] == "v0001-vs-prev"
    metrics = [json.loads(x) for p in r.store.list("metrics/evaluator/")
               for x in r.store.read_bytes(p).decode().splitlines()]
    assert {(m["name"], m["version"], m["opponent"]) for m in metrics} == \
        {("eval.score", 0, "sbtactical"), ("eval.score", 1, "prev"), ("eval.score", 1, "sbtactical")}
    c = r.controller.tick()
    assert c["evals"] == []                                    # every job closed
    md = r.store.read_bytes(ctl.STATUS_MD).decode()
    assert "v0001-vs-prev" in md and "| e1 | evaluator |" in md and "| a1 | actor |" in md


def test_playing_roles_heartbeat_so_a_hung_game_does_not_look_dead(tmp_path):
    """One hung game keeps Gorge.play busy past timeout_s (its pair's replay adds a probe), longer than the
    controller's liveness window: a playing evaluator (and actor) heartbeats every HEARTBEAT_EVERY_S, so the
    controller does not declare it dead and move its shard to another evaluator."""
    r = Run(tmp_path, make_cfg(tmp_path, eval={"opponents": ["sbtactical"], "pairs": 2}))
    r.controller.tick()
    r.learner.step()                                           # v0
    assert r.evaluator.step() == "wait"
    r.controller.tick()                                        # the job's shard goes to e1
    r.controller.live_s, r.controller.late_s = 0.4, 60.0
    r.evaluator.HEARTBEAT_EVERY_S = r.actors[0].HEARTBEAT_EVERY_S = 0.05
    play, seen = r.gorge.play, []

    def hung(a, b, **kw):                                      # longer than the liveness window
        t0 = time.time()
        time.sleep(0.8)
        if kw["record"]:
            seen.append(r.store.read_json("heartbeats/a1.json")["time"] > t0)
        else:
            r.store.write_json("heartbeats/e2.json", {"machine": "e2", "role": "evaluator", "time": time.time()})
            seen.append(r.controller.tick()["evals"][0]["shards"][0]["machine"])
        return play(a, b, **kw)
    r.gorge.play = hung
    assert r.evaluator.step() == "played" and seen == ["e1"]
    assert r.actors[0].step() == "played" and seen == ["e1", True]


def test_controller_reassigns_shards_of_a_quiet_evaluator(tmp_path):
    cfg = make_cfg(tmp_path, eval={"opponents": ["sbtactical"], "pairs": 10})
    s = LocalStore(tmp_path / "st")
    c = Controller(s, cfg, log=lambda *_: None, live_s=100, hb_cache_s=0, write_manifest=lambda *a: None)
    s.write_json("heartbeats/e1.json", {"machine": "e1", "role": "evaluator", "time": 1000})
    s.write_json(ctl.latest_path(), {"version": 0, "path": ctl.version_dir(0)})
    assert c.tick(now=1000)["evals"][0]["shards"][0]["machine"] == "e1"
    s.write_json("heartbeats/e2.json", {"machine": "e2", "role": "evaluator", "time": 1150})
    s.write_json("heartbeats/a1.json", {"machine": "a1", "role": "actor", "time": 1150})
    assert c.tick(now=1150)["evals"][0]["shards"][0]["machine"] == "e2"   # e1 went quiet; actors don't evaluate


# ------------------------------------------------------------------------------------------------ controller

def test_controller_writes_config_and_manifest_once(tmp_path):
    cfg = make_cfg(tmp_path)
    s = LocalStore(tmp_path / "st")
    made = []
    c = Controller(s, cfg, log=lambda *_: None, write_manifest=lambda st, cf: made.append(1) or st.write_json(
        ctl.MANIFEST, {"run": "t"}))
    c.tick()
    assert s.exists(ctl.CONFIG) and s.exists(ctl.MANIFEST) and made == [1]
    from draftzero import config
    p = tmp_path / "back.yml"
    p.write_bytes(s.read_bytes(ctl.CONFIG))
    assert config.config_hash(config.load(p)) == config.config_hash(cfg)
    c2 = Controller(s, cfg, log=lambda *_: None, write_manifest=lambda *a: made.append(2))
    c2.tick()
    assert made == [1] and ctl.read(s)["started"]
    with pytest.raises(ValueError, match="holds run"):
        Controller(s, make_cfg(tmp_path, run="other"), log=lambda *_: None, write_manifest=lambda *a: None).tick()


def test_controller_opens_an_eval_for_every_version_passed(tmp_path):
    for every, want in ((1, [1, 2, 3]), (2, [2])):
        cfg = make_cfg(tmp_path, eval={"every_versions": every, "opponents": ["prev"]})
        s = LocalStore(tmp_path / f"st{every}")
        c = Controller(s, cfg, log=lambda *_: None, write_manifest=lambda *a: None)
        s.write_json(ctl.latest_path(), {"version": 0, "path": ctl.version_dir(0)})
        assert c.tick()["evals"] == []
        s.write_json(ctl.latest_path(), {"version": 3, "path": ctl.version_dir(3)})
        con = c.tick()
        assert con["version"]["player"] == 3 and [j["version"] for j in con["evals"]] == want


def test_plan_changes_apply_at_version_boundaries(tmp_path):
    r = Run(tmp_path, make_cfg(tmp_path))
    r.controller.tick()
    r.learner.step()
    r.controller.tick()
    ctl.request_change(r.store, "actor.sims", 16, "dan")
    ctl.request_change(r.store, "player.arch", "gnn", "dan")             # fixed: rejected
    ctl.request_change(r.store, "actor.simz", 3, "dan")                  # a typo: rejected
    ctl.request_change(r.store, "paused", True, "dan")                   # applies at once
    c = r.controller.tick()
    assert c["paused"] is True and c["play"]["sims"] == 8
    assert [x["path"] for _, x in ctl.pending_requests(r.store)] == ["actor.sims"]
    rejected = [r.store.read_json(p) for p in r.store.list(ctl.REJECTED)]
    assert sorted(x["path"] for x in rejected) == ["actor.simz", "player.arch"] and all(x["error"] for x in rejected)
    ev = ctl.read_events(r.store)
    assert [(e["path"], e["old"], e["new"], e["at_version"], e["kind"]) for e in ev] == \
        [("paused", False, True, 0, "set")]
    assert r.actors[0].step() == "paused"
    ctl.request_change(r.store, "paused", False, "dan")
    r.controller.tick()
    for _ in range(2):
        r.actors[0].step()
    r.learner.step()                                           # v1: a version boundary
    c = r.controller.tick()
    assert c["version"]["player"] == 1 and c["play"]["sims"] == 16 and not ctl.pending_requests(r.store)
    ev = ctl.read_events(r.store)
    e = ev[-1]
    assert (e["path"], e["old"], e["new"], e["at_version"], e["by"]) == ("actor.sims", 8, 16, 1, "dan")
    from draftzero import config
    assert ctl.get(config.effective(r.cfg, ev), "actor.sims") == 16
    r.actors[0].step()
    assert r.gorge.plays[-1]["a"].sims == 16                   # actors follow control["play"]
    ctl.request_change(r.store, "learner.lr", 3e-4, "dan")
    for _ in range(2):
        r.actors[0].step()
    r.controller.tick()
    r.learner.step()                                           # v2 trained before the change applies
    r.controller.tick()                                        # boundary: learner.lr applies from version 2
    for _ in range(2):
        r.actors[0].step()
    r.learner.step()
    argv2, argv3 = r.trains[-2], r.trains[-1]
    assert argv2[argv2.index("--lr") + 1] == "0.0001" and argv3[argv3.index("--lr") + 1] == "0.0003"


def test_deferred_plan_requests_are_checked_in_a_chain(tmp_path):
    """Requests waiting for the next version are checked against the config with the earlier waiting ones applied,
    as `plan set` checks them, so a change that is valid only together is not half rejected."""
    r = Run(tmp_path, make_cfg(tmp_path))
    r.controller.tick()
    r.learner.step()                                           # v0
    r.controller.tick()
    ctl.request_change(r.store, "actor.pcr.cheap_sims", 4, "dan")
    ctl.request_change(r.store, "actor.pcr.prob", 0.25, "dan")                 # needs cheap_sims <= sims (8)
    ctl.request_change(r.store, "actor.pcr.by_kind.target", 1.0, "dan")        # needs prob > 0
    ctl.request_change(r.store, "learner.window_positions", 5000, "dan")
    ctl.request_change(r.store, "learner.min_new_positions", 2000, "dan")      # needs window >= it
    ctl.request_change(r.store, "paused", True, "dan")                         # at once, alone
    c = r.controller.tick()                                    # between versions
    assert not r.store.list(ctl.REJECTED) and c["paused"] and c["play"]["pcr"]["prob"] == 0.0
    assert [x["path"] for _, x in ctl.pending_requests(r.store)] == [
        "actor.pcr.cheap_sims", "actor.pcr.prob", "actor.pcr.by_kind.target", "learner.window_positions",
        "learner.min_new_positions"]
    assert r.controller.cfg.learner.window_positions == 1000   # nothing deferred applies before the version
    ctl.request_change(r.store, "paused", False, "dan")
    r.controller.tick()
    for _ in range(2):
        r.actors[0].step()
    r.learner.step()                                           # v1
    c = r.controller.tick()
    assert not r.store.list(ctl.REJECTED) and not ctl.pending_requests(r.store)
    assert c["play"]["pcr"]["prob"] == 0.25 and c["play"]["pcr"]["by_kind"] == {"target": 1.0}
    assert (r.controller.cfg.learner.window_positions, r.controller.cfg.learner.min_new_positions) == (5000, 2000)


def test_whole_section_plan_changes_are_stored_as_json_and_replay(tmp_path):
    """A change of a whole section (budget, actor.pcr) goes into events.jsonl as a plain mapping, not a dataclass's
    repr, so every role (and a restarted controller) can replay it; budget applies at once, like budget.*."""
    from draftzero import config
    r = Run(tmp_path, make_cfg(tmp_path))
    r.controller.tick()
    r.learner.step()                                           # v0
    r.controller.tick()
    ctl.request_change(r.store, "budget", {"max_versions": 10}, "dan")
    ctl.request_change(r.store, "actor.pcr", "{prob: 0.5, cheap_sims: 4}", "dan")
    r.controller.tick()                                        # no new version: only budget applies
    assert [e["path"] for e in ctl.read_events(r.store)] == ["budget"]
    for _ in range(2):
        r.actors[0].step()
    assert r.learner.step() == "published"                     # v1: actor.pcr applies
    r.controller.tick()
    ev = ctl.read_events(r.store)
    assert [e["path"] for e in ev] == ["budget", "actor.pcr"]
    assert ev[0]["new"] == {"max_versions": 10, "max_hours": None, "max_games": None}
    assert ev[1]["old"] == {"prob": 0.0, "cheap_sims": 40, "by_kind": {}}
    assert ev[1]["new"] == {"prob": 0.5, "cheap_sims": 4, "by_kind": {}}
    eff = config.effective(r.cfg, ev)
    assert eff.actor.pcr.prob == 0.5 and eff.actor.pcr.cheap_sims == 4 and eff.budget.max_versions == 10
    again = Controller(r.store, r.cfg, log=lambda *_: None, write_manifest=lambda *a: None)
    assert again.tick()["play"]["pcr"]["prob"] == 0.5          # a restarted controller replays them
    assert r.evaluator.step() in ("wait", "played") and r.learner.step() == "wait"
    assert r.actors[0].step() == "played" and r.gorge.plays[-1]["a"].pcr == 0.5


def test_budget_drains_then_stops_and_stop_request(tmp_path):
    r = Run(tmp_path, make_cfg(tmp_path, budget={"max_versions": 1}, eval={"opponents": ["prev"], "pairs": 2}))
    r.controller.tick()
    r.learner.step()
    r.controller.tick()
    for _ in range(2):
        r.actors[0].step()
    r.evaluator.step()
    r.learner.step()
    c = r.controller.tick()
    assert c["paused"] and not c["stop"] and [j["job"] for j in c["evals"]] == ["v0001-vs-prev"]
    assert r.actors[0].step() == "paused"
    for _ in range(2):
        r.actors[0].step()
    r.learner.ingest()
    assert r.learner.step() == "wait" and r.store.read_json(ctl.LEARNER_STATUS)["state"] == "done"
    assert r.evaluator.step() == "played"
    c = r.controller.tick()
    assert c["stop"] and c["stop_reason"] == "budget.max_versions 1"
    assert r.learner.step() == "stop" and r.actors[0].step() == "stop" and r.evaluator.step() == "stop"
    # stop.request stops at once; max_hours too
    for budget, req in (({}, True), ({"max_hours": 0.001}, False)):
        s = LocalStore(tmp_path / f"s-{req}")
        c = Controller(s, make_cfg(tmp_path, budget=budget), log=lambda *_: None, write_manifest=lambda *a: None)
        c.tick(now=time.time())
        if req:
            s.put(ctl.STOP, b"stop\n")
        assert c.tick(now=time.time() + 10)["stop"]


def test_a_new_controller_resumes_a_stopped_run(tmp_path):
    cfg = make_cfg(tmp_path)
    s = LocalStore(tmp_path / "st")
    new = lambda: Controller(s, cfg, log=lambda *_: None, write_manifest=lambda *a: None)       # noqa: E731
    c = new()
    c.tick()
    s.put(ctl.STOP, b"stop\n")
    assert c.tick()["stop"]
    assert not new().tick()["stop"] and not s.exists(ctl.STOP)      # restarting a stopped run resumes it
    s.put(ctl.STOP, b"stop\n")                                        # a stop asked while no controller ran
    assert new().tick()["stop"]


def test_max_games_budget(tmp_path):
    r = Run(tmp_path, make_cfg(tmp_path, budget={"max_games": 8}, eval={"opponents": ["prev"]}))
    r.controller.tick()
    r.learner.step()
    r.controller.tick()
    r.actors[0].step()
    assert not r.controller.tick()["stop"]
    r.actors[0].step()
    assert r.controller.tick()["stop_reason"] == "budget.max_games 8"


def test_in_process_run_to_the_version_budget(tmp_path):
    """Every role stepped round-robin until the controller stops the run at budget.max_versions."""
    r = Run(tmp_path, make_cfg(tmp_path, budget={"max_versions": 2}, eval={"pairs": 4, "opponents": ["prev", "nonet"]}),
            actors=("a1", "a2"))
    for _ in range(40):
        c = r.controller.tick()
        if c["stop"]:
            break
        for role in (*r.actors, r.learner, r.evaluator):
            role.step()
    assert c["stop"] and c["stop_reason"] == "budget.max_versions 2"
    assert r.store.read_json(ctl.latest_path())["version"] == 2
    jobs = {p.split("/")[1] for p in r.store.list("evals/") if p.endswith("summary.json")}
    assert jobs == {"v0001-vs-prev", "v0002-vs-prev", "v0002-vs-nonet", "v0001-vs-nonet"}
    actors = {ctl.parse_shard(p)[0] for p in r.store.list("shards/")}
    assert actors == {"a1", "a2"}
    st = r.store.read_json(ctl.STATUS)
    assert st["stop"] and {m["role"] for m in st["machines"]} == {"actor", "learner", "evaluator"}
    assert "budget.max_versions" in status_md(st)


# ------------------------------------------------------------------------------------------------ command line

def test_cli_plan_status_and_stop(tmp_path, capsys):
    from draftzero.loop import cli
    r = Run(tmp_path, make_cfg(tmp_path))
    r.controller.tick()
    st = str(r.store.root)
    assert cli.main(["plan", st, "set", "learner.lr=3e-4", "actor.sims=16", "--by", "dan"]) == 0
    reqs = [x for _, x in ctl.pending_requests(r.store)]
    assert [(x["path"], x["value"], x["by"]) for x in reqs] == [("learner.lr", 3e-4, "dan"), ("actor.sims", 16, "dan")]
    assert cli.main(["plan", "--store", st, "set", "player.arch=gnn"]) == 2       # checked against config.yml
    assert cli.main(["plan", st, "set", "nonsense"]) == 2
    assert cli.main(["plan", st, "list"]) == 0
    assert "pending   learner.lr = 0.0003" in capsys.readouterr().out
    assert cli.main(["status", st]) == 0 and "# Run t" in capsys.readouterr().out
    assert cli.main(["status", st, "--json"]) == 0 and json.loads(capsys.readouterr().out)["run"] == "t"
    assert cli.main(["stop", st]) == 0 and r.store.exists(ctl.STOP)
    assert cli.main(["status"]) == 2
