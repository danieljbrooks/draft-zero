"""The loop's engine-free parts carried over from docs/021 (src/draftzero/loop): settings, the store, evaluation
jobs and the controller. The XMage-era worker, trainer and their record tables are gone with their tests.
"""
from __future__ import annotations

import gzip
import json

import pytest

from draftzero.loop import control as ctl
from draftzero.loop import evals
from draftzero.loop import settings as sc
from draftzero.loop.controller import Controller
from draftzero.loop.store import LocalStore, open_store


# ------------------------------------------------------------------------------------------------ settings, store

def test_config_defaults_and_unknown_keys(tmp_path):
    cfg = sc.load(start="x.pt.gz")
    assert cfg["network"] == "gnn" and cfg["play"]["method"] == "pimc" and cfg["trainer"]["reuse"] == 4.0
    assert sc.bot(cfg) == "gnn@1000" and sc.bot({**cfg, "network": "mlp"}, 100) == "il_bc@100"
    with pytest.raises(ValueError, match="unknown setting play.simulation"):
        sc.load(start="x", play={"simulation": 5})
    with pytest.raises(ValueError, match="start"):
        sc.load()
    p = tmp_path / "c.yml"
    p.write_text("start: s.pt.gz\nplay: {simulations: 8}\ntrainer: {max_versions: 2}\n")
    c = sc.load(p)
    assert c["play"]["simulations"] == 8 and c["play"]["root_noise"] == 0.25 and c["trainer"]["max_versions"] == 2


def test_local_store_round_trips_and_lists(tmp_path):
    s = open_store(tmp_path / "st")
    assert isinstance(s, LocalStore) and s.read_json("control.json") is None
    s.write_json("control.json", {"version": 3})
    src = tmp_path / "w.bin"
    src.write_bytes(b"abc")
    s.put_many({"weights/v0003.pt.gz": src, "games/v0003/m/x.jsonl.gz": b"data"})
    assert s.read_json("control.json") == {"version": 3}
    assert s.list("games/") == ["games/v0003/m/x.jsonl.gz"] and s.list("weights/") == ["weights/v0003.pt.gz"]
    assert s.get("weights/v0003.pt.gz", tmp_path / "out" / "w").read_bytes() == b"abc"
    s.delete(["weights/v0003.pt.gz"])
    assert s.list("weights/") == [] and s.get("weights/v0003.pt.gz", tmp_path / "o2") is None
    with pytest.raises(ValueError):
        s.put("../escape", b"x")


# ------------------------------------------------------------------------------------------------ evaluations

def test_eval_jobs_shards_and_sequential_test():
    job = evals.make_job(4, 0, {"pairs": 25, "shard_pairs": 10, "simulations": 100, "seed": 1, "pool": "p",
                                "heuristic_simulations": 100}, now=0)
    assert job["job"] == "v0004-vs-v0000" and [(s["min_pair"], s["max_pair"]) for s in job["shards"]] == [(0, 9), (10, 19), (20, 24)]
    sc_ = evals.score([{"winner_role": "bot1"}, {"winner_role": "bot2"}, {"winner_role": None}, {"error": "x"}])
    assert (sc_["wins"], sc_["losses"], sc_["draws"], sc_["errors"], sc_["score"]) == (1, 1, 1, 1, 0.5)
    assert evals.sprt(240, 0, 160)["decision"] == "better"
    assert evals.sprt(180, 0, 220)["decision"] == "not better"
    assert evals.sprt(200, 0, 200)["decision"] is None


# ------------------------------------------------------------------------------------------------ controller

def test_controller_publishes_versions_assigns_and_scores_evaluations(tmp_path):
    store = LocalStore(tmp_path / "st")
    cfg = sc.load(start="x", eval={"every_versions": 1, "pairs": 4, "shard_pairs": 2, "sprt_p1": 0.6},
                  controller={"status_every_s": 0})
    c = Controller(store, cfg, log=lambda *_: None)
    assert c.tick(now=100) is None                       # no version 0 yet
    store.write_json(ctl.LATEST, {"version": 0, "path": ctl.weights_path(0), "sha256": "a"})
    assert c.tick(now=110)["version"] == 0 and store.read_json(ctl.CONTROL)["evals"] == []
    store.write_json(ctl.LATEST, {"version": 1, "path": ctl.weights_path(1), "sha256": "b"})
    store.write_json("heartbeats/m1.json", {"machine": "m1", "time": 115})
    store.write_json("heartbeats/m2.json", {"machine": "m2", "time": 116})
    con = c.tick(now=120)
    assert con["version"] == 1 and con["weights"] == "weights/v0001.pt.gz"
    job = con["evals"][0]
    assert job["job"] == "v0001-vs-v0000" and sorted(s["machine"] for s in job["shards"]) == ["m1", "m2"]
    games = [{"winner_role": "bot1"}] * 4
    for sh in job["shards"]:
        store.put(evals.result_path(job["job"], sh["id"]), gzip.compress("".join(json.dumps(g) + "\n" for g in games).encode()))
    con = c.tick(now=130)
    assert con["evals"] == [] and store.read_json("evals/v0001-vs-v0000/summary.json")["result"]["wins"] == 8
    assert "v0001-vs-v0000" in store.read_bytes("status.md").decode()
    store.write_json("learner/status.json", {"done": True})
    assert c.tick(now=140)["stop"] is True


def test_controller_reassigns_late_shards(tmp_path):
    store = LocalStore(tmp_path / "st")
    cfg = sc.load(start="x", eval={"every_versions": 1, "pairs": 2, "shard_pairs": 2, "shard_expected_s": 10,
                                   "late_factor": 2.0})
    c = Controller(store, cfg, log=lambda *_: None)
    store.write_json(ctl.LATEST, {"version": 1, "path": ctl.weights_path(1), "sha256": "b"})
    store.write_json(ctl.CONTROL, {**ctl.new_control({"version": 0, "path": ctl.weights_path(0), "sha256": "a"},
                                                     cfg["play"])})
    store.write_json("heartbeats/m1.json", {"machine": "m1", "time": 100})
    assert c.tick(now=100)["evals"][0]["shards"][0]["machine"] == "m1"
    store.write_json("heartbeats/m1.json", {"machine": "m1", "time": 0})            # m1 went quiet
    store.write_json("heartbeats/m2.json", {"machine": "m2", "time": 125})
    assert c.tick(now=125)["evals"][0]["shards"][0]["machine"] == "m2"
