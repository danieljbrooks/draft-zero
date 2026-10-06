"""The self-play loop (docs/021, src/draftzero/selfplay): everything that runs without the engine.

The engine end of it (play.py's graph records, exploration, a worker's chunks) is checked by the smoke run,
configs/selfplay_smoke.yml.
"""
from __future__ import annotations

import base64
import gzip
import json
import time

import numpy as np
import pytest
import torch

from draftzero.gameplay import graph_net as gn
from draftzero.gameplay import graph_supervised as gs
from draftzero.selfplay import config as sc
from draftzero.selfplay import control as ctl
from draftzero.selfplay import evals, records, tables
from draftzero.selfplay.controller import Controller
from draftzero.selfplay.store import LocalStore, open_store
from draftzero.selfplay.trainer import LeaseHeld, Trainer
from test_graph import toy_graph

TYPE_NAMES = {0: "PRIORITY", 3: "CHOOSE_TARGET", 5: "CHOOSE_USE"}


def b64(xs) -> str:
    return base64.b64encode(np.asarray(xs, "<i4").tobytes()).decode()


def graph_json(g: dict) -> dict:
    """A decoded toy graph in the game player's record layout (GraphRecord.encodeArrays)."""
    opts, k = [], 0
    for n in g["opt_len"]:
        opts.append([int(x) for x in g["opt_node"][k:k + n]])
        k += n
    return {"type": TYPE_NAMES[g["type"]], "text": "priority", "ids": b64(g["ids"]), "values": b64(g["values"]),
            "edge_child": b64(g["child"]), "edge_parent": b64(g["parent"]), "edge_label": b64(g["label"]),
            "options": opts}


def fake_chunk(d, rng, pairs=2, decisions=6, swap_winner=False):
    """play.py's output for `pairs` deck pairs: games.jsonl and per-game records with toy graphs."""
    (d / "records").mkdir(parents=True)
    with open(d / "games.jsonl", "w") as gf:
        for pair in range(pairs):
            for swap in (False, True):
                winner = "A" if (pair + swap + swap_winner) % 2 == 0 else "B"
                gf.write(json.dumps({"pair": pair, "swap": swap, "deck1": "d1", "deck2": "d2", "game_seed": pair,
                                     "budget": 8, "method": "pimc", "winner": winner, "turns": 9, "seconds": 3.0}) + "\n")
                lines = []
                for seat in ("A", "B"):
                    recs = []
                    for i in range(decisions):
                        g = toy_graph(rng, 2 + i % 3, kind="priority" if i % 2 == 0 else "target")
                        n = len(g["opt_len"])
                        v = rng.integers(0, 6, n)
                        v[0] += 1
                        recs.append({"type": TYPE_NAMES[g["type"]], "turn": i + 1, "legal": list(range(n)),
                                     "visits": [int(x) for x in v], "q": float(rng.uniform(-1, 1)), "heuristic": 0.1,
                                     "opt_q": [0.1] * n, "opt_prior": [1 / n] * n, "played": 0, "features": [],
                                     "graph": graph_json(g)})
                    lines.append(json.dumps({"pair": pair, "swap": swap, "seat": seat, "bot": "gnn@8",
                                             "result": 1 if winner == seat else -1, "records": recs}) + "\n")
                with gzip.open(d / "records" / f"p{pair:05d}{'s' if swap else ''}.jsonl.gz", "wt") as f:
                    f.write("".join(lines))


def tiny_checkpoint(path, rng):
    """A small random graph network over the toy graphs' leaves, in graph_supervised's checkpoint layout."""
    from magezero.vocab import FeatureVocab
    leaves = np.unique(np.r_[np.arange(10_000, 10_040), [777, 55555]]).astype(np.int64)
    vocab = FeatureVocab(leaves, feature_hash_bins=gn.GLOBAL_MAX, hash_version=2)
    edge_vocab = FeatureVocab(np.asarray([gn.feature_id("NONE")], np.int64), feature_hash_bins=gn.GLOBAL_MAX, hash_version=2)
    torch.manual_seed(int(rng.integers(1 << 30)))
    model = gn.NetGraph(len(vocab), len(edge_vocab), {"d_model": 32, "heads": 2, "ff": 64, "dropout": 0.0,
                                                      "head_hidden": 32})
    gs.save_weights(path, model, vocab, edge_vocab, {})
    return path


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


# ------------------------------------------------------------------------------------------------ batches, rows

def test_pack_marks_the_starting_networks_seats_as_not_training(tmp_path):
    rng = np.random.default_rng(0)
    fake_chunk(tmp_path / "c", rng, pairs=2, decisions=3)
    s = records.pack(tmp_path / "c", tmp_path / "b.jsonl.gz", run="r", machine="m", chunk="k", version=5,
                     opponent_version=0)
    assert s["games"] == 4 and s["seat_games"] == 8 and s["train_seat_games"] == 4
    lines = list(records.read(tmp_path / "b.jsonl.gz"))
    for sg in lines:
        bot1_seat = "B" if sg["game"]["swap"] else "A"
        assert sg["version"] == (5 if sg["seat"] == bot1_seat else 0) and sg["train"] == (sg["version"] == 5)


def test_graph_rows_targets_select_concat_and_table(tmp_path):
    rng = np.random.default_rng(1)
    fake_chunk(tmp_path / "c", rng, pairs=2, decisions=5)
    records.pack(tmp_path / "c", tmp_path / "b.jsonl.gz", run="r", machine="m", chunk="k", version=1)
    rows = tables.graph_rows(tmp_path / "b.jsonl.gz", lam=0.9, heldout_share=0.5)
    n = tables.n_rows(rows)
    assert n == 8 * 5 and rows["node_ptr"][-1] == len(rows["ids"]) and rows["row_opt_ptr"][-1] == len(rows["opt_p"])
    sums = np.add.reduceat(rows["opt_p"], rows["row_opt_ptr"][:-1])
    assert np.allclose(sums, 1)
    # TD targets: the last decision is lam x result + (1 - lam) x its root value, every one within [-1, 1]
    assert np.all(np.abs(rows["z_td"]) <= 1)
    both = tables.concat([rows, rows])
    assert tables.n_rows(both) == 2 * n and both["opt_ptr"][-1] == len(both["opt_node"])
    half = tables.select(both, np.arange(2 * n) % 2 == 0)
    assert tables.n_rows(half) == n and half["node_ptr"][-1] == len(half["ids"])
    assert np.allclose(np.add.reduceat(half["opt_p"], half["row_opt_ptr"][:-1]), 1)
    ck = tiny_checkpoint(tmp_path / "ck.pt.gz", rng)
    _, vocab, edge_vocab, _ = gs.load_checkpoint(ck)
    t = tables.to_graph_table(rows, vocab, edge_vocab)
    b = gs.make_batch([t], [(0, np.arange(5))])
    assert b["graph_type"].shape[0] == 5 and t.opt_in_set.sum() >= n


def test_td_targets_are_magezeros_labels():
    zt = tables.td_targets(np.asarray([0.2, -0.4]), 1.0, 0.5)
    assert np.allclose(zt, [0.5 * (0.5 * 1 + 0.5 * -0.4) + 0.5 * 0.2, 0.5 * 1 + 0.5 * -0.4])


# ------------------------------------------------------------------------------------------------ GNN training

def test_gnn_update_learns_the_search_and_round_trips(tmp_path):
    from draftzero.selfplay import gnn_train
    rng = np.random.default_rng(2)
    fake_chunk(tmp_path / "c", rng, pairs=3, decisions=6)
    records.pack(tmp_path / "c", tmp_path / "b.jsonl.gz", run="r", machine="m", chunk="k", version=0)
    rows = tables.graph_rows(tmp_path / "b.jsonl.gz", heldout_share=0.0)
    ck = tiny_checkpoint(tmp_path / "v0.pt.gz", rng)
    cfg = {**sc.DEFAULTS["trainer"], "device": "cpu", "batch_rows": 16, "lr": 3e-3, "warmup_steps": 1,
           "reuse": 6.0, "kl_weight": 0.0, "kl_weight_end": 0.0}
    tr = gnn_train.GnnTrainer(ck, ck, cfg)
    rows["heldout"][:] = True
    before = tr.evaluate(rows)
    rows["heldout"][:] = False
    m = tr.train(rows, new_positions=tables.n_rows(rows))
    rows["heldout"][:] = True
    after = tr.evaluate(rows)
    assert m["steps"] >= 10 and np.isfinite(m["loss"]) and after["ce"] < before["ce"]
    assert after["kl_start"] > 0 and before["kl_start"] < 1e-4
    tr.save_weights(tmp_path / "v1.pt.gz", {"version": 1})
    tr.full_state(tmp_path / "full.pt.gz", {"version": 1})
    model, *_ = gs.load_checkpoint(tmp_path / "v1.pt.gz")
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in tr.model.parameters())
    tr2 = gnn_train.GnnTrainer(ck, ck, cfg, full_state=tmp_path / "full.pt.gz")
    assert tr2.step == tr.step and tr2.updates == 1


def test_value_mask_caps_positions_per_game():
    from draftzero.selfplay.gnn_train import value_mask
    game = np.repeat([1, 2, 3], [10, 2, 5])
    mask = value_mask(game, 3, np.random.default_rng(0))
    assert [int(mask[game == g].sum()) for g in (1, 2, 3)] == [3, 2, 3]


# ------------------------------------------------------------------------------------------------ controller, trainer

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


def test_trainer_bootstrap_lease_due_and_prune(tmp_path):
    store = LocalStore(tmp_path / "st")
    start = tmp_path / "start.pt.gz"
    start.write_bytes(b"weights")
    cfg = sc.load(start=str(start), trainer={"min_new_games": 4, "publish_every_s": 100, "keep_versions": 2})
    tr = Trainer(store, cfg, work_dir=tmp_path / "w", log=lambda *_: None)
    tr.lease(now=0)
    other = Trainer(store, cfg, work_dir=tmp_path / "w2", log=lambda *_: None)
    with pytest.raises(LeaseHeld):
        other.lease(now=10)
    latest = tr.bootstrap()
    assert latest["version"] == 0 and store.read_bytes("weights/v0000.pt.gz") == b"weights"
    assert tr.bootstrap()["sha256"] == latest["sha256"]           # idempotent
    tr.pending_games, tr.pending_positions, tr.last_publish = 4, 0, 0
    assert not tr.due(150)                                        # held-out games alone make no version
    tr.pending_positions = 100
    assert not tr.due(50) and tr.due(150)
    tr.pending_games = 8
    assert tr.due(50)
    for v in range(1, 13):
        store.put(ctl.weights_path(v), b"w")
    tr.prune(12)
    left = sorted(int(p[9:13]) for p in store.list("weights/") if p.endswith(".pt.gz"))
    assert left == [0, 10, 11, 12]
