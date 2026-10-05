"""MageZero's graph network in DraftZero (docs/022): the bridge's graph encodings, the graph tables, the
network (graph_net), its imitation trainer (graph_supervised), and graph bots in games. The bridge tests
need the bridge and skip cleanly without it."""
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import torch

from draftzero.gameplay import bridge
from draftzero.gameplay import graph_net as gn
from draftzero.gameplay import graph_supervised as gs
from draftzero.gameplay import graph_tables as gt
from draftzero.gameplay.bridge import golden_specs

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "imitation_scale"
DECKS = Path(__file__).resolve().parents[1] / "assets" / "sample" / "decks"
PROBLEMS = bridge.environment_problems()
needs_worker = pytest.mark.skipif(bool(PROBLEMS), reason="mzbridge unavailable: " + "; ".join(PROBLEMS))


def tool(name: str):
    spec = importlib.util.spec_from_file_location(f"imitation_scale_{name}", TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------
# toy graphs: a root, two players, a hand zone per player, cards with an ability each, shared leaves
# ------------------------------------------------------------------------------------------------

TID = {t: gn.feature_id(t.name) for t in gn.NodeType if t not in (gn.NodeType.ROOT, gn.NodeType.LEAF)}


def toy_graph(rng, n_cards: int, kind: str = "priority") -> dict:
    """A decoded graph (graph_tables.decode's shape). Options: priority -> [Pass, each card's ability];
    target -> [Stop Choosing, each card]; attack -> [Stop Choosing, the opponent]. The first card (option
    1, the toy human's choice) carries a leaf no other object has."""
    ids, values, child, parent = [0], [0], [], []

    def node(i, v=0):
        ids.append(i)
        values.append(v)
        return len(ids) - 1

    def edge(c, p):
        child.append(c)
        parent.append(p)
    pa, pb = node(TID[gn.NodeType.PLAYER]), node(TID[gn.NodeType.PLAYER])
    edge(pa, 0)
    edge(pb, 0)
    stop = node(TID[gn.NodeType.CARD])
    edge(stop, 0)
    pas = node(TID[gn.NodeType.ABILITY])                 # the Pass node, under both players
    edge(pas, pa)
    edge(pas, pb)
    zone = node(TID[gn.NodeType.ZONE])
    edge(zone, pa)
    leaves = [node(int(x)) for x in rng.integers(10_000, 10_040, 12)]
    life = node(777, int(rng.integers(1, 21)))
    edge(life, pa)
    mark = node(55555)                                    # a leaf only the human's choice has
    cards, abil = [], []
    for k in range(n_cards):
        c = node(TID[gn.NodeType.CARD])
        edge(c, zone)
        a = node(TID[gn.NodeType.ABILITY])
        edge(a, c)
        for lf in rng.choice(leaves, 3, replace=False):
            edge(int(lf), c)
        edge(int(rng.choice(leaves)), a)
        if k == 0:
            edge(mark, c)
            edge(mark, a)
        cards.append(c)
        abil.append(a)
    if kind == "priority":
        opts, typ = [[pas]] + [[a] for a in abil], 0
    elif kind == "target":
        opts, typ = [[stop]] + [[c] for c in cards], 3
    else:
        opts, typ = [[stop], [pb]], 3
    return {"type": typ, "ids": np.asarray(ids, np.int32), "values": np.asarray(values, np.int16),
            "child": np.asarray(child, np.uint16), "parent": np.asarray(parent, np.uint16),
            "label": np.asarray([gn.feature_id("NONE")] * len(child), np.int32),
            "opt_node": np.asarray([k for o in opts for k in o], np.int32),
            "opt_len": np.asarray([len(o) for o in opts], np.int32), "missing": 0}


def write_toy_tables(d: Path, split: str, first_game: int, n_games: int, rng) -> None:
    """Flat tables (labels, results) and graph files for turnstart (priority_set), replay_target
    (target) and replay_attack (binary), row for row; the human's choice is option 1."""
    import h5py
    from draftzero.gameplay import imitation as im
    d.mkdir(parents=True, exist_ok=True)
    for name, kind in (("turnstart", "priority"), ("replay_target", "target"), ("replay_attack", "attack")):
        graphs, labels, sets, games, turns = [], [], [], [], []
        for g in range(first_game, first_game + n_games):
            for turn in range(1, 4):
                gr = toy_graph(rng, int(rng.integers(2, 4)), kind)
                n_opt = len(gr["opt_len"])
                lab = ["Pass" if (kind == "priority" and j == 0) else f"opt{j}" for j in range(n_opt)]
                graphs.append(gr)
                labels.append(lab)
                sets.append([lab[1]])
                games.append(g)
                turns.append(turn)
        n = len(graphs)
        path = d / f"{name}_{split}.h5"
        z = np.where(np.asarray(games) % 2 == 0, 1.0, -1.0).astype(np.float32)
        if kind == "attack":
            with h5py.File(path, "w") as f:
                f.create_dataset("indices", data=np.arange(n, dtype=np.int32))
                f.create_dataset("offsets", data=np.arange(n + 1, dtype=np.int64))
                f.create_dataset("y", data=np.ones(n, np.int64))
                f.create_dataset("meta/row", data=np.asarray(games, np.int32))
                f.create_dataset("meta/turn", data=np.asarray(turns, np.int32))
                f.create_dataset("meta/won", data=(z > 0).astype(np.int32))
        else:
            legal = [[0] + list(range(1, len(lab))) if kind == "priority" else list(range(1, len(lab) + 1)) for lab in labels]
            t = {"indices": np.arange(n, dtype=np.int32), "offsets": np.arange(n + 1, dtype=np.int64),
                 "legal_idx": np.asarray([x for li in legal for x in li], np.int32),
                 "legal_indptr": np.r_[0, np.cumsum([len(li) for li in legal])].astype(np.int64),
                 "set_idx": np.asarray([li[1] for li in legal], np.int32),
                 "set_indptr": np.arange(n + 1, dtype=np.int64), "z": z,
                 "meta/label_status": np.ones(n, np.int32), "meta/row": np.asarray(games, np.int32),
                 "meta/turn": np.asarray(turns, np.int32), "labels": labels, "lab_idx": legal}
            im._save_table(t, path)
        flags = [gt.option_labels(lab, st) for lab, st in zip(labels, sets)]
        gt.write(gt.graph_path(path), graphs, [f[0] for f in flags], [f[1] for f in flags], games, turns)


TOY_TABLES = [{"name": "turnstart", "kind": "priority_set"}, {"name": "replay_target", "kind": "target"},
              {"name": "replay_attack", "kind": "binary"}]
TINY = {"d_model": 32, "heads": 2, "ff": 64, "head_hidden": 16, "dropout": 0.0}


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    d = tmp_path_factory.mktemp("graph_h5")
    rng = np.random.default_rng(0)
    write_toy_tables(d, "train", 0, 30, rng)
    write_toy_tables(d, "val", 1000, 6, rng)
    return d


def toy_cfg(d: Path, **over) -> dict:
    return gs.resolve_config({"tables_dir": str(d), "tables": TOY_TABLES, "vocab_k": 0, "edge_vocab_k": 0,
                              "val_rows": None, "arch": TINY, "act_weights": {}, "device": "cpu", "warmup_steps": 2,
                              "batch_rows": 8, "prefetch": 0, "eval_at_start": False, "value_per_game": 2,
                              "td_start_epochs": 0.5, **over})


# ------------------------------------------------------------------------------------------------
# the network
# ------------------------------------------------------------------------------------------------

def test_node_types_and_feature_ids():
    assert gn.feature_id("CARD") == gn.feature_id("CARD") and gn.feature_id("CARD") != gn.feature_id("ZONE")
    ids = np.asarray([0, TID[gn.NodeType.ABILITY], 12345, TID[gn.NodeType.PLAYER]], np.int32)
    assert list(gn.node_types(ids)) == [gn.NodeType.ROOT, gn.NodeType.ABILITY, gn.NodeType.LEAF, gn.NodeType.PLAYER]
    with pytest.raises(ValueError):
        gn.full_arch({"nope": 1})
    assert gn.upstream_loadable({}) and not gn.upstream_loadable({"passes": 1})


def test_segment_logsumexp_and_option_logits_pool_copies():
    x = torch.tensor([0.5, -1.0, 2.0, 3.0])
    seg = torch.tensor([0, 0, 2, 2])
    out = gn.segment_logsumexp(x, seg, 3)
    assert torch.allclose(out[0], torch.logsumexp(x[:2], 0)) and torch.isinf(out[1]) and out[1] < 0
    assert torch.allclose(out[2], torch.logsumexp(x[2:], 0))
    # two copies of a card are one option: their scores add in probability
    o = gn.Out(priority=torch.tensor([0.0, 1.0, 1.0, 2.0]), target=torch.zeros(4), use=torch.tensor([[0.1, 0.7]]),
               value=torch.zeros(1), value_x=torch.zeros(1))
    ob = gn.OptionBatch(torch.tensor([0, 0, 0]), torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 1, 2]), 3)
    lg = gn.option_logits(o, ob, torch.tensor([0]), torch.tensor([0, 1, 2]))
    assert torch.allclose(lg, torch.tensor([0.0, float(torch.logsumexp(torch.tensor([1.0, 1.0]), 0)), 2.0]))
    use = gn.option_logits(o, gn.OptionBatch(torch.tensor([0, 0]), torch.zeros(0, dtype=torch.long),
                                             torch.zeros(0, dtype=torch.long), 2), torch.tensor([5]), torch.tensor([0, 1]))
    assert torch.allclose(use, torch.tensor([0.1, 0.7]))


# ------------------------------------------------------------------------------------------------
# tables and the trainer
# ------------------------------------------------------------------------------------------------

def test_graph_tables_merge_and_read_back(tmp_path):
    build = tool("build")
    rng = np.random.default_rng(1)
    graphs = [toy_graph(rng, k) for k in (2, 3, 2, 3, 2)]
    labs = [["Pass"] + [f"o{j}" for j in range(1, len(g["opt_len"]))] for g in graphs]
    flags = [gt.option_labels(lab, [lab[1]]) for lab in labs]
    parts = [(0, 2), (2, 5)]
    paths = []
    for i, (a, b) in enumerate(parts):
        paths.append(tmp_path / f"p{i}.graph.h5")
        gt.write(paths[-1], graphs[a:b], [f[0] for f in flags[a:b]], [f[1] for f in flags[a:b]],
                 list(range(a, b)), [1] * (b - a))
    merged = tmp_path / "all.graph.h5"
    assert build.merge_h5(paths, merged) == 5
    sel = np.asarray([1, 3, 4])
    g = gs._read_graph(merged, sel, block=2)
    for k, r in enumerate(sel):
        a, b = g["node_ptr"][k], g["node_ptr"][k + 1]
        assert np.array_equal(g["ids"][a:b], graphs[r]["ids"])
        e0, e1 = g["edge_ptr"][k], g["edge_ptr"][k + 1]
        assert np.array_equal(g["child"][e0:e1], graphs[r]["child"])
        o0, o1 = g["row_opt_ptr"][k], g["row_opt_ptr"][k + 1]
        assert np.array_equal(g["in_set"][o0:o1], flags[r][0].astype(bool))
        nodes = g["opt_node"][g["opt_ptr"][o0]:g["opt_ptr"][o1]]
        assert np.array_equal(nodes, graphs[r]["opt_node"])
    assert gt.sizes(merged)["rows"] == 5


def test_labels_view_puts_pass_first_for_supervised_measures():
    legal_ptr, legal, set_ptr, sset = gs._labels_view(np.asarray([0, 3, 5]), np.asarray([0, 1, 0, 1, 1], bool),
                                                      np.asarray([1, 0, 0, 0, 0], bool))
    assert list(legal) == [0, 2, 3, 1, 2] and list(set_ptr) == [0, 1, 3] and list(sset) == [2, 1, 2]


def test_graph_trainer_learns_on_toy_tables_and_checkpoints_round_trip(toy, tmp_path):
    cfg = toy_cfg(toy, max_epochs=6, lr=3e-3)
    data = gs.load_data(cfg, log=lambda *a, **k: None)
    assert [t.n for t in data.train] == [90, 90, 90] and [t.n for t in data.val] == [18, 18, 18]
    tr = gs.Trainer(cfg, data, tmp_path / "run", log=lambda *a, **k: None)
    before = gs.evaluate(tr.model, data.val, cfg, tr.dev)
    s = tr.run()
    assert s["stop"] == "max_epochs" and s["td_refreshes"] >= 1
    after = json.loads((tmp_path / "run" / "evals.jsonl").read_text().splitlines()[-1])
    # the human always chose option 1: the network learns it
    assert after["policy/set_nll"] < before["policy/set_nll"] and after["replay_target/top1"] > 0.9
    assert after["binary/acc"] == 1.0 and "value/auc" in after
    model, vocab, edge_vocab, meta = gs.load_checkpoint(tmp_path / "run" / "final.pt.gz")
    assert meta["arch"]["d_model"] == 32 and len(vocab) == len(data.vocab)
    b = gs.make_batch(data.val, [(0, np.arange(4))])
    with torch.no_grad():
        tr.model.eval()
        assert torch.allclose(model(b["graphs"]).value, tr.model(b["graphs"]).value, atol=1e-6)
    # resuming a finished run picks up its step count
    tr2 = gs.Trainer(cfg, data, tmp_path / "run", log=lambda *a, **k: None, resume=True)
    assert tr2.step == s["step"]


def test_graph_trainer_rejects_misaligned_tables(toy, tmp_path):
    import shutil
    d = tmp_path / "bad"
    shutil.copytree(toy, d)
    rng = np.random.default_rng(5)
    write_toy_tables(tmp_path / "other", "val", 5000, 6, rng)
    shutil.copy(tmp_path / "other" / "turnstart_val.graph.h5", d / "turnstart_val.graph.h5")
    with pytest.raises(ValueError, match="line up"):
        gs.load_data(toy_cfg(d), splits=("val",), vocabs=(gs.build_vocabs([], 0, 0)), log=lambda *a, **k: None)


def test_play_graph_bots_read_the_graph_server():
    play = tool("play")
    o = play.seat_options("gnn@300", 1000, 50052, 300, graph_port=50062)
    assert o["evaluator"] == {"type": "graph", "host": "127.0.0.1", "port": 50062} and o["budget"] == 300
    assert o["priors"] and o["opponentPriors"] == "uniform" and o["isPolicyPerWorld"]
    g = play.seat_options("gnn_policy_greedy", 1000, 50052, 300, graph_port=50062)
    assert g["policyOnly"] and g["policyTemp"] == 0.0
    assert play.seat_options("il_bc", 1000, 50052, 300, graph_port=50062)["evaluator"]["port"] == 50052


# ------------------------------------------------------------------------------------------------
# the bridge
# ------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def worker(tmp_path_factory):
    w = bridge.Bridge("pytest_graph", heap="2g", runtime_root=tmp_path_factory.mktemp("mzbridge_graph"))
    yield w
    w.close()


def spec(name: str) -> dict:
    return json.loads(golden_specs()[name].read_text())


@needs_worker
def test_encode_graph_maps_every_option_to_nodes(worker):
    flat = worker.request("encode", spec("main_phase"), seed=1)
    r = worker.request("encode", spec("main_phase"), seed=1, graph=True)
    assert r["features"] == flat["features"]                          # the graph changes nothing else
    g = gt.decode(r["graph"])
    assert g["type"] == 0 and g["missing"] == 0
    types = gn.node_types(g["ids"])
    assert (types == gn.NodeType.ROOT).sum() == 1 and (types == gn.NodeType.PLAYER).sum() == 2
    assert len(g["opt_len"]) == len(r["decision"]["legal"]) and (g["opt_len"] >= 1).all()
    assert set(types[g["opt_node"]]) == {gn.NodeType.ABILITY}        # plays and Pass are ability nodes
    assert len(g["child"]) == len(g["label"]) and g["child"].max() < len(g["ids"])
    full = gt.decode(worker.request("encode", spec("main_phase"), seed=1, graph=True, perfectInfo=True)["graph"])
    assert len(full["ids"]) > len(g["ids"])                           # the opponent's hand adds nodes
    blk = worker.request("encode", spec("block"), seed=1, decisionPlayer="A", graph=True)
    gb = gt.decode(blk["graph"])
    assert gb["type"] == 3 and gb["missing"] == 0
    labels = [x["label"] for x in blk["decision"]["legal"]]
    stop = gb["opt_node"][np.r_[0, np.cumsum(gb["opt_len"])][labels.index("Stop Choosing")]]
    assert gn.node_types(gb["ids"][[stop]])[0] == gn.NodeType.CARD   # Stop Choosing is a CARD node


class FakeGraphNet:
    """MageZero's graph server protocol with made-up outputs: per-node scores hashed from the node ids,
    a value hashed from the state."""

    def __init__(self):
        import msgpack
        self.calls = 0
        net = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, body: bytes, ctype: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._send(b"ok", "text/plain")

            def do_POST(self):
                data = msgpack.unpackb(self.rfile.read(int(self.headers["Content-Length"])), raw=False)
                self._send(msgpack.packb(net.answer(data), use_bin_type=True, use_single_float=True),
                           "application/x-msgpack")

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def answer(self, data: dict) -> list:
        ids = data["indices"]
        starts = list(data["offsets"]) + [len(ids)]
        out = []
        for a, b in zip(starts, starts[1:]):
            self.calls += 1
            s = ids[a:b]
            sc = [((i * 2654435761) % 1009) / 300.0 for i in s]
            out.append({"policy_priority": sc, "policy_target": sc[::-1], "policy_binary": [0.4, -0.3],
                        "value": (sum(s) % 2001) / 1000.0 - 1.0})
        return out

    def close(self):
        self.server.shutdown()


@needs_worker
def test_play_op_with_a_graph_network(worker):
    net = FakeGraphNet()
    decks = [str(p.resolve()) for p in sorted(DECKS.glob("*.dck"))[:2]]
    try:
        gnn = {"budget": 8, "priors": True, "leaf": "net", "opponentPriors": "uniform", "isPolicyPerWorld": True,
               "evaluator": {"type": "graph", "host": "127.0.0.1", "port": net.port}}
        heur = {"budget": 8, "evaluator": {"type": "offline"}}
        r = worker.request("play", None, deckA=decks[0], deckB=decks[1], seatA=gnn, seatB=heur, seed=3, maxTurns=6,
                           timeout=900)
        pol = {"budget": 4, "policyOnly": True, "policyTemp": 0.0, "priors": True, "leaf": "net",
               "evaluator": {"type": "graph", "host": "127.0.0.1", "port": net.port}}
        p = worker.request("play", None, deckA=decks[0], deckB=decks[1], seatA=pol, seatB=heur, seed=3, maxTurns=6,
                           timeout=900)
        # PIMC (docs/019 §4.6): MageZero's tree search on one world, with the graph network's priors and values
        pm = worker.request("play", None, deckA=decks[0], deckB=decks[1], seatA={**gnn, "method": "pimc"},
                            seatB={**heur, "method": "pimc"}, seed=3, maxTurns=6, timeout=900)
    finally:
        net.close()
    a = r["seats"]["A"]
    assert a["fallbacks"] == 0 and a["sims"] == 8 * a["decisions"] and a["netEvals"] >= a["sims"]
    assert a["netPriors"] > 0 and a["graphPolicyMisses"] >= 0 and r["seats"]["B"]["netEvals"] == 0
    pa = p["seats"]["A"]
    assert pa["policyDecisions"] > 0 and pa["decisions"] == pa["policyDecisions"] + pa["policySearched"]
    ma = pm["seats"]["A"]
    assert ma["fallbacks"] == 0 and ma["decisions"] > 0 and ma["sims"] == 8 * ma["decisions"]
    assert ma["netEvals"] >= ma["sims"] and ma["netPriors"] > 0


# ------------------------------------------------------------------------------------------------
# stage 1 and 4 tools: comparing builds, slim tables, held-out games
# ------------------------------------------------------------------------------------------------

def _reorder_games(src: Path, dst: Path) -> None:
    """dst: the flat tables of src with their games in reverse order (as another build would write them)."""
    import h5py
    build = tool("build")
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.glob("*_val.h5")):
        if p.name.endswith(".graph.h5"):
            continue
        with h5py.File(p, "r") as f:
            g = f["meta/row"][:]
            games = list(dict.fromkeys(g.tolist()))[::-1]
            order = np.concatenate([np.flatnonzero(g == x) for x in games])
            with h5py.File(dst / p.name, "w") as h:
                for k in build._h5_datasets(f):
                    d = f[k][()]
                    if k in ("offsets", "legal_indptr", "set_indptr"):
                        lens = np.diff(d)[order]
                        h.create_dataset(k, data=np.r_[0, np.cumsum(lens)].astype(d.dtype))
                    elif k in ("indices", "legal_idx", "set_idx"):
                        ptr = f[{"indices": "offsets", "legal_idx": "legal_indptr", "set_idx": "set_indptr"}[k]][()]
                        h.create_dataset(k, data=np.concatenate([d[ptr[r]:ptr[r + 1]] for r in order]).astype(d.dtype))
                    elif d.ndim and d.shape[0] == len(g):
                        h.create_dataset(k, data=d[order])
                    else:
                        h.create_dataset(k, data=d)


def test_compare_builds_game_by_game_and_slim_tables(toy, tmp_path):
    import h5py
    build = tool("build")
    _reorder_games(toy, tmp_path / "b")
    res = build.compare(toy, tmp_path / "b", log=lambda *a, **k: None)
    assert res and all(v["same"] for v in res.values())                 # another game order, the same rows
    with h5py.File(tmp_path / "b" / "turnstart_val.h5", "a") as f:
        f["indices"][0] = f["indices"][0] + 1                            # one feature of one row changed
    assert build.compare(toy, tmp_path / "b", ["turnstart_val.h5"], log=lambda *a, **k: None)["turnstart_val.h5"][
        "games_differing"] == 1
    out = build.slim(toy, tmp_path / "slim", log=lambda *a, **k: None)
    assert out["files"] == len(list(toy.glob("*.h5")))
    with h5py.File(tmp_path / "slim" / "turnstart_val.h5", "r") as f:
        assert "indices" not in f and "set_idx" in f and "offsets" in f
    a = gs.load_data(toy_cfg(toy), splits=("val",), vocabs=gs.build_vocabs([], 0, 0), log=lambda *a, **k: None)
    b = gs.load_data(toy_cfg(tmp_path / "slim"), splits=("val",), vocabs=gs.build_vocabs([], 0, 0),
                     log=lambda *a, **k: None)
    assert all(np.array_equal(x.file_rows, y.file_rows) and np.array_equal(x.w, y.w) for x, y in zip(a.val, b.val))


def test_exclude_games_drops_them_from_training_only(toy, tmp_path):
    ex = np.asarray([0, 1, 2, 1000], np.int64)                           # three training games, one validation game
    np.save(tmp_path / "ex.npy", ex)
    d = gs.load_data(toy_cfg(toy, exclude_games=str(tmp_path / "ex.npy")), log=lambda *a, **k: None)
    assert all(not np.isin(t.game, ex).any() for t in d.train) and [t.n for t in d.train] == [81, 81, 81]
    assert any(np.isin(t.game, ex).any() for t in d.val)                 # validation is untouched
