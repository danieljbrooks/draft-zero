"""Tests for the search benchmark's driver options for experiment #4: the leaf evaluator (leaf,
leafMix), the opponent's priors (opponentPriors) and the root values in the output
(java/mzbridge BenchSearch and Bench, tools/search_bench/run.py and analyze.py).

The bench tests need the bridge and skip cleanly without it. They run one worker on the golden
main_phase decision (the opponent holds two instants, so the search meets the opponent's
decisions), with a fake network served in-process: MageZero's protocol, a value hashed from the
state's features and fixed policy logits, so seeded network searches are deterministic.
"""
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from draftzero.gameplay import bridge
from draftzero.gameplay.bridge import BridgeError, golden_specs

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "search_bench"
PROBLEMS = bridge.environment_problems()
needs_worker = pytest.mark.skipif(bool(PROBLEMS), reason="mzbridge unavailable: " + "; ".join(PROBLEMS))


def tool(name: str):
    spec = importlib.util.spec_from_file_location(f"search_bench_{name}", TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------ no JVM needed

def test_run_ids_and_requests_carry_the_leaf_and_opponent_priors():
    run = tool("run")
    item = {"request": {"decisionPlayer": "A"}, "real": {"r": 1}, "worlds": [{"w": k} for k in range(8)], "build_seed": 3}

    def R(method="pimc1", evaluator="remote", priors=False, leaf="net", mix=0.5, opp="uniform"):
        r = {"method": method, "budget": 300, "evaluator": evaluator, "discount": 0.99, "unit": "ply", "seed": 0}
        if priors:
            r["priors"] = True
        return run.set_leaf(r, leaf, mix, opp)

    # experiment #3's runs keep their ids
    assert run.run_id(R()) == "pimc1-b300-remote-d0.99"
    assert run.run_id(R(priors=True, opp="net")) == "pimc1-b300-remote-d0.99-pri"
    assert run.run_id(R(evaluator="offline", leaf="heuristic")) == "pimc1-b300-offline-d0.99"
    # experiment #4's say what differs
    assert run.run_id(R(priors=True)) == "pimc1-b300-remote-d0.99-pri-oppu"
    assert run.run_id(R(priors=True, leaf="heuristic")) == "pimc1-b300-remote-d0.99-pri-oppu-leafh"
    assert run.run_id(R(leaf="mix", mix=0.25)) == "pimc1-b300-remote-d0.99-mix0.25"
    assert run.run_id(R(method="policy", leaf="heuristic")) == "policy-b300-remote"
    # opponentPriors only with priors; leafMix only with mix
    assert "opponentPriors" not in R() and R(priors=True)["opponentPriors"] == "uniform"
    assert "leafMix" not in R() and R(leaf="mix")["leafMix"] == 0.5

    _, opts = run.request_for(item, R(priors=True, leaf="mix", mix=0.25))
    assert opts["priors"] is True and opts["opponentPriors"] == "uniform"
    assert opts["leaf"] == "mix" and opts["leafMix"] == 0.25
    _, opts = run.request_for(item, R(evaluator="offline", leaf="heuristic"))
    assert opts["leaf"] == "heuristic" and "priors" not in opts and "opponentPriors" not in opts and "leafMix" not in opts


def test_policy_per_world_is_ismcts_with_priors_only():
    run = tool("run")
    item = {"request": {"decisionPlayer": "A"}, "real": {"r": 1}, "worlds": [{"w": k} for k in range(8)], "build_seed": 3}

    def R(method, priors=True, pw=True):
        r = {"method": method, "budget": 300, "evaluator": "remote", "discount": 0.99, "unit": "ply", "seed": 0}
        if priors:
            r["priors"] = True
        return run.set_leaf(r, "net", 0.5, "uniform", pw)

    assert run.run_id(R("ismcts")) == "ismcts-b300-remote-d0.99-pri-oppu-pw"
    assert run.run_id(R("ismcts", pw=False)) == "ismcts-b300-remote-d0.99-pri-oppu"
    assert "isPolicyPerWorld" not in R("pimc1") and "isPolicyPerWorld" not in R("ismcts", priors=False)
    _, opts = run.request_for(item, R("ismcts"))
    assert opts["isPolicyPerWorld"] is True
    _, opts = run.request_for(item, R("ismcts", pw=False))
    assert "isPolicyPerWorld" not in opts


def test_run_refuses_a_network_leaf_offline(tmp_path):
    run = tool("run")
    with pytest.raises(SystemExit):
        run.main(["--out", str(tmp_path / "x"), "--evaluator", "offline", "--leaf", "net"])


def test_analyze_tells_experiment_4_runs_apart():
    analyze = tool("analyze")
    old_remote = {"evaluator": "remote", "priors": True}           # a row from before experiment #4
    old_offline = {"evaluator": "offline"}
    assert analyze.leaf_of(old_remote) == "net" and analyze.leaf_of(old_offline) == "heuristic"
    assert analyze.exp3_style(old_remote) and analyze.exp3_style(old_offline)
    assert analyze.exp3_style({"evaluator": "remote", "leaf": "net", "priors": True, "opponentPriors": "net"})
    assert not analyze.exp3_style({"evaluator": "remote", "leaf": "net", "priors": True, "opponentPriors": "uniform"})
    assert not analyze.exp3_style({"evaluator": "remote", "leaf": "heuristic"})
    assert not analyze.exp3_style({"evaluator": "remote", "leaf": "mix", "leafMix": 0.5})
    assert analyze.exp3_style({"evaluator": "remote", "leaf": "net", "opponentPriors": "uniform"})  # priors off


# ------------------------------------------------------------------------------ a fake network

class FakeNet:
    """MageZero's inference protocol (msgpack POST /evaluate, GET /healthz) with made-up heads: a
    value in [-1, 1] hashed from the state's features, and policy logits fixed per action index
    ("hash", different for the player and opponent heads) or all zero ("zero")."""

    WIDTH = 1024

    def __init__(self):
        import msgpack
        self.logits = "hash"
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

            def do_GET(self):  # /healthz
                self._send(b"ok", "text/plain")

            def do_POST(self):  # /evaluate
                data = msgpack.unpackb(self.rfile.read(int(self.headers["Content-Length"])), raw=False)
                self._send(msgpack.packb(net.answer(data), use_bin_type=True), "application/x-msgpack")

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def answer(self, data: dict):
        idx = data.get("indices") or []
        off = data.get("offsets") or [0]
        bags = [idx[a:b] for a, b in zip(off, list(off[1:]) + [len(idx)])]
        self.calls += len(bags)
        outs = [self.one(b) for b in bags]
        return outs[0] if len(outs) == 1 else outs

    def one(self, bag: list) -> dict:
        value = (sum((i * 2654435761) % 1000003 for i in bag) % 2001) / 1000.0 - 1.0
        if self.logits == "zero":
            pl = po = pt = [0.0] * self.WIDTH
            pb = [0.0, 0.0]
        else:
            pl = [((i * 7919) % 101) / 25.0 for i in range(self.WIDTH)]
            po = [((i * 104729) % 97) / 25.0 for i in range(self.WIDTH)]
            pt = [((i * 31) % 53) / 20.0 for i in range(self.WIDTH)]
            pb = [0.4, -0.3]
        return {"policy_player": pl, "policy_opponent": po, "policy_target": pt, "policy_binary": pb, "value": value}

    def close(self):
        self.server.shutdown()


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    p = bridge.BridgePool(1, prefix="pytest_sb", heap="2g", runtime_root=tmp_path_factory.mktemp("mzbridge"))
    yield p
    p.close()


@pytest.fixture(scope="module")
def net():
    pytest.importorskip("msgpack")
    n = FakeNet()
    yield n
    n.close()


SPEC = None


def bench(pool, method="tree", worlds=1, budget=60, evaluator=None, **opts) -> dict:
    global SPEC
    if SPEC is None:
        SPEC = json.loads(golden_specs()["main_phase"].read_text())
    ev = {"type": "remote", "host": "127.0.0.1", "port": evaluator.port} if evaluator else {"type": "offline"}
    return pool.request("bench", None, specs=[SPEC] * worlds, method=method, budget=budget, seed=7,
                        decisionPlayer="A", evaluator=ev, **opts)


def tree_of(r: dict) -> list:
    return [(c["label"], c["N"], c["Q"]) for c in r["children"]]


def assert_same_search(a: dict, b: dict) -> None:
    assert tree_of(a) == tree_of(b)
    assert a["best"] == b["best"] and a["rootQ"] == b["rootQ"] and a["bestQ"] == b["bestQ"]
    assert a["rootValue"] == b["rootValue"]


# clairvoyant-style (one tree), PIMC with 2 worlds, IS-MCTS over 4 worlds (one shared tree,
# re-dealt every iteration, so the opponent's hand and who gets to act at a node vary by world)
METHODS = [("tree", 1), ("tree", 2), ("ismcts", 4)]


# ------------------------------------------------------------------------------ the bench op

@needs_worker
@pytest.mark.parametrize("method,worlds", METHODS)
def test_offline_bench_is_deterministic_and_reports_root_values(pool, method, worlds):
    a = bench(pool, method, worlds)
    b = bench(pool, method, worlds)
    assert_same_search(a, b)
    assert a["settings"]["leaf"] == "heuristic" and a["rootNetValue"] is None
    assert a["stats"]["netEvals"] == 0 and a["stats"]["netPriors"] == 0
    assert isinstance(a["rootQ"], float) and isinstance(a["rootValue"], float)
    assert a["best"] == a["children"][0]["label"] and a["bestQ"] == a["children"][0]["Q"]
    # every simulation passes through a root option and is discounted once on the way to the root
    # (0.99 per ply), so the backed-up root value is 0.99 x the options' visit-weighted Q: pooled
    # over the worlds' roots for PIMC, IS-MCTS's single shared root
    kids = [c for c in a["children"] if c["N"]]
    mean_q = sum(c["N"] * c["Q"] for c in kids) / sum(c["N"] for c in kids)
    assert a["rootQ"] == pytest.approx(0.99 * mean_q, abs=1e-9)
    assert sum(c["N"] for c in kids) == a["stats"]["sims"]


@needs_worker
@pytest.mark.parametrize("method,worlds", METHODS)
def test_leaf_heuristic_with_a_network_matches_offline(pool, net, method, worlds):
    off = bench(pool, method, worlds)
    # priors off: the network is not needed by the search (one call for the reported root value)
    calls = net.calls
    h = bench(pool, method, worlds, evaluator=net, leaf="heuristic")
    assert_same_search(off, h)
    assert h["settings"]["leaf"] == "heuristic" and h["rootNetValue"] is not None
    assert net.calls - calls == h["stats"]["netEvals"] == (1 if method == "ismcts" else worlds)
    # priors on, but uniform ones (zero logits, no bonus off Pass): the network is called at every
    # node with a policy head and its value, never 0 here, must be ignored
    net.logits = "zero"
    try:
        hp = bench(pool, method, worlds, evaluator=net, leaf="heuristic", priors=True, priorBonus=0.0)
    finally:
        net.logits = "hash"
    assert_same_search(off, hp)
    assert hp["stats"]["netEvals"] > 5 and hp["stats"]["netPriors"] > 0
    # mix at lambda 0 is the heuristic
    m0 = bench(pool, method, worlds, evaluator=net, leaf="mix", leafMix=0.0)
    assert_same_search(off, m0)
    assert m0["settings"]["leafMix"] == 0.0


@needs_worker
@pytest.mark.parametrize("method,worlds", METHODS)
def test_leaf_net_is_the_default_and_mix_at_one_is_net(pool, net, method, worlds):
    n = bench(pool, method, worlds, evaluator=net)
    assert n["settings"]["leaf"] == "net"
    assert n["rootNetValue"] == n["rootValue"]  # the root's static value is the network's
    assert_same_search(n, bench(pool, method, worlds, evaluator=net, leaf="net"))
    assert_same_search(n, bench(pool, method, worlds, evaluator=net, leaf="mix", leafMix=1.0))
    h = bench(pool, method, worlds, evaluator=net, leaf="heuristic")
    assert h["rootQ"] != n["rootQ"]
    assert h["rootNetValue"] == n["rootNetValue"]
    m = bench(pool, method, worlds, evaluator=net, leaf="mix", leafMix=0.5)
    if method != "ismcts":  # IS-MCTS scores the root in a re-dealt world: compare tree roots only
        assert m["rootValue"] == pytest.approx(0.5 * n["rootValue"] + 0.5 * h["rootValue"], abs=1e-6)


@needs_worker
@pytest.mark.parametrize("method,worlds", METHODS)
@pytest.mark.parametrize("leaf", ["net", "heuristic"])
def test_uniform_opponent_priors(pool, net, method, worlds, leaf):
    on = bench(pool, method, worlds, budget=120, evaluator=net, leaf=leaf, priors=True, opponentPriors="net")
    assert on["settings"]["opponentPriors"] == "net"
    assert on["stats"]["oppNetPriors"] > 0  # the search meets the opponent's decisions
    u = bench(pool, method, worlds, budget=120, evaluator=net, leaf=leaf, priors=True, opponentPriors="uniform")
    assert u["settings"]["opponentPriors"] == "uniform"
    assert u["stats"]["oppNetPriors"] == 0 and u["stats"]["netPriors"] > 0  # still the searcher's own
    assert_same_search(u, bench(pool, method, worlds, budget=120, evaluator=net, leaf=leaf, priors=True,
                                opponentPriors="uniform"))
    if leaf == "heuristic":  # the network is skipped where its policy would go unused
        assert u["stats"]["netEvals"] < u["stats"]["evals"]


@needs_worker
def test_policy_reports_the_network_value(pool, net):
    p = bench(pool, "policy", evaluator=net)
    assert p["rootNetValue"] == p["rootValue"] is not None
    assert p["rootQ"] is None and p["bestQ"] is None and p["best"] is not None
    # the root is the searcher's: the opponent's priors don't change it
    u = bench(pool, "policy", evaluator=net, opponentPriors="uniform")
    assert [(c["label"], c["prior"]) for c in u["children"]] == [(c["label"], c["prior"]) for c in p["children"]]


@needs_worker
def test_bad_leaf_options_fail_clearly(pool, net):
    with pytest.raises(BridgeError, match="needs a network"):
        bench(pool, leaf="net")
    with pytest.raises(BridgeError, match="needs a network"):
        bench(pool, leaf="mix")
    with pytest.raises(BridgeError, match="leafMix"):
        bench(pool, evaluator=net, leaf="mix", leafMix=1.5)
    with pytest.raises(BridgeError, match="opponentPriors"):
        bench(pool, evaluator=net, priors=True, opponentPriors="both")
    with pytest.raises(BridgeError, match="leaf must be"):
        bench(pool, evaluator=net, leaf="value")


@needs_worker
def test_is_policy_per_world(pool, net):
    """IS-MCTS shares a node across worlds, though its actor and decision type can differ between
    them: isPolicyPerWorld keeps one policy per (actor, decision type), so no prior comes from a
    policy read for another actor or decision type."""
    kw = dict(budget=200, evaluator=net, priors=True, opponentPriors="net")
    old = bench(pool, "ismcts", 4, **kw)
    assert old["settings"]["isPolicyPerWorld"] is False and old["stats"]["policyRefreshes"] == 0
    new = bench(pool, "ismcts", 4, isPolicyPerWorld=True, **kw)
    assert new["settings"]["isPolicyPerWorld"] is True
    assert new["stats"]["policyMismatches"] == 0 and new["stats"]["netPriors"] > 0
    assert_same_search(new, bench(pool, "ismcts", 4, isPolicyPerWorld=True, **kw))   # deterministic
    if old["stats"]["policyMismatches"] == 0 and new["stats"]["policyRefreshes"] == 0:
        assert_same_search(old, new)                    # nothing to fix: the same search
    # one tree per world (PIMC): nothing is shared, so the option changes nothing
    assert_same_search(bench(pool, "tree", 2, **kw), bench(pool, "tree", 2, isPolicyPerWorld=True, **kw))
