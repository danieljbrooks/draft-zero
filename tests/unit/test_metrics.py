"""draftzero.metrics: the metric line schema (interface 9), reading lines back, and the Wilson interval."""
from __future__ import annotations

import json
import math
import threading
from datetime import datetime

import pytest

from draftzero import metrics as M


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_log_writes_one_full_line(tmp_path):
    p = tmp_path / "metrics/learner/d1/000001.jsonl"
    log = M.MetricsLog(p, run="smoke", component="learner", machine="d1")
    x = log.log("learn.policy_ce", 1.25, version=3, step=40)
    assert lines(p) == [x]
    assert list(x)[:len(M.FIELDS)] == list(M.FIELDS)
    assert x["schema"] == M.SCHEMA == "draftzero-metrics/1"
    assert (x["run"], x["component"], x["machine"], x["name"], x["value"]) == ("smoke", "learner", "d1",
                                                                               "learn.policy_ce", 1.25)
    assert (x["version"], x["step"], x["n"], x["lo"], x["hi"]) == (3, 40, None, None, None)
    assert x["time"].endswith("Z") and datetime.fromisoformat(x["time"].replace("Z", "+00:00")).tzinfo is not None


def test_tags_are_extra_keys_and_may_not_shadow_fields(tmp_path):
    log = M.MetricsLog(tmp_path / "m.jsonl", run="r", component="evaluator", machine="m")
    x = log.log("eval.score", 0.56, version=4, n=100, lo=0.49, hi=0.63, opponent="prev", sims=100)
    assert x["opponent"] == "prev" and x["sims"] == 100 and M.tags(x) == {"opponent": "prev", "sims": 100}
    with pytest.raises(ValueError, match="field names"):
        log.log("eval.score", 0.5, run="other")
    assert len(lines(tmp_path / "m.jsonl")) == 1


def test_non_finite_values_become_null(tmp_path):
    log = M.MetricsLog(tmp_path / "m.jsonl", run="r", component="learner", machine="m")
    assert log.log("learn.value_logloss", float("nan"))["value"] is None
    assert log.log("learn.value_logloss", float("inf"), lo=float("-inf"))["lo"] is None
    assert [x["value"] for x in lines(tmp_path / "m.jsonl")] == [None, None]     # still valid JSON


def test_machine_defaults_to_the_host(tmp_path):
    import socket
    assert M.MetricsLog(tmp_path / "m.jsonl", run="r", component="actor").machine == socket.gethostname()


def test_log_rate_and_log_many(tmp_path):
    log = M.MetricsLog(tmp_path / "m.jsonl", run="r", component="evaluator", machine="m")
    x = log.log_rate("eval.score", 60, 100, version=2, opponent="sbtactical")
    assert x["value"] == 0.6 and x["n"] == 100 and (x["lo"], x["hi"]) == pytest.approx(M.wilson(60, 100))
    assert log.log_rate("eval.score", 0, 0)["value"] is None
    xs = log.log_many({"learn.policy_ce": 1.0, "learn.top1_visits": 0.4}, version=2, split="holdout")
    assert [x["name"] for x in xs] == ["learn.policy_ce", "learn.top1_visits"] and xs[1]["split"] == "holdout"
    assert len(lines(tmp_path / "m.jsonl")) == 4


def test_threads_never_interleave_lines(tmp_path):
    log = M.MetricsLog(tmp_path / "m.jsonl", run="r", component="actor", machine="m")

    def work(i):
        for j in range(50):
            log.log("throughput.games_per_hour", j, step=j, worker=i)

    ts = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(lines(tmp_path / "m.jsonl")) == 200


def test_read_files_and_folders_skipping_junk(tmp_path):
    a = M.MetricsLog(tmp_path / "metrics/actor/r1/000001.jsonl", run="r", component="actor", machine="r1")
    b = M.MetricsLog(tmp_path / "metrics/learner/d1/000001.jsonl", run="r", component="learner", machine="d1")
    a.log("throughput.games_per_hour", 10.0)
    b.log("learn.policy_ce", 1.0, version=1)
    with open(tmp_path / "metrics/actor/r1/000001.jsonl", "a") as f:
        f.write("\n[1, 2]\n{\"no_name\": 1}\n{\"name\": \"torn\", \"val")        # junk and a torn last line
    (tmp_path / "metrics/notes.txt").write_text("not metrics\n")
    got = M.read([tmp_path / "metrics"])
    assert [x["name"] for x in got] == ["throughput.games_per_hour", "learn.policy_ce"]
    assert M.read(tmp_path / "metrics/learner/d1/000001.jsonl")[0]["version"] == 1
    assert M.read([tmp_path / "missing.jsonl"]) == []


@pytest.mark.parametrize("k, n, lo, hi", [
    (5, 10, 0.2366, 0.7634),
    (0, 10, 0.0, 0.2775),
    (10, 10, 0.7225, 1.0),
    (50, 100, 0.4038, 0.5962),
    (5.5, 10, 0.2737, 0.7986),                     # a draw counts half
])
def test_wilson(k, n, lo, hi):
    got = M.wilson(k, n)
    assert len(got) == 2 and got == pytest.approx((lo, hi), abs=1e-4)


def test_wilson_edges():
    assert M.wilson(0, 0) == (0.0, 1.0)
    lo, hi = M.wilson(30, 100, z=2.576)
    lo95, hi95 = M.wilson(30, 100)
    assert lo < lo95 < 0.3 < hi95 < hi
    assert all(math.isfinite(v) for v in M.wilson(12, 7))                         # k > n is clamped


def test_append_jsonl_kept(tmp_path):
    p = tmp_path / "a/b.jsonl"
    M.append_jsonl(p, {"x": 1})
    x = json.loads(p.read_text())
    assert list(x) == ["ts", "x"] and x["x"] == 1


def test_standard_names_are_dotted():
    assert {"throughput.games_per_hour", "learn.policy_ce", "eval.score", "eval.elo"} <= set(M.NAMES)
    assert all(n.count(".") == 1 and n.split(".")[0] in ("throughput", "learn", "eval", "data") for n in M.NAMES)
