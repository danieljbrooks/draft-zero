"""draftzero.tracking: Trackio mirroring is optional: a fake trackio module stands in for the real one."""
from __future__ import annotations

import sys
import types
import warnings

import pytest

from draftzero import config as C
from draftzero import metrics as M
from draftzero import tracking as T


@pytest.fixture(autouse=True)
def _fresh_warnings():
    T._warned.clear()
    yield
    T._warned.clear()


class FakeTrackio(types.ModuleType):
    def __init__(self, fail_log: bool = False):
        super().__init__("trackio")
        self.inits, self.logged, self.finished, self.fail_log = [], [], 0, fail_log

    def init(self, **kw):
        self.inits.append(kw)

    def log(self, metrics, step=None):
        if self.fail_log:
            raise RuntimeError("space down")
        self.logged.append((metrics, step))

    def finish(self):
        self.finished += 1


def cfg(on: bool) -> C.RunConfig:
    return C.from_dict({"run": "t1", "tracking": {"trackio": on, "project": "dz-test"}})


def sample_lines(tmp_path):
    ev = M.MetricsLog(tmp_path / "e.jsonl", run="t1", component="evaluator", machine="d1")
    ac = M.MetricsLog(tmp_path / "a.jsonl", run="t1", component="actor", machine="r1")
    return [ev.log("eval.score", 0.6, version=3, n=100, opponent="prev"),
            ac.log("throughput.games_per_hour", 40.0, step=7),
            ac.log("throughput.ms_per_searched", float("nan")),                          # no value: skipped
            ev.log("learn.policy_ce", 1.5)]


def test_off_is_a_silent_no_op(tmp_path, monkeypatch):
    fake = FakeTrackio()
    monkeypatch.setitem(sys.modules, "trackio", fake)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tr = T.Tracker(cfg(False))
        assert tr.enabled is False and tr.mirror(sample_lines(tmp_path)) == 0
        tr.close()
    assert fake.inits == [] and fake.logged == []


def test_missing_trackio_warns_once_and_does_nothing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "trackio", None)                 # import trackio -> ImportError
    with pytest.warns(RuntimeWarning, match="trackio cannot be imported"):
        tr = T.Tracker(cfg(True))
    assert tr.enabled is False and tr.mirror(sample_lines(tmp_path)) == 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        T.Tracker(cfg(True)).close()                                  # the second time: no warning


def test_mirror_logs_each_line_once(tmp_path, monkeypatch):
    fake = FakeTrackio()
    monkeypatch.setitem(sys.modules, "trackio", fake)
    with T.Tracker(cfg(True)) as tr:
        assert tr.enabled
        init = fake.inits[0]
        assert init["project"] == "dz-test" and init["name"] == "t1" and init["config"]["run"] == "t1"
        lines = sample_lines(tmp_path)
        assert tr.mirror(lines) == 3
        assert tr.mirror(lines) == 0                                   # re-reading the same files adds nothing
    assert fake.finished == 1
    assert fake.logged == [({"eval.score/opponent=prev": 0.6}, 3),
                           ({"throughput.games_per_hour/machine=r1": 40.0}, 7),
                           ({"learn.policy_ce": 1.5}, None)]


def test_tracker_failures_never_raise(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "trackio", FakeTrackio(fail_log=True))
    tr = T.Tracker(cfg(True), name="custom")
    with pytest.warns(RuntimeWarning, match="trackio.log failed"):
        assert tr.mirror(sample_lines(tmp_path)) == 0
    tr.close()

    broken = FakeTrackio()
    broken.init = lambda **kw: (_ for _ in ()).throw(ConnectionError("no network"))
    monkeypatch.setitem(sys.modules, "trackio", broken)
    with pytest.warns(RuntimeWarning, match="trackio.init failed"):
        assert T.Tracker(cfg(True)).enabled is False


def test_metric_key():
    assert T.metric_key({"name": "eval.elo", "component": "evaluator", "machine": "d1", "opponent": "bot",
                         "sims": 100}) == "eval.elo/opponent=bot/sims=100"
    assert T.metric_key({"name": "throughput.games_per_hour", "component": "actor", "machine": "r2"}) == \
        "throughput.games_per_hour/machine=r2"
