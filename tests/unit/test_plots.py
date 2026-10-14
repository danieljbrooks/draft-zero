"""The standard figures (draftzero.plots.standard) from a synthetic run's metrics log."""
from __future__ import annotations

import json
import math
import random
import sys
import types
from datetime import datetime, timedelta, timezone

import pytest

from draftzero.plots import standard as st

T0 = datetime(2026, 10, 14, 8, 0, tzinfo=timezone.utc)


def line(name, value, *, component, machine, minutes, version=None, step=None, n=None, lo=None, hi=None, **tags):
    """One metrics line in the contracts §3 schema."""
    return {"schema": "draftzero-metrics/1", "time": (T0 + timedelta(minutes=minutes)).isoformat(), "run": "synth",
            "component": component, "machine": machine, "name": name, "value": value, "version": version,
            "step": step, "n": n, "lo": lo, "hi": hi, **tags}


def synthetic_run(versions=10, actors=("a1", "a2", "a3"), seed=5) -> list[dict]:
    rnd = random.Random(seed)
    out = []
    for v in range(1, versions + 1):
        m = 30 * v
        for split, off in (("holdout", 0.0), ("val", 0.08)):
            out.append(line("learn.policy_ce", 1.6 * math.exp(-v / 6) + 0.6 + off, component="learner", machine="d1",
                            minutes=m, version=v, step=400 * v, split=split))
            out.append(line("learn.value_logloss", 0.69 - 0.1 * (1 - math.exp(-v / 4)) + off, component="learner",
                            machine="d1", minutes=m, version=v, step=400 * v, split=split))
            out.append(line("learn.top1_visits", 0.3 + 0.35 * (1 - math.exp(-v / 5)) - off, component="learner",
                            machine="d1", minutes=m, version=v, step=400 * v, split=split))
            out.append(line("learn.value_auc", 0.6 + 0.15 * (1 - math.exp(-v / 5)) - off, component="learner",
                            machine="d1", minutes=m, version=v, step=400 * v, split=split))
        for opp, s in (("prev", 0.55 + rnd.gauss(0, 0.03)), ("sbtactical", 0.35 + 0.04 * v + rnd.gauss(0, 0.02)),
                       ("nonet", 0.45 + 0.035 * v + rnd.gauss(0, 0.02))):
            if opp == "prev" and v == 1:
                continue
            se = 0.035
            out.append(line("eval.score", min(s, 0.97), component="evaluator", machine="d1", minutes=m + 10, version=v,
                            n=200, lo=s - 1.96 * se, hi=s + 1.96 * se, opponent=opp, n_pairs=100))
    for i, a in enumerate(actors):
        for k in range(0, 36):
            mins = 10 * k + 3 * i
            gph = 220 + 40 * i + rnd.gauss(0, 15)
            out.append(line("throughput.games_per_hour", gph, component="actor", machine=a, minutes=mins))
            out.append(line("throughput.positions_per_hour", 58 * gph, component="actor", machine=a, minutes=mins))
            out.append(line("throughput.ms_per_searched", 1350 + rnd.gauss(0, 60), component="actor", machine=a,
                            minutes=mins))
    return out


def write_run(root, lines):
    """A local store's metrics/<component>/<machine>/<seq>.jsonl files (contracts §4)."""
    by = {}
    for x in lines:
        by.setdefault((x["component"], x["machine"]), []).append(x)
    for (c, m), xs in by.items():
        p = root / "metrics" / c / m / "000000.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(x) + "\n" for x in xs))
    return root


# ----------------------------------------------------------------------------------------------- reading metrics

def test_load_metrics_sources(tmp_path, monkeypatch):
    lines = synthetic_run(versions=3, actors=("a1",))
    root = write_run(tmp_path / "run", lines)
    (root / "metrics" / "learner" / "d1" / "000001.jsonl").write_text(
        '{"name": "learn.x", "value": NaN}\n{"value": 1}\nnot json\n{"name": "learn.x", "value": true}\n'
        '{"name": "learn.x", "value": 2.5, "version": 1}\n{"partial')
    from_dir = st.load_metrics(root)
    assert len(from_dir) == len(lines) + 1                        # only the last good line of the junk file
    assert len(st.load_metrics(lines + [{"name": "x", "value": None}])) == len(lines)
    single = tmp_path / "metrics.jsonl"
    single.write_text("".join(json.dumps(x) + "\n" for x in lines[:5]))
    assert st.load_metrics(single) == lines[:5]
    assert st.load_metrics(tmp_path / "missing") == []

    class FakeStore:                                              # list/read_bytes, as loop.store's stores
        def list(self, prefix):
            return sorted(str(p.relative_to(root)) for p in (root / prefix).rglob("*") if p.is_file())

        def read_bytes(self, path):
            p = root / path
            return p.read_bytes() if p.exists() else None

    assert len(st.load_metrics(FakeStore())) == len(from_dir)
    fake = types.ModuleType("draftzero.loop.store")
    fake.open_store = lambda spec: FakeStore()
    monkeypatch.setitem(sys.modules, "draftzero.loop.store", fake)
    assert len(st.load_metrics("hf://buckets/someone/run")) == len(from_dir)


def test_ratings_from_metrics():
    lines = synthetic_run(versions=6)
    fit = st.ratings_from_metrics(lines)
    assert fit["sbtactical"] == (0.0, 0.0)
    assert set(fit) == {"sbtactical", "nonet"} | {f"v{v}" for v in range(1, 7)}
    assert fit["v6"][0] > fit["v1"][0] > -400 and fit["v6"][1] > 0
    # an evaluator that logs eval.elo is taken as is (latest line per player wins)
    elo = [line("eval.elo", 10.0, component="evaluator", machine="d1", minutes=1, version=3, se=20.0),
           line("eval.elo", 15.0, component="evaluator", machine="d1", minutes=2, version=3, lo=-5.0, hi=35.0),
           line("eval.elo", -50.0, component="evaluator", machine="d1", minutes=2, player="nonet")]
    fit = st.ratings_from_metrics(lines + elo)
    assert fit["v3"] == pytest.approx((15.0, 20 / 1.959964)) and fit["nonet"] == (-50.0, 0.0) and len(fit) == 2
    assert st.ratings_from_metrics([]) == {}


def test_total_throughput_sums_live_machines():
    series = {"a": [(h / 10, 100.0) for h in range(0, 31)], "b": [(h / 10, 200.0) for h in range(10, 21)]}
    tt, total = st._total(series, bins=31)
    at = dict(zip([round(x, 6) for x in tt], total))
    assert at[0.5] == 100 and at[1.5] == 300 and at[2.5] == 100     # b stopped reporting at 2.0 h


def test_needs_matplotlib(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(ImportError, match=r"draftzero\[plots\]"):
        st.report(synthetic_run(versions=2), tmp_path)


# ------------------------------------------------------------------------------------------------------ drawing

def test_report_draws_every_figure_light_and_dark(tmp_path):
    pytest.importorskip("matplotlib")
    from matplotlib.image import imread
    root = write_run(tmp_path / "run", synthetic_run())
    paths = st.report(root, tmp_path / "report")
    names = sorted(p.name for p in paths)
    assert names == sorted(f"{k}-{t}.png" for k in ("learning", "strength", "ratings", "throughput")
                           for t in ("light", "dark"))
    for p in paths:
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        theme = st.THEMES["dark" if p.name.endswith("-dark.png") else "light"]
        px = imread(p)[2, 2, :3]
        want = [int(theme["surface"][i:i + 2], 16) / 255 for i in (1, 3, 5)]
        assert px == pytest.approx(want, abs=2 / 255)


def test_figures_without_data_are_skipped(tmp_path):
    pytest.importorskip("matplotlib")
    only_tp = [x for x in synthetic_run(versions=2, actors=("solo",)) if x["name"].startswith("throughput.")]
    paths = st.report(only_tp, tmp_path, themes=("light",))
    assert [p.name for p in paths] == ["throughput-light.png"]
    assert st.report([], tmp_path / "empty") == []
    # learning curves by version when the learner logs no steps; a single opponent; many machines fold to grey
    lines = [dict(x, step=None) for x in synthetic_run(versions=4, actors=[f"m{i}" for i in range(10)])
             if x["name"] != "eval.score" or x.get("opponent") == "sbtactical"]
    paths = st.report(lines, tmp_path / "v", themes=("dark",))
    assert len(paths) == 4
