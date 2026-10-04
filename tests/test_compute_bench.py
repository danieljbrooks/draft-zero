"""Tests for the compute benchmark's bookkeeping (tools/compute_bench, docs/020): the games' throughput estimates,
the scaling of games to higher budgets, and the task order deploy/compute_bench.sh's drain relies on. No JVM."""
import importlib.util
import json
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def tool(path: str):
    spec = importlib.util.spec_from_file_location(path.replace("/", "_"), TOOLS / f"{path}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def game(pair, swap, seconds, decisions=60, sims=6000):
    seat = {"decisions": decisions, "singleOption": 10, "sims": sims, "evals": sims, "netEvals": sims,
            "searchSeconds": seconds * 0.4, "timedOut": False}
    return {"pair": pair, "swap": swap, "winner": "A", "winner_role": "bot1", "turns": 12, "seconds": seconds,
            "seats": {"A": dict(seat), "B": dict(seat)}}


def test_start_window_counts_long_and_short_games_alike(tmp_path):
    summ = tool("compute_bench/summarize")
    d = tmp_path / "games_b100"
    d.mkdir()
    W = 2
    (d / "bench.json").write_text(json.dumps({"budget": 100, "workers": W, "replicas": 1, "server_device": "cpu"}))
    (d / "t0").write_text("1000")
    (d / "t1").write_text("1600")        # a 600 s run; its second half starts at 300 s
    (d / "t2").write_text("2500")
    # (seconds, finish): two first-round games (started at 0), then games started at 100, 350 and 400 s; the last
    # two started in the second half, one long (finished in the drain) and one short
    finished = [(100, 100), (250, 250), (500, 600), (900, 1250), (200, 600)]
    games = [game(k // 2, bool(k % 2), s) for k, (s, _) in enumerate(finished)]
    (d / "games.jsonl").write_text("".join(json.dumps(g) + "\n" for g in games))
    (d / "started_by_t1").write_text("5")
    (d / "play.log").write_text("".join(f"  {k + 1}/800 games  last: pair {k // 2}  bot1 ({s}.0 s)  elapsed {f} s\n"
                                        for k, (s, f) in enumerate(finished)))
    out = summ.games_summary(d)
    started_late = [900, 200]                                    # starts 350 and 400 s
    assert out["games_per_hour"]["start_window"] == pytest.approx(W * 3600 / (sum(started_late) / 2), abs=0.05)
    assert out["start_window_games"] == 2 and out["start_window_complete"]
    assert out["games_per_hour"]["per_game"] == pytest.approx(W * 3600 / (sum(s for s, _ in finished) / 5), abs=0.05)
    assert out["decisions_per_game"] == 120 and out["sims_per_game"] == 12000


def test_report_scales_games_by_the_search_cost_per_decision():
    rep = tool("compute_bench/report")
    s = {"tag": "m", "sysinfo": {"price_usd_hr": 0.5, "cgroup_cores": 30},
         "games_b100": {"games": 40, "games_per_hour": {"start_window": 60.0}, "decisions_per_game": 120,
                        "setting": {"workers": 28, "server_device": "cuda"}, "load": {}},
         "sb": {"100": {"pod_s_per_decision": 0.2, "workers": 28}, "1000": {"pod_s_per_decision": 1.6, "workers": 28}},
         "train_soft_cpu_t30": {"device": "cpu", "threads": 30, "samples_per_s": 400.0}}
    r = rep.machine_rows(s)
    assert r["g100"] == 60.0 and r["g1000"] == pytest.approx(60.0 * 0.2 / 1.6)
    assert r["train_soft_cpu"] == 400.0 and r["train_soft_cpu_threads"] == 30
    md = rep.tables([r])
    assert "| m | 0.50 | 30 | 28 | cuda | self-play | 100 | 60.0 | 120 |" in md


def test_play_starts_its_games_in_the_order_the_drain_assumes():
    play = tool("imitation_scale/play")
    pairs = [("a", "b"), ("c", "d"), ("e", "f")]
    tasks = play.game_tasks(pairs, seed=7, mirror=True)
    assert [(t["pair"], t["swap"]) for t in tasks] == [(k, s) for k in range(3) for s in (False, True)]
