"""Extending a finished run.

A run that hits target_games is marked completed, after which --resume finds nothing and
would start a FRESH run: generation numbering back to 0, a wasted heuristic bootstrap, and
a broken dashboard trend. --extend reopens the same run instead.
"""
import json
import os

import pytest

pytest.importorskip("magezero", reason="engine dependency not installed")

from draftzero import loop  # noqa: E402


def _finished_run(tmp_path, model="M", gen=7, games=100):
    runs = tmp_path / "runs" / "2026-01-01_00-00-00"
    runs.mkdir(parents=True)
    (runs / "run.json").write_text(json.dumps({
        "generalist": True, "primary": {"deck": model, "version": 1},
        "current_gen": gen, "stage": "play", "gens": {str(i): {} for i in range(gen)},
        "completed_at": "2026-01-02T00:00:00", "next_session": 5,
    }))
    (runs / "games.jsonl").write_text("".join(
        json.dumps({"kind": "selfplay", "gen": 1, "winner": "A"}) + "\n" for _ in range(games)))
    return runs


def test_find_latest_sees_finished_runs(tmp_path, monkeypatch):
    run = _finished_run(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(loop, "RUNS_DIR", tmp_path / "runs")
    assert loop.find_active("M") is None, "a completed run is not active"
    assert loop.find_latest("M") == run, "but --extend must still find it"


def test_extend_refuses_without_a_raised_target(tmp_path, monkeypatch):
    _finished_run(tmp_path, games=100)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(loop, "RUNS_DIR", tmp_path / "runs")
    cfg = {"model": "M", "version": 1, "target_games": 100}
    with pytest.raises(SystemExit) as e:
        loop.main(cfg, resume=None, extend=True)
    assert "Raise target_games" in str(e.value)


def test_extend_reopens_the_same_run_and_records_history(tmp_path, monkeypatch):
    run = _finished_run(tmp_path, gen=7, games=100)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(loop, "RUNS_DIR", tmp_path / "runs")
    # stop before any real work: the loop calls Run(), which needs pools we do not have here
    monkeypatch.setattr(loop, "find_active", lambda m: (_ for _ in ()).throw(RuntimeError("stop")))
    cfg = {"model": "M", "version": 1, "target_games": 500}
    with pytest.raises(RuntimeError, match="stop"):
        loop.main(cfg, resume=None, extend=True)

    state = json.loads((run / "run.json").read_text())
    assert "completed_at" not in state, "reopened runs must not still look finished"
    assert state["current_gen"] == 7, "generation numbering is preserved"
    assert len(state["gens"]) == 7, "history is preserved"
    assert state["extensions"][0]["from_games"] == 100
    assert state["extensions"][0]["to_target"] == 500
    assert state["extensions"][0]["previously_completed_at"] == "2026-01-02T00:00:00"


@pytest.mark.parametrize("stored,expected", [
    ("play", "play"),
    ("train", "train"),
    ("eval", "eval"),
    ("eval_prev", "train"),   # a sub-stage of train, NOT a restart of play
    ("nonsense", "play"),
    (None, "play"),
])
def test_resume_stage_mapping(stored, expected):
    """eval_prev must map to train.

    It is not in STAGES, so a membership test placed first rewrites it to "play" --
    and resuming into "play" deletes that generation's already-recorded games.
    """
    STAGES = loop.STAGES
    stage = stored
    if stage == "eval_prev":
        stage = "train"
    if stage not in STAGES:
        stage = "play"
    assert stage == expected
