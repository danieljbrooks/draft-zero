"""Final eval: the big end-of-run eval has to be comparable with the milestone evals and has to
report honestly even when it is cut short.
"""
import random

import pytest

pytest.importorskip("magezero", reason="engine dependency not installed")

from draftzero import final_eval, hfsync  # noqa: E402

POOL = [f"deck{i}.dck" for i in range(40)]


def test_first_pairs_are_the_milestone_matchups():
    # eval_generation draws its `pairs` matchups exactly like this with fixed_decks on
    rng = random.Random("eval-offline")
    milestone = [tuple(rng.sample(POOL, 2)) for _ in range(20)]
    assert final_eval.eval_pairs(POOL, 100)[:20] == milestone


def test_every_opponent_gets_the_same_decks():
    assert final_eval.eval_pairs(POOL, 100) == final_eval.eval_pairs(POOL, 100)


def test_summarize_counts_only_this_eval():
    games = ([{"kind": "final_eval", "gen": 34, "opponent": "offline", "winner": "A"}] * 3
             + [{"kind": "final_eval", "gen": 34, "opponent": "offline", "winner": "B"}]
             + [{"kind": "final_eval", "gen": 34, "opponent": "gen10", "winner": "A"}]
             + [{"kind": "strength_eval", "gen": 34, "opponent": "offline", "winner": "A"}])
    row = final_eval.summarize(games, 34, "offline")
    assert (row["games"], row["wins"]) == (4, 3)
    assert row["a_winrate"] == pytest.approx(0.75)
    assert row["a_winrate_lo"] < 0.75 < row["a_winrate_hi"]
    assert "3/4" in final_eval.describe(row)


def test_summarize_with_no_games_does_not_crash():
    row = final_eval.summarize([], 34, "gen10")
    assert row["games"] == 0 and row["a_winrate"] is None
    assert "no games" in final_eval.describe(row)


def test_push_without_models_uploads_only_results(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_x"); monkeypatch.setenv("HF_REPO", "u/r")
    run = tmp_path / "runs" / "r1"; run.mkdir(parents=True)
    (run / "final_eval.json").write_text("{}")
    (run / "metrics.jsonl").write_text("")
    models = tmp_path / "models"; models.mkdir()
    (models / "gen0.pt.gz").write_bytes(b"x")
    calls = []

    class FakeApi:
        def upload_file(self, path_or_fileobj, path_in_repo, **kw):
            calls.append(path_in_repo)
    monkeypatch.setattr(hfsync, "_api", lambda: (FakeApi(), "u/r"))

    ok, _ = hfsync.push(run, None, gen=34)
    assert ok and sorted(calls) == ["r1/final_eval.json", "r1/metrics.jsonl"]


def test_server_start_is_retried(monkeypatch):
    calls = []

    def flaky(*a, **kw):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("server process exited before becoming ready")
        return "server"
    monkeypatch.setattr(final_eval, "start_server", flaky)
    assert final_eval.start_server_retry("M", 1, 50052, None, wait_s=0) == "server"
    assert len(calls) == 3


def test_server_start_gives_up_after_attempts(monkeypatch):
    def dead(*a, **kw):
        raise RuntimeError("server process exited before becoming ready")
    monkeypatch.setattr(final_eval, "start_server", dead)
    with pytest.raises(RuntimeError):
        final_eval.start_server_retry("M", 1, 50052, None, attempts=2, wait_s=0)
