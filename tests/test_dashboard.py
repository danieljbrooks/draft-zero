"""The format-knowledge page.

MageZero renders generic training health and knows nothing about draft formats, so this
page is DraftZero's. These tests pin the contract: it degrades quietly on an empty run,
it never double-counts games, and it writes a landing page linking both dashboards.
"""
import json

import pytest

pytest.importorskip("magezero", reason="engine dependency not installed")

from draftzero import dashboard  # noqa: E402


def _run(tmp_path, games):
    (tmp_path / "games.jsonl").write_text("".join(json.dumps(g) + "\n" for g in games))
    (tmp_path / "run.json").write_text(json.dumps({
        "generalist": True, "stage": "play", "current_gen": 2,
        "run_config_snapshot": {"stats": {"window_gens": 2, "min_card_games": 1}},
    }))
    return tmp_path


def _game(gen, kind="selfplay", a="d1", b="d2", winner="A"):
    return {"kind": kind, "gen": gen, "opponent": "self", "agent_side": "both",
            "deck_a": a, "deck_b": b, "colors_a": "WU", "colors_b": "BR",
            "winner": winner, "turns": 20, "drawn_a": ["Swamp"], "drawn_b": ["Island"]}


def test_returns_none_when_run_has_no_games(tmp_path):
    (tmp_path / "run.json").write_text("{}")
    assert dashboard.build_payload(tmp_path) is None
    assert dashboard.render(tmp_path) is None


def test_returns_none_when_games_file_absent(tmp_path):
    assert dashboard.build_payload(tmp_path) is None


def test_renders_page_and_index(tmp_path):
    run = _run(tmp_path, [_game(0), _game(1), _game(1, winner="B")])
    out = dashboard.render(run)
    assert out is not None and out.name == "format.html"
    html = out.read_text()
    assert "Format knowledge" in html
    assert "__DATA__" not in html          # payload was substituted
    idx = run / "index.html"
    assert idx.exists()
    body = idx.read_text()
    assert 'href="dashboard.html"' in body and 'href="format.html"' in body


def test_index_does_not_double_count_recent_games(tmp_path):
    """`recent` is a window inside `network`; summing all three inflates the total."""
    run = _run(tmp_path, [_game(0), _game(1), _game(2)])
    payload = dashboard.build_payload(run)
    sg = payload["selfplay_games"]
    assert sg["recent"] <= sg["network"], "recent must be a subset of network"
    dashboard.render(run)
    shown = int((run / "index.html").read_text().split(" self-play games")[0].split("·")[-1].strip().replace(",", ""))
    assert shown == sg["network"] + sg["heuristic"]
    assert shown != sg["network"] + sg["recent"] + sg["heuristic"] or sg["recent"] == 0


def test_payload_carries_run_state(tmp_path):
    run = _run(tmp_path, [_game(0)])
    p = dashboard.build_payload(run)
    assert p["run_id"] == run.name
    assert p["stage"] == "play"
    assert p["completed"] is False
    assert "cards" in p and "colors" in p and "rho_series" in p
