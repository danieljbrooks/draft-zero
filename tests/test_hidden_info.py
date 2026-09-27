"""The opponent's hand stays out of the network input (docs/009).

Exp #1 encoded both hands in every state: configs/game.yml said `hidden_info:`, the Java reads
`hiddenInfo`, and the mismatch was silently ignored. These pin the fix.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("magezero", reason="engine dependency not installed")

from draftzero import loop  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def test_base_game_yml_hides_the_opponents_hand_under_the_key_java_reads():
    cfg = yaml.safe_load((REPO / "configs" / "game.yml").read_text())
    for side in ("player_a", "player_b"):
        assert "hidden_info" not in cfg[side]
        assert cfg[side]["hiddenInfo"]["see_opponent_hand"] is False


def test_loop_sets_the_switch_and_drops_a_stale_key(tmp_path):
    settings = SimpleNamespace(td_discount=0.95, prior_temperature=1.5,
                               priors=SimpleNamespace(binary=False, priority=False, target=False, opponent=False))
    jvm = {"search_budget": 300, "timeout_ms": 60000}
    p = {"mcts": {}, "priors": {}, "hidden_info": {"see_opponent_hand": True},
         "hiddenInfo": {"see_opponent_hand": True}}
    loop._player(p, "pool.txt", "random", tmp_path / "a.hdf5", "mcts", False, settings, jvm,
                 loop.DEFAULTS["hidden_info"])
    assert "hidden_info" not in p
    assert p["hiddenInfo"] == {"see_opponent_hand": False}
    assert loop.DEFAULTS["hidden_info"]["see_opponent_hand"] is False
