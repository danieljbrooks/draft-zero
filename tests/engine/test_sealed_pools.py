"""draftzero.build.pools on the real `dzgorge sealed-pools` (`pytest -m engine`): draftlab sealed pools are a pure
function of their seed and index, every card is in the deck network's table, and best23's decks are 40 cards from
the pool plus basics. The binary is DZ_SEALED_DZGORGE (default /work/bin-sealed/dzgorge, the gorge fork's
draftzero-sealed branch) with the cards of DZ_GORGE_ROOT (default /work/gorge-fork); skipped without them."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.engine

DZGORGE = Path(os.environ.get("DZ_SEALED_DZGORGE", "/work/bin-sealed/dzgorge"))
CARDS = Path(os.environ.get("DZ_GORGE_ROOT", "/work/gorge-fork")) / ".cards"


@pytest.fixture(scope="module")
def gorge():
    if not DZGORGE.exists() or not (CARDS / "cardsfolder").is_dir():
        pytest.skip(f"no {DZGORGE} or no card corpus at {CARDS}")
    p = subprocess.run([str(DZGORGE), "sealed-pools", "-h"], capture_output=True, text=True, timeout=60)
    if "-best23" not in p.stdout + p.stderr:
        pytest.skip(f"{DZGORGE} has no sealed-pools command")
    return SimpleNamespace(binary=DZGORGE, cards=CARDS)


def test_sealed_pools_are_deterministic_and_fit_the_card_table(gorge, tmp_path):
    from draftzero.build import cards as cardsmod
    from draftzero.build import pools as P
    a = P.cached(gorge, tmp_path / "a", n=6, seed=11, best23=True)
    b = P.cached(gorge, tmp_path / "b", n=6, seed=11, best23=True)
    tail = P.read(P.generate(gorge, tmp_path / "t.jsonl", n=2, first=4, seed=11, best23=True))
    assert a == b and a[4:] == tail
    assert [p["pool_id"] for p in a] == [f"FDN-11-{i:06d}" for i in range(6)]
    other = P.cached(gorge, tmp_path / "a", n=6, seed=12)
    assert {p["seed"] for p in other}.isdisjoint({p["seed"] for p in a}) and "best23" not in other[0]
    C = cardsmod.load()
    for p in a:
        v = P.pool_vector(p)                          # KeyError for a card outside the deck network's table
        assert sum(p["cards"].values()) == 84 and 70 <= v.sum() <= 84
        d = P.deck_vector(p["best23"]["cards"])
        spells = d.copy()
        spells[[C.index[x] for x in cardsmod.BASICS]] = 0
        assert d.sum() == 40 and (spells <= v).all() and p["best23"]["builder"] == "best23@1"
    play = P.cached(gorge, tmp_path / "a", n=2, seed=11, collation="play")
    assert play[0]["collation"] == "play" and play[0]["cards"] != a[0]["cards"]
