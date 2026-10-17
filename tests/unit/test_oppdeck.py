"""The v0 opponent-deck belief: a corpus of deck lists for gorge's :closedlist world (-belief-decks/-belief-pool)."""
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from draftzero.data import pools
from draftzero.oppdeck import belief as ob


@pytest.fixture(autouse=True)
def Belief(monkeypatch):
    """The engine's Belief when it is importable; else a stand-in with the contract's fields (contracts §2)."""
    try:
        from draftzero.engine.gorge import Belief
        return Belief
    except ImportError:
        @dataclass
        class Belief:
            decks: Path
            pool: Path
            split: str = "train"
        pkg, mod = types.ModuleType("draftzero.engine"), types.ModuleType("draftzero.engine.gorge")
        mod.Belief, pkg.gorge = Belief, mod
        monkeypatch.setitem(sys.modules, "draftzero.engine", pkg)
        monkeypatch.setitem(sys.modules, "draftzero.engine.gorge", mod)
        return Belief


DECKS = {
    "zeta": {"name": "zeta", "format": "limited", "cards": [{"name": "Abrade", "count": 2}, {"name": "Mountain", "count": 16}]},
    "alpha": {"Dryad Militant": 3, "Forest": 9, "Plains": 8},
    "mid": {"Island": 17, "colors": "U", "main_colors": "U"},
}


def _dzgorge_read_pool(path, split):
    """cmd/dzgorge/pool.go readPool: skip the header, keep rows with >= 3 fields whose split matches."""
    out = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        v = line.split("\t")
        if i and len(v) >= 3 and (split == "all" or v[1] == split):
            out.append(v[0])
    return out


def test_write_belief_writes_a_corpus_dzgorge_can_read(tmp_path, Belief):
    b = ob.write_belief(DECKS, tmp_path / "bel")
    assert isinstance(b, Belief)
    assert (b.decks, b.pool, b.split) == (tmp_path / "bel/decks", tmp_path / "bel/pool.tsv", "train")
    assert _dzgorge_read_pool(b.pool, "train") == ["alpha", "mid", "zeta"]       # sorted: order changes the draws
    for name in DECKS:
        d = json.loads((b.decks / f"{name}.json").read_text())
        assert d["name"] == name and d["cards"] == sorted(d["cards"], key=lambda e: e["name"])
    rows = {r.deck: r for r in pools.read(b.pool)}
    assert rows["zeta"].cards == 18 and rows["zeta"].colors == "R"
    assert rows["alpha"].colors == "WG"
    assert rows["mid"].colors == "U" and rows["mid"].cards == 17          # given colours win; tags are not cards


def test_write_belief_excludes_the_games_own_decks(tmp_path):
    b = ob.write_belief(DECKS, tmp_path / "bel", exclude=["zeta"], split="pod")
    assert b.split == "pod"
    assert _dzgorge_read_pool(b.pool, "pod") == ["alpha", "mid"]
    assert not (b.decks / "zeta.json").exists()
    with pytest.raises(ValueError):
        ob.write_belief(DECKS, tmp_path / "none", exclude=DECKS)
    with pytest.raises(ValueError):
        ob.write_belief({"../x": {"Island": 1}}, tmp_path / "bad")
    with pytest.raises(ValueError):
        ob.write_belief({"empty": {}}, tmp_path / "empty")


def test_leave_one_out_shares_one_decks_dir(tmp_path):
    per = ob.leave_one_out(DECKS, tmp_path / "loo")
    assert sorted(per) == ["alpha", "mid", "zeta"]
    for held, b in per.items():
        assert b.decks == tmp_path / "loo/decks"
        names = _dzgorge_read_pool(b.pool, "train")
        assert held not in names and sorted(names + [held]) == sorted(DECKS)
    assert sorted(p.name for p in (tmp_path / "loo/decks").iterdir()) == ["alpha.json", "mid.json", "zeta.json"]
    only = ob.leave_one_out(DECKS, tmp_path / "loo2", names=["mid"])
    assert list(only) == ["mid"]
    with pytest.raises(KeyError):
        ob.leave_one_out(DECKS, tmp_path / "loo3", names=["nope"])
    with pytest.raises(ValueError):
        ob.leave_one_out({"a": {"Island": 1}}, tmp_path / "loo4")


def test_corpus_hash_covers_order_split_and_contents(tmp_path):
    b = ob.write_belief(DECKS, tmp_path / "a")
    h = ob.corpus_hash(b)
    assert len(h) == 16 and h == ob.corpus_hash(decks=b.decks, pool=b.pool, split="train")
    assert h == ob.corpus_hash(ob.write_belief(DECKS, tmp_path / "b"))             # same decks, same hash
    rows = pools.read(b.pool)
    pools.write(b.pool, list(reversed(rows)))
    assert ob.corpus_hash(b) != h                                                    # order matters to dzgorge
    pools.write(b.pool, rows)
    assert ob.corpus_hash(b) == h
    (b.decks / "mid.json").write_text("{}")
    assert ob.corpus_hash(b) != h
    assert ob.corpus_hash(decks=b.decks, pool=b.pool, split="eval") != h          # no eval rows: another corpus
    with pytest.raises(ValueError):
        ob.corpus_hash()
