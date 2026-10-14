"""The deck pool in gorge format: pool.tsv, deck files, deterministic samples, the 17lands conversion, assets/sample."""
import json

import pytest

from draftzero.data import decks, paths, pools


def _pool(tmp_path, rows):
    return pools.write(tmp_path / "pool.tsv", [pools.PoolRow(*r) for r in rows])


def test_pool_round_trip_keeps_order_and_columns(tmp_path):
    p = _pool(tmp_path, [("d2", "train", "WB", "WB", 40), ("d1", "eval", "UR", "U", 41)])
    lines = p.read_text().splitlines()
    assert lines[0] == "deck\tsplit\tcolors\tmain_colors\tcards"
    assert lines[1] == "d2\ttrain\tWB\tWB\t40"
    rows = pools.read(p)
    assert [r.deck for r in rows] == ["d2", "d1"] and rows[1] == pools.PoolRow("d1", "eval", "UR", "U", 41)
    # dzgorge's readPool needs at least three tab-separated fields a row, even with empty colours
    pools.write(p, [pools.PoolRow("x", "train")])
    assert len(p.read_text().splitlines()[1].split("\t")) >= 3


def test_read_refuses_a_file_that_is_not_a_pool(tmp_path):
    f = tmp_path / "train.txt"
    f.write_text("FDN_top_00001_WB\n")
    with pytest.raises(ValueError):
        pools.read(f)


def test_names_must_be_file_stems(tmp_path):
    for bad in ("a/b", "", "../x", "a b"):
        with pytest.raises(ValueError):
            pools.write(tmp_path / "p.tsv", [pools.PoolRow(bad, "train")])


def test_splits_and_names(tmp_path):
    p = _pool(tmp_path, [("a", "train", "", "", 0), ("b", "eval", "", "", 0), ("c", "train", "", "", 0)])
    assert pools.splits(p) == {"train": ["a", "c"], "eval": ["b"]}
    assert pools.names(p, "train") == ["a", "c"]
    assert pools.names(p) == pools.names(p, "all") == ["a", "b", "c"]


def test_sample_is_deterministic_and_independent_of_file_order(tmp_path):
    rows = [pools.PoolRow(f"d{i:03d}", "train" if i % 3 else "eval") for i in range(100)]
    a = pools.sample(rows, 5, split="train", seed="s")
    assert a == pools.sample(list(reversed(rows)), 5, split="train", seed="s")
    assert a == sorted(a) and len(set(a)) == 5
    assert all(int(n[1:]) % 3 for n in a)
    # pinned: random.Random(str) seeding is stable across Python versions; a change here changes every sample
    assert a == ["d001", "d023", "d065", "d088", "d092"]
    assert pools.sample(rows, 5, split="train", seed="t") != a
    with pytest.raises(ValueError):
        pools.sample(rows, 1000, split="train")


def test_deck_cards_reads_both_shapes_and_gorge_deck_sorts():
    g = {"name": "x", "cards": [{"name": "Forest", "count": 8}, {"name": "Llanowar Elves", "count": 1},
                                {"name": "Forest", "count": 1}]}
    assert pools.deck_cards(g) == {"Forest": 9, "Llanowar Elves": 1}
    assert pools.deck_cards({"Forest": 9, "colors": "G"}) == {"Forest": 9}
    d = pools.gorge_deck("x", {"Zebra": 1, "Abrade": 2, "None": 0})
    assert d == {"name": "x", "format": "limited", "cards": [{"name": "Abrade", "count": 2}, {"name": "Zebra", "count": 1}]}


def test_deck_colors_from_the_fdn_table():
    assert pools.deck_colors({"Abrade": 4, "Forest": 9}) == ("R", "R")
    colors, main = pools.deck_colors({"Abrade": 1, "Dryad Militant": 3, "Unknown Card": 5})
    assert colors == "WRG" and main == "WG"      # hybrid {G/W} counts both; one Abrade is a splash


def test_subset_copies_decks_byte_for_byte(tmp_path):
    src = tmp_path / "decks"
    src.mkdir()
    for n in ("a", "b", "c"):
        (src / f"{n}.json").write_text(json.dumps({"name": n, "cards": []}, indent=0) + "\n ")
    p = _pool(tmp_path, [("a", "train", "", "", 1), ("b", "eval", "", "", 2), ("c", "train", "", "", 3)])
    d, out = pools.subset(p, src, ["c", "a"], tmp_path / "out")
    assert sorted(x.name for x in d.iterdir()) == ["a.json", "c.json"]
    assert (d / "a.json").read_bytes() == (src / "a.json").read_bytes()
    assert [r.deck for r in pools.read(out)] == ["a", "c"]      # pool order, not the order asked
    with pytest.raises(KeyError):
        pools.subset(p, src, ["zz"], tmp_path / "out2")


def test_pools_cli_sample(tmp_path, capsys):
    p = _pool(tmp_path, [(f"d{i}", "train", "", "", 0) for i in range(10)])
    assert pools.main(["sample", "--pool", str(p), "--n", "3", "--split", "train", "--seed", "x"]) == 0
    assert capsys.readouterr().out.split() == pools.sample(p, 3, split="train", seed="x")


def _row(cards, main="WB", splash="", draft="abc", idx=0):
    return {"main_colors": main, "splash_colors": splash, "cards": cards, "draft_id": draft, "build_index": idx,
            "user_game_win_rate_bucket": 0.62}


def test_stem_matches_the_v010_names():
    assert decks.stem(1, _row({}), "FDN") == "FDN_top_00001_WB"
    assert decks.stem(17, _row({}, main="UR", splash="W"), "FDN") == "FDN_top_00017_URpW"
    assert decks.stem(3, _row({}, main=""), "FDN") == "FDN_top_00003_C"


def test_convert_writes_gorge_decks_and_a_pool(tmp_path):
    jl = tmp_path / "decks.jsonl"
    jl.write_text("\n".join(json.dumps(r) for r in [
        _row({"Plains": 8, "Abrade": 1}), _row({"Island": 9}, main="UR"), _row({"Swamp": 7}, main="BG")]) + "\n")
    split = tmp_path / "decks.tsv"
    split.write_text("deck\tsplit\tcolors\tmain_colors\tplayer_wr_bucket\n"
                     "FDN_top_00001_WB\ttrain\tWB\tWB\t0.6\nFDN_top_00003_BG\teval\tBG\tBG\t0.6\n")
    rows, skipped = decks.convert(jl, split, tmp_path / "gorge")
    assert skipped == 1
    assert [(r.deck, r.split, r.cards) for r in rows] == [("FDN_top_00001_WB", "train", 9), ("FDN_top_00003_BG", "eval", 7)]
    d = json.loads((tmp_path / "gorge/decks/FDN_top_00001_WB.json").read_text())
    assert d["name"] == "FDN_top_00001_WB" and d["cards"][0] == {"name": "Abrade", "count": 1}
    assert "CC BY 4.0" in d["notes"] and "abc:0" in d["notes"]
    assert pools.read(tmp_path / "gorge/pool.tsv") == rows
    # a deck the split lists but decks.jsonl lacks: the split would silently shrink
    split.write_text(split.read_text() + "FDN_top_00009_WB\ttrain\tWB\tWB\t0.6\n")
    with pytest.raises(SystemExit):
        decks.convert(jl, split, tmp_path / "gorge2")


def test_gorge_deck_path():
    assert paths.gorge_deck_path("FDN_top_00001_WB", "/d") == paths.Path("/d/FDN_top_00001_WB.json")
    assert paths.gorge_deck_path("/x/FDN_top_00001_WB.dck", "/d") == paths.Path("/d/FDN_top_00001_WB.json")


def test_the_shipped_sample_is_complete():
    """assets/sample is what a fresh clone's smoke run plays: 40 train + 40 eval decks, every row with its file."""
    rows = pools.read(paths.SAMPLE_POOL)
    assert {s: len(n) for s, n in pools.splits(rows).items()} == {"train": 40, "eval": 40}
    files = sorted(p.stem for p in paths.SAMPLE_DECKS.glob("*.json"))
    assert files == sorted(r.deck for r in rows)
    for split in ("train", "eval"):
        listed = (paths.ASSETS / "sample" / f"{split}.txt").read_text().split()
        assert sorted(listed) == sorted(pools.names(rows, split))
    for r in rows:
        d = pools.load_deck(paths.SAMPLE_DECKS, r.deck)
        assert d["name"] == r.deck and sum(pools.deck_cards(d).values()) == r.cards >= 40
        assert "17lands" in d["notes"] and "CC BY 4.0" in d["notes"]
