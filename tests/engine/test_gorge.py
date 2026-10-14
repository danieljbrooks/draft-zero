"""draftzero.engine.gorge.

The first half runs the wrapper against a fake dzgorge (conftest.Fake): kill, trim, replay, skip and the flags, with
no engine. The second half (marked `engine`: `pytest -m engine`) runs the real driver of the gorge fork: a few
games each at sims <= 16 on 2 workers.
"""
import gzip
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pytest

from draftzero.engine import Belief, Gorge, GorgeError, PolicySpec
from draftzero.engine import gorge as gorge_mod
from draftzero.engine.gorge import gzip_members, prefix_games, shrink_card_raw, trim_run

SP = PolicySpec(sims=8, explore=True, keys=("autopay",))


def _corpus_ids(path: Path) -> list[str]:
    """The corpus's game ids in order, once each."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    with gzip.open(path, "rt") as f:
        return list(dict.fromkeys(json.loads(l)["game_id"] for l in f if l.strip()))


def _dead(pid: int) -> bool:
    try:
        st = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return True
    return st.rsplit(")", 1)[1].split()[0] in ("Z", "X")     # a zombie waiting for a reaper is dead too


# ------------------------------------------------------------------------------------------------
# the wrapper, against a fake dzgorge
# ------------------------------------------------------------------------------------------------

def test_fake_clean_run(fake_dzgorge, tmp_path):
    g = fake_dzgorge.gorge
    r = g.play(SP, SP, pairs=5, seed=7, out_dir=tmp_path / "out", record=True, timeout_s=60)
    assert [x["pair"] for x in r.games] == [0, 1, 2, 3, 4] and r.skipped == []
    assert [x["game"] for x in r.games] == [0, 1, 2, 3, 4]
    assert r.summary == {"games": 5, "pairs": 5, "first_pair": 0, "legs": 1, "fake": True}
    assert json.loads(Path(str(r.games_path) + ".summary.json").read_text()) == r.summary
    assert r.games_path.read_text().count("\n") == 5
    assert _corpus_ids(r.corpus_path) == [f"p{p:07d}g0" for p in range(5)]
    assert len(r.runs) == 1 and not r.runs[0]["killed"] and r.runs[0]["returncode"] == 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["games.jsonl", "games.jsonl.summary.json",
                                                                    "play.log", "visits.jsonl.gz"]
    # the same spec on both sides plays each pair once; different specs twice, swapped
    r2 = g.play(SP, "bot", pairs=2, seed=7, out_dir=tmp_path / "out2", first_pair=10)
    assert [(x["pair"], x["leg"], x["game"]) for x in r2.games] == [(10, 0, 0), (10, 1, 1), (11, 0, 2), (11, 1, 3)]
    assert r2.corpus_path is None and not (tmp_path / "out2" / "visits.jsonl.gz").exists()


def test_fake_hang_is_killed_replayed_and_skipped(fake_dzgorge, tmp_path):
    fake_dzgorge.env(FAKE_HANG="2")
    t0 = time.monotonic()
    r = fake_dzgorge.gorge.play(SP, SP, pairs=5, seed=7, out_dir=tmp_path / "o", record=True, timeout_s=2,
                                probe_timeout_s=1)
    assert time.monotonic() - t0 < 20
    assert r.skipped == [2]
    assert [x["pair"] for x in r.games] == [0, 1, 3, 4]
    assert [x["game"] for x in r.games] == [0, 1, 3, 4]          # numbered as one uninterrupted run
    lines = [json.loads(l) for l in r.games_path.read_text().splitlines()]
    assert lines == r.games
    assert _corpus_ids(r.corpus_path) == [f"p{p:07d}g0" for p in (0, 1, 3, 4)]
    assert [(x["first_pair"], x["pairs"], x["killed"]) for x in r.runs] == [(0, 5, True), (2, 1, True), (3, 2, False)]
    assert r.summary["partial"] and r.summary["games"] == 4 and r.summary["skipped"] == [2]
    assert r.summary["visit_records"] == 8
    calls = fake_dzgorge.calls()
    assert [c[c.index("-first-pair") + 1] for c in calls] == ["0", "2", "3"]
    assert _dead(int(fake_dzgorge.child.read_text()))           # the hung game's child died with the group
    assert not any(p.name.startswith(".run") for p in (tmp_path / "o").iterdir())


def test_fake_hang_replay_alone_finishes(fake_dzgorge, tmp_path):
    fake_dzgorge.env(FAKE_HANG_BUSY="2")
    r = fake_dzgorge.gorge.play(SP, SP, pairs=5, seed=7, out_dir=tmp_path / "o", record=True, timeout_s=2,
                                probe_timeout_s=30)
    assert r.skipped == [] and [x["pair"] for x in r.games] == [0, 1, 2, 3, 4]
    assert [x["game"] for x in r.games] == [0, 1, 2, 3, 4]
    assert _corpus_ids(r.corpus_path) == [f"p{p:07d}g0" for p in range(5)]
    assert [(x["first_pair"], x["pairs"], x["killed"]) for x in r.runs] == [(0, 5, True), (2, 1, False), (3, 2, False)]


def test_fake_paired_keeps_whole_pairs(fake_dzgorge, tmp_path):
    fake_dzgorge.env(FAKE_HANG="1", FAKE_HANG_LEG="1")      # pair 1's leg 0 finishes, its leg 1 hangs
    r = fake_dzgorge.gorge.play(SP, "bot", pairs=3, seed=7, out_dir=tmp_path / "o", timeout_s=2, probe_timeout_s=1)
    assert r.skipped == [1]
    assert [(x["pair"], x["leg"], x["game"]) for x in r.games] == [(0, 0, 0), (0, 1, 1), (2, 0, 4), (2, 1, 5)]


def test_fake_sigterm_ignored_gets_sigkill(fake_dzgorge, tmp_path, monkeypatch):
    monkeypatch.setattr(gorge_mod, "KILL_GRACE_S", 0.5)
    fake_dzgorge.env(FAKE_HANG="0", FAKE_IGNORE_TERM="1")
    r = fake_dzgorge.gorge.play(SP, SP, pairs=2, seed=7, out_dir=tmp_path / "o", timeout_s=1, probe_timeout_s=1)
    assert r.skipped == [0] and [x["pair"] for x in r.games] == [1]
    assert r.runs[0]["returncode"] == -9 and r.runs[0]["killed"]
    assert _dead(int(fake_dzgorge.child.read_text()))


def test_fake_refusal_raises(fake_dzgorge, tmp_path):
    fake_dzgorge.env(FAKE_FAIL="1")
    with pytest.raises(GorgeError, match="unknown key"):
        fake_dzgorge.gorge.play(SP, SP, pairs=2, seed=7, out_dir=tmp_path / "o", timeout_s=30)
    assert len(fake_dzgorge.calls()) == 1          # no replay of a refusal


def test_fake_pairs_file(fake_dzgorge, tmp_path):
    pf = tmp_path / "pairs.tsv"
    pf.write_text("# deckA deckB seed\nd1\td2\t5\n\nd3\td4\nd5\td6\t9\n")
    g = fake_dzgorge.gorge
    r = g.play(SP, SP, pairs=0, seed=1, out_dir=tmp_path / "a", pairs_file=pf, deckfile=tmp_path / "decks.jsonl")
    assert [x["pair"] for x in r.games] == [0, 1, 2]
    r = g.play(SP, "bot", pairs=0, seed=1, out_dir=tmp_path / "b", pairs_file=pf, first_pair=1, single=True)
    assert [(x["pair"], x["leg"]) for x in r.games] == [(1, 0), (2, 0)]
    argv = fake_dzgorge.calls()[-1]
    assert argv[argv.index("-pairs-file") + 1] == str(pf) and "-single" in argv and argv[argv.index("-pairs") + 1] == "2"
    with pytest.raises(ValueError):
        g.play(SP, SP, pairs=0, seed=1, out_dir=tmp_path / "c")


def test_play_argv_flags(tmp_path):
    g = Gorge(tmp_path / "gorge", tmp_path / "decks", tmp_path / "pool.tsv")
    closed = PolicySpec(sims=8, closedlist=True)
    argv = g.play_argv(SP, "bot", pairs=3, seed=11, out=tmp_path / "o.jsonl")
    assert argv[:2] == [str(tmp_path / "gorge" / "bin" / "dzgorge"), "play"]
    opt = lambda a, k: a[a.index(k) + 1]                                         # noqa: E731
    assert opt(argv, "-cards") == str(tmp_path / "gorge" / ".cards")
    assert (opt(argv, "-a"), opt(argv, "-b")) == (SP.render(), "bot")
    assert (opt(argv, "-seed"), opt(argv, "-pairs"), opt(argv, "-first-pair"), opt(argv, "-split")) == ("11", "3", "0", "train")
    assert (opt(argv, "-max-turns"), opt(argv, "-max-intents"), opt(argv, "-mulligans")) == ("100", "20000", "2")
    assert "-mull-heuristic" in argv and opt(argv, "-topk-max-options") == "64"
    # census on is dzgorge's default: no flag; off renders -no-paid-mana-census
    assert "-no-paid-mana-census" not in argv
    off = Gorge(tmp_path / "gorge", tmp_path / "decks", tmp_path / "pool.tsv", census=False, mull_heuristic=False,
                topk_max_options=0)
    a2 = off.play_argv(SP, SP, pairs=1, seed=1, out=tmp_path / "o.jsonl")
    assert "-no-paid-mana-census" in a2 and "-mull-heuristic" not in a2 and "-topk-max-options" not in a2
    # no closedlist, no belief: no belief flags; a closedlist seat gets the engine's train split explicitly
    assert "-belief-decks" not in argv and "-corpus" not in argv
    a3 = g.play_argv(closed, "bot", pairs=1, seed=1, out=tmp_path / "o.jsonl", split="eval")
    assert (opt(a3, "-belief-decks"), opt(a3, "-belief-pool"), opt(a3, "-belief-split")) == (
        str(tmp_path / "decks"), str(tmp_path / "pool.tsv"), "train")
    b = Belief(tmp_path / "bd", tmp_path / "bp.tsv", "train")
    a4 = g.play_argv(closed, closed, pairs=1, seed=1, out=tmp_path / "o.jsonl", belief=b)
    assert (opt(a4, "-belief-decks"), opt(a4, "-belief-pool")) == (str(b.decks), str(b.pool))
    with pytest.raises(ValueError, match="own split"):
        g.play_argv(closed, "bot", pairs=1, seed=1, out=tmp_path / "o", split="eval", belief=Belief(b.decks, b.pool, "eval"))
    # record flags, deckfile / pairs-file / single passthrough, extra flags last
    a5 = g.play_argv(SP, SP, pairs=1, seed=1, out=tmp_path / "o.jsonl", corpus=tmp_path / "v.gz", record_every=4,
                     deckfile=tmp_path / "d.jsonl", pairs_file=tmp_path / "p.tsv", single=True,
                     extra_flags=["-mm-stats", "-gonet-cache", "0"])
    assert (opt(a5, "-record-features"), opt(a5, "-record-every"), opt(a5, "-corpus")) == ("entity", "4", str(tmp_path / "v.gz"))
    assert opt(a5, "-deckfile") == str(tmp_path / "d.jsonl") and opt(a5, "-pairs-file") == str(tmp_path / "p.tsv")
    assert "-single" in a5 and a5[-3:] == ["-mm-stats", "-gonet-cache", "0"]


def test_from_config(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    cfg = NS(engine=NS(name="gorge", root="/opt/gorge", census=False, max_turns=50, max_intents=100, mulligans=1,
                       mull_heuristic=False, topk_max_options=32),
             data=NS(decks="data/decks", pool="data/pool.tsv", train_split="train", eval_split="eval"))
    g = Gorge.from_config(cfg)
    assert g.root == Path("/opt/gorge") and g.binary == Path("/opt/gorge/bin/dzgorge")
    assert g.decks == gorge_mod._repo_root() / "data/decks"          # not under the working directory: the repo's
    (tmp_path / "data" / "pool.tsv").write_text("deck\tsplit\n")
    assert Gorge.from_config(cfg).pool == tmp_path / "data" / "pool.tsv"
    assert (g.census, g.max_turns, g.max_intents, g.mulligans, g.mull_heuristic, g.topk_max_options) == (
        False, 50, 100, 1, False, 32)
    with pytest.raises(ValueError):
        Gorge.from_config(NS(engine=NS(name="xmage", root="/x"), data=cfg.data))
    assert Gorge.from_config({"engine": {"root": "/opt/g"}, "data": {"decks": "/d", "pool": "/p"}}).census is True


def test_trim_run_drops_partial_and_unrecorded_games():
    member = lambda p, leg: gzip.compress(json.dumps({"game_id": f"p{p:07d}g{leg}"}).encode() + b"\n")  # noqa: E731
    line = lambda g, p, leg, v=1: json.dumps({"game": g, "pair": p, "leg": leg, "visits": v}).encode()   # noqa: E731
    games = b"\n".join([line(0, 4, 0), line(1, 5, 0), line(2, 6, 0)]) + b"\n" + b'{"game":3,"pa'
    corpus = member(4, 0) + member(5, 0)                    # game 2's member never made it
    lines, rows, members, k = trim_run(games, corpus, 4, 1)
    assert k == 2 and [r["pair"] for r in rows] == [4, 5] and members == [member(4, 0), member(5, 0)]
    # a game without records has no member and needs none; a truncated member is left out
    games2 = line(0, 4, 0, v=0) + b"\n" + line(1, 5, 0) + b"\n"
    assert trim_run(games2, member(5, 0)[:-3], 4, 1)[3] == 1
    assert [x[2] for x in gzip_members(corpus + member(9, 1)[:10])] == [(4, 0), (5, 0)]
    # paired: an orphan leg is dropped
    paired = line(0, 0, 0) + b"\n" + line(1, 0, 1) + b"\n" + line(2, 1, 0) + b"\n"
    assert trim_run(paired, None, 0, 2)[3] == 1 and len(trim_run(paired, None, 0, 2)[0]) == 2


def _synthetic_pack(root: Path) -> list[Path]:
    from draftzero.nets import packfmt
    return packfmt.write_synthetic(root, 120, seed=3, shard_size=50, game_len=10)


def test_shrink_and_prefix(tmp_path):
    from draftzero.nets.data import ShardSet
    dirs = _synthetic_pack(tmp_path / "p")
    before = [np.load(d / "card_raw.npy") for d in dirs]
    games0 = json.loads((dirs[0] / "meta.json").read_text())["games"]
    for d in dirs:
        assert shrink_card_raw(d) and not shrink_card_raw(d)
        prefix_games(d, "a1-000003-")
    for d, b in zip(dirs, before):
        a = np.load(d / "card_raw.npy")
        assert a.dtype == np.float16 and a.shape == b.shape
        np.testing.assert_allclose(a.astype(np.float32), b, rtol=1e-3, atol=1e-3)
    assert json.loads((dirs[0] / "meta.json").read_text())["games"] == ["a1-000003-" + x for x in games0]
    ds = ShardSet(tmp_path / "p")
    batch = ds.collate(np.arange(10))
    assert batch.card_raw.dtype.is_floating_point and str(batch.card_raw.dtype) == "torch.float32"
    assert len(ds.holdout_mask()) == 120
    ds2 = ShardSet(tmp_path / "p", in_ram=True)
    assert len(ds2) == 120 and str(ds2.collate(np.arange(5)).card_raw.dtype) == "torch.float32"


def test_fake_pack_is_atomic(fake_dzgorge, tmp_path):
    g = fake_dzgorge.gorge
    c = tmp_path / "v.jsonl.gz"
    c.write_bytes(gzip.compress(b"{}\n"))
    shards = g.pack([c, tmp_path / "missing.gz"], tmp_path / "pack", shard=20, game_prefix="x-")
    assert [p.name for p in shards] == ["00000", "00001", "00002"]
    assert all(np.load(p / "card_raw.npy", mmap_mode="r").dtype == np.float16 for p in shards)
    assert all(gid.startswith("x-") for p in shards for gid in json.loads((p / "meta.json").read_text())["games"])
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".")) == []    # no temporary left
    with pytest.raises(FileExistsError):
        g.pack([c], tmp_path / "pack")
    calls = fake_dzgorge.calls()
    assert len(calls) == 1 and calls[0][calls[0].index("-in") + 1] == str(c)    # the missing corpus was left out
    assert g.pack([tmp_path / "empty.gz"], tmp_path / "none") == [] and (tmp_path / "none").is_dir()


# ------------------------------------------------------------------------------------------------
# the real driver (pytest -m engine)
# ------------------------------------------------------------------------------------------------

SELF = PolicySpec(sims=8, explore=True, closedlist=True, pcr=0.5, cheap=4, pcrk={"priority": 0.65})


@pytest.fixture(scope="module")
def selfplay(gorge, tmp_path_factory):
    out = tmp_path_factory.mktemp("selfplay")
    return gorge.play(SELF, SELF, pairs=2, seed=7, out_dir=out, workers=2, record=True, timeout_s=600)


@pytest.mark.engine
def test_engine_version(gorge):
    v = gorge.version()
    assert re.fullmatch(r"[0-9a-f]{40}", v["ref"]) and v["dirty"] in (True, False)
    assert v["binary_sha256"] == v["build"]["binary_sha256"] and v["binary_matches_build"]
    assert v["cards_lock"]["digest"] and v["census"] is True and v["repo"]


@pytest.mark.engine
def test_engine_coverage(gorge):
    c = gorge.coverage()
    assert c["decks"] > 1000 and c["cards"] > 100
    assert c["supported_decks"] == len(c["supported"]) and c["supported_decks"] + c["failed_decks"] <= c["decks"]
    assert len(c["unsupported"]) == c["unsupported_cards"]


@pytest.mark.engine
def test_engine_selfplay_records(selfplay):
    r = selfplay
    assert len(r.games) == 2 and r.skipped == [] and [g["pair"] for g in r.games] == [0, 1]
    assert all(g["result"] in ("A", "B", "draw", "stall") for g in r.games)
    assert all(g["pols"] == [SELF.render()] * 2 for g in r.games)
    assert r.summary["games"] == 2 and r.summary["legs"] == 1 and r.summary["a"] == SELF.render()
    assert r.summary["visit_records"] == sum(g.get("visits", 0) for g in r.games) > 0
    assert "a_pcr_full" in r.summary and r.summary["a_pcr_cheap"] > 0          # the fork took pcr/cheap/pcrk
    assert sorted(set(_corpus_ids(r.corpus_path))) == ["p0000000g0", "p0000001g0"]
    assert "belief:" in (r.games_path.parent / "play.log").read_text()          # closedlist got its corpus


@pytest.mark.engine
def test_engine_pack(gorge, selfplay, tmp_path):
    from draftzero.nets import packfmt
    from draftzero.nets.data import ShardSet
    shards = gorge.pack([selfplay.corpus_path], tmp_path / "pack", game_prefix="actor1-000001-")
    assert [p.name for p in shards] == ["00000"]
    meta = json.loads((shards[0] / "meta.json").read_text())
    assert sorted(meta["games"]) == ["actor1-000001-p0000000g0", "actor1-000001-p0000001g0"]
    assert meta["n"] == selfplay.summary["visit_records"]
    assert np.load(shards[0] / "card_raw.npy", mmap_mode="r").dtype == np.float16
    ds = ShardSet(tmp_path / "pack")
    assert len(ds) == meta["n"]
    assert list(ds.holdout_mask()) == [packfmt.holdout_game(meta["games"][i]) for i in ds.array("game")]
    b = ds.collate(np.arange(min(16, len(ds))))
    assert str(b.card_raw.dtype) == "torch.float32" and b.B == min(16, len(ds))
    raw = gorge.pack([selfplay.corpus_path], tmp_path / "raw", shrink=False)
    f32 = np.load(raw[0] / "card_raw.npy")
    assert f32.dtype == np.float32
    np.testing.assert_allclose(np.load(shards[0] / "card_raw.npy").astype(np.float32), f32, rtol=1e-3, atol=1e-6)


@pytest.mark.engine
def test_engine_timeout_resumes_same_games(gorge, selfplay, tmp_path):
    # every run is killed before its first game (the corpus and belief take seconds to load); each first missing
    # pair is then replayed alone, and the games are the uninterrupted run's
    r = gorge.play(SELF, SELF, pairs=2, seed=7, out_dir=tmp_path, workers=2, record=True, timeout_s=0.3,
                   probe_timeout_s=120)
    assert r.skipped == [] and [x["killed"] for x in r.runs] == [True, False, True, False]
    strip = lambda g: {k: v for k, v in g.items() if k != "ms"}               # noqa: E731
    assert [strip(g) for g in r.games] == [strip(g) for g in selfplay.games]
    assert r.summary["partial"] and r.summary["games"] == 2
    assert _corpus_ids(r.corpus_path) == _corpus_ids(selfplay.corpus_path)


@pytest.mark.engine
def test_engine_timeout_skips(gorge, tmp_path):
    t0 = time.monotonic()
    r = gorge.play(SELF, SELF, pairs=2, seed=7, out_dir=tmp_path, timeout_s=0.3, probe_timeout_s=0.3)
    assert r.games == [] and r.skipped == [0, 1] and all(x["killed"] for x in r.runs)
    assert r.games_path.read_bytes() == b"" and time.monotonic() - t0 < 30


@pytest.mark.engine
def test_engine_census_off(gorge, tmp_path):
    off = Gorge(gorge.root, gorge.decks, gorge.pool, census=False)
    assert "-no-paid-mana-census" in off.play_argv("bot", "sbtactical", pairs=1, seed=3, out=tmp_path / "x")
    assert "-no-paid-mana-census" not in gorge.play_argv("bot", "sbtactical", pairs=1, seed=3, out=tmp_path / "x")
    assert off.version()["census"] is False
    r = off.play("bot", PolicySpec(kind="sbtactical"), pairs=1, seed=3, out_dir=tmp_path, split="eval", timeout_s=300)
    assert [(g["pair"], g["leg"]) for g in r.games] == [(0, 0), (0, 1)] and r.summary["split"] == "eval"


@pytest.mark.engine
def test_engine_bad_spec_raises(gorge, tmp_path):
    with pytest.raises(GorgeError, match="unknown key"):
        gorge.play("az:sims=4:frobnicate", "bot", pairs=1, seed=1, out_dir=tmp_path, timeout_s=120)
    assert os.path.exists(tmp_path / "play.log")
