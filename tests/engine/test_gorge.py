"""draftzero.engine.gorge on the real driver of the gorge fork (`pytest -m engine`): a few games each at sims <= 16
on 2 workers. The wrapper's own logic against a fake driver is in tests/unit/test_gorge_wrapper.py."""

import gzip
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pytest

from draftzero.engine import Gorge, GorgeError, PolicySpec


def _corpus_ids(path: Path) -> list[str]:
    """The corpus's game ids in order, once each."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    with gzip.open(path, "rt") as f:
        return list(dict.fromkeys(json.loads(l)["game_id"] for l in f if l.strip()))


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
def test_engine_pack_keeps_decision_order(gorge, selfplay, tmp_path):
    """dzgorge pack writes the records in corpus order, and the corpus holds each game's records together and in
    decision order (VisitRecord.sequence, the event-log position at the decision, never falls within a game), which
    nets.data.next_own_turn relies on. dense[0] is the record's turn (gorge's Game.Turn) and dense[ACTIVE] marks one
    active seat a turn. next_rv then equals the next own turn's first record found by the corpus's own key."""
    from draftzero.nets import data as D
    with gzip.open(selfplay.corpus_path, "rt") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    gorge.pack([selfplay.corpus_path], tmp_path / "pack", game_prefix="a1-000001-")
    ds = D.ShardSet(tmp_path / "pack", next_rv=True)
    games = np.concatenate([np.array(m["games"])[np.asarray(p["game"])] for p, m in zip(ds.packs, ds.metas)])
    seat, turn = ds.array("seat"), ds.array("turn")
    assert len(ds) == len(recs) > 0
    assert [("a1-000001-" + r["game_id"], r["seat"], r["turn"]) for r in recs] == \
        [(str(g), int(s), int(t)) for g, s, t in zip(games, seat, turn)]
    assert np.array_equal(ds.array("root_value"), np.array([r["root_value"] for r in recs], np.float32))
    ids = [r["game_id"] for r in recs]
    assert [g for i, g in enumerate(ids) if i == 0 or ids[i - 1] != g] == list(dict.fromkeys(ids))   # contiguous
    for g in set(ids):
        seq = [r["sequence"] for r in recs if r["game_id"] == g]
        assert seq == sorted(seq) and len(set(seq)) > 1
    dense = np.concatenate([np.asarray(p["dense"]) for p in ds.packs])
    assert np.array_equal(dense[:, 0], turn) and set(np.unique(dense[:, D.ACTIVE])) <= {0.0, 1.0}
    active = {}
    for g, s, t, a in zip(games, seat, turn, dense[:, D.ACTIVE]):
        assert active.setdefault((g, int(t)), int(s) if a else 1 - int(s)) == (int(s) if a else 1 - int(s))
    # next_rv by the corpus's explicit order key (game, seat, a later own turn, then sequence)
    own = dense[:, D.ACTIVE] > 0.5
    want = []
    for i, r in enumerate(recs):
        later = [(recs[j]["turn"], recs[j]["sequence"], j) for j in range(len(recs))
                 if recs[j]["game_id"] == r["game_id"] and recs[j]["seat"] == r["seat"] and own[j]
                 and recs[j]["turn"] > r["turn"]]
        o = r["outcome"] if r["outcome_known"] else -1.0
        want.append(recs[min(later)[2]]["root_value"] if later else o)
    assert np.allclose(ds.array("next_rv"), np.array(want, np.float32))
    assert (ds.next_own_turn() >= 0).mean() > 0.5


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


# ------------------------------------------------------------------------------------------------
# the fork's decided-game cheap search and forced playouts (a7afdc1): record weights end to end
# ------------------------------------------------------------------------------------------------

# decided=0.6 with decidedn=1, so that one root value outside (0.4, 0.6) starts it even in a tiny game
DECIDED = PolicySpec(sims=8, explore=True, closedlist=True, decided=0.6, decidedn=1, decidedsims=4, decidedw=0.25,
                     forced=2.0)


@pytest.fixture(scope="module")
def decided_selfplay(gorge, tmp_path_factory):
    out = tmp_path_factory.mktemp("decided")
    return gorge.play(DECIDED, DECIDED, pairs=2, seed=11, out_dir=out, workers=2, record=True, timeout_s=600)


@pytest.mark.engine
def test_engine_decided_and_forced_selfplay(decided_selfplay):
    r = decided_selfplay
    assert len(r.games) == 2 and r.skipped == []
    assert all(g["pols"] == [DECIDED.render()] * 2 for g in r.games)          # the fork took the rendered keys
    s = r.summary
    assert s["a_decided"] + s.get("b_decided", 0) > 0 and s["visit_records_weighted"] > 0
    assert s["az_forced_searched"] > 0
    with gzip.open(r.corpus_path, "rt") as f:
        weights = [json.loads(line).get("weight") for line in f if line.strip()]
    assert len(weights) == s["visit_records"] and set(weights) == {None, 0.25}
    assert weights.count(0.25) == s["visit_records_weighted"]


@pytest.mark.engine
def test_engine_decided_pack_has_weights_and_trains(gorge, decided_selfplay, tmp_path):
    import torch
    from draftzero.nets import train
    from draftzero.nets.data import ShardSet
    s = decided_selfplay.summary
    shards = gorge.pack([decided_selfplay.corpus_path], tmp_path / "pack", game_prefix="a1-000001-")
    assert [p.name for p in shards] == ["00000"]
    w = np.load(shards[0] / "weight.npy")
    assert w.dtype == np.float32 and w.shape == (s["visit_records"],)
    assert set(np.unique(w).tolist()) == {0.25, 1.0} and int((w == 0.25).sum()) == s["visit_records_weighted"]
    assert np.load(shards[0] / "card_raw.npy", mmap_mode="r").dtype == np.float16     # shrunk; weight.npy kept
    ds = ShardSet(tmp_path / "pack")
    assert np.array_equal(ds.array("weight"), w)
    b = ds.collate(np.arange(len(ds)))
    assert torch.equal(b.weight, torch.from_numpy(w))
    tiny = json.dumps({"d": 32, "ff": 64, "layers": 1, "d_emb": 16, "card_d": 24, "head_hidden": 32})
    runs = {}
    for name, extra in (("weighted", []), ("unweighted", ["--no-record-weights"])):
        runs[name] = train.main(["--arch", "mlp", "--config", tiny, "--train", str(tmp_path / "pack"),
                                 "--out", str(tmp_path / name), "--max-steps", "4", "--batch", "32", "--warmup", "1",
                                 "--device", "cpu", "--workers", "1", "--eval-every", "4", "--seed", "1", *extra])
        assert runs[name]["steps"] == 4
    a = torch.load(tmp_path / "weighted" / "last.pt", weights_only=True)["state_dict"]
    u = torch.load(tmp_path / "unweighted" / "last.pt", weights_only=True)["state_dict"]
    assert all(torch.isfinite(v).all() for v in a.values() if v.is_floating_point())
    assert any(not torch.equal(a[k], u[k]) for k in a)                         # the weights changed training
