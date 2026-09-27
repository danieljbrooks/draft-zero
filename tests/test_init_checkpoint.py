"""A run that starts from a pretrained network (exp #2 run 2): gen 0 is network self-play,
the start is kept as gen0.pt.gz, and the pretrained checkpoint loads like MageZero's own."""
import gzip
import json
import random

import numpy as np
import pytest

pytest.importorskip("magezero", reason="engine dependency not installed")

from draftzero import loop  # noqa: E402


class _Run(loop.Run):
    def __init__(self, cfg, run_dir, models):        # no deck pools or metadata needed here
        self.cfg, self.dir, self.models = cfg, run_dir, models


def _run(tmp_path, **cfg):
    d = tmp_path / "runs" / "r"
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"gens": {}}))
    base = {**json.loads(json.dumps(loop.DEFAULTS)), "chunk_games": 4, "bootstrap_games": 8,
            "games_per_gen": 8, **cfg}
    return _Run(base, d, tmp_path / "models")


def test_gen0_is_heuristic_without_an_init(tmp_path):
    run = _run(tmp_path)
    jobs = loop.plan_games(run, 0, random.Random(0))
    assert {j["opponent"] for j in jobs} == {"heuristic"} and sum(j["games"] for j in jobs) == 8


def test_gen0_is_network_self_play_from_an_init(tmp_path):
    run = _run(tmp_path, init_checkpoint="x.pt.gz")
    jobs = loop.plan_games(run, 0, random.Random(0))
    assert {j["opponent"] for j in jobs} == {"self"} and all(j["checkpoint"] is None for j in jobs)
    assert sum(j["games"] for j in jobs) == 8


def test_install_init_copies_the_start_as_model_and_gen0(tmp_path):
    src = tmp_path / "pre.pt.gz"
    src.write_bytes(b"weights")
    run = _run(tmp_path, init_checkpoint=str(src))
    loop.install_init(run)
    assert (run.models / "model.pt.gz").read_bytes() == b"weights"
    assert (run.models / "gen0.pt.gz").read_bytes() == b"weights"
    assert run.state["init_checkpoint"]["path"] == str(src)


def test_gen1_mix_is_unchanged_by_an_init(tmp_path):
    run = _run(tmp_path, init_checkpoint="x.pt.gz", mix={"self": 0.2, "past": 0.7, "gen0": 0.1})
    run.models.mkdir()
    for n in ("model", "gen0"):
        (run.models / f"{n}.pt.gz").write_bytes(b"w")
    assert {j["opponent"] for j in loop.plan_games(run, 1, random.Random(0))} == {"self"}
    assert {j["opponent"] for j in loop.plan_games(run, 2, random.Random(0))} == {"self", "gen0"}


def test_pretrained_checkpoint_loads_like_the_server_and_train_py(tmp_path):
    torch = pytest.importorskip("torch")
    from magezero.model import GLOBAL_MAX, NetTransformer, load_model
    from magezero.vocab import FeatureVocab
    from draftzero.gameplay import imitation as im
    from draftzero.gameplay import pretrain
    idx = np.array([5, 9, 5, 9, 1 << 30, 5, 9], np.int64)
    off = np.array([0, 2, 4, 7], np.int64)
    vocab = pretrain.build_vocab(idx, off, k=1)
    assert vocab.feature_hash_bins == GLOBAL_MAX
    model = im.new_net(len(vocab))
    path = tmp_path / "pre.pt.gz"
    pretrain.save_checkpoint(model, vocab, path, {"test": True})
    ck = load_model(str(path))
    v = FeatureVocab.from_state_dict(ck["feature_vocab"])
    v.require_encoding(GLOBAL_MAX)                     # what server.init and train.py check
    served = NetTransformer(len(v), policy_size_pA=im.A_DIM, policy_size_pB=im.A_DIM, policy_size_t=im.A_DIM)
    served.load_state_dict(ck["model_state_dict"])     # server.init builds NetTransformer(len(vocab))
    with gzip.open(path, "rb") as f:
        assert f.read(2)                               # gzip, as MageZero writes it
