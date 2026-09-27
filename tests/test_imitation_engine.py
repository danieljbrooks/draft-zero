"""Imitation's model code runs on MageZero v0.2, not only on experiment #1's engine."""
import gzip

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("magezero")

from draftzero.gameplay import imitation as im  # noqa: E402


def test_new_net_has_the_vocab_width_and_the_trunk_runs():
    model = im.new_net(64)
    assert model.player_priority_head[2].out_features == im.A_DIM
    model.eval()
    idx = torch.tensor([1, 5, 9, 2, 7], dtype=torch.long)
    off = torch.tensor([0, 3], dtype=torch.long)
    with torch.no_grad():
        p, b, v = im.heads(model, im.trunk(model, idx, off, pad_to=8))
    assert p.shape == (2, im.A_DIM) and b.shape == (2, 2) and v.shape == (2,)


def test_load_checkpoint_reads_a_checkpoint_saved_like_magezero_train(tmp_path):
    from magezero.model import GLOBAL_MAX
    from magezero.vocab import FeatureVocab
    vocab = FeatureVocab(feature_hash_bins=GLOBAL_MAX)
    vocab.extend(np.array([11, 22, 33, 44], dtype=np.int64))
    model = im.new_net(len(vocab))
    path = tmp_path / "model.pt.gz"
    with gzip.open(path, "wb") as f:
        torch.save({"model_state_dict": model.state_dict(), "feature_vocab": vocab.state_dict()}, f)
    loaded, v2 = im.load_checkpoint(path)
    assert loaded.embedding.num_embeddings == len(vocab) == len(v2)
    assert loaded.player_priority_head[2].out_features == im.A_DIM
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)
