import json

import numpy as np
import pytest

from dzg import models as M
from dzg import packfmt as pf
from dzg import train

from .conftest import TINY


@pytest.fixture(scope="module")
def shards(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    pf.write_synthetic(root / "train", 12000, seed=11, shard_size=8000, game_len=10)
    pf.write_synthetic(root / "eval", 1000, seed=12, game_len=10)
    return root


def test_auc():
    assert train.auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == 0.75
    assert train.auc(np.array([1.0, 1.0]), np.array([0, 1])) == 0.5


def test_train_smoke(shards, tmp_path):
    out = tmp_path / "run"
    summary = train.main([
        "--arch", "mlp", "--config", json.dumps(TINY["mlp"]), "--train", str(shards / "train"),
        "--val", f"evaldecks={shards / 'eval'}", "--out", str(out), "--epochs", "3", "--batch", "128",
        "--lr", "2e-3", "--warmup", "20", "--eval-every", "30", "--device", "cpu", "--workers", "1",
        "--policy-target", "cq", "--seed", "1"])
    lines = [json.loads(x) for x in (out / "curves.jsonl").read_text().splitlines()]
    evals = [x for x in lines if x["type"] == "eval"]
    trains = [x for x in lines if x["type"] == "train"]
    assert evals and trains
    assert {"step", "epoch", "samples", "wall", "samples_per_s", "policy_ce", "value_logloss"} <= set(trains[0])
    last = evals[-1]["eval"]
    assert set(last) == {"holdout", "evaldecks"}
    h = last["holdout"]
    for k in ("policy_ce", "policy_ce_uniform", "target_entropy", "top1_visits", "top1_bot", "value_logloss",
              "value_auc", "value_brier", "value_base_logloss", "total"):
        assert k in h and h[k] is not None, k
    assert h["value_logloss"] < h["value_base_logloss"] - 0.03, h
    assert evals[0]["eval"]["holdout"]["value_logloss"] > h["value_logloss"]   # it fell
    assert h["policy_ce"] < h["policy_ce_uniform"], h
    assert last["evaldecks"]["value_logloss"] < last["evaldecks"]["value_base_logloss"]
    assert (out / "best.pt").exists() and (out / "last.pt").exists() and (out / "config.json").exists()
    m, ck = M.load(out / "best.pt")
    assert ck["arch"] == "mlp" and ck["meta"]["step"] == summary["best_step"]
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model_config"]["d"] == TINY["mlp"]["d"]
    # a warm start from the run continues from its weights (and its arch and config)
    out2 = tmp_path / "run2"
    s2 = train.main(["--init", str(out / "best.pt"), "--train", str(shards / "train"), "--out", str(out2),
                     "--max-steps", "3", "--batch", "64", "--device", "cpu", "--workers", "1", "--eval-every", "3",
                     "--limit-records", "2000", "--in-ram", "--patience", "1"])
    assert s2["steps"] == 3 and s2["arch"] == "mlp"


@pytest.mark.parametrize("arch", ["transformer", "gnn"])
def test_train_runs_attention_archs(arch, shards, tmp_path):
    s = train.main(["--arch", arch, "--config", json.dumps(TINY[arch]), "--train", str(shards / "train"),
                    "--out", str(tmp_path / arch), "--max-steps", "6", "--batch", "64", "--device", "cpu",
                    "--workers", "1", "--eval-every", "6", "--eval-max", "200", "--limit-records", "3000"])
    assert s["steps"] == 6
    assert np.isfinite(s["best"]["holdout"]["policy_ce"])
