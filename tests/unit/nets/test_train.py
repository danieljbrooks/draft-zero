import json

import numpy as np
import pytest

from draftzero.nets import models as M
from draftzero.nets import packfmt as pf
from draftzero.nets import train

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


# ------------------------------------------------------------------------------------------------ opt-in extras

def _losses_before_extras(model, b, args, weighted=True):
    """train.losses as it was before --surprise-weight / --value-surprise (a3b0893), verbatim."""
    import torch
    import torch.nn.functional as F
    from draftzero.nets import targets as T
    scores, vlogit = model(b)
    z, lp = M.candidate_logits(scores, b)
    tgt = T.policy_target(b, args.policy_target, args.visits_temp, args.cq_c_visit, args.cq_c_scale)
    pm = T.policy_mask(b)
    ce_state = train.seg_sum(-(tgt * lp), b.cand_state, b.B)
    npol = pm.sum()
    pce = torch.where(pm, ce_state, torch.zeros_like(ce_state)).sum() / npol.clamp(min=1)
    vt, vm = T.value_target(b, args.value_blend)
    vl = F.binary_cross_entropy_with_logits(vlogit.float(), vt, reduction="none")
    nval = vm.sum()
    vloss = torch.where(vm, vl, torch.zeros_like(vl)).sum() / nval.clamp(min=1)
    total = pce + args.value_weight * vloss
    return total, pce, vloss, {"scores": scores, "vlogit": vlogit, "z": z, "lp": lp, "tgt": tgt, "pm": pm,
                               "ce_state": ce_state, "npol": npol, "nval": nval}


def _run(shards, out, *extra, steps=12):
    return train.main(["--arch", "mlp", "--config", json.dumps(TINY["mlp"]), "--train", str(shards / "train"),
                       "--val", f"evaldecks={shards / 'eval'}", "--out", str(out), "--max-steps", str(steps),
                       "--batch", "64", "--lr", "2e-3", "--warmup", "4", "--eval-every", "4", "--eval-max", "300",
                       "--limit-records", "3000", "--device", "cpu", "--workers", "1", "--value-blend", "0.5",
                       "--seed", "3", *extra])


def _weights(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)["state_dict"]


def _same(a, b):
    import torch
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _evals(out, key="eval"):
    lines = [json.loads(x) for x in (out / "curves.jsonl").read_text().splitlines()]
    return [(x["step"], x[key]) for x in lines if x["type"] == "eval"]


def test_defaults_are_bit_identical_to_training_before_the_extras(shards, tmp_path, monkeypatch):
    """With every extra at its default (or explicitly 0), training gives the same weights, curves and best step as
    the loss function before the extras existed."""
    s_new = _run(shards, tmp_path / "new")
    s_zero = _run(shards, tmp_path / "zero", "--surprise-weight", "0", "--value-surprise", "0", "--ema-decay", "0",
                  "--value-next", "0")
    monkeypatch.setattr(train, "losses", _losses_before_extras)
    s_old = _run(shards, tmp_path / "old")
    for name in ("best.pt", "last.pt"):
        assert _same(_weights(tmp_path / "new" / name), _weights(tmp_path / "old" / name)), name
        assert _same(_weights(tmp_path / "zero" / name), _weights(tmp_path / "old" / name)), name
    assert _evals(tmp_path / "new") == _evals(tmp_path / "old") == _evals(tmp_path / "zero")
    assert s_new["best_step"] == s_old["best_step"] == s_zero["best_step"]
    assert s_new["best"] == s_old["best"] and "best_weights" not in s_new
    assert not (tmp_path / "new" / "ema.pt").exists() and not (tmp_path / "zero" / "ema.pt").exists()
    meta = M.load(tmp_path / "new" / "best.pt")[1]["meta"]
    assert "weights" not in meta


def _one_batch(n=64, seed=5):
    from draftzero.nets import data as D
    pack, _ = pf.synthetic(n, seed=seed)
    return D.collate(pack, np.arange(n))


def _args(**kw):
    from types import SimpleNamespace
    base = {"policy_target": "visits", "visits_temp": 1.0, "cq_c_visit": 50.0, "cq_c_scale": 0.1, "value_blend": 0.5,
            "value_weight": 1.0, "surprise_weight": 0.0, "value_surprise": 0.0}
    return SimpleNamespace(**{**base, **kw})


def test_surprise_weighted_losses_by_hand():
    import torch
    from draftzero.nets import targets as T
    torch.manual_seed(0)
    model = M.build("mlp", TINY["mlp"])
    b = _one_batch()
    with torch.no_grad():
        t0, p0, v0, x = train.losses(model, b, _args())
        t1, p1, v1, _ = train.losses(model, b, _args(surprise_weight=0.5, value_surprise=0.1))
        te, *_ = train.losses(model, b, _args(surprise_weight=0.5, value_surprise=0.1), weighted=False)
    assert torch.equal(p0, p1) and torch.equal(v0, v1)         # the reported losses stay unweighted
    pm, ce = x["pm"], x["ce_state"]
    kl = T.policy_surprise(b, x["tgt"])
    w = 0.5 + 0.5 * kl / kl[pm].mean()
    assert abs(float(w[pm].mean()) - 1) < 1e-5
    pce_w = float((w * ce)[pm].sum() / pm.sum())
    vt, vm = T.value_target(b, 0.5)
    err = (vt - torch.sigmoid(x["vlogit"].float())).abs()
    vw = 0.9 + 0.1 * err / err[vm].mean()
    vl = torch.nn.functional.binary_cross_entropy_with_logits(x["vlogit"].float(), vt, reduction="none")
    vloss_w = float((vw * vl)[vm].sum() / vm.sum())
    assert abs(float(t1) - (pce_w + vloss_w)) < 1e-5
    assert abs(float(t1) - float(t0)) > 1e-4                   # the weighting changes the training loss
    assert torch.equal(te, t0)                                 # evaluation never weights


def test_surprise_weighting_trains_and_differs(shards, tmp_path):
    base = _run(shards, tmp_path / "base")
    s = _run(shards, tmp_path / "sw", "--surprise-weight", "0.5", "--value-surprise", "0.1")
    assert np.isfinite(s["best"]["holdout"]["total"]) and np.isfinite(s["best"]["evaldecks"]["policy_ce"])
    assert not _same(_weights(tmp_path / "sw" / "last.pt"), _weights(tmp_path / "base" / "last.pt"))
    assert json.loads((tmp_path / "sw" / "config.json").read_text())["args"]["surprise_weight"] == 0.5
    assert base["steps"] == s["steps"] == 12


def test_extras_reject_bad_values(shards, tmp_path):
    for bad in (["--surprise-weight", "1.5"], ["--value-surprise", "-0.1"], ["--ema-decay", "1"],
                ["--export-ema"], ["--ema-decay", "0.9", "--ema-every", "0"], ["--value-next", "-0.1"],
                ["--value-next", "0.6"],                                    # _run's --value-blend 0.5: 0.5 + 0.6 > 1
                ["--value-blend", "-0.5", "--value-next", "1.5"], ["--value-blend", "1.5"]):
        with pytest.raises(SystemExit):
            _run(shards, tmp_path / "bad", *bad)


def test_value_next_trains_on_the_next_turns_root_value(shards, tmp_path, monkeypatch):
    """--value-next: the training value loss is the log loss against (1 - B - F) outcome + B root_value + F next_rv,
    with next_rv from ShardSet(next_rv=True) for the training and validation sets; it trains and differs."""
    import torch
    from draftzero.nets import data as D
    from draftzero.nets import targets as T
    seen = []
    real = D.ShardSet

    def spy(*a, **kw):
        seen.append(kw.get("next_rv", False))
        return real(*a, **kw)
    monkeypatch.setattr(D, "ShardSet", spy)
    base = _run(shards, tmp_path / "base")
    assert seen == [False, False]                               # off: the sets load as before
    s = _run(shards, tmp_path / "vn", "--value-next", "0.5")
    assert seen[2:] == [True, True]
    assert np.isfinite(s["best"]["holdout"]["total"]) and s["steps"] == base["steps"] == 12
    assert not _same(_weights(tmp_path / "vn" / "last.pt"), _weights(tmp_path / "base" / "last.pt"))
    assert json.loads((tmp_path / "vn" / "config.json").read_text())["args"]["value_next"] == 0.5
    model, _ = M.load(tmp_path / "base" / "last.pt")            # trained: a fresh value head scores 0.5 everywhere
    ds = real(shards / "train", next_rv=True)
    b = ds.collate(np.arange(64))
    assert b.next_rv is not None and float((b.next_rv != b.outcome).float().mean()) > 0.2
    with torch.no_grad():
        _, _, v0, x = train.losses(model, b, _args(value_blend=0.25))
        t1, p1, v1, _ = train.losses(model, b, _args(value_blend=0.25, value_next=0.5))
    vt, vm = T.value_target(b, 0.25, 0.5)
    vl = torch.nn.functional.binary_cross_entropy_with_logits(x["vlogit"].float(), vt, reduction="none")
    assert abs(float(v1) - float(vl[vm].mean())) < 1e-5 and abs(float(v1) - float(v0)) > 1e-4
    assert abs(float(t1) - (float(p1) + float(v1))) < 1e-5


# ------------------------------------------------------------------------------------------------ record weights

@pytest.fixture(scope="module")
def weighted_shards(tmp_path_factory):
    """The `shards` data again with weight.npy: 30% of the records at weight 0.1 (a decided-game search's)."""
    root = tmp_path_factory.mktemp("synw")
    pf.write_synthetic(root / "train", 12000, seed=11, shard_size=8000, game_len=10, weighted=0.3)
    pf.write_synthetic(root / "eval", 1000, seed=12, game_len=10, weighted=0.3)
    return root


def _with_ones(src, dst):
    """A copy of a synthetic shard tree with an explicit weight.npy of ones in every shard."""
    import shutil
    shutil.copytree(src, dst)
    for meta in dst.rglob("meta.json"):
        n = json.loads(meta.read_text())["n"]
        np.save(meta.parent / "weight.npy", np.ones(n, np.float32))
    return dst


def test_record_weights_of_one_are_bit_identical(shards, tmp_path, monkeypatch):
    """A batch whose record weights are all 1 trains bit for bit as before record weights (the losses before the
    extras): with no weight.npy (read as ones) and with an explicit weight.npy of ones alike."""
    ones = _with_ones(shards, tmp_path / "ones")
    s_ones = _run(ones, tmp_path / "w1")
    s_none = _run(shards, tmp_path / "w0")
    monkeypatch.setattr(train, "losses", _losses_before_extras)
    s_old = _run(shards, tmp_path / "old")
    for name in ("best.pt", "last.pt"):
        assert _same(_weights(tmp_path / "w1" / name), _weights(tmp_path / "old" / name)), name
        assert _same(_weights(tmp_path / "w0" / name), _weights(tmp_path / "old" / name)), name
    assert _evals(tmp_path / "w1") == _evals(tmp_path / "w0") == _evals(tmp_path / "old")
    assert s_ones["best"] == s_none["best"] == s_old["best"]


def test_record_weighted_losses_by_hand():
    import torch
    from draftzero.nets import targets as T
    torch.manual_seed(0)
    model = M.build("mlp", TINY["mlp"])
    b = _one_batch()
    rw = torch.where(torch.arange(b.B) % 3 == 0, torch.tensor(0.1), torch.tensor(1.0))
    with torch.no_grad():
        t0, p0, v0, x = train.losses(model, b, _args())                        # b.weight None: unweighted
        b.weight = torch.ones(b.B)
        t1, p1, v1, _ = train.losses(model, b, _args())
        assert torch.equal(t1, t0) and torch.equal(p1, p0) and torch.equal(v1, v0)
        b.weight = rw
        tw, pw, vw, _ = train.losses(model, b, _args())
        te, *_ = train.losses(model, b, _args(), weighted=False)
        tn, *_ = train.losses(model, b, _args(record_weights=False))
        ts, *_ = train.losses(model, b, _args(surprise_weight=0.5, value_surprise=0.1))
        b.weight = torch.zeros(b.B)
        tz, *_ = train.losses(model, b, _args())
    assert torch.equal(pw, p0) and torch.equal(vw, v0)                         # the reported losses stay unweighted
    assert torch.equal(te, t0) and torch.equal(tn, t0)                         # evaluation and --no-record-weights
    pm, ce = x["pm"], x["ce_state"]
    pce_w = float((rw * ce)[pm].sum() / rw[pm].sum())
    vt, vm = T.value_target(b, 0.5)
    vl = torch.nn.functional.binary_cross_entropy_with_logits(x["vlogit"].float(), vt, reduction="none")
    vloss_w = float((rw * vl)[vm].sum() / rw[vm].sum())
    assert abs(float(tw) - (pce_w + vloss_w)) < 1e-5 and abs(float(tw) - float(t0)) > 1e-4
    # with surprise weighting too: the record weight multiplies the surprise weight, normalised by the record weights
    sw = 0.5 + 0.5 * T.policy_surprise(b, x["tgt"]) / T.policy_surprise(b, x["tgt"])[pm].mean()
    err = (vt - torch.sigmoid(x["vlogit"].float())).abs()
    vsw = 0.9 + 0.1 * err / err[vm].mean()
    want = float((rw * sw * ce)[pm].sum() / rw[pm].sum() + (rw * vsw * vl)[vm].sum() / rw[vm].sum())
    assert abs(float(ts) - want) < 1e-5
    assert float(tz) == 0.0                                                     # every weight 0: nothing to learn


def test_record_weights_train_and_differ(shards, weighted_shards, tmp_path):
    """Weighted records change training; --no-record-weights trains exactly as the same data without weight.npy."""
    base = _run(shards, tmp_path / "base")
    w = _run(weighted_shards, tmp_path / "w")
    off = _run(weighted_shards, tmp_path / "off", "--no-record-weights")
    assert np.isfinite(w["best"]["holdout"]["total"]) and w["steps"] == base["steps"] == 12
    assert not _same(_weights(tmp_path / "w" / "last.pt"), _weights(tmp_path / "base" / "last.pt"))
    assert _same(_weights(tmp_path / "off" / "last.pt"), _weights(tmp_path / "base" / "last.pt"))
    assert _evals(tmp_path / "off") == _evals(tmp_path / "base")             # evaluation is unweighted either way
    assert json.loads((tmp_path / "off" / "config.json").read_text())["args"]["record_weights"] is False


def test_ema_is_the_bias_corrected_average():
    import torch
    lin = torch.nn.Linear(2, 1)
    ema = train.EMA(lin, 0.5)
    seq = [1.0, 3.0, 5.0]
    for v in seq:
        with torch.no_grad():
            lin.weight.fill_(v)
        ema.update(lin)
    # weights 0.5^2, 0.5, 1 over the three updates; the starting weights are not part of it
    want = (0.25 * 1 + 0.5 * 3 + 1 * 5) / 1.75
    assert torch.allclose(ema.model.weight, torch.full_like(ema.model.weight, want))
    assert ema.n == 3 and not ema.model.weight.requires_grad and lin.weight.requires_grad
    one = train.EMA(lin, 0.999)
    one.update(lin)
    assert torch.allclose(one.model.weight, lin.weight)        # the first update copies the weights


def test_ema_is_evaluated_saved_and_leaves_training_alone(shards, tmp_path):
    base = _run(shards, tmp_path / "base")
    s = _run(shards, tmp_path / "ema", "--ema-decay", "0.9", "--ema-every", "2")
    # the moving average never feeds back: the raw trajectory and best.pt (without --export-ema) are unchanged
    assert _same(_weights(tmp_path / "ema" / "last.pt"), _weights(tmp_path / "base" / "last.pt"))
    assert _same(_weights(tmp_path / "ema" / "best.pt"), _weights(tmp_path / "base" / "best.pt"))
    assert _evals(tmp_path / "ema") == _evals(tmp_path / "base")
    emas = _evals(tmp_path / "ema", "ema")
    assert [st for st, _ in emas] == [4, 8, 12] and set(emas[0][1]) == {"holdout", "evaldecks"}
    assert s["best_weights"] == "raw" and s["ema_updates"] == 6 and s["ema_decay"] == 0.9
    m, ck = M.load(tmp_path / "ema" / "ema.pt")
    assert ck["meta"]["weights"] == "ema" and ck["meta"]["ema_updates"] == 6 and ck["meta"]["step"] == 12
    assert ck["meta"]["metrics"] == emas[-1][1]
    assert not _same(_weights(tmp_path / "ema" / "ema.pt"), _weights(tmp_path / "ema" / "last.pt"))
    assert base["best_step"] == s["best_step"]


@pytest.mark.parametrize("ema_total, want", [(0.5, "ema"), (2.0, "raw"), (1.0, "raw")])
def test_export_ema_rule(shards, tmp_path, monkeypatch, ema_total, want):
    """--export-ema: at each evaluation the lower holdout total of the raw weights and the moving average is the
    candidate (the raw weights on a tie); best.pt keeps the lowest so far and says which it holds."""
    real = train.evaluate

    def fake(m, ds, idx, args, device):
        r = real(m, ds, idx, args, device)
        is_ema = not next(m.parameters()).requires_grad
        return {**r, "total": ema_total if is_ema else 1.0}
    monkeypatch.setattr(train, "evaluate", fake)
    s = _run(shards, tmp_path / "x", "--ema-decay", "0.9", "--export-ema", steps=8)
    assert s["best_weights"] == want and s["best_step"] == 4          # later evaluations tie, so the first stays
    ck = M.load(tmp_path / "x" / "best.pt")[1]
    assert ck["meta"]["weights"] == want and ck["meta"]["step"] == 4
    emas = dict(_evals(tmp_path / "x", "ema"))
    if want == "ema":
        assert ck["meta"]["metrics"] == emas[4] and s["best"] == emas[4]
        assert not _same(_weights(tmp_path / "x" / "best.pt"), _weights(tmp_path / "x" / "last.pt"))
    else:
        assert ck["meta"]["metrics"] == dict(_evals(tmp_path / "x"))[4]
