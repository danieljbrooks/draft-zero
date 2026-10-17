"""Interface 9, the run manifest and metrics schema: the format strings and the keys every reader relies on."""
from __future__ import annotations

import json

from draftzero import config as C
from draftzero import manifest as MF
from draftzero import metrics as M

METRIC_FIELDS = ("schema", "time", "run", "component", "machine", "name", "value", "version", "step", "n", "lo", "hi")
MANIFEST_KEYS = {"format", "run", "created", "draftzero", "engine_pin", "config", "config_hash", "engine", "belief",
                 "python", "torch", "numpy", "platform", "machine", "seed"}
BELIEF_KEYS = {"decks", "pool", "split", "corpus_hash"}
ENGINE_KEYS = {"repo", "ref", "dirty", "binary_sha256", "build", "cards_lock", "census"}


def test_metric_line():
    assert M.SCHEMA == "draftzero-metrics/1" and M.FIELDS == METRIC_FIELDS
    log = M.MetricsLog("unused.jsonl", run="r", component="learner", machine="m")
    x = log.line("learn.policy_ce", 1.0, version=1, step=2, n=3, lo=0.5, hi=1.5, split="holdout")
    assert tuple(x)[:12] == METRIC_FIELDS and x["split"] == "holdout"
    json.loads(json.dumps(x))


def test_metric_names():
    for name in ("throughput.games_per_hour", "throughput.positions_per_hour", "throughput.ms_per_searched",
                 "learn.policy_ce", "learn.value_logloss", "learn.top1_visits", "learn.value_auc", "eval.score",
                 "eval.elo", "data.window_positions"):
        assert name in M.NAMES


def test_manifest(tmp_path):
    cfg = C.from_dict({"run": "c", "engine": {"root": str(tmp_path)},
                       "data": {"decks": "assets/sample/decks", "pool": "assets/sample/pool.tsv"}})
    m = MF.collect(cfg)
    assert MF.FORMAT == "draftzero-manifest/1" and m["format"] == MF.FORMAT
    assert set(m) == MANIFEST_KEYS and set(m["engine"]) == ENGINE_KEYS
    assert set(m["belief"]) == BELIEF_KEYS and len(m["belief"]["corpus_hash"]) == 16   # search.closedlist is on
    assert set(m["draftzero"]) == {"version", "commit", "branch", "dirty"}
    assert set(m["machine"]) >= {"host", "cpu_quota", "mem_gb"}
    assert MF.PATH == "manifest.json"
