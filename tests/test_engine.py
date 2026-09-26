"""The v0.2 engine adapter: parsing MageZero's train/test output, and staging frozen checkpoints."""
import os
from pathlib import Path

import pytest

from draftzero import engine

TRAIN_OUT = """Building feature vocab from dataset (ignore-list rule over observed ids)
Successfully loaded checkpoint from models/T/ver1/model.pt.gz
feature vocab: 2098 kept ids in this dataset, 1281 rows from checkpoint, 817 added -> 2098 embedding rows
[opponent filter] scanned 728 samples  kept 728 (removed 0 opponent states).
Epoch 1  priority_A_loss=0.933  priority_B_loss=0.000 choose_target_loss=0.826 choose_use_loss=0.002 value_loss=0.486 l1_dense=0.0 l1_sparse=0.0 decision_states=318
Epoch 2  priority_A_loss=0.901  priority_B_loss=0.000 choose_target_loss=0.800 choose_use_loss=0.002 value_loss=0.401 l1_dense=0.0 l1_sparse=0.0 decision_states=318
""".splitlines()

TEST_OUT = """feature vocab: 1281 rows
Loaded checkpoint from models/T/ver1/model.pt.gz
Validation loss:  priority_A_loss=0.905  priority_B_loss=0.000 choose_target_loss=0.816 choose_use_loss=0.004 value_loss=0.579 avg_total_loss=2.304 decision_states=410
Test priority_A_accuracy=0.043
No priority B samples in test set to calculate accuracy.
Test choose_target_accuracy=0.000
Test choose_use_accuracy=0.349
""".splitlines()


def test_train_output_becomes_epoch_rows():
    rows = engine.parse_train_output(TRAIN_OUT, "T", 1, gen=3)
    assert [r["epoch"] for r in rows] == [1, 2]
    r = rows[1]
    assert r["kind"] == "train_epoch" and r["gen"] == 3
    assert r["train_value_loss"] == pytest.approx(0.401)
    assert r["train_priority_A_loss"] == pytest.approx(0.901)
    assert r["decision_states"] == 318
    assert r["embed_rows"] == 2098 and r["embed_rows_added"] == 817


def test_test_output_becomes_eval_prev_row():
    row = engine.parse_test_output(TEST_OUT, "T", 1, gen=4)
    assert row["kind"] == "eval_prev_model" and row["gen"] == 4
    assert row["value_loss"] == pytest.approx(0.579)
    assert row["decision_states"] == 410
    assert row["priority_A_acc"] == pytest.approx(0.043)
    assert row["choose_use_acc"] == pytest.approx(0.349)


def test_test_output_without_results_is_skipped():
    assert engine.parse_test_output(["ERROR: Checkpoint not found"], "T", 1, gen=0) is None


def test_frozen_checkpoint_is_staged_as_its_own_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = Path("models/M/ver1/gen5.pt.gz")
    src.parent.mkdir(parents=True)
    src.write_bytes(b"weights")
    assert engine._serve_name("M", 1, None) == "M"
    name = engine._serve_name("M", 1, "gen5")
    staged = Path(f"models/{name}/ver1/model.pt.gz")
    assert staged.read_bytes() == b"weights"
    # staging again (a later generation) replaces the link instead of failing
    assert engine._serve_name("M", 1, "gen5") == name
    with pytest.raises(FileNotFoundError):
        engine._serve_name("M", 1, "gen9")


def test_jvm_command_uses_console_logging_and_private_tmpdir(tmp_path):
    cmd, cwd = engine.jvm_command("game.yml", heap="11g", gc="zgc-gen", tmpdir=tmp_path / "t")
    assert f"-Dlog4j.configuration=file:{engine.LOG4J}" in cmd and engine.LOG4J.exists()
    assert "-Xmx11g" in cmd and "-XX:+ZGenerational" in cmd
    assert f"-Djava.io.tmpdir={(tmp_path / 't').resolve()}" in cmd
    assert cmd[-1] == os.path.abspath("game.yml")
