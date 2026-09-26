"""Checkpoint sync to the persist dir.

In the exp #2 pilot the persist dir was data/persist, inside data/, which is itself synced.
Each rsync copied the previous copy into itself (data/persist/data/persist/...) until it
failed, so no checkpoint ever reached the volume (docs/006).
"""
import shutil

import pytest

pytest.importorskip("magezero", reason="engine dependency not installed")

from draftzero import watchdog  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("rsync"), reason="rsync not installed")


def test_persist_dir_inside_a_synced_dir_is_not_copied_into_itself(tmp_path):
    data = tmp_path / "data"
    (data / "M" / "ver1").mkdir(parents=True)
    (data / "M" / "ver1" / "shard.hdf5").write_text("s")
    persist = data / "persist"
    for _ in range(3):
        ok, detail = watchdog.sync([data], persist)
        assert ok, detail
    assert (persist / "data" / "M" / "ver1" / "shard.hdf5").exists()
    assert not (persist / "data" / "persist").exists()


def test_unrelated_persist_dir_syncs_normally(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "model.pt.gz").write_text("w")
    ok, _ = watchdog.sync([models], tmp_path / "volume")
    assert ok and (tmp_path / "volume" / "models" / "model.pt.gz").exists()
