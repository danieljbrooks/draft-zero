"""Pool path handling — the thing that killed the first two RunPod runs.

Run one: pool files held the laptop's absolute paths, so every game on the pod failed with
"Couldn't load deck, deck size=0". Run two: an rsync from the laptop restored those paths
and it happened again. These tests pin the fix.
"""
from pathlib import Path

import pytest

from draftzero import paths


def test_read_pool_accepts_stems(tmp_path):
    f = tmp_path / "train.txt"
    f.write_text("FDN_top_00001_WB\nFDN_top_00002_BR\n")
    assert paths.read_pool(f) == ["FDN_top_00001_WB", "FDN_top_00002_BR"]


def test_read_pool_normalizes_legacy_absolute_paths(tmp_path):
    f = tmp_path / "train.txt"
    f.write_text("/Users/someone/decks/FDN_top_00001_WB.dck\nFDN_top_00002_BR\n")
    assert paths.read_pool(f) == ["FDN_top_00001_WB", "FDN_top_00002_BR"]


def test_read_pool_ignores_blank_lines(tmp_path):
    f = tmp_path / "train.txt"
    f.write_text("FDN_top_00001_WB\n\n   \nFDN_top_00002_BR\n")
    assert len(paths.read_pool(f)) == 2


def test_resolve_pool_writes_absolute_paths(tmp_path):
    root = tmp_path / "decks"
    root.mkdir()
    for stem in ("a", "b"):
        (root / f"{stem}.dck").write_text("x")
    out = paths.resolve_pool(["a", "b"], root, tmp_path / "resolved.txt")
    lines = out.read_text().split()
    assert lines == [str(root / "a.dck"), str(root / "b.dck")]
    assert all(Path(l).is_absolute() for l in lines)


def test_resolve_pool_raises_when_decks_missing(tmp_path):
    """The whole point: fail here, loudly, not an hour later inside the JVM."""
    with pytest.raises(FileNotFoundError) as e:
        paths.resolve_pool(["nope"], tmp_path / "empty", tmp_path / "out.txt")
    assert "deck_root" in str(e.value)


def test_build_pools_from_split_column(tmp_path):
    meta = tmp_path / "decks.tsv"
    meta.write_text("deck\tsplit\tcolors\nd1\ttrain\tWB\nd2\teval\tUR\nd3\ttrain\tRG\n")
    counts = paths.build_pools(meta, tmp_path / "pools")
    assert counts == {"train": 2, "eval": 1}
    assert (tmp_path / "pools" / "train.txt").read_text().split() == ["d1", "d3"]


def test_normalize_pool_file_is_idempotent(tmp_path):
    f = tmp_path / "train.txt"
    f.write_text("/abs/path/FDN_x.dck\n")
    paths.normalize_pool_file(f)
    assert f.read_text().strip() == "FDN_x"
    assert paths.normalize_pool_file(f) == 0
