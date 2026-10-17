"""Fixtures for the engine tests.

`gorge`: the real driver (tests marked `engine`): the gorge checkout DZ_GORGE_ROOT (default /work/gorge-fork, built by
engine/gorge/build.sh) and the deck pool DZ_GORGE_DATA (default this checkout's data/gorge, else /work/dz-main's).
Skipped when either is missing. The fake driver for the wrapper's own logic is in tests/unit/conftest.py.
"""
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _data_dir() -> Path | None:
    for p in (os.environ.get("DZ_GORGE_DATA"), REPO / "data" / "gorge", "/work/dz-main/data/gorge"):
        if p and (Path(p) / "pool.tsv").exists() and (Path(p) / "decks").is_dir():
            return Path(p)
    return None


@pytest.fixture(scope="session")
def gorge():
    from draftzero.engine import Gorge
    root = Path(os.environ.get("DZ_GORGE_ROOT", "/work/gorge-fork"))
    if not (root / "bin" / "dzgorge").exists() or not (root / ".cards").is_dir():
        pytest.skip(f"no built dzgorge with a card corpus under {root} (engine/gorge/build.sh)")
    data = _data_dir()
    if data is None:
        pytest.skip("no deck pool (data/gorge/decks + pool.tsv); set DZ_GORGE_DATA")
    return Gorge(root, data / "decks", data / "pool.tsv")
