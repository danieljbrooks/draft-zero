"""control.json: what every worker should be doing (docs/021 §3.2). The controller is its only writer.

    {"version": 7,                                  the version workers play
     "weights": "weights/v0007.pt.gz",
     "sha256": "...",
     "start_weights": "weights/v0000.pt.gz",        the starting network (the past-version opponent, evaluations)
     "play": {...},                                 the run's play settings (config.play)
     "evals": [{job}, ...],                         active evaluation jobs (evals.make_job), shards assigned
     "stop": false,
     "updated": <unix time>}

weights/latest.json is the trainer's: {"version", "path", "sha256", "parent", "games", "positions", "metrics",
"time"}. The controller copies a new version from it into control.json.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

CONTROL = "control.json"
LATEST = "weights/latest.json"
STATUS = "status.json"
STATUS_MD = "status.md"


def weights_path(version: int) -> str:
    return f"weights/v{version:04d}.pt.gz"


def games_prefix(version: int | None = None) -> str:
    return "games/" if version is None else f"games/v{version:04d}/"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def new_control(latest: dict, play: dict) -> dict:
    return {"version": latest["version"], "weights": latest["path"], "sha256": latest["sha256"],
            "start_weights": weights_path(0), "play": dict(play), "evals": [], "stop": False, "updated": time.time()}


def read(store) -> dict | None:
    return store.read_json(CONTROL)


def write(store, control: dict, message: str = "") -> None:
    control["updated"] = time.time()
    store.write_json(CONTROL, control, message or f"control: v{control['version']:04d}")
