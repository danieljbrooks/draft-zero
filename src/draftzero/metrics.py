"""metrics.py: a placeholder between the XMage-era log parser and the v0.2.0 metrics log.

The old contents (MageZero JVM log parsing, effect classification, per-game rows) went with XMage. What other code
still uses stays: `wilson` (now defined in draftzero.eval.stats) and `append_jsonl`. The v0.2.0 metrics log
(MetricsLog, the "draftzero-metrics/1" line schema; contracts §3) replaces this module.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from draftzero.eval.stats import wilson

__all__ = ["append_jsonl", "wilson"]


def append_jsonl(path: Path, record: dict) -> None:
    """Append `record` to a JSON-lines file as one line, with a local-time "ts" first; creates the parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(), **record}, default=str) + "\n")
