"""metrics.py — the run's metrics log (interface 9, contracts §3): one JSON line per measurement, one schema.

    log = MetricsLog(path, run="smoke", component="learner", machine="d1")
    log.log("learn.policy_ce", 1.83, version=4, step=1200)
    log.log("eval.score", 0.56, version=4, n=200, lo=0.49, hi=0.63, opponent="prev")
    log.log_rate("eval.score", k=112, n=200, version=4, opponent="prev")      # value k/n with its Wilson interval

Each line is {"schema", "time", "run", "component", "machine", "name", "value", "version", "step", "n", "lo", "hi",
**tags}: every field is present (null when it does not apply); "time" is UTC ISO 8601 ("...Z"); a value that is not
finite (NaN, inf) is written as null, so a bad loss cannot make the file unreadable. Tags (opponent=..., split=...)
are extra keys and may not reuse a field's name. In the store, each component writes
metrics/<component>/<machine>/<seq:06d>.jsonl; `read` takes files or folders and skips lines it cannot parse (a
torn last line from a concurrent writer).

Standard names (NAMES) keep runs comparable; new names are fine, but reuse these where they fit.
`append_jsonl` (a "ts"-first JSON line) stays for older callers.
"""
from __future__ import annotations

import json
import math
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = ["SCHEMA", "FIELDS", "NAMES", "MetricsLog", "read", "tags", "wilson", "append_jsonl"]

SCHEMA = "draftzero-metrics/1"
FIELDS = ("schema", "time", "run", "component", "machine", "name", "value", "version", "step", "n", "lo", "hi")

NAMES = {
    "throughput.games_per_hour": "complete games an hour, per actor machine",
    "throughput.positions_per_hour": "recorded positions an hour, per actor machine",
    "throughput.ms_per_searched": "wall milliseconds per searched decision",
    "learn.policy_ce": "policy cross-entropy against the search's visits (holdout)",
    "learn.value_logloss": "value log loss against the value target (holdout)",
    "learn.top1_visits": "share of positions whose top policy move is the search's most visited (holdout)",
    "learn.value_auc": "value AUC against the game result (holdout)",
    "eval.score": "paired score against the `opponent` tag, with n pairs and a 95% interval",
    "eval.elo": "Bradley-Terry rating (Elo scale), with its interval",
    "data.window_positions": "positions in the learner's training window",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _num(v) -> float | None:
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _int(v) -> int | None:
    return None if v is None else int(v)


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval (lo, hi) for k successes in n trials (k may be fractional: draws count 0.5);
    (0.0, 1.0) when n is 0. eval/stats.py keeps its own (rate, lo, hi) variant for the GIH tables."""
    if n <= 0:
        return 0.0, 1.0
    p = min(max(k / n, 0.0), 1.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


class MetricsLog:
    """Appends metric lines to one JSON-lines file; safe to share between threads of one process."""

    def __init__(self, path: Path, *, run: str, component: str, machine: str | None = None):
        self.path = Path(path)
        self.run = run
        self.component = component
        self.machine = machine or socket.gethostname()
        self._lock = threading.Lock()

    def line(self, name: str, value: float | None, *, version: int | None = None, step: int | None = None,
             n: int | None = None, lo: float | None = None, hi: float | None = None, **tags) -> dict:
        """The line `log` would write, without writing it."""
        clash = sorted(set(tags) & set(FIELDS))
        if clash:
            raise ValueError(f"metric tags may not reuse field names: {clash}")
        return {"schema": SCHEMA, "time": _now(), "run": self.run, "component": self.component,
                "machine": self.machine, "name": str(name), "value": _num(value), "version": _int(version),
                "step": _int(step), "n": _int(n), "lo": _num(lo), "hi": _num(hi), **tags}

    def _append(self, lines: list[dict]) -> None:
        text = "".join(json.dumps(x, default=str, allow_nan=False) + "\n" for x in lines)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(text)

    def log(self, name: str, value: float, *, version: int | None = None, step: int | None = None,
            n: int | None = None, lo: float | None = None, hi: float | None = None, **tags) -> dict:
        """Append one line and return it."""
        x = self.line(name, value, version=version, step=step, n=n, lo=lo, hi=hi, **tags)
        self._append([x])
        return x

    def log_rate(self, name: str, k: float, n: int, *, version: int | None = None, step: int | None = None,
                 z: float = 1.96, **tags) -> dict:
        """A rate k/n with its Wilson interval (lo, hi) and n; value null when n is 0."""
        lo, hi = wilson(k, n, z)
        return self.log(name, k / n if n else None, version=version, step=step, n=n, lo=lo, hi=hi, **tags)

    def log_many(self, values: Mapping[str, float], *, version: int | None = None, step: int | None = None,
                 **tags) -> list[dict]:
        """One line per {name: value}, sharing version, step and tags; written in one append."""
        xs = [self.line(k, v, version=version, step=step, **tags) for k, v in values.items()]
        self._append(xs)
        return xs


def _files(p: Path) -> list[Path]:
    return sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]


def read(paths: Iterable[Path] | Path | str) -> list[dict]:
    """Every metric line in these files (folders: every *.jsonl below them, in path order), in file order.
    Missing files, blank lines, lines that are not JSON objects and lines without a "name" are skipped."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out: list[dict] = []
    for p in paths:
        for f in _files(Path(p)):
            try:
                text = f.read_text()
            except (FileNotFoundError, IsADirectoryError):
                continue
            for raw in text.splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    x = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(x, dict) and "name" in x:
                    out.append(x)
    return out


def tags(line: Mapping[str, Any]) -> dict:
    """A line's tags: its keys that are not schema fields."""
    return {k: v for k, v in line.items() if k not in FIELDS}


def append_jsonl(path: Path, record: dict) -> None:
    """Append `record` to a JSON-lines file as one line, with a local-time "ts" first; creates the parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(), **record}, default=str) + "\n")
