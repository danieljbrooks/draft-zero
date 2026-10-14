"""tracking.py — mirror the run's metric lines into an experiment tracker (Trackio), never depend on it.

The run's files stay the record (interface 9); a tracker is a view onto them. The controller, on a machine with the
token, mirrors the metric lines it reads from the store:

    tr = Tracker(cfg)                          # cfg.tracking: {trackio: true, project: draftzero}
    tr.mirror(metrics.read(paths))             # lines already mirrored are skipped, so re-reading files is fine
    tr.close()

Tracker is a no-op when cfg.tracking.trackio is false, and also (with one warning a process) when trackio is not
installed (`pip install 'draftzero[tracking]'`) or fails: tracking never stops a run. A line becomes one tracker
metric named after the line, with its tags appended ("eval.score/opponent=prev") and, for actor lines, its machine
("throughput.games_per_hour/machine=r1"), logged at step = the line's version (else its step, else none).
Swapping trackers means replacing this module.
"""
from __future__ import annotations

import warnings
from typing import Any, Iterable, Mapping

_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


def metric_key(line: Mapping[str, Any]) -> str:
    """The tracker's name for a metric line: name, then sorted tags (and the machine for actor lines)."""
    from draftzero.metrics import tags
    extra = dict(tags(line))
    if line.get("component") == "actor" and line.get("machine"):
        extra["machine"] = line["machine"]
    return "/".join([str(line.get("name"))] + [f"{k}={extra[k]}" for k in sorted(extra)])


def _identity(line: Mapping[str, Any]) -> tuple:
    return (line.get("run"), line.get("component"), line.get("machine"), line.get("time"), line.get("name"),
            line.get("version"), line.get("step"), metric_key(line))


class Tracker:
    """Mirrors metric lines into Trackio when enabled and available; otherwise every call does nothing."""

    def __init__(self, cfg, *, name: str | None = None, config: Mapping | None = None):
        self.enabled = False
        self._tk = None
        self._seen: set[tuple] = set()
        tr = cfg.tracking
        if not tr.trackio:
            return
        try:
            import trackio
        except Exception as e:  # ImportError, or a broken install
            _warn_once("import", f"tracking.trackio is on but trackio cannot be imported ({e}); metrics stay in the "
                                 f"store only. pip install 'draftzero[tracking]' to mirror them.")
            return
        if config is None:
            from draftzero.config import to_dict
            config = to_dict(cfg)
        try:
            trackio.init(project=tr.project, name=name or cfg.run, config=dict(config), resume="allow")
        except Exception as e:
            _warn_once("init", f"trackio.init failed ({type(e).__name__}: {e}); metrics stay in the store only")
            return
        self._tk = trackio
        self.enabled = True

    def mirror(self, lines: Iterable[Mapping[str, Any]]) -> int:
        """Log the lines not mirrored before (and with a value); returns how many were logged."""
        if not self.enabled:
            return 0
        k = 0
        for line in lines:
            ident = _identity(line)
            if ident in self._seen or line.get("value") is None:
                continue
            self._seen.add(ident)
            step = line.get("version") if line.get("version") is not None else line.get("step")
            try:
                if step is None:
                    self._tk.log({metric_key(line): line["value"]})
                else:
                    self._tk.log({metric_key(line): line["value"]}, step=int(step))
            except Exception as e:
                _warn_once("log", f"trackio.log failed ({type(e).__name__}: {e}); continuing without it")
                continue
            k += 1
        return k

    def close(self) -> None:
        if self.enabled:
            self.enabled = False
            try:
                self._tk.finish()
            except Exception as e:
                _warn_once("finish", f"trackio.finish failed ({type(e).__name__}: {e})")

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
