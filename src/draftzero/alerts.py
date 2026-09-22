"""alerts.py — tell a human when something surprising happens to a run.

A run dies at 3am on a rented box. Nothing watches it, and the first anyone knows is a
status check hours later against an idle machine still billing. These alerts exist so the
interesting events -- a crash, a restart, a shutdown -- reach a phone instead of a log file
nobody is reading.

Two sinks, both optional and independent:
  * alerts.jsonl in the run dir, always written. The durable record.
  * ntfy (https://ntfy.sh or a self-hosted server), when DZ_ALERT_TOPIC is set. It needs no
    account and no credentials on the worker, which is the point -- an SMTP password on a
    rented pod is a worse problem than a missed alert. Set DZ_ALERT_EMAIL and ntfy forwards
    to that address as well.

ntfy topics are public to anyone who guesses the name, so use an unguessable one and keep
message bodies free of anything sensitive: what happened, not what the data contains.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

#: events worth waking someone for, in rough order of how much they matter
CRASH = "crash"
RESTART = "restart"
RESTART_FAILED = "restart_failed"
STALLED = "stalled"
BUDGET = "budget_cap"
TERMINATING = "terminating"
SYNC_FAILED = "sync_failed"
DONE = "done"

PRIORITY = {CRASH: "high", RESTART_FAILED: "urgent", SYNC_FAILED: "urgent",
            TERMINATING: "high", STALLED: "high", BUDGET: "default",
            RESTART: "default", DONE: "default"}

TAG = {CRASH: "boom", RESTART: "arrows_counterclockwise", RESTART_FAILED: "rotating_light",
       STALLED: "warning", BUDGET: "moneybag", TERMINATING: "wastebasket",
       SYNC_FAILED: "rotating_light", DONE: "white_check_mark"}


def send(event: str, summary: str, detail: str = "", run_dir: Optional[Path] = None) -> bool:
    """Record an event and try to push it. Never raises: an alert must not kill a run."""
    rec = {"at": datetime.now(timezone.utc).isoformat(), "event": event,
           "summary": summary, "detail": detail[:2000],
           "run": str(run_dir) if run_dir else None,
           "pod": os.environ.get("RUNPOD_POD_ID") or os.environ.get("HOSTNAME", "")}
    if run_dir:
        try:
            with (Path(run_dir) / "alerts.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass
    return _push(event, summary, detail)


def _push(event: str, summary: str, detail: str) -> bool:
    topic = os.environ.get("DZ_ALERT_TOPIC")
    if not topic:
        return False
    base = os.environ.get("DZ_ALERT_SERVER", "https://ntfy.sh").rstrip("/")
    body = (summary + ("\n\n" + detail if detail else ""))[:3000]
    headers = {
        "Title": f"DraftZero: {event.replace('_', ' ')}",
        "Priority": PRIORITY.get(event, "default"),
        "Tags": TAG.get(event, "information_source"),
    }
    token = os.environ.get("DZ_ALERT_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    email = os.environ.get("DZ_ALERT_EMAIL")

    # Try with e-mail, then WITHOUT it. ntfy.sh rejects anonymous e-mail sending outright
    # (HTTP 400 "anonymous email sending is not allowed"), and if that rejection were left
    # to fail the whole call, one unset token would silently disable every alert -- the
    # push notification included. Push is the channel that must never break.
    attempts = [{**headers, "Email": email}] if email else []
    attempts.append(headers)

    for hdrs in attempts:
        for attempt in range(3):
            try:
                req = urllib.request.Request(f"{base}/{topic}", data=body.encode(), headers=hdrs)
                with urllib.request.urlopen(req, timeout=15) as r:
                    if 200 <= r.status < 300:
                        return True
            except urllib.error.HTTPError as e:
                if 400 <= e.code < 500:
                    break                     # config problem: drop e-mail, try push
                time.sleep(2 * (attempt + 1))
            except Exception:                 # noqa: BLE001 - any failure is non-fatal
                time.sleep(2 * (attempt + 1))
    return False
