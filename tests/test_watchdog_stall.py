"""Stall detection.

A false positive here costs a pod. It happened: games.jsonl only grows when a whole chunk
finishes (~27 min), so a run whose chunk ran long looked stalled at the 45-minute threshold
while its JVM log was being written continuously. The watchdog stopped the trainer, synced,
verified, and destroyed a healthy machine.
"""
import time

import pytest

pytest.importorskip("magezero", reason="engine dependency not installed")

from draftzero import watchdog  # noqa: E402


def test_activity_is_fresh_when_a_file_was_just_written(tmp_path):
    (tmp_path / "jvm_gen1_00.log").write_text("playing")
    assert watchdog.seconds_since_activity(tmp_path) < 5


def test_activity_reports_age_of_the_newest_file(tmp_path):
    old = tmp_path / "old.log"; old.write_text("x")
    past = time.time() - 3600
    import os; os.utime(old, (past, past))
    assert watchdog.seconds_since_activity(tmp_path) > 3000
    # a fresh write anywhere in the dir resets it
    (tmp_path / "new.log").write_text("y")
    assert watchdog.seconds_since_activity(tmp_path) < 5


def test_empty_dir_is_not_reported_as_ancient(tmp_path):
    """No files must not read as 'infinitely quiet' and trip a stall."""
    assert watchdog.seconds_since_activity(tmp_path) == 0.0


def test_missing_dir_is_safe(tmp_path):
    assert watchdog.seconds_since_activity(tmp_path / "nope") == 0.0


def test_stall_requires_all_three_signals():
    """The regression: no new games alone must NOT mean stalled."""
    stall = 45
    def stalled(idle, quiet, alive):
        return idle >= stall and quiet >= stall and not alive

    assert not stalled(50, 0, True),  "JVM writing + trainer alive: healthy, mid-chunk"
    assert not stalled(50, 50, True), "trainer alive: not stalled"
    assert not stalled(50, 2, False), "recent writes: not stalled"
    assert not stalled(10, 50, False), "games still landing: not stalled"
    assert stalled(50, 50, False),    "no games, no writes, no trainer: genuinely stalled"
