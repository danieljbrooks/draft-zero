"""draftzero.resources: the open-file limit the loop roles and the trainer raise at start."""
from __future__ import annotations

import json
import subprocess
import sys


def test_raise_open_files_lifts_the_soft_limit_up_to_the_hard_one():
    """A RunPod pod starts processes at a soft limit of 1,024 open files (hard 1,048,576); a learner's window of
    ~80 shards needs more, since every memory-mapped array holds a descriptor (E2c, 14 Oct)."""
    code = ("import json, resource\n"
            "from draftzero.resources import raise_open_files\n"
            "s, h = resource.getrlimit(resource.RLIMIT_NOFILE)\n"
            "resource.setrlimit(resource.RLIMIT_NOFILE, (256, h))\n"
            "want = 4096 if h == resource.RLIM_INFINITY else min(4096, h)\n"
            "got = raise_open_files(4096)\n"
            "print(json.dumps([got, resource.getrlimit(resource.RLIMIT_NOFILE)[0], want]))\n")
    got, now, want = json.loads(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                               check=True).stdout)
    assert got == now == want


def test_every_loop_role_raises_it(monkeypatch, tmp_path):
    from draftzero import resources
    from draftzero.loop import cli
    calls = []
    monkeypatch.setattr(resources, "raise_open_files", lambda *a, **k: calls.append(1) or 65536)
    cli.main(["status", str(tmp_path / "no-store")])
    assert calls == [1]
