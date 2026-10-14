"""Fixtures for the engine tests.

`gorge`: the real driver (tests marked `engine`): the gorge checkout DZ_GORGE_ROOT (default /work/gorge-fork, built by
engine/gorge/build.sh) and the deck pool DZ_GORGE_DATA (default this checkout's data/gorge, else /work/dz-main's).
Skipped when either is missing.

`fake_dzgorge`: a stand-in driver (a Python script) that plays instant fake games, hangs where told and logs its
command lines, for the wrapper's own logic (resume, kill, trim, flags) without the engine.
"""
import json
import os
import sys
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


FAKE = r'''#!{python}
"""A fake dzgorge: `play` writes instant games (one JSON line each, a gzip member each with -corpus), in game order,
flushed per game; env FAKE_HANG=pairs hang (at leg FAKE_HANG_LEG), FAKE_HANG_BUSY=pairs hang unless played alone,
FAKE_IGNORE_TERM ignores SIGTERM while hanging, FAKE_FAIL refuses like a bad spec, FAKE_PID gets its pid.
`pack` writes synthetic shards."""
import gzip, json, os, signal, subprocess, sys, time

args = sys.argv[1:]
if os.environ.get("FAKE_PID"):
    with open(os.environ["FAKE_PID"], "w") as f:
        f.write(str(os.getpid()))
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps(args) + "\n")
cmd, flags, bools, i = args[0], {{}}, set(), 1
while i < len(args):
    if args[i] in ("-mull-heuristic", "-no-paid-mana-census", "-single"):
        bools.add(args[i]); i += 1
    else:
        flags[args[i]] = args[i + 1]; i += 2
if cmd == "pack":
    from draftzero.nets import packfmt
    packfmt.write_synthetic(flags["-out"], 50, seed=1, shard_size=int(flags["-shard"]), game_len=10)
    sys.exit(0)
if os.environ.get("FAKE_FAIL"):
    print("play: policy 'az:frob': unknown key", file=sys.stderr)
    sys.exit(2)
pairs = lambda k: {{int(x) for x in os.environ.get(k, "").split(",") if x}}
first, n = int(flags["-first-pair"]), int(flags["-pairs"])
legs = 1 if flags["-a"] == flags["-b"] or "-single" in bools else 2
hang, busy, hang_leg = pairs("FAKE_HANG"), pairs("FAKE_HANG_BUSY"), int(os.environ.get("FAKE_HANG_LEG", "0"))
out = open(flags["-out"], "wb")
corpus = open(flags["-corpus"], "wb") if "-corpus" in flags else None
g = 0
for p in range(first, first + n):
    for leg in range(legs):
        if leg == hang_leg and (p in hang or (p in busy and n > 1)):
            out.write(b'{{"game":%d,"pair":%d' % (g, p))
            out.flush()
            if os.environ.get("FAKE_IGNORE_TERM"):
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
            child = subprocess.Popen(["sleep", "3600"])
            with open(os.environ["FAKE_CHILD"], "w") as f:
                f.write(str(child.pid))
            time.sleep(3600)
        row = {{"game": g, "pair": p, "leg": leg, "seed": 7, "result": "AB"[(p + leg) % 2], "ms": 1.5}}
        if corpus:
            row["visits"] = 2
        out.write(json.dumps(row, separators=(",", ":")).encode() + b"\n")
        out.flush()
        if corpus:
            body = "".join(json.dumps({{"game_id": "p%07dg%d" % (p, leg), "i": k}}) + "\n" for k in range(2))
            corpus.write(gzip.compress(body.encode()))
            corpus.flush()
        g += 1
out.close()
with open(flags["-out"] + ".summary.json", "w") as f:
    json.dump({{"games": g, "pairs": n, "first_pair": first, "legs": legs, "fake": True}}, f)
'''


class Fake:
    def __init__(self, tmp: Path, monkeypatch):
        from draftzero.engine import Gorge
        self.root = tmp / "gorge"
        (self.root / "bin").mkdir(parents=True)
        (self.root / ".cards").mkdir()
        self.binary = self.root / "bin" / "dzgorge"
        self.binary.write_text(FAKE.format(python=sys.executable))
        self.binary.chmod(0o755)
        self.log, self.child = tmp / "fake.log", tmp / "fake.child"
        monkeypatch.setenv("FAKE_LOG", str(self.log))
        monkeypatch.setenv("FAKE_CHILD", str(self.child))
        self.monkeypatch = monkeypatch
        self.gorge = Gorge(self.root, tmp / "decks", tmp / "pool.tsv")

    def env(self, **kw):
        for k, v in kw.items():
            self.monkeypatch.setenv(k, str(v))

    def calls(self) -> list[list[str]]:
        return [json.loads(l) for l in self.log.read_text().splitlines()] if self.log.exists() else []


@pytest.fixture
def fake_dzgorge(tmp_path, monkeypatch):
    return Fake(tmp_path, monkeypatch)
