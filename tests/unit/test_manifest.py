"""draftzero.manifest: the run manifest (interface 9) from a fake Gorge, from a gorge checkout's files, with and
without git."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess

import pytest

from draftzero import __version__
from draftzero import config as C
from draftzero import manifest as MF

KEYS = {"format", "run", "created", "draftzero", "engine_pin", "config", "config_hash", "engine", "belief", "python",
        "torch", "numpy", "platform", "machine", "seed"}
SAMPLE = {"decks": "assets/sample/decks", "pool": "assets/sample/pool.tsv"}


def fake_engine_root(tmp_path, lock: bool = True):
    root = tmp_path / "gorge"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "dzgorge").write_bytes(b"\x7fELF fake driver")
    cards = {"repo": "https://github.com/adams-shaun/forge.git", "ref": "098b0553", "digest": "4a9b"}
    build = {"repo": "https://github.com/danieljbrooks/gorge.git", "ref": "5fd039efe" + "0" * 31, "dirty": False,
             "go_version": "go1.25.8", "built_utc": "2026-10-14T15:40:02Z", "binary_sha256": "stale", "cards_lock": cards}
    (root / "bin" / "dzgorge.build.json").write_text(json.dumps(build))
    if lock:
        (root / ".cards").mkdir()
        (root / ".cards" / "cards.lock").write_text(json.dumps({**cards, "files": 34074}))
    return root, build


class FakeGorge:
    def version(self):
        return {"repo": "r", "ref": "f" * 40, "dirty": False, "binary_sha256": "ab", "build": {}, "cards_lock": {},
                "census": False}


@pytest.fixture
def cfg(tmp_path):
    root, _ = fake_engine_root(tmp_path)
    return C.from_dict({"run": "m1", "seed": 7, "engine": {"root": str(root), "census": False}, "data": SAMPLE})


def test_collect_has_every_key(cfg):
    m = MF.collect(cfg, gorge=FakeGorge())
    assert set(m) == KEYS
    assert m["format"] == MF.FORMAT == "draftzero-manifest/1"
    assert (m["run"], m["seed"]) == ("m1", 7)
    assert m["config"] == C.to_dict(cfg) and m["config_hash"] == C.config_hash(cfg)
    assert m["engine"] == FakeGorge().version()
    assert m["draftzero"]["version"] == __version__ and set(m["draftzero"]) == {"version", "commit", "branch", "dirty"}
    assert set(m["machine"]) == {"host", "cpu_quota", "cpus", "mem_gb"} and m["machine"]["cpus"] >= 1
    assert m["created"].endswith("Z") and m["python"].count(".") == 2
    assert m["numpy"] and m["torch"]                                   # read without importing torch
    json.dumps(m)                                                      # plain JSON


def test_engine_record_from_the_checkout_files(cfg, tmp_path):
    e = MF.collect(cfg)["engine"]
    _, build = fake_engine_root(tmp_path / "again")
    assert set(e) == {"repo", "ref", "dirty", "binary_sha256", "build", "cards_lock", "census"}
    assert (e["repo"], e["ref"], e["dirty"], e["census"]) == (build["repo"], build["ref"], False, False)
    assert e["binary_sha256"] == hashlib.sha256(b"\x7fELF fake driver").hexdigest()   # the binary as it is now
    assert e["build"]["binary_sha256"] == "stale" and e["cards_lock"]["files"] == 34074


def test_engine_record_degrades(tmp_path):
    root, build = fake_engine_root(tmp_path, lock=False)
    e = MF.engine_version(root)
    assert e["cards_lock"] == build["cards_lock"]                      # from the build record when .cards is gone
    e = MF.engine_version(tmp_path / "nothing", census=True)
    assert e == {"repo": None, "ref": None, "dirty": None, "binary_sha256": None, "build": None, "cards_lock": None,
                 "census": True}


def test_belief_corpus_is_recorded(cfg, tmp_path):
    from draftzero.oppdeck.belief import corpus_hash
    b = MF.collect(cfg, gorge=FakeGorge())["belief"]
    assert set(b) == {"decks", "pool", "split", "corpus_hash"} and b["split"] == "train"
    assert b["corpus_hash"] == corpus_hash(decks="assets/sample/decks", pool="assets/sample/pool.tsv", split="train")
    off = C.from_dict({**C.to_dict(cfg), "search": {"closedlist": False}})
    assert MF.collect(off, gorge=FakeGorge())["belief"] is None
    gone = C.from_dict({**C.to_dict(cfg), "search": {"belief": {"pool": str(tmp_path / "none.tsv")}}})
    b = MF.collect(gone, gorge=FakeGorge())["belief"]
    assert b["corpus_hash"] is None and "FileNotFoundError" in b["error"]


def test_a_failing_gorge_is_recorded_not_raised(cfg):
    class Broken:
        def version(self):
            raise FileNotFoundError("bin/dzgorge")
    assert MF.collect(cfg, gorge=Broken())["engine"] == {"error": "FileNotFoundError: bin/dzgorge"}


def test_without_a_config_it_is_the_code_and_machine_part():
    m = MF.collect(None)
    assert set(m) == KEYS and m["run"] is None and m["config"] is None and m["config_hash"] is None
    assert m["engine"]["census"] is True and m["belief"] is None


def test_no_git_falls_back_to_the_git_files(tmp_path, monkeypatch, cfg):
    repo = tmp_path / "repo"
    (repo / ".git/refs/heads").mkdir(parents=True)
    (repo / ".git/HEAD").write_text("ref: refs/heads/main\n")
    (repo / ".git/refs/heads/main").write_text("a" * 40 + "\n")
    (repo / "engine/gorge").mkdir(parents=True)
    (repo / "engine/gorge/GORGE_REF").write_text("b" * 40 + "\n")

    def no_git(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(MF.subprocess, "run", no_git)
    m = MF.collect(cfg, gorge=FakeGorge(), repo=repo)
    assert m["draftzero"] == {"version": __version__, "commit": "a" * 40, "branch": "main", "dirty": None}
    assert m["engine_pin"] == {"repo": None, "ref": "b" * 40}

    (repo / ".git/refs/heads/main").unlink()
    (repo / ".git/packed-refs").write_text("# pack-refs\n" + "c" * 40 + " refs/heads/main\n")
    assert MF.git_state(repo) == {"commit": "c" * 40, "branch": "main", "dirty": None}
    (repo / ".git/HEAD").write_text("d" * 40)
    assert MF.git_state(repo) == {"commit": "d" * 40, "branch": None, "dirty": None}
    assert MF.git_state(tmp_path / "not-a-repo") == {"commit": None, "branch": None, "dirty": None}
    assert MF.git_state(None) == {"commit": None, "branch": None, "dirty": None}


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_git_state_of_a_real_repo(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q", "-b", "work"], check=True)
    (repo / "a.txt").write_text("1")
    subprocess.run([*git, "add", "a.txt"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "one"], check=True)
    head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert MF.git_state(repo) == {"commit": head, "branch": "work", "dirty": False}
    (repo / "a.txt").write_text("2")
    assert MF.git_state(repo)["dirty"] is True
    subprocess.run([*git, "checkout", "-q", "--detach"], check=True)
    assert MF.git_state(repo)["branch"] is None


def test_write_and_read(tmp_path, cfg):
    m = MF.collect(cfg, gorge=FakeGorge())
    MF.write(tmp_path / "runs/m1", m)
    assert json.loads((tmp_path / "runs/m1/manifest.json").read_text()) == m
    assert MF.read(tmp_path / "runs/m1") == m and MF.read(tmp_path / "elsewhere") is None

    class FakeStore:
        def __init__(self):
            self.files = {}

        def write_json(self, path, obj, message=""):
            self.files[path] = json.loads(json.dumps(obj))

        def read_json(self, path):
            return self.files.get(path)
    st = FakeStore()
    MF.write(st, m)
    assert st.files == {"manifest.json": m} and MF.read(st) == m


def test_cli_provenance_prints_a_manifest(capsys):
    from draftzero.cli import main
    assert main(["provenance"]) == 0
    m = json.loads(capsys.readouterr().out)
    assert m["format"] == MF.FORMAT and m["config"] is None
