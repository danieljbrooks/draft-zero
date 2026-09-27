"""engine.py — DraftZero's adapter to the MageZero engine (Will's v0.2) and the XMage bundle.

MageZero's own runner is written for a MageZero checkout: it launches its scripts from the
repo-relative path src/magezero, and its train/test don't write metrics. DraftZero doesn't
fork MageZero to change that. It launches MageZero's server.py / train.py / test.py by their
installed path, unmodified, and does everything around them here:

- JVMs: several at once, each with its own temp dir, logging to stdout only.
- Servers: one per checkpoint in use. A past generation's checkpoint (gen5.pt.gz) is served
  by staging it as a model of its own, because v0.2's server only loads model.pt.gz.
- Train / test: their stdout goes to train.log / test.log, and the loss lines become
  metrics.jsonl rows (train_epoch, eval_prev_model), as exp #1's fork used to write.

The engine changes DraftZero needs are opt-in and small, on danieljbrooks/MageZero `draftzero`:
the policy width for a set-wide action vocabulary (MZ_ACTION_VOCAB, WillWroble/MageZero#7), a
CPU fallback for train/test, and the server's HTTP pool size (MZ_SERVER_THREADS).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

import magezero

from draftzero import metrics

PRIMARY_PORT = 50052
OPPONENT_PORT = 50053
TMP_DIR = Path(".mz_tmp")
# MageZero's scripts use flat imports (`from model import ...`), so they run as scripts from
# their own directory, never with -m.
MZ_SRC = Path(list(magezero.__path__)[0])
PYTHON = sys.executable
XMAGE_DIR = Path(os.environ.get("MZ_XMAGE_DIR", "xmage"))
JAVA = os.environ.get("MZ_JAVA", "java")
# stdout only, in the bundle's line format (metrics.LINE_RE parses it). The bundle's own
# log4j.properties also writes a rolling magezero.log in the working directory, which
# several JVMs sharing one xmage/ would fight over.
LOG4J = Path(__file__).resolve().parent / "assets" / "log4j.console.properties"


# ── inference servers ────────────────────────────────────────

def _serve_name(deck: str, version: int, checkpoint: Optional[str]) -> str:
    """The model name to serve. A named checkpoint is staged as models/<deck>.<ckpt>/ver<N>/
    model.pt.gz (a symlink), since v0.2's server loads only model.pt.gz."""
    if not checkpoint:
        return deck
    name = f"{deck}.{checkpoint}"
    src = Path("models") / deck / f"ver{version}" / f"{checkpoint}.pt.gz"
    if not src.exists():
        raise FileNotFoundError(f"no checkpoint {src}")
    dst = Path("models") / name / f"ver{version}" / "model.pt.gz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())
    return name


def start_server(deck: str, version: int, port: int, run_dir: Path,
                 checkpoint: Optional[str] = None, clients: int = 6) -> subprocess.Popen:
    """`clients`: game threads that will query this server. Each request holds one HTTP thread
    until its batch is evaluated, so the pool must cover them all or it caps the batch."""
    name = _serve_name(deck, version, checkpoint)
    label = f"{deck} v{version}" + (f" {checkpoint}" if checkpoint else "")
    print(f"[server] start {label} on :{port}")
    log_path = run_dir / f"server_{port}.log"
    log_file = open(log_path, "a")
    log_file.write(f"\n=== START {datetime.now().isoformat()} {label} ===\n")
    log_file.flush()
    cmd = [PYTHON, "-u", str(MZ_SRC / "server.py"), "--deck", name, "--version", str(version),
           "--port", str(port)]
    env = {**os.environ, "MZ_SERVER_THREADS": str(max(6, 2 * clients))}   # 2x: both players may ask
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    proc._dz_log_file = log_file
    deadline = time.time() + 600
    while time.time() < deadline:
        if proc.poll() is not None:
            log_file.close()
            raise RuntimeError(f"server exited before becoming ready (see {log_path})")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1)
            print(f"[server] ready on :{port}")
            return proc
        except Exception:
            time.sleep(0.5)
    proc.terminate()
    log_file.close()
    raise TimeoutError(f"server on :{port} did not come up within 600s")


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    if hasattr(proc, "_dz_log_file"):
        proc._dz_log_file.close()


# ── XMage JVMs ───────────────────────────────────────────────

def gc_flags(gc: str) -> list[str]:
    """zgc (exp #1), zgc-gen (generational ZGC: +10% in the exp #2 pilot, docs/006), or g1."""
    return {"zgc": ["-XX:+UseZGC"],
            "zgc-gen": ["-XX:+UseZGC", "-XX:+ZGenerational"],
            "g1": ["-XX:+UseG1GC"]}[gc]


def jvm_command(game_yml_path: str, heap: str = "24g", gc: str = "zgc",
                tmpdir: Optional[Path] = None) -> tuple[list[str], str]:
    """(argv, cwd) for one XMage JVM. A private java.io.tmpdir per JVM: concurrent JVMs
    unpacking jhdf5's native library into one shared temp dir killed one of them in the pilot."""
    cmd = [JAVA, f"-Dlog4j.configuration=file:{LOG4J}", "-Xms1g", f"-Xmx{heap}", *gc_flags(gc)]
    if tmpdir is not None:
        tmpdir.mkdir(parents=True, exist_ok=True)
        cmd.append(f"-Djava.io.tmpdir={tmpdir.resolve()}")
    cmd += ["--add-opens=java.base/java.lang=ALL-UNNAMED", "--enable-native-access=ALL-UNNAMED",
            "-jar", "lib/mage-magezero-1.4.58.jar", str(Path(game_yml_path).resolve())]
    return cmd, str(XMAGE_DIR)


def start_jvm(game_yml_path: str, log_path: Path, heap: str = "24g", gc: str = "zgc",
              tmpdir: Optional[Path] = None) -> subprocess.Popen:
    """Start one JVM in the background, stdout and stderr to log_path."""
    print(f"[jvm] launching with {game_yml_path}")
    cmd, cwd = jvm_command(game_yml_path, heap, gc, tmpdir)
    f = open(log_path, "a")
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    proc._dz_log_file = f
    return proc


def wait_jvm(proc: subprocess.Popen) -> int:
    rc = proc.wait()
    proc._dz_log_file.close()
    return rc


def launch_jvm(game_yml_path: str, log_path: Path, heap: str = "24g", gc: str = "zgc",
               tmpdir: Optional[Path] = None) -> None:
    rc = wait_jvm(start_jvm(game_yml_path, log_path, heap, gc, tmpdir))
    if rc != 0:
        raise subprocess.CalledProcessError(rc, "xmage jvm")


# ── train / test ─────────────────────────────────────────────

EPOCH_RE = re.compile(r"^Epoch (\d+)\s+(.*)$")
KV_RE = re.compile(r"(\w+)=(-?[\d.]+(?:[eE][-+]?\d+)?|nan|inf)")
VOCAB_RE = re.compile(r"^feature vocab: (\d+) kept ids.*?(\d+) rows from checkpoint, (\d+) added -> (\d+)")
ACC_RE = re.compile(r"^Test (\w+)_accuracy=([\d.]+)")


def _run_logged(cmd: list[str], log_path: Path, header: str, batch: Optional[int] = None) -> list[str]:
    """Run a MageZero script, appending its output to log_path; return this run's lines.
    `batch`: states per training/test batch (MZ_TRAIN_BATCH; MageZero's default is 512)."""
    env = {**os.environ, **({"MZ_TRAIN_BATCH": str(batch)} if batch else {})}
    with open(log_path, "a") as f:
        f.write(f"\n=== {header} {datetime.now().isoformat()} ===\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        f.write(proc.stdout)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout[-4000:])
    return proc.stdout.splitlines()


def _kv(s: str) -> dict:
    return {k: float(v) for k, v in KV_RE.findall(s)}


def parse_train_output(lines: list[str], deck: str, version: int, gen: int) -> list[dict]:
    rows, vocab = [], {}
    for l in lines:
        if (m := VOCAB_RE.match(l)):
            vocab = {"features_kept": int(m.group(1)), "embed_rows": int(m.group(4)),
                     "embed_rows_added": int(m.group(3))}
        elif (m := EPOCH_RE.match(l)):
            kv = _kv(m.group(2))
            rows.append({"kind": "train_epoch", "deck": deck, "version": version, "gen": gen,
                         "epoch": int(m.group(1)), **{f"train_{k}": v for k, v in kv.items()
                                                      if k != "decision_states"},
                         "decision_states": int(kv.get("decision_states", 0)), **vocab})
    return rows


def parse_test_output(lines: list[str], deck: str, version: int, gen: int) -> Optional[dict]:
    row: dict = {}
    for l in lines:
        if l.startswith("Validation loss:"):
            row.update(_kv(l))
        elif (m := ACC_RE.match(l)):
            row[f"{m.group(1)}_acc"] = float(m.group(2))
    if not row:
        return None
    row["decision_states"] = int(row.get("decision_states", 0))
    return {"kind": "eval_prev_model", "deck": deck, "version": version, "gen": gen, **row}


def run_train(deck: str, version: int, epochs: int, use_checkpoint: bool, run_dir: Path, gen: int,
              steps: Optional[int] = None, batch: Optional[int] = None, keep_as: Optional[str] = None) -> None:
    """Train, log per-epoch losses to metrics.jsonl, and keep this generation's checkpoint as
    gen<N>.pt.gz (the league and the evals play against those), or as `keep_as`."""
    cmd = [PYTHON, "-u", str(MZ_SRC / "train.py"), "--deck", deck, "--version", str(version),
           "--epochs", str(epochs)]
    if steps:
        cmd += ["--steps", str(steps)]
    if use_checkpoint:
        cmd.append("--checkpoint")
    lines = _run_logged(cmd, run_dir / "train.log", f"GEN {gen} TRAIN {deck}", batch)
    for row in parse_train_output(lines, deck, version, gen):
        metrics.append_jsonl(run_dir / "metrics.jsonl", row)
    models = Path("models") / deck / f"ver{version}"
    shutil.copyfile(models / "model.pt.gz", models / f"{keep_as or f'gen{gen}'}.pt.gz")


def run_test(deck: str, version: int, run_dir: Path, gen: int, batch: Optional[int] = None) -> None:
    """The previous generation's model on this generation's new games (before training on them)."""
    lines = _run_logged([PYTHON, "-u", str(MZ_SRC / "test.py"), "--deck", deck, "--version", str(version)],
                        run_dir / "test.log", f"GEN {gen} TEST {deck}", batch)
    row = parse_test_output(lines, deck, version, gen)
    if row:
        metrics.append_jsonl(run_dir / "metrics.jsonl", row)


def refresh_dashboard(run_dir: Path) -> None:
    try:
        from draftzero import report
        report.render(run_dir, run_dir / "dashboard.html")
    except Exception as e:
        print(f"[metrics] WARNING dashboard render failed: {e}")
