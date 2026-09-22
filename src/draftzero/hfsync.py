"""hfsync.py — push checkpoints to a Hugging Face repo so a pod can be destroyed safely.

A network volume outlives its pod; a container disk does not. On the cheap multi-GPU pods
that have no volume, the only durable copy of the weights is one pushed off the machine.
This is that push. It is deliberately the thing the watchdog VERIFIES before it is allowed
to run --on-complete, so "terminate the pod" can never precede "the weights are safe".

Credentials come from the environment (HF_TOKEN, HF_REPO), loaded by secrets.load() from a
600 file on the pod's container disk -- never the volume, never the repo, never a log.

Uploads are best-effort per file and never raise: a failed push must leave the run running,
not kill it. The caller decides what a failure means for termination.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# What to push, in priority order. Weights first: if only one file makes it, make it a model.
PATTERNS = ("*.pt.gz", "run.json", "metrics.jsonl", "games.jsonl", "STATUS.json",
            "deck_records.tsv", "alerts.jsonl")


def configured() -> bool:
    return bool(os.environ.get("HF_TOKEN") and os.environ.get("HF_REPO"))


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ["HF_TOKEN"]), os.environ["HF_REPO"]


def push(run_dir: Path, models_dir: Path, gen: Optional[int] = None) -> tuple[bool, str]:
    """Upload checkpoints and run artifacts. Returns (ok, detail). Never raises.

    Files land under a prefix so several runs can share one repo:
        <run_name>/models/...   <run_name>/metrics.jsonl   etc.
    """
    if not configured():
        return False, "HF_TOKEN / HF_REPO not set"
    try:
        api, repo = _api()
    except Exception as e:  # noqa: BLE001 - import or auth shape varies
        return False, f"hf init failed: {e}"

    run_dir, models_dir = Path(run_dir), Path(models_dir)
    prefix = run_dir.name
    label = f"gen {gen}" if gen is not None else "sync"

    # (local path, path in repo) pairs
    items: list[tuple[Path, str]] = []
    if models_dir.exists():
        for f in models_dir.rglob("*.pt.gz"):
            items.append((f, f"{prefix}/models/{f.relative_to(models_dir)}"))
    for pat in PATTERNS:
        if pat == "*.pt.gz":
            continue
        for f in run_dir.glob(pat):
            items.append((f, f"{prefix}/{f.name}"))

    if not items:
        return False, "nothing to push yet"

    pushed = 0
    errs: list[str] = []
    for local, remote in items:
        try:
            api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                            repo_id=repo, repo_type="model",
                            commit_message=f"draftzero {label}: {remote}")
            pushed += 1
        except Exception as e:  # noqa: BLE001
            errs.append(f"{remote}: {str(e)[:80]}")

    ok = pushed > 0 and not errs
    detail = f"pushed {pushed}/{len(items)} to {repo}/{prefix}"
    if errs:
        detail += f" | {len(errs)} failed: {errs[0]}"
    return ok, detail


def has_checkpoint(run_name: str) -> tuple[bool, str]:
    """Confirm at least one checkpoint is actually in the repo. The remote analogue of
    verify(): a terminate must not fire on a push we only THINK happened."""
    if not configured():
        return False, "HF not configured"
    try:
        api, repo = _api()
        files = api.list_repo_files(repo_id=repo, repo_type="model")
    except Exception as e:  # noqa: BLE001
        return False, f"list failed: {e}"
    ckpts = [f for f in files if f.startswith(f"{run_name}/models/") and f.endswith(".pt.gz")]
    if not ckpts:
        return False, f"no checkpoints under {run_name}/ in {repo}"
    return True, f"{len(ckpts)} checkpoints in {repo}/{run_name}"
