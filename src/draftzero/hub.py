"""hub.py — Hugging Face model files by URI.

    resolve("runs/x/model.pt")                                     a local path, as is
    resolve("hf://danbrooks/draftzero-v0.1.0/player/model.pt")    downloads (and caches) that file of a model repo
    resolve("hf://danbrooks/draftzero-e1@v0042/model.dzgw")        ...at a revision: a tag, branch or commit
    resolve("hf://datasets/danbrooks/draftzero-games/x.jsonl")      a dataset repo's file (spaces/ likewise)
    resolve("hf://danbrooks/draftzero-e1@v0042")                   no path: the whole repo at that revision (a folder)

A revision holding '/' is percent-encoded (refs%2Fpr%2F1). Private repos need HF_TOKEN or `hf auth login`.
hf://buckets/... names a run store, not a model file: open it with draftzero.loop.store.open_store.
This is gameplay/imitation_net.resolve (v0.1.0) plus revisions, repo types and whole-repo snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

PREFIX = "hf://"
_TYPES = {"datasets": "dataset", "spaces": "space"}


@dataclass(frozen=True)
class HubRef:
    repo_id: str                  # owner/repo
    path: str | None              # the file in the repo; None = the whole repo
    revision: str | None = None
    repo_type: str = "model"      # model | dataset | space


def is_hub(uri: str | Path) -> bool:
    return str(uri).startswith(PREFIX)


def parse(uri: str) -> HubRef:
    """hf://[datasets/|spaces/]owner/repo[@rev][/path] -> HubRef; ValueError if it is not one."""
    s = str(uri)
    if not s.startswith(PREFIX):
        raise ValueError(f"{s}: not an hf:// URI")
    parts = s[len(PREFIX):].split("/")
    repo_type = "model"
    if parts and parts[0] in _TYPES:
        repo_type = _TYPES[parts.pop(0)]
    if parts and parts[0] == "buckets":
        raise ValueError(f"{s}: a bucket is a run store, not a model file (draftzero.loop.store.open_store)")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"{s}: want hf://<owner>/<repo>[@<revision>][/<path in the repo>]")
    owner, repo, rest = parts[0], parts[1], parts[2:]
    revision = None
    if "@" in repo:
        repo, revision = repo.split("@", 1)
        revision = unquote(revision) or None
        if not repo:
            raise ValueError(f"{s}: empty repo name")
    path = "/".join(rest).strip("/") or None
    return HubRef(f"{owner}/{repo}", path, revision, repo_type)


def resolve(uri: str | Path, *, cache_dir: str | Path | None = None, token: str | None = None) -> Path:
    """A local path for `uri`: local paths as they are; hf:// files downloaded into the Hugging Face cache (a
    repeat call is a cache hit); an hf:// repo without a path as a snapshot folder."""
    if not is_hub(uri):
        return Path(uri).expanduser()
    ref = parse(str(uri))
    kw = {"repo_type": ref.repo_type, "revision": ref.revision, "cache_dir": cache_dir, "token": token}
    if ref.path is None:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(ref.repo_id, **kw))
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(ref.repo_id, ref.path, **kw))
