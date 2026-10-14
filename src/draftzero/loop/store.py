"""The shared store (docs/021 §3.2): the only thing the machines share.

    open_store("runs/selfplay/smoke")                                    a local folder: one machine
    open_store("hf://danbrooks/draftzero-selfplay-games")                a Hugging Face dataset
    open_store("hf://danbrooks/draftzero-selfplay-games?weights=danbrooks/draftzero-selfplay-weights")
                                                                         ...with the weights in a private repo

Paths are store-relative ("control.json", "games/v0003/r1/....jsonl.gz", "weights/v0003.pt.gz"). In an HFStore,
paths under weights/ and learner/full go to the weights repo when one is given (docs/021: the weights stay
private for the first version); everything else goes to the dataset. A put of several files is one commit
(Hugging Face allows 128 an hour per repo on a free account), and every read of a small file goes through the
repo's latest commit, not a cached copy.

Tokens: HF_SELFPLAY_TOKEN, else HF_TOKEN, else the machine's login. A worker's token needs write access to the
dataset and read access to the weights repo.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path


class Store:
    """The interface every role uses. Writes are whole files: a reader never sees half of one."""

    def put_many(self, files: dict[str, Path | bytes], message: str = "") -> None:
        raise NotImplementedError

    def get(self, path: str, dest: Path) -> Path | None:
        """Copy `path` to `dest` (None if it doesn't exist)."""
        raise NotImplementedError

    def list(self, prefix: str) -> list[str]:
        """Every file under `prefix`, sorted."""
        raise NotImplementedError

    def delete(self, paths: list[str], message: str = "") -> None:
        raise NotImplementedError

    def read_bytes(self, path: str) -> bytes | None:
        raise NotImplementedError

    # conveniences
    def put(self, path: str, src: Path | bytes, message: str = "") -> None:
        self.put_many({path: src}, message or f"put {path}")

    def read_json(self, path: str) -> dict | None:
        b = self.read_bytes(path)
        return None if b is None else json.loads(b.decode())

    def write_json(self, path: str, obj, message: str = "") -> None:
        self.put(path, json.dumps(obj, indent=1, sort_keys=True).encode(), message or f"update {path}")

    def exists(self, path: str) -> bool:
        return self.read_bytes(path) is not None

    def describe(self) -> str:
        return type(self).__name__


class LocalStore(Store):
    """A folder: single-machine mode, and the tests."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError(f"{path} is outside the store")
        return p

    def put_many(self, files, message=""):
        for path, src in files.items():
            dst = self._p(path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=".tmp-")
            os.close(fd)
            if isinstance(src, (bytes, bytearray)):
                Path(tmp).write_bytes(src)
            else:
                shutil.copyfile(src, tmp)
            os.replace(tmp, dst)

    def get(self, path, dest):
        src = self._p(path)
        if not src.exists():
            return None
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        return dest

    def list(self, prefix):
        base = self._p(prefix) if prefix else self.root
        if base.is_file():
            return [prefix]
        if not base.exists():
            return []
        return sorted(str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file() and not p.name.startswith(".tmp-"))

    def delete(self, paths, message=""):
        for path in paths:
            p = self._p(path)
            if p.exists():
                p.unlink()

    def read_bytes(self, path):
        p = self._p(path)
        return p.read_bytes() if p.exists() else None

    def describe(self):
        return f"local folder {self.root}"


class HFStore(Store):
    """A public Hugging Face dataset, plus an optional (private) model repo for the weights."""

    WEIGHT_PREFIXES = ("weights/", "learner/full")

    def __init__(self, dataset: str, weights_repo: str | None = None, token: str | None = None):
        from huggingface_hub import HfApi
        self.dataset = dataset
        self.weights_repo = weights_repo
        self.token = token or os.environ.get("HF_SELFPLAY_TOKEN") or os.environ.get("HF_TOKEN") or None
        self.api = HfApi(token=self.token)

    def _repo(self, path: str) -> tuple[str, str]:
        if self.weights_repo and path.startswith(self.WEIGHT_PREFIXES):
            return self.weights_repo, "model"
        return self.dataset, "dataset"

    def _sha(self, repo: str, kind: str) -> str:
        return self.api.repo_info(repo, repo_type=kind).sha

    def put_many(self, files, message=""):
        from huggingface_hub import CommitOperationAdd
        by: dict[tuple[str, str], list] = {}
        for path, src in files.items():
            obj = bytes(src) if isinstance(src, (bytes, bytearray)) else str(src)
            by.setdefault(self._repo(path), []).append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=obj))
        for (repo, kind), ops in by.items():
            self.api.create_commit(repo, operations=ops, repo_type=kind, commit_message=message or "selfplay")

    def get(self, path, dest):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
        repo, kind = self._repo(path)
        try:
            src = hf_hub_download(repo, path, repo_type=kind, revision=self._sha(repo, kind), token=self.token)
        except EntryNotFoundError:
            return None
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        return dest

    def list(self, prefix):
        repo, kind = self._repo(prefix)
        files = self.api.list_repo_files(repo, repo_type=kind, revision=self._sha(repo, kind))
        return sorted(f for f in files if f.startswith(prefix))

    def delete(self, paths, message=""):
        from huggingface_hub import CommitOperationDelete
        by: dict[tuple[str, str], list] = {}
        for path in paths:
            by.setdefault(self._repo(path), []).append(CommitOperationDelete(path_in_repo=path))
        for (repo, kind), ops in by.items():
            self.api.create_commit(repo, operations=ops, repo_type=kind, commit_message=message or "selfplay: delete")

    def read_bytes(self, path):
        with tempfile.TemporaryDirectory() as d:
            p = self.get(path, Path(d) / "f")
            return None if p is None else p.read_bytes()

    def squash(self) -> None:
        """Drop replaced files from both repos' history (docs/021 §3.2): run right after a new version."""
        for repo, kind in {(self.dataset, "dataset"), *([(self.weights_repo, "model")] if self.weights_repo else [])}:
            self.api.super_squash_history(repo, repo_type=kind)

    def describe(self):
        return f"Hugging Face dataset {self.dataset}" + (f", weights in {self.weights_repo}" if self.weights_repo else "")


def open_store(spec: str | Path) -> Store:
    s = str(spec)
    if s.startswith("hf://"):
        from urllib.parse import parse_qs, urlparse
        u = urlparse(s)
        q = parse_qs(u.query)
        return HFStore(f"{u.netloc}{u.path}".strip("/"), (q.get("weights") or [None])[0])
    return LocalStore(s)
