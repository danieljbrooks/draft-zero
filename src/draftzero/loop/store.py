"""The shared store (contracts §4): the only thing the roles share.

    open_store("runs/smoke")                       a local folder: one machine (and the tests)
    open_store("hf://buckets/danbrooks/dz-e1")     a private Hugging Face Storage Bucket: many machines
    open_store("runpodvol://<vol>@EUR-IS-1/runs/e1/store")   a RunPod network volume: its mount on a pod, else S3

Paths are store-relative and use "/" ("control.json", "shards/pod1/000003/meta.json"). A prefix that ends in "/"
is a folder; `list` returns every file under it, sorted. Writes are whole files: a reader never sees half of one.
A put of several files is one call (one Hub API call on a bucket), so an actor uploads a shard, its heartbeat and
its metrics together. It is not atomic across files: writers put the file that marks a set complete (a shard's
meta.json, a version's latest.json) last, and readers treat a set without it as not there yet.

A bucket is mutable, unversioned storage (no git commits, so no 128-commits-an-hour cap); every call still counts
against the account's Hub API limit (1,000 per 5 minutes on a free account), so roles poll at most every few
seconds and read small files only when they need them. Token: the `token` argument, else $DZ_STORE_TOKEN (a
fine-grained token limited to the run's bucket), else $HF_TOKEN, else the machine's `hf auth login`.

An ssh:// store is not opened here: a machine without a token writes to a local folder and a relay
(draftzero.loop.relay) on a machine with the token moves files between that folder and the bucket.

A folder can also be shared: a RunPod network volume mounted on several pods (MooseFS, docs/035). Every file has one
writer and is renamed into place whole, so readers see the old file or the new one; a reader whose cached directory
entry lags a rename by up to ~1 s can find the file gone between looking and opening it, so reads try again
(VANISH_TRIES) before they say it is not there.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

VANISH_TRIES = 3             # a file seen but gone when opened (a shared folder's cached directory entry): look again
VANISH_PAUSE_S = 0.3


class Store:
    """The interface every role uses."""

    remote = False          # True: every call is a network request (roles poll less often)

    def put_many(self, files: dict[str, Path | bytes], message: str = "") -> None:
        """Write every file (local path or bytes) at its store path, in the dict's order."""
        raise NotImplementedError

    def get(self, path: str, dest: Path) -> Path | None:
        """Copy `path` to `dest` (None if it doesn't exist)."""
        raise NotImplementedError

    def list(self, prefix: str) -> list[str]:
        """Every file under `prefix` (a folder ending in "/", or one file's path), sorted."""
        raise NotImplementedError

    def delete(self, paths: list[str], message: str = "") -> None:
        raise NotImplementedError

    def read_bytes(self, path: str) -> bytes | None:
        raise NotImplementedError

    def ensure(self) -> None:
        """Create the store if it doesn't exist yet (the controller, at the start of a run)."""

    # conveniences
    def put(self, path: str, src: Path | bytes, message: str = "") -> None:
        self.put_many({path: src}, message or f"put {path}")

    def read_json(self, path: str):
        b = self.read_bytes(path)
        return None if b is None else json.loads(b.decode())

    def write_json(self, path: str, obj, message: str = "") -> None:
        self.put(path, dumps(obj), message or f"update {path}")

    def exists(self, path: str) -> bool:
        return self.read_bytes(path) is not None

    def describe(self) -> str:
        return type(self).__name__


def dumps(obj) -> bytes:
    """The store's JSON: indented, sorted keys, UTF-8."""
    return json.dumps(obj, indent=1, sort_keys=True, default=str).encode()


def _under(path: str, prefix: str) -> bool:
    """`path` is `prefix` itself, or under it: "shards/a" holds shards/a/x but not shards/ab/x."""
    if not prefix:
        return True
    if prefix.endswith("/"):
        return path.startswith(prefix)
    return path == prefix or path.startswith(prefix + "/")


class LocalStore(Store):
    """A folder: single-machine mode, a relay's local side, and the tests."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError(f"{path} is outside the store")
        return p

    def put_many(self, files, message=""):
        """Each file goes to a temp file beside its target, is flushed to disk (fsync: on a network filesystem the
        rename must not reach the other machines before the data) and renamed into place."""
        for path, src in files.items():
            dst = self._p(path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=".tmp-")
            try:
                with os.fdopen(fd, "wb") as f:
                    if isinstance(src, (bytes, bytearray)):
                        f.write(src)
                    else:
                        with open(src, "rb") as s:
                            shutil.copyfileobj(s, f, 1 << 20)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, dst)
            except BaseException:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise

    def get(self, path, dest):
        src = self._p(path)
        dest = Path(dest)
        for _ in range(VANISH_TRIES):
            if not src.is_file():
                return None
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(src, dest)
                return dest
            except FileNotFoundError:            # replaced or removed between the look and the open
                time.sleep(VANISH_PAUSE_S)
        return None

    def list(self, prefix):
        """Every file under prefix, except dot-files and anything in a dot-folder: this store's own `.tmp-*`
        files, rsync's temporary `.name.XXXXXX` files while a relay pulls, and the relay's `.relay-tmp/`.
        os.walk, not rglob: no stat per file, which on a network filesystem is a round trip each."""
        base = self._p(prefix) if prefix else self.root
        if base.is_file():
            return [prefix]
        if not base.is_dir():
            return []
        out = []
        for d, dirs, names in os.walk(base):
            dirs[:] = [x for x in dirs if not x.startswith(".")]
            rel = Path(d).relative_to(self.root)
            if any(part.startswith(".") for part in rel.parts):
                continue
            out += [(rel / n).as_posix() for n in names if not n.startswith(".")]
        return sorted(out)

    def delete(self, paths, message=""):
        for path in paths:
            p = self._p(path)
            if p.exists():
                p.unlink(missing_ok=True)

    def read_bytes(self, path):
        p = self._p(path)
        for _ in range(VANISH_TRIES):
            if not p.is_file():
                return None
            try:
                return p.read_bytes()
            except FileNotFoundError:            # replaced or removed between the look and the open
                time.sleep(VANISH_PAUSE_S)
        return None

    def exists(self, path):
        return self._p(path).is_file()

    def ensure(self):
        self.root.mkdir(parents=True, exist_ok=True)

    def describe(self):
        return f"local folder {self.root}"


def _token(token: str | None) -> str | None:
    return token or os.environ.get("DZ_STORE_TOKEN") or os.environ.get("HF_TOKEN") or None


class HFBucketStore(Store):
    """A Hugging Face Storage Bucket ("owner/name"), through huggingface_hub's bucket API (2.2):
    batch_bucket_files (put_many, delete), list_bucket_tree (list), download_bucket_files (get),
    get_bucket_paths_info (exists) and create_bucket (ensure: private, exist_ok).

    `api` is anything with those HfApi methods (the tests pass a fake). Transient errors are retried `retries`
    times with a growing pause; a missing file is not an error (get and read_bytes return None).
    """

    remote = True

    def __init__(self, bucket: str, token: str | None = None, *, api=None, retries: int = 3, backoff_s: float = 2.0):
        if bucket.count("/") != 1 or not all(bucket.split("/")):
            raise ValueError(f"a bucket id is owner/name, not {bucket!r}")
        self.bucket = bucket
        self.token = _token(token)
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi(token=self.token)
        self.api = api
        self.retries, self.backoff_s = retries, backoff_s

    def _call(self, fn, *args, **kw):
        for attempt in range(self.retries + 1):
            try:
                return fn(*args, **kw)
            except Exception as e:                       # noqa: BLE001 - the Hub raises many kinds
                if _missing(e) or attempt == self.retries or not _transient(e):
                    raise
                time.sleep(self.backoff_s * (attempt + 1))

    def put_many(self, files, message=""):
        add = [(bytes(src) if isinstance(src, (bytes, bytearray)) else str(src), path) for path, src in files.items()]
        if add:
            self._call(self.api.batch_bucket_files, self.bucket, add=add)

    def get(self, path, dest):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".tmp-")
        os.close(fd)
        try:
            try:
                self._call(self.api.download_bucket_files, self.bucket, [(path, tmp)], raise_on_missing_files=True)
            except Exception as e:                       # noqa: BLE001
                if _missing(e):
                    return None
                raise
            os.replace(tmp, dest)
            return dest
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def list(self, prefix):
        items = self._call(lambda: list(self.api.list_bucket_tree(self.bucket, prefix=prefix or None, recursive=True)))
        return sorted(i.path for i in items if getattr(i, "type", "file") == "file" and _under(i.path, prefix))

    def delete(self, paths, message=""):
        paths = list(paths)
        if paths:
            self._call(self.api.batch_bucket_files, self.bucket, delete=paths)

    def read_bytes(self, path):
        with tempfile.TemporaryDirectory() as d:
            p = self.get(path, Path(d) / "f")
            return None if p is None else p.read_bytes()

    def exists(self, path):
        infos = self._call(lambda: list(self.api.get_bucket_paths_info(self.bucket, [path])))
        return any(i.path == path for i in infos)

    def ensure(self):
        self._call(self.api.create_bucket, self.bucket, private=True, exist_ok=True)

    def describe(self):
        return f"Hugging Face bucket {self.bucket}"


def _missing(e: Exception) -> bool:
    return type(e).__name__ in ("EntryNotFoundError", "RemoteEntryNotFoundError", "FileNotFoundError")


def _transient(e: Exception) -> bool:
    """Network trouble and the Hub's 408/429/5xx are worth a retry; any other HTTP error (401, 403, a missing bucket,
    400, 422) is not. The status decides first: the Hub's HfHubHTTPError subclasses OSError (through
    httpx2.HTTPError), so an OSError check alone would retry every HTTP error."""
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status is not None:
        return status in (408, 429, 500, 502, 503, 504)
    for name in ("httpx2", "httpx"):                     # huggingface_hub's HTTP client (httpx2 from 2.x)
        try:
            client = __import__(name)
        except ImportError:
            continue
        if isinstance(e, client.TransportError):         # connect, read, write and pool errors and timeouts
            return True
        if isinstance(e, client.HTTPError):              # any other HTTP-level error without a status
            return False
    return isinstance(e, (ConnectionError, TimeoutError, OSError)) and not isinstance(e, FileNotFoundError)


def open_store(spec: str | Path, *, token: str | None = None) -> Store:
    """A store from its spec: a folder path, hf://buckets/<owner>/<name>, or runpodvol://<volume>@<DC>/<path> (a
    folder on a RunPod network volume: this machine's mount of it, else RunPod's S3 API; draftzero.loop.s3)."""
    s = str(spec)
    if s.startswith("runpodvol://"):
        from draftzero.loop.s3 import open_volume_store
        return open_volume_store(s)
    if s.startswith("hf://"):
        rest = s[len("hf://"):].strip("/")
        if not rest.startswith("buckets/"):
            raise ValueError(f"{s}: a run store on Hugging Face is a Storage Bucket, hf://buckets/<owner>/<name> "
                             "(a dataset or model repo commits on every write; publish finished runs there instead)")
        return HFBucketStore(rest[len("buckets/"):], token=token)
    if "://" in s:
        raise ValueError(f"{s}: an ssh:// (or other remote) store is reached through a relay (draftzero.loop.relay): "
                         "run this role against a local folder and the relay on a machine that holds the token")
    return LocalStore(s)
