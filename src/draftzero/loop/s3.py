"""s3.py — a run's store on a RunPod network volume, reached through RunPod's S3-compatible API (docs/035 §6).

Pods mount the volume at /workspace and open the store there as a folder; a machine outside RunPod (d1) reaches the
same files through `https://s3api-<dc>.runpod.io/`, where the bucket is the volume id, the region is the data center
and an object key is a file's path on the volume. One spec names the store on both:

    open_store("runpodvol://<volume-id>@<DATA-CENTER>/<path on the volume>")
        on a pod that mounts the volume   -> LocalStore("<mount>/<path>")   (from /proc/self/mountinfo)
        anywhere else                     -> S3Store(<volume-id>, "<path>", region=<DATA-CENTER>)

Credentials: an S3 API key from the RunPod console (Settings -> S3 API Keys: an access key `user_...` and a secret
`rps_...`, separate from the API key), in /root/.runpod/s3_env as RUNPOD_S3_ACCESS_KEY_ID and
RUNPOD_S3_SECRET_ACCESS_KEY (parsed, never sourced or exported), else $DZ_S3_ACCESS_KEY_ID / $DZ_S3_SECRET_ACCESS_KEY.
The key reaches every volume on the account: it stays on the machine that needs it, travels only in a signed header
and never appears in an error. Pods need none.

What RunPod's S3 API supports (October 2026): GetObject, PutObject (under 500 MB), HeadObject, DeleteObject,
ListObjectsV2, HeadBucket. Not DeleteObjects (so a delete is one call per file) or CreateBucket (a volume is made with
`dz pod volume create`). Listings include folders (keys ending in "/") and dot-files, which the store skips, and
502s happen, which are retried with the other network errors, 408, 429 and 5xx. Requests are signed with AWS
Signature Version 4 (checked against AWS's published examples in the tests), path-style, with the payload's SHA-256.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional

from draftzero import secrets
from draftzero.loop.store import Store, _under

S3_ENV = "/root/.runpod/s3_env"
MAX_PUT = 500 * 1024 * 1024 - 1          # RunPod's PutObject limit (multipart above it; never needed for a store)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_UNRESERVED = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.~"

# (method, url, body, headers) -> (status, headers, body)
S3Http = Callable[[str, str, Optional[bytes], dict], tuple[int, dict, bytes]]


class S3Error(RuntimeError):
    pass


def endpoint_for(datacenter: str) -> str:
    return f"https://s3api-{datacenter.lower()}.runpod.io"


def _enc(s: str, keep_slash: bool) -> str:
    return urllib.parse.quote(s, safe=_UNRESERVED + ("/" if keep_slash else ""))


def sigv4_headers(method: str, url: str, headers: dict, payload_sha256: str, *, region: str, access_key: str,
                  secret_key: str, now: _dt.datetime, service: str = "s3") -> dict:
    """`headers` plus x-amz-date, x-amz-content-sha256 and the Authorization header of AWS Signature Version 4.
    Every header given is signed (with host). The URL's path and query must already be in their final encoding."""
    u = urllib.parse.urlsplit(url)
    amz = now.strftime("%Y%m%dT%H%M%SZ")
    day = amz[:8]
    out = {**headers, "x-amz-date": amz, "x-amz-content-sha256": payload_sha256}
    canon = {k.lower(): " ".join(str(v).split()) for k, v in out.items()}
    canon["host"] = u.netloc
    names = sorted(canon)
    query = sorted(urllib.parse.parse_qsl(u.query, keep_blank_values=True))
    request = "\n".join([
        method,
        u.path or "/",
        "&".join(f"{_enc(k, False)}={_enc(v, False)}" for k, v in query),
        "".join(f"{n}:{canon[n]}\n" for n in names),
        ";".join(names),
        payload_sha256,
    ])
    scope = f"{day}/{region}/{service}/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz, scope, hashlib.sha256(request.encode()).hexdigest()])
    key = ("AWS4" + secret_key).encode()
    for part in (day, region, service, "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    sig = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    out["Authorization"] = (f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, SignedHeaders={';'.join(names)}, "
                            f"Signature={sig}")
    return out


def credentials(env_file: str | None = None) -> tuple[str, str]:
    """(access key, secret) for RunPod's S3 API. Never print them."""
    try:
        vals = secrets.load(env_file or S3_ENV, export=False)
    except OSError:                                   # e.g. /root/.runpod for a user who isn't root: not there
        vals = {}
    ak = vals.get("RUNPOD_S3_ACCESS_KEY_ID") or os.environ.get("DZ_S3_ACCESS_KEY_ID")
    sk = vals.get("RUNPOD_S3_SECRET_ACCESS_KEY") or os.environ.get("DZ_S3_SECRET_ACCESS_KEY")
    if not (ak and sk):
        raise S3Error(f"no RunPod S3 key: put RUNPOD_S3_ACCESS_KEY_ID=... and RUNPOD_S3_SECRET_ACCESS_KEY=... in "
                      f"{env_file or S3_ENV} (chmod 600); create the key in the RunPod console (Settings -> S3 API Keys)")
    return ak, sk


def urllib_http(method: str, url: str, body: Optional[bytes], headers: dict) -> tuple[int, dict, bytes]:
    r = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=120) as f:
            return f.status, dict(f.headers), f.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read()


def parse_spec(spec: str) -> tuple[str, str, str]:
    """runpodvol://<volume-id>@<DATA-CENTER>/<path> -> (volume id, data center, path)."""
    rest = spec[len("runpodvol://"):] if spec.startswith("runpodvol://") else ""
    head, _, path = rest.partition("/")
    vol, _, dc = head.partition("@")
    path = path.strip("/")
    if not (vol and dc and path) or any(c in vol + dc for c in " :"):
        raise ValueError(f"{spec}: runpodvol://<volume-id>@<DATA-CENTER>/<path on the volume>, e.g. "
                         "runpodvol://5njr2ebemc@EUR-IS-1/runs/e1/store")
    return vol, dc, path


def volume_mount(volume_id: str, mountinfo: str = "/proc/self/mountinfo") -> Optional[Path]:
    """Where this machine mounts network volume `volume_id` (a RunPod pod: /workspace), else None. RunPod mounts
    `mfs#<dc>.runpod.net:9421[/networkvolumes/<id>]`, so the mount's root field names the volume."""
    try:
        lines = Path(mountinfo).read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        f = line.split()
        if len(f) > 4 and f[3].rstrip("/") == f"/networkvolumes/{volume_id}":
            return Path(f[4].replace("\\040", " "))
    return None


class S3Store(Store):
    """A folder on a network volume through RunPod's S3 API. `http` is pluggable (the tests pass a fake server)."""

    remote = True

    def __init__(self, bucket: str, prefix: str = "", *, region: str, endpoint: str | None = None,
                 access_key: str | None = None, secret_key: str | None = None, http: S3Http | None = None,
                 retries: int = 4, backoff_s: float = 2.0, clock: Callable[[], _dt.datetime] | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        if not bucket or "/" in bucket:
            raise ValueError(f"bucket {bucket!r}: a volume id")
        self.bucket, self.prefix, self.region = bucket, prefix.strip("/"), region
        self.endpoint = (endpoint or endpoint_for(region)).rstrip("/")
        if access_key is None or secret_key is None:
            access_key, secret_key = credentials()
        self._ak, self._sk = access_key, secret_key
        self.http = http or urllib_http
        self.retries, self.backoff_s, self.sleep = retries, backoff_s, sleep
        self.clock = clock or (lambda: _dt.datetime.now(_dt.timezone.utc))

    def __repr__(self) -> str:
        return f"<S3Store {self.bucket}/{self.prefix} at {self.endpoint}>"

    # ---------------------------------------------------------------------------- requests
    def _key(self, path: str) -> str:
        if path.startswith("/") or ".." in path.split("/"):
            raise ValueError(f"{path} is outside the store")
        return f"{self.prefix}/{path}" if self.prefix else path

    def _url(self, key: str = "", query: dict | None = None) -> str:
        url = f"{self.endpoint}/{_enc(self.bucket, False)}" + (f"/{_enc(key, True)}" if key else "")
        if query:
            url += "?" + "&".join(f"{_enc(k, False)}={_enc(str(v), False)}" for k, v in sorted(query.items()))
        return url

    def _scrub(self, text: str) -> str:
        for s in (self._sk, self._ak):
            if s and len(s) >= 8:                       # a real key; a short one would scrub ordinary text
                text = text.replace(s, secrets.redact(s))
        return text

    def _request(self, method: str, key: str = "", *, query: dict | None = None, body: bytes | None = None,
                 ok_missing: bool = False) -> tuple[int, dict, bytes]:
        url = self._url(key, query)
        sha = hashlib.sha256(body).hexdigest() if body is not None else EMPTY_SHA256
        for attempt in range(self.retries + 1):
            headers = sigv4_headers(method, url, {}, sha, region=self.region, access_key=self._ak,
                                    secret_key=self._sk, now=self.clock())
            try:
                status, hdrs, data = self.http(method, url, body, headers)
            except (OSError, TimeoutError) as e:          # connection trouble: worth a retry
                if attempt == self.retries:
                    raise S3Error(f"{method} {url}: {self._scrub(str(e))[:300]}") from None
                self.sleep(self.backoff_s * (attempt + 1))
                continue
            if status == 404 and ok_missing:
                return status, hdrs, data
            if status < 300:
                return status, hdrs, data
            if status in (408, 429) or status >= 500:
                if attempt < self.retries:
                    self.sleep(self.backoff_s * (attempt + 1))
                    continue
            raise S3Error(f"{method} {url}: HTTP {status}: {self._scrub(data.decode(errors='replace'))[:400]}")
        raise S3Error(f"{method} {url}: no answer")              # not reached

    # ---------------------------------------------------------------------------- the Store
    def put_many(self, files, message=""):
        for path, src in files.items():
            body = bytes(src) if isinstance(src, (bytes, bytearray)) else Path(src).read_bytes()
            if len(body) > MAX_PUT:
                raise S3Error(f"{path}: {len(body)} bytes, above RunPod's 500 MB PutObject limit")
            self._request("PUT", self._key(path), body=body)

    def read_bytes(self, path):
        status, _, data = self._request("GET", self._key(path), ok_missing=True)
        return None if status == 404 else data

    def get(self, path, dest):
        data = self.read_bytes(path)
        if data is None:
            return None
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".tmp-")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)
        return dest

    def exists(self, path):
        status, _, _ = self._request("HEAD", self._key(path), ok_missing=True)
        return status != 404

    def list(self, prefix):
        """Every file under prefix: ListObjectsV2 pages, without folder keys and dot-files (a pod's .tmp-* files,
        RunPod's .s3compat_uploads/)."""
        base = f"{self.prefix}/" if self.prefix else ""
        out, token, pages = [], None, 0
        while True:
            q = {"list-type": "2", "prefix": base + prefix, "max-keys": "1000"}
            if token:
                q["continuation-token"] = token
            _, _, data = self._request("GET", query=q)
            root = ET.fromstring(data)
            ns = root.tag[:root.tag.index("}") + 1] if root.tag.startswith("{") else ""
            for c in root.iter(f"{ns}Contents"):
                k = c.findtext(f"{ns}Key") or ""
                if not k.startswith(base) or k.endswith("/"):
                    continue
                rel = k[len(base):]
                if rel and not any(p.startswith(".") for p in rel.split("/")) and _under(rel, prefix):
                    out.append(rel)
            nxt = root.findtext(f"{ns}NextContinuationToken")
            pages += 1
            if (root.findtext(f"{ns}IsTruncated") or "").lower() != "true" or not nxt:
                break
            if nxt == token or pages > 100_000:
                raise S3Error(f"list {prefix}: the listing repeats its continuation token")
            token = nxt
        return sorted(set(out))

    def delete(self, paths, message=""):
        for path in paths:
            self._request("DELETE", self._key(path), ok_missing=True)

    def ensure(self):
        status, _, _ = self._request("HEAD", ok_missing=True)
        if status == 404:
            raise S3Error(f"no network volume {self.bucket} at {self.endpoint}: create it with `dz pod volume create`")

    def describe(self):
        return f"network volume {self.bucket}/{self.prefix} over S3 ({self.region})"


def open_volume_store(spec: str, *, mountinfo: str = "/proc/self/mountinfo", **kw) -> Store:
    """runpodvol://<volume-id>@<DC>/<path>: the folder on this machine's mount of the volume, else S3Store."""
    from draftzero.loop.store import LocalStore
    vol, dc, path = parse_spec(spec)
    mount = volume_mount(vol, mountinfo)
    if mount is not None:
        return LocalStore(mount / path)
    return S3Store(vol, path, region=dc, **kw)
