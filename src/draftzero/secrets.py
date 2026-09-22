"""secrets.py — read credential files without executing them.

Sourcing a secrets file into a shell is a trap: if the file holds a bare value rather than
`export KEY=value`, the shell tries to RUN it and echoes the secret in the error. That
happened here with a Hugging Face token, which then had to be revoked. Parse, never source.

Accepts either shape, so a hand-pasted file works as well as a generated one:

    export HF_TOKEN=hf_xxx        # shell-style
    HF_TOKEN=hf_xxx               # plain key=value
    hf_xxx                        # a bare value, when `default_key` says what it is
"""
from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Optional


def load(path, default_key: Optional[str] = None, export: bool = True) -> dict:
    """Parse a credential file into a dict, optionally into os.environ.

    Never executes the file. Warns when the file is group- or world-readable, because a
    credential that anyone on the box can read is a credential you should assume is loose.
    """
    p = Path(path).expanduser()
    if not p.exists():
        return {}
    mode = p.stat().st_mode
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        print(f"WARNING: {p} is readable beyond its owner ({oct(mode & 0o777)}); chmod 600 it")

    out: dict[str, str] = {}
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip("'\"")
        elif default_key and default_key not in out:
            out[default_key] = line          # a bare value, e.g. a pasted token
    if export:
        os.environ.update(out)
    return out


def redact(value: str, keep: int = 4) -> str:
    """For logs. Never print a credential whole, not even one you are about to discard."""
    if not value:
        return "<empty>"
    return f"{value[:keep]}...<{len(value)} chars>"
