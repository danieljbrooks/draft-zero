#!/bin/bash
# bootstrap.sh — make a worker able to run DraftZero. Idempotent; safe to re-run.
# Works on any Debian-family worker: a rented pod, a LAN server, or this laptop.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== draftzero bootstrap in $(pwd)"

need_sudo() { [ "$(id -u)" -eq 0 ] && echo "" || echo "sudo"; }
SUDO="$(need_sudo)"

if ! command -v java >/dev/null 2>&1; then
  echo "== installing jdk 21"
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq openjdk-21-jdk-headless
fi
command -v rsync >/dev/null 2>&1 || $SUDO apt-get install -y -qq rsync
java -version 2>&1 | head -1

echo "== python deps"
# Some images ship an externally-managed (PEP 668) python whose debian-installed blinker
# has no RECORD file, so pip cannot replace it when flask wants a newer one. Without both
# flags the whole install transaction aborts.
PIPFLAGS="--root-user-action=ignore"
python -c "import sysconfig,sys; sys.exit(0 if sysconfig.get_config_var('EXT_SUFFIX') else 0)" 2>/dev/null || true
if python -m pip install --help 2>/dev/null | grep -q break-system-packages; then
  PIPFLAGS="$PIPFLAGS --break-system-packages"
fi
python -m pip install -q $PIPFLAGS --ignore-installed blinker -e . || {
  echo "!! editable install failed; falling back to PYTHONPATH=src"; }

echo "== sanity"
python - <<'PY'
import importlib.util, sys
for m in ("torch", "magezero", "draftzero"):
    print(f"   {m:10} {'ok' if importlib.util.find_spec(m) else 'MISSING'}")
try:
    import torch
    print(f"   cuda={torch.cuda.is_available()} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
except Exception as e:
    print(f"   torch check skipped: {e}")
PY

echo "== capacity (what the container may ACTUALLY use, not what it advertises)"
python -c "
from magezero.resources import cpu_quota, mem_limit_gb
import psutil
print(f'   visible: {psutil.cpu_count()} cores / {psutil.virtual_memory().total/1024**3:.0f} GB')
print(f'   quota  : {cpu_quota()} cores / {mem_limit_gb() and round(mem_limit_gb())} GB')
" 2>/dev/null || echo "   (draftzero not importable yet)"

test -d xmage/lib || echo "!! xmage/ missing — link or fetch the fdn-generalist XMage build"
echo "== ready"
