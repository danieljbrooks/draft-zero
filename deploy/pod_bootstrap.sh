#!/bin/bash
# pod_bootstrap.sh [ROOT] [ENGINE]: make a pod that draftzero.workers.runpod.bundle filled able to run DraftZero
# actors, and prove it. Idempotent; safe to re-run.
#   ROOT    (default /root/dz)          holds draftzero/ (the repo, with data/gorge/{decks/,pool.tsv}); venv/ goes here
#   ENGINE  (default /work/gorge-fork)  holds bin/dzgorge, bin/dzgorge.build.json and .cards (the path d1's config names)
# Env: TORCH_INDEX (default the CPU wheels: actors search in Go, torch is only imported), SMOKE_A/SMOKE_B (the smoke
# game's policies, default bot), SMOKE_SPLIT (train).
# Steps: apt (python3-venv, rsync) if missing -> venv with the image's packages visible -> torch (CPU wheel unless
# the image has one; then pinned so pip cannot swap it) -> pip install -e . -> dz version -> one dzgorge game.
# The last line is "pod_bootstrap: ready" (workers.runpod.bootstrap checks for it).
set -euo pipefail
ROOT=${1:-/root/dz}
ENGINE=${2:-/work/gorge-fork}
REPO=$ROOT/draftzero
VENV=$ROOT/venv
log() { echo "== pod_bootstrap: $*"; }
cd "$REPO"

export DEBIAN_FRONTEND=noninteractive PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
need=()
command -v python3 > /dev/null || need+=(python3)
python3 -c 'import ensurepip, venv' 2> /dev/null || need+=(python3-venv)
command -v rsync > /dev/null || need+=(rsync)
if [ ${#need[@]} -gt 0 ]; then
  log "apt install ${need[*]}"
  apt-get update -qq && apt-get install -y -qq "${need[@]}" > /dev/null
fi

# --system-site-packages: a CUDA image's torch stays usable; a slim image has none and gets the CPU wheel
[ -x "$VENV/bin/python" ] || python3 -m venv --system-site-packages "$VENV"
PY=$VENV/bin/python
"$PY" -m pip install -q --upgrade pip
if ! "$PY" -c 'import torch' 2> /dev/null; then
  log "torch (CPU wheel)"
  "$PY" -m pip install -q torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
fi
# Pin the torch that is there ("2.9.1+cpu" -> torch==2.9.1, which the local build satisfies): an unpinned
# `pip install -e .` may otherwise pull a 2 GB CUDA wheel from PyPI over it.
"$PY" -c "import torch; print('torch==' + torch.__version__.split('+')[0])" > "$ROOT/constraints.txt"
log "pip install -e . ($(cat "$ROOT/constraints.txt"))"
"$PY" -m pip install -q -c "$ROOT/constraints.txt" -e .
{ printf '#!/bin/sh\nexec %s "$@"\n' "$VENV/bin/dz" > /usr/local/bin/dz && chmod +x /usr/local/bin/dz; } 2> /dev/null || true   # dz on PATH for people

log "versions"
echo "draftzero $("$VENV/bin/dz" version)"
"$PY" -c "import sys, numpy, torch, yaml; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'torch', torch.__version__)"
"$PY" -c "from draftzero.resources import cpu_quota, mem_limit_gb; import os
print('capacity: quota', cpu_quota(), 'cores,', mem_limit_gb(), 'GB (nproc', os.cpu_count(), '- size workers from the quota)')"

log "engine"
DZG=$ENGINE/bin/dzgorge
for f in "$DZG" "$ENGINE/bin/dzgorge.build.json" "$ENGINE/.cards" data/gorge/decks data/gorge/pool.tsv; do
  [ -e "$f" ] || { echo "pod_bootstrap: missing $f (run workers.runpod.bundle first)"; exit 1; }
done
"$PY" -c "import json, sys; b = json.load(open(sys.argv[1])); print('dzgorge', b['ref'][:12], 'dirty' if b.get('dirty') else 'clean', b['binary_sha256'][:12])" "$ENGINE/bin/dzgorge.build.json"
SMOKE=$ROOT/smoke
rm -rf "$SMOKE" && mkdir -p "$SMOKE"
timeout 300 "$DZG" play -cards "$ENGINE/.cards" -decks data/gorge/decks -pool data/gorge/pool.tsv \
  -split "${SMOKE_SPLIT:-train}" -a "${SMOKE_A:-bot}" -b "${SMOKE_B:-bot}" -pairs 1 -single -workers 1 -seed 1 \
  -mulligans 2 -mull-heuristic -topk-max-options 64 -progress 0 -out "$SMOKE/games.jsonl" > "$SMOKE/play.log" 2>&1 \
  || { echo "pod_bootstrap: dzgorge smoke failed:"; tail -20 "$SMOKE/play.log"; exit 1; }
"$PY" - "$SMOKE/games.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
assert len(rows) == 1, f"smoke: {len(rows)} games, want 1"
g = rows[0]
assert g.get("result") in ("A", "B", "draw", "stall"), f"smoke game: {g.get('result')} {g.get('error', '')}"
print(f"smoke game: {g['decks'][0]} vs {g['decks'][1]}: {g['result']} in {g['turns']} turns, {g['ms']:.0f} ms")
PY
echo "pod_bootstrap: ready"
