#!/bin/bash
# launch.sh — start the DraftZero loop plus its watchdog on this worker.
#   bash deploy/launch.sh [config.yml] [--fresh|--resume]
#
# Env:
#   MZ_DECK_DIR      where the .dck files live (default data/decks)
#   DZ_PERSIST       where checkpoints are mirrored (default data/persist)
#   DZ_ON_COMPLETE   shell command run after a verified final sync (e.g. destroy the pod)
#   DZ_STALL_MINUTES no-new-game timeout before the run is declared stalled (default 45)
#   DZ_MAX_HOURS     wall-clock budget cap; the worker stops itself after this many hours
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH:-$(pwd)/src}"
export MZ_ACTION_VOCAB="${MZ_ACTION_VOCAB:-$(pwd)/assets/vocab/FDN_SPG.tsv}"
export MZ_DECK_DIR="${MZ_DECK_DIR:-$(pwd)/data/decks}"
export DZ_PERSIST="${DZ_PERSIST:-$(pwd)/data/persist}"
export MZ_BATCH_SIZE="${MZ_BATCH_SIZE:-32}"
export MZ_INFER_DTYPE="${MZ_INFER_DTYPE:-float16}"

CFG="${1:-configs/runpod.yml}"
MODE="${2:---fresh}"
mkdir -p "$DZ_PERSIST" logs

# Pool files are stored as deck stems; confirm they resolve before burning an hour finding
# out from the JVM that every game failed with "deck size=0".
python -c "
from pathlib import Path
from draftzero import paths
stems = paths.read_pool(Path('data/pools/train.txt'))
paths.resolve_pool(stems[:50], Path('${MZ_DECK_DIR}'), Path('logs/_poolcheck.txt'))
print(f'== pools ok: {len(stems)} decks under ${MZ_DECK_DIR}')
"

TARGET="$(python -c "import yaml;print(yaml.safe_load(open('$CFG')).get('target_games') or 0)")"
echo "== config $CFG | target_games=$TARGET | persist=$DZ_PERSIST"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "   (no GPU: CPU-only worker)"

# Distinguish THIS run's directory from any that already exist: taking the newest
# unconditionally hands the watchdog a finished run, which then looks stalled and gets a
# healthy worker torn down underneath it.
MARKER="$(mktemp)"; mkdir -p runs
nohup python -u -m draftzero.loop --config "$CFG" "$MODE" > logs/loop.log 2>&1 &
LOOP=$!
echo "== loop pid $LOOP -> logs/loop.log"

# --resume / --extend continue an EXISTING run dir, so no new one ever appears and the
# newer-than-marker search below would spin for 3 minutes and then abort a healthy restart.
RUN_DIR=""
case "$MODE" in
  --resume|--extend)
    RUN_DIR="$(ls -1dt runs/*/ 2>/dev/null | head -1)"
    RUN_DIR="${RUN_DIR%/}"
    [ -n "$RUN_DIR" ] || { echo "!! $MODE but no existing run dir found"; rm -f "$MARKER"; exit 1; }
    echo "== $MODE: continuing $RUN_DIR"
    ;;
esac

for _ in $(seq 1 90); do
  [ -n "$RUN_DIR" ] && break
  RUN_DIR="$(find runs -maxdepth 1 -mindepth 1 -type d -newer "$MARKER" 2>/dev/null | head -1)"
  [ -n "$RUN_DIR" ] && break
  kill -0 "$LOOP" 2>/dev/null || { echo "!! loop exited before creating a run dir:"; tail -20 logs/loop.log; rm -f "$MARKER"; exit 1; }
  sleep 2
done
rm -f "$MARKER"
[ -n "$RUN_DIR" ] || { echo "!! no run dir appeared"; tail -20 logs/loop.log; exit 1; }
echo "== run dir $RUN_DIR"

WD=(--run-dir "$RUN_DIR" --persist "$DZ_PERSIST" --interval "${DZ_WD_INTERVAL:-300}"
    --stall-minutes "${DZ_STALL_MINUTES:-45}")
[ -n "${DZ_MAX_HOURS:-}" ] && WD+=(--max-hours "$DZ_MAX_HOURS")
WD+=(--grace-hours "${DZ_GRACE_HOURS:-3}")   # finish the current generation before stopping
[ "$TARGET" -gt 0 ] && WD+=(--target "$TARGET")
[ -n "${DZ_ON_COMPLETE:-}" ] && WD+=(--on-complete "$DZ_ON_COMPLETE")

nohup python -u -m draftzero.watchdog "${WD[@]}" > logs/watchdog.log 2>&1 &
echo "== watchdog pid $! -> logs/watchdog.log"
echo "   tail -f logs/loop.log logs/watchdog.log"
