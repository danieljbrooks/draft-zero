#!/usr/bin/env bash
# docs/028 stage 1 on one machine: pack the shared games, train each arm on them (skipping arms already trained), then
# play each arm's policy alone against the start's (paired, greedy, the evaluation decks) and score it by deck pair.
#   source ~/sp_env.sh; bash deploy/sp_stage1.sh <spec> <games run> <arm> [<arm> ...]
#   e.g. bash deploy/sp_stage1.sh configs/sp/stage1.yml runs/gnn/games/d0-v0-n100 visits kl cq10 cq30
# Env: M0 (the start), PAIRS (1000), WORKERS (12), HEAP (1400m), TRAIN_GPU (1), SKIP_GAMES=1 (train only), EXTRA_ARMS (checkpoints
# to evaluate without training: name=path ...), PH_PAIRS (0: skip; else also each policy alone against heuristic@100
# on docs/024's ladder deals, seed 20261001). Results: runs/sp/arms/<arm>/, runs/sp/h2h/{p0,ph}-<arm>/ and
# runs/sp/stage1_p0.jsonl, stage1_ph.jsonl (one line per arm).
set -uo pipefail
cd "$(dirname "$0")/.."
SPEC=$1 GAMES=$2; shift 2
: "${M0:?set M0 to the start network}"
PAIRS=${PAIRS:-1000} WORKERS=${WORKERS:-12} GPU=${TRAIN_GPU:-1}
log() { echo "[$(TZ=America/Los_Angeles date '+%a %-I:%M %p PT')] stage1: $*"; }
mkdir -p runs/sp/arms runs/sp/h2h
B=runs/sp/batches/$(basename "$GAMES")
if [ ! -s "$B/r1-$(basename "$GAMES").jsonl.gz" ]; then
  mkdir -p "$B"
  python - "$GAMES" "$B/r1-$(basename "$GAMES").jsonl.gz" <<'PY'
import sys
from draftzero.selfplay import records
print(records.pack(sys.argv[1], sys.argv[2], run="stage0", machine="r1", chunk="d0", version=0))
PY
fi
for arm in "$@"; do
  out=runs/sp/arms/$arm
  if [ -s "$out/summary.json" ]; then log "$arm already trained"; continue; fi
  log "training $arm"
  CUDA_VISIBLE_DEVICES=$GPU timeout -k 60 3h python tools/selfplay_transition/sp_train.py --spec "$SPEC" --arm "$arm" \
    --batches "$B" --out "$out" > "$out.log" 2>&1 || { log "$arm: training failed (see $out.log)"; continue; }
  log "$arm trained: $(cat "$out/summary.json" | tr -d '\n ')"
done
[ "${SKIP_GAMES:-0}" = 1 ] && exit 0
EVAL=()
for arm in "$@"; do EVAL+=("$arm=runs/sp/arms/$arm/final.pt.gz"); done
for x in ${EXTRA_ARMS:-}; do EVAL+=("$x"); done
for e in "${EVAL[@]}"; do
  arm=${e%%=*} ck=${e#*=}
  [ -s "$ck" ] || { log "$arm: no checkpoint $ck"; continue; }
  run=p0-$arm
  log "policy alone: $arm against the start ($PAIRS pairs)"
  timeout -k 60 4h bash deploy/sp_h2h.sh "$run" "$ck" "$M0" 2 --bot1 gnn_policy_greedy --bot2 gnn_policy_greedy \
    --pool data/pools/eval.txt --pairs "$PAIRS" --seed 20261006 --method pimc --workers "$WORKERS" --heap ${HEAP:-1400m} \
    --max-turns 50 --search-timeout 900 --game-timeout 7200 > "runs/sp/h2h/$run.log" 2>&1
  res=$(python tools/selfplay_transition/paired.py "runs/sp/h2h/$run")
  log "$arm: $res"
  echo "{\"arm\": \"$arm\", \"checkpoint\": \"$ck\", \"p0\": $res}" >> runs/sp/stage1_p0.jsonl
done
if [ "${PH_PAIRS:-0}" -gt 0 ]; then
  for e in "${EVAL[@]}"; do
    arm=${e%%=*} ck=${e#*=}
    [ -s "$ck" ] || continue
    run=ph-$arm
    log "policy alone: $arm against heuristic@100 ($PH_PAIRS pairs)"
    timeout -k 60 4h bash deploy/sp_h2h.sh "$run" "$ck" - 2 --bot1 gnn_policy_greedy --bot2 heuristic@100 \
      --pool data/pools/eval.txt --pairs "$PH_PAIRS" --seed 20261001 --method pimc --workers "$WORKERS" --heap ${HEAP:-1400m} \
      --max-turns 50 --search-timeout 900 --game-timeout 7200 > "runs/sp/h2h/$run.log" 2>&1
    res=$(python tools/selfplay_transition/paired.py "runs/sp/h2h/$run")
    log "$arm against heuristic@100: $res"
    echo "{\"arm\": \"$arm\", \"checkpoint\": \"$ck\", \"ph\": $res}" >> runs/sp/stage1_ph.jsonl
  done
fi
log "done"
