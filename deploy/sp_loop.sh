#!/usr/bin/env bash
# docs/028 stage 2: continue one recipe for more generations. Each generation: the last network plays itself (100
# simulations, stage 0's exploration, the training decks, records on), its games are packed, the next network trains
# from the last one on the last two generations' games (anchored, if the recipe has an anchor, to the start M0), and
# its policy alone plays the start's and the last network's, and heuristic@100.
#   source ~/sp_env.sh; bash deploy/sp_loop.sh <loop> <spec> <arm> <gen-1 checkpoint> <gen-1 batch dir> <first gen> <last gen>
#   e.g. bash deploy/sp_loop.sh recipe configs/sp/stage1.yml cq30 runs/sp/arms/cq30/final.pt.gz runs/sp/batches/d0-v0-n100 2 3
# Env: M0, SP_PAIRS (250: 500 games a generation), WORKERS (20), TRAIN_GPU (1), PAIRS (1000, policy-alone matches),
# PH_PAIRS (150). Results: runs/sp/loops/<loop>/g<g>/ (the network and its curves), runs/sp/h2h/{sp,p0,pp,ph}-<loop>-g<g>/,
# runs/sp/loops/<loop>/results.jsonl.
set -uo pipefail
cd "$(dirname "$0")/.."
LOOP=$1 SPEC=$2 ARM=$3 PREV=$4 PREVB=$5 G0=$6 G1=$7
: "${M0:?set M0 to the start network}"
SPP=${SP_PAIRS:-250} WORKERS=${WORKERS:-20} GPU=${TRAIN_GPU:-1} PAIRS=${PAIRS:-1000} PHP=${PH_PAIRS:-150}
log() { echo "[$(TZ=America/Los_Angeles date '+%a %-I:%M %p PT')] loop $LOOP: $*"; }
D=runs/sp/loops/$LOOP
mkdir -p "$D" runs/sp/h2h
for g in $(seq "$G0" "$G1"); do
  run=sp-$LOOP-g$g
  B=runs/sp/batches/$LOOP-g$g
  if [ ! -s "$B/r1-$LOOP-g$g.jsonl.gz" ]; then
    log "generation $g: self-play by $PREV ($SPP pairs)"
    timeout -k 60 14h bash deploy/sp_h2h.sh "$run" "$PREV" - 4 --pool data/pools/train.txt --bot1 gnn@100 --bot2 gnn@100 \
      --method pimc --root-noise 0.25 --root-noise-alpha 0.3 --sample-turns 3 --record --pairs "$SPP" --seed $((100 * g + 1)) \
      --workers "$WORKERS" --heap 2500m --max-turns 50 --search-timeout 1800 --game-timeout 14400 > "runs/sp/h2h/$run.log" 2>&1
    mkdir -p "$B"
    python - "runs/sp/h2h/$run" "$B/r1-$LOOP-g$g.jsonl.gz" "$g" <<'PY'
import sys
from draftzero.selfplay import records
print(records.pack(sys.argv[1], sys.argv[2], run=f"loop-g{sys.argv[3]}", machine="r1", chunk=f"g{sys.argv[3]}",
                   version=int(sys.argv[3]) - 1))
PY
  fi
  out=$D/g$g
  if [ ! -s "$out/summary.json" ]; then
    log "generation $g: training from $PREV on $PREVB + $B"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=$GPU timeout -k 60 3h python tools/selfplay_transition/sp_train.py --spec "$SPEC" --arm "$ARM" \
      --batches "$PREVB" "$B" --out "$out" --set "start=$PREV" --set "ref=$M0" > "$out.log" 2>&1 \
      || { log "training failed (see $out.log)"; exit 1; }
  fi
  ck=$out/final.pt.gz
  for kind in p0 pp ph; do
    r=$kind-$LOOP-g$g
    [ -s "runs/sp/h2h/$r/summary.json" ] && continue
    case $kind in
      p0) opp=$M0;   args=(--bot2 gnn_policy_greedy --pairs "$PAIRS" --seed 20261006) ;;
      pp) opp=$PREV; args=(--bot2 gnn_policy_greedy --pairs "$PAIRS" --seed 20261006) ;;
      ph) opp=-;     args=(--bot2 heuristic@100 --pairs "$PHP" --seed 20261001) ;;
    esac
    timeout -k 60 4h bash deploy/sp_h2h.sh "$r" "$ck" "$opp" 2 --bot1 gnn_policy_greedy "${args[@]}" \
      --pool data/pools/eval.txt --method pimc --workers "$WORKERS" --heap 2500m --max-turns 50 --search-timeout 900 \
      --game-timeout 7200 > "runs/sp/h2h/$r.log" 2>&1
    res=$(python tools/selfplay_transition/paired.py "runs/sp/h2h/$r")
    log "generation $g, $kind: $res"
    echo "{\"loop\": \"$LOOP\", \"gen\": $g, \"kind\": \"$kind\", \"checkpoint\": \"$ck\", \"res\": $res}" >> "$D/results.jsonl"
  done
  PREV=$ck PREVB=$B
done
log "done"
