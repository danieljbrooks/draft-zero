#!/usr/bin/env bash
# A dzg network's paired games on the eval decks.
#
#   NET=runs/gen1/mlp/best.pt OUT=data/gorge/runs/eval/gen1-mlp SIMS=100 PAIRS=300 bash gorge/dzg_eval.sh
#
# ARMS (space-separated, default "full") picks the matches, each against the same search without
# the network at SIMS simulations unless named otherwise:
#   full     the network's prior and value          value   its value only (uniform prior)
#   prior    its prior only (heuristic leaf)        alone   its policy alone against gorge's bot
#   prev     the full search against the full search with the network PREV (a previous generation)
#   double   the full search against the search without a network at twice the simulations
set -euo pipefail
source "$(dirname "$0")/dzg_lib.sh"
NET="${NET:?NET=checkpoint}"; OUT="${OUT:?OUT=dir}"; SIMS="${SIMS:-100}"; PAIRS="${PAIRS:-300}"
W="${W:-60}"; SEED="${SEED:-77}"; ARMS="${ARMS:-full}"; MAXMIN="${MAXMIN:-120}"
mkdir -p "$OUT"
serve "$NET" "$OUT/serve"; addr="$ADDR"
paddr=""
if [[ " $ARMS " == *" prev "* ]]; then serve "${PREV:?PREV=checkpoint for the prev arm}" "$OUT/serve-prev"; paddr="$ADDR"; fi
base="az:sims=$SIMS"
for arm in $ARMS; do
  case "$arm" in
    full)   a="$base:remote=$addr";               b="$base" ;;
    value)  a="$base:prior=uniform:remote=$addr"; b="$base" ;;
    prior)  a="$base:leaf=heuristic:remote=$addr"; b="$base" ;;
    alone)  a="prior:remote=$addr";               b="bot" ;;
    prev)   a="$base:remote=$addr";               b="$base:remote=$paddr" ;;
    double) a="$base:remote=$addr";               b="az:sims=$((2 * SIMS))" ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
  pairs=$PAIRS
  [ "$arm" = alone ] && pairs=$((PAIRS * 4))
  echo "== $arm: $a  vs  $b ($pairs pairs)"
  play "$MAXMIN" -split eval -a "$a" -b "$b" -pairs "$pairs" -workers "$W" -seed "$SEED" -out "$OUT/$arm.jsonl" \
    | grep -E '"(a_score|games|games_per_hour|az_ms_per_searched|remote_[a-z_]+)"' || true
done
