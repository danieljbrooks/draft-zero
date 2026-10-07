#!/usr/bin/env bash
# One AlphaZero generation's self-play with a dzg network, packed for training.
#
#   NET=runs/gen1/mlp/best.pt OUT=data/gorge/runs/gen1 SIMS=100 PAIRS=8000 W=60 SEED=201 bash gorge/dzg_gen.sh
#
# The search plays both seats on the train decks, sampling moves by visits (with root noise) on
# turns 1-4, and records every searched decision in the entity encoding. Without NET it is
# generation 0: a uniform prior and gorge's heuristic leaf. Stops after MAXMIN minutes
# (a hard kill; the packer keeps every finished game).
set -euo pipefail
source "$(dirname "$0")/dzg_lib.sh"
OUT="${OUT:?OUT=dir}"; SIMS="${SIMS:-100}"; PAIRS="${PAIRS:-8000}"; W="${W:-60}"; SEED="${SEED:-201}"
MAXMIN="${MAXMIN:-120}"; SPLIT="${SPLIT:-train}"; EXTRA="${EXTRA:-}"
mkdir -p "$OUT"
spec="az:sims=$SIMS:explore$EXTRA"
if [ -n "${NET:-}" ]; then
  serve "$NET" "$OUT/serve"
  spec+=":remote=$ADDR"
fi
echo "self-play: $spec, $PAIRS games on $SPLIT, $W workers"
play "$MAXMIN" -split "$SPLIT" -a "$spec" -b "$spec" -pairs "$PAIRS" -workers "$W" -seed "$SEED" \
  -record-features entity -out "$OUT/selfplay.jsonl" -corpus "$OUT/selfplay.visits.jsonl.gz" || echo "self-play stopped (exit $?)"
stop_servers
("$GORGE_DIR/bin/dzgorge" pack -in "$OUT/selfplay.visits.jsonl.gz" -out "$OUT/pack" -shard 200000)
