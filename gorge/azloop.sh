#!/usr/bin/env bash
# One AlphaZero-style generation on gorge (docs/025 §4): search self-play on the TRAIN decks
# records every searched decision's visit counts, gorge's trainer (policytrain -visits-corpus) fits
# a policy+value network to them, and the network is then played on the EVAL decks.
#
#   bash gorge/azloop.sh <gen>          # gen 0: no network (uniform prior, gorge's heuristic leaf)
#   NET=data/gorge/runs/az/gen1.gpol bash gorge/azloop.sh 1
#
# Knobs (env): SIMS (search simulations, 50), PAIRS (self-play deck pairs, 1500), W (workers, 4),
# TRAIN_CORPORA (comma list; default this generation's corpus), TEMP (visit target temperature, 1),
# RESIDUAL (fixed bonus for the bot's own answer, 0), EPOCHS (12).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
GORGE_DIR="${GORGE_DIR:-$(cd "$REPO/.." && pwd)/ext/gorge}"
GEN="$1"
SIMS="${SIMS:-50}"; PAIRS="${PAIRS:-1500}"; W="${W:-4}"
TEMP="${TEMP:-1}"; RESIDUAL="${RESIDUAL:-0}"; EPOCHS="${EPOCHS:-12}"
D="$REPO/data/gorge"; RUN="$D/runs/az"; mkdir -p "$RUN"
NETARG=""; [ -n "${NET:-}" ] && NETARG=":net=$NET"
cd "$GORGE_DIR"
play() { ./bin/dzgorge play -cards .cards -decks "$D/decks" -pool "$D/pool.tsv" -workers "$W" "$@"; }

CORPUS="$RUN/gen$GEN.visits.jsonl.gz"
if [ ! -s "$RUN/gen$GEN.selfplay.jsonl.summary.json" ]; then
  echo "== gen $GEN self-play: az sims $SIMS${NET:+ with $NET}, $PAIRS train-deck pairs"
  P="az:sims=$SIMS$NETARG:explore:nonoise"
  play -split train -a "$P" -b "$P" -pairs "$PAIRS" -seed $((1000 + GEN)) \
    -corpus "$CORPUS" -out "$RUN/gen$GEN.selfplay.jsonl" 2> "$RUN/gen$GEN.selfplay.log"
fi

NEXT="$RUN/gen$((GEN + 1)).gpol"
if [ ! -s "$NEXT" ]; then
  echo "== train gen $((GEN + 1)) on ${TRAIN_CORPORA:-$CORPUS}"
  ./bin/policytrain -visits-corpus "${TRAIN_CORPORA:-$CORPUS}" -out "$NEXT" -epochs "$EPOCHS" -lr 0.1 -clip 5 \
    -value-weight 1 -value-blend 0.05 -visits-temp "$TEMP" -residual-init "$RESIDUAL" -seed 1 \
    > "$RUN/gen$((GEN + 1)).train.log" 2>&1
  tail -4 "$RUN/gen$((GEN + 1)).train.log"
fi
