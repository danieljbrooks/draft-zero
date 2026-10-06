#!/usr/bin/env bash
# Evaluate one trained network on the EVAL decks (docs/025 §4), paired games:
#   the policy alone against the bot and against random; and inside the search, against the
#   network-free search at the same budget (prior and leaf; leaf only; prior only).
#
#   NET=data/gorge/runs/az/gen1.gpol bash gorge/evalnet.sh
# Knobs: SIMS (25), PAIRS (search arms, 150), PRIOR_PAIRS (policy-alone arms, 1000), W (4), ARMS.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
GORGE_DIR="${GORGE_DIR:-$(cd "$REPO/.." && pwd)/ext/gorge}"
NET="$(cd "$(dirname "$NET")" && pwd)/$(basename "$NET")"
SIMS="${SIMS:-25}"; PAIRS="${PAIRS:-150}"; PRIOR_PAIRS="${PRIOR_PAIRS:-1000}"; W="${W:-4}"
D="$REPO/data/gorge"; NAME="$(basename "$NET" .gpol)"; OUT="$D/runs/eval/$NAME"; mkdir -p "$OUT"
ARMS="${ARMS:-prior_v_bot prior_v_random full_v_az leaf_v_az prioronly_v_az full_v_bot}"
cd "$GORGE_DIR"
arm() { # name, pairs, A, B
  [ -s "$OUT/$1.jsonl.summary.json" ] && { echo "skip $1"; return; }
  echo "== $NAME $1: $3 vs $4"
  ./bin/dzgorge play -cards .cards -decks "$D/decks" -pool "$D/pool.tsv" -split eval -workers "$W" \
    -pairs "$2" -seed 77 -a "$3" -b "$4" -out "$OUT/$1.jsonl" 2> "$OUT/$1.log" | grep -E '"(a_score|games_per_hour)"'
}
for a in $ARMS; do
  case $a in
    prior_v_bot)    arm $a "$PRIOR_PAIRS" "prior:net=$NET" bot ;;
    prior_v_random) arm $a "$PRIOR_PAIRS" "prior:net=$NET" random ;;
    full_v_az)      arm $a "$PAIRS" "az:sims=$SIMS:net=$NET" "az:sims=$SIMS" ;;
    leaf_v_az)      arm $a "$PAIRS" "az:sims=$SIMS:net=$NET:prior=uniform" "az:sims=$SIMS" ;;
    prioronly_v_az) arm $a "$PAIRS" "az:sims=$SIMS:net=$NET:leaf=heuristic" "az:sims=$SIMS" ;;
    full_v_bot)     arm $a "$PAIRS" "az:sims=$SIMS:net=$NET" bot ;;
  esac
done
