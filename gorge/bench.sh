#!/usr/bin/env bash
# Throughput and strength benchmark on the eval split (docs/026 §2-3).
#
#   bash gorge/bench.sh [workers]      # -> data/gorge/runs/bench/*.jsonl{,.summary.json}
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
GORGE_DIR="${GORGE_DIR:-$(cd "$REPO/.." && pwd)/ext/gorge}"
W="${1:-4}"
D="$REPO/data/gorge"
OUT="$D/runs/bench"
mkdir -p "$OUT"
cd "$GORGE_DIR"
play() { # name, then dzgorge play flags
  local name="$1"; shift
  [ -s "$OUT/$name.jsonl.summary.json" ] && { echo "skip $name"; return; }
  echo "== $name: $*"
  ./bin/dzgorge play -cards .cards -decks "$D/decks" -pool "$D/pool.tsv" -split eval -workers "$W" \
    -out "$OUT/$name.jsonl" "$@" 2> "$OUT/$name.log" | grep -E '"(games|games_per_hour|a_score|wall_s|az_ms_per_searched)"'
}
# One core: the per-core rate of the cheap policies.
play w1_bot_self   -a bot -b bot -pairs 2000 -seed 11 -workers 1
play w1_random_self -a random -b random -pairs 2000 -seed 12 -workers 1
# All workers.
play random_self   -a random -b random -pairs 5000 -seed 21
play bot_self      -a bot -b bot -pairs 5000 -seed 22
play bot_v_random  -a bot -b random -pairs 1000 -seed 23
play az10_v_bot    -a az:sims=10 -b bot -pairs 200 -seed 24
play az25_v_bot    -a az:sims=25 -b bot -pairs 200 -seed 25
play az100_v_bot   -a az:sims=100 -b bot -pairs 100 -seed 26
play az100_v_az25  -a az:sims=100 -b az:sims=25 -pairs 100 -seed 27
play az25_self     -a az:sims=25 -b az:sims=25 -pairs 200 -seed 28
