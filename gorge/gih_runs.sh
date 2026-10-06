#!/usr/bin/env bash
# Self-play for 17lands card statistics (docs/025 §5): each policy against itself on decks drawn
# from the whole pool, one game per deck pair.
#
#   bash gorge/gih_runs.sh            # -> data/gorge/runs/gih/*.jsonl
#   python gorge/analyze.py gih data/gorge/runs/gih/bot.jsonl
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
GORGE_DIR="${GORGE_DIR:-$(cd "$REPO/.." && pwd)/ext/gorge}"
W="${W:-4}"
D="$REPO/data/gorge"; OUT="$D/runs/gih"; mkdir -p "$OUT"
cd "$GORGE_DIR"
self() { # name, policy, pairs, seed
  [ -s "$OUT/$1.jsonl.summary.json" ] && { echo "skip $1"; return; }
  echo "== $1: $2 x $3"
  ./bin/dzgorge play -cards .cards -decks "$D/decks" -pool "$D/pool.tsv" -split all -workers "$W" \
    -a "$2" -b "$2" -pairs "$3" -seed "$4" -out "$OUT/$1.jsonl" 2> "$OUT/$1.log" | grep -E '"(games|games_per_hour)"'
}
self bot    bot    100000 31
self random random  20000 32
for net in ${NETS:-}; do # NETS="data/gorge/runs/az/gen1.gpol ..."
  self "prior_$(basename "$net" .gpol)" "prior:net=$net" 20000 33
done
