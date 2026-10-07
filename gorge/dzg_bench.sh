#!/usr/bin/env bash
# Games an hour with search at 100, 1,000 and 10,000 simulations, with and without a dzg network.
#
#   OUT=data/gorge/runs/bench-3080ti NETS="none mlp=path/best.pt" SIMS="100 1000 10000" bash gorge/dzg_bench.sh
#
# For each network (none = gorge's heuristic leaf and a uniform prior; NAME=CKPT, or NAME=random:ARCH
# for an untrained one: the speed doesn't depend on the weights) and each budget it plays, on the eval
# decks:
#   eval:     the search for one seat against gorge's bot (paired: each deck pair twice)
#   selfplay: the search for both seats, as generation self-play does (with exploration on turns 1-4)
# Games per run scale down with the budget; each run is killed after MAXMIN minutes. Searches with a
# network run WNET workers (default 2x the cores) so games waiting on the GPU don't idle the CPU.
set -euo pipefail
source "$(dirname "$0")/dzg_lib.sh"
OUT="${OUT:?OUT=dir}"; NETS="${NETS:-none}"; SIMS="${SIMS:-100 1000 10000}"; MODES="${MODES:-eval selfplay}"
NCPU="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
W="${W:-$NCPU}"; WNET="${WNET:-$((2 * NCPU))}"; MAXMIN="${MAXMIN:-30}"; SEED="${SEED:-4242}"
mkdir -p "$OUT"
{ echo "host: $(hostname)"; nproc; lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)|Thread'; nvidia-smi -L 2>/dev/null || true
  cat /sys/fs/cgroup/cpu.max 2>/dev/null || true; } > "$OUT/host.txt"
for net in $NETS; do
  name="${net%%=*}"; ckpt="${net#*=}"; remote=""; w="$W"
  if [ "$name" != none ]; then serve "$ckpt" "$OUT/serve-$name"; remote=":remote=$ADDR"; w="$WNET"; fi
  for s in $SIMS; do
    # about 8 games a worker at 100 simulations, 2 at 1,000, 1 at 10,000
    per=$(( s <= 100 ? 8 : (s <= 1000 ? 2 : 1) ))
    for mode in $MODES; do
      if [ "$mode" = eval ]; then
        a="az:sims=$s$remote"; b="bot"; pairs=$(( (w * per + 1) / 2 ))
      else
        a="az:sims=$s:explore$remote"; b="$a"; pairs=$(( w * per ))
      fi
      tag="$name-$mode-$s"
      echo "== $tag: $pairs pairs, $w workers"
      play "$MAXMIN" -split eval -a "$a" -b "$b" -pairs "$pairs" -workers "$w" -seed "$SEED" -out "$OUT/$tag.jsonl" \
        > "$OUT/$tag.log" 2>&1 || echo "  ($tag stopped: exit $?)"
      grep -E '"(games|games_per_hour|az_ms_per_searched|az_decisions_searched|remote_states_per_batch|remote_wait_ms_per_miss)"' \
        "$OUT/$tag.jsonl.summary.json" 2>/dev/null | tr -d '\n' ; echo
    done
  done
  stop_servers
done
