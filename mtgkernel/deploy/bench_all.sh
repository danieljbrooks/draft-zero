#!/bin/bash
# bench_all.sh OUT_DIR THREADS: dzk bench on the eval pairs (generated + fixture decks) for each bot, one JSON each.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=$1; T=${2:-1,8}
mkdir -p "$OUT"
D=decks/gen_v1:decks/fixtures
for spec in "random 20" "mcts:100 60" "mcts:300 90" "mcts:1000 120"; do
  set -- $spec
  ./target/release/dzk bench --decks-dir $D --pairs pairs/eval_v1.tsv --bot "$1" --seconds "$2" --threads "$T" \
    > "$OUT/bench_${1//[:,=]/_}.json" 2> "$OUT/bench_${1//[:,=]/_}.err"
done
uptime > "$OUT/uptime.txt"; nproc >> "$OUT/uptime.txt"
cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu.max 2>/dev/null >> "$OUT/uptime.txt"
lscpu > "$OUT/lscpu.txt" 2>/dev/null || sysctl -n machdep.cpu.brand_string > "$OUT/lscpu.txt"
