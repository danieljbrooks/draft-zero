#!/bin/bash
# pilot_bench.sh — experiments 2 and 3 of the exp #2 pilot, on ONE pod so every number is
# comparable (CPUs differ between hosts even for the same GPU type). Run after pilot_setup.sh.
#   bash deploy/pilot_bench.sh [minutes per run, default 10]
#
# Exp 3 (Will's #1 issue): the same total game threads as one big JVM vs several 4-thread JVMs,
#   first offline (XMage only, isolates the JVM layout), then with the network, where each JVM
#   gets its own inference server or all share one (ROADMAP B3).
# Exp 2: single-JVM tuning with the network on: generational ZGC, and fewer threads.
# Results: bench/<name>/result.json, summarized in bench/summary.jsonl.
set -uo pipefail
cd "$(dirname "$0")/.."
MIN="${1:-10}"

export PATH="$PATH"
export MZ_ACTION_VOCAB="$(pwd)/assets/vocab/FDN_SPG.tsv"
export MZ_DECK_DIR="$(pwd)/data/deckgen/FDN_PremierDraft_wr60/top_player_FDN_decks"
export MZ_INFER_DTYPE=float16 MZ_BATCH_SIZE=32 MZ_TORCH_THREADS=1

# The container's REAL core quota. Never nproc / os.cpu_count(): on a pod those report the
# host (e.g. 384 on a box capped at ~40 cores), which would launch ~96 JVMs.
CORES=$(awk '$1 != "max" {print int($1/$2)}' /sys/fs/cgroup/cpu.max 2>/dev/null)
[ -n "$CORES" ] || { echo "!! no cgroup cpu quota found; refusing to guess from nproc"; exit 1; }
K=$(( CORES / 4 )); [ "$K" -lt 2 ] && K=2
T=$(( K * 4 ))                                  # total game threads, identical across layouts
echo "== $CORES usable cores -> $T game threads: 1x$T vs ${K}x4 | $MIN min per run"
mkdir -p bench

run() {   # server threads, name, then throughput_bench args
  export MZ_SERVER_THREADS="$1"; local name="$2"; shift 2
  echo "== $(date +%H:%M) $name: $*"
  python tools/throughput_bench.py --out "bench/$name" --minutes "$MIN" "$@" 2>&1 | tail -1 | tee -a bench/summary.jsonl
}

# ── Exp 3: JVM layout ──
run 1 off_1x$T      --jvms 1  --threads $T --heap 48g
run 1 off_${K}x4    --jvms $K --threads 4
run $T net_1x$T        --jvms 1  --threads $T --heap 48g --mode network --checkpoint gen33
run 4 net_${K}x4_own  --jvms $K --threads 4 --mode network --checkpoint gen33
run $T net_${K}x4_shared --jvms $K --threads 4 --mode network --checkpoint gen33 --shared-server

# ── Exp 2: single-JVM tuning (network, the real workload) ──
run $T net_1x${T}_zgcgen --jvms 1 --threads $T --heap 48g --gc zgc-gen --mode network --checkpoint gen33
H=$(( T / 2 ))
run $H net_1x$H          --jvms 1 --threads $H --heap 48g --mode network --checkpoint gen33

echo "== bench done $(date +%H:%M)"
