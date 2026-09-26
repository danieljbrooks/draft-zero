#!/bin/bash
# pilot_v02.sh — the v0.2 pilot: the loop (configs/exp2_v02_pilot_pod.yml) through launch.sh
# and its watchdog, then two 10-minute benchmarks of the shared inference server. Run after
# deploy/pilot_v02_setup.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
export MZ_DECK_DIR="$(pwd)/data/deckgen/FDN_PremierDraft_wr60/top_player_FDN_decks"
export MZ_ACTION_VOCAB="$(pwd)/assets/vocab/FDN_SPG.tsv"
CFG=configs/exp2_v02_pilot_pod.yml

DZ_STALL_MINUTES=90 bash deploy/launch.sh "$CFG" --fresh || { echo "!! launch failed"; exit 1; }
RUN="$(ls -1dt runs/*/ | head -1)"; RUN="${RUN%/}"
echo "== $(date -u +%H:%M) waiting for $RUN"
while true; do
  python -c "import json,sys; sys.exit(0 if json.load(open('$RUN/run.json')).get('completed_at') else 1)" && break
  # the watchdog restarts a crashed loop; only give up once no loop and no watchdog are left
  pgrep -f "draftzero.loop" >/dev/null || pgrep -f "draftzero.watchdog" >/dev/null || { echo "!! loop gone"; break; }
  sleep 60
done
echo "== $(date -u +%H:%M) loop finished"

# The shared server, v0.2's fixed HTTP pool (6 threads: --clients 3) vs one sized to the 28
# game threads, on the pilot's own gen-1 checkpoint.
for C in 3 28; do
  echo "== $(date -u +%H:%M) bench shared server, clients=$C"
  python tools/throughput_bench.py --out "bench/v02_7x4_shared_c$C" --jvms 7 --threads 4 --heap 11g \
    --gc zgc-gen --mode network --shared-server --model FDN_exp2_v02 --checkpoint gen1 --clients "$C" \
    --minutes 10 --deck-root "$MZ_DECK_DIR" 2>&1 | tail -1 | tee -a bench/summary.jsonl
done
echo "== pilot v02 done $(date -u +%H:%M)"
