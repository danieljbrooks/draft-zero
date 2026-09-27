#!/bin/bash
# pilot_v02.sh — the v0.2 pilot: the loop (configs/exp2_v02_pilot_pod.yml) through launch.sh
# and its watchdog, then two 10-minute benchmarks of the shared inference server. Run after
# deploy/pilot_v02_setup.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
export MZ_DECK_DIR="$(pwd)/data/deckgen/FDN_PremierDraft_wr60/top_player_FDN_decks"
export MZ_ACTION_VOCAB="$(pwd)/assets/vocab/FDN_SPG.tsv"
# Size the layout from THIS pod's cgroup (never nproc): K JVMs x 4 threads, K = cores / 4,
# 70% of the RAM split across the heaps, 2 games per game thread per generation, and one
# eval pair per JVM. configs/exp2_v02_pilot_pod.yml is the 31-core version of the same.
mkdir -p logs
CFG=logs/pilot_v02.yml
MODE=--fresh
if [ "${RESUME:-}" = 1 ] && [ -f "$CFG" ]; then
  # continue the newest run at its interrupted stage, with the layout it was sized with
  MODE=--resume
  python - <<'PY'
import yaml
fresh = yaml.safe_load(open("configs/exp2_v02_pilot_pod.yml"))
cfg = yaml.safe_load(open("logs/pilot_v02.yml"))
cfg["train_batch"] = fresh.get("train_batch")      # settings added since it was sized
yaml.safe_dump(cfg, open("logs/pilot_v02.yml", "w"), sort_keys=False)
PY
  read K HEAP <<< "$(python -c "import yaml; j=yaml.safe_load(open('$CFG'))['jvm']; print(j['jvms'], j['heap'])")"
else
read K HEAP <<< "$(python - <<'PY'
import yaml
from draftzero.resources import cpu_quota, mem_limit_gb
cores, ram = cpu_quota(), mem_limit_gb()
if not cores or not ram:
    raise SystemExit("!! no cgroup cpu/memory limit found; refusing to size from nproc")
k = max(1, int(cores // 4))
heap = max(4, int(ram * 0.7 / k))
cfg = yaml.safe_load(open("configs/exp2_v02_pilot_pod.yml"))
games = k * 4 * 2
cfg["jvm"].update(jvms=k, heap=f"{heap}g")
cfg.update(bootstrap_games=games, games_per_gen=games, chunk_games=8)
cfg["eval"]["pairs"] = k
yaml.safe_dump(cfg, open("logs/pilot_v02.yml", "w"), sort_keys=False)
print(k, f"{heap}g")
PY
)"
fi
[ -n "$K" ] || { echo "!! sizing failed"; exit 1; }
echo "== layout: $K JVMs x 4 threads, heap $HEAP, $((K * 8)) games per generation"

DZ_STALL_MINUTES=90 bash deploy/launch.sh "$CFG" "$MODE" || { echo "!! launch failed"; exit 1; }
RUN="$(ls -1dt runs/*/ | head -1)"; RUN="${RUN%/}"
echo "== $(date -u +%H:%M) waiting for $RUN"
while true; do
  python -c "import json,sys; sys.exit(0 if json.load(open('$RUN/run.json')).get('completed_at') else 1)" && break
  # the watchdog restarts a crashed loop; only give up once no loop and no watchdog are left
  pgrep -f "draftzero.loop" >/dev/null || pgrep -f "draftzero.watchdog" >/dev/null || { echo "!! loop gone"; break; }
  sleep 60
done
echo "== $(date -u +%H:%M) loop finished"

# The shared server, v0.2's fixed HTTP pool (6 threads: --clients 3) vs one sized to all
# K x 4 game threads, on the pilot's own gen-1 checkpoint.
mkdir -p bench
for C in 3 $((K * 4)); do
  echo "== $(date -u +%H:%M) bench shared server, clients=$C"
  python tools/throughput_bench.py --out "bench/v02_${K}x4_shared_c$C" --jvms "$K" --threads 4 --heap "$HEAP" \
    --gc zgc-gen --mode network --shared-server --model FDN_exp2_v02 --checkpoint gen1 --clients "$C" \
    --minutes 10 --deck-root "$MZ_DECK_DIR" 2>&1 | tail -1 | tee -a bench/summary.jsonl
done
echo "== pilot v02 done $(date -u +%H:%M)"
