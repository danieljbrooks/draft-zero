#!/usr/bin/env bash
# Grab a RunPod CPU pod as soon as one is rentable (docs/020): every TRY_EVERY seconds, ask the v2 catalog which data
# centers have the flavor at each size (GET api.runpod.io/v2/catalog/cpus/<flavor>?include=AVAILABILITY), and run
# tools/compute_bench/run_pod.sh there, largest size first. CPU pods are often listed as available but refuse
# creates; a create pinned to the listed data center is the only kind that has worked (3-4 October 2026).
#   bash tools/compute_bench/cpu_grab.sh <tag> <flavor> "<sizes, largest first>" -- <KEY=VALUE knobs for compute_bench.sh>
#   e.g. bash tools/compute_bench/cpu_grab.sh cpu3c cpu3c "8 4 2" -- GAMES_MIN=40 ...
# Env: TRIES (default 60), TRY_EVERY (120). The knobs WORKERS/TRAIN_THREADS/TSTEP_THREADS are set from the size.
set -uo pipefail
cd "$(dirname "$0")/../.."
TAG=$1 FLAVOR=$2 SIZES=$3; shift 3; [ "${1:-}" = "--" ] && shift
source ~/.runpod/env
PRICE_PER_VCPU=$(curl -s -m 30 "https://api.runpod.io/v2/catalog/cpus/$FLAVOR" -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "User-Agent: curl/8.7.1" | python3 -c "import json,sys; print(json.load(sys.stdin)['price']['securePerVcpu'])")
for k in $(seq 1 "${TRIES:-60}"); do
  for v in $SIZES; do
    DCS=$(curl -s -m 30 "https://api.runpod.io/v2/catalog/cpus/$FLAVOR?include=AVAILABILITY&product=POD&vcpuCount=$v" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" -H "User-Agent: curl/8.7.1" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(' '.join(x['id'] for x in d.get('dataCenters', []) if x.get('availability') not in (None, 'NONE')))" 2>/dev/null)
    for dc in $DCS; do
      price=$(python3 -c "print(round($PRICE_PER_VCPU * $v, 3))")
      W=$(( v > 2 ? v - 2 : 1 ))
      echo "[$(date -u +%H:%M:%S)] cpu_grab $TAG: trying $FLAVOR-$v in $dc (\$$price/hr)"
      if SLIM=1 TRIES=1 DEADLINE_MIN=${DEADLINE_MIN:-170} bash tools/compute_bench/run_pod.sh "$TAG-$v" "$price" \
           "$FLAVOR-$v-$((v * 2)) Secure CPU pod, $dc; slim image" --cpu "$FLAVOR" --vcpu "$v" --dc "$dc" --disk $(( v * 10 > 40 ? 40 : v * 10 )) \
           -- REPLICAS=1 WORKERS=$W TRAIN_THREADS=$v TSTEP_THREADS=$v "$@"; then
        echo "[$(date -u +%H:%M:%S)] cpu_grab $TAG: started $FLAVOR-$v in $dc"; exit 0
      fi
    done
  done
  sleep "${TRY_EVERY:-120}"
done
echo "[$(date -u +%H:%M:%S)] cpu_grab $TAG: gave up"; exit 1
