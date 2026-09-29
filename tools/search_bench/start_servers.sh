#!/usr/bin/env bash
# Start R value-only servers (ports 50052 + 100 i) for a checkpoint, and optionally MageZero's own
# server on 50900 (the policy reference). Logs in runs/search_bench/servers/.
#   bash tools/search_bench/start_servers.sh <model.pt.gz> <R> [full-deck full-ckpt]
set -euo pipefail
cd "$(dirname "$0")/../.."
export MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv
MODEL=$1; R=$2
mkdir -p runs/search_bench/servers
for i in $(seq 0 $((R - 1))); do
  port=$((50052 + 100 * i))
  nohup setsid python tools/search_bench/value_server.py --model "$MODEL" --port $port --threads 24 \
    > runs/search_bench/servers/value_$port.log 2>&1 < /dev/null &
done
if [ $# -ge 4 ]; then
  nohup setsid python tools/search_bench/serve.py --deck "$3" --checkpoint "$4" --replicas 1 --base-port 50900 --clients 16 \
    > runs/search_bench/servers/full.log 2>&1 < /dev/null &
fi
for i in $(seq 0 $((R - 1))); do
  port=$((50052 + 100 * i))
  for k in $(seq 1 90); do curl -s localhost:$port/healthz > /dev/null && break; sleep 2; done
done
[ $# -ge 4 ] && for k in $(seq 1 90); do curl -s localhost:50900/healthz > /dev/null && break; sleep 2; done
echo "servers up"
