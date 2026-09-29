#!/usr/bin/env bash
# Follow-up (the extra budget): the network's policy heads as PUCT priors, MageZero's setPriors
# (softmax at temperature 1.5, +0.1 for anything but Pass). Needs MageZero's own server, which
# returns the policy heads: 4 replicas on 50052.. replace the value-only servers.
set -uo pipefail
cd "$(dirname "$0")/../.."
export MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv
bash tools/search_bench/stop_servers.sh
nohup setsid python tools/search_bench/serve.py --deck FDN_exp2 --checkpoint gen18 --replicas 4 --base-port 50052 \
  --clients 10 --log-dir runs/search_bench/servers_full > runs/search_bench/serve_full.log 2>&1 < /dev/null &
for port in 50052 50152 50252 50352; do
  for k in $(seq 1 120); do curl -s localhost:$port/healthz > /dev/null && break; sleep 2; done
done
P=50052,50152,50252,50352
R="python tools/search_bench/run.py --items data/search_bench/sb-v1 --split test --evaluator remote --ports $P --workers 32 --heap 2g"
$R --out runs/search_bench/priors_network --grid priors --budgets 300,1000 --methods pimc1,pimc4,ismcts,clairvoyant
echo "plan_priors done"
