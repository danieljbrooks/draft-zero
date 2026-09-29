#!/usr/bin/env bash
# Follow-up (docs/016 §8): #2b's human-pretrained starting network inside the search. Its policy
# as PUCT priors (MageZero's setPriors), and its value head (trained on human game results) at the
# leaves; PIMC with 1 world, the cheapest fair method. Plus the same network with priors off.
set -uo pipefail
cd "$(dirname "$0")/../.."
export MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv
bash tools/search_bench/stop_servers.sh
nohup setsid python tools/search_bench/serve.py --deck FDN_exp2_imit --checkpoint gen0 --replicas 4 --base-port 50052 \
  --clients 10 --log-dir runs/search_bench/servers_imit > runs/search_bench/serve_imit.log 2>&1 < /dev/null &
for port in 50052 50152 50252 50352; do
  for k in $(seq 1 120); do curl -s localhost:$port/healthz > /dev/null && break; sleep 2; done
done
P=50052,50152,50252,50352
R="python tools/search_bench/run.py --items data/search_bench/sb-v1 --split test --evaluator remote --ports $P --workers 32 --heap 2g --net imit"
$R --out runs/search_bench/imit_network --grid priors --budgets 100,300,1000 --methods pimc1
$R --out runs/search_bench/imit_network --grid e2 --budgets 1000 --methods pimc1
echo "plan_imit done"
