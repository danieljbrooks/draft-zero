#!/usr/bin/env bash
# The network half, part three (the second pod, after plan_offline.sh): E2b with the discounts
# matched from the network's clairvoyant 1,000 run, the leak test's network spot check, then
# the priors follow-up on MageZero's own (policy-serving) server.
set -uo pipefail
cd "$(dirname "$0")/../.."
export MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv
bash tools/search_bench/start_servers.sh models/FDN_exp2/ver1/gen18.pt.gz 4
P=50052,50152,50252,50352
R="python tools/search_bench/run.py --items data/search_bench/sb-v1 --split test --evaluator remote --ports $P --workers 32 --heap 2g"
read DA DT < <(python tools/search_bench/match_discount.py runs/search_bench/e2_network_ref clairvoyant-b1000-remote-d0.99)
echo "E2b network: per-action $DA, per-turn $DT"
$R --out runs/search_bench/e2b_network --grid e2b --d-action $DA --d-turn $DT
python tools/search_bench/leak.py run --out runs/search_bench/e1_network --workers 32 --heap 2g --seeds 16 --budget 3000 \
  --probes counterspell,cantrip,counterspell_x,cantrip_x --methods clairvoyant,pimc4 --evaluator remote --port 50052
python tools/search_bench/leak.py analyze --out runs/search_bench/e1_network > runs/search_bench/e1_network/analyze.txt
echo "plan_net_c done"
