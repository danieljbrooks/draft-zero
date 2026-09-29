#!/usr/bin/env bash
# The first experiment's network half, part two: budget 3,000, then E2b with the discounts matched
# from E2's network clairvoyant 1,000 run. Servers: start_servers.sh gen18 4 first.
set -uo pipefail
cd "$(dirname "$0")/../.."
export MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv
P=${P:-50052,50152,50252,50352}
R="python tools/search_bench/run.py --items data/search_bench/sb-v1 --split test --evaluator remote --ports $P --workers 32 --heap 2g"
$R --out runs/search_bench/e2_network --grid e2 --budgets 3000
read DA DT < <(python tools/search_bench/match_discount.py runs/search_bench/e2_network clairvoyant-b1000-remote-d0.99)
echo "E2b network: per-action $DA, per-turn $DT"
$R --out runs/search_bench/e2b_network --grid e2b --d-action $DA --d-turn $DT
echo "plan_net_b done"
