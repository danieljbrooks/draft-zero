#!/usr/bin/env bash
# The first experiment's network half, part two: budget 3,000 for the four methods.
# Servers: start_servers.sh gen18 4 first.
set -uo pipefail
cd "$(dirname "$0")/../.."
export MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv
P=${P:-50052,50152,50252,50352}
R="python tools/search_bench/run.py --items data/search_bench/sb-v1 --split test --evaluator remote --ports $P --workers 32 --heap 2g"
$R --out runs/search_bench/e2_network --grid e2 --budgets 3000
echo "plan_net_b done"
