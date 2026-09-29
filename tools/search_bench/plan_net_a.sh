#!/usr/bin/env bash
# Pod 1 of the first experiment: network runs, budgets 100-1,000, then the E0 network references.
set -uo pipefail
cd "$(dirname "$0")/../.."
export MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv
P=50052,50152,50252,50352
R="python tools/search_bench/run.py --items data/search_bench/sb-v1 --split test --evaluator remote --ports $P --workers 32 --heap 2g"
$R --out runs/search_bench/e2_network --grid e2 --budgets 100,300,1000
$R --out runs/search_bench/e0_network --grid e0 --only pimc4-b1000-remote-d0.99-s1
echo "plan_net_a done"
