#!/usr/bin/env bash
# Extra budget: IS-MCTS offline at 10,000 simulations (it was still gaining at 3,000). The 300 run
# first warms the JVMs so the 10,000 timing is clean.
set -uo pipefail
cd "$(dirname "$0")/../.."
export MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv
python tools/search_bench/run.py --items data/search_bench/sb-v1 --split test --evaluator offline --workers 30 --heap 3g \
  --out runs/search_bench/e2_offline_is10k --grid e2 --budgets 300,10000 --methods ismcts
echo "plan_is10k done"
