#!/usr/bin/env bash
# Extra budget: the best offline method at 10,000 simulations (docs/012 §2.7's optional point),
# where the offline curves flatten. 25 workers with 4 GB heaps (bigger trees).
set -uo pipefail
cd "$(dirname "$0")/../.."
export MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv
bash tools/search_bench/stop_servers.sh
python tools/search_bench/run.py --items data/search_bench/sb-v1 --split test --evaluator offline --workers 25 --heap 4g \
  --out runs/search_bench/e2_offline_10k --grid e2 --budgets 300,10000 --methods pimc1
echo "plan_10k done"
