#!/usr/bin/env bash
# The first experiment's offline half: E1 (leak test), E2 offline (4 methods x 4 budgets), E0's
# second seed, then E2b offline with the discounts matched from E2's clairvoyant 1,000 run.
set -uo pipefail
cd "$(dirname "$0")/../.."
export MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv
W=${W:-30}
python tools/search_bench/leak.py run --out runs/search_bench/e1 --workers $W --seeds 16 --budget 3000
python tools/search_bench/leak.py analyze --out runs/search_bench/e1 > runs/search_bench/e1/analyze.txt
R="python tools/search_bench/run.py --items data/search_bench/sb-v1 --split test --evaluator offline --workers $W"
$R --out runs/search_bench/e2_offline --grid e2 --budgets 100,300,1000
$R --out runs/search_bench/e0_offline --grid e0
read DA DT < <(python tools/search_bench/match_discount.py runs/search_bench/e2_offline clairvoyant-b1000-offline-d0.99)
echo "E2b offline: per-action $DA, per-turn $DT"
$R --out runs/search_bench/e2b_offline --grid e2b --d-action $DA --d-turn $DT
$R --out runs/search_bench/e2_offline --grid e2 --budgets 3000
echo "plan_offline done"
