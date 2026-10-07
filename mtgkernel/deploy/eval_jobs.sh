#!/bin/bash
# eval_jobs.sh NET TAG THREADS [PART]: write jobs/eval_TAG[_PART].tsv, docs/019's ladder for one network on the eval
# pairs: the policy alone (greedy) against heuristic search at 100 and against random moves; network-guided search at
# 100, 300 and 1,000 simulations against heuristic search at 100; and both at 1,000. PART a|b splits the jobs over
# two machines. Seeds are fixed per matchup, so every network meets the same games.
set -euo pipefail
cd "$(dirname "$0")/.."
NET=$1; TAG=$2; T=$3; PART=${4:-all}
D=decks/gen_v1:decks/fixtures
P=pairs/eval_v1.tsv
a=(
"${TAG}_net_vs_mcts100	3600	--decks-dir $D --pairs $P --bot1 net:path=$NET --bot2 mcts:100 --seed 51 --threads $T --n-pairs 500"
"${TAG}_net_vs_random	1800	--decks-dir $D --pairs $P --bot1 net:path=$NET --bot2 random --seed 52 --threads $T --n-pairs 100"
"${TAG}_pmcts100_vs_mcts100	3600	--decks-dir $D --pairs $P --bot1 pmcts:100,net=$NET --bot2 mcts:100 --seed 53 --threads $T --n-pairs 250"
"${TAG}_pmcts300_vs_mcts100	3600	--decks-dir $D --pairs $P --bot1 pmcts:300,net=$NET --bot2 mcts:100 --seed 54 --threads $T --n-pairs 250"
)
b=(
"${TAG}_pmcts1000_vs_mcts100	5400	--decks-dir $D --pairs $P --bot1 pmcts:1000,net=$NET --bot2 mcts:100 --seed 55 --threads $T --n-pairs 250"
"${TAG}_pmcts1000_vs_mcts1000	5400	--decks-dir $D --pairs $P --bot1 pmcts:1000,net=$NET --bot2 mcts:1000 --seed 56 --threads $T --n-pairs 124"
)
case $PART in a) jobs=("${a[@]}");; b) jobs=("${b[@]}");; *) jobs=("${a[@]}" "${b[@]}");; esac
out=jobs/eval_${TAG}$([ "$PART" = all ] || echo "_$PART").tsv
printf '%s\n' "${jobs[@]}" > "$out"
echo "$out"
