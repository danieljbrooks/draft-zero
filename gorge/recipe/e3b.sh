#!/usr/bin/env bash
# docs/031 E3, from generation FROM on: loops A and B interleaved as in e3.sh, each generation also against gorge's bot;
# after each round, the next level of the search ladder (ladder.sh) for LADDER_NET: LEVELS, about x5 apart (Dan).
set -uo pipefail
cd "$(dirname "$0")/../.."
R=data/gorge/runs/r31
export START=$R/e2/visits-result WIDTH=128 GAMES=${GAMES:-8000} GENS=1
levels=(${LEVELS:-100 500 3000 15000})
li=0
for ((k = ${FROM:-1}; k <= ${MAXGEN:-8}; k++)); do
  [ -f $R/STOP ] && { echo "STOP file"; break; }
  NAME=A MODE=scratch FIRST=$k bash gorge/recipe/loop2.sh
  if [ $k = 1 ]; then mkdir -p $R/loops/B/gen1; [ -e $R/loops/B/gen1/sp ] || ln -s ../../A/gen1/sp $R/loops/B/gen1/sp; fi
  [ -f $R/STOP ] && { echo "STOP file"; break; }
  NAME=B MODE=warm FIRST=$k bash gorge/recipe/loop2.sh
  [ -f $R/STOP ] && { echo "STOP file"; break; }
  if [ $li -lt ${#levels[@]} ]; then
    lad=${levels[$li]}; li=$(( li + 1 ))
    NET=${LADDER_NET:-$R/e2/visits-result} NAME=${LADDER_NAME:-start} SIMS=$lad bash gorge/recipe/ladder.sh
  fi
done
echo "E3_DONE $(date)"
