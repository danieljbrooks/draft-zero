#!/usr/bin/env bash
# docs/031 E3: loops A (from scratch on every game) and B (warm start), interleaved generation by generation so they
# get equal laptop time. Generation 1's games come from the same start network, so B reuses A's.
#   nohup caffeinate -i bash gorge/recipe/e3.sh > data/gorge/runs/r31/e3.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
R=data/gorge/runs/r31
export START=$R/e2/visits-result WIDTH=128 GAMES=${GAMES:-8000} GENS=1
for ((k = 1; k <= ${MAXGEN:-8}; k++)); do
  [ -f $R/STOP ] && { echo "STOP file"; break; }
  NAME=A MODE=scratch FIRST=$k bash gorge/recipe/loop.sh
  if [ $k = 1 ]; then mkdir -p $R/loops/B/gen1; [ -e $R/loops/B/gen1/sp ] || ln -s ../../A/gen1/sp $R/loops/B/gen1/sp; fi
  [ -f $R/STOP ] && { echo "STOP file"; break; }
  NAME=B MODE=warm FIRST=$k bash gorge/recipe/loop.sh
done
echo "E3_DONE $(date)"
