#!/usr/bin/env bash
# docs/031 E3, round-robin over several loops, one generation each per round, from generation FROM:
#   LOOPS="A:scratch:all B:warm:all D:scratch:recent4"  NAME:MODE:WINDOW
#     all      stage 0's 80,000 games plus every generation's (BASE default, WINDOW 20)
#     recentN  the loop's last N generations only (no stage-0 games)
# A loop that starts later (D) has its first generations linked to another loop's (see the doc's run log).
set -uo pipefail
cd "$(dirname "$0")/../.."
R=data/gorge/runs/r31
export START=$R/e2/visits-result WIDTH=128 GAMES=${GAMES:-8000} GENS=1
for ((k = ${FROM:?}; k <= ${MAXGEN:-40}; k++)); do
  for spec in ${LOOPS:?}; do
    IFS=: read -r name mode win <<< "$spec"
    [ -f $R/STOP ] && { echo "STOP file"; exit 0; }
    if [ "$win" = all ]; then
      NAME=$name MODE=$mode FIRST=$k bash gorge/recipe/loop2.sh
    else
      NAME=$name MODE=$mode FIRST=$k BASE=" " WINDOW=${win#recent} bash gorge/recipe/loop2.sh
    fi
  done
done
echo "E3_DONE $(date)"
