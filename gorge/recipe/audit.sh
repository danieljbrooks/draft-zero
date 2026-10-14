#!/usr/bin/env bash
# docs/031 S3: audit agents against an oracle (the same network, ORACLE_SIMS simulations, no top-k cut), playing
# gorge's bot: a quarter of each agent's searched decisions, PAIRS deck pairs per agent.
#   NET=<best.dzgw> OUT=<dir> WORKERS=4 bash gorge/recipe/audit.sh
set -uo pipefail
source "$(dirname "$0")/lib.sh"
NET=${NET:?}; OUT=${OUT:?}; PAIRS=${PAIRS:-100}; OS=${ORACLE_SIMS:-2000}; RATE=${RATE:-0.25}
mkdir -p $OUT
oracle="az:sims=$OS:$FULL:gonet=$NET"
agents=${AGENTS:-"nonet100=az:sims=100:$FULL net25=az:sims=25:$FULL:topk=6:gonet=$NET net100=az:sims=100:$FULL:topk=6:gonet=$NET net100c05=az:sims=100:$FULL:topk=6:cpuct=0.5:gonet=$NET net500=az:sims=500:$FULL:topk=6:gonet=$NET"}
for spec in $agents; do
  name=${spec%%=*}; pol=${spec#*=}
  [ -f $OUT/$name.done ] && continue
  hardkill 28800 "$G/bin/dzgorge" play "${COMMON[@]}" -split eval -seed 3131 -pairs $PAIRS -workers "${WORKERS:-4}" \
    -a "$pol" -b bot -out $OUT/$name.games.jsonl -audit-oracle "$oracle" -audit-rate $RATE \
    -audit-out $OUT/$name.audit.jsonl > $OUT/$name.log 2>&1 && touch $OUT/$name.done
  echo "== $name: $(score $OUT/$name.games.jsonl) vs bot $(date)"
  $PY gorge/recipe/audit_report.py $OUT/$name.audit.jsonl
done
echo AUDIT_DONE
