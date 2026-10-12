#!/usr/bin/env bash
# docs/031 E6: capacity at scale. MLPs of widths 128 / 256 / 384, from scratch, on every game of loop B's lineage (stage 0's
# 80,000 plus generations 1..LAST), then each against loop B's generation LAST head to head (cpuct 0.25, 100 simulations).
set -uo pipefail
source "$(dirname "$0")/lib.sh"
R=data/gorge/runs/r31; L=$R/loops/B; O=$R/e6; LAST=${LAST:?}
packs=($R/g0/pack $R/g0x/pack); for ((j = 1; j <= LAST; j++)); do packs+=($L/gen$j/sp/pack); done
for w in ${WIDTHS:-128 256 384}; do
  echo "== w$w on ${#packs[@]} packs $(date)"
  train $O/w$w $w 21600 --train "${packs[@]}" --epochs 6 --patience 6 --lr 3e-4 --policy-target visits --value-blend 0
  tail -1 $O/w$w/train.log | cut -c1-300
done
for w in ${WIDTHS:-128 256 384}; do
  match $O/w$w-vs-B$LAST.jsonl 100 500 "az:sims=100:$FULL:topk=6:cpuct=0.25:gonet=$O/w$w/best.dzgw" \
    "az:sims=100:$FULL:topk=6:cpuct=0.25:gonet=$L/gen$LAST/best.dzgw"
  echo "w$w vs B$LAST: $(score $O/w$w-vs-B$LAST.jsonl) $(date)"
done
echo E6_DONE
