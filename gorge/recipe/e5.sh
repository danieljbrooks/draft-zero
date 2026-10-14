#!/usr/bin/env bash
# docs/031 E5: targets on the loop's own games. Warm starts from loop B's generation 4, trained on B's generations 1-4
# (32,000 games played with a network: their root values come from the network's search), then 1,000 games against
# gorge's bot at 100 simulations each.
set -uo pipefail
source "$(dirname "$0")/lib.sh"
R=data/gorge/runs/r31; L=$R/loops/B; O=$R/e5
packs=($L/gen1/sp/pack $L/gen2/sp/pack $L/gen3/sp/pack $L/gen4/sp/pack)
arm() {
  local name=$1; shift
  echo "== $name $(date)"
  train $O/$name 128 7200 --train "${packs[@]}" --init $L/gen4/best.pt --epochs 2 --lr 1e-4 --warmup 100 --patience 4 "$@"
  vsbot $O/$name/best.dzgw $O/$name/vsbot 100 500
  echo "   $name vs bot at 100: $(score $O/$name/vsbot.jsonl) $(date)"
}
arm ctrl  --policy-target visits --value-blend 0
arm vt05  --policy-target visits --visits-temp 0.5 --value-blend 0
arm cq03  --policy-target cq --cq-c-scale 0.03 --value-blend 0
arm vb25  --policy-target visits --value-blend 0.25
arm vb50  --policy-target visits --value-blend 0.5
echo "E5_DONE $(date)"
