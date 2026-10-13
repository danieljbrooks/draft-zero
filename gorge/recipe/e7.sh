#!/usr/bin/env bash
# docs/031 E7: is a deeper search's root value a better value label than the game's result?
# Self-play with NET at SIMS simulations (cpuct 0.25, every decision recorded) for GAMES games; then three warm starts
# from NET on those games only (2 passes, lr 1e-4): the result, half and half, the root value alone; each against NET
# head to head (cpuct 0.25, 100 simulations, 500 deck pairs).
set -uo pipefail
source "$(dirname "$0")/lib.sh"
R=data/gorge/runs/r31; O=${OUT:-$R/e7}; NET=${NET:?dir with best.pt and best.dzgw}; SIMS=${SIMS:-200}; GAMES=${GAMES:-1200}
mkdir -p $O
RECORD_EVERY=1 SP_EXTRA=":cpuct=0.25" CHUNK=100 CHUNK_SECS=7200 selfplay $O/sp $GAMES 7700 $NET/best.dzgw $SIMS
echo "self-play: $(wc -l < $O/sp/games.jsonl) games $(date)"
for vb in ${VBS:-0 0.5 1}; do
  train $O/vb$vb 128 7200 --train $O/sp/pack --init $NET/best.pt --epochs 2 --lr 1e-4 --warmup 100 --patience 4 \
    --policy-target visits --value-blend $vb
  echo "vb$vb: $(grep -o '"best_step": [0-9]*' $O/vb$vb/train.log)"
done
for vb in ${VBS:-0 0.5 1}; do
  match $O/vb$vb-vs-net.jsonl 100 500 "az:sims=100:$FULL:topk=6:cpuct=0.25:gonet=$O/vb$vb/best.dzgw" \
    "az:sims=100:$FULL:topk=6:cpuct=0.25:gonet=$NET/best.dzgw"
  echo "value blend $vb vs NET: $(score $O/vb$vb-vs-net.jsonl) $(date)"
done
echo E7_DONE
