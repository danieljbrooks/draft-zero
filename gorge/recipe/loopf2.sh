#!/usr/bin/env bash
# docs/031 loop F' (r1 down): loop F's recipe on the laptop and d1. Each generation's GAMES games of self-play at SIMS
# simulations (cpuct 0.25, every decision recorded) are split: d1 plays deck pairs 0..GAMES/2-1, the laptop the rest
# (same seed, so the union is exactly the games one machine would play). The laptop merges and packs them, warm-starts
# the next network on the last WINDOW generations (root value as the value's label) and plays it against REF head to
# head (cpuct 0.25, 100 simulations, 500 deck pairs). Resumable: finished steps are skipped.
#   START=<dir with best.pt/best.dzgw> FIRST=6 REF=<best.dzgw> bash gorge/recipe/loopf2.sh
set -uo pipefail
source "$(dirname "$0")/lib.sh"
R=data/gorge/runs/r31; L=$R/loops/F2; REF=${REF:?}; START=${START:?}; FIRST=${FIRST:?}
SIMS=${SIMS:-200}; GAMES=${GAMES:-1200}; WINDOW=${WINDOW:-3}; VB=${VB:-1}
D1=/work/dz031/wt; D1R=$D1/$R; half=$((GAMES / 2))
d1() { perl -e 'alarm shift; exec @ARGV' 600 ssh -o BatchMode=yes -o ConnectTimeout=20 d1 "$@"; }
mkdir -p $L/gen$((FIRST - 1))
for f in best.pt best.dzgw; do [ -e $L/gen$((FIRST - 1))/$f ] || cp $START/$f $L/gen$((FIRST - 1))/; done
for ((k = FIRST; k <= ${MAXGEN:-100}; k++)); do
  [ -f $R/STOP_F2 ] && { echo "STOP_F2"; break; }
  prev=$L/gen$((k - 1)); cur=$L/gen$k; seed=$((7800 + k))
  echo "== F' gen $k $(date)"
  if [ ! -d $cur/sp/pack ]; then
    # d1: the first half (its own selfplay, unpacked: the laptop packs the union)
    d1 "mkdir -p $D1R/loops/F2/gen$((k - 1)) $D1R/loops/F2/gen$k/sp"
    perl -e 'alarm 300; exec @ARGV' scp -q -o BatchMode=yes $prev/best.dzgw d1:$D1R/loops/F2/gen$((k - 1))/best.dzgw
    d1 "cd $D1 && tmux kill-session -t f2 2>/dev/null; tmux new-session -d -s f2 'bash gorge/recipe/f2_half.sh $k $seed $SIMS $half > $D1R/loops/F2/gen$k/d1.log 2>&1'"
    # the laptop: the second half
    mkdir -p $cur/lap
    [ -f $cur/lap/next ] || echo $half > $cur/lap/next
    p="az:sims=$SIMS:explore:$FULL:cpuct=0.25:topk=6:gonet=$prev/best.dzgw"
    s=$(cat $cur/lap/next)
    while [ $s -lt $GAMES ]; do
      e=$(( s + 100 < GAMES ? s + 100 : GAMES ))
      m=$(RECORD_EVERY=1 _chunk $cur/lap $s $((e - s)) $seed "$p" 7200)
      s=$((s + m)); [ $s -lt $e ] && s=$((s + 1)); echo $s > $cur/lap/next
    done
    until d1 "test -f $D1R/loops/F2/gen$k/d1/DONE"; do sleep 120; done
    mkdir -p $cur/sp $cur/d1
    d1 "cd $D1R/loops/F2/gen$k/d1 && tar -cf - games.jsonl visits.jsonl.gz" | tar -xf - -C $cur/d1
    cat $cur/d1/games.jsonl $cur/lap/games.jsonl > $cur/sp/games.jsonl
    cat $cur/d1/visits.jsonl.gz $cur/lap/visits.jsonl.gz > $cur/sp/visits.jsonl.gz
    "$G/bin/dzgorge" pack -in $cur/sp/visits.jsonl.gz -out $cur/sp/pack -shard 200000 > $cur/sp/pack.log 2>&1 \
      && "$PY" gorge/recipe/shrink.py $cur/sp/pack > /dev/null
  fi
  echo "   $(wc -l < $cur/sp/games.jsonl) games $(date)"
  packs=(); for ((j = k - WINDOW + 1; j <= k; j++)); do [ -d $L/gen$j/sp/pack ] && packs+=($L/gen$j/sp/pack); done
  train $cur 128 14400 --train "${packs[@]}" --init $prev/best.pt --epochs 2 --lr 1e-4 --warmup 100 --patience 4 \
    --policy-target visits --value-blend $VB
  match $cur/vsref.jsonl 100 500 "az:sims=100:$FULL:topk=6:cpuct=0.25:gonet=$cur/best.dzgw" \
    "az:sims=100:$FULL:topk=6:cpuct=0.25:gonet=$REF"
  echo "   F'$k vs reference: $(score $cur/vsref.jsonl) $(date)"
done
echo LOOPF2_DONE
