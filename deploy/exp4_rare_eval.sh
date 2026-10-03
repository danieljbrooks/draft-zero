#!/bin/bash
# After the MLP follow-up waves (configs/exp4_sweep_rare_a.yml, _b, _k3, _k1, _b2) finish on the second machine: the rare-decision
# evaluation (tools/imitation_scale/rare_eval.py) of every run and of the baselines (the 10% and 30% best MLPs, stage
# 3's transformer), and how far each run's feature embeddings moved from their init (tools/imitation_scale/emb_drift.py).
# Waits (up to 6 hours) for every run's summary.json, then evaluates what exists; finished outputs are skipped on a rerun.
#   nohup bash deploy/exp4_rare_eval.sh [GPU] >> runs/exp4/rare_eval.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
GPU=${1:-0}
export MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv PATH=~/venv/bin:$PATH
OUT=runs/exp4/rare_eval; mkdir -p $OUT
RUNS=$(python -c "
import yaml
for w in ('a', 'b', 'k3', 'k1', 'b2'):
    for r in yaml.safe_load(open(f'configs/exp4_sweep_rare_{w}.yml'))['runs']: print(f'runs/exp4/sweep_rare_{w}/runs/' + r['name'])
")
for i in $(seq 1 72); do
  left=0; for r in $RUNS; do [ -f $r/summary.json ] || left=$((left + 1)); done
  [ $left -eq 0 ] && break
  echo "[$(date -u +%H:%M)] waiting: $left runs unfinished"; sleep 300
done
CKS=""
for c in runs/exp4/sweep_mlp2/runs/m4-best runs/exp4/sweep_mlp_s30/runs/s30-mlp-m3best-w1024 runs/exp4/main $RUNS; do
  n=$(basename $c); [ $n = main ] && n=main
  [ -f $c/best_policy.pt.gz ] && [ ! -f $OUT/$n-best_policy.json ] && CKS="$CKS $c/best_policy.pt.gz"
done
echo "[$(date -u +%H:%M)] rare_eval on:$CKS"
[ -n "$CKS" ] && CUDA_VISIBLE_DEVICES=$GPU python tools/imitation_scale/rare_eval.py $CKS --out $OUT --rows 20000 2>&1 | grep -v -i -E "warning|findfont"
for r in runs/exp4/sweep_mlp2/runs/m4-best $RUNS; do
  n=$(basename $r)
  case $n in *from-transformer) continue;; esac          # its init is the transformer's rows, not the keyed init
  [ -f $r/best_policy.pt.gz ] && [ ! -f $OUT/drift-$n.txt ] && \
    CUDA_VISIBLE_DEVICES=$GPU python tools/imitation_scale/emb_drift.py $r/best_policy.pt.gz 0.02 2>&1 | grep -v -i -E "warning|findfont" > $OUT/drift-$n.txt
done
echo "[$(date -u +%H:%M)] rare_eval done"
