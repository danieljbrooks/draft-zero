#!/bin/bash
# Queue MLP follow-up sweeps on one GPU of the second machine (memory: two 10% sweeps fill most of its 40 GB, so at
# most one per GPU): wait for every run of AFTER's spec to finish, stop that sweep's idle --follow loop (its STOP
# file), then run each SPEC in turn (configs/exp4_sweep_rare_<spec>.yml -> runs/exp4/sweep_rare_<spec>).
#   nohup bash deploy/exp4_rare_chain.sh GPU AFTER SPEC... >> runs/exp4/rare_chain_GPU.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
GPU=$1 AFTER=$2; shift 2
export MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv PATH=~/venv/bin:$PATH
RUNS=$(python -c "
import yaml
for r in yaml.safe_load(open('configs/exp4_sweep_rare_$AFTER.yml'))['runs']: print('runs/exp4/sweep_rare_$AFTER/runs/' + r['name'])
")
for i in $(seq 1 120); do
  left=0; for r in $RUNS; do [ -f $r/summary.json ] || left=$((left + 1)); done
  [ $left -eq 0 ] && break
  echo "[$(date -u +%H:%M)] waiting for sweep_rare_$AFTER: $left runs unfinished"; sleep 120
done
touch runs/exp4/sweep_rare_$AFTER/STOP
for i in $(seq 1 30); do pgrep -f "spec configs/exp4_sweep_rare_$AFTER.yml" > /dev/null || break; sleep 20; done
for s in "$@"; do
  echo "[$(date -u +%H:%M)] starting sweep_rare_$s on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU python -m draftzero.gameplay.supervised sweep --spec configs/exp4_sweep_rare_$s.yml \
    --out runs/exp4/sweep_rare_$s --tables-dir data/imitation_scale/h5 >> runs/exp4/sweep_rare_$s.log 2>&1
  echo "[$(date -u +%H:%M)] sweep_rare_$s done"
done
