#!/usr/bin/env bash
# docs/022's planning measurements for MageZero's graph network, on one GPU pod: on a small graph build's tables
# (build.py build/tables --graph, copied to data/imitation_graph/h5), measure
#   1. speed: inference (GPU at batches 1-128, CPU at 1 and 4 threads) and training (states a second at
#      batch 64, 128, 256), MageZero's default graph network on real decision graphs;
#   2. learning: one epoch of the GNN (configs/gnn_train.yml) and of experiment #4's MLP recipe
#      (configs/exp4_train_mlp_1ep.yml) on the same rows, quarter-epoch learning curves, then three GNN
#      epochs and two learning-rate / dropout variants. Small data (a few thousand games), so this checks
#      that the GNN learns, not how well;
#   3. serving: graph_server.py (the one-epoch GNN) on the GPU under 1-56 concurrent single-state clients
#      (search workers);
#   4. rare decisions (tools/imitation_scale/rare_eval.py) for the MLP and the two GNN runs.
# docs/022 §2 is its output on 4,000 games (a Community RTX 3090, ~75 minutes).
#   bash deploy/gnn_bench.sh [out]        (out: runs/gnn_bench)
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-runs/gnn_bench}
T=${TABLES:-data/imitation_graph/h5}
mkdir -p "$OUT"
PY=$(command -v python || command -v python3)
log() { echo "[$(date -u +%H:%M:%S)] gnn_bench: $*"; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$OUT/gpu.txt" 2>&1 || true
lscpu | grep -E 'Model name|^CPU\(s\)' > "$OUT/cpu.txt" 2>&1 || true

log "1. speed"
$PY -m draftzero.gameplay.graph_supervised bench --tables-dir "$T" --set val_rows=4000 --cpu-threads 1,4 \
    --train-steps 100 --json "$OUT/bench_b64.json" > "$OUT/bench_b64.log" 2>&1 || log "bench b64 failed"
for B in 128 256; do
  $PY -m draftzero.gameplay.graph_supervised bench --tables-dir "$T" --set val_rows=4000 --set batch_rows=$B \
      --cpu-threads 1 --seconds 2 --train-steps 60 --json "$OUT/bench_b$B.json" > "$OUT/bench_b$B.log" 2>&1 || log "bench b$B failed"
done

log "2. learning"
ROWS=$($PY -c "
import h5py, sys
n = sum(h5py.File(f'$T/{t}_train.h5', 'r')['offsets'].shape[0] - 1 for t in
        ('turnstart', 'replay_priority', 'opp_priority', 'replay_attack', 'replay_target', 'opp_block'))
print(n)")
Q=$(( ROWS / 64 / 4 ))      # quarter epochs, at 64 states a step
log "training rows $ROWS, eval every $Q steps"
$PY -m draftzero.gameplay.graph_supervised train --config configs/gnn_train.yml --tables-dir "$T" --out "$OUT/gnn_1ep" \
    --set max_epochs=1 --set warmup_steps=300 --set eval_every_steps=$Q --set data_cache=null > "$OUT/gnn_1ep.log" 2>&1 || log "gnn 1ep failed"
$PY -m draftzero.gameplay.supervised train --config configs/exp4_train_mlp_1ep.yml --tables-dir "$T" --out "$OUT/mlp_1ep" \
    --set warmup_steps=300 --set eval_every_steps=$(( ROWS / 38 / 4 )) --set data_cache=null > "$OUT/mlp_1ep.log" 2>&1 || log "mlp 1ep failed"
$PY -m draftzero.gameplay.graph_supervised train --config configs/gnn_train.yml --tables-dir "$T" --out "$OUT/gnn_3ep" \
    --set max_epochs=3 --set warmup_steps=300 --set eval_every_steps=$Q --set data_cache=null > "$OUT/gnn_3ep.log" 2>&1 || log "gnn 3ep failed"
for V in "3.0e-4" "1.0e-3"; do
  $PY -m draftzero.gameplay.graph_supervised train --config configs/gnn_train.yml --tables-dir "$T" --out "$OUT/gnn_1ep_lr${V/.0/}_d0.1" \
      --set max_epochs=1 --set warmup_steps=300 --set eval_every_steps=$Q --set lr=$V --set arch.dropout=0.1 \
      --set data_cache=null > "$OUT/gnn_1ep_lr${V/.0/}_d0.1.log" 2>&1 || log "variant lr $V failed"
done

log "3. serving"
$PY tools/imitation_scale/graph_server.py --model "$OUT/gnn_1ep/best_policy.pt.gz" --port 50062 > "$OUT/graph_server.log" 2>&1 &
SRV=$!
for k in $(seq 1 90); do curl -s localhost:50062/healthz > /dev/null && break; sleep 2; done
$PY tools/imitation_scale/graph_load.py --tables-dir "$T" --port 50062 --clients 1,8,28,56 --seconds 20 \
    --json "$OUT/load.json" > "$OUT/load.log" 2>&1 || log "load test failed"
kill $SRV 2>/dev/null

log "4. rare decisions"
$PY tools/imitation_scale/rare_eval.py "$OUT/mlp_1ep/best_policy.pt.gz" "$OUT/gnn_1ep/best_policy.pt.gz" \
    "$OUT/gnn_3ep/best_policy.pt.gz" --out "$OUT/rare_eval" --tables-dir "$T" --rows 20000 \
    --ref-vocab "$OUT/mlp_1ep/best_policy.pt.gz" > "$OUT/rare_eval.log" 2>&1 || log "rare eval failed"
log "done"
echo "GNN_BENCH_DONE"
