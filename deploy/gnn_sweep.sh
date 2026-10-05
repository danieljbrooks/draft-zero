#!/usr/bin/env bash
# docs/022 stage 2 on one GPU: some of configs/gnn_sweep.yml's runs (the sweep is split between machines with --only,
# each writing its own runs/ directory; finished runs are read back, a stopped one resumes).
#   bash deploy/gnn_sweep.sh <gpu> <run,run,...> [tables dir]
# The tables default to data/imitation_graph/h5 (the build's own); on a machine without them, the slim tables
# (graph files + labels) come from the HF repo into data/imitation_graph/slim, given HF_TOKEN in the environment.
# Env: OUT (runs/gnn/sweep), SPEC (configs/gnn_sweep.yml), SETS (extra --set arguments, e.g. "data_cache=null").
set -uo pipefail
cd "$(dirname "$0")/.."
GPU=$1 ONLY=$2 T=${3:-data/imitation_graph/h5}
OUT=${OUT:-runs/gnn/sweep}
SPEC=${SPEC:-configs/gnn_sweep.yml}
PY=$(command -v python || command -v python3)
if [ ! -f "$T/turnstart_train.graph.h5" ]; then
  : "${HF_TOKEN:?no tables at $T: set HF_TOKEN to fetch the slim ones from the HF repo}"
  T=data/imitation_graph/slim
  HF_HUB_DISABLE_PROGRESS_BARS=1 $PY - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("danbrooks/draftzero-checkpoints", allow_patterns=["exp4/tables_graph/slim/*"],
                  local_dir="data/imitation_graph/hf", token=os.environ["HF_TOKEN"])
PY
  mkdir -p "$T"
  for f in data/imitation_graph/hf/exp4/tables_graph/slim/*; do ln -sf "$PWD/$f" "$T/$(basename "$f")"; done
fi
unset HF_TOKEN       # only the download needs it: not in the long-running sweep's environment
ARGS=()
for s in ${SETS:-}; do ARGS+=(--set "$s"); done
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPU exec $PY -m draftzero.gameplay.graph_supervised sweep --spec "$SPEC" --out "$OUT" \
  --tables-dir "$T" --only "$ONLY" "${ARGS[@]}"
