#!/usr/bin/env bash
# docs/022 stage 3 on one pod: the GNN's large training (configs/gnn_train.yml) from the slim tables on the HF repo,
# then its best_policy and best_value scored on the test split (stage 4). Results go to the HF repo (gnn/main/)
# every 15 minutes and at the end; a stopped run resumes from latest.pt. The first load holds every training row in
# RAM (~100 GB at peak) before the memory-mapped cache takes over: rent ~150 GB.
#   HF_TOKEN=<write token> bash deploy/gnn_train.sh
# Env: OUT (runs/gnn/main), CONFIG (configs/gnn_train.yml), HOURS (9: the external hard kill), T (a tables dir that
# already holds the graph tables; default: the slim ones from HF).
set -uo pipefail
cd "$(dirname "$0")/.."
: "${HF_TOKEN:?set HF_TOKEN to a token that can write danbrooks/draftzero-checkpoints}"
export PYTHONPATH=$PWD/src${PYTHONPATH:+:$PYTHONPATH} PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
OUT=${OUT:-runs/gnn/main}
CONFIG=${CONFIG:-configs/gnn_train.yml}
PY=$(command -v python || command -v python3)
log() { echo "[$(TZ=America/Los_Angeles date '+%a %-I:%M %p PT')] gnn_train: $*"; }
T=${T:-data/imitation_graph/slim}
if [ ! -f "$T/turnstart_train.graph.h5" ]; then
  log "fetching the slim tables"
  $PY - <<'PY' || { log "fetch failed"; exit 2; }
import os
from huggingface_hub import snapshot_download
snapshot_download("danbrooks/draftzero-checkpoints", allow_patterns=["exp4/tables_graph/slim/*"],
                  local_dir="data/imitation_graph/hf", token=os.environ["HF_TOKEN"])
PY
  mkdir -p "$T"
  for f in data/imitation_graph/hf/exp4/tables_graph/slim/*; do ln -sf "$PWD/$f" "$T/$(basename "$f")"; done
fi
mkdir -p "$OUT"

upload() {
  $PY - "$OUT" <<'PY' || true
import os, sys
from huggingface_hub import HfApi
out = sys.argv[1]
HfApi(token=os.environ["HF_TOKEN"]).upload_folder(repo_id="danbrooks/draftzero-checkpoints", folder_path=out,
    path_in_repo="gnn/main", allow_patterns=["evals.jsonl", "summary.json", "config.json", "train.log", "test_*.json",
    "best_policy.pt.gz", "best_value.pt.gz", "final.pt.gz"], commit_message="docs/022 stage 3: the GNN's large training")
PY
}
( while sleep 900; do upload > /dev/null 2>&1; done ) &
UPLOADER=$!

RESUME=()
[ -f "$OUT/latest.pt" ] && RESUME=(--resume)
log "training ($CONFIG -> $OUT${RESUME:+, resuming})"
timeout -k 120 "${HOURS:-9}h" $PY -m draftzero.gameplay.graph_supervised train --config "$CONFIG" --out "$OUT" \
  --tables-dir "$T" "${RESUME[@]}" 2>&1 | tee -a "$OUT/train.log"
STATUS=${PIPESTATUS[0]}
log "training exit $STATUS"
for ck in best_policy best_value; do
  [ -f "$OUT/$ck.pt.gz" ] || continue
  timeout -k 60 1h $PY -m draftzero.gameplay.graph_supervised evaluate --config "$CONFIG" --checkpoint "$OUT/$ck.pt.gz" \
    --split test --tables-dir "$T" --json "$OUT/test_$ck.json" >> "$OUT/train.log" 2>&1 && log "test: $ck scored" || log "test: $ck FAILED"
done
kill $UPLOADER 2>/dev/null
upload
log "done"
echo "GNN_TRAIN_DONE status=$STATUS"
