#!/usr/bin/env bash
# docs/022 stage 5: one game run on a pod, after deploy/exp4_games_setup.sh (the bundle, decks, belief data): <replicas>
# graph servers of the GNN on the GPUs (tools/imitation_scale/graph_server.py; one process feeds ~650 states a second,
# docs/022 §2.2, so a 28-worker run wants 2-3), value servers of a flat network too when MLP is set (for gnn against
# mlp), the belief service for closed decklists, then play.py into runs/gnn/games/<name>. Results go to the HF repo
# (gnn/games/runs/<name>/) every 10 minutes and at the end; a stopped run resumes where it left off.
#   HF_TOKEN=<write token> bash deploy/gnn_games_run.sh <name> <gnn checkpoint> <replicas> <play.py args...>
#   e.g. bash deploy/gnn_games_run.sh gnn100 models/gnn/main/best_policy.pt.gz 3 \
#          --bot1 gnn@100 --bot2 heuristic@100 --pairs 50 --workers 28 --heap 2500m
#        MLP=models/exp4/mlp_1ep/best_policy.pt.gz bash deploy/gnn_games_run.sh gnn100-mlp100 <gnn> 2 \
#          --bot1 gnn@100 --bot2 il_bc@100 --pairs 50 --workers 28 --heap 2500m
# Env: MLP (a flat checkpoint for the il_bc / policy bots), MLP_REPLICAS (2), VALUE_MODEL (the graph servers take the
# value from this checkpoint instead: graph_server.py --value-model).
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=$1 MODEL=$2 REPLICAS=$3; shift 3
export MZ_XMAGE_DIR=$PWD/xmage MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv HF_HUB_DISABLE_PROGRESS_BARS=1
OUT=runs/gnn/games/$NAME
mkdir -p "$OUT/logs"
log() { echo "[$(TZ=America/Los_Angeles date '+%a %-I:%M %p PT')] gnn games $NAME: $*"; }

wait_up() { for k in $(seq 1 120); do curl -s -m 2 "localhost:$1/healthz" > /dev/null && return 0; sleep 2; done; return 1; }
GPORTS=""
NGPU=$(nvidia-smi -L 2>/dev/null | grep -c ^GPU); [ "$NGPU" -ge 1 ] || NGPU=1
for i in $(seq 0 $((REPLICAS - 1))); do
  p=$((50062 + 100 * i)); GPORTS="${GPORTS:+$GPORTS,}$p"
  curl -s -m 2 "localhost:$p/healthz" > /dev/null && continue
  # replicas round-robin over the machine's GPUs (r1 has two)
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$((i % NGPU))} nohup python tools/imitation_scale/graph_server.py --model "$MODEL" --port "$p" --threads 16 \
    ${VALUE_MODEL:+--value-model "$VALUE_MODEL"} > "$OUT/logs/graph_server_$p.log" 2>&1 < /dev/null &
done
PORTS=""
if [ -n "${MLP:-}" ]; then
  for i in $(seq 0 $((${MLP_REPLICAS:-2} - 1))); do
    p=$((50052 + 100 * i)); PORTS="${PORTS:+$PORTS,}$p"
    curl -s -m 2 "localhost:$p/healthz" > /dev/null && continue
    nohup python tools/search_bench/value_server.py --model "$MLP" --port "$p" --threads 8 --policy \
      > "$OUT/logs/server_$p.log" 2>&1 < /dev/null &
  done
fi
for p in ${GPORTS//,/ } ${PORTS//,/ }; do wait_up "$p" || { log "server on $p never came up"; exit 2; }; done
if ! echo " $* " | grep -q -- " --open-decklists "; then
  curl -s -m 2 localhost:50070/healthz > /dev/null || \
    nohup python tools/imitation_scale/belief_server.py --port 50070 > "$OUT/logs/belief.log" 2>&1 < /dev/null &
  wait_up 50070 || { log "belief service never came up"; exit 2; }
fi
log "graph servers on $GPORTS${PORTS:+, flat servers on $PORTS}; playing: $*"

upload() {
  [ -n "${HF_TOKEN:-}" ] || return 0
  python - "$OUT" "gnn/games/runs/$NAME" <<'PY' || true
import os, sys
from huggingface_hub import HfApi
out, dst = sys.argv[1], sys.argv[2]
HfApi(token=os.environ["HF_TOKEN"]).upload_folder(repo_id="danbrooks/draftzero-checkpoints", folder_path=out,
    path_in_repo=dst, allow_patterns=["games.jsonl", "config.json", "summary.json", "play.log"],
    commit_message=f"gnn games: {dst}")
PY
}
( while sleep 600; do upload > /dev/null 2>&1; done ) &
UPLOADER=$!
python tools/imitation_scale/play.py --deck-root data/deckgen/FDN_PremierDraft_wr60/top_player_FDN_decks \
  --graph-ports "$GPORTS" ${PORTS:+--ports "$PORTS"} --out "$OUT" "$@" 2>&1 | tee -a "$OUT/play.log"
STATUS=${PIPESTATUS[0]}
kill $UPLOADER 2>/dev/null
upload
log "done (play.py exit $STATUS)"
exit "$STATUS"
