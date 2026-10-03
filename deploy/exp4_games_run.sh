#!/usr/bin/env bash
# One of experiment #4's game runs on a pod (docs/018, phase C), after deploy/exp4_games_setup.sh: <replicas>
# inference servers of the network on the GPU (value_server.py --policy), the belief service for closed
# decklists, then tools/imitation_scale/play.py into runs/exp4/games/<name>. Its results go to the HF repo
# (exp4/games/runs/<name>/) every 10 minutes and at the end; a stopped run resumes where it left off.
#   HF_TOKEN=<write token> bash deploy/exp4_games_run.sh <name> <network> <replicas> <play.py args...>
#   e.g. bash deploy/exp4_games_run.sh c2-ilbc300-s0 models/exp4/stage3/best_policy.pt.gz 4 \
#          --bot1 il_bc@300 --bot2 heuristic@100 --pairs 50 --workers 30 --heap 1500m --shard 0/2
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=$1 MODEL=$2 REPLICAS=$3; shift 3
export MZ_XMAGE_DIR=$PWD/xmage MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv HF_HUB_DISABLE_PROGRESS_BARS=1
OUT=runs/exp4/games/$NAME
mkdir -p "$OUT/logs"
log() { echo "[$(date -u +%H:%M:%S)] games $NAME: $*"; }

PORTS=""
for i in $(seq 0 $((REPLICAS - 1))); do
  p=$((50052 + 100 * i)); PORTS="${PORTS:+$PORTS,}$p"
  curl -s -m 2 "localhost:$p/healthz" > /dev/null && continue
  nohup python tools/search_bench/value_server.py --model "$MODEL" --port "$p" --threads 8 --policy \
    > "$OUT/logs/server_$p.log" 2>&1 < /dev/null &
done
for p in ${PORTS//,/ }; do
  for k in $(seq 1 120); do curl -s -m 2 "localhost:$p/healthz" > /dev/null && break; sleep 2; done
done
if ! echo " $* " | grep -q -- " --open-decklists "; then
  curl -s -m 2 localhost:50070/healthz > /dev/null || \
    nohup python tools/imitation_scale/belief_server.py --port 50070 > "$OUT/logs/belief.log" 2>&1 < /dev/null &
  for k in $(seq 1 120); do curl -s -m 2 localhost:50070/healthz > /dev/null && break; sleep 2; done
fi
log "servers on $PORTS; playing: $*"

upload() {
  [ -n "${HF_TOKEN:-}" ] || return 0
  python - "$OUT" "exp4/games/runs/$NAME" <<'PY' || true
import os, sys
from huggingface_hub import HfApi
out, dst = sys.argv[1], sys.argv[2]
HfApi(token=os.environ["HF_TOKEN"]).upload_folder(repo_id="danbrooks/draftzero-checkpoints", folder_path=out,
    path_in_repo=dst, allow_patterns=["games.jsonl", "config.json", "summary.json", "play.log"],
    commit_message=f"exp4 games: {dst}")
PY
}
( while sleep 600; do upload > /dev/null 2>&1; done ) &
UPLOADER=$!
python tools/imitation_scale/play.py --deck-root data/deckgen/FDN_PremierDraft_wr60/top_player_FDN_decks \
  --ports "$PORTS" --out "$OUT" "$@" 2>&1 | tee -a "$OUT/play.log"
STATUS=${PIPESTATUS[0]}
kill $UPLOADER 2>/dev/null
upload
log "done (play.py exit $STATUS)"
exit "$STATUS"
