#!/usr/bin/env bash
# docs/028 on a rented pod (after deploy/exp4_games_setup.sh): play one match with deploy/sp_h2h.sh and mirror its
# results to the HF repo (sp028/h2h/<name>/) every 10 minutes and at the end, so a pod that self-destructs loses at
# most 10 minutes of games. Checkpoints the match needs that aren't on HF are copied in beforehand (scp).
#   HF_TOKEN=<write token> bash deploy/sp_pod_job.sh <name> <ckpt A> <ckpt B|-> <replicas each> <play.py args...>
# The worker count comes from deploy/pod_workers.py at 3 GB a worker unless the args give --workers.
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=$1; shift
OUT=runs/sp/h2h/$NAME
upload() {
  [ -n "${HF_TOKEN:-}" ] || return 0
  python3 - "$OUT" "sp028/h2h/$NAME" <<'PY' || true
import os, sys
from huggingface_hub import HfApi
out, dst = sys.argv[1], sys.argv[2]
if os.path.isdir(out):
    HfApi(token=os.environ["HF_TOKEN"]).upload_folder(repo_id="danbrooks/draftzero-checkpoints", folder_path=out,
        path_in_repo=dst, allow_patterns=["games.jsonl", "config.json", "summary.json", "play.log", "networks.txt"],
        commit_message=f"docs/028 match: {dst}")
PY
}
( while sleep 600; do upload > /dev/null 2>&1; done ) &
UP=$!
W=()
echo " $* " | grep -q -- " --workers " || W=(--workers "$(python3 deploy/pod_workers.py 3)")
bash deploy/sp_h2h.sh "$NAME" "$@" "${W[@]}"
STATUS=$?
kill $UP 2>/dev/null
upload
python3 tools/selfplay_transition/paired.py "$OUT" || true
echo "SP_POD_JOB_DONE $NAME $STATUS"
