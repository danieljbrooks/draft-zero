#!/usr/bin/env bash
# docs/022 stage 1 on one pod: every top player's game rebuilt with graph encodings, the tables, the check that the
# flat tables equal experiment #4's game by game, and the slim tables (graph files + labels) uploaded to the HF repo
# (exp4/tables_graph/slim/) for the GNN's training elsewhere.
#   HF_TOKEN=<write token> bash deploy/gnn_build.sh
# Before it: a bootstrapped repo (deploy/slim_bootstrap.sh or deploy/bootstrap.sh), and data/17lands/ (the FDN replay
# file, cards.csv, abilities.csv, xmage_Foundations_SpecialGuests.json) and data/imitation_scale/row_split_exp4.npy
# copied in: they aren't on HF. ~3 h of build and ~2 h of tables on 24 workers; the shards need ~65 GB of disk.
# Env: WORKERS (default: the cgroup's cores - 2), HEAP (2500m), OUT (data/imitation_graph), SKIP_BUILD=1 (tables on),
# TABLE_JOBS (8: parts turned into tables at once).
set -uo pipefail
cd "$(dirname "$0")/.."
: "${HF_TOKEN:?set HF_TOKEN to a token that can write danbrooks/draftzero-checkpoints}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
OUT=${OUT:-data/imitation_graph}
PY=$(command -v python || command -v python3)
log() { echo "[$(date -u +%H:%M:%S)] gnn_build: $*"; }
CORES=$($PY -c "
import os
try:
    q, p = open('/sys/fs/cgroup/cpu.max').read().split()                    # cgroup v2
    print(int(int(q) / int(p)) if q != 'max' else os.cpu_count())
except Exception:
    try:                                                                     # cgroup v1 (nproc shows the host's)
        q = int(open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read())
        p = int(open('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read())
        print(q // p if q > 0 else os.cpu_count())
    except Exception:
        print(os.cpu_count())")
W=${WORKERS:-$(( CORES - 2 ))}
for f in data/17lands/replay_data_public.FDN.PremierDraft.csv.gz data/17lands/cards.csv data/17lands/abilities.csv \
         data/17lands/xmage_Foundations_SpecialGuests.json data/imitation_scale/row_split_exp4.npy; do
  [ -f "$f" ] || { log "missing $f (copy it in first)"; exit 2; }
done

log "setup: the v0.2 bundle, its card database, the bridge"
$PY - <<'PY'
import os, shutil, tarfile
from huggingface_hub import hf_hub_download
repo, tok = "danbrooks/draftzero-checkpoints", os.environ["HF_TOKEN"]
if not os.path.isdir("xmage/lib"):
    tarfile.open(hf_hub_download(repo, "xmage/generalist-xmage-v0.2-48e49184.tar.gz", token=tok)).extractall(".")
os.makedirs("xmage/db", exist_ok=True)
if not os.path.exists("xmage/db/cards.h2.mv.db"):
    shutil.copy(hf_hub_download(repo, "exp4/games/cards.h2.mv.db", token=tok), "xmage/db/cards.h2.mv.db")
os.chmod("xmage/db/cards.h2.mv.db", 0o644)
print("   xmage/lib jars:", len(os.listdir("xmage/lib")))
PY
export MZ_XMAGE_DIR=$PWD/xmage MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv
sh java/mzbridge/build.sh || exit 3

if [ -z "${SKIP_BUILD:-}" ]; then
  log "build: $W workers"
  $PY tools/imitation_scale/build.py build --graph --out "$OUT" --workers "$W" --heap "${HEAP:-2500m}" || { log "build failed"; exit 4; }
fi
log "tables"
$PY tools/imitation_scale/build.py tables --graph --out "$OUT" --jobs "${TABLE_JOBS:-8}" || { log "tables failed"; exit 5; }

log "compare with experiment #4's tables"
$PY - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("danbrooks/draftzero-checkpoints", allow_patterns=["exp4/tables/h5/*.h5"], local_dir="data/exp4_tables",
                  token=os.environ["HF_TOKEN"])
PY
$PY tools/imitation_scale/build.py compare "$OUT/h5" data/exp4_tables/exp4/tables/h5 --json "$OUT/compare.json"
SAME=$?
log "compare: $([ $SAME = 0 ] && echo identical || echo DIFFERENT, see $OUT/compare.json)"

log "slim tables, upload"
$PY tools/imitation_scale/build.py slim "$OUT/h5" "$OUT/slim" > "$OUT/slim.json" || { log "slim failed"; exit 6; }
cp "$OUT/shards/build_stats.json" "$OUT/tables_stats.json" "$OUT/compare.json" "$OUT/slim/" 2>/dev/null
$PY - <<PY || { log "upload failed"; exit 7; }
import os
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_folder(repo_id="danbrooks/draftzero-checkpoints", folder_path="$OUT/slim",
    path_in_repo="exp4/tables_graph/slim", commit_message="docs/022 stage 1: graph tables (slim: graph files + labels)")
PY
log "done"
echo "GNN_BUILD_DONE same=$SAME"
