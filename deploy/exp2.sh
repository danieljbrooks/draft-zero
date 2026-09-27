#!/bin/bash
# exp2.sh — launch experiment #2 on this pod under a hard dollar budget. Run from the repo root,
# after deploy/pilot_v02_setup.sh.
#   DZ_BUDGET_USD=29 bash deploy/exp2.sh [config]      (default configs/exp2.yml)
#
# The budget becomes three limits, all from this pod's own hourly price:
#   - the watchdog stops at (budget hours - 0.75 h): it lets the current generation finish for
#     up to 0.5 h, syncs, pushes the weights to Hugging Face, verifies, then removes the pod;
#   - a self-destruct removes the pod at exactly the budget hours, whatever state it is in;
#   - weights are pushed to Hugging Face after every generation, so nothing depends on
#     anyone pulling results before the pod goes.
# The HF push needs /root/.dz_env (mode 600) holding HF_TOKEN (write access to HF_REPO only)
# and HF_REPO. Without it this refuses to start: a removed pod keeps nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="${1:-configs/exp2.yml}"
BUDGET="${DZ_BUDGET_USD:-29}"
ENVF=/root/.dz_env

grep -q '^HF_TOKEN=' "$ENVF" 2>/dev/null && grep -q '^HF_REPO=' "$ENVF" \
  || { echo "!! $ENVF needs HF_TOKEN and HF_REPO: a budget-stopped pod would keep no weights"; exit 1; }
chmod 600 "$ENVF"
: "${RUNPOD_POD_ID:?not on a RunPod pod (RUNPOD_POD_ID unset)}"

RATE="${DZ_COST_PER_HR:-$(runpodctl pod get "$RUNPOD_POD_ID" 2>/dev/null \
  | python -c "import json,sys; print(json.load(sys.stdin).get('costPerHr') or '')" 2>/dev/null || true)}"
[ -n "$RATE" ] || { echo "!! could not read this pod's price; set DZ_COST_PER_HR"; exit 1; }
read HARD MAXH <<< "$(python -c "h = $BUDGET / $RATE; print(f'{h:.2f} {max(0.5, h - 0.75):.2f}')")"
SECS="$(python -c "print(int($HARD * 3600))")"
echo "== budget \$$BUDGET at \$$RATE/hr: stop at ${MAXH} h, hard removal at ${HARD} h"

# the hard backstop, armed before anything else can fail
nohup setsid bash -c "sleep $SECS; runpodctl remove pod $RUNPOD_POD_ID || runpodctl pod remove $RUNPOD_POD_ID" \
  > /root/selfdestruct.log 2>&1 < /dev/null &

export MZ_DECK_DIR="$(pwd)/data/deckgen/FDN_PremierDraft_wr60/top_player_FDN_decks"
export MZ_ACTION_VOCAB="$(pwd)/assets/vocab/FDN_SPG.tsv"
mkdir -p logs
CFG=logs/exp2.yml
# size the layout from THIS pod's cgroup (never nproc): K JVMs x 4 threads, K = cores / 4,
# 70% of RAM across the heaps, 4 waves of games per generation, and a 200-game eval
python - "$BASE" "$CFG" <<'PY'
import sys, yaml
from draftzero.resources import cpu_quota, mem_limit_gb
cores, ram = cpu_quota(), mem_limit_gb()
if not cores or not ram:
    raise SystemExit("!! no cgroup cpu/memory limit found; refusing to size from nproc")
cfg = yaml.safe_load(open(sys.argv[1]))
k = max(1, int(cores // 4))
threads = cfg["jvm"]["threads"]
cfg["jvm"].update(jvms=k, heap=f"{max(4, int(ram * 0.7 / k))}g")
cfg["bootstrap_games"] = cfg["games_per_gen"] = 4 * k * threads
yaml.safe_dump(cfg, open(sys.argv[2], "w"), sort_keys=False)
print(f"== layout: {k} JVMs x {threads} threads, heap {cfg['jvm']['heap']}, "
      f"{cfg['games_per_gen']} games per generation, cores {cores:.1f}, RAM {ram:.0f} GB")
PY

DZ_MAX_HOURS="$MAXH" DZ_GRACE_HOURS=0.5 DZ_STALL_MINUTES=90 DZ_HF_ENV="$ENVF" \
DZ_ON_COMPLETE="runpodctl remove pod $RUNPOD_POD_ID || runpodctl pod remove $RUNPOD_POD_ID" \
  bash deploy/launch.sh "$CFG" --fresh
