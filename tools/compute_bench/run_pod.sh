#!/usr/bin/env bash
# One machine of the compute benchmark (docs/020), from the laptop: rent a RunPod pod (retrying while none is in
# stock), wait for SSH, clone the repo and copy the benchmark's files, arm the pod's self-destruct, then start the
# setup and deploy/compute_bench.sh in the background. The pod removes itself when the battery is done (or at the
# deadline). Results go to the HF repo, exp4/bench/runs/<tag>/.
#   bash tools/compute_bench/run_pod.sh <tag> <price> "<offer>" <pods.py create args...> -- [KEY=VALUE ...]
#   e.g. bash tools/compute_bench/run_pod.sh cpu5c-32 1.12 "cpu5c-32-64 Secure" --cpu cpu5c --vcpu 32 -- GAMES_MIN=25
# Env: TRIES (create attempts, default 1), TRY_EVERY (s, 120), DEADLINE_MIN (self-destruct, 150), SLIM=1 (RunPod's
# 0.73 GB base image and deploy/slim_bootstrap.sh instead of the 10.7 GB PyTorch image; SLIM_TORCH: the torch wheel
# index, CPU-only by default).
set -uo pipefail
cd "$(dirname "$0")/../.."
TAG=$1 PRICE=$2 OFFER=$3; shift 3
CREATE=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do CREATE+=("$1"); shift; done; [ $# -gt 0 ] && shift
KNOBS=("$@")
S=${SCRATCH:-/tmp}/cb-$TAG; mkdir -p "$S"
log() { echo "[$(date -u +%H:%M:%S)] pod $TAG: $*" | tee -a "$S/driver.log"; }
source ~/.runpod/env; source ~/.runpod/hf_env
SSHO=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=30 -o LogLevel=ERROR -i ~/.ssh/id_ed25519)

ID=""
for k in $(seq 1 "${TRIES:-1}"); do
  out=$(python3 tools/compute_bench/pods.py create --name "cb-$TAG" "${CREATE[@]}" \
        ${SLIM:+--image runpod/base:1.4.0-ubuntu2404} 2>&1)
  ID=$(echo "$out" | python3 -c "import json,sys; print(json.load(sys.stdin).get('id') or '')" 2>/dev/null)
  [ -n "$ID" ] && break
  log "create attempt $k: $(echo "$out" | head -c 160)"
  [ "$k" -lt "${TRIES:-1}" ] && sleep "${TRY_EVERY:-120}"
done
[ -n "$ID" ] || { log "no pod"; exit 1; }
echo "$ID" > "$S/pod_id"
log "created $ID: $(echo "$out" | tr -d '\n' | head -c 400)"

IP="" PORT=""
for k in $(seq 1 90); do
  j=$(python3 tools/compute_bench/pods.py get "$ID" 2>/dev/null)
  IP=$(echo "$j" | python3 -c "import json,sys; print(json.load(sys.stdin).get('publicIp') or '')" 2>/dev/null)
  PORT=$(echo "$j" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ssh_port') or '')" 2>/dev/null)
  [ -n "$IP" ] && [ -n "$PORT" ] && break
  sleep 10
done
[ -n "$IP" ] && [ -n "$PORT" ] || { log "no public ip/port after 15 min: removing"; python3 tools/compute_bench/pods.py rm "$ID"; exit 1; }
echo "$j" > "$S/pod.json"
log "ssh root@$IP -p $PORT"
for k in $(seq 1 60); do ssh "${SSHO[@]}" -p "$PORT" "root@$IP" true 2>/dev/null && break; sleep 10; done
ssh "${SSHO[@]}" -p "$PORT" "root@$IP" true 2>/dev/null || { log "ssh never answered: removing"; python3 tools/compute_bench/pods.py rm "$ID"; exit 1; }
echo "ssh ${SSHO[*]} -p $PORT root@$IP" > "$S/ssh_cmd"
log "ssh up"

tar czf "$S/bench_files.tgz" tools/compute_bench deploy/compute_bench.sh deploy/slim_bootstrap.sh deploy/exp4_games_setup.sh \
  configs/compute_bench_soft.yml configs/compute_bench_human.yml
scp "${SSHO[@]}" -P "$PORT" "$S/bench_files.tgz" "root@$IP:/root/bench_files.tgz" > /dev/null
ENVS="HF_TOKEN=$HF_TOKEN PRICE=$PRICE OFFER='$OFFER' ${KNOBS[*]:-}"
ssh "${SSHO[@]}" -p "$PORT" "root@$IP" "bash -s" > "$S/remote_start.log" 2>&1 <<EOF
set -e
cd /root
command -v git > /dev/null || { apt-get update -qq && apt-get install -y -qq git > /dev/null; }
[ -d draft-zero ] || git clone -q https://github.com/danieljbrooks/draft-zero.git
cd draft-zero && tar xzf /root/bench_files.tgz
bash deploy/runpod_arm.sh $(( ${DEADLINE_MIN:-150} * 60 ))
cat > /root/bench_job.sh <<'JOB'
set -o pipefail
cd /root/draft-zero
rc=0
if [ -n "\${SLIM:-}" ]; then TORCH_INDEX="\${SLIM_TORCH:-https://download.pytorch.org/whl/cpu}" bash deploy/slim_bootstrap.sh || rc=1; export NO_BOOTSTRAP=1; fi
[ \$rc = 0 ] && { bash deploy/exp4_games_setup.sh exp4/bench/mlp-d1-s30-combined/best_policy_fp16.pt.gz || rc=2; }
[ \$rc = 0 ] && { bash deploy/compute_bench.sh "\$TAG" || rc=3; }
echo "JOB_DONE rc=\$rc"
# the job's own log goes to the HF repo whatever happened, so a failed pod still says why
\$(command -v python || command -v python3) -c "import os; from huggingface_hub import HfApi; HfApi(token=os.environ['HF_TOKEN']).upload_file(path_or_fileobj='/root/bench.out', path_in_repo='exp4/bench/runs/' + os.environ['TAG'] + '/bench.out', repo_id='danbrooks/draftzero-checkpoints')" || true
[ \$rc = 0 ] || sleep 600      # a failure: ten minutes to look before the pod goes
sleep 60; /root/selfdestruct_now.sh
JOB
nohup bash -c "export $ENVS TAG=$TAG ${SLIM:+SLIM=1} ${SLIM_TORCH:+SLIM_TORCH=$SLIM_TORCH}; bash /root/bench_job.sh" > /root/bench.out 2>&1 < /dev/null &
echo started
EOF
log "remote: $(tail -2 "$S/remote_start.log" | tr '\n' ' ')"
