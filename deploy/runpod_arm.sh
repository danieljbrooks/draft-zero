#!/bin/bash
# runpod_arm.sh <seconds>: (re)arm this RunPod pod's self-destruct: after <seconds>, GraphQL podTerminate with the
# pod's own key from PID 1's environment (an SSH session doesn't inherit it; the image's runpodctl 1.x can't
# authenticate with it). Re-running moves the deadline. Tested on a throwaway pod (docs/018, pods).
# The terminate is retried with backoff (15 s doubling to 5 min, 30 s timeout each) for up to 30 min until RunPod
# accepts it, and every try is logged to /root/selfdestruct.log. Without the key or pod id it arms nothing and fails.
# Tests only: DZ_SELFDESTRUCT_DIR (default /root), DZ_SELFDESTRUCT_ENVIRON (default /proc/1/environ),
# DZ_SELFDESTRUCT_DELAY (the first pause, default 15 s).
set -e
DIR=${DZ_SELFDESTRUCT_DIR:-/root}
[[ "$1" =~ ^[0-9]+$ ]] || { echo "runpod_arm: want the seconds until the self-destruct, not '$1'" >&2; exit 2; }
eval "$(tr '\0' '\n' < "${DZ_SELFDESTRUCT_ENVIRON:-/proc/1/environ}" | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
if [ -z "${RUNPOD_API_KEY:-}" ] || [ -z "${RUNPOD_POD_ID:-}" ]; then
  echo "runpod_arm: RUNPOD_API_KEY or RUNPOD_POD_ID is not in PID 1's environment: NOT armed" >&2
  exit 1
fi
( umask 077
  { echo '#!/bin/bash'
    echo '# written by runpod_arm.sh: terminate this pod, retrying until RunPod accepts it (see that script)'
    printf 'RUNPOD_API_KEY=%q\nRUNPOD_POD_ID=%q\n' "$RUNPOD_API_KEY" "$RUNPOD_POD_ID"
    cat <<'EOS'
end=$(( $(date +%s) + 1800 )); delay=${DZ_SELFDESTRUCT_DELAY:-15}; try=1
body=$(printf '{"query":"mutation { podTerminate(input:{podId:\\"%s\\"}) }"}' "$RUNPOD_POD_ID")
while :; do
  out=$(curl -sS --connect-timeout 10 --max-time 30 -w '\nHTTP %{http_code}' -X POST "https://api.runpod.io/graphql" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -d "$body" 2>&1)
  echo "$(date -u +%FT%TZ) selfdestruct try $try: $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-300)"
  case "$out" in
    *'"errors"'*) ;;
    *'"podTerminate"'*'HTTP 200') echo "$(date -u +%FT%TZ) selfdestruct: terminate accepted"; exit 0 ;;
  esac
  if [ "$(date +%s)" -ge "$end" ]; then
    echo "$(date -u +%FT%TZ) selfdestruct: GAVE UP after $try tries: the pod may still be billing"
    exit 1
  fi
  sleep "$delay"; delay=$(( delay * 2 > 300 ? 300 : delay * 2 )); try=$(( try + 1 ))
done
EOS
  } > "$DIR/selfdestruct_now.sh" )
chmod 700 "$DIR/selfdestruct_now.sh"
pkill -f "^selfdestruct_sleep " || true
DEADLINE=$(( $(date +%s) + $1 ))
echo "$DEADLINE $(date -u -d @$DEADLINE)" > "$DIR/selfdestruct.deadline"
nohup bash -c "exec -a selfdestruct_sleep sleep $1" > /dev/null 2>&1 &
SPID=$!
nohup bash -c "while kill -0 $SPID 2>/dev/null; do sleep 5; done; [ -f '$DIR/selfdestruct.deadline' ] && [ \$(date +%s) -ge $DEADLINE ] && '$DIR/selfdestruct_now.sh'" >> "$DIR/selfdestruct.log" 2>&1 &
sleep 1; pgrep -f "^selfdestruct_sleep " > /dev/null && echo "armed: terminates $RUNPOD_POD_ID at $(date -u -d @$DEADLINE)"
