#!/bin/bash
# runpod_arm.sh <seconds>: (re)arm this RunPod pod's self-destruct: after <seconds>, GraphQL podTerminate with the
# pod's own key from PID 1's environment (an SSH session doesn't inherit it; the image's runpodctl 1.x can't
# authenticate with it). Re-running moves the deadline. Tested on a throwaway pod (docs/018, pods).
set -e
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
cat > /root/selfdestruct_now.sh <<EOS
#!/bin/bash
curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"query":"mutation { podTerminate(input:{podId:\"$RUNPOD_POD_ID\"}) }"}'
EOS
chmod 700 /root/selfdestruct_now.sh
pkill -f "selfdestruct_sleep" || true
DEADLINE=$(( $(date +%s) + $1 ))
echo "$DEADLINE $(date -u -d @$DEADLINE)" > /root/selfdestruct.deadline
nohup bash -c "exec -a selfdestruct_sleep sleep $1" > /dev/null 2>&1 &
SPID=$!
nohup bash -c "while kill -0 $SPID 2>/dev/null; do sleep 5; done; [ -f /root/selfdestruct.deadline ] && [ \$(date +%s) -ge $DEADLINE ] && /root/selfdestruct_now.sh" > /root/selfdestruct.log 2>&1 &
sleep 1; pgrep -f selfdestruct_sleep > /dev/null && echo "armed: terminates $RUNPOD_POD_ID at $(date -u -d @$DEADLINE)"
