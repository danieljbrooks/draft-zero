#!/usr/bin/env bash
# docs/028: a match between two graph networks (or one against the heuristic bot), each network on its own graph
# servers, which this script starts, records and kills by PID when it ends (no reuse of whatever answers on a port).
#   bash deploy/sp_h2h.sh <name> <ckpt A> <ckpt B|-> <replicas each> <play.py args...>
#   e.g. bash deploy/sp_h2h.sh lam95-vs-v0-p0 runs/sp/arms/lam95/best.pt.gz $M0 2 \
#          --bot1 gnn_policy_greedy --bot2 gnn_policy_greedy --pool data/pools/eval.txt --pairs 500 --workers 12
# bot1 reads A's servers, bot2 B's (play.py --bot2-graph-ports, so both games of a pair share their deal). With B
# "-", only A's servers start (a ladder against the heuristic). Env: PORT_BASE (default 51062; A at PORT_BASE+100i,
# B at PORT_BASE+400+100i), BELIEF_PORT (default 50070; shared, started if nothing answers). Output: runs/sp/h2h/<name>.
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=$1 A=$2 B=$3 REPLICAS=$4; shift 4
export MZ_XMAGE_DIR=$PWD/xmage MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv
OUT=runs/sp/h2h/$NAME
mkdir -p "$OUT/logs"
BASE=${PORT_BASE:-51062} BP=${BELIEF_PORT:-50070}
log() { echo "[$(TZ=America/Los_Angeles date '+%a %-I:%M %p PT')] sp_h2h $NAME: $*"; }
wait_up() { for k in $(seq 1 150); do curl -s -m 2 "localhost:$1/healthz" > /dev/null && return 0; sleep 2; done; return 1; }
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; }
trap cleanup EXIT
NGPU=$(nvidia-smi -L 2>/dev/null | grep -c ^GPU); [ "$NGPU" -ge 1 ] || NGPU=1
start_set() {   # <ckpt> <first port> -> the comma-separated ports
  local ck=$1 p0=$2 ports="" i p
  for i in $(seq 0 $((REPLICAS - 1))); do
    p=$((p0 + 100 * i)); ports="${ports:+$ports,}$p"
    if curl -s -m 2 "localhost:$p/healthz" > /dev/null; then log "port $p is taken: refusing to reuse it"; exit 3; fi
    CUDA_VISIBLE_DEVICES=$((i % NGPU)) nohup python tools/imitation_scale/graph_server.py --model "$ck" --port "$p" \
      --threads 16 > "$OUT/logs/graph_server_$p.log" 2>&1 < /dev/null &
    PIDS+=($!)
  done
  echo "$ports"
}
APORTS=$(start_set "$A" "$BASE") || exit 3
BARGS=()
if [ "$B" != "-" ]; then
  BPORTS=$(start_set "$B" $((BASE + 400))) || exit 3
  BARGS=(--bot2-graph-ports "$BPORTS")
fi
for p in ${APORTS//,/ } ${BPORTS:-}; do [ -n "$p" ] || continue; for q in ${p//,/ }; do wait_up "$q" || { log "server $q never came up"; exit 2; }; done; done
if ! echo " $* " | grep -q -- " --open-decklists "; then
  if ! curl -s -m 2 "localhost:$BP/healthz" > /dev/null; then
    nohup python tools/imitation_scale/belief_server.py --port "$BP" > "$OUT/logs/belief.log" 2>&1 < /dev/null &
    PIDS+=($!)
    wait_up "$BP" || { log "belief service never came up"; exit 2; }
  fi
  curl -s -m 20 -X POST "localhost:$BP/sample" -d '{"k":1,"hand":7,"seed":1}' | grep -q samples \
    || { log "belief service answers /healthz but not /sample"; exit 2; }
fi
{ echo "A $(sha256sum "$A" | cut -c1-16) $A"; [ "$B" != "-" ] && echo "B $(sha256sum "$B" | cut -c1-16) $B"; } > "$OUT/networks.txt"
log "A on $APORTS${BPORTS:+, B on $BPORTS}; playing: $*"
python tools/imitation_scale/play.py --deck-root data/deckgen/FDN_PremierDraft_wr60/top_player_FDN_decks \
  --graph-ports "$APORTS" "${BARGS[@]}" --belief-port "$BP" --out "$OUT" "$@" 2>&1 | tee -a "$OUT/play.log"
STATUS=${PIPESTATUS[0]}
rm -rf "data/mzbridge/runtime/play_$(basename "$OUT")"_* 2>/dev/null
log "done (play.py exit $STATUS)"
exit "$STATUS"
