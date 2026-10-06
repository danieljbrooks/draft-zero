#!/bin/bash
# docs/022 stage 5 with docs/019 §4.6's search: the GNN's games with PIMC on one belief world (closed decklists), on
# the same deck pairs, seed and limits as deploy/pimc_games.sh's MLP games, so each GNN game replays an MLP game but
# for the network. Runs go one after another through deploy/gnn_games_run.sh (graph servers on the GPU), each topped
# up to at least 100 valid games (results to HF gnn/games/runs/<name>).
#   HF_TOKEN=<write token> nohup bash deploy/gnn_pimc_games.sh GNN_MODEL RUN:WORKERS:HEAP ... >> /root/games.log 2>&1 &
#   e.g. bash deploy/gnn_pimc_games.sh models/gnn/main/best_policy.pt.gz policy:20:2g gnn100:24:2500m gnn1000:24:2500m
# RUN: policy (gnn_policy_greedy against heuristic@100), gnnN (gnn@N against heuristic@100), or h2hN (gnn@N against
# the MLP's il_bc@N; MLP=<flat checkpoint> in the environment). After deploy/exp4_games_setup.sh.
# Env: REPLICAS (3 graph servers: one serves ~650 states a second), DECKLISTS (closed | open), MLP, TAG (a name tag:
# pimc-gnn-<TAG>-gnn100, e.g. the network's), SHARD (I/N: this pod's deck pairs, pair % N == I; not topped up), PAIRS0 (deck pairs to start: 50).
set -u
cd "$(dirname "$0")/.."
export PATH=$HOME/venv/bin:/root/venv/bin:$PATH
MODEL=$1; shift
log() { echo "[$(TZ=America/Los_Angeles date '+%a %-I:%M %p PT')] gnn_pimc_games: $*"; }
valid() { python -c "import json,sys; print(json.load(open(sys.argv[1]))['games'])" "$1/summary.json" 2>/dev/null || echo 0; }

DECKLISTS=${DECKLISTS:-closed}
case $DECKLISTS in closed) PREFIX=pimc-gnn OPEN="" ;; open) PREFIX=pimc-open-gnn OPEN=--open-decklists ;;
  *) echo "DECKLISTS must be closed or open"; exit 2 ;; esac
PREFIX=$PREFIX${TAG:+-$TAG}

for item in "$@"; do
  IFS=: read -r RUN W HEAP <<< "$item"
  case $RUN in
    policy) NAME=$PREFIX-pvh-t0 B1=gnn_policy_greedy B2=heuristic@100 GT=7200 ;;
    gnn*) N=${RUN#gnn}; NAME=$PREFIX-gnn$N B1=gnn@$N B2=heuristic@100 GT=28800 ;;
    h2h*) N=${RUN#h2h}; NAME=$PREFIX-h2h$N B1=gnn@$N B2=il_bc@$N GT=28800
          : "${MLP:?h2h needs MLP=<a flat checkpoint>}" ;;
    *) log "unknown run $RUN"; exit 2 ;;
  esac
  PAIRS=${PAIRS0:-50} ROUND=0
  while :; do
    log "$NAME: $B1 against $B2, PIMC ($DECKLISTS decklists), $PAIRS pairs, $W workers, heap $HEAP"
    bash deploy/gnn_games_run.sh "$NAME${SHARD:+-s${SHARD%/*}}" "$MODEL" "${REPLICAS:-3}" --bot1 "$B1" --bot2 "$B2" \
      --method pimc $OPEN ${SHARD:+--shard "$SHARD"} --pairs "$PAIRS" --workers "$W" --heap "$HEAP" --max-turns 50 \
      --search-timeout 900 --game-timeout "$GT"
    V=$(valid "runs/gnn/games/$NAME${SHARD:+-s${SHARD%/*}}")
    log "$NAME: $V valid games"
    { [ -n "${SHARD:-}" ] || [ "$V" -ge 100 ]; } && break
    ROUND=$((ROUND + 1))
    [ "$ROUND" -gt 3 ] && { log "$NAME: still short after 3 top-ups; moving on"; break; }
    PAIRS=$((PAIRS + (100 - V + 1) / 2 + 1))
  done
done
log "done: $*"
echo JOB_DONE
