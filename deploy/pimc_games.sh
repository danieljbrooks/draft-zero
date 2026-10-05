#!/bin/bash
# The MLP's games with PIMC instead of IS-MCTS (docs/019 §4.2's follow-up, Dan, Sun 4 October): both bots search
# one world with MageZero's tree search (play.py --method pimc), on the same deck pairs, seed and limits as
# deploy/exp4_mlp_games.sh, so each game replays an IS-MCTS game of docs/019 but for the search method. Runs go one
# after another through deploy/exp4_games_run.sh, each topped up to at least 100 valid games (results to HF
# exp4/games/runs/<name>).
#   nohup deploy/pimc_games.sh MODEL RUN:WORKERS:HEAP ... >> /root/podrun.log 2>&1 < /dev/null &
#   e.g. deploy/pimc_games.sh models/exp4/mlp_1ep/best_policy.pt.gz policy:20:2g ilbc100:18:2500m ilbc1000/1/2:18:2500m
# RUN: policy (policy_greedy against heuristic@100: "0 simulations" for the network) or ilbcN[/I/S] (il_bc@N against
# heuristic@100; /I/S plays shard I of S, for two pods, and is not topped up).
# DECKLISTS=closed (default; runs pimc-mlp-<run>): the world is a sample from the belief service. DECKLISTS=open (runs
# pimc-open-mlp-<run>): the world is re-dealt from the opponent's real decklist, the information docs/019's IS-MCTS
# games had (their belief service failed every call, so every decision fell back to the real decklist).
# Ends by writing JOB_DONE, which the pod's watcher uses to terminate it.
set -u
cd "$(dirname "$0")/.."
export PATH=$HOME/venv/bin:/root/venv/bin:$PATH
MODEL=$1; shift
log() { echo "[$(TZ=America/Los_Angeles date '+%a %-I:%M %p PT')] pimc_games: $*"; }
valid() { python -c "import json,sys; print(json.load(open(sys.argv[1]))['games'])" "$1/summary.json" 2>/dev/null || echo 0; }

DECKLISTS=${DECKLISTS:-closed}
case $DECKLISTS in closed) PREFIX=pimc-mlp OPEN="" ;; open) PREFIX=pimc-open-mlp OPEN=--open-decklists ;;
  *) echo "DECKLISTS must be closed or open"; exit 2 ;; esac

for item in "$@"; do
  IFS=: read -r RUN W HEAP <<< "$item"
  SI="" SN=""
  if [ "$RUN" = policy ]; then
    NAME=$PREFIX-pvh-t0 BOT=policy_greedy GT=7200
  else
    IFS=/ read -r BUDGETRUN SI SN <<< "$RUN"
    N=${BUDGETRUN#ilbc}
    NAME=$PREFIX-ilbc$N${SI:+-s$SI} BOT=il_bc@$N GT=28800
  fi
  PAIRS=50 ROUND=0
  while :; do
    log "$NAME: $BOT against heuristic@100, PIMC ($DECKLISTS decklists), $PAIRS pairs${SI:+, shard $SI/$SN}, $W workers, heap $HEAP"
    bash deploy/exp4_games_run.sh "$NAME" "$MODEL" 4 --bot1 "$BOT" --bot2 heuristic@100 --method pimc $OPEN \
      ${SI:+--shard "$SI/$SN"} --pairs "$PAIRS" --workers "$W" --heap "$HEAP" --max-turns 50 --search-timeout 900 --game-timeout "$GT"
    V=$(valid "runs/exp4/games/$NAME")
    log "$NAME: $V valid games"
    { [ -n "$SI" ] || [ "$V" -ge 100 ]; } && break
    ROUND=$((ROUND + 1))
    [ "$ROUND" -gt 3 ] && { log "$NAME: still short after 3 top-ups; moving on"; break; }
    PAIRS=$((PAIRS + (100 - V + 1) / 2 + 1))
  done
done
log "done: $*"
echo JOB_DONE
