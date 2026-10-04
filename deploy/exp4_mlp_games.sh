#!/bin/bash
# Experiment #4: the best MLP's evaluation games on the second machine (Dan, Sat 3 October; docs/018, "Extending run
# 2"), after deploy/exp4_games_setup.sh. Phase C's settings and deck pairs (the same seed and pool, so each game
# replays a transformer game but for the network): greedy self-play for the 17lands statistics, and the MLP searching
# (il_bc@N) against heuristic@100. Runs go one after another through deploy/exp4_games_run.sh into
# runs/exp4/games/mlp-<run>; a search run with fewer than 100 valid games (engine errors left out) is topped up with
# more deck pairs (Dan: at least 100 valid games a budget). Value servers holding another network are stopped first.
# The second machine has a 40 GiB memory limit, so each run's workers and Java heap are given:
#   nohup deploy/exp4_mlp_games.sh MODEL RUN:WORKERS:HEAP ... >> runs/exp4/mlp_games.log 2>&1 < /dev/null &
#   e.g. deploy/exp4_mlp_games.sh runs/exp4/mlp_1ep/best_policy.pt.gz selfplay:20:2000m ilbc100:14:2500m \
#        ilbc300:12:2500m ilbc1000:9:3g ilbc3000/1/2:6:5g
# RUN: selfplay (10,000 games, policy_greedy both seats, open decklists) or ilbcN[/I/S] (il_bc@N against
# heuristic@100, closed decklists, 50 deck pairs; /I/S plays shard I of S, and is not topped up).
set -u
cd "$(dirname "$0")/.."
export PATH=$HOME/venv/bin:$PATH
MODEL=$1; shift
log() { echo "[$(TZ=America/Los_Angeles date '+%a %-I:%M %p PT')] mlp_games: $*"; }

for pid in $(pgrep -f "[v]alue_server.py --model"); do
  if ! tr '\0' ' ' < /proc/$pid/cmdline | grep -q -- "--model $MODEL "; then
    log "stopping value server $pid (another network)"; kill "$pid"; sleep 5
  fi
done

valid() { python -c "import json,sys; print(json.load(open(sys.argv[1]))['games'])" "$1/summary.json" 2>/dev/null || echo 0; }

for item in "$@"; do
  IFS=: read -r RUN W HEAP <<< "$item"
  if [ "$RUN" = selfplay ]; then
    NAME=mlp-selfplay-t0
    log "$NAME: $W workers, heap $HEAP"
    bash deploy/exp4_games_run.sh "$NAME" "$MODEL" 2 --bot1 policy_greedy --bot2 policy_greedy --open-decklists \
      --pairs 5000 --workers "$W" --heap "$HEAP" --max-turns 50 --search-timeout 300 --game-timeout 7200
    continue
  fi
  IFS=/ read -r BUDGETRUN SI SN <<< "$RUN"
  N=${BUDGETRUN#ilbc}
  NAME=mlp-ilbc$N${SI:+-s$SI}
  GT=28800; ST=900
  [ "$N" -gt 1000 ] && { GT=86400; ST=2700; }
  PAIRS=50 ROUND=0
  while :; do
    log "$NAME: il_bc@$N against heuristic@100, $PAIRS pairs${SI:+, shard $SI/$SN}, $W workers, heap $HEAP"
    bash deploy/exp4_games_run.sh "$NAME" "$MODEL" 2 --bot1 "il_bc@$N" --bot2 heuristic@100 --pairs "$PAIRS" \
      ${SI:+--shard "$SI/$SN"} --workers "$W" --heap "$HEAP" --max-turns 50 --search-timeout "$ST" --game-timeout "$GT"
    V=$(valid "runs/exp4/games/$NAME")
    log "$NAME: $V valid games"
    { [ -n "$SI" ] || [ "$V" -ge 100 ]; } && break
    ROUND=$((ROUND + 1))
    [ "$ROUND" -gt 3 ] && { log "$NAME: still short after 3 top-ups; moving on"; break; }
    PAIRS=$((PAIRS + (100 - V + 1) / 2 + 1))
  done
done
log "done: $*"
