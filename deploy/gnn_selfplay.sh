#!/bin/bash
# docs/024: the GNN's greedy policy against itself, for docs/019 §4.4's 17lands analysis (each card's games-in-hand
# win rate against 17lands'). One shard of 10,000 games (5,000 deck pairs, both seatings), with experiment #4's
# settings (deploy/exp4_mlp_games.sh: open decklists, no search, 50-turn cap); results to HF gnn/games/runs/<name>.
#   HF_TOKEN=<write token> bash deploy/gnn_selfplay.sh MODEL I/N WORKERS HEAP [REPLICAS]
#   e.g. bash deploy/gnn_selfplay.sh models/gnn/full_r1/best_policy.pt.gz 0/3 12 3g
# Env: TAG (a name tag: gnn-<TAG>-selfplay-t0-s<I>).
set -u
cd "$(dirname "$0")/.."
MODEL=$1 SHARD=$2 W=$3 HEAP=$4 REPLICAS=${5:-2}
NAME=gnn${TAG:+-$TAG}-selfplay-t0-s${SHARD%/*}
bash deploy/gnn_games_run.sh "$NAME" "$MODEL" "$REPLICAS" --bot1 gnn_policy_greedy --bot2 gnn_policy_greedy \
  --open-decklists --pairs 5000 --shard "$SHARD" --workers "$W" --heap "$HEAP" --max-turns 50 --search-timeout 300 \
  --game-timeout 7200
echo JOB_DONE
