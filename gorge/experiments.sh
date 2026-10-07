#!/usr/bin/env bash
# docs/026 §4.2–4.5 and §5, after gen-0 (gorge/azloop.sh 0): the exact runs behind the doc's tables, in order.
# Each step skips work whose output exists. Takes about 3.5 hours on 4 vCPUs.
#
#   bash gorge/experiments.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
GORGE_DIR="${GORGE_DIR:-$(cd "$REPO/.." && pwd)/ext/gorge}"
D="$REPO/data/gorge"; R="$D/runs/az"; C="$D/runs/eval/confirm"
mkdir -p "$C"
cd "$GORGE_DIR"
play() { # out, then dzgorge play flags
  local out="$1"; shift
  [ -s "$out.summary.json" ] || ./bin/dzgorge play -decks "$D/decks" -pool "$D/pool.tsv" -workers 4 -out "$out" "$@" 2> "${out%.jsonl}.log" >/dev/null
}
train() { # out, log, then policytrain flags
  local out="$1" log="$2"; shift 2
  [ -s "$out" ] || ./bin/policytrain -out "$out" "$@" > "$log" 2>&1
}
readout() { # network: its policy and value on the held-out searched positions
  [ -s "$R/$1.evalset.json" ] || ./bin/policytrain -visits-corpus "$R/evalset.visits.jsonl.gz" -init "$R/$1.gpol" -epochs 0 \
    -visits-eval "$R/evalset.visits.jsonl.gz" -visits-report "$R/$1.evalset.json" > "$R/$1.evalset.log" 2>&1
}
G0="$R/gen0.visits.jsonl.gz"
[ -s "$R/gen0.visits.sorted.jsonl.gz" ] && G0="$R/gen0.visits.sorted.jsonl.gz"  # written by an earlier dzgorge, in completion order
RECIPE=(-lr 0.1 -clip 5 -value-weight 1 -value-blend 0.05 -visits-temp 1 -seed 1)

# §4.2: held-out searched positions (the gen-0 search on the eval decks), gen 1, its games
play "$R/evalset.selfplay.jsonl" -split eval -a az:sims=50 -b az:sims=50 -pairs 150 -seed 900 -corpus "$R/evalset.visits.jsonl.gz"
train "$R/gen1.gpol" "$R/gen1.train.log" -visits-corpus "$G0" -epochs 12 "${RECIPE[@]}" -residual-init 0
readout gen1
(cd "$REPO" && NET="$R/gen1.gpol" bash gorge/evalnet.sh)
# §4.3: a value from cheap games
play "$R/v.selfplay.jsonl" -split train -a az:sims=2 -b az:sims=2 -pairs 15000 -seed 2002 -corpus "$R/v.visits.jsonl.gz"
train "$R/v1.gpol" "$R/v1.train.log" -visits-corpus "$R/v.visits.jsonl.gz" -visits-label bot -epochs 3 -lr 0.1 -clip 5 \
  -value-weight 1 -value-blend 0 -seed 1 -visits-eval "$R/evalset.visits.jsonl.gz" -visits-report "$R/v1.evalset.json"
(cd "$REPO" && ARMS="leaf_v_az" NET="$R/v1.gpol" bash gorge/evalnet.sh)
# §4.4: two variants of gen 1
train "$R/gen1e2.gpol" "$R/gen1e2.train.log" -visits-corpus "$G0" -epochs 2 "${RECIPE[@]}" -residual-init 0
train "$R/gen1r.gpol" "$R/gen1r.train.log" -visits-corpus "$G0" -epochs 12 "${RECIPE[@]}" -residual-init 2
readout gen1e2
readout gen1r
(cd "$REPO" && ARMS="prior_v_bot full_v_az" NET="$R/gen1r.gpol" bash gorge/evalnet.sh)
(cd "$REPO" && ARMS="leaf_v_az full_v_az" NET="$R/gen1e2.gpol" bash gorge/evalnet.sh)
# §4.5: the 2-epoch network on fresh pairs, deeper, and at equal time per decision
play "$C/gen1e2_full_v_az_s78.jsonl" -split eval -a "az:sims=25:net=$R/gen1e2.gpol" -b az:sims=25 -pairs 250 -seed 78
play "$C/v1_leaf_v_az_s100.jsonl" -split eval -a "az:sims=100:net=$R/v1.gpol:prior=uniform" -b az:sims=100 -pairs 75 -seed 79
play "$C/gen1e2_full_v_az_s100.jsonl" -split eval -a "az:sims=100:net=$R/gen1e2.gpol" -b az:sims=100 -pairs 75 -seed 79
play "$C/gen1e2_s25_v_az50.jsonl" -split eval -a "az:sims=25:net=$R/gen1e2.gpol" -b az:sims=50 -pairs 150 -seed 80
play "$C/az50_v_az25.jsonl" -split eval -a az:sims=50 -b az:sims=25 -pairs 150 -seed 80
# §5: card statistics from the gen-1 policy and from search
(cd "$REPO" && NETS="$R/gen1.gpol" bash gorge/gih_runs.sh)
play "$D/runs/gih/az10.jsonl" -split all -a az:sims=10 -b az:sims=10 -pairs 8000 -seed 34
# §2: throughput of network self-play
play "$C/gen1e2_s25_self.jsonl" -split eval -a "az:sims=25:net=$R/gen1e2.gpol" -b "az:sims=25:net=$R/gen1e2.gpol" -pairs 200 -seed 81
