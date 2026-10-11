# docs/031's helpers for the laptop (macOS: no `timeout`, so a perl alarm is each job's hard kill).
#   source gorge/recipe/lib.sh
# Paths are relative to the worktree root; data/ holds the decks and the runs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
G="${GORGE_DIR:-../ext/gorge}"
PY="${PY:-../draft-zero-v02/.venv/bin/python}"
FULL="autopay:oppnodes:oppfull:morekinds"
COMMON=(-cards "$G/.cards" -decks data/gorge/decks -pool data/gorge/pool.tsv -mulligans 2 -mull-heuristic -progress 0)

hardkill() { local s=$1; shift; perl -e 'alarm shift; exec @ARGV' "$s" "$@"; }

# width -> MLP config (docs/027's MLP64 at 64; models.py's default at 384)
mlp_cfg() {
  case "$1" in
    64)  echo '{"d":64,"ff":128,"layers":1,"d_emb":32,"card_d":32,"head_hidden":64}' ;;
    128) echo '{"d":128,"ff":256,"layers":1,"d_emb":32,"card_d":64,"head_hidden":128}' ;;
    256) echo '{"d":256,"ff":512,"layers":2,"d_emb":64,"card_d":128,"head_hidden":256}' ;;
    384) echo '{}' ;;
    *) echo "unknown width $1" >&2; return 2 ;;
  esac
}

# train OUT WIDTH SECONDS -- extra dzg.train args (--train ..., targets, --limit-records ...)
train() {
  local out=$1 width=$2 secs=$3; shift 3
  [ -f "$out/best.dzgw" ] && { echo "skip $out (done)"; return 0; }
  mkdir -p "$out"
  PYTHONPATH=gorge hardkill "$secs" "$PY" -m dzg.train --arch mlp --config "$(mlp_cfg "$width")" \
    --val evaldecks=data/gorge/runs/r31/val/pack --device mps --in-ram "$@" --out "$out" > "$out/train.log" 2>&1 \
    && PYTHONPATH=gorge "$PY" -m dzg.export "$out/best.pt" "$out/best.dzgw" >> "$out/train.log" 2>&1
  tail -1 "$out/train.log"
}

# strength NET.dzgw OUT [PAIRS] [SIMS] [WORKERS]: the full search with the network against the zero point
strength() {
  local net=$1 out=$2 pairs=${3:-500} sims=${4:-100} w=${5:-7}
  mkdir -p "$(dirname "$out")"
  hardkill 7200 "$G/bin/dzgorge" play "${COMMON[@]}" -split eval -seed 3131 -pairs "$pairs" -workers "$w" \
    -a "az:sims=$sims:$FULL:topk=6:gonet=$net" -b "az:sims=$sims:$FULL" -out "$out.jsonl" > "$out.log" 2>&1
}

# h2h NET_A NET_B OUT [PAIRS]: the full search with network A against the full search with network B
h2h() {
  local a=$1 b=$2 out=$3 pairs=${4:-500}
  hardkill 7200 "$G/bin/dzgorge" play "${COMMON[@]}" -split eval -seed 3131 -pairs "$pairs" -workers 7 \
    -a "az:sims=100:$FULL:topk=6:gonet=$a" -b "az:sims=100:$FULL:topk=6:gonet=$b" -out "$out.jsonl" > "$out.log" 2>&1
}

# alone NET.dzgw OUT [PAIRS]: the network's policy alone against gorge's bot
alone() {
  local net=$1 out=$2 pairs=${3:-1000}
  hardkill 3600 "$G/bin/dzgorge" play "${COMMON[@]}" -split eval -seed 3131 -pairs "$pairs" -workers 7 \
    -a "prior:gonet=$net" -b "bot" -out "$out.jsonl" > "$out.log" 2>&1
}

# selfplay OUT GAMES SEED [NET.dzgw] [SIMS]: cheap games (record one searched decision in four), packed
selfplay() {
  local out=$1 n=$2 seed=$3 net=${4:-} sims=${5:-25} pol
  pol="az:sims=$sims:explore:$FULL"
  [ -n "$net" ] && pol="$pol:topk=6:gonet=$net"
  mkdir -p "$out"
  [ -d "$out/pack" ] && { echo "skip $out (packed)"; return 0; }
  hardkill $(( n + 1800 )) "$G/bin/dzgorge" play "${COMMON[@]}" -record-features entity -record-every 4 \
    -split train -a "$pol" -b "$pol" -pairs "$n" -workers 7 -seed "$seed" -out "$out/games.jsonl" \
    -corpus "$out/visits.jsonl.gz" > "$out/play.log" 2>&1
  "$G/bin/dzgorge" pack -in "$out/visits.jsonl.gz" -out "$out/pack" -shard 200000 >> "$out/play.log" 2>&1 \
    && "$PY" gorge/recipe/shrink.py "$out/pack" > /dev/null
}

# score FILE.jsonl: side A's paired score and n
score() { "$PY" gorge/recipe/score.py "$@"; }
