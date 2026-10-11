# docs/031's helpers for the laptop (macOS: no `timeout`, so a perl alarm is each job's hard kill).
#   source gorge/recipe/lib.sh
# Paths are relative to the worktree root; data/ holds the decks and the runs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
G="${GORGE_DIR:-../ext/gorge}"
PY="${PY:-../draft-zero-v02/.venv/bin/python}"
FULL="autopay:oppnodes:oppfull:morekinds"
COMMON=(-cards "$G/.cards" -decks data/gorge/decks -pool data/gorge/pool.tsv -mulligans 2 -mull-heuristic -progress 0
        -topk-max-options 64)   # patch 0008: no network ranking at a decision with >64 options (they hung games)

# hardkill SECS CMD...: run CMD; after SECS send it TERM, then KILL. (An alarm signal is not enough: Go programs such as
# dzgorge catch SIGALRM and ignore it, so the first version of this never stopped a hung game.)
hardkill() {
  local s=$1; shift
  perl -e 'my $t = shift; my $pid = fork(); if (!$pid) { exec @ARGV or die "exec: $!" }
    $SIG{ALRM} = sub { kill "TERM", $pid; sleep 10; kill "KILL", $pid }; alarm $t;
    waitpid($pid, 0); exit($? & 127 ? 128 + ($? & 127) : $? >> 8)' "$s" "$@"
}

# match OUT.jsonl SIMS PAIRS POLICY_A POLICY_B: paired games on the eval decks (seed 3131), in chunks of deck pairs,
# each with its own kill, so one hung game costs its chunk's time, not the match. A killed chunk keeps its complete
# games (score.py counts complete pairs only) and resumes after the pair of the first missing game. Resumable.
match() {
  local out=$1 sims=$2 pairs=$3 a=$4 b=$5 s m e cp secs
  cp=$(( sims <= 500 ? 50 : 15 ))
  secs=$(( 36 * sims * cp / 500 + 900 ))   # about 2x a chunk's usual time (4 s a pair at 100 simulations), plus 15 min
  secs=${MATCH_CHUNK_SECS:-$secs}
  touch "$out"
  s=$(cat "$out.next" 2>/dev/null || echo 0)
  while [ "$s" -lt "$pairs" ]; do
    e=$(( s + cp < pairs ? s + cp : pairs ))
    rm -f "$out.chunk"
    hardkill "$secs" "$G/bin/dzgorge" play "${COMMON[@]}" -split eval -seed 3131 -first-pair "$s" -pairs $(( e - s )) \
      -workers "${WORKERS:-7}" -a "$a" -b "$b" -out "$out.chunk" >> "${out%.jsonl}.log" 2>&1
    touch "$out.chunk"
    m=$("$PY" -c "import sys; b=open(sys.argv[1],'rb').read(); k=b[:b.rfind(b'\n')+1]; open(sys.argv[1],'wb').write(k); print(k.count(b'\n'))" "$out.chunk")
    cat "$out.chunk" >> "$out"; rm -f "$out.chunk"
    if [ $(( s + m / 2 )) -lt "$e" ] && [ "$m" -lt $(( 2 * (e - s) )) ]; then
      echo "pair $(( s + m / 2 )) did not finish in ${secs}s; skipped $(date)" | tee -a "${out%.jsonl}.skipped"
      s=$(( s + m / 2 + 1 ))
    else
      s=$e
    fi
    echo "$s" > "$out.next"
  done
}

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
    --val evaldecks=data/gorge/runs/r31/val/pack --device "${DEVICE:-auto}" --in-ram "$@" --out "$out" > "$out/train.log" 2>&1 \
    && PYTHONPATH=gorge "$PY" -m dzg.export "$out/best.pt" "$out/best.dzgw" >> "$out/train.log" 2>&1
  tail -1 "$out/train.log"
}

# strength NET.dzgw OUT [PAIRS] [SIMS]: the full search with the network against the zero point (no network)
strength() {
  local net=$1 out=$2 pairs=${3:-500} sims=${4:-100}
  mkdir -p "$(dirname "$out")"
  match "$out.jsonl" "$sims" "$pairs" "az:sims=$sims:$FULL:topk=6:gonet=$net" "az:sims=$sims:$FULL"
}

# h2h NET_A NET_B OUT [PAIRS]: the full search with network A against the full search with network B
h2h() {
  local a=$1 b=$2 out=$3 pairs=${4:-500}
  match "$out.jsonl" 100 "$pairs" "az:sims=100:$FULL:topk=6:gonet=$a" "az:sims=100:$FULL:topk=6:gonet=$b"
}

# alone NET.dzgw OUT [PAIRS]: the network's policy alone against gorge's bot
alone() {
  local net=$1 out=$2 pairs=${3:-1000}
  match "$out.jsonl" 1 "$pairs" "prior:gonet=$net" "bot"
}

# selfplay OUT GAMES SEED [NET.dzgw] [SIMS]: cheap games (record one searched decision in four), packed.
# Played in chunks of CHUNK games, each with its own hard kill: gorge writes games in order, so one game that never
# ends (a decision with a huge option list, seen once with a network in the search) would otherwise hold every later
# game in memory until the kill, and lose them. After a killed chunk, trim.py keeps its complete games; the first
# missing game is replayed alone (PROBE_SECS) and, if it hangs again, logged in OUT/skipped.txt and skipped. Games
# are a pure function of the seed and pair index, so resuming a range reproduces the same games.
_chunk() {  # _chunk OUT FIRST N SEED POLICY SECS -> echoes complete games appended
  local out=$1 first=$2 n=$3 seed=$4 pol=$5 secs=$6
  rm -f "$out/chunk.jsonl" "$out/chunk.gz"
  hardkill "$secs" "$G/bin/dzgorge" play "${COMMON[@]}" -record-features entity -record-every 4 -split train \
    -a "$pol" -b "$pol" -pairs "$n" -first-pair "$first" -workers "${WORKERS:-7}" -seed "$seed" \
    -out "$out/chunk.jsonl" -corpus "$out/chunk.gz" >> "$out/play.log" 2>&1
  touch "$out/chunk.jsonl" "$out/chunk.gz"
  "$PY" gorge/recipe/trim.py "$out/chunk.jsonl" "$out/chunk.gz"
  cat "$out/chunk.jsonl" >> "$out/games.jsonl"
  cat "$out/chunk.gz" >> "$out/visits.jsonl.gz"
  rm -f "$out/chunk.jsonl" "$out/chunk.gz"
}

selfplay() {
  local out=$1 n=$2 seed=$3 net=${4:-} sims=${5:-25} pol s m e chunk=${CHUNK:-500}
  pol="az:sims=$sims:explore:$FULL"
  [ -n "$net" ] && pol="$pol:topk=6:gonet=$net"
  mkdir -p "$out"
  [ -d "$out/pack" ] && { echo "skip $out (packed)"; return 0; }
  s=$(cat "$out/next" 2>/dev/null || echo 0)
  while [ "$s" -lt "$n" ]; do
    e=$(( s + chunk < n ? s + chunk : n ))
    m=$(_chunk "$out" "$s" $(( e - s )) "$seed" "$pol" "${CHUNK_SECS:-900}")
    s=$(( s + m ))
    if [ "$s" -lt "$e" ]; then        # killed: replay the first missing game alone
      m=$(_chunk "$out" "$s" 1 "$seed" "$pol" "${PROBE_SECS:-300}")
      [ "$m" = 0 ] && echo "game $s (seed $seed) did not finish; skipped $(date)" | tee -a "$out/skipped.txt"
      s=$(( s + 1 ))
    fi
    echo "$s" > "$out/next"
  done
  "$G/bin/dzgorge" pack -in "$out/visits.jsonl.gz" -out "$out/pack" -shard 200000 >> "$out/play.log" 2>&1 \
    && "$PY" gorge/recipe/shrink.py "$out/pack" > /dev/null
}

# score FILE.jsonl: side A's paired score and n
score() { "$PY" gorge/recipe/score.py "$@"; }

# vsbot NET.dzgw|"" OUT SIMS [PAIRS]: the full search at SIMS (with the network, or without one) against gorge's
# heuristic bot, the fixed baseline; the same deals (seed 3131) at every level
vsbot() {
  local net=$1 out=$2 sims=$3 pairs=${4:-500} pol
  pol="az:sims=$sims:$FULL"
  [ -n "$net" ] && pol="$pol:topk=6:gonet=$net"
  mkdir -p "$(dirname "$out")"
  match "$out.jsonl" "$sims" "$pairs" "$pol" bot
}
