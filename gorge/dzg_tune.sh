#!/usr/bin/env bash
# Tuning arms: paired games of any policy against any other on the eval decks, with the dzg network
# NET served once and written into a spec wherever NET appears (as remote=ADDR).
#
#   NET=best.pt OUT=data/gorge/runs/tune/x PAIRS=300 bash gorge/dzg_tune.sh \
#     "autopay=az:sims=100:autopay:NET~az:sims=100" "cands12=az:sims=100:cands=12:NET~az:sims=100"
#
# Each argument is NAME=SPEC_A~SPEC_B. Every arm uses the same seed, so every arm plays the same deck pairs.
# SERVE_ARGS passes extra server arguments (--prior-temp, --value-temp); NET2 is a second network (NET2 in
# a spec), served with SERVE_ARGS2. EXTRA_PLAY adds dzgorge play flags to every arm (e.g. -mulligans 2 -mull-heuristic).
set -euo pipefail
source "$(dirname "$0")/dzg_lib.sh"
OUT="${OUT:?OUT=dir}"; PAIRS="${PAIRS:-300}"; W="${W:-60}"; SEED="${SEED:-77}"; MAXMIN="${MAXMIN:-45}"
mkdir -p "$OUT"
addr=""; addr2=""
if [ -n "${NET:-}" ]; then serve "$NET" "$OUT/serve"; addr="$ADDR"; fi
if [ -n "${NET2:-}" ]; then SERVE_ARGS="${SERVE_ARGS2:-}" serve "$NET2" "$OUT/serve2"; addr2="$ADDR"; fi
for arm in "$@"; do
  name="${arm%%=*}"; specs="${arm#*=}"; a="${specs%%~*}"; b="${specs#*~}"
  a="${a//NET2/remote=$addr2}"; b="${b//NET2/remote=$addr2}"
  a="${a//NET/remote=$addr}"; b="${b//NET/remote=$addr}"
  echo "== $name: $a  vs  $b ($PAIRS pairs)"
  # shellcheck disable=SC2086
  play "$MAXMIN" -split eval -a "$a" -b "$b" -pairs "$PAIRS" -workers "$W" -seed "$SEED" ${EXTRA_PLAY:-} -out "$OUT/$name.jsonl" \
    > "$OUT/$name.log" 2>&1 || echo "  ($name stopped: exit $?)"
  { grep -E '"(a_score|games|games_per_hour|az_decisions_searched|az_ms_per_searched)"' "$OUT/$name.jsonl.summary.json" 2>/dev/null \
    || echo "  (no summary)"; } | tr -d '\n'; echo
done
