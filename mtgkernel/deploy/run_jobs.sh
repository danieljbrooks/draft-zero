#!/bin/bash
# run_jobs.sh JOBS_FILE [RUNS_DIR]: run `dzk play` jobs one after another, each under a hard wall-clock kill.
# JOBS_FILE lines: `name<TAB>max_seconds<TAB>dzk play arguments (without --out)`; blank lines and # comments skipped.
# Each job writes RUNS_DIR/name/games.jsonl (+ summary.json, stderr.log) and resumes if the file exists, so the
# whole file can be re-run after a kill. Writes RUNS_DIR/JOBS_DONE when every job has finished.
#   bash deploy/run_jobs.sh jobs/ladder.tsv runs/ladder    (from mtgkernel/)
set -uo pipefail
cd "$(dirname "$0")/.."
JOBS=$1
RUNS=${2:-runs}
DZK=${DZK:-./target/release/dzk}
mkdir -p "$RUNS"
rm -f "$RUNS/JOBS_DONE"
while IFS=$'\t' read -r name secs args; do
  [[ -z "${name// }" || "$name" == \#* ]] && continue
  out="$RUNS/$name"
  mkdir -p "$out"
  resume=()
  [ -s "$out/games.jsonl" ] && resume=(--resume)
  echo "$(date -u +%FT%TZ) start $name ($secs s): $args" | tee -a "$RUNS/jobs.log"
  # shellcheck disable=SC2086
  timeout --signal=TERM --kill-after=30 "$secs" "$DZK" play $args --out "$out/games.jsonl" "${resume[@]}" \
    2> "$out/stderr.log"
  rc=$?
  echo "$(date -u +%FT%TZ) end $name rc=$rc games=$(wc -l < "$out/games.jsonl" 2>/dev/null || echo 0)" | tee -a "$RUNS/jobs.log"
done < "$JOBS"
touch "$RUNS/JOBS_DONE"
