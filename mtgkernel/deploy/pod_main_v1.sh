#!/bin/bash
# The first pod session's work (docs/025): benchmarks, then the baseline ladder, then the GIH self-play runs.
cd "$(dirname "$0")/.."
bash deploy/bench_all.sh runs/bench_v1 1,4,8,14,16 > runs/bench_v1.log 2>&1
bash deploy/run_jobs.sh jobs/ladder_v1.tsv runs/ladder_v1
bash deploy/run_jobs.sh jobs/gih_v1.tsv runs/gih_v1
touch runs/MAIN_V1_DONE
