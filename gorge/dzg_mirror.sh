#!/usr/bin/env bash
# Mirror the small result files (game summaries, training curves and configs, logs) from every machine
# into data/gorge/results/<machine>/runs/, for dzg_collect.py and the figures. No checkpoints, no corpora.
#
#   bash gorge/dzg_mirror.sh research=dz-gorge dzpod=/root/dz-gorge dz3090=/root/dz-gorge
set -uo pipefail
DZ="$(cd "$(dirname "$0")/.." && pwd)"
for spec in "$@"; do
  host="${spec%%=*}"; repo="${spec#*=}"; name="$host"; [ "$host" = research ] && name=r1
  mkdir -p "$DZ/data/gorge/results/$name"
  rsync -a --prune-empty-dirs --include='*/' --include='bench-*/*.jsonl' --include='*.summary.json' --include='curves.jsonl' \
    --include='config.json' --include='*.log' --include='host.txt' --include='train-*.json' --exclude='*' \
    "$host:$repo/data/gorge/runs" "$DZ/data/gorge/results/$name/" || echo "mirror $host failed"
done
