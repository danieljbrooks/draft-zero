#!/bin/bash
# freeze.sh NAME: copy gorge/sealed to data/sealed/code/NAME/sealed, so a long run keeps its code while the live code
# changes (spawned build workers re-import from disk). Run from inside that directory with DZ_REPO set:
#   cd data/sealed/code/NAME && DZ_REPO=<checkout> python -m sealed.loop <absolute paths>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
D="$ROOT/data/sealed/code/$1"
rm -rf "$D"; mkdir -p "$D"
cp -r "$ROOT/gorge/sealed" "$D/sealed"; rm -rf "$D/sealed/__pycache__"
(cd "$ROOT" && git rev-parse --short HEAD) > "$D/COMMIT"
echo "$D"
