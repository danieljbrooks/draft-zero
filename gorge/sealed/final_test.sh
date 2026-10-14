#!/bin/bash
# docs/032's final numbers: the builders chosen on val pools, scored once on the TEST pools.
#   bash gorge/sealed/final_test.sh ENS.pt WORKERS [SIMS...]      (default SIMS: 100)
# Writes data/sealed/eval/test-<builder>-<kind>-s<sims>/result.json for the self-play builder (ENS.pt), the 17lands
# heuristic and the imitation reference, on all 1,000 sealed test pools (2,000 games each) and the 1,000 draft test
# pools. Resumable: evalbuild skips games already played.
set -euo pipefail
ENS=$1; W=$2; shift 2
SIMS=${*:-100}
cd "$(dirname "$0")/.."
PY=${PY:-python}
E=../data/sealed/eval
for s in $SIMS; do
  for kind in sealed draft; do
    $PY -m sealed.evalbuild $E/test-selfplay-$kind-s$s --builder ens:$ENS --kind $kind --split test --pools 1000 --sims $s --workers $W --procs 8
    $PY -m sealed.evalbuild $E/test-gih-$kind-s$s --builder gih --kind $kind --split test --pools 1000 --sims $s --workers $W
    $PY -m sealed.evalbuild $E/test-imit-$kind-s$s --builder ens:../data/sealed/imit/ens.pt --kind $kind --split test --pools 1000 --sims $s --workers $W --procs 8
  done
done
