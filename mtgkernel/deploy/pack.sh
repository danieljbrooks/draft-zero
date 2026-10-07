#!/bin/bash
# pack.sh [out.tgz]: bundle what a pod needs to build and run dzk — the mtgkernel/ folder (no build outputs, no
# mtg-kernel clone), the fixture decks, the 17lands reference and draft-zero's GIH tool. The pod clones mtg-kernel
# itself (mtgkernel/setup.sh). Nothing is pushed anywhere: the bundle goes to the pod over scp.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${1:-/tmp/dzk_bundle.tgz}
tar czf "$OUT" --exclude='mtgkernel/target' --exclude='mtgkernel/.deps' --exclude='mtgkernel/runs' \
  --exclude='__pycache__' --exclude='*.pt' --exclude='mtgkernel/data_out' \
  mtgkernel assets/reference/FDN_gih.json assets/sample/decks/FDN_top_04956_UG.dck \
  assets/sample/decks/FDN_top_20626_WG.dck tools/imitation_scale/gih.py deploy/runpod_arm.sh
ls -la "$OUT"
