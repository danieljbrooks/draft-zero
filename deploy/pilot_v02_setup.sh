#!/bin/bash
# pilot_v02_setup.sh — prepare a fresh pod for the v0.2 pilot. Run from the repo root.
#   HF_TOKEN=<read token> bash deploy/pilot_v02_setup.sh
#
# Fetches the generalist XMage bundle for v0.2 (danieljbrooks/mage v0.2-generalist built into
# Will's v0.2.0-alpha release bundle; see build-generalist-bundle.sh there) from the private
# HF repo, rebuilds the deck pool from 17lands' public data (byte-identical to exp #1's), and
# runs the normal bootstrap. No exp #1 checkpoint: v0.1 checkpoints don't load under v0.2.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_TOKEN:?set HF_TOKEN to a token that can read danbrooks/draftzero-checkpoints}"
BUNDLE="${DZ_XMAGE_BUNDLE:-xmage/generalist-xmage-v0.2-48e49184.tar.gz}"

bash deploy/bootstrap.sh

echo "== XMage bundle from the private HF repo: $BUNDLE"
BUNDLE="$BUNDLE" python - <<'PY'
import os, tarfile
from huggingface_hub import hf_hub_download
tgz = hf_hub_download("danbrooks/draftzero-checkpoints", os.environ["BUNDLE"], token=os.environ["HF_TOKEN"])
if not os.path.isdir("xmage/lib"):
    tarfile.open(tgz).extractall(".")
print("   xmage/lib jars:", len(os.listdir("xmage/lib")),
      "| source", open("xmage/GENERALIST_SOURCE_COMMIT").read().strip()[:8])
PY

echo "== deck pool from 17lands public data"
python tools/extract_decks.py --set FDN --format PremierDraft --min-winrate 0.60 --dck \
  --xmage-jar xmage/lib/mage-sets-1.4.58.jar | tail -1
python -m draftzero.cli pools build
python -m draftzero.cli pools check --deck-root data/deckgen/FDN_PremierDraft_wr60/top_player_FDN_decks

echo "== pilot v0.2 setup done"
