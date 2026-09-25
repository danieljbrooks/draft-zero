#!/bin/bash
# pilot_setup.sh — prepare a fresh pod for the experiment #2 pilot. Run from the repo root.
#   HF_TOKEN=<read token> bash deploy/pilot_setup.sh
#
# Fetches exp #1's engine (v0.1 generalist XMage build, not public) and exp #1's gen-33
# checkpoint from the private HF repo, rebuilds the deck pool from 17lands' public data
# (byte-identical to exp #1's), and runs the normal bootstrap.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_TOKEN:?set HF_TOKEN to a token that can read danbrooks/draftzero-checkpoints}"

bash deploy/bootstrap.sh

echo "== engine and checkpoint from the private HF repo"
python - <<'PY'
import os, shutil, tarfile
from huggingface_hub import hf_hub_download
repo = "danbrooks/draftzero-checkpoints"
tgz = hf_hub_download(repo, "pilot/generalist-xmage-v0.1-5a32441c.tar.gz", token=os.environ["HF_TOKEN"])
if not os.path.isdir("xmage/lib"):
    tarfile.open(tgz).extractall(".")
print("   xmage/lib jars:", len(os.listdir("xmage/lib")))
ck = hf_hub_download(repo, "2026-09-22_02-46-01/models/FDN_generalist/ver1/gen33.pt.gz", token=os.environ["HF_TOKEN"])
os.makedirs("models/FDN_generalist/ver1", exist_ok=True)
shutil.copy(ck, "models/FDN_generalist/ver1/gen33.pt.gz")
print("   exp #1 gen33 checkpoint in place")
PY

echo "== deck pool from 17lands public data"
python tools/extract_decks.py --set FDN --format PremierDraft --min-winrate 0.60 --dck \
  --xmage-jar xmage/lib/mage-sets-1.4.58.jar | tail -1
python -m draftzero.cli pools build
python -m draftzero.cli pools check --deck-root data/deckgen/FDN_PremierDraft_wr60/top_player_FDN_decks

echo "== pilot setup done"
