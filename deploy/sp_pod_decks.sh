#!/usr/bin/env bash
# docs/028 on a rented pod (after deploy/exp4_games_setup.sh): the training decks for self-play, from the HF repo
# (sp028/decks/: top_player_FDN_decks.tgz and train.txt). Needs HF_TOKEN.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
import os, shutil, tarfile
from huggingface_hub import hf_hub_download
repo, tok = "danbrooks/draftzero-checkpoints", os.environ["HF_TOKEN"]
tarfile.open(hf_hub_download(repo, "sp028/decks/top_player_FDN_decks.tgz", token=tok)).extractall(
    "data/deckgen/FDN_PremierDraft_wr60")
os.makedirs("data/pools", exist_ok=True)
shutil.copy(hf_hub_download(repo, "sp028/decks/train.txt", token=tok), "data/pools/train.txt")
PY
echo "decks: $(ls data/deckgen/FDN_PremierDraft_wr60/top_player_FDN_decks | wc -l), train pool: $(wc -l < data/pools/train.txt)"
