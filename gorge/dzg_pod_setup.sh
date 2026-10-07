#!/usr/bin/env bash
# Set up a fresh GPU pod (RunPod's runpod/pytorch image: Python, torch, git) for dzg:
# Go, this repo's branch, gorge at the pin with the card corpus, and the deck pool.
#
#   scp decks.jsonl pod:/root/   # tools/extract_decks.py's FDN pool (data/deckgen/FDN_PremierDraft_wr60)
#   ssh pod 'bash -s' < gorge/dzg_pod_setup.sh
set -euo pipefail
BRANCH="${BRANCH:-claude/gorge-fdn-selfplay}"
cd /root
if [ ! -x /root/go-sdk/bin/go ]; then
  curl -sSL https://go.dev/dl/go1.25.3.linux-amd64.tar.gz -o /tmp/go.tgz
  mkdir -p /root/go-sdk && tar -C /root/go-sdk --strip-components=1 -xzf /tmp/go.tgz && rm /tmp/go.tgz
fi
export PATH=/root/go-sdk/bin:$PATH
command -v timeout >/dev/null || { echo "need coreutils timeout"; exit 1; }
[ -d /root/dz-gorge/.git ] || git clone -q --branch "$BRANCH" https://github.com/danieljbrooks/draft-zero.git /root/dz-gorge
cd /root/dz-gorge && git pull -q --ff-only
mkdir -p /root/ext
GORGE_DIR=/root/ext/gorge GOFLAGS="-p=$(nproc) -trimpath" bash gorge/build.sh
python3 gorge/decks.py --decks /root/decks.jsonl
python3 -c "import numpy" 2>/dev/null || pip install -q --break-system-packages numpy
python3 -c "import torch, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo POD_SETUP_DONE
