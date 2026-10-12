#!/usr/bin/env bash
# Set up a fresh RunPod pod (runpod/pytorch image: Python, torch, git) as a sealed game worker (docs/032):
# Go, this repo's sealed-deckbuilding branch at /root/dz-sealed, gorge at the pin with the patches and the card corpus.
#
#   ssh pod 'bash -s' < gorge/sealed/pod_setup.sh
#   rsync -a data/sealed/{sealed.jsonl,draft.jsonl,nets} pod:/root/dz-sealed/data/sealed/
set -euo pipefail
BRANCH="${BRANCH:-sealed-deckbuilding}"
cd /root
if [ ! -x /root/go-sdk/bin/go ]; then
  curl -sSL https://go.dev/dl/go1.25.3.linux-amd64.tar.gz -o /tmp/go.tgz
  mkdir -p /root/go-sdk && tar -C /root/go-sdk --strip-components=1 -xzf /tmp/go.tgz && rm /tmp/go.tgz
fi
export PATH=/root/go-sdk/bin:$PATH
command -v timeout >/dev/null || { echo "need coreutils timeout"; exit 1; }
command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync > /dev/null)
[ -d /root/dz-sealed/.git ] || git clone -q --branch "$BRANCH" https://github.com/danieljbrooks/draft-zero.git /root/dz-sealed
cd /root/dz-sealed && git pull -q --ff-only
mkdir -p /root/ext /root/dz-sealed/data/sealed
GORGE_DIR=/root/ext/gorge-sealed GOFLAGS="-p=$(nproc) -trimpath" bash gorge/build.sh
python3 -c "import numpy" 2>/dev/null || pip install -q --break-system-packages numpy scipy
python3 -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__)"
echo POD_SETUP_DONE
