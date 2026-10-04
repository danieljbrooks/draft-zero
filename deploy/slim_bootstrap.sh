#!/bin/bash
# slim_bootstrap.sh — make a pod started from RunPod's small base image (runpod/base:1.4.0-ubuntu2404, 0.73 GB
# against the PyTorch image's 10.7 GB) able to run the games and the trainer: a JDK, a venv with torch and this
# repo, and `python` on the PATH. Community hosts often stall for 20+ minutes pulling the big image (docs/005,
# docs/020); this pulls ~1 GB and installs the rest from PyPI.
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash deploy/slim_bootstrap.sh   # a GPU pod
#   bash deploy/slim_bootstrap.sh                                                       # CPU-only torch (default)
# Then run deploy/exp4_games_setup.sh with NO_BOOTSTRAP=1.
set -euo pipefail
cd "$(dirname "$0")/.."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq openjdk-21-jdk-headless python3-venv python3-pip git rsync curl > /dev/null
[ -d /root/venv ] || python3 -m venv /root/venv
# shellcheck disable=SC1091
. /root/venv/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.9.1 --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
pip install -q -e . huggingface_hub
# wrappers, not symlinks: a venv's python reached through a symlink outside the venv doesn't load the venv
for t in python pip; do rm -f "/usr/local/bin/$t"; printf '#!/bin/sh\nexec /root/venv/bin/%s "$@"\n' "$t" > "/usr/local/bin/$t"; chmod +x "/usr/local/bin/$t"; done
python -c "import torch, magezero, draftzero, huggingface_hub; print('slim bootstrap: torch', torch.__version__, 'cuda', torch.cuda.is_available())"
java -version 2>&1 | head -1
