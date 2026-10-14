#!/usr/bin/env bash
# Set up a machine as a sealed game worker (docs/032): Go, this repo's sealed-deckbuilding branch at $BASE/dz-sealed,
# gorge at the pin with the patches and the card corpus at $BASE/ext/gorge-sealed, and a Python with numpy.
#
#   ssh pod 'bash -s' < gorge/sealed/pod_setup.sh                    # RunPod (runpod/pytorch image), BASE=/root
#   ssh d1 'BASE=/work VENV=1 bash -s' < gorge/sealed/pod_setup.sh   # a plain box: everything under /work, a venv
#   rsync -a data/sealed/{sealed.jsonl,draft.jsonl,nets} host:$BASE/dz-sealed/data/sealed/
set -euo pipefail
BRANCH="${BRANCH:-sealed-deckbuilding}"
BASE="${BASE:-/root}"
cd "$BASE"
if [ ! -x "$BASE/go-sdk/bin/go" ]; then
  curl -sSL https://go.dev/dl/go1.25.3.linux-amd64.tar.gz -o /tmp/go.tgz
  mkdir -p "$BASE/go-sdk" && tar -C "$BASE/go-sdk" --strip-components=1 -xzf /tmp/go.tgz && rm /tmp/go.tgz
fi
export PATH=$BASE/go-sdk/bin:$PATH GOPATH=$BASE/gopath GOCACHE=$BASE/.gocache
command -v timeout >/dev/null || { echo "need coreutils timeout"; exit 1; }
command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync > /dev/null)
[ -d "$BASE/dz-sealed/.git" ] || git clone -q --branch "$BRANCH" https://github.com/danieljbrooks/draft-zero.git "$BASE/dz-sealed"
cd "$BASE/dz-sealed" && git pull -q --ff-only
mkdir -p "$BASE/ext" "$BASE/dz-sealed/data/sealed"
GORGE_DIR="$BASE/ext/gorge-sealed" GOFLAGS="-p=$(nproc) -trimpath" bash gorge/build.sh
if [ -n "${VENV:-}" ]; then
  [ -x "$BASE/venv/bin/python" ] || python3 -m venv "$BASE/venv"
  "$BASE/venv/bin/python" -c "import numpy" 2>/dev/null || "$BASE/venv/bin/pip" install -q numpy
  "$BASE/venv/bin/python" -c "import numpy; print('numpy', numpy.__version__)"
else
  python3 -c "import numpy" 2>/dev/null || pip install -q --break-system-packages numpy scipy
  python3 -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__)"
fi
echo POD_SETUP_DONE
