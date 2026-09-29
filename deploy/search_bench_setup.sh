#!/usr/bin/env bash
# Pod setup for docs/012's search benchmark (tools/search_bench/run.py): the v0.2.0 XMage bundle,
# the bridge, and the two experiment #2 networks it serves. About 5 minutes.
#   HF_TOKEN=<read token> bash deploy/search_bench_setup.sh
# The bundle has no card database: copy xmage/db/cards.h2.mv.db from a laptop build first
# (scp ... root@pod:/root/cards.h2.mv.db), and this script moves it into place.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_TOKEN:?set HF_TOKEN to a token that can read danbrooks/draftzero-checkpoints}"

bash deploy/bootstrap.sh

echo "== XMage bundle, networks"
python - <<'PY'
import os, shutil, tarfile
from huggingface_hub import hf_hub_download
repo, tok = "danbrooks/draftzero-checkpoints", os.environ["HF_TOKEN"]
if not os.path.isdir("xmage/lib"):
    tgz = hf_hub_download(repo, "xmage/generalist-xmage-v0.2-48e49184.tar.gz", token=tok)
    tarfile.open(tgz).extractall(".")
print("   xmage/lib jars:", len(os.listdir("xmage/lib")))
for src, dst in [("2026-09-27_01-59-54/models/FDN_exp2/ver1/gen18.pt.gz", "models/FDN_exp2/ver1/gen18.pt.gz"),
                 ("2026-09-27_03-42-21/models/FDN_exp2_imit/ver1/gen0.pt.gz", "models/FDN_exp2_imit/ver1/gen0.pt.gz")]:
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(hf_hub_download(repo, src, token=tok), dst)
    print("  ", dst, os.path.getsize(dst))
PY
mkdir -p xmage/db
[ -f xmage/db/cards.h2.mv.db ] || mv /root/cards.h2.mv.db xmage/db/cards.h2.mv.db
ls -la xmage/db/cards.h2.mv.db

echo "== bridge"
sh java/mzbridge/build.sh
echo "== search bench setup done"
