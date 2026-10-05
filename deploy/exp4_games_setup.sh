#!/usr/bin/env bash
# Pod setup for experiment #4's games and search bench (docs/018, phase C): the v0.2 XMage bundle and its card
# database, the bridge, the eval pool's decks with the belief service's decks.jsonl, sb-v2's items, and a network, all from the project's HF
# repo. About 5 minutes on a RunPod pod (from the repo root).
#   HF_TOKEN=<read token> bash deploy/exp4_games_setup.sh [<checkpoint in the HF repo>]
# The checkpoint defaults to stage 3's best_policy; it lands in models/exp4/ under its repo path.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_TOKEN:?set HF_TOKEN to a token that can read danbrooks/draftzero-checkpoints}"
export CKPT=${1:-exp4/stage3/best_policy.pt.gz}
export HF_HUB_DISABLE_PROGRESS_BARS=1

# NO_BOOTSTRAP=1 on a box whose Python and JDK are already set up and shared (the second machine): bootstrap's
# editable install would repoint that venv's draftzero at this checkout
[ -n "${NO_BOOTSTRAP:-}" ] || bash deploy/bootstrap.sh

echo "== XMage bundle, card database, decks, network"
python - <<'PY'
import os, shutil, tarfile
from huggingface_hub import hf_hub_download
repo, tok = "danbrooks/draftzero-checkpoints", os.environ["HF_TOKEN"]
get = lambda f: hf_hub_download(repo, f, token=tok)   # noqa: E731
if not os.path.isdir("xmage/lib"):
    tarfile.open(get("xmage/generalist-xmage-v0.2-48e49184.tar.gz")).extractall(".")
os.makedirs("xmage/db", exist_ok=True)
if not os.path.exists("xmage/db/cards.h2.mv.db"):
    shutil.copy(get("exp4/games/cards.h2.mv.db"), "xmage/db/cards.h2.mv.db")
# HF's cache files are read-only and the copy keeps the mode; H2 refuses a read-only database (root never notices)
os.chmod("xmage/db/cards.h2.mv.db", 0o644)
if not os.path.exists("data/pools/eval.txt"):          # play.py's default pool; its order fixes the deck pairs
    os.makedirs("data/pools", exist_ok=True)
    shutil.copy(get("exp4/games/eval.txt"), "data/pools/eval.txt")
for f in ("items.jsonl.gz", "build.json"):            # sb-v2: the 1,000 held-out decisions (tools/search_bench/run.py)
    if not os.path.exists(f"data/search_bench/sb-v2/{f}"):
        os.makedirs("data/search_bench/sb-v2", exist_ok=True)
        shutil.copy(get(f"exp4/games/sb-v2/{f}"), f"data/search_bench/sb-v2/{f}")
root = "data/deckgen/FDN_PremierDraft_wr60"
if not os.path.exists(f"{root}/decks.jsonl"):
    os.makedirs(root, exist_ok=True)
    tarfile.open(get("exp4/games/eval_decks.tgz")).extractall(root)
# the belief service's deck pool (belief.DeckPool's cache; without it every sample fails and closed decklists fall
# back to the real decklist, as in experiment #4's games)
if not os.path.exists("data/gameplay/deckpool_FDN_PremierDraft.npz"):
    os.makedirs("data/gameplay", exist_ok=True)
    shutil.copy(get("exp4/games/deckpool_FDN_PremierDraft.npz"), "data/gameplay/deckpool_FDN_PremierDraft.npz")
dst = os.path.join("models", os.environ["CKPT"])
if not os.path.exists(dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(get(os.environ["CKPT"]), dst)
print("   xmage/lib jars:", len(os.listdir("xmage/lib")), "| decks:", len(os.listdir(f"{root}/top_player_FDN_decks")),
      "| network:", dst, os.path.getsize(dst))
PY

echo "== bridge"
MZ_XMAGE_DIR=$PWD/xmage sh java/mzbridge/build.sh
echo "== exp4 games setup done"
