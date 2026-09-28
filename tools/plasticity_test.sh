#!/bin/bash
# plasticity_test.sh — does a starting network still learn? (docs/013 §2.3)
#
# Trains four starting networks with MageZero's own train.py on the same self-play data and scores
# each on the same held-out generation, before training (test.py) and after each of 2 epochs:
#   fresh          a new network
#   pretrained     the run's gen0.pt.gz (for exp #2 run 2, the human-pretrained start)
#   lightstart     another run's gen-0 network (exp #2 run 1's, trained on 108 heuristic games)
#   shrinkperturb  the pretrained start after shrink and perturb (Ash & Adams 2020):
#                  theta <- 0.4 * theta + 0.1 * std(theta) * noise
# Run from the repo root on a GPU pod that holds the run's replay shards, after the run stops:
#   bash tools/plasticity_test.sh runs/<run> <model> <train lo> <train hi> <test gen> <light start.pt.gz>
#   e.g. bash tools/plasticity_test.sh runs/2026-09-27_03-42-21 FDN_exp2_imit 8 11 12 /root/run1_gen0.pt.gz
# Consolidated from the two scripts that ran on 2026-09-28 (their logs are plasticity.log and
# pretest.log in run 2's logs.tar.gz). About 8 minutes per start for 55k states on an RTX 3090.
set -uo pipefail
RUN=$1 MODEL=$2 LO=$3 HI=$4 TEST=$5 LIGHT=$6
export MZ_ACTION_VOCAB=${MZ_ACTION_VOCAB:-$PWD/assets/vocab/FDN_SPG.tsv} MZ_TRAIN_BATCH=${MZ_TRAIN_BATCH:-64}
export PYTHONUNBUFFERED=1
MZ=$(python -c "import magezero, os; print(os.path.dirname(magezero.__file__))" 2>/dev/null \
     || python -c "import magezero; print(list(magezero.__path__)[0])")
RUN="$RUN" MODEL="$MODEL" LO="$LO" HI="$HI" TEST="$TEST" LIGHT="$LIGHT" python - <<'PY'
import gzip, io, json, os, shutil
import torch
from magezero.model import load_model
e = os.environ
r = json.load(open(f"{e['RUN']}/run.json"))
sid_gen = {j["sid"]: int(g) for g, rec in r["gens"].items() for j in rec.get("jobs", [])}
src = f"data/{e['MODEL']}/ver1/training"
lo, hi, test = int(e["LO"]), int(e["HI"]), int(e["TEST"])
starts = {"pretrained": f"models/{e['MODEL']}/ver1/gen0.pt.gz", "lightstart": e["LIGHT"]}
ck = load_model(starts["pretrained"])
torch.manual_seed(0)
sd = ck["model_state_dict"]
for k, v in sd.items():
    if v.is_floating_point() and v.dim() >= 1:
        sd[k] = 0.4 * v + 0.1 * torch.randn_like(v) * (v.std() if v.numel() > 1 else 0)
buf = io.BytesIO()
torch.save(ck, buf)
os.makedirs("models/PT_starts", exist_ok=True)
with gzip.open("models/PT_starts/shrinkperturb.pt.gz", "wb", compresslevel=1) as f:
    f.write(buf.getvalue())
starts["shrinkperturb"] = "models/PT_starts/shrinkperturb.pt.gz"
for cfg in ("fresh", "pretrained", "lightstart", "shrinkperturb"):
    for tag in (f"PT_{cfg}", f"PTpre_{cfg}"):
        for sub in ("training", "testing"):
            os.makedirs(f"data/{tag}/ver1/{sub}", exist_ok=True)
        os.makedirs(f"models/{tag}/ver1", exist_ok=True)
        for f in os.listdir(src):
            g = sid_gen.get(int(f.split("_")[0][7:]), -1)
            sub = "training" if lo <= g <= hi else "testing" if g == test else None
            dst = f"data/{tag}/ver1/{sub}/{f}"
            if sub and not os.path.exists(dst):
                os.symlink(os.path.abspath(f"{src}/{f}"), dst)
    if cfg in starts:
        shutil.copyfile(starts[cfg], f"models/PT_{cfg}/ver1/model.pt.gz")
        shutil.copyfile(starts[cfg], f"models/PTpre_{cfg}/ver1/model.pt.gz")
print("setup ok")
PY
for cfg in pretrained lightstart shrinkperturb; do
  echo "=== before training: $cfg"
  python -u "$MZ/test.py" --deck "PTpre_$cfg" --version 1 2>&1 | grep -E "Validation loss|accuracy|ERROR"
done
for cfg in fresh pretrained lightstart shrinkperturb; do
  flag=""; [ "$cfg" != fresh ] && flag="--checkpoint"
  echo "=== train: $cfg $(date -u +%T)"
  python -u "$MZ/train.py" --deck "PT_$cfg" --version 1 --epochs 2 $flag 2>&1 \
    | grep -E "Epoch|Validation loss|accuracy|vocab|Error"
done
echo "=== done $(date -u +%T)"
