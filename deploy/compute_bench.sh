#!/usr/bin/env bash
# The compute benchmark (docs/020) on one machine: the MLP's inference speed, self-play games with search (the
# MLP's policy as prior, its value at the leaves, both seats), search decisions at higher budgets, and training
# throughput, with the machine's facts and load. Run after the games setup with the benchmark's MLP:
#   HF_TOKEN=<token> bash deploy/exp4_games_setup.sh exp4/bench/mlp-d1-s30-combined/best_policy_fp16.pt.gz
#   HF_TOKEN=<token> PRICE=0.24 OFFER="..." bash deploy/compute_bench.sh <tag> [phase ...]
# Phases, in this order by default: sysinfo infer games sb tables train upload
#   sysinfo  tools/compute_bench/sysinfo.py: CPU model, cgroup cores and memory, GPU, the quote (PRICE, OFFER)
#   infer    the MLP's evaluations a second (supervised bench-speed): CPU at 1 thread, the GPU if any
#   games    self-play il_bc@$BUDGET (closed decklists, records kept) for $GAMES_MIN minutes: $WORKERS JVMs of
#            $HEAP, $REPLICAS inference servers on $SERVER_DEVICE (cpu or cuda), load every 10 s
#            (tools/compute_bench/monitor.py). Unfinished games are dropped when the time is up
#   sb       sb-v2's held-out decisions (the same $SB_LIMIT on every machine) at each of $SB_BUDGETS simulations,
#            with each of $SB_METHODS (default ismcts; "pimc1 ismcts" compares PIMC on one belief world, docs/021)
#   tables   the games' records -> soft tables (tools/imitation_scale/selfplay_tables.py)
#   train    the trainer from the MLP, $TRAIN_STEPS steps: stage 6's recipe on those tables (configs/
#            compute_bench_soft.yml) and the imitation recipe on two small 17lands tables (compute_bench_human.yml),
#            on the CPU at each of $TRAIN_THREADS threads, then on the GPU if any
#   games_cpuserve, games_gc, games_more, games_fewer, sb10k: variants for the hypotheses (not in the default list)
#   tstep    (not in the default list) where a training step goes: the trainer's step on real batches split into
#            forward, backward and optimizer, as it is, with fused AdamW, and with the feature table frozen (the
#            limit of a sparse table update), on the CPU at $TSTEP_THREADS and on the GPU if any
#   upload   everything but records, tables and JVM logs to the HF repo, exp4/bench/runs/<tag>/
# Results: runs/compute_bench/<tag>/; summarise with tools/compute_bench/summarize.py.
set -uo pipefail
cd "$(dirname "$0")/.."
TAG=$1; shift
PHASES=${*:-${PHASES:-sysinfo cpuprobe infer games sb tables train upload}}   # arguments, else $PHASES, else these
export MZ_XMAGE_DIR=${MZ_XMAGE_DIR:-$PWD/xmage} MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv HF_HUB_DISABLE_PROGRESS_BARS=1
MODEL=${MODEL:-models/exp4/bench/mlp-d1-s30-combined/best_policy_fp16.pt.gz}
DECKS=data/deckgen/FDN_PremierDraft_wr60/top_player_FDN_decks
OUT=runs/compute_bench/$TAG
mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%S)] bench $TAG: $*" | tee -a "$OUT/bench.log"; }
tmo() {   # tmo <seconds> cmd...: GNU timeout where there is one, perl's alarm elsewhere (macOS)
  if command -v timeout > /dev/null; then timeout "$@"; else local t=$1; shift; perl -e 'alarm(shift); exec(@ARGV)' "$t" "$@"; fi
}

HAS_GPU=$(python -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null || echo 0)   # torch's view: a CPU wheel on a GPU pod has none
# the cores the container may use: its CFS quota, or (CPU pods pin a cpuset instead of a quota) its CPU affinity
QUOTA=$(python -c "import os; from draftzero.resources import cpu_quota; aff = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count(); print(int(round(min(cpu_quota() or os.cpu_count(), aff))))")
MEM=$(python -c "from draftzero.resources import mem_limit_gb; import psutil; print(int(mem_limit_gb() or psutil.virtual_memory().total / 2**30))")
SERVER_DEVICE=${SERVER_DEVICE:-$([ $HAS_GPU = 1 ] && echo cuda || echo cpu)}
REPLICAS=${REPLICAS:-4}
BUDGET=${BUDGET:-100}
HEAP=${HEAP:-1500m}
WORKERS=${WORKERS:-$(( QUOTA - REPLICAS > 2 ? QUOTA - REPLICAS : 2 ))}
# a JVM's RSS can reach its -Xmx plus ~0.45 GB (metaspace, JIT code, GC): cap the workers by memory too, leaving
# 1.5 GB a server and 3 GB for the rest
wcap() {   # wcap <heap> <workers>: the workers that fit in memory at that heap
  python -c "h='$1'.lower(); g=float(h[:-1]) / (1024 if h.endswith('m') else 1); print(min($2, max(2, int(($MEM - 3 - 1.5 * $REPLICAS) / (g + 0.45)))))"
}
WMEM=$(wcap "$HEAP" 100000)
[ "$WORKERS" -gt "$WMEM" ] && WORKERS=$WMEM
GAMES_MIN=${GAMES_MIN:-25}
SB_BUDGETS=${SB_BUDGETS:-100 1000}
SB_HEAP=${SB_HEAP:-2500m}
SB_LIMIT=${SB_LIMIT:-112}
SB_METHODS=${SB_METHODS:-ismcts}
TRAIN_STEPS=${TRAIN_STEPS:-400}
TRAIN_THREADS=${TRAIN_THREADS:-$QUOTA $(( QUOTA / 2 ))}
GPU_TRAIN_STEPS=${GPU_TRAIN_STEPS:-3000}
log "phases: $PHASES | quota $QUOTA cores, $MEM GB | gpu $HAS_GPU | servers $REPLICAS on $SERVER_DEVICE | workers $WORKERS, heap $HEAP"

PORTS=""
start_servers() {   # (re)start $REPLICAS servers on $SERVER_DEVICE unless they already answer
  PORTS=""
  local dev_env=""; [ "$SERVER_DEVICE" = cpu ] && dev_env="CUDA_VISIBLE_DEVICES="
  for i in $(seq 0 $((REPLICAS - 1))); do
    local p=$((50052 + 100 * i)); PORTS="${PORTS:+$PORTS,}$p"
    curl -s -m 2 "localhost:$p/healthz" > /dev/null && continue
    env $dev_env nohup python tools/search_bench/value_server.py --model "$MODEL" --port "$p" --threads 8 --policy \
      > "$OUT/server_$p.log" 2>&1 < /dev/null &
  done
  for p in ${PORTS//,/ }; do
    for k in $(seq 1 120); do curl -s -m 2 "localhost:$p/healthz" > /dev/null && break; sleep 2; done
  done
  curl -s -m 2 localhost:50070/healthz > /dev/null || \
    nohup python tools/imitation_scale/belief_server.py --port 50070 > "$OUT/belief.log" 2>&1 < /dev/null &
  for k in $(seq 1 120); do curl -s -m 2 localhost:50070/healthz > /dev/null && break; sleep 2; done
  log "servers on $PORTS ($(grep -ho 'device=[a-z]*' "$OUT"/server_*.log | sort | uniq -c | tr '\n' ' '))"
}
stop_servers() {
  pkill -f "[v]alue_server.py --model" 2>/dev/null; pkill -f "[b]elief_server.py --port" 2>/dev/null; sleep 2
}

phase_sysinfo() { python tools/compute_bench/sysinfo.py --out "$OUT/sysinfo.json" ${PRICE:+--price $PRICE} ${OFFER:+--offer "$OFFER"} > /dev/null; }
phase_cpuprobe() {   # what a vCPU is: one alone, two at once (SMT siblings?), all at once (tools/compute_bench/cpuprobe.py)
  python tools/compute_bench/cpuprobe.py --out "$OUT/cpuprobe.json" > /dev/null 2>&1
  log "cpuprobe: $(python -c "import json; d=json.load(open('$OUT/cpuprobe.json')); print({k: v for k, v in d.items() if k.startswith('score') or k in ('cpuset', 'affinity_cpus')})" 2>&1)"
}

phase_infer() {
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m draftzero.gameplay.supervised bench-speed --checkpoint "$MODEL" \
    --device cpu --tokens 816 --batches 1,8,32 --seconds 3 --json "$OUT/infer_cpu_t1.json" > "$OUT/infer_cpu_t1.log" 2>&1
  if [ $HAS_GPU = 1 ]; then
    python -m draftzero.gameplay.supervised bench-speed --checkpoint "$MODEL" --device cuda --tokens 816 \
      --batches 1,8,32,128 --seconds 3 --json "$OUT/infer_cuda.json" > "$OUT/infer_cuda.log" 2>&1
  fi
  log "infer: $(grep -h 'batch' "$OUT"/infer_*.log | tr -s ' ' | tr '\n' ';')"
}

phase_games() {
  local D=$OUT/games_b${BUDGET}${GAMES_SUFFIX:-}
  mkdir -p "$D"
  start_servers
  python tools/compute_bench/monitor.py --out "$D/load.jsonl" & local MON=$!
  date +%s > "$D/t0"
  # a process group of its own, so the JVMs go with it when the time is up
  local SETSID=""; command -v setsid > /dev/null && SETSID=setsid
  MZB_JAVA_OPTS="${JAVA_OPTS:-}" $SETSID python tools/imitation_scale/play.py --deck-root "$DECKS" --pairs 400 \
    --seed 20261003 --bot1 "il_bc@$BUDGET" --bot2 "il_bc@$BUDGET" --ports "$PORTS" --workers "$WORKERS" \
    --heap "$HEAP" --record --game-timeout 14400 --out "$D" > "$D/play.log" 2>&1 < /dev/null &
  local PLAY=$!
  echo "{\"budget\": $BUDGET, \"workers\": $WORKERS, \"replicas\": $REPLICAS, \"server_device\": \"$SERVER_DEVICE\", \"heap\": \"$HEAP\", \"java_opts\": \"${JAVA_OPTS:-}\", \"minutes\": $GAMES_MIN, \"quota\": $QUOTA, \"mem_gb\": $MEM}" > "$D/bench.json"
  log "games: il_bc@$BUDGET self-play, $WORKERS workers, $GAMES_MIN minutes (pid $PLAY)"
  local end=$(( $(date +%s) + GAMES_MIN * 60 ))
  while [ "$(date +%s)" -lt $end ] && kill -0 $PLAY 2>/dev/null; do sleep 15; done
  date +%s > "$D/t1"
  # then let every game started by now finish, new games keeping the load up meanwhile, so the games started in
  # the measured window all count, short and long alike (summarize.py: the start-window estimate)
  local started=$(( $(grep -c . "$D/games.jsonl" 2>/dev/null || echo 0) + WORKERS ))
  echo "$started" > "$D/started_by_t1"
  local drain_end=$(( $(date +%s) + ${GAMES_DRAIN_MAX:-30} * 60 ))
  while [ "$(date +%s)" -lt $drain_end ] && kill -0 $PLAY 2>/dev/null; do
    python - "$D" "$started" <<'PY' && break
import json, sys
d, n = sys.argv[1], int(sys.argv[2])
cfg = json.load(open(f"{d}/config.json"))
done = {(g["pair"], g["swap"]) for g in map(json.loads, open(f"{d}/games.jsonl"))}   # errors are finished too
first = [(k, s) for k in range(cfg["pairs"]) for s in (False, True)][:n]   # play.py starts its tasks in this order
sys.exit(0 if all(t in done for t in first) else 1)
PY
    sleep 15
  done
  date +%s > "$D/t2"
  if [ -n "$SETSID" ]; then kill -INT -- -$PLAY 2>/dev/null; sleep 8; kill -KILL -- -$PLAY 2>/dev/null
  else kill -INT $PLAY 2>/dev/null; sleep 8; pkill -KILL -f "[o]rg.draftzero.mzbridge.Worker" 2>/dev/null; kill -KILL $PLAY 2>/dev/null; fi
  sleep 2
  kill $MON 2>/dev/null
  log "games: $(grep -c . "$D/games.jsonl" 2>/dev/null || echo 0) finished"
}

# variants of the games phase, each in its own directory (games_b<budget><suffix>), for the hypotheses (docs/020 §6)
phase_games_cpuserve() {   # the network served on the CPU (what a CPU-only pod does), same workers
  stop_servers; SERVER_DEVICE=cpu GAMES_SUFFIX=_cpuserve phase_games; stop_servers
}
phase_games_gc() {         # each JVM with 2 parallel GC threads, 1 concurrent and 2 JIT compiler threads
  JAVA_OPTS="-XX:ParallelGCThreads=2 -XX:ConcGCThreads=1 -XX:CICompilerCount=2" GAMES_SUFFIX=_gc phase_games
}
phase_games_more() {       # 25% more workers than the quota's default (SMT siblings, memory permitting)
  local W0=$WORKERS; WORKERS=$(( WORKERS * 5 / 4 )); [ "$WORKERS" -gt "$WMEM" ] && WORKERS=$WMEM
  GAMES_SUFFIX=_w$WORKERS phase_games; WORKERS=$W0
}
phase_games_fewer() {      # 25% fewer workers
  local W0=$WORKERS; WORKERS=$(( WORKERS * 3 / 4 ))
  GAMES_SUFFIX=_w$WORKERS phase_games; WORKERS=$W0
}
phase_sb10k() {            # 10,000 simulations: a round of decisions, one per worker, the JVMs logging their GCs
  mkdir -p "$OUT/sb"
  JAVA_OPTS="${JAVA_OPTS:-} -Xlog:gc:file=gc.log:uptime" SB_BUDGETS=10000 phase_sb
  python - "$OUT" <<'PY' > "$OUT/sb/gc10k.json" 2>/dev/null || true
import glob, json, re, sys
peak = {}
for f in glob.glob("data/mzbridge/runtime/sb_sb_*/gc.log"):
    after = [int(m.group(2)) for m in re.finditer(r"(\d+)M->(\d+)M\(", open(f).read())]
    if after:
        peak[f.split("/")[-2]] = max(after)
v = sorted(peak.values())
print(json.dumps({"jvms": len(v), "max_live_mb_after_gc": v[-1] if v else None,
                  "median_peak_mb": v[len(v) // 2] if v else None, "peaks_mb": v}, indent=1))
PY
  log "sb10k gc: $(head -c 200 "$OUT/sb/gc10k.json" | tr '\n' ' ')"
}

phase_sb() {   # the budgets below 10,000 in one run.py call (one pool: the first budget warms the JVMs for the rest)
  mkdir -p "$OUT/sb"
  start_servers
  python tools/compute_bench/monitor.py --out "$OUT/sb/load_monitor.jsonl" & local MON=$!
  local small="" big=""
  for B in $SB_BUDGETS; do if [ "$B" -ge 10000 ]; then big="$big $B"; else small="${small:+$small,}$B"; fi; done
  if [ -n "$small" ]; then
    local Wk; Wk=$(wcap "$SB_HEAP" "$WORKERS")
    log "sb: $SB_LIMIT decisions at $small simulations ($SB_METHODS), $Wk workers, heap $SB_HEAP"
    # SB_MAX_MIN caps a small machine's run: the decisions done by then are kept (each is compared item by item)
    MZB_JAVA_OPTS="${JAVA_OPTS:-}" tmo $(( ${SB_MAX_MIN:-600} * 60 )) python tools/search_bench/run.py --items data/search_bench/sb-v2 \
      --split test --out "$OUT/sb" --evaluator remote --ports "$PORTS" --grid priors --methods "${SB_METHODS// /,}" --budgets "$small" \
      --leaf net --net mlpd1 --workers "$Wk" --heap "$SB_HEAP" --limit "$SB_LIMIT" >> "$OUT/sb/run.log" 2>&1
    pkill -f "[o]rg.draftzero.mzbridge.Worker" 2>/dev/null
    log "sb: $(grep finished "$OUT/sb/run.log" | tail -2 | tr '\n' ' ') ($(cat "$OUT"/sb/decisions/*.jsonl 2>/dev/null | wc -l) decisions)"
  fi
  for B in $big; do
    local H=${SB_HEAP_10K:-12g} Wk; Wk=$(wcap "${SB_HEAP_10K:-12g}" "${SB_WORKERS_10K:-$WORKERS}")
    local L=${SB_LIMIT_10K:-$Wk}
    log "sb: $L decisions at $B simulations, $Wk workers, heap $H"
    MZB_JAVA_OPTS="${JAVA_OPTS:-}" python tools/search_bench/run.py --items data/search_bench/sb-v2 --split test \
      --out "$OUT/sb" --evaluator remote --ports "$PORTS" --grid priors --methods "${SB_METHODS// /,}" --budgets "$B" --leaf net \
      --net mlpd1 --workers "$Wk" --heap "$H" --limit "$L" >> "$OUT/sb/run.log" 2>&1
    log "sb: $(grep finished "$OUT/sb/run.log" | tail -1)"
  done
  kill $MON 2>/dev/null
}

phase_tables() {
  python tools/imitation_scale/selfplay_tables.py --records "$OUT"/games_b*/records --out "$OUT/h5_soft" > "$OUT/tables.log" 2>&1
  log "tables: $(python -c "import h5py; f=h5py.File('$OUT/h5_soft/selfplay_train.h5'); print(len(f['offsets']) - 1, 'training rows')" 2>&1)"
}

human_tables() {   # two small 17lands tables (the validation files), linked as both splits
  local H=data/compute_bench/h5_human
  mkdir -p "$H"
  python - "$H" <<'PY' >&2
import os, shutil, sys
from huggingface_hub import hf_hub_download
h = sys.argv[1]
for t in ("turnstart", "replay_attack"):
    if not os.path.exists(f"{h}/{t}_val.h5"):
        shutil.copy(hf_hub_download("danbrooks/draftzero-checkpoints", f"exp4/tables/h5/{t}_val.h5",
                                    token=os.environ.get("HF_TOKEN")), f"{h}/{t}_val.h5")
    if not os.path.exists(f"{h}/{t}_train.h5"):
        os.symlink(f"{t}_val.h5", f"{h}/{t}_train.h5")
PY
  echo "$H"
}

train_one() {   # name config tables device threads steps
  local d=$OUT/train_$1
  rm -rf "$d"
  OMP_NUM_THREADS=$5 MKL_NUM_THREADS=$5 python -m draftzero.gameplay.supervised train --config "configs/$2" --out "$d" \
    --set tables_dir="$3" --set init_checkpoint="$MODEL" --set device="$4" --set max_steps="$6" > "$d.log" 2>&1
  log "train $1: $(python -c "import json; s=json.load(open('$d/summary.json')); print(s['step'], 'steps', s['seen'], 'samples', s['train_time_s'], 's:', round(s['samples_per_s'], 1), 'samples/s')" 2>&1 | tail -1)"
  rm -f "$d"/*.pt "$d"/*.pt.gz; rm -rf "$d/ckpt"     # weights are not the result, and they fill the disk
}

phase_train() {
  local HH; HH=$(human_tables)
  for T in $TRAIN_THREADS; do
    train_one "soft_cpu_t$T" compute_bench_soft.yml "$OUT/h5_soft" cpu "$T" "$TRAIN_STEPS"
    train_one "human_cpu_t$T" compute_bench_human.yml "$HH" cpu "$T" "$TRAIN_STEPS"
  done
  if [ $HAS_GPU = 1 ]; then
    train_one soft_cuda compute_bench_soft.yml "$OUT/h5_soft" cuda "$QUOTA" "$GPU_TRAIN_STEPS"
    train_one human_cuda compute_bench_human.yml "$HH" cuda "$QUOTA" "$GPU_TRAIN_STEPS"
  elif python -c "import sys, torch; sys.exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    train_one human_mps compute_bench_human.yml "$HH" mps "$QUOTA" "${MPS_TRAIN_STEPS:-400}"   # Apple's GPU
  fi
}

phase_tstep() {   # where a training step goes (tools/compute_bench/train_step_bench.py): dense / fused AdamW / frozen table
  local HH; HH=$(human_tables)
  python tools/compute_bench/train_step_bench.py --checkpoint "$MODEL" --tables-dir "$HH" --threads "${TSTEP_THREADS:-$QUOTA,8}" \
    --steps 20 --variants "${TSTEP_VARIANTS:-dense,fused,frozen}" --out /tmp/tstep_cpu --json "$OUT/tstep_cpu.json" > "$OUT/tstep_cpu.log" 2>&1
  log "tstep cpu: $(grep -h samples_per_s "$OUT/tstep_cpu.log" | python -c "import json,sys; print([(r['threads'], r['variant'], r['samples_per_s']) for r in map(json.loads, sys.stdin)])" 2>&1 | tail -1)"
  if [ $HAS_GPU = 1 ]; then
    python tools/compute_bench/train_step_bench.py --checkpoint "$MODEL" --tables-dir "$HH" --device cuda --threads 1 \
      --steps 50 --variants dense,fused,frozen --out /tmp/tstep_cuda --json "$OUT/tstep_cuda.json" > "$OUT/tstep_cuda.log" 2>&1
    log "tstep cuda: $(grep -h samples_per_s "$OUT/tstep_cuda.log" | python -c "import json,sys; print([(r['variant'], r['samples_per_s']) for r in map(json.loads, sys.stdin)])" 2>&1 | tail -1)"
  fi
}

phase_upload() {
  [ -n "${HF_TOKEN:-}" ] || { log "upload: no HF_TOKEN"; return; }
  python tools/compute_bench/summarize.py "$OUT" > "$OUT/summary.txt" 2>&1
  python - "$OUT" "exp4/bench/runs/$TAG" <<'PY' && log "upload: done" || log "upload: FAILED"
import os, sys
from huggingface_hub import HfApi
out, dst = sys.argv[1], sys.argv[2]
HfApi(token=os.environ["HF_TOKEN"]).upload_folder(repo_id="danbrooks/draftzero-checkpoints", folder_path=out,
    path_in_repo=dst, ignore_patterns=["**/records/**", "h5_soft/**", "**/*.pt", "**/*.pt.gz", "**/runtime/**"],
    commit_message=f"compute bench: {dst}")
PY
}

for ph in $PHASES; do
  log "phase $ph"
  "phase_$ph"
done
stop_servers
log "BENCH_DONE"
