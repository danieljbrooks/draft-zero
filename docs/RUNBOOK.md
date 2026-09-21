# FDN long run — runbook

Target: a generalist that plays FDN limited with any deck, trained on one machine for
~10 days at ~$70, with a checkpoint every ~3 hours and enough eval signal to tell whether
it is actually improving.

| | |
|---|---|
| Config | `configs/fdn_5090.yml` (see also `fdn_long.yml` for an 8-vCPU box) |
| Worker | RunPod RTX 5090, **secure**, EUR-IS-1, $0.99/hr — 48 vCPU (40.8 after cgroup), 94 GB |
| Volume | `draftzero-l4` (`ejj4atzg14`), 60 GB, EUR-IS-1 — holds the code, decks and XMage build |
| Budget cap | `DZ_MAX_HOURS=40` → ~$40 of compute, leaving most of the $98 for the extended run |
| Expected | ~30 generations, ~2,500 games, ~2 days |

Secure cloud is required: network volumes do not exist on community cloud, and without the
volume a terminated pod takes the weights with it.

## 1. Provision

```bash
source ~/.runpod/env
runpodctl pod create --name fdn-long \
  --gpu-id "NVIDIA RTX 4000 Ada Generation" --cloud-type SECURE \
  --data-center-ids EU-RO-1 --network-volume-id uot2qdgwry \
  --image "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404" \
  --container-disk-in-gb 60 --ports "22/tcp" --wait
```

If that GPU is out of stock, pick another by **vCPU per dollar**, not VRAM — self-play is
CPU-bound and the GPU idles at 2–8%:

```bash
dz workers rank --min-vcpu 8
```

## 2. Launch

The volume already carries everything, so this is just bootstrap + launch:

```bash
ssh -i ~/.ssh/id_ed25519 -p <port> root@<ip>
cd /workspace/draftzero
bash deploy/bootstrap.sh                      # idempotent; prints the REAL cpu/ram quota
DZ_MAX_HOURS=230 bash deploy/launch.sh configs/fdn_long.yml --fresh
```

`bootstrap.sh` prints both the advertised and the cgroup capacity. If the quota line reads
much below ~7.65 cores, you got a smaller slice than planned — lower `jvm.threads` to match
before launching, or the threads will just contend.

## 3. First 24 hours — confirm the plan survives contact

The whole config rests on a fitted throughput model. Generation 1 is the first honest test
of it, so check these once gen 1 completes (~5h in, after the ~2h gen-0 bootstrap):

| Check | Where | Expected | If wrong |
|---|---|---|---|
| Throughput | `env_gph` | ~21 games/hr | Well below → see the next two rows |
| Searches truncated | `env_timeout` | low | High → raise `timeout_ms`; do **not** lower `search_budget` first |
| CPU vs quota | `res_cpu` | 70–90% | <50% → threads are blocking on inference, raise `jvm.threads` |
| Container RAM | `res_mem` | well under 46 GB | Near the cap → lower `jvm.heap` |
| GPU | `res_gpu` | 5–15%, idle is normal | Pegged at 100% → inference is the bottleneck, unexpected |
| Generation wall time | `env_gen_minutes` | ~190 min | Much longer → checkpoint cadence is worse than planned |

A generation that takes 8 hours instead of 3 is not a disaster, but it halves the number of
policy updates in the budget. Better to notice on day one and cut `games_per_gen` than to
discover it on day nine.

## 4. Is it learning?

Three independent signals, all on the dashboards (`runs/<id>/index.html`):

- **Losses** (`dashboard.html` → Learning): value and policy loss should fall and then
  flatten. Falling loss alone is not strength — it can just mean fitting its own noise.
- **Strength vs fixed baselines** (`dashboard.html` → Actual strength): win rate against
  `offline` (search with no network) and `minimax`. These opponents never change, so this
  is the honest measure. Pooled over 8 generations, ~64 games per point. **Always read the
  n alongside the rate** — a 60% win rate over 8 games means nothing.
- **Format knowledge** (`format.html`): Spearman ρ against 17lands GIH win rates. Gen 0's
  heuristic ρ is the baseline for how much card sense search alone has; beating it means
  the network learned something about cards that search did not already know.

Expect all three to move slowly. Beating `offline` at all is the first real milestone: it
means the network is adding value over raw search.

## 5. Crash recovery

Checkpoints land in `models/FDN_generalist/ver1/` — `gen{N}.pt.gz` per generation plus a
rolling `model.pt.gz` — and the watchdog mirrors them to `$DZ_PERSIST` on the volume within
5 minutes. The volume outlives the pod, so nothing is lost when a pod dies.

To resume, on a fresh pod attached to the same volume:

```bash
cd /workspace/draftzero
bash deploy/bootstrap.sh
DZ_MAX_HOURS=<remaining> bash deploy/launch.sh configs/fdn_long.yml --resume
```

`--resume` continues the newest unfinished run and redoes the interrupted stage. Worst case
a crash costs one chunk (8 games) plus the current stage.

## 6. Extending the run

This run is deliberately small — it proves the pipeline and gives a learning curve. When it
finishes, continue it rather than starting over:

```bash
# raise target_games in the config (or pass --target), then:
DZ_MAX_HOURS=<more> bash deploy/launch.sh configs/fdn_3090.yml --extend
```

`--extend` reopens the **same** run: generation numbering, `games.jsonl`, the dashboard
trend line and the pool of past checkpoints that the `past` mix draws from all carry over.
A fresh run would restart at generation 0 and burn a heuristic bootstrap it does not need,
and the strength curve would start over with it.

It refuses to run if `target_games` has not been raised above the games already played,
since the loop would otherwise stop again immediately. Each extension is recorded in
`run.json` under `extensions`.

Because extension is the plan, **pull the checkpoints off the pod before destroying it**:

```bash
rsync -rlptz --no-o --no-g -e "ssh -i ~/.ssh/id_ed25519 -p <port>" \
  root@<ip>:/workspace/draftzero/{models,runs} ./recovered/
```

On community cloud there is no network volume, so this is the only copy.

## 7. How it stops

The watchdog ends the run on whichever comes first: `target_games` reached, `DZ_MAX_HOURS`
elapsed, or no new game for `DZ_STALL_MINUTES` (45). In every case it syncs, **verifies the
sync**, writes `STATUS.json`, and only then runs `$DZ_ON_COMPLETE`. A failed sync never
triggers it — losing a pod is cheap, losing the weights is not.

To have the pod destroy itself when done, set `DZ_ON_COMPLETE` at launch. That needs the
API key on the worker; if you would rather not put it there, leave it unset and terminate
by hand:

```bash
runpodctl pod remove <pod_id>
```

Check on it without SSH:

```bash
runpodctl user          # balance and current spend rate
runpodctl pod list      # is it still running
```

## What this budget does and does not buy

~4,100 games across a 31.5k-deck pool is enough to answer **"does this approach learn?"** —
whether strength against fixed baselines trends up and whether ρ beats the gen-0 heuristic.

It is not enough to produce a strong FDN player. AlphaZero-style training on a format this
wide wants orders of magnitude more self-play. Treat this run as the experiment that tells
you whether scaling up is worth funding, not as the finished agent.
