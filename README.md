# DraftZero

One network that plays a Magic limited format with **any** deck. Every game draws both
decks at random from ~31.5k Foundations (FDN) Premier Draft decks built by top 17lands
players, and the agent is scored with 17lands-style statistics: card GIH win rate, rank
correlation against the public data, and per-colour deck records.

This is deliberately the opposite of [MageZero](https://github.com/WillWroble/MageZero),
which decomposes Magic into deck-local subgames. DraftZero uses MageZero as a library —
its RL framework, XMage bridge, trainer and evaluator — and adds the format-level parts.

## How the pieces relate

```
mage (XMage fork, Java)      pinned release artifact — rules engine, deck pools,
  ▲                          GAME_SUMMARY lines, action vocab
  │
MageZero (Python)            pinned dependency — model, trainer, server, runner, report
  ▲
DraftZero                    this repo — pools, generalist loop, format stats, workers
```

Dependencies point one way only. Nothing in MageZero knows DraftZero exists.

## Layout

| Path | What |
|---|---|
| `src/draftzero/loop.py` | the generation loop: play → train → eval |
| `src/draftzero/stats.py` | GIH win rate, Spearman ρ vs 17lands, deck/colour records |
| `src/draftzero/pools.py` | build deck pools from 17lands exports |
| `src/draftzero/reference.py` | build the 17lands GIH reference |
| `src/draftzero/paths.py` | resolve deck stems against a machine-local `deck_root` |
| `src/draftzero/provenance.py` | record which code produced a run |
| `src/draftzero/resources.py` | CPU/RAM/GPU sampling that respects cgroup limits |
| `src/draftzero/watchdog.py` | persist weights, detect stalls, end the run |
| `src/draftzero/workers/` | where self-play runs: local, ssh, runpod |
| `assets/` | small versioned inputs: deck metadata, action vocab, GIH reference |
| `configs/` | run configs and the hyperparameter curriculum |
| `data/` | **generated, gitignored**: decks, pools, runs, checkpoints |

`assets/` versus `data/` is the important line. Assets are small and reviewable and you
want to diff them. Data is large and regenerable and must never be committed.

## Install

```bash
pip install -e .
```

MageZero is pinned by git URL in `pyproject.toml`. **Replace the branch pin with a commit
SHA before any run you intend to reproduce** — a branch pin silently changes what your
results were produced against.

For local development against a MageZero checkout:

```bash
pip install -e ../MageZero && pip install -e . --no-deps
```

## Data

Nothing large is committed. `assets/decks.tsv` is the source of truth for the train/eval
split; the pool files derive from it:

```bash
dz pools build                 # assets/decks.tsv -> data/pools/{train,eval}.txt
dz pools check                 # confirm every stem resolves under data/decks
```

The `.dck` files themselves and the XMage build are fetched separately; point at them with
`MZ_DECK_DIR` and the `xmage` symlink.

## Running

```bash
bash deploy/bootstrap.sh                       # install deps, report the REAL cpu/ram quota
bash deploy/launch.sh configs/smoke.yml        # ~20 min end-to-end check
bash deploy/launch.sh configs/runpod.yml       # the real run
```

`launch.sh` starts the loop and a watchdog. The watchdog mirrors checkpoints to
`$DZ_PERSIST`, and when the run finishes or stalls it runs `$DZ_ON_COMPLETE` — **but only
after verifying the final sync**. A failed sync never triggers it. Losing a worker is
cheap; losing the weights is not.

## Workers

```python
from draftzero.workers import make_worker

w = make_worker("local")
w = make_worker("ssh", host="gpubox.lan", user="dan", workdir="~/draftzero")
w = make_worker("runpod", gpu_id="NVIDIA RTX 4000 Ada Generation", datacenter="EU-RO-1")
```

Adding a worker means implementing `run`, `push` and `fetch`, plus `provision`/`teardown`
if the machine is rented. Everything above that interface is identical.

### Sizing a worker

Self-play is **CPU-bound** XMage/MCTS. The GPU only serves small batched inference and
measures 2–8% utilisation. Pick a machine by vCPU, not VRAM:

```bash
dz workers rank --min-vcpu 8    # in-stock RunPod GPUs by vCPU per dollar
```

Two things that cost real time to learn: containers report the **host's** CPU and RAM
(a pod sold as 9 vCPU / 50 GB reports `nproc=48` / 251 GB and is cgroup-capped to ~7.65
cores / 46 GB), so never size threads from `nproc`; and the JVM thread pool is
`min(jvm.threads, chunk_games)`, so `chunk_games` must be ≥ `threads` or the extra threads
do nothing.

## Known gap

MageZero's dashboard renders the generic charts (throughput, losses, win rates, CPU/GPU/RAM).
The **format-level** sections — card GIH win rate tables, Spearman ρ against 17lands, deck
colour records — are computed here by `stats.py` but not yet rendered: that rendering lived
in a patched `report.py` inside the old fork and was deliberately left out of the engine
branch, since MageZero should not know what a draft format is. DraftZero needs to own that
rendering. `stats.py` already produces the payload; only the HTML/JS side is missing.

## Measured throughput

RTX 4000 Ada, 7.65-core quota, 9 vCPU pod:

| Setting | games/hr |
|---|---|
| gen 0 heuristic, `search_budget` 40, 6 threads | 112 |
| gen 0 heuristic, `search_budget` 200, 12 threads | 45 |
| network self-play, `search_budget` 40 | 29 |
| network self-play, `search_budget` 200 (projected) | ~12 |

Network play is ~4× slower than gen-0 heuristic, and games roughly double in length once
a net is driving them (20 → 40 turns). Budget from the bottom row, not the top.
