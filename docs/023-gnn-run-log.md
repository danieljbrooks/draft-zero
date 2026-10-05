# MageZero's graph network: the run log

*The stages of [docs/022](022-gnn-imitation-test-plan.md), as they run. Started 4 October 2026, 6:20 PM PT, when
Dan approved the plan ("push to main, use r1 if available, budget is okay"). All times are Pacific.*

## Status (Sunday 4 October, 10:30 PM PT)

| Stage | Status | Where | Spend |
|---|---|---|---|
| 0. Engineering | done, on main (5d1766a, a72792d, c18a727, c646d46) | laptop | – |
| 1. Build | **done** 9:14 PM PT (2.7 h, no errors); tables, slim tables uploaded 9:49 PM PT. 0.3-0.7% of games differ from experiment #4's: a pre-existing leak between games in the bridge's workers, not the graph code (below) | pod `gnn-stage1` | ~$1.50 |
| 2. Sweep | **running**: round 1 (batch 64) on the pod, round 2 (batch 256) on r1. **The GNN trails the MLP clearly at 10% too**: set NLL 0.315–0.336 against 0.258, value AUC 0.56–0.62 against 0.75. Batch 256 costs nothing; small embeddings help the policy a little, not the value | pod, r1 | |
| 3. Scale check, large training | | | |
| 4. Offline evaluation | held-out cards chosen (below); the tooling is on main | | |
| 5. Games | | | |

## Where things run

- **r1** (Dama's box): GPU 0 is free; GPU 1 and 8 CPU workers run another session's MLP games. The home disk has
  25 GB free, too little for the full tables (~30 GB with their flat twins), and the RAM (40 GiB, 16 in use) too
  little for 28 build workers. So the build runs on a pod. r1 takes part of the sweep from the **slim** tables
  (graph files + labels, `build.py slim`), in its own checkout (`~/dz-gnn`, `PYTHONPATH=src`), so the other session's
  running games are untouched.
- **GNN speed on r1's GPU 0** (the planning tables, Will's network): 1,177 states a second at batch 64, no faster
  than the 3090 (1,233): a step at 64 states is overhead-bound (~54 ms). At batch 256 r1 does 3,950 a second (the
  3090: 1,596). Inference at batch 128: 15,600 a second (the 3090: 5,400). The sweep's batch-256 arm decides whether
  the faster batch is free.
- **Pod `gnn-stage1`**: Secure RTX 3090, $0.50 an hour (EU-CZ-1, AMD EPYC 7C13, a 31-core quota, 125 GB RAM, 200 GB
  disk), rented 6:21 PM PT. cgroup v1, so `nproc` shows the host's 256 cores (`gnn_build.sh` now reads the v1
  quota). Self-destruct armed for 6:22 AM PT Tuesday. It runs the build, then training.

## Stage 1: the build

- 6:30 PM PT: `WORKERS=28 bash deploy/gnn_build.sh` started, after the slim bootstrap (~25 minutes, mostly the CUDA
  torch download) and copying `data/17lands/` and the split file from the laptop (~6 minutes).
- **Speed:** 13.4 games a second at 8,000 games (experiment #4's flat build: 17.6 on a different pod), so ~3.3 hours
  for the 161,206 games.
- **Shards:** 1.1 GB per 5,000-game part, ~36 GB in all (the plan guessed 65 GB from the laptop's smaller parts).
- 7:22 PM PT: 52,000 games at 16.5 a second.
- **Tables in parallel.** `build.py tables` turned the 33 parts into tables one at a time (exp4: 1 hour; with graphs
  longer), with the pod's cores idle. `tables --jobs N` now builds N parts at once; on the planning build, 4 jobs gave
  the same flat tables (`compare`) and the same graph datasets as one. A hand-off on the pod (`/root/handoff.sh`)
  stops `gnn_build.sh` when it reaches its tables step, pulls main and reruns it with `SKIP_BUILD=1 TABLE_JOBS=8`
  (the tables step resumes from finished parts).

- **Done 9:14 PM PT:** 161,206 games, no errors, 2.73 hours at 16.4 a second. Turns reproduced: 1,194,754 of the
  player's (86.9%) and 1,084,551 of the opponent's (83.6%), against experiment #4's 1,194,268 and 1,084,253.
- **Tables:** 8 parts at a time, ~7 minutes for the parts (experiment #4's serial step: an hour), then the merge.
  Slim tables (graph files + labels): uploaded to HF `exp4/tables_graph/slim/` at 9:49 PM PT; r1 fetched them
  (15 GB). The comparison with experiment #4's tables ran in parallel, so the sweep didn't wait for it.

### The 0.3% of games that differ from experiment #4's tables

`build.py compare`: every table has 0.3–0.7% of its games different (block_test: 20 of 5,734; block_train: 423 of
117,858; opp_block_train: 609 of 92,987), with row counts within 0.1%. In the differing rows the labels are the same
and the new build's states have **2–12 extra flat features**, the same in every row of the game.

**Not the graph code.** Two such games (83334, 118707) rebuilt on the laptop, each in a fresh bridge worker, give
experiment #4's rows exactly, with the graph encoding off and on. The extra features appear only in the pod build,
where a worker runs thousands of requests: something from an earlier game in the same JVM leaks into later games'
states (as the play op's "seeded games replay exactly only in a fresh JVM", docs/018). Which games it touches depends
on which worker gets them, so two builds differ in a few hundred games. **Consequence here:** experiment #4's final
MLP and transformer are scored on this build's validation and test rows (`/root/ref_evals.sh` on the pod), so every
comparison stays row for row. The leak itself is a bridge bug to fix separately (it touches experiment #4's tables
the same way).

## Stage 2: the sweep

`configs/gnn_sweep.yml`, the same 10% of the training games for every run, 20,000 validation rows a table (the rows
experiment #4's sweeps validated on, if `compare` passes):

| Machine | Runs |
|---|---|
| pod, RTX 3090 (`runs/gnn/sweep_pod`) | g-base, g-lr3e-4, g-lr3e-5, g-emb-lr10, g-drop0.1, g-leafdrop0.1, g-act5 |
| r1, GPU 0 (`runs/gnn/sweep_r1`, slim tables from HF, in RAM) | g-batch256, g-base-seed1, g-3ep, g-passes1, g-global1, g-d256, g-global4 |

### Results

Validation, 20,000 rows a table, one epoch on the same 10% of the training games (14,590 games, 1.1M decisions):

| Run | Set NLL | Top-1 acted | Attacks | Blocks | Targets | Value AUC | Pass on top, opp. turn (humans 0.934) | Train time |
|---|---|---|---|---|---|---|---|---|
| g-base (pod, batch 64) | 0.336 | 0.734 | 0.736 | 0.694 | 0.582 | 0.559 | 0.967 | 22.7 min (3090) |
| g-base-seed1 (r1, batch 64) | 0.324 | 0.732 | 0.734 | 0.690 | 0.586 | 0.614 | 0.938 | 7.6 min |
| g-batch256 (r1) | 0.334 | 0.731 | 0.737 | 0.689 | 0.585 | 0.623 | 0.947 | 4.5 min |
| **g256-emb0.02** (r1) | **0.315** | **0.740** | **0.742** | 0.689 | **0.614** | 0.615 | 0.936 | 4.6 min |
| *the GNN at 2.4% (docs/022 §2.3)* | *0.343* | *0.721* | *0.711* | *0.722* | *0.540* | *0.536* | *0.982* | |
| *experiment #4's MLP at 10% (best)* | *0.258* | *0.807* | *0.827* | *0.711* | *0.617* | *0.751* | | |
| *experiment #4's transformer at 10% (act3-td99)* | *0.285* | *0.791* | | | | *0.749* | *0.937* | |
| *experiment #4's MLP, all the games, on this build's rows* | *0.227* | *0.832* | *0.870* | *0.744* | *0.734* | *0.787* | *0.930* | |
| *experiment #4's transformer, all the games, on this build's rows* | *0.234* | *0.828* | *0.856* | *0.741* | *0.707* | *0.781* | *0.927* | |

- **Four times the data barely moved the GNN** (set NLL 0.343 → 0.334), while the flat networks gained a lot between
  similar sizes. If that holds for the batch-64 base, the plan's scaling question already leans against Will's
  data-hunger reading at these sizes.
- **The seed-to-seed band at batch 64 is wide:** set NLL 0.324 and 0.336, value AUC 0.614 and 0.559. The plan's rule
  (§4.2: a setting must beat the base by twice the seed difference) asks for 0.024 in set NLL.
- **Batch 256 is free:** 0.334 sits inside the batch-64 band, at a quarter of the steps. Round 2 and the later stages
  use it.
- **Small embeddings help the policy, not the value.** Experiment #4's transformer trained well only after its
  embeddings started at std 0.02 instead of 1. Will's network draws its leaf, type and value embeddings and CLS from
  N(0, 1). With `emb_init_std` 0.02: set NLL 0.315 (0.019 better than batch 256's base, short of the rule's 0.024
  until round 2's second batch-256 seed), targets +0.029, attacks +0.005, value AUC unchanged (0.615).
- **The value head is the weakest part.** Value AUC 0.56–0.62 in every run against the MLP's 0.75 at the same data,
  and still climbing at the end of the epoch (0.58 → 0.61 over the last half). The three-epoch arm says whether it
  is slow or stuck.
- **The scaling check (§4.2).** The gap to the MLP was 0.076 at 2.4% of the games; at 10% it is 0.066–0.078 for the
  base and 0.057 with small embeddings. Barely smaller: stage 3's 30% check decides.

### Round 2: batch 256 on r1

A batch-256 run takes ~5 minutes on r1, four times faster than batch 64. So once r1 finishes round 1's second seed
(the batch-64 noise band), it runs round 2 at batch 256 instead of round 1's remaining batch-64 arms:
emb_init_std 0.02 (alone, with dropout 0.1, with lr 5e-4, for 3 epochs), g-batch256 again with quarter-epoch curves,
and round 1's shape arms (1 local pass, 1 and 4 global layers, width 256). The pod keeps round 1's batch-64 arms
(learning rates, the leaf table's rate, dropout, the passivity fix).

Round 1's leftover arm (g-passes1) on r1 was stopped at 10:15 PM PT; GPU 0 runs only round 2.

**Why batch 256 runs first.** A profile of a training step on r1 (`torch.profiler`, the planning tables): at 64
states a step the CPU spends ~39 ms issuing ~3,000 small kernels while the GPU works ~16 ms. The step is launch-bound
whatever the GPU, so a bigger batch is nearly free: 3.4x the throughput on r1, 1.3x on the 3090. If it costs no
accuracy, stages 3 and later use it.

## Stage 4, ahead of time: the held-out cards

`heldout_cards.py choose` over the top players' training games (145,903). Five cards, each a different kind, each
sharing its mechanics with other cards: a vanilla creature, a defender, burn, a removal aura and a combat trick.
A game counts as showing a card when it is in the player's decklist or either player is seen with it.

| Card | Type | Rarity | Share of training games |
|---|---|---|---|
| Savannah Lions | Creature (2/1 for W) | uncommon | 6.9% |
| Claws Out | Instant (W) | uncommon | 5.9% |
| Imprisoned in the Moon | Aura (U) | uncommon | 4.6% |
| Gleaming Barrier | Artifact creature, wall | common | 4.4% |
| Goblin Negotiation | Sorcery (R) | uncommon | 3.3% |
| **Any of them** | | | **22.1%** (32,283 games) |

The list is `data/imitation_graph/heldout/{cards.json,exclude_games.npy}` (regenerate with
`heldout_cards.py choose --cards "Savannah Lions,Gleaming Barrier,Goblin Negotiation,Imprisoned in the Moon,Claws Out"`).

## Pods

| Pod | What | Rented | Removed | Hours | $/h | Cost |
|---|---|---|---|---|---|---|
| `gnn-plan` (Community 3090, FR) | docs/022 §2's planning measurements | 4:53 PM PT | 6:07 PM PT | 1.2 | 0.22 | $0.27 |
| `gnn-stage1` (Secure 3090, CZ) | stage 1, then training | 6:21 PM PT | | | 0.50 | |
