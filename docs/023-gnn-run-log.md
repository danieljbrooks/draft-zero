# MageZero's graph network: the run log

*The stages of [docs/022](022-gnn-imitation-test-plan.md), as they run. Started 4 October 2026, 6:20 PM PT, when
Dan approved the plan ("push to main, use r1 if available, budget is okay"). All times are Pacific.*

## Status (Monday 5 October, 6:45 AM PT)

| Stage | Status | Where | Spend |
|---|---|---|---|
| 0. Engineering | done, on main (5d1766a, a72792d, c18a727, c646d46) | laptop | – |
| 1. Build | **done** 9:14 PM PT (2.7 h, no errors); tables, slim tables uploaded 9:49 PM PT. 0.3-0.7% of games differ from experiment #4's: a pre-existing leak between games in the bridge's workers, not the graph code (below) | pod `gnn-stage1` | ~$1.50 |
| 2. Sweep | **done** (6 rounds, 44 runs). **The recipe: width 256, embeddings at std 0.02, dropout 0.1, the game result as the value target, lr 1e-4, batch 256.** Three epochs of 10%: set NLL 0.256, top-1 0.813, value AUC 0.776, past the MLP's best at 10% (0.258 / 0.807 / 0.751); five: 0.247. Will's settings gave 0.324–0.336 | pod, r1 | |
| 3. Scale check, large training | **done.** 30%: one epoch 0.250 (the gap to the MLP closes with data: 0.037 at 10%, 0.009 at 30%), three epochs 0.232. **All the games, three epochs (6:35 AM PT): test set NLL 0.214, top-1 0.847, value AUC 0.797, ahead of experiment #4's MLP (0.229 / 0.829 / 0.784) and transformer on every measure.** HF `gnn/main/` | `gnn-stage1`, `gnn-full` | $7.31 (`gnn-full`) |
| 4. Offline evaluation | **held-out cards done:** without the cards' games, the GNN loses about half what the MLP loses on decisions where a held-out card is legal (top-1 −6.4 points against −10.9); the test split runs at the end of the large training | `gnn-stage1` | |
| 5. Games | | | |

## Where things run

- **r1** (Dama's box): GPU 0 is free; GPU 1 and 8 CPU workers run another session's MLP games. The home disk has
  25 GB free, too little for the full tables (~30 GB with their flat twins), and the RAM (40 GiB, 16 in use) too
  little for 28 build workers. So the build runs on a pod. r1 takes part of the sweep from the **slim** tables
  (graph files + labels, `build.py slim`), in its own checkout (`~/dz-gnn`, `PYTHONPATH=src`), so the other session's
  running games are untouched.
- **r1's disk filled at ~11:45 PM PT**, from round 2's checkpoints (~1 GB a run with the resume state). Round 2's
  last run (g256-global4) and round 3's start died, and two of the other session's game workers lost their log
  threads (`mlp-ilbc3000-top`: "No space left on device"; the games kept running). Cleared to 6.9 GB free; r1's
  runner (`~/r1_round3.sh`) now keeps only best_policy and best_value of finished runs and stops the sweep under
  2 GB free.
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
| *round 1, batch 64 (pod)* | | | | | | | | |
| g-lr3e-4 | 0.375 | 0.717 | 0.628 | 0.687 | 0.535 | 0.507 | 0.993 | 22.6 min |
| g-lr3e-5 | 0.332 | 0.735 | 0.753 | 0.690 | 0.590 | **0.688** | 0.924 | 22.2 min |
| g-emb-lr10 (leaf tables' rate ×10) | 0.317 | 0.740 | 0.741 | 0.693 | 0.584 | 0.656 | 0.918 | 22.6 min |
| g-act5 (passivity fix ×5) | 0.352 | 0.735 | 0.745 | 0.689 | 0.579 | 0.552 | 0.948 | 23.0 min |
| g-drop0.1 | 0.318 | 0.741 | 0.746 | 0.689 | 0.587 | 0.653 | 0.931 | 22.8 min |
| g-leafdrop0.1 (and leaf dropout 0.1) | 0.324 | 0.739 | 0.788 | 0.700 | 0.597 | 0.666 | 0.938 | 22.6 min |
| *round 2, batch 256 (r1)* | | | | | | | | |
| g256-base-again | 0.328 | 0.735 | 0.742 | 0.690 | 0.598 | 0.667 | 0.932 | 4.8 min |
| **g256-emb0.02-drop0.1** | **0.305** | **0.744** | 0.753 | 0.690 | **0.619** | 0.671 | 0.924 | 4.7 min |
| g256-emb0.02-lr5e-4 | 0.368 | 0.724 | 0.631 | 0.688 | 0.544 | 0.503 | 0.986 | 4.8 min |
| g256-emb0.02-3ep (3 epochs) | 0.300 | 0.747 | 0.753 | 0.691 | 0.656 | 0.683 | 0.925 | 14.5 min |
| g256-passes1 | 0.340 | 0.734 | 0.719 | 0.687 | 0.560 | 0.582 | 0.941 | 3.2 min |
| g256-global1 | 0.346 | 0.735 | 0.742 | 0.690 | 0.560 | 0.630 | 0.980 | 4.2 min |
| g256-d256 (width 256) | 0.319 | 0.735 | **0.759** | 0.696 | 0.581 | 0.636 | 0.922 | 3.3 min |
| g256-global4 | 0.327 | 0.739 | 0.734 | 0.690 | 0.582 | 0.624 | 0.928 | 6.0 min |
| *round 3, the value head: small embeddings, batch 256 (r1)* | | | | | | | | |
| g256-v-result (the game result, no TD) | 0.316 | 0.734 | 0.725 | 0.689 | 0.612 | **0.674** | 0.926 | 4.6 min |
| g256-v-td95 (TD 0.95, MageZero's) | 0.310 | 0.741 | 0.735 | 0.689 | 0.620 | 0.588 | 0.924 | 4.9 min |
| g256-v-weight2 (value loss ×4) | 0.347 | 0.730 | 0.721 | 0.688 | 0.563 | 0.664 | 0.968 | 5.0 min |
| g256-v-only (no policy losses) | 0.935 | 0.523 | 0.502 | 0.637 | 0.365 | 0.521 | 0.502 | 4.7 min |
| *round 4, the candidate: small embeddings, dropout 0.1, result targets, batch 256 (r1)* | | | | | | | | |
| c-base | 0.303 | 0.749 | 0.762 | 0.689 | 0.639 | 0.702 | 0.931 | 4.8 min |
| c-seed1 | 0.306 | 0.742 | 0.778 | 0.698 | 0.625 | 0.729 | 0.923 | 5.0 min |
| c-lr1e-4 | 0.292 | 0.753 | 0.769 | **0.709** | 0.639 | **0.747** | 0.926 | 5.0 min |
| **c-d256 (width 256)** | **0.286** | **0.778** | 0.768 | 0.690 | 0.641 | 0.723 | 0.928 | **3.0 min** |
| c-emb-lr10 | 0.303 | 0.750 | 0.758 | 0.697 | **0.649** | 0.723 | 0.923 | 4.6 min |
| *round 5: round 4's recipe at width 256 and lr 1e-4 (r1)* | | | | | | | | |
| r5-base | 0.295 | 0.754 | 0.770 | 0.701 | 0.624 | 0.749 | 0.926 | 3.4 min |
| r5-seed1 | 0.296 | 0.752 | 0.792 | 0.689 | 0.636 | 0.751 | 0.922 | 3.3 min |
| r5-d128 (width 128) | 0.323 | 0.743 | 0.770 | 0.698 | 0.607 | 0.738 | 0.936 | 2.5 min |
| r5-lr5e-5 | 0.304 | 0.748 | 0.767 | 0.692 | 0.610 | 0.717 | 0.930 | 3.0 min |
| **r5-3ep (three epochs)** | **0.256** | **0.813** | **0.839** | **0.726** | **0.702** | **0.776** | 0.928 | 8.8 min |
| *the GNN at 2.4% (docs/022 §2.3)* | *0.343* | *0.721* | *0.711* | *0.722* | *0.540* | *0.536* | *0.982* | |
| *experiment #4's MLP at 10% (best)* | *0.258* | *0.807* | *0.827* | *0.711* | *0.617* | *0.751* | | |
| *experiment #4's transformer at 10% (act3-td99)* | *0.285* | *0.791* | | | | *0.749* | *0.937* | |
| *experiment #4's MLP, all the games, on this build's rows* | *0.227* | *0.832* | *0.870* | *0.744* | *0.734* | *0.787* | *0.930* | |
| *experiment #4's transformer, all the games, on this build's rows* | *0.234* | *0.828* | *0.856* | *0.741* | *0.707* | *0.781* | *0.927* | |

- **Four times the data barely moved the GNN** (set NLL 0.343 → 0.334), while the flat networks gained a lot between
  similar sizes. If that holds for the batch-64 base, the plan's scaling question already leans against Will's
  data-hunger reading at these sizes.
- **The seed bands:** batch 64, set NLL 0.324 / 0.336 and value AUC 0.614 / 0.559; batch 256, 0.334 / 0.328 and
  0.623 / 0.667. Value AUC moves by 0.05 between seeds. The plan's rule (§4.2: beat the base by twice the seed
  difference, no worse on value AUC) asks for ~0.012 in set NLL at batch 256.
- **Batch 256 is free** (inside the batch-64 band, a quarter of the steps). Every later run uses it.
- **Small embeddings + dropout 0.1 pass the rule:** set NLL 0.305 against the base's 0.331 (−0.026), targets
  +0.03, attacks +0.01, value AUC 0.671 (base 0.623–0.667). Small embeddings alone: 0.315, value AUC 0.615.
- **High learning rates flatten the attack and value outputs:** at 3e-4 (batch 64) and 5e-4 (batch 256) attacks fall
  to 0.63 and value AUC to 0.50, while the priority heads still learn. A low one helps the value: 3e-5 at batch 64
  gives the best one-epoch value AUC (0.688) at the base's set NLL.
- **More epochs help:** three epochs of the same 10% (small embeddings, dropout 0.25) reach 0.300 and value AUC
  0.683, still falling at the end. Experiment #4's MLP gained less from repeats.
- **Shape:** width 256 is as good as 512 (0.319, the best attacks, 30% faster); one local pass or one global layer is
  worse (0.340, 0.346).
- **The value head is still the weakest part:** 0.50–0.69 in every run against the MLP's 0.751.
- **Round 3, the value head** (against small embeddings alone, TD(0.99): set NLL 0.315, value AUC 0.615):
  - **The game result is the better target for the GNN:** value AUC 0.674 (+0.06), log-loss 0.622 (−0.028), the
    policy unchanged (0.316). The TD trap: TD(0.99) blends in the network's own still-poor value.
  - **TD(0.95), MageZero's λ, is the transformer's trade again:** the policy a little better (0.310, within noise),
    the value worse (0.588). In MageZero the blend is with the search's root score, not the network's value.
  - **A heavier value loss lifts the value and costs the policy** (0.347): the shared features are contested.
  - **The value alone learns nothing in an epoch** (AUC 0.52): the value head lives on the features the policy
    losses build. A separate value tower (experiment #4's `x-value-tower`) wouldn't help the GNN without them.
- **Round 1 (pod):** the leaf tables' rate ×10 helps a little (0.317, inside twice the batch-64 band); the passivity
  fix at ×5 moves Pass on top toward the humans (0.948, humans 0.934) at a cost (0.352).
- **The scaling check (§4.2).** The gap to the MLP in set NLL was 0.076 at 2.4% of the games (Will's settings); at
  10% it is 0.066–0.078 with Will's settings and 0.028 with round 4's best. Smaller, as the plan's rule asks to go
  on: stage 3's 30% check decides whether it keeps closing.

### Round 2: batch 256 on r1

A batch-256 run takes ~5 minutes on r1, four times faster than batch 64. So once r1 finishes round 1's second seed
(the batch-64 noise band), it runs round 2 at batch 256 instead of round 1's remaining batch-64 arms:
emb_init_std 0.02 (alone, with dropout 0.1, with lr 5e-4, for 3 epochs), g-batch256 again with quarter-epoch curves,
and round 1's shape arms (1 local pass, 1 and 4 global layers, width 256). The pod keeps round 1's batch-64 arms
(learning rates, the leaf table's rate, dropout, the passivity fix).

Round 1's leftover arm (g-passes1) on r1 was stopped at 10:15 PM PT; GPU 0 runs only round 2. Round 2 ended at
11:45 PM PT (g256-global4 lost to the full disk; rerun after round 3).

## Stage 3: the scale check and the large training

**30% of the games** (`configs/gnn_scale30.yml`, the pod's 3090, 3.3M decisions), against experiment #4's 30%
networks (scored on experiment #4's rows, which differ in 0.3% of games):

| 30% of the games, one epoch | Set NLL | Top-1 acted | Attacks | Blocks | Targets | Value AUC | Value log-loss | Train time |
|---|---|---|---|---|---|---|---|---|
| GNN, width 256 (`s30-d256`) | 0.250 | 0.821 | 0.842 | 0.722 | 0.718 | 0.769 | 0.556 | 21 min |
| GNN, width 512 (`s30-d512`, Will's width) | 0.269 | 0.779 | 0.814 | 0.721 | 0.704 | 0.773 | 0.553 | 36 min |
| **GNN, width 256, three epochs** (`s30-d256-3ep`) | **0.232** | **0.838** | **0.860** | **0.752** | **0.744** | **0.799** | **0.525** | 65 min |
| *MLP, wave D's combination (`d1`, `d2`)* | ***0.241** / 0.243* | *0.820 / 0.821* | ***0.855** / 0.851* | ***0.728** / 0.727* | *0.673 / 0.665* | ***0.775** / 0.776* | *0.571 / 0.573* | |
| *transformer (docs/022 §4.3's reference)* | *0.256* | | | | | | | |

- **The gap to the MLP closes with data.** One epoch of the same recipe: 0.295 against 0.258 at 10% (0.037), 0.250
  against 0.241 at 30% (0.009). From 10% to 30% the GNN gains 0.045, the MLP 0.017, the transformer 0.029. Will's
  reading, that the GNN is the data-hungry one, holds once it trains properly.
- **At 30% the GNN matches the MLP's top-1 and beats its targets by 4.5 points** and its value log-loss; the MLP
  still leads on attacks, blocks and set NLL.
- **Width 512 trails 256 at 30% too** (0.269 against 0.250 at one epoch), as at 10%: at these data sizes Will's
  width is the slower learner, not the better one.
- **Three epochs of 30%: set NLL 0.232, ahead of the 30% MLP on every measure** (top-1 +1.8 points, attacks +0.5,
  blocks +2.4, targets +7.1, value AUC +0.024, log-loss −0.046). Against experiment #4's MLP trained on *all* the
  games, scored on these rows (0.227 / 0.832 / 0.870 / 0.744 / 0.734 / 0.787): better top-1, blocks, targets and
  value, a little behind on set NLL (+0.005) and attacks (−1.0 point), from a third of the data. The MLP gains
  nothing from repeats (experiment #4), so one epoch is its best at 30%.
- **The curve:** 0.253 at one epoch, 0.236 at two, 0.232 at three; the value AUC still rising (0.775 → 0.788 →
  0.799).
- **Inference** (`graph_supervised bench`, r1's GPU 0; batch 1 / 8 / 32 / 128): width 256 (9.2M parameters) 120 /
  1,058 / 3,559 / 10,844 states a second, width 512 (35.1M) 140 / 1,095 / 3,958 / 14,167. The GPU is launch-bound
  either way. On one CPU thread at batch 1 the narrower network is 6.6× faster (99 against 15 states a second).

**The large training** (`configs/gnn_train.yml`, `deploy/gnn_train.sh`): all 10.9M training decisions, three epochs,
the sweep's recipe. Its first load holds every row in RAM (~80 GB from the 30% run's 24.5 GB) before the memory-mapped
cache takes over, too much for `gnn-stage1`'s 125 GB alongside its runs. So it runs on a second pod, `gnn-full`:
Secure RTX PRO 6000 (r1's GPU), 188 GB, 27 cores of an EPYC 9554, $2.09 an hour (a Community L40S at $0.79 had none
free). Results go to HF `gnn/main/` every 15 minutes; the test split is scored at the end.

- 3:14 AM PT: started; all the rows loaded in 29 minutes (vocabs: 1,932 leaves, 22 edge labels); training at ~3,250 states a second
  (r1 trains the same network at ~5,800: this pod's GPU sits at ~47%), so three epochs take ~2.8 hours.
- Quarter epochs: set NLL 0.256 → 0.243 → 0.238; top-1 0.817 → 0.831; attacks 0.793 → 0.857; value AUC 0.759 →
  0.784.
- **Done 6:35 AM PT** (3 epochs, 128,361 steps, 3.4 hours with the load); the test split scored by 6:37; everything
  on HF `gnn/main/` (best_policy, best_value, final, evals, the test scores); `gnn-full` removed at 6:39 AM PT
  (3.5 hours, $7.31). Validation: 0.256 at a quarter epoch, 0.232 at one, 0.220 at two, 0.214 at three; still
  falling slowly (0.2137 → 0.2133 over the last quarter).

**The test split** (stage 4), the same rows for all three networks (experiment #4's scored on this build's rows):

| All the training games | Set NLL | Top-1 acted | Attacks | Blocks | Targets | Value AUC | Value log-loss | Pass on top, opp. turn (humans 0.934) |
|---|---|---|---|---|---|---|---|---|
| **GNN, width 256, three epochs** | **0.214** | **0.847** | **0.878** | **0.783** | **0.779** | **0.797** | **0.526** | 0.927 |
| MLP (experiment #4, `mlp_1ep`) | 0.229 | 0.829 | 0.873 | 0.750 | 0.730 | 0.784 | 0.552 | 0.931 |
| transformer (experiment #4, stage 3, three epochs) | 0.235 | 0.825 | 0.862 | 0.748 | 0.709 | 0.781 | 0.549 | 0.928 |

- **The GNN leads on every measure:** set NLL −0.015 against the MLP, top-1 +1.8 points, blocks +3.3, targets +4.9,
  value AUC +0.013. Its gains are largest on the decisions about specific cards and creatures (targets, blocks),
  which the graph represents directly.
- best_policy and best_value are the same checkpoint (the last evaluation was best on both).

### Round 5: width 256 at lr 1e-4 (r1, 1:35–2:03 AM PT), and round 6

- **Width 256 and lr 1e-4 don't stack at one epoch:** 0.295 / 0.296 (two seeds), against width 256 at 2e-4's 0.286,
  but the value AUC is lr 1e-4's (0.749 / 0.751). Width 128 (0.323) and lr 5e-5 (0.304) are worse.
- **Three epochs of the same 10%: set NLL 0.256**, top-1 0.813, attacks 0.839, blocks 0.726, targets 0.702, value
  AUC 0.776. **Past the MLP's best at 10% on every measure** (0.258 / 0.807 / 0.827 / 0.711 / 0.617 / 0.751), and
  experiment #4's transformer at three epochs of 10% (`ep3`: 0.269 / 0.807, value AUC 0.747). Still falling at the
  end (0.2604 → 0.2570 → 0.2560 over the last half epoch), and the value AUC still rising. The GNN gains far more from
  repeats than the transformer did (−0.039 from one epoch to three, against −0.018).
- **Round 6, repeats** (`configs/gnn_sweep_r6.yml`, r1, 2:15–3:05 AM PT):

  | 10% of the games | Set NLL | Top-1 acted | Attacks | Blocks | Targets | Value AUC | Train time |
  |---|---|---|---|---|---|---|---|
  | three epochs (r5-3ep) | 0.256 | 0.813 | 0.839 | 0.726 | 0.702 | **0.776** | 8.8 min |
  | **five epochs** | **0.247** | **0.824** | 0.833 | **0.737** | **0.722** | 0.771 | 14.7 min |
  | three epochs at lr 2e-4 | 0.259 | 0.810 | 0.836 | 0.716 | 0.702 | 0.758 | 9.0 min |
  | three epochs at width 512 | 0.269 | 0.784 | 0.820 | 0.721 | 0.695 | 0.769 | 13.8 min |

  Repeats keep paying for the policy (five epochs: −0.009 more), while the value peaks around four (0.774) and
  then drifts. lr 1e-4 beats 2e-4 once there are repeats. **Width 512 doesn't catch up with more steps** (0.269
  against 0.256): the narrower network is better at this data size.
- **Round 1's last arms (pod):** dropout 0.1 at batch 64 (0.318) as at 256; leaf dropout adds nothing (0.324).
- **The pod** finished round 1 at ~1:50 AM PT, then sat idle ~15 minutes (its runner waited for a marker the first
  sweep's launcher never wrote); the 30% check started at 2:05 AM PT.

### Round 4: the candidate recipe (r1, 12:52–1:26 AM PT)

`configs/gnn_sweep_r4.yml`: small embeddings, dropout 0.1, the game result as the value target, and one change each:
a second seed, learning rate 1e-4, width 256, the leaf tables' rate ×10.

- **The fixes stack:** the recipe reaches set NLL 0.303–0.306 and value AUC 0.70–0.73, against 0.315 / 0.615 for small
  embeddings alone. Its two seeds differ by 0.003, so the rule's bar is ~0.006.
- **Width 256 is the best network yet:** set NLL 0.286 (−0.019), top-1 0.778 (+0.03), at 60% of the training time.
  At 10% of the games the GNN now matches experiment #4's transformer (0.285 / 0.791) and trails the MLP by 0.028.
- **Learning rate 1e-4:** set NLL 0.292 (−0.013), value AUC 0.747, the MLP's level (0.751), and the best blocks.
- **The leaf tables' rate ×10** adds nothing on top of small embeddings (0.303).
- **Next:** round 5 (`configs/gnn_sweep_r5.yml`, r1) puts width 256 and lr 1e-4 together, with a second seed, width
  128, lr 5e-5 and three epochs. Stage 3's 30% check (`configs/gnn_scale30.yml`, the pod) runs the same recipe at
  widths 256 and 512: is Will's wider network the one that needs more data? Then three epochs at width 256. r1's 58
  GB of RAM, shared with another session's games, can't hold 30% of the tables.

### Round 3: the value head (r1)

`configs/gnn_sweep_r3.yml`, on small embeddings, one change each: the game result as the target (no TD); the value
loss ×4; the value alone (no policy losses); TD(0.95), MageZero's own λ. TD(0.99) over ~75 positions a game takes
about half of an early position's target from the network's own value. That helps a network whose value is good and
may trap one whose value is poor. In MageZero (README; `configs/curriculum.yml`: 0.95 at generation 0, down to 0.70
by generation 3) the blended value is the search's root score, not the network's. Experiment #4's transformer at
0.95 against 0.99: set NLL 0.287 / 0.292, value AUC 0.739 / 0.749.

**Why batch 256 runs first.** A profile of a training step on r1 (`torch.profiler`, the planning tables): at 64
states a step the CPU spends ~39 ms issuing ~3,000 small kernels while the GPU works ~16 ms. The step is launch-bound
whatever the GPU, so a bigger batch is nearly free: 3.4x the throughput on r1, 1.3x on the 3090. If it costs no
accuracy, stages 3 and later use it.

## Stage 4: the held-out cards

**Results (done 6:06 AM PT).** Each network's 30% recipe (one epoch) trained twice: on the 30% subset, and on it
without every game that shows one of the five cards (9,799 of 43,769 games, 22.4%; `configs/gnn_heldout.yml`,
`configs/exp4_heldout_mlp.yml`; the MLP's full-subset network is experiment #4's `d1-s30-combined`). Scored on the
test split by `heldout_cards.py eval`: "involved" are the decisions where a held-out card is a legal option (3,761;
2,173 where the human acted), "chosen" those where the human's move is a held-out card (766; 636 acted), "control"
5,000 other decisions a table (24,981; 11,295 acted).

| Top-1 when the human acted (set NLL) | Involved | Chosen | Involved, not chosen | Control |
|---|---|---|---|---|
| GNN, all games | 0.794 (0.437) | 0.777 (0.570) | 0.802 | 0.761 (0.410) |
| GNN, cards held out | 0.730 (0.568) | 0.756 (0.640) | 0.719 | 0.749 (0.428) |
| **GNN, change** | **−6.4 (+0.130)** | −2.1 (+0.071) | **−8.2** | −1.2 (+0.019) |
| MLP, all games | 0.794 (0.433) | 0.774 (0.572) | 0.802 | 0.740 (0.414) |
| MLP, cards held out | 0.685 (0.680) | 0.841 (0.373) | 0.621 | 0.730 (0.423) |
| **MLP, change** | **−10.9 (+0.248)** | +6.7 (−0.199) | **−18.1** | −1.0 (+0.009) |

*"Involved, not chosen" is derived from the other columns (chosen decisions are a subset of involved).*

- **The GNN keeps more of what it knew.** On every decision where a held-out card is legal, it loses 6.4 points of
  top-1 and 0.13 of set NLL; the MLP loses 10.9 and 0.25. Both lose about the same on the control decisions (1.0-1.2
  points: 22% fewer games), so the excess loss from the unseen cards is about half the MLP's (5.2 points against
  9.9). With ~2,200 acted decisions, the difference is several times its sampling noise.
- **Both over-pick a card they have never seen, the MLP far more.** When the human played something else, the MLP
  without the cards drops from 0.802 to 0.621, while it *gains* on the decisions where the human did play the card
  (0.774 → 0.841). It puts the unknown card on top whatever the situation, which is right when the human cast it and
  wrong otherwise. The cause: its policy heads score slots of a fixed action vocabulary, and training's softmax runs
  over the legal options only, so an unseen card's slot keeps its initial score while every rival it would have
  faced was pushed down. The GNN builds the card from its parts (types, cost, power, abilities), and its
  drop when the human played something else is less than half the MLP's (−8.2 against −18.1).
- **The GNN's reading is supported, with a limit.** It generalizes better to unseen cards than the flat network, but
  it still loses 5 points beyond the control on them: the known parts don't fully stand in for having seen the card.

### The choice

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
| `gnn-full` (Secure RTX PRO 6000, IS) | stage 3's large training | 3:10 AM PT | 6:39 AM PT | 3.5 | 2.09 | $7.31 |
