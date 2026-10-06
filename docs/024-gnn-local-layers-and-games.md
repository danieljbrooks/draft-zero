# MageZero's graph network, part 2: local layers, a full run on Dama's GPUs, and games

*Follows [docs/023](023-gnn-run-log.md), where the GNN at width 256 beat experiment #4's MLP and transformer on
every offline measure (test set NLL 0.214 against 0.229 and 0.234). Started Monday 5 October 2026, 8:30 AM PT, at
Dan's request after Will's review ("this looks great! also try to ablate GNN local layers ... the num local layers is
what im most interested in though at high data scale"). All times are Pacific.*

## Status (Monday 5 October, 3:20 PM PT)

| Stage | Status | Where | Spend |
|---|---|---|---|
| 1. Code: `local_depth`, the streamed cache, the wsd schedule | done, on main | laptop | – |
| 2. Round 7: local layers at 10%, three epochs | **done**: local depth 3, 3 passes x depth 2 and FFN 1,024 gain ~0.008 in set NLL | r1 | free |
| 3. Round 8: long runs (8 epochs of 30%) | **done**: FFN 1,024 (0.2210, the best value head, 1.04x the base's cost) chosen; 3 passes x depth 2 ties it at 2x the cost; base 0.2221; dropout 0.2 0.2286 | r1 | free |
| 4. The full run | **running** on r1's GPU 0 at ~6,500 states/s (block-local sampling since 4:55 PM PT): all the games, width 256, FFN 1,024; 16 epochs end ~12:30-1 AM PT, the decay starts ~10:45 PM PT | r1 | free |
| 5. Games | a preview ladder of docs/023's network: **policy alone 42.7%, 100 simulations 64.1%** (103 games each; the MLP's PIMC: 40%, 57%); 300 simulations running | Community A4000 | ~$1.20 so far |
| 6. 17lands analysis (docs/019 §4.4) | | laptop | – |

**Dan's added goals (9:10 AM PT):** save and plot every run's curves, to audit whether the network is still learning
at the end; and the full model may be used in production, so it may train for as long as it keeps improving (up to
~48 hours). The sweep's goal is the final model, not the best short run: short runs favour fast learners (how width
256 and dropout 0.1 won in docs/023), and both choices may reverse in a long run. Hence round 8's long runs, and the
**wsd learning-rate schedule** (`lr_schedule: wsd`): warm up, hold the peak, cosine down over the last
`wsd_decay_frac` (20%). A run still holding the peak can be extended by resuming with a larger `max_epochs`, so the
full run's length needn't be fixed in advance.

## The plan (Dan, 5 October)

1. **Small-scale experiments, Will's question first:** the number of local layers. In MageZero's network each of
   the 2 bottom-up passes runs one `LocalLayer` per node type. docs/023 varied the passes (1 was worse) and the
   global layers, but not the layers per type. `local_depth` (graph_net.py) stacks that many LocalLayers per type
   within a pass, each re-attending from the node's updated embedding to the same children; depth 1 is Will's
   network, with his checkpoint layout. Width stays 256 (Dan: for speed).
2. **"At high data scale":** the most promising shapes again on 30% of the games.
3. **A full-scale run** of the best recipe on all the training games, **on Dama's RTX 6000s** (r1).
4. **Evaluation**, on cheap rented GPUs (RTX 3070/3090, Community): 100 paired games at 0, 100, 300, 1,000 and
   3,000 simulations against the baseline (MageZero's heuristic bot at 100), with PIMC as in docs/019 §4.6; and
   10,000 policy-only self-play games for docs/019 §4.4's 17lands analysis. Games an hour and games per dollar.

## 1. Fitting all the rows on r1

r1's container has 40 GiB of RAM. All the training rows hold 2.50 billion graph nodes and 5.10 billion edges (the
six tables), ~58 GB in the trainer's arrays: docs/023's full run went to a rented 188 GB pod for that reason.

- **Compact dtypes after mapping** (`_compact`): node rows as int16 (1,932 leaves) and edge labels as int8 (22
  labels), 2.5 of a node's 7 bytes and 3 of an edge's 8: ~38 GB for all the rows. Every run gets this, in RAM or not.
- **`stream_cache`**: the vocab is counted a chunk of rows at a time (300,000), then each table is mapped and
  written a chunk at a time into preallocated `.npy` files, memory-mapped for training. No step holds a whole
  table. Same vocabs and arrays as the in-memory path (`test_stream_cache_matches_the_in_memory_path`). The cache
  sits on the container's scratch disk (`/var/tmp`, 153 GB free; lost on a restart), and the page cache keeps most of
  it resident.

## 2. Round 7: the number of local layers (10% of the games, three epochs)

`configs/gnn_sweep_r7.yml`, r1's two GPUs, 8:50-10:00 AM PT. The base is docs/023's three-epoch recipe at 10%
(width 256, embeddings at std 0.02, dropout 0.1, the game result as the value target, lr 1e-4 cosine, batch 256).

| Run | Parameters | Set NLL | Top-1 acted | Attacks | Blocks | Targets | Value AUC | Value log-loss | Train time |
|---|---|---|---|---|---|---|---|---|---|
| base, seed 0 (docs/023's r5-3ep) | 9.2M | 0.256 | 0.813 | 0.839 | 0.726 | 0.702 | 0.776 | 0.555 | 8.8 min |
| base, seed 1 | 9.2M | 0.254 | 0.824 | 0.822 | 0.719 | 0.699 | 0.772 | 0.556 | 9.4 min |
| local depth 2 | 16.6M | 0.256 | 0.811 | 0.830 | 0.727 | 0.709 | 0.778 | 0.549 | 13.7 min |
| **local depth 3** | 24.0M | **0.247** | 0.824 | 0.835 | **0.730** | 0.709 | 0.780 | 0.552 | 17.8 min |
| **3 passes, local depth 2** | 24.0M | **0.247** | **0.828** | 0.829 | 0.725 | 0.711 | **0.782** | **0.547** | 20.1 min |
| 4 passes | 16.6M | 0.259 | 0.805 | 0.830 | 0.725 | 0.706 | 0.776 | 0.553 | 15.5 min |
| **FFN 1,024** | 13.4M | 0.248 | 0.823 | 0.833 | 0.726 | **0.713** | 0.773 | 0.556 | 11.6 min |
| 3 passes | 12.9M | *lost at 2.25 epochs: 0.258, the base's curve* | | | | | | | |
| 8 heads, 3 global layers, EMA | | *lost to the restart* | | | | | | | |

![Six panels of validation curves over three epochs of 10% of the games: set NLL, non-Pass top-1, attack accuracy, target top-1, value AUC and value log-loss. Local depth 3 and 3 passes x depth 2 lead on set NLL and top-1 from about half way; FFN 1,024 joins them by the end; local depth 2 tracks the two base seeds.](img/024-r7-curves-light.png)

*Figure 1. Round 7's validation curves (the two grey dashed lines are the base's two seeds).*

- **Will's local layers help, in some shapes.** Three LocalLayers per type (depth 3), or 2 per type over 3 passes,
  gain ~0.008 in set NLL over the base (two seeds: 0.254, 0.256), about 4x the seed-to-seed difference, with top-1
  +0.01, blocks and targets up and the value log-loss down. Depth 2 alone and 4 passes of depth 1 don't.
- **A wider FFN does almost as well for less:** 0.248 at 1.2x the base's training time (the deeper shapes take 2x).
- **Every curve is still falling at three epochs**, the deeper ones fastest: round 8 trains them longer.
- Depth costs inference: depth 3 and 3 passes x depth 2 have 2.6x the parameters and roughly twice the local-layer
  work of the base (measured in the games, below, once the shape is chosen).

### r1's restarts (10:00 and 11:24 AM PT): the container's storage cap

r1's container restarted twice, each time killing every process (round 7's last four arms; round 8's first 3.5
epochs and a cache rebuild) and wiping `/var/tmp`. **The cause was our caches in `/var/tmp`**, which sits in the
container's own writable layer on the host's disk. Kubernetes caps that ephemeral storage per container (here, it
seems, ~40 GB) and evicts a container over it; Dan got a low-disk alert for r1. Both times we had written ~39-42 GB
there: the full cache (37 GB) plus the start of round 8's 30% cache, then the 30% cache (11 GB) plus 31 GB of the
full cache's rebuild. A first guess, running out of memory, was wrong: a memory guard (`~/r1_memguard.sh`, killing
our newest job past 32 GiB of anonymous + dirty memory) was running the second time and never fired.

- **Only round 8's 30% cache (11 GB) goes in `/var/tmp`**; round 8 resumed from its checkpoints (the base's run
  saves `latest.pt` every 15 minutes in the persistent home).
- **The full cache (37 GB) needs a persistent disk.** r1's home (197 GB) has 5.8 GB free: experiment #4's runs
  (98 GB), its MLP data caches (39 GB) and a Hugging Face cache (18 GB) fill it. Freeing space there, or more disk
  from Dama, is Dan's call (docs/024 §4).

## 5. Games: the pipeline and its cost

**Pods.** Community availability was poor on 5 October: two Community 3070s ($0.13/h) landed on the same host, where
CUDA fails ("CUDA unknown error", with any torch build); a Community 3090 ($0.22/h) had a 6.8-core quota (an
i7-11700F) and downloaded at 1.4 MB/s; no Community 4090 or any 3090 with 16+ vCPU was free. The eval pod is a
**Community RTX A4000 at $0.17/h**: a 13.6-core quota (EPYC 7452), 62 GB, 83 MB/s; ~$0.0125 per core-hour, below a
Secure 3090's $0.016. The games are CPU-bound, so cores per dollar is the measure.

**Throughput test** (docs/023's network, 12 workers, 2 graph servers):

| Games | Valid | Errors | Median game | Games an hour | $ a game |
|---|---|---|---|---|---|
| PIMC, `gnn@100` against `heuristic@100`, guessed decks | 12 | 0 | 8.8 min | 53 | $0.0032 |
| greedy self-play, open decklists | 35 of 40 | 3 Java heap (2 GB heaps), 2 engine ("Error in unit tests") | 49 s | ~530 steady | $0.0003 |

- The search games are slow next to the MLP's (median 2.8 minutes at 100 simulations, docs/019): the pod runs at a
  load of ~28 on 13.6 cores, and each simulation waits on the graph server. Heaps are 3 GB from here.
- **The evaluation's estimate on A4000s:** 10,000 self-play games ~19 pod-hours; the PIMC ladder ~1.5 (100
  simulations), ~3.5 (300), ~11 (1,000) and ~33 pod-hours (3,000): ~60 pod-hours, ~$10, spread over 4-6 pods.
- **A preview ladder** of docs/023's network (0, 100 and 300 simulations, 100 games each) is running on the A4000
  while the final model trains: early playing-strength numbers against the MLP's PIMC ladder (docs/019 §4.6: 40%,
  57%, 60% with guessed decks), and measured games an hour.

## 3. Round 8: long runs at 30% of the games

`configs/gnn_sweep_r8.yml`: 8 epochs of 30% of the games (3.3M decisions), lr 1e-4 held to epoch 6.4 and cosine down
over the last 20% (wsd); r1's two GPUs, reading one memory-mapped 30% cache. The runs restarted twice with r1's
container (§2) and resumed from their checkpoints.

| Run | Parameters | Set NLL (best) | Top-1 acted | Attacks | Blocks | Targets | Value AUC | Value log-loss | Train time |
|---|---|---|---|---|---|---|---|---|---|
| base (2 passes, depth 1) | 9.2M | 0.2221 (0.2217) | **0.847** | 0.874 | 0.773 | **0.786** | 0.789 | 0.558 | 1.4 h |
| 3 passes, local depth 2 | 24.0M | **0.2210 (0.2207)** | 0.846 | **0.876** | **0.783** | 0.784 | **0.794** | 0.557 | 2.8 h |
| **FFN 1,024** | 13.4M | **0.2210 (0.2210)** | **0.847** | 0.872 | 0.775 | 0.775 | **0.796** | **0.541** | 1.5 h |
| dropout 0.2 | 9.2M | 0.2286 | 0.841 | 0.865 | 0.762 | 0.763 | 0.791 | 0.545 | 1.3 h |
| *docs/023's 3 epochs, cosine (s30-d256-3ep)* | *9.2M* | *0.2317* | *0.838* | *0.860* | *0.752* | *0.744* | *0.799* | *0.525* | *1.1 h* |

- **Will's local layers at high data scale: the gain shrinks with data and training.** At 10% of the games and three
  epochs, 3 passes x depth 2 led the base by 0.008 in set NLL; on 30% for eight epochs, by 0.001 (0.2207 against
  0.2217), for 2.6x the parameters and twice the training and inference cost. It keeps a small edge on blocks (+1
  point) and the value AUC (+0.005). At this scale the extra capacity isn't what limits the network.
- **Eight epochs beat three:** set NLL 0.2217 against docs/023's 0.2317 on the same 30%, top-1 +0.9 points, targets
  +4. The policy still improved through the last epoch.
- **The value head memorises late, except with the wider FFN.** The base's and the deep network's value log-loss
  rose from ~0.542 to ~0.557 in the decay, while their policies kept improving; FFN 1,024's stayed at 0.541, with
  the best value AUC (0.796). Dropout 0.2 held the value off too (0.545) but cost the policy 0.007.
- **The value head prefers a decaying rate to more epochs.** docs/023's three-epoch run, whose rate decayed
  throughout, reached a value log-loss of 0.525 and AUC 0.799 at 10M rows, better than any eight-epoch run (0.541-
  0.558, ~0.79) with 2.6x the rows; its policy, though, ends 0.010 behind. The value head improves while the rate
  falls and stops once each game has been seen a few times; the policy keeps gaining from repeats. For the full run:
  `best_value` will likely come early and `best_policy` late, and the trainer keeps both.
- **The choice: FFN 1,024 at width 256** (13.4M parameters): the deep network's set NLL at 1.04x the base's training
  time (the deep network: 2x), the best value head, and no late value overfit. Width stays 256, as Dan asked.

![Six panels of validation curves over eight epochs of 30% of the games: set NLL, non-Pass top-1, attack accuracy, target top-1, value AUC and value log-loss. The base, FFN 1,024 and 3 passes x depth 2 overlap on the policy and end near 0.221 set NLL; dropout 0.2 trails throughout. docs/023's three-epoch cosine run (grey) leads at equal rows on every panel and reaches the lowest value log-loss, 0.525.](img/024-r8-curves-light.png)

*Figure 2. Round 8's validation curves against docs/023's three-epoch run on the same 30% (grey).*

### The full run's first try: starved of page cache

The full run (all 10.9M decisions; `configs/gnn_full_r1.yml`) started at 1:54 PM PT on GPU 0 beside round 8 on GPU 1.
Its memory-mapped data (37 GB) and round 8's (12 GB) don't fit together in r1's 40 GiB, so the full run's random
reads mostly went to disk: GPU 0 at 0%, 13M major page faults, 0.25 epochs in 68 minutes (~670 states a second, a
tenth of its speed). Stopped at 3:02 PM PT; restarted alone at 4:30 PM PT with round 8's winner (FFN 1,024; the base's 0.25 epochs
don't carry over to another shape), round 8's 30% cache deleted to leave it the whole page cache.

**Still starved, then fixed (4:55 PM PT).** Alone, it ran at 2,182 states a second with GPU 0 at 0-11%: 37 GB of
memory-mapped rows and ~3 GB of the process sit right at the 40 GiB cap, so a random order of rows faulted pages
back from disk (~1,800 major faults a second). **Block-local sampling** (`sample_block_rows: 32`,
`sample_window_blocks: 64`): the order visits random blocks of 32 consecutive rows and shuffles the rows of 64
blocks at a time, so every row is still seen once an epoch and a batch still draws from ~64 places, but the disk reads
contiguous runs. Resumed from its checkpoint: GPU 0 at 57-78%, ~6,500 states a second (r1's GPU at batch 256).

### The full run's curves so far

![Six panels of validation curves against training rows seen on all the training games: set NLL, non-Pass top-1, attack accuracy, target top-1, value AUC and value log-loss. The GNN full run (blue, 4.5 epochs so far, learning rate at its peak) reaches set NLL ~0.216 and top-1 ~0.853 at 49M rows; docs/023's GNN (orange, 3 epochs, cosine) ends at 0.213; the MLP (green, 1 epoch) ends at 0.227 at 11M rows and the transformer (purple, 3 epochs) at 0.234. The full run's value curves are noisier and below docs/023's.](img/024-full-curves-light.png)

*Figure 3. Validation curves on all the training games (6:30 PM PT, 4.5 epochs): the full run against docs/023's GNN
and experiment #4's MLP and transformer, scored on experiment #4's rows for the latter two.*

- **Both GNNs sit well below the flat networks at every equal number of rows seen**: at 11M rows (one epoch) the
  MLP reaches 0.227 and the full run 0.236 with its rate still at the peak; past one epoch the MLP gains nothing
  (experiment #4), while the GNN keeps falling (0.216 at 49M rows).
- **docs/023's run leads the full run at equal rows,** as round 8's three-epoch run led its eight-epoch runs: its
  rate decayed throughout. The full run's held rate trades that for room to keep improving; its decay (epochs
  12.8-16) decides. The held rate also shows in the value curves, which are noisy (log-loss 0.52-0.56) and below
  docs/023's.

## 5b. The preview ladder: docs/023's network in games

docs/023's full GNN (`gnn/main`, three epochs at width 256) against MageZero's heuristic bot at 100 simulations, PIMC
on one world for both bots with guessed decks, the same deck pairs and seed as the MLP's PIMC ladder (docs/019 §4.6);
Community A4000, 12 workers, 3 GB heaps.

| GNN's simulations | GNN | MLP (docs/019 §4.6, guessed decks) |
|---|---|---|
| 0: the policy alone | **42.7%** of 103 | 40% of 100 |
| 100 | **64.1%** of 103 | 57% of 100 |
| 300 | **68.0%** of 103 | 60% of 103 |

At ~100 games a point each difference is within the noise (a 95% interval of about ±9 points on each), but all
three point the GNN's way, by 3-8 points. Records: HF `gnn/games/runs/pimc-gnn-stage3-*`.

**Games an hour** (Community A4000 at $0.17/h, 12 workers, 13.6 cores; the first 100 games of each rung):

| GNN's simulations | Wall time, 100 games | Games an hour | Median game | GNN decisions searched a game | $ a game |
|---|---|---|---|---|---|
| 0 (policy alone; the heuristic searches) | 16 min | ~380 | 1.6 min | – | $0.0004 |
| 100 | 76 min | ~79 | 7.8 min | 84 | $0.0022 |
| 300 | 3.8 h | ~27 | 20.4 min | 84 | $0.0064 |
| 1,000 (projected, x3.3 the 300 rung's time) | ~12.5 h | ~8 | ~65 min | | ~$0.021 |
| 3,000 (projected, x10) | ~37 h | ~2.7 | ~3.4 h | | ~$0.063 |
| greedy self-play (docs/023's network, 40-game test) | – | ~530 | 49 s | – | $0.0003 |

For the final model's ladder: ~55 pod-hours (~$9) for 0-3,000 simulations, ~19 for 10,000 self-play games (~$3),
spread over several pods so it finishes in under a day.

## Pods and spend

| Pod | What | Rented | Removed | Cost |
|---|---|---|---|---|
| `gnn-eval-a`, `-b` (Community RTX 3070, $0.13/h) | the pipeline test; CUDA failed on that host | 8:44 AM PT | 9:25 AM PT | ~$0.10 |
| `gnn-eval-c` (Community RTX 3090, $0.22/h) | a 6.8-core quota and 1.4 MB/s: replaced | 9:30 AM PT | 10:00 AM PT | ~$0.11 |
| `gnn-eval-c` (Community RTX A4000, $0.17/h) | throughput tests and the preview ladder | 9:58 AM PT | 4:43 PM PT | ~$1.15 |
| r1 (Dama's RTX PRO 6000s) | rounds 7-8, the full run | | | free |
