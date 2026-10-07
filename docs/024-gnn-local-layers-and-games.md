# MageZero's graph network, part 2: local layers, a full run on Dama's GPUs, and games

*Follows [docs/023](023-gnn-run-log.md), where the GNN at width 256 beat experiment #4's MLP and transformer on
every offline measure (test set NLL 0.214 against 0.229 and 0.234). Started Monday 5 October 2026, 8:30 AM PT, at
Dan's request after Will's review ("this looks great! also try to ablate GNN local layers ... the num local layers is
what im most interested in though at high data scale"). All times are Pacific. Training ran on r1, the machine with
two RTX PRO 6000s that Dama lent us.*

**Status (Tuesday 6 October, 11:15 PM PT): done.** The sweep, the full run, the ladder, 10,000 self-play games, the
17lands analysis and a follow-up test of the value head (§5) are all finished; the network is on Hugging Face
(`danbrooks/draftzero-checkpoints`, `gnn/full_r1/best_policy_calibrated.pt.gz`).

## Summary

| | **GNN, full run (this doc)** | GNN, docs/023 | MLP (experiment #4) | Transformer (experiment #4) |
|---|---|---|---|---|
| Parameters | 13.4M | 9.2M | | |
| Training | 22.3 epochs, ~12 h on one RTX PRO 6000 | 3 epochs | 1 epoch | 3 epochs |
| Test set NLL (lower is better) | **0.2027** | 0.2142 | 0.2289 | 0.2346 |
| Picks the player's move (top-1, non-Pass) | **0.859** | 0.847 | 0.829 | 0.825 |
| Won against the heuristic bot: policy alone | **44.7%** of 103 | 42.7% of 103 | 40.0% of 100 | |
| ... at 100 simulations | 62.5% of 104 | **64.1%** of 103 | 57.0% of 100 | |
| ... at 300 simulations | **69.2%** of 104 | 68.0% of 103 | 60.2% of 103 | |
| ... at 1,000 simulations | 69.3% of 101 | | **73.8%** of 103 | |
| ... at 1,000, with the epoch-7 value head (§5) | **72.8%** of 103 | | | |
| Self-play card ratings against 17lands: commons | 0.27 | | 0.34 | **0.41** |
| ... every non-basic card | 0.39 | | 0.40 | **0.43** |

*The games use PIMC search with guessed decks against MageZero's heuristic bot at 100 simulations (§5). The card
ratings are Spearman rank correlations with 17lands' games-in-hand win rates, from 10,000 games of each network's
policy against itself; 17lands' own games would reach about 0.80 at this sample size (§6).*

- **Will's question, the number of local layers:** more local layers help when data and training are short, and the
  gain disappears at scale. At 10% of the games for three epochs, three local layers per node type gained 0.008 in set
  NLL; on 30% for eight epochs, the best deep shape gained 0.001 at twice the training cost. A wider feed-forward
  layer (1,024) matched the deep shapes at 1.04x the base's cost, and the final network uses it (§2, §3).
- **The full run is our best imitation network by a clear margin:** 0.011 below docs/023's GNN in test set NLL, and
  it was still improving slowly at the end. Its value head (the win-probability estimate) overfit; a one-number
  temperature calibration fixed its overconfidence (§4).
- **In games, it beats the MLP with little or no search, and stops gaining past 300 simulations.** It won 45%, 62.5%
  and 69% at 0, 100 and 300 simulations, 5 to 10 points above the MLP, but 69% at 1,000, where the MLP reached 74%.
  Its value head from epoch 7, before it overfit, scored 72.8% at 1,000, which is within chance of both, so the test
  doesn't pin the flat top on the value head. It plays no better than docs/023's GNN, despite the better offline
  scores (§5).
- **Its own self-play rates cards less like 17lands than the flat networks did** (0.27 on commons, against the
  MLP's 0.34 and the transformer's 0.41), and twice as many of its games reach the 50-turn cap. Better imitation of
  single decisions did not make policy-only self-play more human-like (§6).
- **Cost:** searched games are CPU-bound, and every pod type cost about the same per core-hour. A game cost $0.0008
  with the policy alone, $0.0035 at 100 simulations, $0.011 at 300 and $0.02-0.03 at 1,000; a policy-only self-play
  game cost ~$0.0003. The evaluation cost ~$5 of RunPod time and the value-head test ~$8 more (§7).

## The plan (Dan, 5 October)

1. **Small-scale experiments, Will's question first:** the number of local layers. In MageZero's network each of
   the 2 bottom-up passes runs one `LocalLayer` per node type. docs/023 varied the passes (1 was worse) and the
   global layers, but not the layers per type. `local_depth` (graph_net.py) stacks that many LocalLayers per type
   within a pass, each re-attending from the node's updated embedding to the same children; depth 1 is Will's
   network, with his checkpoint layout. Width stays 256 (Dan: for speed).
2. **"At high data scale":** the most promising shapes again on 30% of the games.
3. **A full-scale run** of the best recipe on all the training games, **on Dama's RTX 6000s** (r1).
4. **Evaluation**, on cheap rented GPUs (RTX 3070/3090, Community): 100 paired games at 0, 100, 300 and 1,000
   simulations against the baseline (MageZero's heuristic bot at 100), with PIMC as in docs/019 §4.6 (3,000 dropped
   for expediency: Dan, 5 October, 10:15 PM PT); and 10,000 policy-only self-play games for docs/019 §4.4's 17lands
   analysis. Games an hour and games per dollar.

**Dan's added goals (9:10 AM PT):** save and plot every run's curves, to audit whether the network is still learning
at the end; and the full model may be used in production, so it may train for as long as it keeps improving (up to
~48 hours). The sweep's goal is the final model, not the best short run: short runs favour fast learners (how width
256 and dropout 0.1 won in docs/023), and both choices may reverse in a long run. Hence round 8's long runs, and the
**wsd learning-rate schedule** (`lr_schedule: wsd`): warm up, hold the peak, cosine down over the last
`wsd_decay_frac` (20%). A run still holding the peak can be extended by resuming with a larger `max_epochs`, so the
full run's length needn't be fixed in advance.

## 1. Fitting all the rows on r1

r1's container has 40 GiB of RAM. All the training rows hold 2.50 billion graph nodes and 5.10 billion edges (the
six tables), ~58 GB in the trainer's arrays: docs/023's full run went to a rented 188 GB pod for that reason.

- **Compact dtypes after mapping** (`_compact`): node rows as int16 (1,932 leaves) and edge labels as int8 (22
  labels), 2.5 of a node's 7 bytes and 3 of an edge's 8: ~38 GB for all the rows. Every run gets this, in RAM or not.
- **`stream_cache`**: the vocab is counted a chunk of rows at a time (300,000), then each table is mapped and
  written a chunk at a time into preallocated `.npy` files, memory-mapped for training. No step holds a whole
  table. Same vocabs and arrays as the in-memory path (`test_stream_cache_matches_the_in_memory_path`). The page
  cache keeps most of it resident. The full cache (37 GB) lives in r1's persistent home (`~/gnn-cache`), not the
  container's scratch disk (below).

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

### r1's restarts (10:00 and 11:24 AM PT): the container's storage cap

r1's container restarted twice, each time killing every process (round 7's last four arms; round 8's first 3.5
epochs and a cache rebuild) and wiping `/var/tmp`. **The cause was our caches in `/var/tmp`**, which sits in the
container's own writable layer on the host's disk. Kubernetes caps that ephemeral storage per container (here, it
seems, ~40 GB) and evicts a container over it; Dan got a low-disk alert for r1. Both times we had written ~39-42 GB
there: the full cache (37 GB) plus the start of round 8's 30% cache, then the 30% cache (11 GB) plus 31 GB of the
full cache's rebuild. A first guess, running out of memory, was wrong: a memory guard (`~/r1_memguard.sh`, killing
our newest job past 32 GiB of anonymous + dirty memory) was running the second time and never fired.

- **Only round 8's 30% cache (11 GB) went in `/var/tmp`**; round 8 resumed from its checkpoints (each run saves
  `latest.pt` every 15 minutes in the persistent home).
- **The full cache (37 GB) went to the persistent home,** after we deleted experiment #4's data caches there and Dama
  enlarged it to 394 GB.

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
  falls and stops once each game has been seen a few times; the policy keeps gaining from repeats.
- **The choice: FFN 1,024 at width 256** (13.4M parameters): the deep network's set NLL at 1.04x the base's training
  time (the deep network: 2x), the best value head, and no late value overfit. Width stays 256, as Dan asked.

![Six panels of validation curves over eight epochs of 30% of the games: set NLL, non-Pass top-1, attack accuracy, target top-1, value AUC and value log-loss. The base, FFN 1,024 and 3 passes x depth 2 overlap on the policy and end near 0.221 set NLL; dropout 0.2 trails throughout. docs/023's three-epoch cosine run (grey) leads at equal rows on every panel and reaches the lowest value log-loss, 0.525.](img/024-r8-curves-light.png)

*Figure 2. Round 8's validation curves against docs/023's three-epoch run on the same 30% (grey).*

## 4. The full run

All 10.9M training decisions, `configs/gnn_full_r1.yml`: width 256, FFN 1,024, dropout 0.1, lr 1e-4 with the wsd
schedule, batch 256, on one of r1's GPUs.

**Starved of page cache, then fixed (Monday 1:54-4:55 PM PT).** Beside round 8, the full run's 37 GB of memory-mapped
rows and round 8's 12 GB didn't fit in r1's 40 GiB together, so its random reads mostly went to disk: GPU at 0%, 0.25
epochs in 68 minutes. Alone, it still ran at only 2,182 states a second, its data right at the memory cap.
**Block-local sampling** (`sample_block_rows: 32`, `sample_window_blocks: 64`) fixed it: the order visits random
blocks of 32 consecutive rows and shuffles the rows of 64 blocks at a time, so every row is still seen once an epoch
and a batch still draws from ~64 places, but the disk reads contiguous runs. Then ~6,500 states a second, GPU at
57-78%.

**Extended to 24 epochs (10:05 PM PT).** At 9.2 epochs the set NLL was still falling (~0.0015 an epoch) with no sign
of overfitting, and r1 was otherwise idle overnight: resumed with `max_epochs: 24` while still holding the peak rate.

**The decay started early (Tuesday 2:20 AM PT).** By epoch 17.5 the set NLL had been level at ~0.205 since epoch ~14
and the value log-loss was drifting up (0.518 at epoch 7, ~0.540 at 16-17.5). On Dan's call the run resumed with
`max_epochs: 22.29`, which put the decay's start at the current step: a cosine from 1e-4 to 1e-5 over the last 4.45
epochs. It ended at 4:31 AM PT.

![Six panels of validation curves against training rows seen on all the training games: set NLL, non-Pass top-1, attack accuracy, target top-1, value AUC and value log-loss. The GNN full run (blue, 22.3 epochs, 245M rows) falls to set NLL ~0.205 by 150M rows, holds there, and drops to 0.201 in the decay; its top-1 climbs to 0.866. docs/023's GNN (orange, 3 epochs) ends at 0.213, the MLP (green) at 0.227 and the transformer (purple) at 0.234. The full run's value AUC peaks near 0.803 around 100M rows and falls to 0.789, and its value log-loss, lowest near 0.52 at 75-150M rows, rises to 0.575.](img/024-full-curves-light.png)

*Figure 3. Validation curves on all the training games: the full run against docs/023's GNN and experiment #4's MLP
and transformer (the latter two scored on experiment #4's rows).*

- **Is it still learning at the end?** The policy, slowly. The validation set NLL fell ~0.002 an epoch from epoch 3
  to 9 and ~0.0008 an epoch from 9 to 15, then held level to 17.8; the decay took off another 0.003 (0.2046 to
  0.2015), and top-1 rose throughout. More epochs at the peak rate would have gained little more; an epoch takes
  about half an hour.
- **The value head stopped improving by epoch 7-9** and then overfit (below).
- **docs/023's run leads at equal rows,** because its rate decayed throughout; the full run's held rate traded that
  for room to keep improving, and ends 0.011 lower.

### The full run's result (test split)

Every network scored on the same test rows (experiment #4's on this build's rows):

| Test split | Set NLL | Top-1 acted | Attacks | Blocks | Targets | Value AUC | Value log-loss | Calibration error |
|---|---|---|---|---|---|---|---|---|
| **full run, best policy, calibrated** (epoch 19.8, T = 1.61) | **0.2027** | **0.859** | **0.893** | **0.810** | **0.819** | 0.783 | 0.539 | **0.023** |
| full run, best policy, raw | 0.2027 | 0.859 | 0.893 | 0.810 | 0.819 | 0.783 | 0.581 | 0.082 |
| full run, final (epoch 22.3) | 0.2029 | 0.858 | 0.893 | 0.808 | 0.818 | 0.783 | 0.587 | 0.086 |
| full run, epoch 7 (`best_value`) | 0.2129 | 0.850 | 0.881 | 0.789 | 0.801 | **0.792** | **0.527** | 0.023 |
| full run, value head re-fit on the frozen final network | 0.2029 | 0.858 | 0.893 | 0.808 | 0.818 | 0.781 | 0.562 | 0.056 |
| *docs/023's GNN (3 epochs)* | *0.2142* | *0.847* | *0.878* | *0.783* | *0.779* | *0.797* | *0.526* | |
| *experiment #4's MLP* | *0.2289* | *0.829* | *0.873* | *0.750* | *0.730* | *0.784* | *0.552* | |
| *experiment #4's transformer* | *0.2346* | *0.825* | *0.862* | *0.748* | *0.709* | *0.781* | *0.549* | |

- **The policy is the best we have trained:** 0.011 below docs/023's GNN in test set NLL and 0.026 below the MLP;
  top-1 +1.2 and +3.0 points, targets +4.0 and +8.9, blocks +2.7 and +6.0.
- **The value head overfit.** Its validation log-loss was best at epoch 7 (0.518) and ended at 0.575, against ~0.43
  on training rows: one noisy bit per game, shown ~350 times over 22 epochs, from positions of a game that a
  network can recognise (two specific decklists). AlphaGo met the same failure (Silver et al., 2016: "successive
  positions are strongly correlated ... the regression target is shared for the entire game") and fixed it with one
  position per game from 30M self-play games. The policy, with 10.9M distinct labels, kept generalising.
- **Re-fitting the value head on the frozen network didn't fix it:** the frozen features already identify the training
  games, so the new head re-learned their results (training loss 0.38, validation back up to 0.588 after a brief dip),
  and the ranking (AUC) can't move without new features.
- **Temperature calibration fixes the overconfidence:** P(win) = sigmoid(2x / T), with one number T fitted on half the
  validation games (`tools/imitation_scale/value_temperature.py`): T = 1.61; on the other half, log-loss 0.584 ->
  0.544 and calibration error 0.084 -> 0.020; on test, 0.581 -> 0.539 and 0.082 -> 0.023, the AUC unchanged by
  construction. T is folded into the value head's last layer: one ordinary network. **The games below use it**
  (Dan).

## 5. Playing strength: games against the heuristic bot

**The setup** is docs/019 §4.6's. The network's bot plays MageZero's heuristic bot, which searches 100 simulations a
decision. Both bots search with **PIMC**: each decision deals the hidden cards once into a world that fits what the
bot has seen, and runs MageZero's tree search on it. The opponent's deck is guessed from real 17lands decks that fit
the cards seen so far, so neither bot knows anything it couldn't know in a real game. The network's bot searches 0
(its policy's first choice), 100, 300 or 1,000 simulations a decision; the network supplies the search's priors
(policy) and its position evaluations (value). Each budget plays the same deck pairs and shuffles as the MLP's
ladder, each pair twice with the seats swapped, which lowers the variance; a game that ends in an engine error is
left out, and one capped at 50 turns would count as half (none were).

| Network's simulations | **GNN, full run** | GNN, docs/023 | MLP (docs/019 §4.6) |
|---|---|---|---|
| 0: the policy alone | **44.7%** of 103 games | 42.7% of 103 | 40.0% of 100 |
| 100 | 62.5% of 104 | **64.1%** of 103 | 57.0% of 100 |
| 300 | **69.2%** of 104 | 68.0% of 103 | 60.2% of 103 |
| 1,000 | 69.3% of 101 | | **73.8%** of 103 |

![Line chart of games won against the heuristic bot by the network's simulations per decision, PIMC with guessed decks. The full GNN (blue): 44.7%, 62.5%, 69.2%, 69.3% at 0, 100, 300 and 1,000 simulations. docs/023's GNN (orange, dashed): 42.7%, 64.1%, 68.0% at 0 to 300. The MLP (green): 40.0%, 57.0%, 60.2%, 73.8%. A hollow blue diamond at 1,000 marks the full GNN's policy with its epoch-7 value head: 72.8%. A dashed line marks 50%.](img/024-ladder-light.png)

*Figure 4. Win rate against the heuristic bot by search budget: the full GNN, docs/023's GNN and the MLP, each
point about 100 paired games. The diamond: the full GNN's policy with its epoch-7 value head (below).*

- **With little or no search, the GNN beats the MLP:** on the same deck pairs it scores 6.0, 5.0 and 9.6 points more
  at 0, 100 and 300 simulations. At ~100 games a point, each gap alone could be chance; all three pointing the same
  way, and docs/023's GNN doing the same, is the evidence. This is where the policy matters most.
- **Its gain stops at 300 simulations,** while the MLP's continues: 69.3% at 1,000, 4.8 points below the MLP on the
  same deck pairs (again within chance at this sample size). Deeper searches lean more on the value head, the full
  run's weak spot (§4); the test below finds no clear effect.
- **The better policy didn't show in games:** the full run and docs/023's GNN score within 3 points of each other at
  every budget (+2.9, −1.9 and +1.9 points on the same deck pairs), despite 0.011 in test set NLL.

### Is the value head why the gain stops? (Tuesday afternoon and evening)

At 1,000 simulations the search leans on the value head more than at 300, and the full run's value head overfit after
epoch 7 (§4). So we replayed the 1,000-simulation rung with the full run's policy and **the value head of its epoch-7
checkpoint** (`best_value`: test value log-loss 0.527 and AUC 0.792, against the calibrated head's 0.539 and 0.783).
The graph server takes the value from a second checkpoint (`graph_server.py --value-model`), so each evaluation runs
both networks. The deck pairs, seats and shuffles are the rung's.

| 1,000 simulations, the same 101 games | Won |
|---|---|
| Full run's policy, its own value head (calibrated) | 69.3% |
| **Full run's policy, epoch-7 value head** | **72.3%** |
| MLP (docs/019 §4.6) | 73.3% |

*The epoch-7 arm won 72.8% of all its 103 games (figure 4's diamond); the table compares the games all three
played.*

- **The earlier value head scores 2.9 points more on the same games,** well within chance: the two arms' results
  differ in 29 of the 101 games, 16 going the epoch-7 head's way and 13 the other. It ties the MLP (73.3% each).
- **So the test doesn't pin the flat top on the value head.** If the overfit value costs anything at 1,000
  simulations, it's a few points, below what ~100 games can show. The flat top itself (69.2% at 300, 69.3% at 1,000)
  may be partly chance: the MLP's ladder sags at 300 too (60.2%) before its 73.8% at 1,000.
- **Interim counts mislead here:** at 95 games the epoch-7 head led by 7.7 points; the last, longest games mostly
  went the other way.
- **Two networks per evaluation double the server's work:** on Secure RTX 3090s the games took a median 2.1-2.4 hours
  (the rung's own: ~1 hour on Community pods), ~$0.12 a game (§7).

## 6. Do the cards rank like 17lands? (docs/019 §4.4)

17lands' best-known statistic is a card's **games-in-hand win rate:** how often a deck wins the games in which that
card was drawn. If a network plays like people do, the same statistic from its own games should rank the cards the
same way. As in docs/019, we played **10,000 games of the network's policy against itself** (its most likely move
every time, no search, open decklists, a 50-turn cap; on r1's CPUs, Tuesday 5:26 AM to 12:58 PM PT) and compared
each card's win rate with 17lands' (791,000 player-games) by Spearman rank correlation. 9,546 games finished without
an engine error.

| Rank correlation with 17lands | Cards | **GNN, full run** | MLP | Transformer | Noise ceiling [5–95%] |
|---|---|---|---|---|---|
| **Commons** | 90 | 0.27 | 0.34 | **0.41** | 0.80 [0.74, 0.86] |
| Uncommons | 98 | 0.40 | 0.43 | **0.45** | |
| Rares and mythics | 79 | **0.44** | 0.43 | 0.42 | |
| Every non-basic card | 267 | 0.39 | 0.40 | **0.43** | 0.80 [0.74, 0.84] |

*The noise ceiling: 17lands' own games, drawn down to the GNN's 16,536 player-games with a winner, correlated with
the full file (60 draws; `gih_ceiling.py`).*

- **The GNN's self-play ranks commons less like 17lands than the flat networks' does:** 0.27 against 0.34 and 0.41.
  Re-drawing the games (bootstrap), the gap to the transformer, −0.13, is larger than chance (5–95%: −0.26 to
  −0.02); the gap to the MLP, −0.07, could be chance (−0.19 to +0.02). Across every card the three are close (0.39,
  0.40, 0.43).
- **The GNN disagrees with the flat networks more than they do with each other:** on commons, 0.63-0.64 between the
  GNN's ratings and either flat network's, against 0.91 between the two flat networks'.
- **Twice as many of its games hit the 50-turn cap:** 13.4%, against 6.7% for the MLP and 6.1% for the transformer
  (most in white and green decks: 18-19% for WG, WR and WB). The greedy GNN reaches more board stalls that neither
  side breaks. The cap doesn't explain the correlations: counting capped games as half a win changes none by more
  than 0.005.

**Card by card.** Among the 90 commons that all three networks drew at least 30 times, each win rate relative to its
source's average over those commons (54.0% for 17lands, 49.8% for the GNN's self-play), and each card's rank of 90:

*17lands' six best commons, and where each network ranks them:*

| 17lands rank | Card | 17lands | GNN (rank) | MLP rank | Transformer rank |
|---|---|---|---|---|---|
| 1 | Bake into a Pie | +4.0 | +1.0 (41) | 27 | 44 |
| 2 | Burst Lightning | +3.9 | −1.7 (61) | 77 | 83 |
| 3 | Stab | +3.9 | −0.2 (54) | 37 | 43 |
| 4 | Dazzling Angel | +3.8 | +8.3 (1) | 2 | 1 |
| 5 | Luminous Rebuke | +3.7 | +4.7 (13) | 9 | 10 |
| 6 | Refute | +3.6 | −0.2 (52) | 24 | 21 |

*The GNN's six best commons, and where 17lands and the flat networks rank them:*

| GNN rank | Card | GNN | 17lands (rank) | MLP rank | Transformer rank |
|---|---|---|---|---|---|
| 1 | Dazzling Angel | +8.3 | +3.8 (4) | 2 | 1 |
| 2 | Cackling Prowler | +7.8 | +0.1 (46) | 17 | 26 |
| 3 | Felidar Savior | +7.2 | +3.4 (10) | 7 | 6 |
| 4 | Blossoming Sands | +7.1 | −0.3 (55) | 16 | 14 |
| 5 | Treetop Snarespinner | +6.9 | +0.6 (39) | 40 | 9 |
| 6 | Healer's Hawk | +6.6 | +3.5 (7) | 4 | 5 |

*The commons the GNN ranks furthest below 17lands:*

| Card | 17lands rank | GNN rank | MLP rank | Transformer rank |
|---|---|---|---|---|
| Involuntary Employment | 28 | 90 | 88 | 88 |
| Fake Your Own Death | 21 | 87 | 79 | 79 |
| Eaten Alive | 20 | 82 | 52 | 71 |
| Fleeting Flight | 16 | 80 | 61 | 53 |
| Infestation Sage | 13 | 68 | 57 | 55 |
| Refute | 6 | 52 | 24 | 21 |

![Two scatter panels of each card's win rate when drawn in the GNN's self-play (vertical) against 17lands' (horizontal): commons on the left (Spearman 0.27, noise ceiling 0.80) and every non-basic card by rarity on the right (0.39). Fake Your Own Death, Involuntary Employment, Eaten Alive and Fleeting Flight sit far below the trend; Gleaming Barrier, Apothecary Stomper, Blossoming Sands and Cackling Prowler sit far above it.](img/024-gih-light.png)

*Figure 5. The GNN's self-play: each card's win rate when drawn, against 17lands'.*

- **Spells whose value is in their timing fall furthest,** as in docs/019: combat tricks (Fake Your Own Death,
  Fleeting Flight), a counterspell (Refute), and removal with a cost or a condition (Eaten Alive sacrifices a
  creature; Involuntary Employment steals one for a turn). The GNN ranks Refute 52nd where the flat networks put it
  21st-24th.
- **Burst Lightning does better** than for the flat networks (61st against 77th and 83rd), though still far from
  17lands' 2nd.
- **Big and defensive creatures rise,** green ones most: Apothecary Stomper (17lands 74th, GNN 8th), Treetop
  Snarespinner, Cackling Prowler and Gleaming Barrier (17lands' last of these 90, the GNN's 24th). That fits the
  board stalls above: in games that stall, the bigger bodies decide them.
- **The GNN's self-play spreads card win rates a little less** than the flat networks' (standard deviation 4.5
  points over the 90 commons, against 4.8-5.1), but still about twice as wide as 17lands' (2.5).

![Deck win rate by colour pair, relative to each source's average, for the GNN's, the MLP's and the transformer's self-play and 17lands' top players, sorted by the GNN's. The GNN puts white-green and white-blue on top (+5 points) and black-red last (−7); it rates green pairs higher than the flat networks do and blue-black lower (−4, where the flat networks and top players sit near 0). Top players stay within about ±3 points.](img/024-colours-light.png)

*Figure 6. Deck win rate by colour pair: each network's self-play and 17lands' top players, relative to each
source's average.*

**Colour pairs.** The GNN's self-play spreads the colour pairs less widely than the flat networks' (43% to 56%,
against 43% to 60% for the MLP and 40% to 59% for the transformer), but still about three times wider than top
players (61% to 66%). It shifts toward green (white-green +5 points, red-green +3) and away from blue-black (−4),
which top players and both flat networks put near average. Black-red is last for all three networks.

**Reading this.** The GNN imitates single decisions best (§4) and plays best with little search (§5), yet its
policy-only self-play is no more human-like in how it values cards, and on commons less. Self-play without search
compounds whatever a policy gets systematically wrong over a whole game, and the GNN's systematic errors differ from
the flat networks' (passive boards, timing spells). As docs/019 said, card ratings from self-play *with* search
should be the better test of how the networks value cards.

## 7. What the games cost: games an hour and games per dollar

**Pods.** The games are CPU-bound: each worker runs a Java game engine, and the GPU serves the network to all of
them. So the measure is cores per dollar, and the three pod types we used cost about the same per core-hour
(about $0.01-0.016). Community stock was poor both days: every RTX 3070 we rented (five on 6 October, two on 5 October)
and two of the 3090s had broken CUDA on their hosts (`fleet_up.sh` now checks CUDA first and removes those, at a few
cents each). The ladder's first three budgets ran on a **Secure** RTX 3090 for that reason.

| Games | Pod | Workers | Median game | Games an hour | $ a game |
|---|---|---|---|---|---|
| PIMC, policy alone | Secure RTX 3090, 32 vCPU, $0.50/h | 30 | 1.9 min | ~610 | $0.0008 |
| PIMC, 100 simulations | Secure RTX 3090 | 30 | 8.6 min | ~140 | $0.0035 |
| PIMC, 300 simulations | Secure RTX 3090 | 30 | 24 min | ~47 | $0.011 |
| PIMC, 1,000 simulations | Community RTX 3090, 16 vCPU, $0.22/h | 12 | 51 min | ~11 | $0.019 |
| PIMC, 1,000 simulations | Community RTX A4000, 13.6-core quota, $0.17/h | 12 | 67 min | ~6.6 | $0.026 |
| PIMC, 1,000 simulations | Community RTX 3090, 25 vCPU, $0.22/h | 9 | 65 min | ~6.4 | $0.034 |
| PIMC, 1,000 simulations, epoch-7 value head (two networks) | Secure RTX 3090, 32 vCPU, $0.50/h | 13-14 | 124-142 min | ~4.1-4.6 | $0.11-0.12 |
| PIMC, 1,000 simulations, epoch-7 value head | r1, Dama's (free) | 6 | 34 min | ~8.7 | free |
| Greedy self-play (policy only) | r1, Dama's (free) | 6 | ~6 s | ~1,330 | free |
| Greedy self-play, docs/023's network | Community RTX A4000, $0.17/h | 12 | 49 s | ~530 | $0.0003 |

*Games an hour: the first 100 games of a budget (the 1,000-simulation rows: one shard of ~34 games each), wall time
including the slow last games. The 1,000-simulation shards ran three ways in parallel, so their per-pod rates include a
long tail.*

- **Search budget drives the cost:** each step from 100 to 300 to 1,000 simulations costs ~3x per game or more
  (a game at 1,000 costs 5-10 times one at 100).
- **A GNN game is slow next to the MLP's:** at 100 simulations, a median 8.6 minutes against the MLP's 2.8 (docs/019
  §4.6, on a different 3090), because every simulation waits on the graph server. Serving the network faster is
  the main speed-up left. With two networks per evaluation, a 3090 pod's GPU was the limit: at 99% busy with four
  server processes it fed 26 workers ~220 evaluations a second, 8 a worker, while r1's faster GPU and CPU gave each of
  its 6 workers ~29.
- **10,000 policy-only self-play games** took ~7.5 hours on r1 (three shards of 3,334 games, 2.4-2.6 hours each),
  and would take ~19 pod-hours (~$3) on a Community A4000 or 3090.

### The preview ladder: docs/023's network (Monday)

While the full run trained, docs/023's three-epoch GNN played the ladder's first three budgets on a Community A4000
(12 workers): 42.7%, 64.1% and 68.0% (in §5's table). That pod's rates: ~380 games an hour at 0 simulations, ~79 at
100 (median 7.8 minutes, $0.0022 a game) and ~27 at 300 (median 20 minutes, $0.0064).

## Pods and spend

| Pod | What | Rented | Removed | Cost |
|---|---|---|---|---|
| `gnn-eval-a`, `-b` (Community RTX 3070, $0.13/h) | the pipeline test; CUDA failed on that host | Mon 8:44 AM | 9:25 AM | ~$0.10 |
| `gnn-eval-c` (Community RTX 3090, $0.22/h) | a 6.8-core quota and 1.4 MB/s: replaced | Mon 9:30 AM | 10:00 AM | ~$0.11 |
| `gnn-eval-c` (Community RTX A4000, $0.17/h) | throughput tests and the preview ladder | Mon 9:58 AM | 4:43 PM | ~$1.15 |
| `eval-k1-s0` (Community RTX 3090, 16 vCPU, $0.22/h) | 1,000 simulations, shard 1 of 3 | Tue ~5:20 AM | 8:44 AM | ~$0.75 |
| `eval-k1-s2` (Community RTX 3090, 25 vCPU, $0.22/h) | 1,000 simulations, shard 3 | Tue ~5:20 AM | 10:47 AM | ~$1.20 |
| `eval-ladder` (Secure RTX 3090, 32 vCPU, $0.50/h) | 0, 100 and 300 simulations | Tue ~6:25 AM | 10:21 AM | ~$1.97 |
| `eval-k1-s1` (Community RTX A4000, $0.17/h) | 1,000 simulations, shard 2 | Tue ~6:28 AM | 11:42 AM | ~$0.89 |
| seven pods with broken CUDA (five 3070s, two 3090s) | removed by `fleet_up.sh`'s check | Tue 5-6:30 AM | minutes later | ~$0.15 |
| `eval-v7-s1` (Secure RTX 3090, 32 vCPU, $0.50/h; no Community stock) | the value-head test, shard 2 of 3 | Tue ~2:13 PM | 10:51 PM | ~$4.32 |
| `eval-v7-s2` (Secure RTX 3090, 32 vCPU, $0.50/h) | the value-head test, shard 3 | Tue ~2:37 PM | 10:06 PM | ~$3.75 |
| r1 (Dama's two RTX PRO 6000s) | rounds 7-8, the full run, the 10,000 self-play games, the value-head test's shard 1 | | closed by Dama Tue ~10 PM | free |
| **Total** | | | | **~$14.40** |


**Records** (Hugging Face `danbrooks/draftzero-checkpoints`): the network and its test results in `gnn/full_r1/`
(`best_policy_calibrated.pt.gz` is the one evaluated; also `best_policy`, `best_value`, `final`,
`value_temperature.json`); the games in `gnn/games/runs/`: `pimc-gnn-full-*` (the ladder), `gnn-full-selfplay-t0-s0..2`
(self-play), `pimc-gnn-full-v7-*` (the value-head test) and `pimc-gnn-stage3-*` (the preview); rounds 7-8's curves in
`gnn/sweeps/r1/` (`sweep_r1g`, `r1h`, `r8g`, `r8h`, `r8i`; their checkpoints went with r1). Figures: `tools/imitation_scale/fig_doc024.py` (4, 6),
`fig_gih.py` (5) and `fig_mlp_curves.py` (1-3).
