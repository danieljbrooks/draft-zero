# From imitation to self-play: results

*Status (final, Friday 9 October 2026, 1:30 PM PT): every stage planned in docs/028 ran except the plain-AlphaZero
loop's second generation (dropped for compute, §6). The plan is [docs/028](028-imitation-to-self-play-plan.md); the code is on the
[`selfplay-transition`](https://github.com/danieljbrooks/draft-zero/tree/selfplay-transition) branch.*

docs/028 planned how to move our imitation-learned graph network (GNN, docs/022-024) to **self-play** without losing
what it learned from top 17lands players, and looked for a recipe we can carry to the faster engine. This doc reports
what happened. The runs used r1 (the machine with two RTX PRO 6000s that Dama lends us) and RunPod Community pods.

A few terms, as in docs/028:
- **Policy:** how likely the network is to pick each legal move. **Value:** its estimate of the chance of winning.
- **Simulations:** a search at N simulations plays N short look-ahead lines before each decision (PIMC: it deals the
  hidden cards once into one consistent world, then searches it). "Policy alone" plays the network's most likely
  move with no search.
- **Paired games** play each pair of decks twice with the seats swapped. "Won 54% of 1,000 paired games" counts a game
  stopped at the 50-turn cap as half.

## Summary

**What we found:**

1. **Train the policy toward the network's own policy tilted by the search's move values, not toward the search's
   visit counts.**
   - Visit counts carry the search's exploration. Trained on them, the policy flattened (entropy 0.26 to 0.66 nats)
     and lost agreement with people: the 17lands validation NLL rose from 0.201 to 0.349, more than all of the GNN's
     gain over the MLP.
   - The tilted target, π'(a) ∝ π(a)·exp(s·(Q(a) − Q_root)) at s = 10, changed only the moves the search disagreed
     with. It kept 17lands NLL at 0.201, and its policy alone played as well as any arm:
     - against the start's policy alone, it won 53.6% of 1,888 paired games;
     - against heuristic@100, 47.1% where the start won 43.0%.
2. **Expect the imitation value to be optimistic, and let one pass of self-play results fix it.**
   - On its own games the start predicted a 61.7% chance of winning where the player to move won 55.7%. The 17lands
     data is recorded from top players' seats, and they won ~60% of their games.
   - One training pass brought the mean to 56.8% and the calibration error from 0.060 to 0.020 (log-loss 0.453 to
     0.417) in every arm.
3. **The TD discount λ (MageZero's `td_discount`) trades calibration on self-play against what the value remembers
   about human games:**
   - λ = 1.0 (the result alone) reaches the true base rate;
   - λ = 0.95 keeps the start's optimism, because its targets are mostly the search's root values, which inherit the
     start's bias;
   - λ = 0.99 sits between them and has the best log-loss.
4. **A KL anchor on top of visit counts is a blunt fix.** It halved the drift away from people but played worse than
   either of the others, policy alone.
5. **Mixing 25% human decisions into every batch is cheap and harmless.** All 10.95M human rows load from the
   imitation run's memory-mapped cache in 7 seconds. It gave the best policy alone against the heuristic bot (48.4%).
6. **With search (100 simulations against the start at 100), every arm won 53-55%,** and the arms tie with each
   other. The value-only arm, whose policy is the start's, got most of the gain: with search, the retrained value
   carries the improvement.
7. **Generation 2 kept improving, smoothly:**
   - policy alone, it beat generation 1 in 52.0% of 756 paired games;
   - with search at 100 simulations, it beat the start in 61.6% of 142, against generation 1's 55.1% on the same deck
     pairs (+6.9 points, about 2 standard errors).

   Its value's offline numbers, though, stopped improving: it trained mostly on games it had already seen.

## 1. What ran

| Stage | What | Where | Result |
|---|---|---|---|
| 0 | the start network plays itself: 1,411 games (2,822 seat-games, 236,707 searched decisions), 100 simulations, exploration on (root Dirichlet noise 0.25, α 0.3; the first 3 turns sampled by visits), the training decks | r1 and four Community pods, split by deck pair | §2 |
| – | the start at 100 simulations against heuristic@100, 150 deck pairs | a Community pod | 68.2% of 286 paired games (docs/024: 62.5% of 104; on the same 52 pairs, within 1 point) |
| 1a | four policy targets, each one training step from the start on the same games | r1 | §3 |
| 1b | six value and anchor variants on the 1a winner | r1 | §4 |
| 1 (games) | each arm's policy alone against the start's (1,000-2,000 paired games) and against heuristic@100 (~286) | r1 | §3-4 |
| 1 (100 simulations) | three arms at 100 simulations against the start at 100 (~180-280 paired games each) | RunPod pods (Community, one Secure) | §5 |
| 2 | generation 2 of the recipe: 400 self-play games, training, checks | a Secure RTX 4090, then Community RTX 4070 Ti and 5080 pods | §6 |

**Training, the same for every arm:**
- 2 passes over the 209,969 training decisions (10% of deck pairs held out by hash, the same for every arm);
- batch 256, AdamW at 3e-5 with 100 warm-up steps and a cosine to a tenth;
- the value on at most 16 positions a game, weight 0.5, TD λ = 0.99 unless the arm changes it;
- 3-6 minutes an arm on one RTX PRO 6000.

The checks ran every 400 steps:
- on the held-out self-play decisions: policy entropy, KL from the start, agreement with the search, and the value's
  log-loss, AUC and calibration error;
- on the 17lands validation decisions: set NLL, non-Pass top-1, the Pass rate, and the value's log-loss and AUC.

## 2. What the start does in self-play

- **Speed:**
  - a game took a median 6.4 minutes of one r1 core at 100 simulations (19 turns, 80 searched decisions);
  - r1 ran 14-20 workers, about 100-130 games an hour;
  - a Community 3090 pod managed 55-75 an hour.
- **Errors:** 4-6% of games ended in an XMage engine error (as in docs/024); none hit the 50-turn cap.
- **Its value is optimistic and its ranking is good** (figure 2, the blue line). Over the held-out games:
  - AUC 0.871;
  - mean prediction 61.7% against 55.7% won (the winner of a game makes more decisions, so more than half of the
    decisions belong to winners);
  - the search's root value is no better calibrated (59.3%).
- **The search's visit counts sit far from the network's policy:**
  - KL 0.93 nats against a policy entropy of only 0.27;
  - the search changes the top move in 12% of decisions;
  - the visits also spread onto options the network rejects, because of the root noise, MageZero's prior temperature
    (1.5) and its 0.1 bonus for everything but Pass (docs/028 §3).

## 3. Stage 1a: the policy target

![Four panels of validation curves over two passes of training, for four policy targets: 17lands validation set NLL (visit counts rise from 0.20 to 0.35 within half a pass; visits with a KL anchor to 0.26; the two tilted targets stay at 0.20), policy entropy on held-out self-play (visit counts 0.26 to 0.66, KL anchor 0.51, tilted s = 30 0.31, s = 10 0.27), value log-loss on held-out self-play (all four fall from 0.453 to about 0.417-0.419) and value calibration error (all four fall from 0.060 to about 0.02).](img/029-curves-light.png)

*Figure 1. Stage 1a's arms as they train (`tools/selfplay_transition/fig_doc029.py`).*

| Arm | Policy target | Entropy | KL from start | 17lands NLL | Top-1 (non-Pass) | vs start's policy | vs heuristic@100 |
|---|---|---|---|---|---|---|---|
| start | – | 0.26 | 0 | **0.2010** | 0.864 | – | 43.0% of 286 |
| `visits` | the search's visit shares | 0.66 | 0.163 | 0.3485 | 0.848 | 52.9% of 952 | 46.4% of 292 |
| `kl` | visits + 1.0 × KL(start ‖ network) | 0.51 | 0.072 | 0.2601 | 0.862 | 51.9% of 952 | 44.4% of 286 |
| `cq10` | the network's policy tilted by the search's values, s = 10 | 0.27 | 0.004 | 0.2012 | **0.864** | 53.6% of 1,888 | **47.1%** of 280 |
| `cq30` | the same, s = 30 | 0.31 | 0.011 | 0.2016 | 0.863 | **55.1%** of 948 | 46.9% of 286 |

*Policy alone, greedy. Each "vs" column counts paired games, a 50-turn stop counting half; 13-16% of the policy-alone
games against the start stalled to the cap (as in docs/024's self-play). Engine errors dropped both games of a pair:
3-4% of games.*

- **Every target made a better policy than the start's.** Even visit counts, which flattened it: greedy play takes
  the top option, and the flattening rarely changed the top option.
- **The tilted targets did best on every measure.** They gained 4-5 points against the heuristic bot on the same deck
  pairs, and 3.6-5.1 points head to head against the start, with agreement with people unchanged.
- **`cq30` against `cq10`:** 1.5 points apart head to head, within docs/028's 2-point tie band, so the rule chose the
  one closer to the start: **`cq10` won stage 1a.**
- **Where they changed the policy** (5,477 held-out decisions):
  - **ordinary decisions:** `cq10` changed the top move in 1.6% of priority decisions (what to cast or activate) and
    3.2% of target decisions (targets, attacks, blocks), close to the 12% where the search disagreed;
  - **the yes/no head:** in the rare (1.3%) "may" decisions, the start agreed with the search only 46% of the time:
    imitation never trained that head. `cq30` raised that to 64%.

## 4. Stage 1b: the value and the anchor

![The calibration of the value on 26,738 held-out self-play decisions: the share of positions whose player won against the predicted chance, in ten bins. The start network (mean 61.7% against 55.7% won, calibration error 0.060) and the search's root value (59.3%, 0.066) run below the diagonal between 0.3 and 0.8; after one training step at λ 0.99 (56.8%, 0.020) and λ 1.0 (55.6%, 0.018) the curves sit near the diagonal; λ 0.95 (58.6%, 0.042) stays between.](img/029-calib-light.png)

*Figure 2. The value on held-out self-play games before and after one training step (`value_calib.py`).*

| Arm (on `cq10`) | Change | Value log-loss | Calibration error | Mean prediction | Human-game value log-loss | 17lands NLL | vs start's policy | vs heuristic@100 |
|---|---|---|---|---|---|---|---|---|
| start | – | 0.453 | 0.060 | 61.7% | **0.531** | 0.2010 | – | 43.0% |
| `cq10` | λ 0.99 | **0.417** | 0.020 | 56.8% | 0.554 | 0.2012 | 53.6% | 47.1% |
| `b_lam1` | λ 1.0 (the result alone) | 0.420 | **0.017** | 55.6% | 0.583 | 0.2012 | 53.3% | 46.6% |
| `b_lam95` | λ 0.95 | 0.427 | 0.042 | 58.6% | 0.533 | 0.2013 | 53.7% | 45.3% |
| `b_vwarm` | the value head alone first (a quarter of the steps, the result as target, lr 3e-4), then everything | 0.420 | 0.027 | 56.5% | 0.551 | 0.2013 | 52.5% | 43.8% |
| `b_vonly` | no policy loss; the policy pinned by a KL to the start (10) | 0.420 | 0.029 | 57.2% | 0.546 | 0.2003 | 50.2% | 43.8% |
| `b_human` | 25% of every batch from the human training decisions (their policy loss only) | 0.418 | 0.024 | 57.0% | 0.550 | **0.2003** | 52.7% | **48.4%** |
| `b_lr1e4` | learning rate 1e-4 | 0.422 | 0.017 | 56.9% | 0.572 | 0.2027 | *on r1's disk (r1 down)* | 47.5% |

*Value columns: the 26,738 held-out self-play decisions, where the player to move won 55.7%. Human-game value:
17lands validation. Policy-alone matches as in §3: ~950 paired games against the start, ~286 against the heuristic
bot.*

![Two panels of horizontal bars, one per stage-1 arm: the policy alone against the start's policy alone (left; visit counts 52.9%, KL anchor 51.9%, tilted s = 10 53.6%, s = 30 55.1%, then the 1b variants between 50.2% and 53.6%) and against heuristic@100 (right; the start 43.0%, the arms 43.8% to 48.4%).](img/029-stage1-light.png)

*Figure 3. Every stage-1 arm's policy alone: blue, the policy targets; orange, the value and anchor variants on
`cq10`.*

- **λ changes calibration and memory, not ranking:**
  - every arm ranks self-play positions about equally (AUC 0.884-0.887);
  - λ = 1.0 learns the new base rate but forgets the most about human games (log-loss 0.531 to 0.583);
  - λ = 0.95 keeps the human-game value but stays optimistic: bootstrapped targets inherit the bias of the value they
    bootstrap from.

  Will found λ the most influential setting in XMage. Here its effect shows in the value, and whether it matters to
  play needs the search (§5).
- **Policy alone can't separate value changes:**
  - the value-only arm scored 50.2% against the start, as it must with the policy pinned;
  - its 43.8% against the heuristic bot equals the start's 43.0% within chance;
  - the gains in §3 come from the policy, not from a better value in the policy's few searched fallback decisions.
- **Human rows:** the best 17lands NLL (0.2003) and the best score against the heuristic bot. At this size the
  rehearsal costs nothing.
- **A higher learning rate moved the policy twice as far** (KL 0.009) and helped nothing offline.

## 5. With search: 100 simulations against the start at 100

Each network searched 100 simulations a decision, as did the start, on the evaluation decks (docs/024's deck-pair
seed). These are the slowest games in the study, two GNNs searching against each other: 150 deck pairs per match, 100
for the value-only arm.

| Network at 100 simulations | Against the start at 100 | Same deck pairs, compared with `cq10` |
|---|---|---|
| `visits` (policy: visit counts) | **55.3%** of 274 paired games | +0.2 points (133 pairs) |
| `cq10` (policy: Q-tilted, s = 10) | 54.3% of 276 | – |
| `b_vonly` (the value only; the policy pinned to the start's) | 53.3% of 182 | −2.2 points (89 pairs) |

*Engine errors dropped 12-13 pairs per match; no game hit the 50-turn cap.*

- **With search, every arm beats the start by 3-5 points.**
- **The arms tie with each other.** Visit counts and the Q-tilted target are 0.2 points apart on the same deals (48 of
  133 pairs differ).
- **The value-only arm gets most of the gain.** Its policy is the start's, so the retrained value carries most of
  the improvement with search.
- **What the policy target buys:**
  - with search at 100 simulations, it mattered little here;
  - without search, the tilted target played best (§3) and kept agreement with people, where visit counts lost it.

## 6. Stage 2: generation 2

The 1a winner continued for one more generation:
- **Self-play:** `cq10` (generation 1) played 400 games against itself (200 deck pairs, 388 finished), with stage 0's
  settings.
- **Training:** generation 2 trained from generation 1 with `cq10`'s recipe on stage 0's games plus the new ones:
  1,799 games, 267,357 decisions, 2 passes, 5.5 minutes on an RTX 4090.

The plain-AlphaZero loop's second generation was dropped: r1 was down and Community pods were scarce (§8).

**Offline,** on the held-out games of both generations (32,891 decisions):

| | Generation 1 (`cq10`) | Generation 2 |
|---|---|---|
| Value log-loss | **0.411** | 0.414 |
| Value AUC | **0.891** | 0.889 |
| Value calibration error | 0.020 | **0.017** |
| Policy KL from the start | **0.004** | 0.012 |
| Policy entropy | 0.261 | 0.260 |

**In games:**

| Generation 2 against | Won | Generation 1 (`cq10`) against the same |
|---|---|---|
| the start's policy alone (policy alone) | 53.7% of 750 paired games | 53.6% of 1,888 |
| generation 1's policy alone (policy alone) | **52.0%** of 756 | – |
| heuristic@100 (policy alone) | **47.6%** of 290 | 47.1% of 280 (the start: 43.0% of 286) |
| the start at 100 (both at 100 simulations) | **61.6%** of 142 (deck pairs 0-74) | 55.1% of 138 on the same deck pairs (54.3% of 276 on all 150) |

*Policy alone: 11-15% of games against the start or generation 1 stalled to the 50-turn cap (counted as half), as
in stage 1; engine errors dropped 3-4% of games.*

- **No drift:**
  - the policy stayed sharp and close to people;
  - generation 2 keeps generation 1's edge over the start;
  - it beats generation 1 head to head by 2 points (52.0% of 756 paired games, about 2 standard errors);
  - it scores a little higher against the heuristic bot.
- **With search, generation 2 improved further:**
  - 61.6% against the start at 100 simulations, where generation 1 won 55.1% on the same 69 deck pairs;
  - +6.9 points, about 2 standard errors; the two matches differ in 25 of the 69 pairs.
- **Its value's offline numbers did not improve:**
  - most of generation 2's decisions came from stage 0's games, now trained on for a third and fourth time
    (docs/027's lesson: games, not positions, limit the value);
  - yet it played better with search;
  - the held-out log-loss on these games is a weak guide to play, as in docs/014 and docs/024.

![Three lines across the start, generation 1 and generation 2: 100 simulations against the start at 100 on deck pairs 0-74 (50%, 55.1%, 61.6%), the policy alone against the start's policy alone (50%, 53.6%, 53.7%), and the policy alone against heuristic@100 (43.0%, 47.1%, 47.6%), with one-standard-error bars.](img/029-generations-light.png)

*Figure 4. Generation by generation (`fig_doc029.py --gens`).*

## 7. What we'd do on the new engine

1. **Measure the start on its own self-play games before training:**
   - the value's calibration (`value_calib.py`);
   - how far the search's visit counts sit from the policy (`target_stats.py`).

   Both cost minutes and predicted what went wrong here: an optimistic value, and targets 1 nat from the policy.
2. **Policy target:** the network's own policy tilted by the search's values,
   π'(a) ∝ π(a)·exp(s·w(a)·(Q(a) − Q_root)) with s ≈ 10-30 and w(a) = n/(n + 2).
   - Raw visit counts carry the search's exploration, prior temperature and prior bonus, and they flatten the policy.
   - With search at play time the two tied (§5); without search, the tilted target played better and kept agreement
     with people.
3. **Value target:** the game result blended with the search's root values, λ = 0.99 (MageZero's labels).
   - One pass recalibrated the imitation value.
   - Lower λ keeps more of the start's bias, because bootstrapped targets inherit it.
   - λ = 1.0 forgets the most about human games.
   - The value is what improved play with search (§5).
4. **Keep human data in the batches:** 25% human rows cost nothing with a memory-mapped cache and kept 17lands NLL at
   the start's level. A KL anchor on top of raw visits is a poor substitute for a better target.
5. **Games, not positions:**
   - each generation needs mostly new games: reusing old games stalled the value's offline numbers in generation 2;
   - cap reuse at ~2-4 passes;
   - on a fast engine, many thousands of games a generation are cheap.
6. **Judge a recipe over at least two generations.** Generation 2 gained 6.9 more points with search than generation
   1, with no drift away from people. One step from the start understates where the recipe goes.
7. **Learning rate:** 3e-5 for two passes was enough; 1e-4 moved the policy further with no gain.
8. **Judge cheaply, then expensively:**
   - offline checks and policy-alone matches (thousands of games in an hour) for every arm;
   - search matches only for the finalists;
   - deck pairs played both ways, and scores read at a fixed size, not as they come in.

## 8. Operations

- **r1 changed under us:**
  - the container restarted twice (7:25 PM and 1:19 AM PT) and came back with 40 GiB and 24 cores instead of
    54 GiB and 30;
  - another tenant appeared on one GPU;
  - on Thursday afternoon it went down for Dama's RAM upgrade.

  Everything resumed, because the runs write results as they go, but each restart cost the games in progress.
- **Memory:**
  - JVMs at 2.5 GB of heap grew toward the container's limit in self-play, so 1.4 GB was used there;
  - 1.4 GB was too small for policy-alone games: one out-of-memory error poisoned a JVM, and every later game on it
    failed;
  - 3 GB fixed it.
- **Community pods:**
  - three RTX 3090 Ti hosts couldn't reach GitHub or Hugging Face;
  - one 3090 host with broken CUDA was offered again and again;
  - stock ran out for most of Thursday.
- **A pod's host rebooted twice on Friday morning** (RunPod's "network heartbeat" notice):
  - the games in progress died each time;
  - the matches resumed from their saved games.

  The last pod ran the end of generation 2's match, then removed itself.
- **The first 100-simulation match on an RTX 4070 Ti pod crawled:** JVMs at a 1.4 GB heap spent most of their time in
  garbage collection, with Java sizing its collector for the host's 112 CPUs. A 2.2 GB heap and capped collector
  threads (`MZB_JAVA_OPTS="-XX:ParallelGCThreads=4 -XX:ConcGCThreads=1"`) fixed it.
- **Cost:** $17.10 of RunPod, against a $15 budget (Dan allowed going slightly over to finish):
  - stage 0's four pods ~$4.20;
  - the ladder ~$0.70;
  - the stage-1 100-simulation matches ~$4.50, one on Secure;
  - generation 2's self-play and training on a Secure RTX 4090 ~$3.50;
  - generation 2's checks on a 4070 Ti and a 5080 ~$1.70;
  - idle time ~$2. After the laptop slept, a finished pod sat for two hours: the fetch scripts that removed pods had
    crashed when they were edited while running.
  - failed or probe pods ~$0.50.

  r1's time was free.

## 9. Next steps: gorge and SpellBench

### What should carry over to gorge

docs/026-027 trained gorge's networks from gorge's own search, not from people. Their problems were the same as here:
- the value memorised after half an epoch;
- visit targets at 100 simulations were near-uniform;
- games, not positions, limited learning.

What this study adds:

1. **Diagnose before training:**
   - `target_stats.py` and `value_calib.py` take minutes;
   - here they showed, before any training, that the visit targets sat a nat from the policy and that the value was
     6 points optimistic.
2. **Compare recipes on one shared batch of games.** Ten arms trained in minutes each on the same 1,411 games, so the
   comparison isolated the recipe from the luck of the games. On gorge, where games are ~100x cheaper, the shared
   batch can be large enough to settle the value questions this study couldn't.
3. **Use the Q-tilted target, with care.** docs/027's completed-Q target sharpened the visits but didn't help the
   policy alone; it was applied to networks trained from scratch. The tilt here is a small, controlled step from a
   policy that's already good (s = 10-30, KL 0.004-0.014 per step). It applies to gorge once a gorge network starts
   from a strong prior, for example a port of this GNN.
4. **Separate the policy's contribution from the value's:**
   - add a value-only arm;
   - match each policy alone, and with search.

   Here that split showed the value carrying the gain with search, which offline scores didn't predict.
5. **λ = 0.99, mostly new games each generation, at most ~2-4 passes over any game.** Then judge over at least two
   generations: generation 2 gained more than generation 1 here.

### Putting these networks on SpellBench

**Same process and architecture as the imitation network.**
- Every network here is the same GNN, with the same vocabulary and graph encoder, fine-tuned from the imitation
  checkpoint.
- So whatever SpellBench builds to run the imitation GNN (`danbrooks/draftzero-fdn-gnn`) runs these too:
  - the graph StateEncoder on its reconstructed worlds;
  - NetGraph inference;
  - the search.
- A self-play network is a weight swap: re-export it with the same release tool, regenerate its goldens, and publish
  it as a separate entry.

**What does differ is how they should be run:**
1. **With search.** Generation 2's clearest gain was at 100 simulations (61.6% against the start), and with search the
   gain came mostly from its value. The value is now calibrated on self-play, so no temperature fix is needed.
   - SpellBench's entry should use PIMC with our settings (100+ simulations, c_puct 1, prior temperature 1.5, prior
     bonus 0.1, the policy as priors and the value at leaves).
   - With different search settings, the strength we measured won't transfer.
2. **The right network.**
   - Generation 2 (`sp028/loops/recipe/g2`) is the strongest;
   - `cq10` (generation 1) is the safest step from the imitation network, as human-like as it by every measure.
   - Both are in the private repo; publishing one means the same public release as the imitation GNN.
3. **One more head is now trained.** The yes/no head ("may" choices) never trained on human data. Self-play trained
   it, so a SpellBench backend should route those decisions to it, not around it.
