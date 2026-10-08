# From imitation to self-play: what the literature suggests, and a plan of runs

*Status (Wednesday 7 October 2026, 7 PM PT): plan written; the shared self-play games for stage 1 are being played on
r1. Results will go in docs/029. The code is on the
[`selfplay-transition`](https://github.com/danieljbrooks/draft-zero/tree/selfplay-transition) branch.*

Our best network imitates top 17lands players: the graph network (GNN) from docs/022-024. This doc plans how to
start improving it by **self-play**, where the network plays itself with search and learns from those games, without
losing what it learned from people. The goal is a **recipe for the handoff from imitation to self-play** that we can
carry to the faster engine we are moving to. The budget is about $15 of rented GPUs plus r1, the machine with two RTX
PRO 6000s that Dama lends us.

A few terms:
- The network has two outputs: a **policy**, how likely it is to pick each legal move, and a **value**, its estimate
  of the chance of winning from the position.
- A **search** at N **simulations** plays N short look-ahead lines before each decision, using the policy to choose
  which moves to try and the value to score where the lines end. Ours is **PIMC**: it deals the hidden cards once into
  one world consistent with what the player has seen, then searches that world.
- In **expert iteration** (AlphaZero's training), the network learns to predict what its own search chose, and the
  value learns from how the games ended. A better network makes a better search, which makes better training data.
- **Paired games** play each pair of decks twice with the seats swapped, which removes most of the luck of the decks.

## Summary

**What can go wrong at the handoff.** The literature and our own history agree on four failure modes:

1. **The policy drifts and forgets.** Every source that started reinforcement learning from an imitation policy and
   then let go of it saw a dip:
   - In Diplomacy, a human-initialised agent fell from ~61% to ~46% agreement with human moves and lost strength
     against fixed bots.
   - In NetHack, plain fine-tuning fell from a ~5,200 score to 647.
   - Sources that kept the imitation policy in the loop did better. AlphaStar kept a KL penalty toward its
     supervised policy for its whole run: +84 Elo for the human start alone, +464 more with the penalty.
2. **The value doesn't fit the new games.** A value head calibrated on human games need not be calibrated on
   self-play positions (ours isn't, §3). Value heads also memorise results when one game's outcome labels many
   positions: AlphaGo's value network had a squared error of 0.19 on its training games and 0.37 on others, until it
   used one position per game. Ours overfit too (docs/024).
3. **There are too few games.** We can afford a few thousand searched games. AlphaGo used 30 million for its value
   network.
4. **Plasticity is lost.** Our one previous imitation start stopped learning in self-play (experiment #2b,
   docs/013).

**What our first 27 self-play games show** (the start network against itself at 100 simulations, §3):
- **The start's value is optimistic on its own games.** It predicts a 61% chance of winning on average, where the
  player to move won 56% of the time. It learned from top players' seats, and they won ~60% of their games.
- **The search's visit counts make a noisy policy target.**
  - Exploration noise and the search's own settings spread the visits onto moves the network rejects. Visit shares
    sit 1.0 nats of KL away from the network's policy, while the network's own entropy is only 0.25 nats.
  - Yet the search changes the top move in only 11% of decisions.
  - Tilting the network's own policy by the search's move values keeps most of those changes (9.1% at scale 30) at a
    seventh of the KL (0.14). This is the kind of target Gumbel AlphaZero uses.
- **Games are cheaper than we estimated:** a median 5.3 minutes of one r1 core at 100 simulations, ~160 games an hour
  on r1 with 14 workers.

**The plan:**
- **Stage 1: one step from the start, many recipes, the same games.** The start plays 1,500 games against itself, now
  running on r1. About ten training recipes ("arms") each train one new network on those games. That isolates the
  recipe from the luck of the games, and costs only training.
  - Stage 1a picks the policy target: raw visits, visits with a KL anchor, or the network's policy tilted by the
    search's values (two strengths).
  - Stage 1b varies the value and the anchor on top of the winner:
    - the value target, λ = 1.0 / 0.99 / 0.95;
    - training the value first;
    - training only the value;
    - mixing in human decisions;
    - a higher learning rate.
- **Stage 2: does it compound?** Two loops of two more generations each, 500 new games a generation: the plain
  AlphaZero recipe against the best stage-1 recipe.
- **Judging:**
  - **every arm:** free offline checks, then its policy alone against the start's (1,000 paired games, cheap);
  - **the finalists and each loop's last network:** paired games at 100 simulations against the start, and the
    ladder against the heuristic bot.
- **A smooth transition means:**
  - each new network beats the start, with and without search;
  - agreement with top players falls only a little;
  - the value's calibration on self-play games improves;
  - each loop generation keeps improving on the last.

**For reviewers:**
1. Is "one shared batch, many recipes, then two short loops" the right shape for this budget?
2. Should the anchor to the imitation network stay for the whole run (AlphaStar, Diplodocus), fade out
   (kickstarting), or follow the previous generation (DeepNash's regularised Nash dynamics, which converge in
   two-player imperfect-information games where plain self-play cycles)? Stage 2's loop keeps it on the start network
   unless stage 1 says otherwise.

Settled already:
- **Dan:** keep the TD discount λ at 0.95 or above; self-play trained fine at 0.99.
- **Will:** in XMage his best setup used an exponential moving average of the current and all future root values,
  and the rate (MageZero's `td_discount`, our λ) was the single most influential setting. So λ is an axis of stage 1b.

## 1. Where we start

| | The start network (`gnn/full_r1/best_policy_calibrated`, docs/024) |
|---|---|
| Network | MageZero's graph network: 13.4M parameters, a graph of ~224 nodes per position |
| Imitation | 10.9M decisions of top players (17lands, FDN), 22 epochs; test NLL 0.203, the player's move first in 85.9% of non-Pass decisions |
| Value | overfit after epoch 7; a one-number temperature (T = 1.61) fixed its overconfidence on human games |
| Against the heuristic bot at 100 simulations | 44.7% with its policy alone, 62.5% at 100 simulations, 69.2% at 300, 69.3% at 1,000 (~100 paired games each) |
| Known weaknesses | timing spells (combat tricks, counterspells), board stalls (13% of its policy-only self-play hits the 50-turn cap) |

## 2. What the literature suggests

Five research agents read the AlphaGo line (AlphaGo, AlphaGo Zero, AlphaZero, expert iteration, MuZero, Gumbel),
KataGo and Leela, human-regularised self-play (AlphaStar, piKL, Diplodocus), imperfect-information card games
(Suphx, DouZero, Skat, Stratego), and the machine-learning literature on fine-tuning from offline data. A second set
of agents checked every claim against the papers and corrected the numbers below. The tricks, ranked by fit to our
setting:

| Problem | Trick | What the sources did and found | In this plan |
|---|---|---|---|
| Policy drift | **Keep the imitation policy in the loss:** a KL penalty toward it | AlphaStar kept KL(supervised ‖ current) for its whole 44-day run: Elo 936 supervised, 1,020 with the human start alone, 1,400 with the KL (Fig. 3E). Diplodocus anchored its search and training to the human policy throughout. | arm `kl` |
| Forgetting | **Rehearse human data** in every batch | Wolczyk et al. 2024 (NetHack): plain fine-tuning fell from ~5.2K to 647; with behavioural-cloning rehearsal 7,610; with kickstarting 10,588. InstructGPT's mix of pretraining gradients removed benchmark regressions where a 100x larger KL did not. | arm `human` |
| Value at the handoff | **Train the value first, keep the imitation policy** | DORA (Bakhtin et al. 2021): a fixed human policy with a value learned in the loop scored 45.6% against DipNet, against 30.6% when the policy also trained. VAPO (Yue et al. 2025): fitting the value under the fixed starting policy before RL took a math benchmark from 11 to 60. LP-FT (Kumar et al. 2022): fit the new head on frozen features first, then fine-tune everything. PPG (Cobbe et al. 2021): a value phase with a KL that holds the policy still. AlphaGo kept its supervised policy as the search prior. | arms `vwarm`, `vonly` |
| Value overfitting | **Count value data in games**, not positions | AlphaGo's value net: 0.19 / 0.37 squared error on training / held-out positions from whole human games; 0.226 / 0.234 with one position from each of 30M games. AlphaGo Zero weighted its value loss 0.01 on human data. Expert iteration: a value needs >10^5 independent samples. | a cap of 16 positions a game; curves per arm |
| Value target | **TD or root-value targets** instead of the bare result | MageZero (Will): the moving average of future root values, λ the key setting. docs/025: the root value gave the policy +11.5 points over the result. KataGo keeps the result on its main head and adds short-term value heads. | arms `lam1`, `lam95` (and λ 0.99 everywhere else) |
| Noisy policy targets | **Q-based targets with a trust dial** | Gumbel AlphaZero (Danihelka et al. 2022): the network's logits plus the search's completed values; learns Go at 2 simulations where visit targets fail at ≤16. Grill et al. 2020: regularised targets help at 5 simulations, little at 50. Hamrick et al. 2021: values of rarely tried moves are the least reliable, so we discount each value by its visits. | arms `cq10`, `cq30` |
| Noisy policy targets | **Keep exploration out of the target** | KataGo prunes forced exploration visits from its policy targets: 1.25x faster learning. | measured offline (§3) |
| Few games | **Playout cap randomisation**: a full search on 25% of moves, a fast one on the rest | KataGo: equal strength at 1/1.37 of the compute; more games for the value. | not built: for the new engine |
| Plasticity | **Low learning rate with warm-up; shrink and perturb at most once** | Diplodocus trained RL at 1/20 of its imitation rate with 10k warm-up steps; AlphaStar's RL rate was 3e-5. Shrink and perturb once restores trainability (Ash and Adams 2020), repeatedly it hurts (Lyle et al. 2023). | arm `lr1e4` |
| Beating only itself | **Play some games against past versions** | AlphaStar: pure self-play's worst win rate against past versions was 46%, against 69-71% with a mix of opponents. OpenAI Five played 20% of its games against past versions. | every network is judged against the start |
| How far from human | **Average the weights** of the start and a trained network | WiSE-FT (Wortsman et al. 2022): interpolating pretrained and fine-tuned weights gained 4-6 points under distribution shift over either end. A free dial, no retraining. | `mid`: halfway to the 1a winner |

**Ruled out for this budget:**
- Gating each new network with searched games: 400 games a gate is about our whole budget.
- Plain policy gradient: it failed from an imitation start in Diplomacy and lost to expert iteration in Hex.
- A league with exploiters.
- Reanalyse: it needs engine state restored from stored games.
- Perfect-information "oracle" guiding.

**Where the sources disagree:**
- **The anchor.** AlphaStar and Diplodocus keep it for the whole run. Kickstarting fades it fast: a constant weight
  plateaued at the teacher's score (41.2), one faded to 0 reached 59.4.
- **The value target.** MuZero Unplugged favours real results; CICERO and piKL bootstrap from a value model because
  results are noisy.
- **Continuing or restarting.** AlphaZero trains one network continuously; expert iteration restarted from scratch
  each round.

## 3. What the first self-play games show

Calibration run: 28 games of the start network against itself on the training decks, 100 simulations a decision,
14 workers on r1. Self-play exploration was on:
- Dirichlet noise on the root's move priors (weight 0.25, α 0.3);
- moves sampled in proportion to visits for each player's first 3 turns.

**Speed.**
- A game lasted a median 5.3 minutes of one worker (18 turns, ~143 decisions, ~80 of them searched).
- No engine errors and no 50-turn stops.
- The 28 games took 16 minutes of wall time, ~105 games an hour including the slowest. Steadily it is ~160 an hour
  with 14 workers; the 1,500-game batch runs 20.

**The value.**

| Seat-decisions of the 27 finished games | |
|---|---|
| Share won by the player to move (each seat's decisions count) | 55.7% |
| The network's mean predicted chance of winning | 61.0% |
| The search's mean root value, as a chance | 59.1% |
| AUC of the network's value (ranking wins above losses) | 0.833 |
| AUC of the search's root value | 0.839 |

The value ranks positions well (0.833) but is about 5 points optimistic. Our explanation: 17lands records top
players' seats, who win ~60% of their games, so "the player to move usually wins" was true in its training data. In
self-play it is false by construction. The search's root value is a little better on both counts. A temperature
fitted on one distribution can't be expected to hold on another (Ovadia et al. 2019), so the value's calibration is
measured on self-play games from here on.

**The policy targets.**

![Policy targets from 27 self-play games at 100 simulations: x is KL(target ‖ the network's policy) in nats, y is the share of decisions where the target's top move differs from the network's. Visit shares sit at 1.0 nats and 11.1%; sharpening them to visits^(1/τ) lowers the KL to 0.49 (τ 0.5) and 0.34 (τ 0.25) without changing the 11.1%. The network's policy tilted by the search's values rises from the origin: 0.004 nats and 1.8% at s = 3, 0.03 and 4.5% at s = 10, 0.14 and 9.1% at s = 30, 0.51 and 17.7% at s = 100.](img/028-targets-light.png)

*Figure 1. How far each candidate policy target sits from the network that played the games
(`tools/selfplay_transition/target_stats.py`, `fig_doc028.py`).*

- **The network is sharp:** its policy has 0.25 nats of entropy; 60% of the decisions are yes/no, with 2.7 options
  on average.
- **The raw visit shares are flat and noisy,** at 0.59 nats of entropy and 1.0 nats of KL. A typical decision:
  visits [1, 3, 2, 6, 88] where the network's policy gives the first four options <1% each. Those visits come from:
  - exploration noise;
  - a prior temperature of 1.5;
  - a bonus of 0.1 that MageZero's search adds to every option except Pass.

  Training on them would teach the network to spread out, not to play better.
- **Sharpening the visits doesn't remove the noise:** the KL stays at 0.34-0.49 nats, because a single visit on an
  option the network rejects still costs a lot.
- **The tilted target** is π'(a) ∝ π(a)·exp(s·w(a)·(Q(a) − Q_root)):
  - the network's own policy, raised where the search valued a move above the position and lowered where below;
  - w(a) = n/(n + 2) discounts a value backed by few visits n;
  - an option the search never tried keeps its policy weight.

  At s = 30 it changes the top move in 9.1% of decisions (raw visits: 11.1%) for 0.14 nats. The scale s is an
  explicit dial for how far each step may move from the network.

Both findings feed the plan: the arms test these targets, and the value arms test how the value should adapt.

## 4. The plan

### 4.1 Stage 0: the shared games (running)

- **The batch:** the start network plays 750 deck pairs, so 1,500 games against itself, on the training decks.
  Settings: 100 simulations, PIMC with guessed opponent decks, the exploration above, a 50-turn cap. It started at
  6:30 PM PT on r1 (20 workers, a 14-hour hard kill).
- **Held out:** 10% of the deck pairs, chosen by hash, the same for every arm. These games are used only for the
  offline checks.
- **What each game records:** for every searched decision, the position's graph, each option's visits, value and
  prior, the root value, and the move played.

### 4.2 Stage 1: one step from the start

Every arm starts from the start network and trains on the same games. **Base recipe:**
- 2 passes over the training decisions, batch 256;
- AdamW at 3e-5: 100 warm-up steps, then a cosine to a tenth;
- the value on at most 16 positions a game, weight 0.5, target TD λ = 0.99 (MageZero's labels);
- no anchor, no human rows.

**Stage 1a: the policy target.**

| Arm | Policy target | The question |
|---|---|---|
| `visits` | the search's visit shares | the plain AlphaZero recipe (control) |
| `kl` | visit shares, plus 1.0 × KL(start ‖ network) | does docs/021's anchor (in effect, half visits, half the start's policy) stop the drift? |
| `cq10` | the network's policy tilted by the search's values, s = 10 | a gentle trust-region step |
| `cq30` | the same, s = 30 | a larger step |

**Stage 1b: the value and the anchor,** each on top of the 1a winner.

| Arm | Change | The question |
|---|---|---|
| `lam1` | λ = 1.0: the game result alone | Will's question: does the TD discount matter here? |
| `lam95` | λ = 0.95: mostly the next ~20 root values | the same |
| `vwarm` | the first quarter of the steps trains only the value head, on the results; then everything as usual | does recalibrating the value before the policy moves avoid a dip? |
| `vonly` | no policy loss; a strong KL to the start (10) pins the policy; the value trains through the whole network | does a better value alone, under the imitation policy, beat moving both? |
| `human` | 25% of every batch from the human training decisions (their policy loss only) | does rehearsal keep agreement with people without costing strength? |
| `lr1e4` | learning rate 1e-4 (the imitation run's) | is 3e-5 too timid? |
| `mid` | no training: the weights halfway between the start and the 1a winner | is half a step better than a whole one? (policy alone only) |

**How each arm is judged:**

1. **Offline, free.** These are checked every 400 steps during training, because a dip right after the switch can be
   transient and is missed by checking only at the end (the "stability gap", De Lange et al. 2023).
   - **On the held-out self-play games:**
     - the policy's agreement with the search;
     - its KL from the start and its entropy;
     - the value's log-loss, calibration error and AUC against the results.
   - **On the 17lands validation decisions:**
     - set NLL and top-1 (non-Pass);
     - how often Pass is the top choice;
     - the value's log-loss.
2. **Policy alone against the start's policy alone.** 500 deck pairs (1,000 games) on the evaluation decks, both
   greedy (each always plays its most likely move). This is cheap: no search, ~minutes per 100 games on r1. At
   1,000 games, about 3 points is within chance.
3. **At 100 simulations against the start at 100.** 150 deck pairs (300 games), for:
   - the control;
   - the 1a winner;
   - `vonly`;
   - the 1b winner.

   About 6 points is within chance at 300 games.
4. **Against the heuristic bot at 100,** on docs/024's deck pairs, for the final networks only. This is the outside
   reference: the start won 62.5% of these games at 100 simulations.

**Decision rules, set before the results:**
- **1a winner:** the best policy-alone score against the start.
  - It is disqualified if its 17lands set NLL rose by more than 0.02, or its Pass rate moved by more than 5 points.
  - If two arms are within 2 points of each other, the one closer to the start (lower KL) wins.
- **1b recipe:** a value arm joins the recipe only if it wins at 100 simulations, or ties there and wins offline on
  the value's log-loss and calibration.

### 4.3 Stage 2: does it compound?

Two loops, each continuing from its stage-1 network for two more generations:
- **plain:** the `visits` recipe, the plain AlphaZero handoff;
- **recipe:** the best stage-1 settings.

Each generation:
- **Data:** 500 new games of the newest network against itself, with stage 0's settings.
- **Training:** from the previous generation's weights, on its own games plus the previous generation's (a window of
  two). Each position is used at most about 4 times overall, KataGo's cap.
- **Anchor:** any KL anchor or human rows stay tied to the start network, not to the previous generation.
- **Checks:**
  - every generation: the offline checks, and its policy alone against the start's and the previous generation's
    (1,000 games each);
  - the final network of each loop: 300 paired games at 100 simulations against the start, and the heuristic ladder
    at 100.

### 4.4 Compute and order

| Step | Where | Games | Time |
|---|---|---|---|
| Stage 0 | r1 | 1,500 at 100 simulations | ~10 hours (tonight) |
| Stage 1 training | r1 GPU | – | ~10-20 minutes an arm |
| Policy-alone evaluations | r1 | ~1,000 an arm | ~20-40 minutes an arm |
| 100-simulation evaluations | RunPod Community pods (~$0.20/h), some on r1 | ~300 an arm, ~8 arms | ~$10 |
| Stage 2 | r1 | 4 × 500 at 100 simulations | ~14 hours |

Stage 1's policy-alone results come first. They decide which arms earn 100-simulation games.

## 5. What would show a smooth transition

| Signal | A smooth handoff | A dip, or collapse |
|---|---|---|
| Policy alone against the start | ≥ 50% from the first network on | < 50%: the policy got worse before it got better |
| 100 simulations against the start | ≥ 50% and rising by generation | < 50% while the policy alone is ≥ 50%: the value or the search's use of it got worse |
| 17lands set NLL, non-Pass top-1 | rises a little (search disagrees with people where it finds better moves) | rises sharply, or Pass becomes the top choice far more or less often |
| Value on held-out self-play games | calibration error and log-loss fall | log-loss rises: memorising results |
| Generation against the previous | ≥ 50% each step | falls back: forgetting, or beating only itself |
| Against the heuristic bot | at least the start's 62.5% at 100 | below the start's |

## 6. Risks

- **Effects smaller than the noise:**
  - Stage 1's arms may all land within 3 points of each other, policy alone.
  - In that case the offline checks rank them, and the doc says so plainly.
- **The value can't move:** 1,350 training games are far fewer than any source's value data. If no value arm beats
  the start at 100 simulations, that is itself a finding for the new engine, where games are ~100x cheaper.
- **r1 is shared and preemptible:**
  - Games are written as they finish, so a stopped run resumes.
  - Every job has a hard time limit.
  - A memory guard kills our newest process before the container could run out of memory.
- **The evaluation decks overlap the imitation data.** 89% of their drafts are in the imitation training set
  (docs/019). That is fine for comparing arms; it flatters agreement with people.

## 7. The code

On the [`selfplay-transition`](https://github.com/danieljbrooks/draft-zero/tree/selfplay-transition) branch:

- **`src/draftzero/selfplay/gnn_train.py`:** new options for the self-play trainer, all off by default, so the loop's
  recipe is unchanged:
  - `policy_weight`;
  - `policy_target: visits | cq` with `cq_scale`, and `visit_temp`;
  - `train_only` (freeze the rest);
  - `value_head_init: keep | reset | shrink`;
  - `kl_mask_use`: no KL on yes/no rows, whose head the imitation never trained;
  - `human_value_weight`;
  - `human_data_cache`: human rows read from the imitation run's memory-mapped cache;
  - `cq_visit_shrink`: the tilted target's discount of values backed by few visits;
  - `seed`;
  - a held-out check that adds entropy and calibration error.
- **`src/draftzero/selfplay/tables.py`:** keeps each decision's root value (`q_root`) for re-labelling at another λ
  and for the tilted targets.
- **`tools/selfplay_transition/`:**
  - `sp_train.py` trains one arm on fixed games (phases, per-phase λ, checks every N steps, 17lands validation);
  - `target_stats.py` produces figure 1's measurements;
  - `paired.py` scores a match by deck pair, counting only pairs with both games free of engine errors;
  - `fig_doc028.py` draws figure 1;
  - `interp.py` averages two networks' weights.
- **`deploy/sp_h2h.sh`:** a match between two networks. Each network gets its own servers, started and killed by
  this script, so a match can't silently reuse another network's server.

## References

- Silver et al. 2016, *Mastering the game of Go with deep neural networks and tree search*, Nature 529:484.
- Silver et al. 2017, *Mastering the game of Go without human knowledge*, Nature 550:354; Silver et al. 2018,
  AlphaZero, Science 362:1140.
- Anthony, Tian and Barber 2017, *Thinking Fast and Slow with Deep Learning and Tree Search*, arXiv:1705.08439.
- Wu 2019, *Accelerating Self-Play Learning in Go* (KataGo), arXiv:1902.10565; KataGo's `docs/KataGoMethods.md`.
- Danihelka, Guez, Schrittwieser and Silver 2022, *Policy improvement by planning with Gumbel*, ICLR.
- Grill et al. 2020, *Monte-Carlo Tree Search as Regularized Policy Optimization*, ICML, arXiv:2007.12509.
- Schrittwieser et al. 2021, *Online and Offline Reinforcement Learning by Planning with a Learned Model* (MuZero
  Unplugged), arXiv:2104.06294.
- Vinyals et al. 2019, *Grandmaster level in StarCraft II using multi-agent reinforcement learning*, Nature 575:350.
- Jacob et al. 2022, *Modeling Strong and Human-Like Gameplay with KL-Regularized Search* (piKL), arXiv:2112.07544.
- Bakhtin et al. 2021, *No-Press Diplomacy from Scratch* (DORA), arXiv:2110.02924.
- Bakhtin et al. 2023, *Mastering No-Press Diplomacy via Human-Regularized RL and Planning* (Diplodocus),
  arXiv:2210.05492.
- Schmitt et al. 2018, *Kickstarting Deep Reinforcement Learning*, arXiv:1803.03835.
- Yue et al. 2025, *VAPO*, arXiv:2504.05118; Kumar et al. 2022, *Fine-Tuning can Distort Pretrained Features*
  (LP-FT), arXiv:2202.10054; Cobbe et al. 2021, *Phasic Policy Gradient*, arXiv:2009.04416.
- Hamrick et al. 2021, *On the role of planning in model-based deep reinforcement learning*, arXiv:2011.04021.
- Wortsman et al. 2022, *Robust fine-tuning of zero-shot models* (WiSE-FT), arXiv:2109.01903.
- Ovadia et al. 2019, *Can You Trust Your Model's Uncertainty?*, arXiv:1906.02530.
- De Lange et al. 2023, *Continual evaluation for lifelong learning: identifying the stability gap*, ICLR.
- Perolat et al. 2022, *Mastering the game of Stratego with model-free multiagent reinforcement learning*
  (DeepNash), Science 378:990.
- Wołczyk et al. 2024, *Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem*,
  ICML, arXiv:2402.02868.
- Ouyang et al. 2022, *Training language models to follow instructions with human feedback*, arXiv:2203.02155.
- Uchendu et al. 2023, *Jump-Start Reinforcement Learning*, arXiv:2204.02372.
- Ash and Adams 2020, *On Warm-Starting Neural Network Training*, arXiv:1910.08475; Lyle et al. 2023,
  *Understanding Plasticity in Neural Networks*, arXiv:2303.01486.
- Berner et al. 2019, *Dota 2 with Large Scale Deep Reinforcement Learning* (OpenAI Five), arXiv:1912.06680.
- Our own: docs/013 (an imitation start that lost plasticity), docs/021 (the self-play loop), docs/024 (the start
  network), docs/025-027 (self-play on mtg-kernel and gorge).
