# Experiment #4: Scaling up imitation learning

*September–October 2026. The plan, and the results as the stages run (§8). It follows experiment
#3 ([docs/016](016-search-benchmark-results.md)) and builds on its §9, "A potential plan for
bootstrapping limited agents with imitation learning".*

## The question

Experiment #3 found that a network trained to imitate human decisions agreed with top players as
well as the best search we can afford, with no search at all and for about a thousandth of the
compute. That network learned from 1.7% of the 17lands games we have, and from only some kinds of
decision.

This experiment asks how far supervised learning from top players' games goes:

- **the policy by imitation learning**, on as many of the top players' decisions as we can
  rebuild, including timing: the second main phase, the end of turn, and answers to a spell on the
  stack;
- **the value by behaviour cloning**, learned from the same games' results, benchmarked against
  the hand-written heuristic it would replace (§4);
- **every evaluation mixes the two halves:** the imitation policy or the default policy as the
  search's prior, and the cloned value or the heuristic at its leaves;
- the deciding test is play: the imitation-plus-cloned-value bot against the heuristic bot, at a
  search budget of 1,000;
- then two ways to keep training from that start without losing the human policy.

Every search uses IS-MCTS. It runs on RunPod over a few days. The initial budget was $30–40, and
it can grow as needed (§6).

## Short answers

| Question | Answer | Details |
|---|---|---|
| Which data? | **Top players only:** the 60%-and-up win-rate bucket, 21% of the file, about 166k games. 90% of them, about 140k games, train; that's 10× #2b's. The rest of the file is a reserve, most useful for the value head. | §2.1 |
| All decisions? | **Most of them:** the plays at turn start and later in the turn, attacks, blocks, spell targets where the outcome settles them, and the timing decisions: second main phase, end of turn, and answers to the stack. About 7M decisions, 50× #2b's. 17lands pins timing to the turn, not the step, so timing labels come in two grades. | §2.2 |
| Can a fair search (IS-MCTS or PIMC) use an imitated policy *and* value? | **The policy, yes.** **The value, only indirectly.** Humans don't report values, only results. We can learn *the value of human play* from game results. | §3 |
| Can behaviour cloning bootstrap a value head better than the heuristic? | **Better at predicting results, not yet better in search.** In a laptop pilot on 16,067 held-out human positions, #2b's head predicts who wins with AUC 0.701, against 0.661 for the heuristic (paired +0.041, 95% CI +0.028 to +0.055). Inside the search, and in play, it was no better. | §4 |
| Are multiple epochs viable? | **Yes.** Plan for several, with checkpoints, and stop when validation plateaus. Stop the value head separately, because it overfits first. | §5 |
| Which GPU? | **Measure first.** Experiment #3's network searches kept the RTX 3090 96–99% busy, but at only about a tenth of its peak arithmetic. A faster inference server may matter as much as the card. The L40S is the main alternative. | §6.7 |

## 1. Why imitation, at our compute budget

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/017-games-dark.png">
  <img alt="Horizontal bars on a log scale of games: DraftZero #2a self-play, 2.7 thousand games; #2b's human games, 13.5 thousand; experiment #4's training split of top players' games, about 140 thousand; all 17lands FDN Premier Draft games, 791 thousand; for scale, AlphaGo's human games from KGS, 160 thousand, and AlphaZero's chess self-play, 44 million." src="img/017-games-light.png">
</picture>

*How many games each agent learned from. Self-play games cost us compute; the 17lands games are
free.*

- **Self-play needs far more games than we can play.**
  - Experiment #2a played 2,673 games over 18 generations in 43.6 pod-hours
    ([docs/014](014-exp2-run1-report.md)). Its network then scored positions no better than the
    hand-written heuristic ([docs/016](016-search-benchmark-results.md) §4.3).
  - AlphaZero's chess run played 44 million games.
  - At #2a's rate, the ~140k top players' games we'll train on would take about 2,300 pod-hours of
    self-play, or ~$1,100. And those games would be played by a far weaker agent than the people
    in the 17lands file.
- **Search is expensive and short-sighted.**
  - The best fair searches in experiment #3 reach 0.636 on the balanced score (IS-MCTS, 10,000
    simulations, 14.8 pod-seconds per decision) and 0.631 (PIMC with 1 world, 6.5 pod-seconds).
  - Even at 3,000 simulations they look less than one turn ahead. They act far more often than top
    players, because they can't see the value of waiting (docs/016 §8).
- **A little imitation already matches the best search.**
  - Experiment #2b's network was trained for 40 minutes on one RTX 3090 (~$0.33), on 13,533 games.
  - Its policy alone scores **0.646**, for 0.014 pod-seconds per decision. That's about a
    thousandth of IS-MCTS's compute, and a five-hundredth of PIMC's (docs/016 §7).
  - It has the habit the searches lack: knowing when to do nothing.
- **The data is there, and it maps onto the engine.** The 17lands mapping exercise
  ([docs/008](008-gameplay-data.md)) found:
  - 791,159 FDN Premier Draft games, licensed CC BY 4.0;
  - every card, ability, cast and land play maps to XMage and MageZero;
  - every visible card is pinned down in 88.7% of turn-start states;
  - 88% of whole turns replay in XMage to the same next state.
- **How the human knowledge reaches the agent matters** ([docs/013](013-exp2-imitation-report.md),
  docs/016 §7).
  - #2b warm-started self-play with the policy priors off, so the search never read the human
    policy.
  - Heavy pretraining (17 passes over the data) cost the network its ability to keep learning.
    Self-play then overwrote the human policy in one epoch.
  - Experiment #3 tried the human policy as the search's prior. The search kept the policy's
    quality at every budget, but didn't improve on it. What search can add is limited by the value
    at the leaves, and #2b's value head, trained on human results, was no better than #2a's.
- **The closest precedent is AlphaGo, not AlphaZero.** AlphaGo learned its policy from 160k human
  games before any self-play. At our budget we're in that regime.

So experiment #4 scales up the two things #2b had too little of: data, and the parts of the
network that the search reads.

## 2. The data: top players' games, as many decisions as we can rebuild

### 2.1 Which games

- **Top players:** `user_game_win_rate_bucket` of 0.60 or more. In a 1-in-20 sample of the file
  (39,558 games), that's 21.0%, which scales to about 166k games and 1.43M turn-start decisions.
- **More, in a pinch:** the other 79% of the file (players under 60%). Their play may be weaker, so
  they'd feed the value head before the policy, since a result means the same whoever played.
- **Selection on the result:** the bucket is computed over games that include the game itself
  (docs/008 §3.5). So it favours wins, most for players with few games.

  | Players in the top bucket | Share of its games | Their win rate in those games |
  |---|---|---|
  | 100+ games in the file | 55% | 63.9% |
  | 50–99 games | 25% | 62.6% |
  | fewer than 50 games | 20% | 68.2% |
  | *whole file, for reference* | | *54.7%* |

  The policy doesn't care much. The value head learns from these results, so it gets checked by
  group (its calibration on each row). If the under-50 group skews the head, drop that group from
  the value loss.
- **The split:** 90 / 5 / 5 by a salted hash of the draft, as #2b's tables did. Drafts linked by a
  mirrored game (13.7% of rows are one game recorded by both players) share a split.
  - About 140k training games, 10× #2b's 13,533.
  - Validation and test have about 7.7k games each, about 65k turn-start decisions each.
- **Held out from everything:** every draft of experiment #3's benchmark games (sb-v1) and of
  their mirrored partners: 842 drafts, 6,139 rows (docs/016 §9.1). Holding out whole components
  instead would drop 76k rows, because mirrored games chain drafts into components of up to ~10k
  rows. The new benchmark (§6.5) comes from the new test split.
- **Measured** (`tools/imitation_scale/build.py splits`): 709,859 train, 39,841 validation and
  35,320 test rows in the whole file. The top-player filter applies at build time.

### 2.2 Which decisions

| Decision | Network head | Label | Per game | Training set | Status |
|---|---|---|---|---|---|
| First main-phase priority of the player's turn | player priority | the set of the turn's plays (17lands has no order) | 8.6 | 1.2M | built; #2b trained on 115k |
| Later priority decisions in the player's main phases (turn replay) | player priority | each play, in an imputed order; the Pass once the plays are done is exact | 19 | 2.7M | built for 1.25 turns a game; never trained on |
| **Timing, in the player's own turn:** the second main phase, combat and the end step, with an instant or ability available | player priority | Pass or a play. Holding to the next turn is exact; main phase 1 against 2 is imputed unless the outcome settles it | 18 | 2.7M | **built** (`allStops`) |
| **Timing, in the opponent's turn:** instants, flash and abilities, including answers to a spell on the stack | player priority | the turn of each play is exact; the step is imputed, except a counterspell (it must answer a spell) and a trick (combat deaths show it) | 19 | 2.8M | **built:** the opponent's turn replayed |
| "Attack with X?" (turn replay) | binary | exact | 8.8 | 1.2M | built for 1.25 turns a game; #2b trained on 17k |
| "Which attacker does X block?" | target | the recorded pairing, unique in about 88% | 3.3 | 0.5M | **built:** every blocker from the opponent's-turn replay; the first of each attack also from its end-of-turn state |
| Spell targets (turn replay) | target | exact when what died settles it, about half | 3 | ~0.2M exact | built; never trained on |
| The opponent's plays, from the player's seat | opponent priority | the set of the opponent's recorded plays | 8.6 | 1.2M | **not in the first run:** the search uses the baseline prior at the opponent's nodes (§3.2) |
| Modes, X values | – | rarely recoverable | | | left out |
| Mulligans | – | exact | ~1 | | left out: mulligans are off in MageZero's games |
| Value | value | the game's result (§4) | every row | ~140k independent results | built for turn starts |

- **Where the rates come from.**
  - Turn starts and replay yields are from #2b's build: 8.56 turn starts a game; 87.8% of 20,596
    replayed turns reproduced, yielding 2.56 priority and 1.17 attack decisions each.
  - Blocks are from a 1-in-40 sample of the file (19,779 games). "Up to 8" counts the player's
    creatures in play when the opponent attacked. Humans blocked with 1.6 of them a game, which
    fits the 20% block rate in docs/016.
  - The top players' games have 8.6 turn starts a game too (1-in-20 sample).
- **#2b replayed only 1.25 turns a game.** That is why docs/016 §9.1 counted about 1.3 attacks a
  game. Replaying every turn gives about 7× more.
- **Timing is where good players differ.** Top players hold a spell to the last useful moment
  (docs/016 §8.2), and the searches don't. So timing gets its own rows in training and its own
  items in the benchmark (§6.5). Two grades of label:
  - **Exact:** whether a play happened this turn or a later one. 17lands records each play's turn,
    so "held it through my end step and cast it in the opponent's turn" is exact. So is "cast
    nothing all turn".
  - **Imputed:** which step within the turn. The replay tries main phase 1 or 2, and an instant's
    window is read off the card: a counterspell in response, a trick after blocks, removal after
    attackers, the rest at the end step. When more than one window reproduces the turn, the first
    one tried wins.
  - Train on both, with imputed rows at lower weight, and build the timing items of the benchmark
    from exact labels only.
  - **Keep both sides of each timing question.** In the opponent's turn, a Pass is exact only when
    the player cast nothing that turn; when they did cast something, the stop is imputed. Training
    on the exact passes alone would teach the policy never to act in the opponent's turn. So these
    rows come from replaying the opponent's turn, which labels the casts too, not from the passes
    alone.
- **The replay needs two changes for timing.**
  - Today it records the player's priority decisions in its main phases, and at other stops only
    when it plays something there. It should record every stop with a real choice. A Pass there is
    exact when nothing of the player's is left for the turn.
  - It should also replay the opponent's turns: the opponent's recorded plays, the player's
    recorded instants in their windows, and the player's blocks. That's the same code with the
    seats swapped.
- **The later priority decisions teach when to stop.** Only 5% of turn-start labels are Pass, so a
  model trained on them alone learns to always act. docs/008 found that mixing in replayed rows
  raised accuracy on the exact "stop acting" decisions from 48% to 77%. That's experiment #3's
  hold problem: #2b's policy matched only 34% of holds.
- **In all:** about 6.5M decisions before the new timing rows, 50× #2b's 132,603.
- **Measured on a 20-game smoke build** (stage 0; a small sample):
  - **per game:** 9.1 turn starts, 38 later priority decisions in the player's own turn, 10.6
    attacks, 1.3 exact spell targets, 19 priority decisions in the opponent's turn, and 3.3 blocks
    (every blocker), plus 3 first-block questions from end-of-turn states;
  - **in all:** about 84 table rows a game, so about 12.6M for the training split;
  - **reproduced turns:** 85% of the player's turns and 85% of the opponent's;
  - **timing:** about 270 exact Pass decisions at combat stops, the second main phase and the end
    step, including instants and flash creatures held through the player's own turn.
  - **In the opponent's turns:** counterspells cast in response to a spell, removal after attackers,
    flash creatures and burn at the end step, and passes at every other stop. About 17 plays
    against 350 passes in 20 games.
  - **Label sets:** in the player's main phases, a decision's label is every play the player has
    left this turn (17lands has no order). At other stops it's only the plays due at that stop. So
    an instant cast later isn't labelled at an earlier stop: before this fix, 5% of those passes
    were labelled with a later play.

## 3. What a fair search can take from human games

### 3.1 What the search reads

The search code (`BenchSearch`) encodes every position it evaluates from the **searching player's
seat, with the opponent's hand hidden.** That holds for PIMC and IS-MCTS alike. The hidden cards
the search samples shape its tree, but never enter the network. A 17lands row records exactly that
view: the player's own hand, and the opponent's as a count.

| Head | Read by the search at | Human labels | Trained in #2b? |
|---|---|---|---|
| player priority | the searcher's priority decisions | the player's plays | yes, turn starts only |
| opponent priority | the opponent's priority decisions | the opponent's recorded plays | **no.** In experiment #3's human-prior search, the opponent's priors came from an untrained head. The first run uses uniform priors there |
| binary | yes/no questions, mostly attacks, for both players | attacks, exact | yes, 17k attacks |
| target | blocks and spell targets, for both players | blocks; targets that what died settles | **no.** That's why #2b's blocks were at chance in experiment #3 |
| value | every new leaf | game results | yes, turn starts only |

### 3.2 The policy: yes, for both players

- **The player's heads** are behaviour cloning, as in #2b, extended to every row of §2.2.
- **The opponent's head** would predict what the opponent will do, given what the searcher can see.
  A fair search needs exactly that at the opponent's nodes, and 17lands records the opponent's
  plays from the player's seat. **It isn't in the first run.** The search uses the baseline,
  uniform priors, at the opponent's nodes, instead of the untrained head experiment #3's human-prior
  search read there.
  - The opponent's hand is hidden, so their legal options aren't known. Train the head over the
    whole action vocabulary ("which card will they play?"). The search already renormalises the
    prior over the options legal in each sampled world.
  - It suits IS-MCTS: the opponent's nodes are shared across sampled worlds, and the head's input
    doesn't depend on the world.

### 3.3 The value: learned from results, not imitated

Humans don't output values; the data gives one result per game. Fitting results to positions
learns **the value of human play:** the chance of winning from what the player can see, if both
sides then play like the people in the file.

- For a search whose prior is the human policy, that's a sensible value to want.
- It's a hidden-information value by construction. It averages over the opponent hands that
  actually occurred, so it can't leak the opponent's real hand.

§4 covers how to train it, and how to tell whether it beats the hand-written heuristic.

### 3.4 IS-MCTS for every search

- **Decided: every search in experiment #4 uses IS-MCTS,** in the benchmark and in games. Both
  IS-MCTS and PIMC read the network the same way, so one network serves either.
- **What it costs:** in experiment #3, IS-MCTS cost 1.2–1.25× PIMC with 1 world with the network,
  and 2.8–3.3× offline, because it replays its path in a freshly dealt world every simulation
  (docs/016 §1.4). At 3,000 simulations that's 5.3 pod-seconds a decision with the network and 3.6
  offline.
- **What it bought there:** with the network it agreed with top players no better than PIMC with 1
  world (−4.8 points at 1,000 simulations, −2.5 at 3,000). Offline it tied PIMC, and it was the
  best search at 10,000 simulations. Agreement can't test its main advantage: one tree across
  sampled worlds, with no strategy fusion, so it can value bluffs and playing around tricks.
- **Where its worlds come from:**
  - **In the benchmark:** the item's 8 belief worlds, as in experiment #3. Their decklists come
    from the belief model below, so the benchmark has no decklist leak.
  - **In games:** every simulation re-deals the opponent's hand and both libraries from the cards
    the searcher hasn't seen. Experiment #3's IS-MCTS already does that.
- **Closed decklists in games (decided).** The game player samples the opponent's hidden cards
  from the same belief model as the benchmark, so neither bot knows the opponent's real list. That
  model is `belief.py`'s `OpponentModel` (docs/008 §8.2):
  - **The posterior:** it weighs real 17lands decks by the colours revealed and the cards seen.
    Decks missing a seen card stay possible but lose weight.
  - **Sampling:** a sampled deck is forced to contain every seen card, and the hidden hand is drawn
    from the rest.
- **How the game player uses it:**
  - **Every decision:** it asks a small service (`tools/imitation_scale/belief_server.py`) for 8
    samples, given the opponent's cards in known zones (battlefield, graveyard, face-up exile,
    the stack) and its hand size.
  - **Each world:** the search's starting state with the opponent's hand and library replaced by
    one sample's cards. IS-MCTS re-deals within those worlds every simulation, as in the
    benchmark.
  - **No leak back in:** the service leaves the opponent deck's own draft out of its pool, or the
    posterior would find the real list again. A world that can't replay to the decision is
    dropped, so the real hidden cards never enter a search.
- **Its limits, as built:**
  - "Seen" means the cards in known zones now. A card that left them for a hidden zone (bounced,
    shuffled in) isn't remembered, and a known card in the opponent's hand counts as unknown.
  - The model's optional hand-retention weighting is off.
  - A decision about the opponent's hidden cards themselves (a "look at their hand" effect) has
    different options in every sampled world. If none of the searched options exists in the real
    game, the player plays MageZero's first option (counted as `fallbacks`; once in the first test
    game).
- **Tested:** in a closed-decklist game every decision built its 8 worlds (600 in all), none
  failed, and the game ran at the same speed as an open one.

## 4. Bootstrapping the value head by behaviour cloning

Behaviour cloning gives a policy directly. For the value head, the nearest equivalent is learning
from the human games' results which positions win. The question is whether such a head can replace
the hand-written heuristic as the search's evaluator, and how to tell.

**The heuristic** is offline MageZero's `GameStateEvaluator3`: a hand-tuned score of life, cards
and board, from the player's seat. Experiment #3's offline searches used it at every leaf.

### 4.1 What we already know: a better predictor, not a better evaluator

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/017-value-pilot-dark.png">
  <img alt="Left: AUC predicting who won, by the player's turn number, for three evaluators on 16,067 held-out human turn starts. All three start near 0.55 at turns 1-2 and rise. The behaviour-cloned head (#2b) is highest from turn 3 on, reaching 0.80 at turns 7-9, overall 0.701; the heuristic (0.661) and the self-play head (#2a gen 18, 0.654) track each other around 0.73. Right: inside experiment #3's search (PIMC with 1 world, 1,000 simulations, priors off) the three heads score 0.597, 0.604 and 0.599 balanced, with overlapping 95% intervals." src="img/017-value-pilot-light.png">
</picture>

*Left: a laptop pilot for this doc. Each of #2b's 16,067 clean held-out turn starts (1,882 games)
was rebuilt in XMage and scored by the heuristic. 99.7% of the rebuilt positions encode exactly as
in #2b's table, so all three evaluators see the same states. Right: experiment #3's search
(docs/016 §7).*

Three evaluators, measured three ways:

| | Heuristic | #2a gen 18 (self-play) | #2b (behaviour cloning) |
|---|---|---|---|
| **Offline:** AUC predicting held-out human results | 0.661 (0.644–0.677) | 0.654 (0.636–0.673) | **0.701** (0.687–0.714) |
| Log-loss, each calibrated by a logistic fit | 0.649 | 0.654 | **0.621** |
| AUC at turns 1–2, mostly the deck and who's on the play | 0.562 | 0.542 | 0.562 |
| AUC at turns 7–9 | 0.726 | 0.727 | **0.796** |
| AUC of the change since turns 1–2, at turns 5+ (the deck and the play order cancel) | 0.713 | 0.712 | **0.764** |
| **In search:** experiment #3's balanced score, PIMC with 1 world, 1,000 simulations, priors off | 0.597 | 0.604 | 0.599 |
| **In play:** against raw search with the heuristic, budget 300, about 200 games | (the baseline) | 49.7% at gen 16 (docs/014) | 47.2% at gen 0 (docs/013) |

- **Offline, the behaviour-cloned head is clearly better.** It beats the heuristic by +0.041 AUC
  (paired over games, 95% CI +0.028 to +0.055). The self-play head is level with the heuristic
  (−0.007, CI −0.018 to +0.006).
- **Its edge isn't knowing which decks win.** At turns 1–2, where a position is mostly the deck,
  all three are at 0.54–0.56. The edge grows with the turn. It also holds for the change in value
  since the first turns, where the deck cancels.
- **It isn't confined to the positions it trained on.** On 5,443 replayed mid-turn positions it
  never saw in training, it still beats the self-play head (AUC 0.742 against 0.712). The heuristic
  wasn't scored there.
- **Yet inside the search, and in play, it's no better than the heuristic.** Predicting who wins is
  necessary for a leaf evaluator, but not enough.
- **Others found the same.**
  - In Diplomacy, Diplodocus found that search over a human-imitation policy and value gained
    nothing over plain search; the gain came only once RL had improved the value (Bakhtin et al.,
    2023, appendix H.2).
  - AlphaGo Zero's value net trained on human games predicted professional results worse than its
    self-play value net (MSE 0.207 against 0.177; Silver et al., 2017).

### 4.2 Why a better predictor needn't search better

- **Search compares siblings; AUC compares games.** The search ranks positions one action apart in
  the same game. AUC ranks positions from different games, where one side may be three cards and
  eight life ahead. Experiment #3's disagreements with humans were mostly near-ties, a median gap of
  0.043 in the search's own values (docs/016 §8.4). A head can order games well and still barely
  separate "attack" from "hold back".
- **Results teach that local difference only weakly.** Two siblings share their game's result. The
  network can learn their difference only from similar positions in other games that went
  differently. More games, and targets carrying more than one bit a game, address this directly.
- **Leaves fall where #2b's head never looked.** The search's leaves are mid-turn, in combat and in
  the opponent's turn. #2b's head never saw the opponent's turn. The heuristic and the self-play
  head have seen it.
- **Search noise.** At 1,000 simulations the methods in experiment #3 differed by 1–5 points, close
  to the benchmark's resolution. A modestly better evaluator could hide in that.

So the benchmark has to judge a value head inside the search and in play, not only by AUC (§4.4).
The training has to aim at local discrimination: more results, all position types, and denser
targets (§4.3).

### 4.3 Techniques

What the literature and the pilot suggest, and what goes into the experiment. A literature survey
for this doc checked each source against the paper itself.

| Technique | What it changes | Evidence | In the experiment |
|---|---|---|---|
| **Many more results** | 700k independent results, not 13.5k | AlphaGo's value net, trained on human games, memorised them: MSE 0.19 on training games, 0.37 on test. ALLIE's value, trained only on the results of 91M human chess games, lifted MCTS from a 2136 to a 2318–2528 rating | all arms; the learning curve measures it |
| **A few positions per game in each epoch** | stops the head memorising each game's result | AlphaGo kept one position per game; Maia sampled 1 in 32; AlphaGo Zero weighted the value loss 0.01 instead | all arms |
| **Every position type, both seats' turns** | trains on the positions the search asks about | docs/013, lesson 5. In Hearthstone, a winner predictor trained on random bots' games fell from 0.79 to 0.75 AUC on MCTS bots' games (Janusz et al., 2017) | all arms |
| **Cross-entropy on P(win), not MSE through tanh** | calibrated, without tanh's vanishing gradients. (1 + tanh x)/2 is a sigmoid, so the head itself is unchanged | for chess values, classification beat L2 regression, 61.8% against 58.9% move accuracy (Ruoss et al., 2024) | all arms |
| **Condition on skill** | separates the position from the players; set to the top rank when used in search | Maia-2, ALLIE and AlphaStar Unplugged condition on rating. 17lands' win-rate bucket includes the game's own result (docs/008 §3.5); the player's Arena rank (98.7% filled) doesn't | optional: the pool is top players already |
| **TD(λ) along each game** | lower-variance targets from the following snapshots | self-play's λ = 0.95. But Monte Carlo targets matched TD in Amiranashvili et al. (2018), and a TD value diverged in AlphaStar Unplugged | a sweep option (§6.3) |
| **Auxiliary targets:** final life difference, turns to the end, card advantage | more than one bit a game into the shared trunk | KataGo: dropping its score and ownership targets cost 1.65× the training (Wu, 2019) | a sweep option (§6.3) |
| **Rollout distillation** | fresh, independent results: roll the cloned policy forward from human positions | the best-documented route. piKL (appendix J) and Cicero's pre-RL value roll the human policy forward a few turns and train on the value after; AlphaGo's 30M-game set kept one position per game | later; stage 6's frozen-policy value training (§6.6) is the same idea on games the bot plays |
| **Mix with the heuristic at the leaf** | a second opinion, at no training cost | AlphaGo: value net alone 2177 Elo, rollouts alone 2416, the 50/50 mix 2890 | an extra leaf setting in the cheap evaluation, if the budget allows |
| **Expectile value (IQL)** | values the better-than-average continuation | IQL (Kostrikov et al., 2022); untested in games | no. With result-only targets it only rescales P(win); worth a try later with TD |
| **A privileged teacher from mirrored games** | in the 13.7% of games recorded from both seats, a value that sees both hands teaches the fair one | AlphaStar gave its value function the opponent's observations during training: 22% → 82% in its ablation | later |

Two card-game results bear on the choice. In Hearthstone, a winner predictor trained on bot games
replaced random rollouts inside MCTS. It lifted the weaker deck from 26.5% to 50.2% against plain
MCTS (Świechowski et al., 2018). In Legends of Code and Magic, a value cloned from bot games, with
one turn of search, lifted a bot from 26.8% to 51.4% against the champion (a 2026 preprint). Both
learned from bots, not people. We found no published work that trains a Magic value on human
games.

### 4.4 How the value head is judged

Every comparison pairs a cloned head with the heuristic, on the same positions or the same deals.

1. **Offline, on the test split** (free on the training pod). The pilot's measures: AUC,
   calibrated log-loss, AUC by turn, and the AUC of the change since turns 1–2. Run them on every
   kind of position, including the opponent's turn. The build records the heuristic's score for
   every decision (the bridge's `encode` has a `heuristic` option now), so it's in every
   comparison at no extra cost.
2. **Through the search, on 1,000 held-out decisions** (§6.5). After 300 and after 3,000
   simulations, how well the search's root value predicts who won the game, with each leaf
   evaluator and each prior. The search's agreement with the human's choice comes from the same
   runs.
3. **In play:** the imitation-plus-cloned-value bot against the heuristic bot at 1,000 simulations,
   200 games (§6.5).

## 5. Multiple epochs

- **Yes.** #2b made 17 passes over its 132,603 decisions (docs/013 §1.3):
  - the policy's validation loss (set NLL) was 0.574 after 1.7 passes, best at 0.548 after about
    4, and flat after that;
  - the value head was best after about 9 passes, then drifted worse.
- **Several epochs are affordable.** One epoch over about 7M decisions is 3.5× all the samples #2b
  saw (2.0M), about 2.3 hours on an RTX 3090 at #2b's speed. Plan for several, save checkpoints as
  it goes, and stop once validation has been flat for about an epoch.
- **Repetition is nearly free up to a point.** In language-model studies, up to about 4 epochs of
  repeated data are almost as good as new data, and returns fade after that (Muennighoff et al.,
  2023, *Scaling Data-Constrained Language Models*).
- **The value head overfits first.** Positions from one game share its result and most of its
  cards, and a draft's ~5.8 games share a deck. So the network can learn to recognise a game and
  recall its result. To guard against it:
  - keep the best value head separately from the best policy heads;
  - keep the value weight low (docs/008 §7.4);
  - use only a few positions per game for the value loss in each epoch.
- **Plasticity** ([docs/013](013-exp2-imitation-report.md) §2.3): long training made #2b's network
  hard to train further. That doesn't matter for using the network in search as it is. If it seeds
  self-play later, apply shrink and perturb and screen it with docs/013's test first.

## 6. The plan: about $36–44 on RunPod

### 6.1 Stages

| Stage | What | Where | Pod-hours | Cost |
|---|---|---|---|---|
| 0. Engineering | the pieces in §6.2, each tested on the laptop | laptop | – | – |
| 1. Build | the tables of §2.2 from the top players' games, with the heuristic's score recorded | CPU-heavy pod | 5 | $2.50 |
| 2. Hyperparameter sweep | about 13 short runs on a 10% subset; a plot and a report (§6.3) | GPU pod | 6 | $3–6 |
| 3. Large training | the chosen settings, all training games, several epochs, checkpoints throughout (§6.4) | GPU pod | 12 | $6–8 |
| 4. Cheap evaluation | 1,000 held-out decisions, IS-MCTS at 300 and 3,000 simulations, the four prior and leaf mixes (§6.5) | GPU pod | 6 | $3–6 |
| 5. Play | the heuristic bot against the imitation-plus-cloned-value bot, IS-MCTS at 1,000 simulations, 200 games (§6.5) | GPU pod, 24+ vCPU | 10.5 | $5.25 |
| 6. Follow-up training | about 300 self-play games at 1,000 simulations (a quarter held out), then three training arms (§6.6) | GPU pod | 21 | $10.50 |
| 7. Evaluate them | 80 games each at 1,000 against the heuristic bot, and the 1,000 decisions at 300 and 3,000 | GPU pod | 12 | $6 |
| **Total** | | | **~72** | **~$36–44** |

- **The low end assumes RTX 3090 rates ($0.50/hr);** the high end, an L40S for the training stages
  (§6.7).
- **The games run at 1,000 simulations,** not 3,000. That cuts the play stages from about 51
  pod-hours to 18, and the total from ~$45–55 to ~$30–38 (stage 1 later grew to include the
  opponent's turns). IS-MCTS (§3.4) still costs more than
  PIMC would have, mostly for the heuristic bot.
- **The game stages are estimates.** A game is assumed to have about 140 searched decisions (from
  #2a's 61 games an hour at 300 simulations). At 1,000 simulations IS-MCTS costs 1.7 pod-seconds a
  decision with the network and 1.0 for the heuristic seat (docs/016 §4.1). That gives about 19
  games a pod-hour. Stage 0 measures it.
- **If a stage runs 50% over its estimate, stop and report** before going on.

### 6.2 Engineering before the first pod (stage 0)

1. **The table builder:**
   - top players' games only, and every turn replayed;
   - every stop where the player has a real choice;
   - the opponent's turns replayed, for blocks and the player's instants;
   - labels graded exact or imputed;
   - the heuristic's score recorded (done: `encode` takes a `heuristic` option).
2. **The trainer:**
   - the player priority, binary and target heads, and the value with cross-entropy;
   - the best policy and the best value kept separately;
   - checkpoints throughout, and resumable;
   - options for the sweep: layers, width, transformer or MLP, learning rate, value targets.
3. **The search (`BenchSearch`):**
   - the four mixes (the imitation policy or uniform priors, the cloned value or the heuristic at
     the leaves);
   - uniform priors at the opponent's nodes;
   - the root value in its output.
4. **The 1,000 held-out decisions (sb-v2),** from the new test split, with timing items built
   from end-of-turn snapshots (§6.5). Timing in the middle of a turn (the second main phase, the
   stack) would need a mid-turn search root, which experiment #3's pipeline couldn't build
   (docs/016 §1.1). It needs a converter from the engine's state back to a spec, a later addition.
5. **A game player:**
   - full games with the same search (IS-MCTS) at every decision, re-dealing hidden cards every
     simulation (§3.4);
   - deck pairs from the eval pool, seats swapped;
   - it also writes the self-play training data for stage 6.
6. **A GPU check:** evaluations a second and training samples a second, per GPU type, and a quick
   attempt at a faster inference server (§6.7).

### 6.3 Hyperparameter sweep (stage 2)

- **What varies,** one setting at a time around the current default (2 layers, width 512, learning
  rate 3e-4):
  - layers 1 or 4;
  - width 256 or 768;
  - transformer against MLP (MageZero's own `Net`);
  - learning rate 1e-4 or 1e-3;
  - the value target: the result alone, or with TD(λ) and auxiliary targets (§4.3).

  That's about 11 runs, plus the default with a second seed to measure the noise.
- **Each run** uses the same 10% of the training games, the same time budget (about 25 minutes)
  and the same validation rows.
- **Measures:**
  - the policy's non-Pass top-1 and set NLL, by decision type, timing included;
  - the value's log-loss and AUC;
  - inference speed, since the search pays for every evaluation.
- **Report:** a plot of each measure by setting, with the two default seeds as the noise band.
- **The rule: keep the default** unless a setting beats it by more than twice the seed-to-seed
  difference, is no worse on the other head, and slows inference by less than 25%.

### 6.4 Large training (stage 3)

- The chosen settings, all training games, several epochs.
- Every ~30 minutes: a checkpoint, and the validation measures. Keep the best policy and the best
  value states.
- Stop once validation has been flat for about an epoch.
- **Report:** learning curves, and the test-split measures of §4.4, tier 1.

### 6.5 Evaluation (stages 4 and 5)

**Always a mix of approaches:**

| | Heuristic at the leaves | Cloned value at the leaves |
|---|---|---|
| **Default (uniform) prior** | the heuristic bot: experiment #3's offline search | the value alone |
| **Imitation prior** | the policy alone, guiding the search | **the imitation-plus-cloned-value bot** |

The policy alone and the value head alone, with no search, are references.

**Cheap: 1,000 held-out decisions (sb-v2).**
- From the top players' test games, as experiment #3's sb-v1: 300 spells, 125 holds, 225 attacks
  and 150 blocks.
- Plus 200 timing items, with exact labels only:
  - 100 `endstep`: the player's own end step, holding an instant, flash card or ability;
  - 100 `oppwindow`: the player's first window in the opponent's turn, in a turn in which they cast
    nothing.

  Both are labelled Pass, and both feed the cast-or-pass question of the balanced score.
- The second main phase and answers to the stack are scored on the policy alone, on the replayed
  test-split rows, until mid-turn search roots exist (§6.2).
- IS-MCTS at 300 and 3,000 simulations, with each item's 8 belief worlds, for each of the four
  mixes. Two scores from each run:
  - **does the policy predict human moves:** the balanced score and agreement by type, timing
    included;
  - **does the value predict results:** the AUC of the search's root value against the game's
    result.

**Sizable: play.**
- The heuristic bot against the imitation-plus-cloned-value bot, IS-MCTS at 1,000
  simulations, both seats.
- 200 games: 100 deck pairs from the eval pool, each played with seats swapped. That gives a 95% CI
  of about ±7 points.
- **Closed decklists:** each bot samples the opponent's hidden cards from the belief model (§3.4).

### 6.6 Two follow-up checkpoints (stages 6 and 7)

**The starting point is the IL+BC network:** stage 3's output, the policy learned by imitation and
the value by behaviour cloning from top players' games. Self-play shouldn't throw away what it
learned from the experts. So these stages test whether training on the bot's own games keeps it,
or collapses as #2b's did, when its policy overwrote the human one within one epoch (docs/013).

*Agreed in review; the code is built and tested on the laptop (§8.1).*

**Stage 6a: high-quality self-play games.**
- About 300 games of the IL+BC bot against itself, at **1,000 simulations** per decision (the
  evaluation budget, since quality is the point), with closed decklists. That's about 20
  pod-hours, ~$10.
- The two games of a deck pair get different seeds in self-play. With one seed, the swapped game
  would replay the first exactly, since the same bot sits in both seats.
- Each decision records the state, the search's visit counts by action index (the policy
  target), its root value, the heuristic's score and the turn, with each seat's result. Each game
  is written to its own file, whole, so a stopped pod loses only unfinished games.
- The games split by deck pair: 65% training, 10% validation, and 25% held out as the evaluation
  set, never trained on.

**Stage 6b: evaluate on the bot's own games.** The held-out games are the positions the search's
leaves actually reach, which human test games aren't.
- **The value:** the IL+BC network's value head against those games' results (AUC, log-loss,
  calibration error), and the heuristic's AUC on the same positions beside it. Does a value
  learned from human results hold up on bot positions?
- **The policy:** top-1 agreement with the search's most-visited option, and cross-entropy against
  its visit distribution. That shows how much search changes the human policy.

**Stage 6c: three training arms**, each from the IL+BC network, on the same games and human rows,
for the same 8 epochs (an epoch is one pass over the self-play training rows), at learning rate
1e-4:
1. **No anchor**, the control: self-play targets only. Expected to drift as #2b did.
2. **Anchored:** the same, plus λ·KL(π_IL+BC ‖ π) on every self-play position, with λ annealed from
   1.0 to 0.1 over the run (piKL, AlphaStar, Diplodocus). 15% of the samples come from the human
   tables, with stage 3's losses.
   - The value is anchored only through the human data, not by a KL to the cloned value. The
     cloned value is the part self-play should improve (§4.1), and a tight anchor would keep its
     weakness. A small value-KL is an extra arm if the budget allows.
3. **Frozen policy:** arm 2 with the trunk and policy heads frozen, so only the value head learns.

**Targets on the self-play positions:**
- **Policy:** cross-entropy against the search's visit distribution over the legal options. Each
  decision type trains its own head: player priority, targets, yes/no.
- **Value:** about 300 games give only about 300 independent results. So each position's target
  blends the game's result with the search's root values, backwards through the seat's decisions,
  as MageZero's self-play does (TD(λ), λ = 0.95).
- **The KL's reference** is the IL+BC network's policy on the full state, computed once.

**Collapse is measured offline** at every evaluation, which costs almost nothing:
- on the human validation rows: top-1 and non-Pass top-1, how often the top choice is Pass
  against how often the humans only passed, and the policy's entropy;
- on the held-out self-play games: agreement with the search, KL to the IL+BC policy, and the
  value's AUC, log-loss and calibration error;
- sb-v2's balanced score for checkpoints that look promising.

**Stage 7:** the best arm or two play 80 games each at 1,000 simulations against the heuristic bot
(about ±11 points), and run sb-v2 at 300 and 3,000.

### 6.7 Compute: GPU and storage

- **What's GPU-bound:** training, network search and network games.
- **What's CPU-bound:** the build, the heuristic bot, and the engine side of every game.
- **Experiment #3's network searches kept the RTX 3090 96–99% busy** at 550–700 evaluations a
  second (docs/016 §1.4). At about 9 GFLOP an evaluation (2 layers, width 512, about 760 tokens),
  that's roughly a tenth of its fp16 peak. So the limit is how the server batches, as much as the
  card. Stage 0 measures before choosing:
  - evaluations a second and training samples a second on an RTX 3090 and an L40S, 15 minutes each;
  - a batched fp16 inference server.
- **Quotes on 2026-09-30,** one GPU, stock low everywhere:

  | Pod | $/hr | vCPU | RAM |
  |---|---|---|---|
  | L40S, Secure | 1.09 | 32 | 125 GB |
  | L40S, Community | 0.79 | 24 | 251 GB |
  | RTX 5090, Community | 0.69 | 32 | 107 GB |
  | RTX A6000, Secure | 0.53 | 16 | 62 GB |
  | RTX 3090, Secure | not in stock (it was $0.50 with 32 vCPU in experiments #2–3) | | |

  Community pods may have no public IP for SSH (docs/005).
- **Choose by dollars per evaluation, from the check.** The L40S pays only if it's at least 2.2×
  the 3090's throughput. Take 24+ vCPUs for the game stages, and the cheapest high-vCPU pod for
  the build.
- **Storage:**
  - tables of about 33 GB (2.6 KB a row × ~12.6M);
  - gzipped build shards of about 30 GB;
  - checkpoints of about 175 MB each with optimizer state, so keep weights only for most of them;
  - a few GB of self-play data.

  Request about 150 GB. A network volume (Secure Cloud only) keeps the data between pods;
  otherwise, a container disk of 100 GB or more. The default 40 GB filled up in docs/013. Load the
  tables into RAM for training: the pods have 100+ GB.

### 6.8 Running it: handoff to the run session

The stages are run from a fresh session that starts from this section. Read §6–8 first; §7 has the
decisions.

**Rules:**
- Every stage writes its results into §8 as it finishes: numbers, plots, pod-hours and dollars,
  and each pod's quote and what it actually got (docs/005).
- Every pod gets a self-destruct when it's created (docs/005), sized to its stage's estimate in
  §6.1 plus about half.
- Track spend against the budget (§6.1). If a stage runs 50% over its estimate, stop and report.
- Remove pods (`runpodctl pod remove`, not `stop`) as soon as a stage's outputs are copied off.

**Before the first pod:**
1. **The code is on main** (§8.1 lists the pieces). Pods check it out from GitHub.
2. **Copy the data a pod can't download** (from this laptop; none of it is committed):
   - `data/17lands/`: the FDN replay file (438 MB), `cards.csv`, `abilities.csv`, and
     `xmage_Foundations_SpecialGuests.json`;
   - `data/gameplay/`: `pairs_FDN_PremierDraft.jsonl` and `deckpool_FDN_PremierDraft.npz`;
   - `data/deckgen/FDN_PremierDraft_wr60/decks.jsonl`: the deck to draft map, for closed
     decklists;
   - `data/imitation_scale/row_split_exp4.npy` (the splits, §2.1);
   - `data/search_bench/sb-v2/` (the benchmark set, built on the laptop);
   - `xmage/db/cards.h2.mv.db` (the v0.2 bundle ships without its card database:
     `deploy/search_bench_setup.sh`).
3. **Set up the pod:** `HF_TOKEN=... bash deploy/search_bench_setup.sh` (the bundle, the bridge,
   experiment #2's networks), then `export MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv`.
   Run the bridge tests (`pytest tests/test_gameplay_bridge.py tests/test_search_bench.py
   tests/test_imitation_scale.py`) before anything long.
   - **The bridge needs the v0.2 bundle at run time too** (`MZ_XMAGE_DIR`). Against the v0.1
     bundle in the repo's `xmage/`, a jar built for v0.2 fails at run time with an
     `IllegalAccessError` on `MCTSNode.state`. On the laptop: `export
     MZ_XMAGE_DIR=~/Desktop/Code/mage-builds/generalist-v0.2/xmage`.
4. **Do the GPU check** on the first GPU pod (§6.7): `supervised bench-speed` and a short network
   search at 32 workers. Compare $ per evaluation for a 3090 and an L40S, then choose.

**Stage commands:**
- **Stage 1, build** (a CPU-heavy pod, 28 workers, a 150 GB disk):

  ```bash
  python tools/imitation_scale/build.py build --workers 28     # ~3.7 h; shards in data/imitation_scale/shards
  python tools/imitation_scale/build.py tables                 # HDF5 tables in data/imitation_scale/h5
  ```

  Check `build_stats.json` and `tables_stats.json` against §2.2's smoke-build rates. Copy the
  tables to the GPU pod, or keep them on a network volume.
- **Stages 2 and 3, sweep and training** (a GPU pod; the tables copied over or on a network volume).
  First a short CUDA check, which the laptop couldn't run:

  ```bash
  python -m draftzero.gameplay.supervised train --config configs/exp4_train.yml --out runs/exp4/cuda_check \
      --tables-dir data/imitation_scale/h5 --set max_steps=200
  python -m draftzero.gameplay.supervised bench-speed --checkpoint runs/exp4/cuda_check/final.pt.gz \
      --table data/imitation_scale/h5/turnstart_val.h5
  python -m draftzero.gameplay.supervised sweep --spec configs/exp4_sweep.yml --out runs/exp4/sweep \
      --tables-dir data/imitation_scale/h5         # ~13 runs x 25 min; re-running resumes
  # set the sweep's choices in configs/exp4_train.yml (arch, lr, value target), then:
  python -m draftzero.gameplay.supervised train --config configs/exp4_train.yml --out runs/exp4/main \
      --tables-dir data/imitation_scale/h5
  python -m draftzero.gameplay.supervised train --out runs/exp4/main --resume      # after a stop
  python -m draftzero.gameplay.supervised evaluate --checkpoint runs/exp4/main/best.pt.gz --split test
  ```

  - **Network shape:** only the default network (2 layers, width 512) loads into MageZero's server
    unchanged. Another architecture can win the sweep only with a server change; the rule in §6.3
    already leans to the default.
  - **Outputs:** the run keeps `best_policy`, `best_value` and `best` checkpoints in MageZero's
    format, a checkpoint every 30 minutes, and `latest.pt` for resuming.
- **Stage 4, sb-v2:** serve the stage 3 checkpoint with `tools/search_bench/serve.py` (MageZero's
  server, which returns the policy heads). Then `tools/search_bench/run.py --items
  data/search_bench/sb-v2 --split test --methods ismcts --budgets 300,3000`, once per mix:

  | Mix | Flags |
  |---|---|
  | heuristic bot | `--evaluator offline --grid e2` |
  | value alone | `--evaluator remote --grid e2 --leaf net` |
  | policy guiding the heuristic | `--evaluator remote --grid priors --leaf heuristic` |
  | imitation plus cloned value | `--evaluator remote --grid priors --leaf net` |
  | references | `--evaluator remote --grid policy` (the policy alone) |

  The defaults are experiment #4's: uniform priors at the opponent's nodes, and IS-MCTS policies
  per actor and decision type. The value score is the AUC of each decision's `rootQ` against the
  item's `won`.
- **Stage 5, play** (a GPU pod with 24+ vCPUs; the stage 3 network served as in stage 4; the eval
  deck pool, built by `tools/extract_decks.py`, or the experiment #1 release on Hugging Face):

  ```bash
  python tools/imitation_scale/play.py --pool data/pools/eval.txt --deck-root $MZ_DECK_DIR \
      --pairs 100 --budget 1000 --bot1 il_bc --bot2 heuristic --ports 50052,50152,50252,50352 \
      --workers 28 --out runs/imitation_scale/play_stage5
  ```

  - **Pairing:** each deck pair is played twice. The decks keep their seats and the bots swap, so
    each bot plays each deck once and goes first once.
  - **Resuming:** `games.jsonl` grows a line per game, and re-running skips finished games.
  - **The result:** `summary.json` has bot1's score and its 95% interval.
  - **First, time it:** a few games at budget 1,000, to check §6.1's estimate of about 19 games per
    pod-hour.
  - **Report `activationFailures`** from each game's seat stats: the plays MageZero couldn't carry
    out, where the bot passed instead (§8.1). The smoke test's rate was about 0.8% of decisions.
- **Closed decklists:** start the belief service before any games:
  `python tools/imitation_scale/belief_server.py --port 50070`. It needs
  `data/deckgen/FDN_PremierDraft_wr60/decks.jsonl` and `data/gameplay/deckpool_FDN_PremierDraft.npz`
  (copy both). `play.py` refuses to run without it unless given `--open-decklists`.
- **Stage 6** (§6.6). First set `init_checkpoint` in `configs/exp4_stage6.yml` to stage 3's network
  (the IL+BC network). The games are the cost; training takes minutes on a GPU.

  ```bash
  # 6a: ~300 self-play games at 1,000 simulations (the stage 3 network served as in stage 4)
  python tools/imitation_scale/play.py --pool data/pools/eval.txt --deck-root $MZ_DECK_DIR \
      --pairs 150 --budget 1000 --bot1 il_bc --bot2 il_bc --record --ports 50052,50152,50252,50352 \
      --workers 28 --out runs/exp4/stage6/play          # records/: one file per game
  python tools/imitation_scale/selfplay_tables.py --records runs/exp4/stage6/play/records \
      --out data/imitation_scale/h5                     # selfplay_{train,val,test}.h5; the heuristic's AUC per split
  # 6b: the IL+BC network on the held-out games (and the human test split)
  python -m draftzero.gameplay.supervised evaluate --checkpoint runs/exp4/main/best.pt.gz \
      --config configs/exp4_stage6.yml --split test --json runs/exp4/stage6/ilbc_test.json
  # 6c: the three arms; each run's evals.jsonl is its collapse curve, sweep.csv sums them up
  python -m draftzero.gameplay.supervised sweep --spec configs/exp4_stage6_arms.yml \
      --out runs/exp4/stage6/arms
  ```

  - **The test split's rows** are capped at `val_rows` (20,000) per table, whole games; the
    self-play test split is smaller than that.
  - **The arms' checkpoints** (`ckpt/`, every 2 minutes) are what stage 7 picks from, by the
    collapse measures.
- **Stage 7:** `play.py` with `--pairs 40` (80 games) for each new checkpoint, and stage 4's runs
  for each.

## 7. Decisions from review, and what's still open

**Decided:**
1. **Data:** top players only. Players under 60% are a reserve, first for the value head.
2. **The deciding test:** 200 games at 1,000 simulations against the heuristic bot, then 80 games
   for each follow-up checkpoint, also at 1,000. The 1,000 benchmark decisions stay at 300 and
   3,000.
3. **No opponent head in the first run:** uniform priors at the opponent's nodes.
4. **Rollout distillation waits.** Stage 6's frozen-policy value training is the same idea, on games
   the bot plays.
5. **IS-MCTS for every search** (§3.4); the budget can grow to cover it (§6.1).
6. **The game player** is new, built on experiment #3's IS-MCTS. MageZero's own evaluator has no
   IS-MCTS: its search sees the hidden cards, and it walks its whole tree every simulation, which
   is slow at high budgets.
7. **Closed decklists in games** (§3.4): the opponent's hidden cards are sampled from the belief
   model, as in the benchmark, never dealt from its real list.
8. **80 games** for each follow-up checkpoint (item 2).
9. **Stage 6** (§6.6): about 300 self-play games at 1,000 simulations, a quarter held out as an
   evaluation set, then three arms from the IL+BC network (no anchor, anchored, frozen policy). The
   anchor is a policy KL toward the IL+BC network plus human data in every batch. About $10.50 for
   the games and training.

Nothing is awaiting review.

## 8. Results

*Filled in as the stages run. Spend so far: $0 (initial budget $30–40, which can grow, §6.1).*

| Stage | Status | Pod-hours | Cost | Result |
|---|---|---|---|---|
| 0. Engineering | done on the laptop (§8.1) | – | – | |
| 1. Build | not started | | | |
| 2. Hyperparameter sweep | not started | | | |
| 3. Large training | not started | | | |
| 4. Cheap evaluation | not started | | | |
| 5. Play | not started | | | |
| 6. Follow-up checkpoints | not started | | | |
| 7. Their evaluation | not started | | | |

### 8.1 Stage 0: engineering

*All of stage 0's code is on main.*

**Stage 0 is done, stage 6's code included.** Every piece below is tested on the laptop; all
384 tests pass (one skipped) with `MZ_XMAGE_DIR` set to the v0.2 bundle (§6.8). Nothing ran on
a pod.

- **The table builder** (`tools/imitation_scale/build.py`; `splits`, `build`, `tables`) covers:
  - top players only, every turn replayed, every stop with a real choice;
  - the opponent's turns replayed, for the player's off-turn plays and every block;
  - spell targets settled by the outcome.

  Rows come with exact or imputed labels: a training weight of 1 for exact, 0.5 for imputed. Every
  row carries the heuristic's score. Shards are gzipped. Speed: 0.5–2.5 worker-seconds a game, so
  1–4 hours on a 28-worker pod.
  - **The tables** (per split): `turnstart`, `replay_priority`, `replay_attack`, `replay_target`,
    `block`, `opp_priority` and `opp_block`. `block` (the first block question of each attack,
    from its end-of-turn state) and `opp_block` (every blocker, from the replay) overlap; the
    configs train on `opp_block`.
  - **End to end:** a 120-game build trained in the trainer.
- **Bridge changes:**
  - the replay records every stop with a real choice (`allStops`), and the non-active seat's
    decisions (`recordSeat`);
  - replay and `encode` decisions carry the heuristic's score (`heuristic`);
  - `encode` can return the built state's dump (`dump`), which labels blocks;
  - a new `play` op (§6.8, stage 5).
- **The opponent's-turn replay** (`turnreplay.replay_opp_turn`, `labels.opponent_turn_label`)
  reproduces about 85% of opponent turns.
- **The search's options** (from a background agent, then extended): the four prior and leaf mixes,
  uniform priors at the opponent's nodes, and the root values.
  - **A bug it found in experiment #3's IS-MCTS:** a shared node's policy was read once, in the
    world that created the node, and applied in every world. In other worlds the node can belong
    to the other player or be another kind of decision: up to 7–12% of prior applications per
    decision.
  - **The fix:** an option, `isPolicyPerWorld`, on by default in `run.py` and `play.py`, keeps one
    policy per (actor, decision type) at each shared node. Experiment #3's plans pin it off, so its
    results reproduce. `policyMismatches` counts the old behaviour's wrong applications.
- **The new benchmark set (sb-v2) is built:** `data/search_bench/sb-v2`, from
  `tools/search_bench/items.py build --version sb-v2` (6 minutes on the laptop).
  - 1,000 decisions from 541 top players' test games: 300 spells, 125 holds, 225 attacks, 150
    blocks, 100 `endstep` and 100 `oppwindow` items.
  - All tiers T0 or T1, each with its game's result (`won`) and 8 belief worlds for IS-MCTS. The
    players won 64.6% of these games.
  - The second main phase and stack decisions are scored on the policy alone, from the test
    split's replayed rows. A search root in the middle of a turn needs a converter from the
    engine's state back to a spec.
- **The trainer** (`src/draftzero/gameplay/supervised.py`, from a background agent; configs
  `configs/exp4_sweep.yml` and `configs/exp4_train.yml`):
  - **What it trains:** every head, with the value as cross-entropy on P(win), optionally on a few
    positions per game.
  - **Its options:** TD(λ) and auxiliary value targets, transformer or MLP, layers and width.
  - **Its outputs:** separate best policy and best value checkpoints, checkpoints every 30 minutes,
    an exact resume, a plateau stop, a sweep with a plot, and `bench-speed` and `evaluate`
    commands.
  - **Checked:** on #2b's network it reproduces #2b's published numbers (73.1% turn-start top-1,
    set NLL 0.543).
  - **Not yet run:** its CUDA path, since the laptop has no GPU.
- **The game player** (`mage.player.ai.BenchPlayer`, the bridge's `play` op,
  `tools/imitation_scale/play.py`):
  - **How it works:** a MageZero player whose search is IS-MCTS from MageZero's own root at every
    decision. Hidden cards are re-dealt every simulation.
  - **Closed decklists:** 8 belief worlds per decision, from the belief service (§3.4).
  - **Training records:** each decision is recorded as MageZero's self-play does (state, visit
    counts by action index, root value), plus the heuristic's score, for stage 6. One file per
    game.
  - **Reproducible per fresh worker:** a seeded game replays exactly in a new worker (checked
    twice), but a worker that has already played other games can play it differently: some
    engine state carries over between games. So a run of `play.py` doesn't replay exactly (games
    go to whichever worker is free); the scores don't depend on it.
  - **When MageZero can't carry out the chosen play** ("failed to activate chosen ability", an
    invalid script), the player now passes instead and counts it (`activationFailures`).
    MageZero throws there, and a test-mode game ends with XMage's "Error in unit tests": 2.6% of
    experiment #2's self-play games (docs/013 §4.4, docs/014), and 2 of 12 in this smoke test.
    - With the fallback all 12 finished. Two games had failures, 2 and 10 of them, each from one
      seat: about 0.8% of the decisions. The one traced was a seat repeatedly choosing to cast
      Burst Lightning (kicker, any target).
    - The fallback restores the state and takes a new last-priority snapshot, which clears the
      action histories MageZero rebuilds its roots from. Then it passes.
  - **Self-play seeds:** in a mirror match (`il_bc` against `il_bc`) the swapped game of a deck pair
    gets its own seed. With one seed it replayed the first game exactly, since the same bot sat in
    both seats: half of stage 6's games would have been duplicates.
  - **Tested on the laptop:** heuristic against heuristic (16–23 turns; 30–100 s a game at 50
    simulations), and the imitation-plus-cloned-value bot against the heuristic with #2b's network
    on CPU. The network bot made 1,189 and 3,265 network calls in its two games, the per-world
    policy fix was active, and every chosen option matched one of MageZero's.
- **Stage 6's code** (§6.6):
  - **Records to tables:** `tools/imitation_scale/selfplay_tables.py` turns `play.py`'s records
    into `selfplay_{train,val,test}.h5`, split 65/10/25 by deck pair. Each row has the visit shares
    over the legal options, the decision type, the result, MageZero's TD(0.95) value target, and
    the heuristic's score.
  - **Trainer options** (`supervised.py`):
    - a `soft` table kind (cross-entropy against visit shares, each decision type on its head);
    - a per-table `value_column`, so the value trains on the TD targets and is scored on results;
    - `init_checkpoint` (start from a network, with its shape and feature vocab);
    - `kl_weight` / `kl_weight_end` / `kl_ref`, a KL toward a frozen reference policy;
    - `freeze` (trunk, policy or value);
    - table `group`s with fixed `group_shares` (the human mix). With groups, an epoch is a pass
      over the self-play rows.
  - **The collapse measures** are in every evaluation: Pass at the top against the humans', and
    entropy, on human rows; on self-play rows, agreement with the search, KL to the reference,
    and the value's calibration error.
  - **Configs:** `configs/exp4_stage6.yml` (shared settings; it also trains the anchored arm alone
    and drives stage 6b's `evaluate`) and `configs/exp4_stage6_arms.yml` (the three arms as a
    sweep; a sweep spec can now name a `base_config`).
  - **Checked end to end:** 12 recorded heuristic-against-heuristic games (50 simulations, 1,290
    rows: 751 priority, 251 target, 303 yes/no decisions), tables, then all three arms for 12 steps
    from #2b's network on CPU with the smoke build's human tables. The frozen arm's KL to the
    reference stayed exactly 0. A 30-step anchored run moved the KL from 0 to 0.008 while the
    weight annealed from 1.0 to 0.1. Synthetic-table tests cover exact resume, the frozen
    parameters, the shares and the KL at the start.
- **Value weight:** #2b used 0.5 with MSE; with cross-entropy the scale differs, and the sweep
  doesn't vary it. Worth one more sweep run (`value_weight: 0.1`).

## Later ideas (not in this experiment)

- **Sharper beliefs in games** (§3.4): remember cards that left the known zones, keep known cards
  in the opponent's hand, and turn on the hand-retention weighting.
- **The opponent head** (§3.2), in place of uniform priors at the opponent's nodes.
- **Rollout distillation** (§4.3), if stage 6's value training helps.
- **A one-move-deep test** of each value head on the benchmark decisions: score the position after
  every option and pick the best (Ruoss et al., 2024). It isolates the sibling comparison that
  search depends on (§4.2).
- **Everyone's games for the value head,** if its learning curve is still rising at 140k games.
- **Self-play from this start,** with the human anchors of docs/016 §9 step 3: a KL penalty toward
  the frozen human policy, and human decisions mixed into every batch.
- **A larger network.** 700k games may support one, but it would slow every search evaluation.
- **Relabelling human positions with search** (Reanalyse), once the value is good enough to trust.
  AlphaStar Unplugged found that training on search targets offline collapsed its policy; it
  searched only at inference.
- **Short rollouts at the leaf:** play the cloned policy to the end of the turn, then apply the
  value head, as SearchBot did in Diplomacy and the Hearthstone bot did. It costs more per
  simulation.
- **An expectile (IQL) value with TD targets,** and **a privileged teacher** from the mirrored
  games (§4.3).
- **Other sets:** 17lands publishes replay data for 27 more sets (docs/008).

## References

- Amiranashvili et al. (2018), *TD or not TD: Analyzing the Role of Temporal Differencing in Deep
  Reinforcement Learning*, [arXiv 1806.01175](https://arxiv.org/abs/1806.01175).
- Bakhtin et al. (2023), *Mastering the Game of No-Press Diplomacy via Human-Regularized
  Reinforcement Learning and Planning* (Diplodocus), [arXiv 2210.05492](https://arxiv.org/abs/2210.05492).
- Gray et al. (2021), *Human-Level Performance in No-Press Diplomacy via Equilibrium Search*
  (SearchBot), [arXiv 2010.02923](https://arxiv.org/abs/2010.02923).
- Jacob et al. (2022), *Modeling Strong and Human-Like Gameplay with KL-Regularized Search* (piKL),
  [arXiv 2112.07544](https://arxiv.org/abs/2112.07544).
- Janusz, Tajmajer and Świechowski (2017), *Helping AI to Play Hearthstone: AAIA'17 Data Mining
  Challenge*, [arXiv 1708.00730](https://arxiv.org/abs/1708.00730).
- Kostrikov, Nair and Levine (2022), *Offline Reinforcement Learning with Implicit Q-Learning*,
  [arXiv 2110.06169](https://arxiv.org/abs/2110.06169).
- Mathieu et al. (2023), *AlphaStar Unplugged: Large-Scale Offline Reinforcement Learning*,
  [arXiv 2308.03526](https://arxiv.org/abs/2308.03526).
- McIlroy-Young et al. (2020), *Aligning Superhuman AI with Human Behavior: Chess as a Model System*
  (Maia), [arXiv 2006.01855](https://arxiv.org/abs/2006.01855).
- Meta FAIR Diplomacy Team (2022), *Human-level play in the game of Diplomacy by combining language
  models with strategic reasoning* (Cicero), *Science* 378:1067.
- Muennighoff et al. (2023), *Scaling Data-Constrained Language Models*,
  [arXiv 2305.16264](https://arxiv.org/abs/2305.16264).
- Rubin (2026), *Unsound Search with Policy and Value Networks in Legends of Code and Magic*,
  preprint, [arXiv 2609.06816](https://arxiv.org/abs/2609.06816).
- Ruoss et al. (2024), *Grandmaster-Level Chess Without Search*,
  [arXiv 2402.04494](https://arxiv.org/abs/2402.04494).
- Silver et al. (2016), *Mastering the game of Go with deep neural networks and tree search*
  (AlphaGo), *Nature* 529:484.
- Silver et al. (2017), *Mastering the game of Go without human knowledge* (AlphaGo Zero), *Nature*
  550:354.
- Świechowski, Tajmajer and Janusz (2018), *Improving Hearthstone AI by Combining MCTS and
  Supervised Learning Algorithms*, [arXiv 1808.04794](https://arxiv.org/abs/1808.04794).
- Tang et al. (2024), *Maia-2: A Unified Model for Human-AI Alignment in Chess*,
  [arXiv 2409.20553](https://arxiv.org/abs/2409.20553).
- Vinyals et al. (2019), *Grandmaster level in StarCraft II using multi-agent reinforcement
  learning* (AlphaStar), *Nature* 575:350.
- Wu (2019), *Accelerating Self-Play Learning in Go* (KataGo),
  [arXiv 1902.10565](https://arxiv.org/abs/1902.10565).
- Zhang et al. (2025), *Human-Aligned Chess With a Bit of Search* (ALLIE),
  [arXiv 2410.03893](https://arxiv.org/abs/2410.03893).

## Appendix: reproducing

- **The games figure:** `python tools/imitation_scale/fig_games.py`.
- **The per-game rates of §2.2** come from #2b's `build_stats.json` and tables (docs/011 §1). They
  also come from a 1-in-40 pass over the replay file with
  `draftzero.gameplay.replay.iter_games(every=40, start=3)`, counting each opponent turn with
  attackers and the player's creatures in play before it.
- **The value pilot (§4.1),** on the laptop with #2b's tables (`data/gameplay/imitation` in the
  `exp2-run2` worktree) and the v0.2 XMage bundle:

  ```bash
  export MZ_XMAGE_DIR=<v0.2 bundle>/xmage MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv
  python tools/imitation_scale/value_pilot.py heuristic --tables <run2>/data/gameplay/imitation
  python tools/imitation_scale/value_pilot.py score --tables <run2>/data/gameplay/imitation \
    --net 2b=<HF 2026-09-27_03-42-21/models/FDN_exp2_imit/ver1/gen0.pt.gz> \
    --net 2a=<HF 2026-09-27_01-59-54/models/FDN_exp2/ver1/gen18.pt.gz>
  python tools/imitation_scale/value_pilot.py figure
  ```

  The heuristic stage takes under a minute with 3 workers. `score` also runs the mid-turn check:
  both networks on `replay_priority_test.h5`, restricted to the clean games.
