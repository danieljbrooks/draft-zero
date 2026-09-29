# Search benchmark, first experiment: results

**Status: partial draft, 2026-09-29 05:15 PDT.** The plan is [docs/012](012-search-benchmark.md) §2,
and this report follows its order.

- **Done:** all 16 offline runs, 15 of 16 network runs, E0, E1 offline and E2b offline.
- **Still running:** IS-MCTS with the network at 3,000 simulations; E2b with the network (4 of 10
  runs); E1's network spot check; the extra-budget follow-ups (§7).

Numbers may move by a point or two, and conclusions marked *tentative* may change.

## Summary

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/016-frontier-dark.png">
  <img alt="Two panels, offline search and the trained network, each plotting agreement with top 17lands players (A_set, 35 to 75 percent) against pod-seconds per decision on a log axis. Every search method lies between 47 and 54 percent, a few points above the rule heuristic (47 percent) and chance (44 percent). The fair methods (PIMC with 1 and 4 worlds, IS-MCTS; colored) sit at or above the clairvoyant search (gray, hollow). Lines rise a few points from 100 to 1,000 simulations and flatten after. On the network panel, the network's policy with no search sits far above everything at 69 percent." src="img/016-frontier-light.png">
</picture>

- **Every search agrees with top players about equally, and not much more than a simple
  heuristic.** Across four methods, four budgets and both evaluators, A_set ranges from 47% to 54%,
  macro-averaged over the four decision types. The references: chance 43.7%, the rule heuristic
  46.9%. The best configurations are offline PIMC with 1 world and offline IS-MCTS at 3,000
  simulations (both 54.2%), and PIMC with 1 world on the network at 3,000 (53.3%).
- **Fairness costs nothing on this benchmark.** No fair method is significantly behind today's
  clairvoyant search at any budget, and PIMC with 1 world is ahead by 2–3 points with the network
  (not significant). This is expected: on 17lands positions the clairvoyant search reads a
  *guessed* opponent hand (docs/012 §2.4), so peeking buys nothing here. The leak test shows it
  peeks where it can (next point).
- **The hidden-information test separates the methods cleanly.** Clairvoyant MCTS fails five of
  six probe pairs. On the counterspell pair it drops Serra Angel's value by 0.37 when the opponent
  really holds Refute. PIMC with 1 and 4 worlds and IS-MCTS pass all six, with identical searches in
  both worlds of every pair (§3).
- **More search helps a little, then flattens.** From 100 to 3,000 simulations, clairvoyant MCTS and
  PIMC with 1 world gain 6–7 points offline (significant). From 1,000 to 3,000 they gain 1–3, mostly not
  significant. Almost all of the gain is on holds: searches cast spells where top players held back,
  and do so less with more budget (§4.2).
- **One world is enough; four don't help.** PIMC with 4 worlds is never better than with 1, and
  offline at 1,000 it is 2.5 points worse (CI −5.2 to +0.5). Each of its trees gets a quarter of
  the budget.
- **IS-MCTS matches PIMC offline at about 3× the cost, and trails it with the network.** Offline at
  3,000 it ties PIMC with 1 world (54.2% each) at 3.6 against 1.3 pod-seconds per decision. With
  the network at 1,000 it is 4.8 points behind PIMC with 1 world (CI −7.7 to −1.7).
- **The trained network adds nothing to search, but its policy alone beats every search.** Network
  and offline search agree within ±3 points at every setting (§4.3). With priors off, the search
  reads only #2a's value head, which docs/014 found weak. The network's *policy* with no search
  scores 68.6%. That is inflated by passing, since it holds 88% of the time; its strict score is
  53.9%. Using the policy as priors inside the search is the obvious next step, and it is running
  (§7).
- **The backprop discount doesn't matter for agreement.** No arm beats the default 0.99 per ply.
  0.9 per ply costs clairvoyant MCTS 3.2 points offline (significant). Per-action and per-turn
  discounts are within noise (§5).
- **Compute:** offline search costs 0.02–3.6 pod-seconds per decision, network search 0.14–4.3. The
  network runs are GPU-bound at about 550–700 evaluations per second across the pod. The
  experiment has cost about $7 in pods so far, including calibration.
- **The proxy's weak spot showed up.** Differences between search methods are 1–5 points,
  comparable to the benchmark's resolution. The one large effect in the data, the policy's 68.6%,
  comes from a style of play (passing), not from strength. docs/012 §2.10 warned about both.

---

## 1. What ran, and how it differs from the plan

The first experiment compares four search methods at four budgets with two evaluators, on 1,000
held-out decisions by top 17lands players, and tests each method for hidden-information leaks.

| | Planned (docs/012 §2) | Ran |
|---|---|---|
| Methods | clairvoyant MCTS; PIMC with 1 and 4 worlds; IS-MCTS | the same |
| Budgets | 100, 300, 1,000, 3,000 simulations | the same |
| Evaluators | offline search; experiment #2's network | offline search; #2a's gen 18 network |
| Decisions | 1,000 test + 300 dev, six types | 1,000 test + 300 dev, **four types** (§1.1) |
| Leak test | about two dozen probe scenarios | six probe pairs, 16 seeds per world (§3) |
| Discount sweep (E2b) | 1.0, 0.95, 0.9 per ply; 0.95-matched per action and per turn | the same |
| Compute | pod-seconds per decision on the RTX 3090 pod | the same (§1.4) |

### 1.1 The decisions (sb-v1)

Built by `tools/search_bench/items.py` from 1,874 held-out games by top players
(`user_game_win_rate_bucket` ≥ 0.60 and at least 100 games), sampled every 36th row of the FDN
replay file. Held out means no row of any imitation table (docs/008 §7.3 and experiment #2b used
the same 16,483 games) and no mirrored partner of one. At most two decisions per game (a mirrored
pair counts as one game), turns 3–12, fidelity tiers T0 and T1.

| Type | What | Label | Test | Dev | Options (mean) | Chance |
|---|---|---|---|---|---|---|
| **Spell** | the first main-phase priority after the human's land drop, in a turn in which they cast a spell or activated an ability | that turn's casts and activations legal here, plus Pass when the human attacked | 375 | 110 | 3.99 | 48% |
| **Hold** | the same decision in a turn in which the human cast and activated nothing, with a spell castable | Pass | 125 | 40 | 3.40 | 33% |
| **Attack** | the first "attack with X?" question, in a turn in which the human cast nothing | exact | 300 | 90 | 2.00 | 50% |
| **Block** | the first "what does X block?" question in the opponent's next turn | exact, a unique pairing | 200 | 60 | 2.37 | 44% |

- **Two of docs/012's six types are left out:** mid-turn priority and spell targets. The
  pipeline can't yet rebuild a mid-turn state as a search root, and 17lands records no target
  labels that the current resolver can match. Their 225 test slots went to the other four types.
- **The land drop is played first.** A new bridge option (`preLand`) plays the human's land at the
  first main-phase priority, then the search decides. Without it, "play a land" would match the
  human's label almost every time.
- **Attack decisions come from turns in which the human cast nothing,** so the pre-combat state
  is the turn start plus the land. That selects quieter turns.
- **Holds are rare in top players' games:** 3,001 hold candidates failed because the human cast
  something. The 125 test holds came from the last games scanned.
- **Each decision carries its worlds** (docs/012 §2.4). `real` is the position with the opponent's
  hand and deck filled from the belief model, with the item's own seed: what clairvoyant MCTS
  searches. `worlds` are eight more belief samples with the search seed. PIMC with 1 world uses
  the first, PIMC with 4 the first four, IS-MCTS all eight. The belief model leaves out the game's
  own drafts and a mirrored partner's.

### 1.2 The search code

The four methods run through one new driver, `mage.player.ai.BenchSearch` in `java/mzbridge`,
reached by a new `bench` op. It uses MageZero's nodes only to step the engine and list options,
and keeps its own statistics. That's how the benchmark gets:

- **Fresh simulations only:** a fresh tree for every decision, and the budget counts simulations
  run, not root visits.
- **The discount's unit** (E2b): per ply, per logical action (only edges out of priority
  decisions), or per turn.
- **No whole-tree walks:** MageZero walks the tree twice per iteration for its node and depth
  limits, and searches for duplicate states even with pruning off (docs/012 §2.9).
- **Synchronous network evaluation:** no virtual loss, so its sign problem at opponent nodes
  (docs/012 §2.9) can't bias anything. A seeded offline search is deterministic.

Everything else follows ComputerPlayerMCTS2: PUCT with c = 1, unvisited options valued 0,
uniform priors (priors off, as in experiment #2), the single-option shortcut, offline leaves
scored by `GameStateEvaluator3` at priority decisions with micro decisions inheriting their
parent's score, and the final choice by visits.

- **Clairvoyant MCTS:** one tree on the real world.
- **PIMC:** one tree per belief world, the budget split evenly, root options merged by label, and
  the choice by visits summed over worlds.
- **IS-MCTS** (single-observer, Cowling, Powley and Whitehouse 2012):
  - one tree whose edges are world-independent keys (an ability's text and source name; a
    target's name, state, zone and controller);
  - every iteration picks one of the eight belief worlds, re-deals it (the opponent's hand redrawn
    from that world's unseen cards, both libraries shuffled, with the search's own random stream)
    and replays its path from the root in that world, following only options legal there;
  - selection uses availability counts; a node is scored once, in the world of the iteration that
    creates it; opponent nodes are shared across worlds, as published.

  So every iteration sees a fresh hand and fresh library orders, and deck compositions come from
  eight belief decks. It is the vanilla form docs/012 chose, with a finite pool of decks.

### 1.3 The network

Experiment #2a's final checkpoint (gen 18, `FDN_exp2`, SHA-256 `ac9b2d4f…`), with the opponent's
hand hidden from the encoder (`perfectInfo` false, as in training) and priors off, so the search
reads only the value head. It is served by `tools/search_bench/value_server.py`: MageZero v0.2's
model, forward pass and batching, returning only the value. MageZero's own server returns all four
1,024-wide policy heads as Python lists for every state. On 60 searches the two servers' values
agreed within 0.0005 (fp16), and the choices matched on 59. The policy references use MageZero's
own server.

### 1.4 Compute

- **Pods:** two Secure RTX 3090 pods, each with a 31.1-core quota and 116 GB, at $0.50/hr: the
  reference pod of docs/012 §2.6. Pod 1 ran the network runs at budgets 100–3,000. Pod 2 ran the
  offline half and E1, then E2b with the network.
- **Load:** 30 bridge workers (one JVM, one search each) for offline search. 32 workers and four
  value-server replicas sharing the GPU for network search. Every search is fresh, so the pod-seconds
  per decision are a batch's wall-clock at full load divided by its 1,000 decisions.
- **The network runs are GPU-bound.** At 32 workers the GPU is 96–99% busy, at about 550–700
  evaluations per second across the pod. A single server replica was worse: 35% GPU and 0.88
  against 0.57 pod-seconds per decision at budget 300, because Python's GIL throttles one process.
  About 13 ms of each network simulation is inference latency and 6 ms engine work, so many
  workers are needed to fill the GPU.
- **Offline search is CPU-bound** at about 2,600–3,000 simulations per second across the pod for
  the tree methods. That is about 5× docs/012's estimate of 550, which came from self-play with its
  whole-tree walks.
- **IS-MCTS pays for its replay:** 6.3 engine steps per simulation at 100 simulations, 9.8 at
  1,000 and 11.5 at 3,000, against 1.02 for the tree methods. Per decision, that is 2.8–3.3× the
  tree methods' cost offline, close to docs/012's guess of 3×.
- **Warm timing.** Each batch's first run includes JVM start-up, about 45 pod-seconds in all. That
  matters only for clairvoyant MCTS offline at 100 simulations, which was re-timed warm: 0.027
  pod-seconds per decision, against 0.074 cold.

## 2. E0: references and noise

All on the 1,000 test decisions. A_set is macro-averaged over the four decision types; "strict"
drops the lenient Pass on spell decisions (§1.1).

| Reference | A_set | 95% CI | Strict | Spell | Hold | Attack | Block |
|---|---|---|---|---|---|---|---|
| Chance (uniform over the distinct legal options) | 43.7% | | | 48.3% | 32.6% | 50.0% | 44.0% |
| Rule heuristic, no search | 46.9% | | | | | | |
| #2a's gen 18 network: its policy, no search | **68.6%** | 65.7–71.6 | 53.9% | 59% | **88%** | 63% | 64% |
| #2b's starting network, pretrained on human decisions: its policy, no search | 54.7% | 51.6–58.2 | 54.4% | 68% | 34% | 75% | 43% |

- **The rule heuristic barely beats chance.** It is "cast the biggest spell; attack when power is
  at least the best blocker's toughness; block when the blocker kills the attacker and survives".
- **#2a's policy scores highest of anything in this report, mostly by passing.** It passes on 88%
  of holds, and Pass counts on spell decisions where the human attacked. Its strict score, 53.9%,
  is about where the searches are. Its policy heads learned from self-play searches' visit counts
  (300 simulations with tree reuse), yet they prefer Pass far more than any search here does. Why
  isn't clear from this data. With priors off, the search never reads them.
- **#2b's human-pretrained start agrees best on spells (68%) and attacks (75%),** the two types its
  pretraining covered (turn-start plays and attacks, docs/011). It is near chance on holds and
  blocks. It is the floor under the ceiling docs/012 §2.3 asked for, but only for spells and
  attacks.
- **Test–retest noise is small.** PIMC with 4 worlds at 1,000 simulations, run again with a second
  seed, changed its A_set by +0.2 points offline (95% CI −0.9 to +1.1) and −0.5 with the network
  (−1.8 to +0.9). Paired differences between methods of about 2–3 points are detectable, as docs/012
  §2.3 estimated. The unpaired CIs in the tables (about ±3.3 points) are much wider than the paired
  ones, so methods are compared paired (§4).
- **Not run:** XMage's MAD AI. `MadProbe` evaluates a position without advancing it, and the
  benchmark's positions start at the end of the previous turn, so it needs an advance step first.
- **Plies per logical action and per turn,** measured on clairvoyant MCTS's 1,000-simulation trees,
  set E2b's matched discounts. Offline: 1.43 plies per action and 16.2 per turn, so 0.95 per ply
  matches 0.930 per action and 0.435 per turn. With the network: 1.43 and 15.8, so 0.930 and 0.446.

## 3. E1: the hidden-information test

**Clairvoyant MCTS fails. PIMC with 1 and 4 worlds and IS-MCTS pass, with every difference exactly
zero.** `tools/search_bench/leak.py`, offline search, 3,000 simulations, 16 seeds per world, seeds
paired across the two worlds of a pair.

Each pair is two positions that look the same to the deciding player and differ only in cards it
can't see. They are docs/009's two positions plus extreme versions, a decklist pair and a canary:

| Pair | World X / world Y | Clairvoyant MCTS: worst option's ΔQ (X − Y), 95% CI | Its choices, X / Y | Verdict |
|---|---|---|---|---|
| counterspell | the opponent holds Refute + Island / Island + Island | Cast Serra Angel −0.37 [−0.45, −0.29] | Pass 10 of 16 / Cast 16 of 16 | fails |
| counterspell, extreme | Refute ×3 / Island ×3 | −0.39 [−0.44, −0.34] | Pass 16 / Cast 16 | fails |
| cantrip | the searcher's own next draw is Llanowar Elves / a Plains | Cast Helpful Hunter +0.073 [+0.067, +0.079] | Cast 16 / Cast 15 | passes the rule |
| cantrip, extreme | its next five draws are Elves / Plains | +0.086 [+0.070, +0.102] | Cast 16 / Cast 14 | fails |
| decklist | the opponent's hidden hand is dealt from a list with four counterspells / none | −0.10 [−0.20, −0.00] | Pass 3 / Pass 2 | fails |
| canary | the opponent holds Counterspell ×2, a card outside FDN that no belief can deal / Island ×2 | −0.31 [−0.41, −0.21]; the canary reached all 16 of its searches | Pass 14 / Cast 16 | fails |

The rule (docs/012 §2.5): a pair passes when every option's paired ΔQ has a 95% CI inside ±0.1 and
the chosen actions don't differ (a permutation test, p > 0.01).

- **Clairvoyant MCTS fails five of the six pairs.** It fails as docs/009 found: it plays around a
  counterspell only when one is really there, and values a cantrip by the card it will draw. The
  realistic cantrip pair shows a real leak (+0.073, far from zero) that is inside the ±0.1 window,
  so it passes the rule. Its extreme version fails, which is what the extreme versions are for.
- **The fair methods give identical searches in both worlds of every pair,** ΔQ 0.000 and the same
  choice in all 16 seeds. That is by construction, and it is what the test checks: their worlds are
  drawn from a public view of the position (the opponent's hand as a count, its decklist unknown,
  the searcher's library top unknown), so the same seed gives the same worlds, and offline search is
  deterministic given its seed. No canary reached any of their worlds.
- **What it doesn't test.** There are no known-card controls. The public view drops *every*
  hidden card, including ones the player legitimately knows, such as a card it saw bounced to the
  opponent's hand or a card it scried to the top. So the fair methods also ignore information they
  are entitled to. That costs quality, not fairness. Nothing reuses trees, so docs/012's tree-reuse
  probe doesn't apply.
- **The network spot check** runs the counterspell and cantrip pairs with the network
  *(pending)*.

## 4. E2: method × budget

### 4.1 The grid

A_set, macro-averaged over the four types (95% CI about ±3.3 points; paired differences are much
tighter). Pod-seconds per decision in brackets. † still running.

| | 100 | 300 | 1,000 | 3,000 |
|---|---|---|---|---|
| **Offline search** | | | | |
| Clairvoyant MCTS (fails E1) | 46.9% (0.027) | 50.5% (0.08) | 52.7% (0.34) | 53.5% (1.42) |
| PIMC, 1 world | 48.1% (0.026) | 50.9% (0.08) | 52.7% (0.34) | **54.2%** (1.27) |
| PIMC, 4 worlds | 50.3% (0.023) | 50.4% (0.07) | 50.2% (0.32) | 52.9% (1.18) |
| IS-MCTS | 49.5% (0.09) | 49.1% (0.28) | 52.1% (1.04) | **54.2%** (3.59) |
| **Network (#2a gen 18, priors off)** | | | | |
| Clairvoyant MCTS (fails E1) | 48.1% (0.15) | 47.6% (0.42) | 49.7% (1.40) | 50.9% (4.27) |
| PIMC, 1 world | 48.7% (0.14) | 49.2% (0.41) | 52.8% (1.39) | **53.3%** (4.21) |
| PIMC, 4 worlds | 51.4% (0.14) | 50.0% (0.41) | 50.4% (1.38) | 51.5% (4.16) |
| IS-MCTS | 47.4% (0.18) | 47.3% (0.49) | 48.1% (1.66) | † |

Paired differences (A − B, points, 95% CI over games):

| Comparison | Offline | Network |
|---|---|---|
| PIMC 1 − clairvoyant, 3,000 | +0.6 (−2.2, +3.5) | +2.4 (−0.5, +5.3) |
| IS-MCTS − clairvoyant, 3,000 | +0.7 (−2.0, +3.5) | † |
| PIMC 4 − PIMC 1, 1,000 | −2.5 (−5.2, +0.5) | −2.4 (−5.2, +0.5) |
| PIMC 4 − PIMC 1, 3,000 | −1.2 (−4.0, +1.5) | −1.7 (−5.1, +1.3) |
| IS-MCTS − PIMC 1, 1,000 | −0.6 (−3.1, +2.0) | **−4.8 (−7.7, −1.7)** |
| IS-MCTS − PIMC 1, 3,000 | 0.0 (−3.0, +3.0) | † |
| PIMC 1: 3,000 − 100 | **+6.1 (+2.3, +9.6)** | **+4.6 (+0.9, +8.2)** |
| PIMC 1: 3,000 − 1,000 | +1.4 (−1.1, +4.2) | +0.4 (−2.2, +3.2) |
| Clairvoyant: 3,000 − 100 | **+6.7 (+3.1, +10.6)** | +2.7 (−0.8, +6.4) |
| IS-MCTS: 3,000 − 1,000 | +2.0 (0.0, +4.1) | † |

### 4.2 By decision type

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/016-by-type-dark.png">
  <img alt="Four panels, one per decision type, plotting A_set against simulations for each method, offline solid and network dashed, with chance as a gray dashed line. Spell: 60 to 69 percent, flat. Hold: 14 to 41 percent, rising with budget from below chance (33 percent) to above it. Attack: 53 to 64 percent. Block: 46 to 62 percent." src="img/016-by-type-light.png">
</picture>

- **Spells (375):** 58–69%, against chance at 48%. More budget doesn't help: offline methods sit at
  64–69% from 100 simulations on. The network searches are 3–6 points lower.
- **Holds (125): where budget matters.** At 100 simulations the searches pass on only 14–29% of the
  decisions where the human held, below chance (33%). They cast whatever they can. At 3,000,
  clairvoyant MCTS and PIMC with 1 world reach 34–41%. PIMC with 4 worlds holds least, 19–32% at
  every budget. Top players hold far more often than any search.
- **Attacks (300):** 53–64%, against chance at 50%, and flat in budget. Offline search is a few
  points better than the network here. Offline, a micro decision inherits its parent's score, so
  attacks resolve only once combat reaches a priority decision (docs/012 §2.1).
- **Blocks (200):** 46–62%, against chance at 44%. The network's PIMC with 4 worlds at 100
  simulations is the best cell (62%). Most other settings are 50–55%.

### 4.3 Offline against network

Network minus offline, same method and budget: between −4.1 and +1.3 points, and only one of 15
comparisons is significant (IS-MCTS at 1,000, −4.1). With priors off, #2a's value head makes the
search no better than the hand-written heuristic. That fits docs/014, where #2a never beat raw
search in play and its value head improved slowly.

## 5. E2b: the backprop discount

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/016-discount-dark.png">
  <img alt="A_set at 1,000 simulations for six discount arms (1.0, 0.99, 0.95, 0.9 per ply; 0.95-matched per action; 0.95-matched per turn), for clairvoyant MCTS and PIMC with 4 worlds, offline (filled) and network (hollow). Everything lies between 49 and 53 percent with overlapping CIs; 0.9 per ply is lowest." src="img/016-discount-light.png">
</picture>

Paired differences from the default 0.99 per ply, at 1,000 simulations (points, 95% CI). The
network's PIMC arms and per-action and per-turn arms are still running.

| Arm | Clairvoyant, offline | PIMC 4, offline | Clairvoyant, network |
|---|---|---|---|
| 1.0 per ply (no discount) | −0.3 (−1.5, +0.9) | −0.1 (−1.1, +0.7) | +0.8 (−0.6, +2.4) |
| 0.95 per ply | −0.8 (−2.8, +1.3) | −0.5 (−2.3, +1.3) | 0.0 (−2.3, +2.3) |
| 0.9 per ply | **−3.2 (−5.6, −0.5)** | −1.2 (−3.5, +0.8) | −1.0 (−3.9, +1.9) |
| 0.930 per action (matches 0.95 per ply) | +0.3 (−2.0, +2.9) | +1.3 (−0.8, +3.3) | † |
| 0.435 per turn (matches 0.95 per ply) | −0.2 (−3.0, +2.6) | +2.0 (−0.4, +4.3) | † |

- **No arm improves agreement.** The only significant effect is a loss: 0.9 per ply costs
  clairvoyant MCTS 3.2 points offline, mostly on attacks (61% → 53%).
- **The prediction isn't borne out in agreement.** docs/012 predicted that the discount would help
  clairvoyant MCTS more than PIMC, because the stalling it fixes needs knowledge of the top card.
  Agreement can't see that kind of stalling (§6).
- **The unit barely matters.** Per action and per turn are within noise of per ply at matched
  strength. The per-turn discount lifts PIMC with 4 worlds on attacks (60% → 70%) but not overall.
- **The free check:** a stronger per-ply discount shifts little search away from options that take
  sub-decisions (a spell's target, the next attacker's question). Among decisions offering both
  kinds, the visit share on multi-step options is 0.394 at 1.0 per ply and 0.388 at 0.9 per ply for
  clairvoyant MCTS, and 0.399 to 0.386 for PIMC with 4 worlds. The per-turn discount moves it most
  (0.383 and 0.380).

## 6. Discussion (tentative)

- **Is agreement measuring anything here?** Partly. Search beats chance by 5–10 points and the rule
  heuristic by up to 7, and more budget helps where it should (holds). But the spread between
  sensible methods, 1–5 points, is close to the benchmark's resolution. As in Allie's chess
  results (docs/012 §2.10), agreement may be separating broken from decent without separating
  decent from good. docs/012's first follow-up, validation by play (E9), is what would settle
  which of these differences matter.
- **What is worth carrying into self-play, on this evidence:**
  - PIMC with 1 world costs the same as today's clairvoyant search and agrees at least as well. It
    is the cheapest way to stop the search reading hidden cards.
  - PIMC with 4 worlds buys nothing at these budgets.
  - IS-MCTS isn't worth its 3× cost on agreement. Its value would have to show in play, for
    example in bluffs and in playing around tricks, which agreement doesn't test.
- **The network's policy is the strongest single signal.** With priors off, the search ignores it.
  Whether it helps as priors is the most useful follow-up (§7). Priors raise human agreement for
  reasons that are partly style, though. The policy's Pass habit shows how far style alone can move
  the score.
- **The benchmark's own limits:**
  - two of the six planned decision types are missing;
  - the hold type is small (125) and depends on the human's timing, which 17lands records only per
    turn;
  - the lenient Pass on spell decisions rewards passive play. The strict score is 0.5–2 points
    lower for the searches, but 15 points lower for the policy.

## 7. Follow-ups with the extra budget *(running)*

- **The network's policy as priors** (MageZero's setPriors: softmax at temperature 1.5, plus 0.1 for
  anything but Pass), for all four methods at 300 and 1,000 simulations, on MageZero's own server,
  which returns the policy heads. A 40-decision dev test moved PIMC with 4 worlds at 100
  simulations from 39% to 57%. Too small to trust, but the strongest lead. About $1.75.
- **PIMC with 1 world, offline, at 10,000 simulations:** docs/012's optional top budget, to see
  where the curve flattens. About $0.65.

## Appendix: reproducing

The code is on branch `search-benchmark`:

- `java/mzbridge/src/mage/player/ai/BenchSearch.java`: the search driver.
- `java/mzbridge/src/org/draftzero/mzbridge/Bench.java`: the `bench` op.
- `tools/search_bench/`: the decision builder, runner, leak test, value server, analysis and pod
  plans.

```bash
bash deploy/search_bench_setup.sh                       # on a pod, with HF_TOKEN
python tools/search_bench/items.py build --every 36     # sb-v1, about 12 minutes on a pod
bash tools/search_bench/plan_offline.sh                 # E1, E2 offline, E0, E2b offline
bash tools/search_bench/start_servers.sh models/FDN_exp2/ver1/gen18.pt.gz 4
bash tools/search_bench/plan_net_a.sh                   # E2 network 100-1,000, E0
python tools/search_bench/analyze.py --runs runs/search_bench/e2_* ... --plot docs/img/016-frontier
```

The decision set and the per-decision outputs are git-ignored: they hold mirrored-pair rows.
