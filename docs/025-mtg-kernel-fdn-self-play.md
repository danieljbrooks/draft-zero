# Self-play on mtg-kernel with FDN limited decks

*Started Tuesday 6 October 2026, 10:05 AM PT, at Dan's request, after mtg-kernel's maintainer reported that our two
FDN fixture decks play complete games
([mtg-kernel#110](https://github.com/jackmaiorino/mtg-kernel/issues/110#issuecomment-6017220636)). Follows
[docs/015](015-rules-engine-comparison.md), which compared the engines. All times are Pacific. **The code is on the
[`test-mtg-kernel`](https://github.com/danieljbrooks/draft-zero/tree/test-mtg-kernel) branch** (folder `mtgkernel/`),
not on main.*

*Status, Tuesday 6 October, 8:45 PM PT: done; every run finished and the rented machines are off ($2.94 of the $10
budget).*

**For reviewers:**

1. Should we offer our two mtg-kernel patches upstream and report its Ward halt (§8)?
2. Does the speed gap (§3), together with the card-pool blocker (§8), justify planning self-play training on
   mtg-kernel once more FDN cards land?

## Summary

**The question:** how far can we get training on mtg-kernel today, using only the FDN cards it already implements?
Play games, measure games per hour, train or port a model, evaluate it against baseline bots, and compute 17lands
card statistics as we did on XMage.

**What we found:**

- **Self-play games cost hundreds of times less than on XMage.** On one $0.22/hour rented machine, mtg-kernel plays
  3.6 million random games an hour, 46,800 of MageZero's heuristic search bot at 100 simulations a decision, and
  32,800 of our network-guided search at 100. For network-guided search that is about 360 times XMage's games per
  machine-hour and 800 times its games per dollar (docs/020); allowing for XMage's slower search method and longer
  games, roughly 100 times (§3).
- **Only 38 FDN cards are playable, all white, blue or green but one.** No real 17lands deck except our two fixture
  decks can be built from them, so we generated 120 two-colour decks.
- **A network trained from random weights by expert iteration beats the heuristic bot.** Its first generation
  learned from 6,000 games of the heuristic search itself (no human data); two more generations learned from its
  own search's games, 12,000 games in all, about 2 machine-hours. Its policy alone, with no search, won 63.7% of 998
  paired games against the heuristic search at 100 simulations; guiding a search, it won 71.4% of 500 at equal
  budget (100 simulations) and 70.2% of 248 at 1,000. Generation 1 improved on generation 0 everywhere; generation
  2 improved the policy alone again, but searched no better than generation 1.
- **Its card ratings resemble 17lands' when it searches.** In the self-play of generation 1's network-guided search,
  card win rates rank the 37 cards of our decks like 17lands' at a Spearman rank correlation of 0.30 (1 = the same
  order, 0 = unrelated), and 0.47 after removing deck and colour strength; random play gives −0.23. Five of 17lands'
  eight best cards are among the bot's eight best. It underrates cheap combat tricks: cards whose value depends on
  timing, as docs/019's XMage bots underrated burn and removal.
- **We did not port a draft-zero network.** Their inputs are XMage's own feature strings: a port is 2–5 days for the
  graph network and up to 8 for the others (§6); a new network was faster to train.
- **Blockers** (§8): the card pool; no trainer, search or network in mtg-kernel accepts 40-card decks yet, so we
  built our own on its rules engine, with two small patches (visibility and speed, no rule changes); its observation
  layer halts about 1 game in 400–500 on Ward when a bot plays without search; and our search assumes open
  decklists.

**Terms used throughout:**

- **Fixture decks:** two real 17lands top-player decks (blue-green and white-green, our "pair A" from docs/015) that
  mtg-kernel's maintainer implemented every card of.
- **Simulation:** one step of the tree search: try a move, look one decision ahead, score the position.
- **Heuristic search (`mcts:N`):** the baseline bot: draft-zero's port of MageZero's tree search, N simulations a
  decision, scoring positions with MageZero's hand-written evaluator (GameStateEvaluator3: life, cards in hand,
  permanents). draft-zero calls it heuristic@N.
- **Policy, value:** a network's probability for each legal move, and its estimate of the chance of winning.
- **Expert iteration:** a search bot plays itself, a network learns to imitate the search's choices and estimates,
  then the next search uses that network; repeat (AlphaZero's loop).
- **Paired games:** every deck pair is played twice with the two bots swapping seats and the same shuffle, which
  removes much of the luck of the draw (variance 30–60% lower in experiment #4). Games that end in an engine error
  are left out (at most 2 a row below).
- **GIH WR:** 17lands' games-in-hand win rate: how often a deck wins the games in which a card was drawn.

## 1. What mtg-kernel can play today

**Cards.** On mtg-kernel's main branch (commit `a4e1474`, 6 October), 43 of the 286 FDN card names have full engine
support: the 5 basic lands and 38 others. 36 of the 38 are the non-basic cards of the two fixture decks; the other
two, Tolarian Terror and Goblin Bushwhacker, are cards mtg-kernel already had for Pauper that are also in FDN.
Ajani, Caller of the Pride is partly done. The maintainer's next batch (PR #153, open) adds 7 creatures; we did not
use it.

| Colour | Playable cards |
|---|---|
| White | 11 |
| Blue | 10 |
| Green | 12 |
| Gold (WG, UG) | 2 |
| Red | 1 (Goblin Bushwhacker, a Special Guest card) |
| Black | 0 |
| Lands | 2 gainlands (Blossoming Sands, Thornwood Falls) + 5 basics |

That makes a three-colour mini-format: white, blue and green. Of the 31,516 top-player 17lands decks in draft-zero,
only our two fixture decks can be played; the next closest is one card short (Giant Growth).

**Decks.** So we generated decks from the playable cards (`mtgkernel/py/gen_decks.py`): 40 for each of the three
colour pairs (WG, UG, WU), each 23 spells and 17 lands. Spells are drawn at random from the pair's cards, each card
weighted by its copy limit (3 copies of a common, 2 of an uncommon, 1 of a rare), under limits of 14–17 creatures,
at least 5 cards of mana value 2 or less, at most 4 of 5 or more, and at least 5 cards of each colour. Commons end up
in 85% of on-colour decks and rares in 42%. Goblin Bushwhacker, the only red card, is left out. **What matters for
the card statistics** is that a card's presence doesn't depend on how good the rest of the deck is: in 17lands'
data, good cards end up in good drafters' decks. 30 of the 120 decks (10 a colour pair) are held out for evaluation,
together with the two fixture decks.

**Rules.** We use the most complete rules mtg-kernel offers for custom decks (its `kernel_limited_env` "schema 4"):
every priority window, London mulligans, and Foundations combat (damage assignment and trample). In a code review,
our harness and mtg-kernel's own JSON server played 150 games in lockstep and agreed on all 20,287 decisions (the
check script lives in the review's scratch space, not the repo).

## 2. How the games run

mtg-kernel ships no trainer, search or network that accepts 40-card decks: its Python trainer plays only its fixed
Pauper decks, against a random opponent; its native trainers require 60-card decks; and its searches run only on
its Pauper session type and fail on Limited decisions (mulligans, damage assignment). The maintainer lists "fair
Limited search" and "DraftZero integration" as milestones 6 and 7 of #110. So we built a small Rust harness, `dzk`,
on top of mtg-kernel's rules engine:

- **Game wrapper.** It drives mtg-kernel's own Limited game session in-process (no JSON pipe), with the same
  arguments its JSON server uses. Mana abilities are hidden from the bots whenever another move exists, because
  casting pays mana automatically. After that, about 80% of the engine's decisions have only one move and are played
  without asking the bot.
- **Two small patches to mtg-kernel** (`mtgkernel/patches/`), changing no rules or cards:
  1. Make the custom-deck session constructor public; give mutable access to the game state (which upstream
     deliberately withholds) and read access to the policy surface, so a search can re-deal hidden cards on a copy;
     make the re-seeding of a copy's future shuffles public.
  2. An opt-in switch that skips hashing the full observation at every step, and a step call that doesn't
     deep-copy the observation and every legal action. In a profile, the hash and the copies took 77–81% of the CPU
     (the hash alone about two thirds). The observation is still built. Games are byte-identical with and without
     the switch.
- **Bots:** `random`; the heuristic search `mcts:N` (PUCT with uniform priors, a 0.99 discount per step, the
  most-visited move, as draft-zero's `BenchSearch`); and the network bots of §5.
- **Hidden information.** The search plays on a re-dealt copy of the game: PIMC (perfect-information Monte Carlo)
  with one world. The opponent's unseen cards and both libraries are re-shuffled from what the searcher knows, and
  future shuffles are re-seeded. **It assumes open decklists:** the searcher knows which 40 cards are in the
  opponent's deck, but not which are in hand or the library's order.
- **Records** use draft-zero's `games.jsonl` format, so `gih.py` and `card_stats.py` read them, with the opening hand
  counted the way 17lands does (the last 7-card deal, before London bottoming).

Five adversarial code reviews (three of the harness, two of the network code) found real problems, all fixed before
the results below were kept. The worst: the search, when behind, steered into moves that trigger mtg-kernel's Ward
halt (§8), and those games were then dropped as errors, inflating the losing searcher's score; opening hands missed
the cards bottomed by a mulligan; the trainer exported its first-epoch network and let the value head memorize
results (§5); and a resumed run could mix two networks' games.

## 3. Speed: games per hour and per dollar

![Horizontal bars on a log scale, self-play games per dollar: on mtg-kernel, random moves 16.2 million, generation 0's policy alone 16.9 million, heuristic search at 100 simulations 213,000, network-guided search at 100 simulations 149,000 and at 1,000 simulations 14,500; on XMage (docs/020), network-guided search at 100 simulations 183 and at 1,000 simulations 18.](img/025-games-per-dollar-light.png)

*Figure 1. Self-play games per dollar: mtg-kernel on a Community RTX 3090 pod at $0.22/hour; XMage on a Secure RTX
3090 pod at $0.50/hour with IS-MCTS (docs/020). The network rows use generation 0's network.*

*One rented RTX 3090 machine (Community cloud, $0.22/hour) whose container gets 13.6 vCPUs of quota on an AMD
Threadripper PRO 5975WX, 14 game threads; self-play on the 62 evaluation deck pairs. Whole-machine rates are steady
state (games still running at the deadline excluded).*

| Bot | Games/hour, 1 thread | Games/hour, whole machine | Games per dollar | Simulations/s, whole machine |
|---|---|---|---|---|
| Random moves | 275,000 | 3,560,000 | 16,200,000 | – |
| Random moves through mtg-kernel's JSON interface, Python client (fixture pair, 13 processes) | 16,600 | 214,000 | 973,000 | – |
| Heuristic search, 100 simulations | 4,190 | 46,800 | 213,000 | 129,000 |
| Heuristic search, 300 simulations | 1,320 | 15,600 | 70,900 | 129,000 |
| Heuristic search, 1,000 simulations | 478 | 4,540 | 20,700 | 130,000 |
| Generation 0's policy alone (§5) | 281,000 | 3,710,000 | 16,900,000 | – |
| Generation 0's network-guided search, 100 simulations | 2,540 | 32,800 | 149,000 | 99,800 |
| Generation 0's network-guided search, 1,000 simulations | 284 | 3,180 | 14,500 | 99,900 |

**Against XMage.** docs/020's best machine for XMage self-play (a Secure $0.50 RTX 3090 pod, 27 workers, IS-MCTS
with experiment #4's MLP) played 92 games an hour at 100 simulations and 9 at 1,000: 183 and 18 games a dollar. Our
network-guided search plays about **360 times as many games per machine-hour** (32,800 against 92) and about 800
times as many per dollar. Three things besides the engine are in that gap: the cheaper Community pod (2.3 times per
dollar), XMage's IS-MCTS, which re-deals hidden cards at every simulation (docs/019 §4.6 measured one-world PIMC
games about twice as fast on XMage), and shorter games (about 110 searched decisions a game here, against XMage's
169). Allowing for the last two, the engines differ by roughly 100 times. docs/019 priced one XMage game at 1,000
simulations at $0.055 (18 a dollar); here a dollar buys 14,500.

**Where the time goes.** A heuristic-search simulation costs about 85 µs on one thread (82–89 µs at 100 to 1,000
simulations), almost all of it the engine playing to the next decision; the evaluator takes about 0.2 µs. A
network-guided simulation costs about 125 µs: the network's forward pass adds about 40 µs on these cores. Before our
speed patch the harness already ran random games 5.5 times faster per thread than mtg-kernel's JSON interface on the
fixture pair (92,000 against 16,600 games an hour); the patch made it about 3.5 times faster again.

**Other machines.** On Dan's laptop (M1 Pro, 8 threads, shared with other work), random games ran at about 1.9
million an hour and heuristic search at 100 simulations at about 28,000. A second rented RTX 3090 machine (23.8 vCPUs
of an EPYC 7663, same price) was 14–46% slower per thread depending on the bot: 14% for heuristic search at 1,000,
about 40% for network-guided search.

## 4. Baselines: heuristic search against random play and against itself

*Paired games on the 62 evaluation pairs (the 30 held-out generated decks and the two fixture decks).*

| Bot | Opponent | Paired games | Won |
|---|---|---|---|
| Heuristic search, 100 | random moves | 743 | 93.5% |
| Heuristic search, 300 | heuristic search, 100 | 744 | 56.3% |
| Heuristic search, 1,000 | heuristic search, 100 | 744 | 62.2% |
| Heuristic search, 3,000 | heuristic search, 100 | 744 | 65.6% |
| Heuristic search, 10,000 | heuristic search, 100 | 248 | 70.6% |
| Heuristic search, 100, 4 worlds | heuristic search, 100, 1 world | 124 | 40.3% |
| Heuristic search, 1,000, 4 worlds | heuristic search, 1,000, 1 world | 248 | 49.2% |

- **More search keeps helping:** each tenfold increase adds 8–12 points, less at higher budgets.
- **Re-dealing several worlds doesn't help** at these budgets: splitting 100 simulations over 4 worlds loses, and at
  1,000 it ties. That is consistent with docs/019 §4.6, where one-world PIMC played at least as well as IS-MCTS.

## 5. A new network trained by expert iteration

**The network** (`mtgkernel/src/encode.rs`, `nn.rs`, `py/model.py`) reads only what the player to move can see, from
mtg-kernel's own observation: 96 numbers about the game (life, cards in hand and library, turn and step, mana, the
kind of decision), one token per visible card (its card id, zone, tapped, damage, power and toughness, counters,
keywords, types) and one token per legal move (its kind, the card it uses and the card or player it targets). Card
tokens pass through a small network and are pooled by zone (my hand, my battlefield, the opponent's battlefield,
the rest); each move is scored against that summary, and a value head estimates the chance of winning. 485,000
parameters, all learned from scratch. The same forward pass is written in Rust for play: about 40 µs a decision on
one pod core (about 60 µs on the M1 Pro laptop), matching PyTorch to 2.4e-6 on the pod. It never sees hidden cards
(tested by re-dealing them).

**Training.** AlphaZero-style expert iteration at small scale, with one change: the value learns the search's own
estimate of the position (its root value), not the game result (see "What made the difference" below). Each
generation, a search bot plays itself on the 90 training decks; the network learns to predict the share of
simulations the search gave each move, and the search's root value; the next generation's search uses the network
for its move priors and position values. Each generation is trained from the previous network (a warm start).

| Generation | Self-play | New games | Trained on | Training (RTX 3090) |
|---|---|---|---|---|
| 0 | heuristic search, 1,000 simulations | 6,000 | 576,000 decisions | 2.0 minutes |
| 1 | generation 0's network-guided search, 300 simulations, exploration noise | 3,000 | 875,000 (generations 0–1) | 2.3 minutes |
| 2 | generation 1's network-guided search, the same | 3,000 | 1,180,000 (generations 0–2) | 2.9 minutes |

Exploration noise is AlphaZero's: random (Dirichlet) noise on the root's move priors (weight 0.25, α 0.3), and the
first 8 searched moves of each game sampled in proportion to their simulations. Evaluation games never use it.

**Results.** *Paired games on the 62 evaluation pairs, with the same deck pairs and shuffles for every generation.
"Policy alone" plays the network's most likely move with no search.*

| Bot | Opponent | Paired games | Generation 0 | Generation 1 | Generation 2 |
|---|---|---|---|---|---|
| Policy alone | random moves | 199–200 | 94.0% | **99.5%** | 98.0% |
| Policy alone | heuristic search, 100 | 998–1,000 | 52.9% | 60.6% | **63.7%** |
| Network-guided search, 100 | heuristic search, 100 | 500 | 68.8% | 70.2% | **71.4%** |
| Network-guided search, 300 | heuristic search, 100 | 500 | 73.8% | 76.8% | **77.2%** |
| Network-guided search, 1,000 | heuristic search, 100 | 500 | 78.4% | 79.8% | **80.8%** |
| Network-guided search, 1,000 | heuristic search, 1,000 | 248 | 67.3% | 69.4% | **70.2%** |

![Line chart of the win rate against heuristic search at 100 simulations by simulations per decision, log scale. Heuristic search: 56% at 300, 62% at 1,000, 66% at 3,000 and 71% at 10,000. Network-guided search, generation 0: 69% at 100, 74% at 300, 78% at 1,000. Generation 2: 71%, 77% and 81%. A dashed line marks 50%.](img/025-ladder-light.png)

*Figure 2. Win rate against heuristic search at 100 simulations, by search budget: generations 0 and 2 of the
network-guided search, and the heuristic search itself.*

*Generation 2 against the earlier networks, on the same evaluation pairs:*

| Generation 2's bot | Opponent | Paired games | Generation 2 won |
|---|---|---|---|
| Policy alone | generation 0's policy alone | 998 | 59.5% |
| Policy alone | generation 1's policy alone | 999 | 51.7% |
| Network-guided search, 100 | generation 0's, 100 | 500 | 53.8% |
| Network-guided search, 100 | generation 1's, 100 | 500 | 51.0% |

- **The policy alone beats the heuristic search at 100 simulations.** With no search at all, generation 0's network
  won 52.9% of 1,000 paired games against it and generation 2's won 63.7% of 998, at 13 ms a game. For comparison
  only (another engine, harder decks), experiment #4's imitation-learned policies won 38–40% against the same kind of
  baseline on XMage (docs/019 §4.2).
- **With search, the network bot beats the heuristic bot at equal budget:** generation 2 won 71.4% of 500 paired
  games at 100 simulations and 70.2% of 248 at 1,000. Network-guided search at 100 simulations (71.4%) is as strong
  as heuristic search at 10,000 (70.6% of 248, §4), for about 1/70 of the compute a decision.
- **Self-play helped, with returns falling fast at this size.** Generation 1 improved on generation 0 in every
  matchup; generation 2 improved the policy alone again (63.7% of 998 against 60.6% of 999 versus the heuristic
  bot), but with search it is indistinguishable from generation 1 (51.0% of 500 head to head). Each later
  generation added only 3,000 games to generation 0's 6,000.

**What made the difference** *(the policy alone of generation 0's network, against heuristic search at 100):*

| Training | Paired games | Won |
|---|---|---|
| First trainer: value trained on the game result, the epoch with the lowest total loss kept (the first), 8,000 games of heuristic search at 1,000 | 619 | 35.5% |
| Value trained on the search's root value, the epoch with the best policy kept, 3,600 games | 999 | 50.2% |
| The same, 6,000 games | 1,000 | 52.9% |

The game result is a noisy target for the ~110 decisions of a game: the value head memorized it, and through the
shared layers that hurt the policy (on 988 laptop games, +11.5 points for the root-value target). The fixes mattered
for the policy alone: guiding a search, the first trainer's network was already as strong as generation 0 (69.0% of
620 paired games at 100 simulations, against 68.8% of 500).

## 6. Porting draft-zero's networks

We did not port a draft-zero network. It is possible, but it is days of work, not hours, because of what the
networks read:

| | Graph network (docs/024, `gnn/full_r1`) | MLP (`exp4/mlp_1ep`) | Transformer (`exp4/stage3`) |
|---|---|---|---|
| Size | 13.4M parameters | 115M (98M in embeddings) | 40.6M (37.1M in embeddings) |
| Input | a graph of typed nodes (players, zones, cards, abilities) whose leaves are XMage strings: card names, rule and effect text, costs, `CanActivate` | the set of hashed feature ids from MageZero's `StateEncoder`: every feature string is hashed with its whole path (`Player#1/Battlefield#1/Forest#1/Tapped#1`), and numbers are thermometer-coded | the same as the MLP |
| Moves | the nodes of the legal options | XMage's ability strings (`Cast Llanowar Elves`, rule text) and target names, hashed into 1,024 slots | the same as the MLP |

mtg-kernel's observation is structured data (card ids, tapped, power and toughness, keyword flags), not XMage's
text, and its moves are its own kinds (cast, target, one attacker or blocker at a time, mulligans, damage
assignment). Two ways across, estimated from reading the encoders' code (no prototype):

- **XMage as the encoder (2–4 days).** Rebuild each mtg-kernel position inside XMage with draft-zero's existing
  state-builder (`mzbridge`'s StateSpec) and encode it there. Faithful for all three networks, but every decision
  goes through a Java process, which gives back much of the speed of §3.
- **Re-implement the encoder (3–5 days for the graph network, 5–8 for the others).** The graph network is the easiest:
  its 1,932 leaf strings can be recovered (we matched 753 of them, including every card in the fixture decks), and
  each card's subgraph depends only on the card and its zone. The flat networks hash whole paths with XMage's own
  ordering and counts, so one wrong count shifts many ids.

Either way, some decisions have no head in those networks (mulligans, trigger order, damage assignment), mtg-kernel
asks about blocks one pair at a time where XMage asks once per blocker, and the networks were trained on about
146,000 top-player games of all of FDN (22,230 drafts), not on 38 cards. **Training a new network on mtg-kernel's
own observations (§5) was faster and fits the engine.**

## 7. Card statistics against 17lands

As in docs/019 §4.4: each bot plays itself on all 120 generated decks, and each card's games-in-hand win rate in its
games is compared with 17lands' (all 791,000 FDN Premier Draft player-games) by Spearman rank correlation, over the
37 cards of our decks (19 commons), each in hand at least 30 times. Two more measures:

- **Deck-adjusted GIH WR:** the card's GIH WR minus the average win rate of the decks that saw it, on our side only
  (17lands' side stays its raw GIH WR). Every card is played only in decks of its own colours, and the colour pairs
  differ a lot in strength in this mini-format: in every bot's self-play white-green decks win 55–60% of their games
  and white-blue 36–46%. So raw GIH WR partly measures colour.
- **IWD** (17lands' improvement when drawn): GIH WR minus the win rate of the same decks in games where the card was
  not seen, on both sides.

**The noise ceiling** is matched card by card: for each card we draw as many of 17lands' games with it in hand as our
self-play has, recompute its GIH WR and correlate with the full data (60 draws). Each card is in hundreds to
thousands of our games (at least 753), so the ceiling is 0.94–1.00: game sampling explains little of the gap.

*Spearman rank correlation with 17lands, by bot (1 = 17lands' order, 0 = unrelated):*

| Measure | Random moves | Heuristic search, 100 | Heuristic search, 1,000 | Gen. 0 policy alone | Gen. 1 policy alone | Gen. 1 network search, 100 |
|---|---|---|---|---|---|---|
| Player-games | 199,620 | 50,400 | 12,000 | 50,266 | 50,302 | 25,198 |
| GIH WR, all 37 cards | −0.23 | 0.15 | **0.34** | 0.06 | 0.10 | 0.30 |
| GIH WR, 19 commons | −0.27 | 0.06 | **0.23** | 0.04 | 0.06 | 0.13 |
| Deck-adjusted GIH WR, all | 0.00 | 0.39 | **0.48** | 0.29 | 0.35 | 0.47 |
| IWD, all | 0.07 | 0.44 | 0.49 | 0.31 | 0.40 | **0.58** |
| Noise ceiling, GIH WR, all | 1.00 | 0.99 | 0.94 | 0.99 | 0.99 | 0.97 |

**Card by card.** *17lands' eight best of the 37 cards, and where the two bots that agree best rank them by
deck-adjusted GIH WR. 17lands' win rates are in points above or below its average over these cards (55.8%); the
bots' are in points above their decks' average.*

| 17lands rank | Card | Rarity | 17lands | Heuristic search, 1,000 (rank) | Gen. 1 network search, 100 (rank) |
|---|---|---|---|---|---|
| 1 | Celestial Armor | rare | +6.2 | +2.9 (16) | +3.5 (11) |
| 2 | Sylvan Scavenging | rare | +6.0 | +5.6 (3) | +4.4 (6) |
| 3 | Kiora, the Rising Tide | rare | +4.6 | +3.9 (7) | +5.2 (4) |
| 4 | Mischievous Mystic | uncommon | +3.8 | +3.7 (9) | +2.9 (16) |
| 5 | Exemplar of Light | rare | +3.2 | +6.8 (1) | +6.4 (1) |
| 6 | Felling Blow | uncommon | +2.9 | +3.4 (13) | +5.7 (2) |
| 7 | Spectral Sailor | uncommon | +2.8 | +3.0 (15) | +2.7 (20) |
| 8 | Sun-Blessed Healer | uncommon | +2.7 | +3.1 (14) | +4.8 (5) |

- **Search makes card ratings more like 17lands', as docs/019 expected.** Random play ranks cards against 17lands'
  order (−0.23). Heuristic search reaches 0.15 at 100 simulations and 0.34 at 1,000. The network's policy alone,
  with no search, scores 0.06 (generation 0) and 0.10 (generation 1); its search at 100 simulations reaches 0.30,
  close to what the heuristic search needs 1,000 simulations for. On IWD, which compares a card only with its own
  decks, the network search agrees best (0.58).
- **At the top, the network-search bot agrees with 17lands:** five of 17lands' eight best cards (Exemplar of Light,
  Felling Blow, Kiora, Sun-Blessed Healer, Sylvan Scavenging) are among its eight best.
- **The bots underrate cheap tricks and cheap evasive creatures** (the network search's rank against 17lands'):
  Fleeting Flight 37th against 15th, Joust Through 31st against 16th, Healer's Hawk 27th against 11th, Spectral
  Sailor 20th against 7th. Tricks need timing; on XMage, docs/019's bots underrated burn and removal for the same
  kind of reason.
- **And overrate green creatures:** Dwynen, Gilt-Leaf Daen 12th against 32nd, Treetop Snarespinner 7th against 26th,
  Cackling Prowler 10th against 27th.
- **Caveats.** Over only 37 cards a rank correlation is coarse (about ±0.17, from which cards happen to be in the
  pool); game sampling adds little. This is a white-blue-green mini-format with generated decks, so card values here
  genuinely differ from 17lands' full FDN format: a perfect bot would not reach 1.

## 8. Blockers and what to ask upstream

**Blocking a real FDN agent:**

1. **The card pool.** 38 of 286 FDN cards, no black, one red card. No real 17lands deck can be played except our two
   fixture decks, and card statistics cover only 37 cards (§7). The maintainer sizes the rest at several weeks of
   batches (milestone 5). By a greedy count, after PR #153's 7 cards, the 30 cards that unlock the most decks (mostly
   green, among the maintainer's easier tiers) would make 98 of draft-zero's 31,516 top-player decks playable.
2. **No Limited-capable training, search or network in mtg-kernel.** We built our own (§2, §5). For it to be
   supported upstream, mtg-kernel would need what our patches do:
   - a public constructor for the custom-deck Limited session (now crate-private), a public way to re-seed a copy's
     future shuffles, read access to the policy surface, and mutable access to the game state, which upstream
     deliberately does not give, for re-dealing hidden cards on a copy (patch 0001);
   - an opt-in way to step without hashing the full observation or deep-copying it and every legal action (patch
     0002): in a profile, the two took 77–81% of the CPU.

   Both change no rules and fit the maintainer's milestone 7 (DraftZero integration). **Ask:** would the
   maintainer take them as PRs?
3. **Fair hidden information.** Our search, like mtg-kernel's own, re-deals the opponent's hidden cards from their
   actual decklist (open decklists). The maintainer's milestone 6 is a sampler from public information; draft-zero
   has one for XMage (`belief.py`, closed decklists).

**Rough edges we worked around:**

4. **Ward halts the observation layer.** Without search, about 1 game in 400–500 on the generated decks stops with
   "legacy observation cannot identify the Ward-bound stack item" (190 of 100,000 random games; 49–64 of 25,200
   network-policy games); on the fixture pair, one game in 6,963. The rules engine itself is fine: skipping the
   observation lets the game finish. Our search drops branches that halt, so searched self-play never halted (0 in
   55,800 games); a halted game is recorded as an error and left out of the win rates. **To report upstream.**
5. **Runaway games.** Homunculus Horde with Strix Lookout doubles its tokens every turn: one test game reached 236
   tokens by round 16 and took 410 s, with 5,040-way trigger-order decisions (every ordering of 7 triggers is one
   menu). The rules are right; the game just needs a cap. We end a game as truncated (half a point each) when the two
   battlefields together reach 100 permanents (normal games peaked at 39 in 4,320 test games), and cap the network's
   input at 255 card tokens.
6. **Mana payment.** With mana abilities hidden, mtg-kernel's payment solver picks the first source in battlefield
   order, so it can tap Llanowar Elves while Forests stay untapped. Its cost is not measured; 70 of the 120 generated
   decks run Llanowar Elves.
7. **Moving target.** The catalog identity changes with every card batch (PR #153 moves it to v52), and much of
   mtg-kernel is guarded by frozen hashes, so our patches pin commit `a4e1474`.

**Not blocking, but worth knowing:** mtg-kernel's own trainers and searches are built around its Pauper decks, with
Windows- or CUDA-only production paths; its README calls the results "engineering evidence, not competitive
benchmarks".

## 9. Cost and reproducing

**Cost: $2.94 of the $10 budget** (RunPod; the laptop did the building and the analysis). The expert iteration's
self-play and training took about 2 machine-hours of that; the rest went to benchmarks, the heuristic ladders, the
card-statistics games, evaluations, and a first generation 0 run (8,000 games) whose data a code fix made unusable.

| Machine | Hours | Cost | Used for |
|---|---|---|---|
| RTX 3090 Community pod 1 (Threadripper PRO 5975WX, 13.6 vCPUs of quota), $0.22/h | 8.7 | $1.92 | benchmarks, the heuristic ladders, card statistics for the heuristic search, random play and generation 0, parts of generations 0 and 2, evaluations |
| RTX 3090 Community pod 2 (EPYC 7663, 23.8 vCPUs of quota), $0.22/h | 4.6 | $1.02 | self-play, training on its GPU, card statistics for generation 1, evaluations |

**Code** (all in `mtgkernel/` on the
[`test-mtg-kernel`](https://github.com/danieljbrooks/draft-zero/tree/test-mtg-kernel) branch; see its README):

| Path | What |
|---|---|
| `setup.sh`, `patches/` | clone mtg-kernel at `a4e1474` into `.deps/` and apply our two patches |
| `src/` (Rust crate `dzk`) | the game wrapper, bots (`random`, `mcts:N`, `net`, `pmcts:N`), PIMC re-dealing, encoder, network forward pass, game records, training data; `dzk play`, `bench`, `summarize`, `parity`, `nnbench` |
| `py/train.py`, `model.py`, `data.py`, `parity.py`, `loop.py` | the trainer (PyTorch), the network, the data loader, the Rust–PyTorch parity check, the expert-iteration driver |
| `py/gen_decks.py`, `decks/` | the generated decks (`gen_v1`) and copies of the two fixture decks |
| `py/gih_report.py` | the card statistics against 17lands, with the noise ceiling and the deck-adjusted GIH WR |
| `py/jsonl_bench.py`, `py/fig_doc025.py` | the JSON-interface benchmark, this doc's figures |
| `deploy/` | pod setup (`pack.sh`, `pod_bootstrap.sh`), job lists (`run_jobs.sh`, `bench_all.sh`, `eval_jobs.sh`) |

**To reproduce** (on the `test-mtg-kernel` branch; Rust 1.94.1, which on a Mac comes from Homebrew's rustup):

```bash
bash mtgkernel/setup.sh
```

```bash
cd mtgkernel && cargo build --release && cargo test --release -- --test-threads=1
```

One rung of §4's ladder (§4 pools this run, 620 games, with 124 more played with `--seed 14 --n-pairs 62`):

```bash
./target/release/dzk play --decks-dir decks/gen_v1:decks/fixtures --pairs pairs/eval_v1.tsv --bot1 mcts:1000 --bot2 mcts:100 --seed 34 --threads 8 --n-pairs 310 --out runs/ladder/games.jsonl
```

Generation 0's data and network, on a pod (`deploy/pod_bootstrap.sh` installs Rust and torch):

```bash
./target/release/dzk play --decks-dir decks/gen_v1 --pairs decks/gen_v1/pairs_train.tsv --bot1 mcts:1000 --bot2 mcts:1000 --seed 200 --threads 24 --pair-range 0:3000 --out runs/ei2/g0/sp/games.jsonl --data-out runs/ei2/g0/data
```

```bash
/root/venv/bin/python py/train.py --data runs/ei2/g0/data --out runs/ei2/g0/train --epochs 6 --device cuda
```

The figures:

```bash
python3 mtgkernel/py/fig_doc025.py --runs mtgkernel/runs --spec mtgkernel/py/fig_doc025_spec.json
```

Game records, training data and networks are in the git-ignored `mtgkernel/runs/` (laptop copies of the pods'
results): the three networks are 1.9 MB each (`runs/pod2/ei2/g{0,1,2}/train/net.dzkn`), and the 12,000 self-play
games' training data is 650 MB.
