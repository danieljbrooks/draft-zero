# Building sealed decks from self-play: FDN on gorge

*Status (Monday 12 October 2026, 6:45 AM PT): trained only on its own games, the builder learns to build simple,
white-based creature decks that its pilot (gorge's search with docs/031's network) can play well; played by that pilot
on both sides, those decks beat the human decks from the same pools 66% of the time (§5.1). The choices mirror the
pilot's weaknesses, so the next lever is gameplay, not deck building. The code is on the
[`sealed-deckbuilding`](https://github.com/danieljbrooks/draft-zero/tree/sealed-deckbuilding) branch (`gorge/sealed/`).*

**The goal:** a model that builds a 40-card sealed deck from a pool of six FDN packs, trained **only from games it
plays**, never from human decklists, and that builds decks which beat the decks humans built from the same pools.

## 1. The task, in brief

In sealed, each player opens six packs (about 81 cards in FDN once basic lands are set aside) and builds a 40-card
deck: usually about 17 lands and 23 spells in two colours. The edge cases are what make it hard to hand-code:
non-land cards that work like lands, synergy decks with unusual shapes, three- to five-colour decks that need a mana
base to support them. Every new set changes which shapes work, so whatever the model knows about curves, land counts
and colours has to be learnable from games.

Terms:
- **The play bot** plays every game: gorge's search with docs/031's loop B network (generation 7) inside it, at N
  **simulations** per decision (look-ahead lines tried before each move), exploration constant 0.5 (docs/031 found
  it played best). At 0 simulations, gorge's built-in heuristic bot plays instead.
- **A paired game** plays the same deal twice with the seats swapped, which cancels much of the luck of the draw.
- **A generation** is one round of: build decks, play them against each other, retrain.

## 2. Data: pools, human decks and the splits

From 17lands' public FDN data (CC BY 4.0), `gorge/sealed/data.py`:

| Set | Pools | Use |
|---|---|---|
| Sealed (Bo1 + Bo3), train | 6,240 | **pool contents only**: the pools the self-play builder practises on |
| Sealed, val | 1,000 | choosing settings |
| Sealed, test | 1,000 | final numbers, used once |
| Premier Draft, val + test | 1,000 + 1,000 | the secondary test: does a sealed builder build draft decks well? |

- A 17lands game row lists the player's deck and the rest of their pool, so each pool comes with the deck(s) the
  human built from it. When a player rebuilt a pool, the build with the most games is the **reference deck**.
- Human reference decks on val: 94% have two main colours, 86% play 17 lands, about 23 spells.

## 3. How a builder is judged

| Measure | How |
|---|---|
| **Card agreement** (primary) | cards the builder's deck shares with the human's deck from the same pool, out of 40 (basics included); also the non-land cards shared, and whether the main colours match |
| **Win rate against human decks** (primary) | pools paired (i, j): the builder's deck from pool i plays the human's deck from pool j, and the builder's deck from pool j plays the human's deck from pool i, each as a paired game: 4 games a pool pair. A builder identical to the humans scores exactly 50% |
| Secondary | land counts, colour counts, how the builder's per-card values rank against 17lands' games-in-hand win rate |

Both primary measures run on the same held-out pools, so a builder can't trade one off silently against the other.

## 4. The method

```
                 ┌──────────────────────────── the self-play loop (one generation) ───────────────────────────┐
                 │                                                                                             │
  train pools ──►│  BUILD: for each pool, decks from the current         PLAY: decks from different pools      │
  (contents      │  value network (best, explore, perturbed, random) ──► play paired games in gorge            │
   only)         │        ▲                                              (the play bot at N simulations)       │
                 │        │                                                          │                         │
                 │  TRAIN: the deck value network on the last 4 generations' results ◄┘                         │
                 └─────────────────────────────────────────────────────────────────────────────────────────────┘

  BUILD, for one pool:  pick colours (score a quick deck for every colour combination)
                        → refine the best 4 by one-card swaps until no swap helps
                        → keep the candidate the network likes best
```

**A deck value network** (`gorge/sealed/model.py`) gives every deck a strength s. A game between decks a and b is a
comparison: P(a wins) = sigmoid(s(a) − s(b) + an on-the-play term), fit to game results (Bradley-Terry). s has three
parts:
1. **Card quality:** one learned number per card, the sum over the deck. It plays the role of 17lands' games-in-hand
   win rate, learned from our own games.
2. **Structure:** a small network over deck totals built from learnable card *roles*: how much a card counts as a
   land, its effective mana value, whether it is a creature, which colours of mana it makes. Each role starts at the
   printed card and may move if the games say so, so the network learns this format's good curves, land counts and
   colour requirements, and can learn that a card with misleading printed facts (a 6-drop with landcycling) partly
   counts as a land.
3. **Context:** a set network over the deck's cards, for synergies the first two miss.

Card inputs are printed facts only (mana cost, types, power and toughness, keywords, rarity, the colours a land
makes). Five networks trained on resampled games form an ensemble; their disagreement says where the model is unsure.

**The builder** (`gorge/sealed/build.py`) follows the human process:
1. **Pick colours:** for each colour combination (5 mono, 10 pairs, 10 triples), a quick deck of its best cards by
   card quality, scored by the network.
2. **Build candidates:** the top four combinations are refined by swapping one card at a time (any deck card out, any
   pool card or basic land in) whenever the network scores the new deck higher, until no swap helps. Swaps can splash
   a card, change the land count or change a basic land's colour. This is the iterative "re-rank every card in the
   context of the current deck" idea, run to convergence.
3. **Choose** the candidate the network likes best.

**The self-play loop** (`gorge/sealed/loop.py`): each generation builds four decks for each of 800 train pools: the
best deck, a deck that one ensemble member alone likes best (exploring where the members disagree), the best deck
with 1-3 random swaps, and, 15% of the time, a random deck. Decks from different pools play paired games; the next
ensemble trains on the last four generations. Generation 0 builds random deck-shaped decks with 1-5 colours and 12-21
lands, so the network also sees decks that don't work.

The search over decks exploits the network's mistakes: in the first test, a network trained on random decks built
five-colour 15-land piles it had never seen lose. Playing the built decks and retraining on them each generation
is what corrects this (the same fix the Hearthstone deck-building literature found, §6).

**Search depth.** Every game costs more as the play bot searches more. Per game, on one laptop core:

| Play bot | CPU seconds a game | Games an hour on 15 of r1's cores |
|---|---|---|
| gorge's heuristic bot | 0.04 | ~350,000 (laptop, 4 workers) |
| search, 25 simulations | 2.5 | ~12,000 |
| search, 100 simulations | 11 | ~2,900 |
| search, 300 simulations | ~35 (est.) | ~1,000 (est.) |

The plan is a curriculum: many cheap generations with the heuristic bot (phase A), then generations at
increasing search depth that continue from phase A's network, with every builder judged at 100 and 300
simulations.

## 5. Results

### 5.1 Headline: the builder learns decks its pilot can play (Monday 12 October, 6:45 AM PT)

**Trained only on its own games, the builder learns to build simple, white-based creature decks that the bot playing
them can pilot well.** White is in 92% of its decks and white-green in 54% (the humans: 17%). It plays 17 creatures to
the humans' 15 and 5 non-creature spells to their 8, and skips the cards that reward skilful play: planeswalkers,
tricks, card advantage, fixing lands, and some strong cards outright (Giada, Font of Hope, 0% of the pools that have
her; humans 61%). This is how a newer human player builds: decks that play themselves.

**Played by its pilot, these decks win.** Both decks in every test game are played by the same bot: gorge's full
search at 100 simulations with docs/031's loop B network. Against the decks the humans built from the same val pools,
the best builder's decks (phase B, generation 4) win 66.1% of 1,000 games, where the 17lands heuristic's decks win
56.6% and a builder that imitates the human decks wins 54.3%. The human decks' strengths need play this pilot doesn't
have, so the win rate measures how well a deck suits the pilot more than how good it is.

| Who played the games | Phase |
|---|---|
| gorge's heuristic bot, no search | phase A's training games |
| the full search at 25 simulations with the network | phase B's training games |
| the same search at 100 simulations | phase C's training games (from 6:30 AM), and every test game in this report (phase A's generation 2 also at 300: 59.4%) |

**What it means for the project** (the goal is bots that play sealed well): the builder works, in that it finds what
wins for whoever plays its decks, and it will follow the pilot as the pilot improves. The ceiling now is the pilot.
docs/031's search ladder found the same: against gorge's heuristic bot, strength levels off near 82% between 500 and
3,000 simulations.

![Win rate against the human decks at 100 simulations, per builder](img/032-builders-light.png)

| Builder | Trained on | Cards shared with the human deck (of 40) | Main colours match | Won against the human decks, search at 100 simulations |
|---|---|---|---|---|
| Random deck-shaped build | nothing | 14.3 | 11% | 35.8% of 400 |
| 17lands heuristic | human games (17lands statistics) | 20.2 | 29% | 56.6% of 1,000 |
| Imitation of human decks (same network and builder) | human decks | **23.8** | **36%** | 54.3% of 1,000 |
| Self-play, phase A, generation 1 | its own games only (gorge's heuristic bot) | 18.2 | 18% | 52.3% of 1,000 |
| Self-play, phase A, generation 2 | its own games only (gorge's heuristic bot) | 18.6 | 21% | 60.9% of 1,000 |
| **Self-play, phase B, generation 3** | **its own games only** (then the search at 25 simulations) | 18.7 | 25% | **65.4% of 1,000** |

*Agreement on 1,000 val pools (the self-play rows: 500). Win rates: 500 val pools, 4 games a pool pair; the random
build's is the search-depth test's (§5.3). One standard error is about 1.5 points at 1,000 games, 2.5 at 400.*

### 5.2 Card agreement is hard in sealed

Even the imitation builder, trained on 6,240 human sealed decks, shares only 23.7 of 40 cards with the human deck and
picks the same main colours a third of the time. A sealed pool usually supports several reasonable colour pairs,
and once the colours differ most of the 40 cards do. Draft pools (about 41 cards, already mostly two colours) are far
easier:

| Builder | Sealed: cards shared (of 40) | Draft: cards shared (of 40) |
|---|---|---|
| Random deck-shaped build | 14.3 | 28.3 |
| 17lands heuristic | 20.3 | 33.3 |
| Imitation of human decks | 23.7 | 34.6 |

*1,000 val pools each.*

### 5.3 Does the play bot's search depth change which deck wins?

Human reference decks against two kinds of decks from other val pools (4 games per pool pair):

| Play bot | Human decks vs random builds | Human decks vs the 17lands heuristic | Games each |
|---|---|---|---|
| gorge's heuristic bot | 65.4% | 46.5% | 2,000 |
| search, 25 simulations | 67.6% | 47.4% | 800 |
| search, 100 simulations | 64.2% | 45.8% | 400 |
| search, 300 simulations | 66.2% | 45.8% | 400 / 240 |

- **For coarse deck quality, depth hardly matters:** human decks beat random builds about two games in three, and
  lose to the 17lands heuristic about 46-54, at every level from no search to 300 simulations.
- **The 17lands heuristic beats the human decks** at every level. The humans' decks are a modest bar in gorge: many
  17lands players are average, and the bot may reward raw card quality more than human opponents do.
- Since the cheap heuristic bot ranks these decks like the search does, at 1/60th of the cost of 25 simulations,
  the loop starts there (phase A) and moves to search-level games (phase B) once the cheap games stop teaching.
  Whether the finer choices (the 22nd and 23rd card, a splash) also agree across depths is what phase B tests.

### 5.4 Phase A: self-play with gorge's heuristic bot (laptop)

Each generation: 800 train pools, four or five decks each, 40,000 games played by gorge's heuristic bot (about 8
minutes on 4 laptop cores). The last column is the measure that matters: the search at 100 simulations playing the
generation's decks against the human decks, on the same deals for every builder.

| Generation | Cards shared (of 40) | Main colours match | Lands | Won against the human decks, heuristic bot | Won against the human decks, search at 100 simulations |
|---|---|---|---|---|---|
| 0 (trained on random decks) | 17.2 | 13% | 14.3 | 48.1% | |
| 1 | 18.9 | 22% | 17.0 | 67.5% | 52.3% |
| **2** | 18.7 | 26% | 17.4 | 71.1% | **60.9%** |
| 3 | 19.1 | 26% | 18.1 | 73.5% | |
| 4 | 18.8 | 27% | 18.2 | 68.3% | |
| 5 | 19.0 | 30% | 18.1 | 67.1% | 57.6% |
| 6 | 18.5 | 29% | 19.4 | 60.1% | |
| 7 | 19.8 | 30% | 18.1 | 71.6% | |
| 8 (support region, land exploration, draw term from here) | 19.4 | 30% | 18.7 | 71.9% | |
| 9 | 18.9 | 27% | 18.7 | 71.2% | |
| 10 | 18.8 | 27% | 19.2 | 68.1% | |
| 11 | 18.4 | 27% | 18.7 | 68.7% | |
| 12 | 19.1 | 26% | 18.5 | 72.5% | |

![Phase A: win rate against the human decks per generation, played by the heuristic bot and by the search](img/032-phaseA-light.png)

*Agreement on 200 val pools; win rates on 500 val pools (1,000 games; one standard error about 1.5 points). The
heuristic-bot column of generations 0-3 played under the Omniscience bug (§5.5).*

- **Cheap games teach the basics fast:** one generation fixes the land count (14 to 17), and by generation 2 the
  builder beats the human decks 60.9% when the search plays.
- **Then they mislead.** Later generations keep winning about 70% under the heuristic bot but fall to 57.6% under
  the search (generation 5): the builder drifts toward what the heuristic bot rewards (more lands, 18-19; more
  creatures; a higher curve). This is the search-depth question Dan raised, answered: the weak bot is a good
  teacher for two generations and a bad one after. Phase B continues from generation 2 with games played by the
  search.
- **What the bot-taught builder builds** (60 val pools, generation 1): 19.9 creatures against the humans' 15.4, an
  average mana value of 3.6 against 3.1, almost no non-basic lands (0.1 against 1.4). In the first val pool it
  built a red-white creature deck where the human built black-white with five removal spells.

### 5.5 What the self-play builder builds: decks for its pilot

The best builder (phase B, generation 4) against the humans, on the same 500 val pools:

| Main colours | Builder | Humans | Builder's decks won against the human decks |
|---|---|---|---|
| White-green | **54%** | 17% | 69.9% (269 pools) |
| White-black | 22% | 18% | 61.5% (109) |
| White-blue | 8% | 6% | 55.3% (38) |
| White-red | 7% | 8% | 67.6% (34) |
| Blue-green | 2% | 9% | 50.0% (12) |
| Black-red | 2% | 6% | 72.7% (11) |
| Black-green | 2% | 12% | 55.6% (9) |
| Red-green, blue-black, blue-red | under 1% each | 9%, 7%, 4% | |
| Three colours | 3% | 1% | |

| | Builder | Humans |
|---|---|---|
| Creatures | 17.2 | 15.4 |
| Non-creature spells | 5.4 | 7.8 |
| Average mana value of the spells | 3.39 | 3.06 |
| Lands (non-basic) | 17.5 (0.0) | 16.9 (1.3) |
| Splashes a third colour | 58 decks | 191 decks |
| Same main colours as the human's deck | 125 of 500 | |

**An example (val pool 0).** Both built white-black and share 30 of 40 cards; the builder won 2 of its 4 games.

- **Builder (17 lands, 8 Plains and 9 Swamp):** 16 creatures, including 2 Crypt Feaster, Campus Guide and Meteor Golem
  where the human has Giada, Font of Hope, Diregraf Ghoul and Dauntless Veteran; 7 spells, including a second Eaten
  Alive.
- **Human (16 lands, with 2 Evolving Wilds, Scoured Barrens and Soulstone Sanctuary):** 15 creatures; 10 spells,
  including Stab, Stroke of Midnight, Claws Out and Rise of the Dark Realms.
- **Two clear mistakes by a human player's standard** (Dan): no Giada and no Scoured Barrens. In the builder's own
  games, decks that drew Giada won 37.9% of 494 games, about 5 points below those decks' average, and Evolving Wilds
  32.5% of 1,301, so the network learned both as weak for this pilot. A paired test of each card in 100 of the
  builder's decks, with and without it against the same human decks on the same deals, is running to separate "weak
  for this pilot" from "the builder never learned better".

- **It never plays** planeswalkers (Ajani, Caller of the Pride: 0% of the pools that have it, against the humans'
  100%; Chandra, Flameshaper and Kaito likewise), equipment (Leyline Axe: 0% against 78%), Evolving Wilds (0% against
  83%), Day of Judgment or Phyrexian Arena, and plays plain creatures instead (Savannah Lions, Juggernaut,
  Armasaur Guide, Campus Guide: 90-100% against 10-30%).
- **Per card, the rate at which it plays a card from the pool** has a rank correlation of 0.47 with the humans'. Its
  colour pairs are far from theirs (total variation distance 0.54). Each card's games-in-hand win rate in its own
  games has a rank correlation of 0.39 with 17lands' across 249 cards.
- **The reading:** a builder trained on games learns what wins *for the player who plays its decks*. gorge's
  heuristic bot, which played phase A's games, attacks with creatures well and gets little from planeswalkers,
  equipment and card advantage; the builder followed. The decks still beat the humans' 61% when the search plays
  them, so the search seems to share some of that taste; phase B, whose games the search plays, tests how much.
- **Building cautiously doesn't help (experiment E2):** building for the ensemble's mean less one standard deviation,
  to avoid decks the networks disagree on, won 54.4% of 1,000 games against the plain mean's 57.6% on the same
  network (generation 5). Disagreement marks the unexplored decks, and some of those are the good ones.

- **Picking among candidates by playing them doesn't help either (experiment E4).** The last step of the human
  process is to build a few decks and keep the one that plays best. For each val pool, generation 2's top three
  candidates (from different colour configurations) each played 80 games against a gauntlet of 40 self-play decks
  (gorge's heuristic bot; 101,840 games), and a Bradley-Terry posterior with the network's score as its prior picked
  one. The playoff overrode the network in 222 of 500 pools, almost always between near-ties (median score gap 0.05),
  and its decks won 60.6% against the network's 60.9%: on the same deals, a paired difference of -0.3 points
  (standard error 1.3). Eighty cheap games per candidate tell the builder nothing the network, trained on hundreds
  of thousands of them, did not already know.

### 5.6 The builder found a rules bug in gorge

Phase A's networks ranked **Omniscience**, a 10-mana enchantment that 17lands' players rank near the bottom, as the
best card in FDN. The games agreed: decks with it won 60% of their bot games, and 82% of the games where it was drawn.
gorge let Omniscience's own "cast spells from your hand without paying their mana costs" work **while it was still
in hand**, so it cast itself for free and then made every later spell free.

- **The fix** (gorge patch 0009, `gorge/patches/`): a card's own static ability that names no zone works only on the
  battlefield, as the rules say. With it, the same 600 deck pairings give Omniscience decks 42.7% of 613 games and
  39.5% of the 256 where it was drawn: the dead card it should be.
- **The same bug** let 44 other card/zone pairs in gorge's full card pool grant themselves a permission (Future
  Sight playing itself from the top of the library, for example); none of them are in FDN. The fix and a census of
  the affected cards went to gorge's author as an issue and a pull request.
- **What it changes here:** every machine runs the fixed engine from Monday 12 October, 12:00 AM PT. Phase A's
  generations 0-3 and the first 100-simulation results played under the bug. 24 of generation 2's 500 val decks
  include Omniscience; on the fixed engine the same decks win 60.9% instead of 61.3%, so the bug barely touched
  the headline.
- **A lesson for the method:** a builder that optimises win rate exploits whatever the engine gets wrong, which makes
  self-play deck building a useful engine audit. A card whose learned value is far from its human rating is worth
  checking in the rules before trusting the model.

### 5.7 Two failure modes of searching against a learned value, and the fixes

**Extrapolation.** Phase A's generation 6 built 43 decks with 23-24 lands and 7 with 7-8 lands on 200 val pools,
although its own training games had 21-land decks winning 31% and 15-16-land decks 31-37%. All five ensemble members
agreed that 24 lands beat 16 by about 2 logits on one pool: the network learned "a basic land beats a weak spell"
from games with 17-19 lands, and the deck search found the land counts where that extrapolation was most wrong. Its
win rate against the human decks (heuristic bot) fell from 73.5% to 60.1%; generation 7 recovered to 71.6% by chance.

- **Fix 1, a support region:** the builder considers only land counts that make up at least 2% of the training
  window's games (17-20 at generation 6). Nothing is hand-set: the region is read from the data and widens as the
  data does. This is the pessimism of offline model-based optimisation, applied to the one deck feature where it
  failed.
- **Fix 2, targeted exploration:** a fifth deck per pool, the best deck with 1-3 lands more or fewer, so the network
  sees real results at the edges of the region.

**Noise in card values.** A game result credits all 40 cards, but only the ~17 a player drew can have mattered.
17lands' games-in-hand win rate uses this, and so now does the network: the result also depends on how much better
or worse the drawn cards were than the deck's average (a draw term). Which cards are drawn is random, so this is a
natural experiment free of the builder's own choices. On 152,000 of phase A's games it leaves the deck-level
held-out loss unchanged (0.6510 against 0.6506) and makes the card values steadier: separately trained networks
rank the cards alike with a rank correlation of 0.914, up from 0.854. Both fixes run from phase A's generation 8
and phase B's generation 1.

### 5.8 Phase B: self-play games played by the search (r1 and a pod)

Phase B continues from phase A's generation 2 with games played by the search at 25 simulations: 800 train pools
and 20,000 games a generation, about 45 minutes on r1's 15 cores and a RunPod RTX 3080 Ti. Each generation is judged
like the others: 1,000 games against the human decks at 100 simulations.

| Generation | Lands | Won against the human decks, 25 simulations | Won against the human decks, 100 simulations |
|---|---|---|---|
| 0 | 17.9 | 63.1% | 59.7% |
| 1 | 18.7 | 58.3% | 56.0% |
| **2** | 18.0 | 61.7% | **61.6%** |
| **3** | 17.9 | 65.4% | **65.4%** |
| **4** | 17.5 | 67.7% | **66.1%** |
| 5 (castability from here) | 17.0 | 65.4% | 63.0% |

*Generation 1 is the first with the support region, land exploration and the draw term (§5.7).*

**Do the builder's extra lands cost it games? No (a hypothesis tested and rejected).** Across builders, more lands
went with lower win rates against the human decks (phase A generation 2: 17.3 lands, 60.9%; phase B generation 1:
18.7 lands, 56.0%), and within phase B generation 1's games its 18-land decks won 61%, its 19-land decks 53% and its
20-land decks 48%. Its own self-play games also showed one land fewer costing 5.4 points (standard error 2.7) but one
or two more costing only about 1, which suggested that its slow, creature-heavy field made flooding cheap where the
humans' decks would punish it. The direct test says otherwise: the same 500 decks with their land count set to 17
(the weakest spells swapped in or out) won 2.0 points *less* against the human decks on the same deals (standard error
1.6), and no group gained (decks built with 18, 19 or 20 lands: -2.6, -0.8, -4.1 points). The gradient was the pools:
the builder plays more lands in pools that are weaker to begin with. The lesson for the method: correlations within
an evaluation mislead; a paired intervention on the same deals is cheap and decisive.

**Castability, a rule of the game in the network.** A colour-diverse league experiment (E5, below) produced a network
that built decks with red spells and no Mountains. The exploration decks explain it: 70% of the perturbed decks carry
about one card the deck has no mana for, and they win only 1-2 points less, so the network learned that uncastable
cards are nearly free and the search extrapolated to eight of them. The network now counts each spell in proportion
to how castable its deck makes it: a smooth, learned function of the deck's sources of each colour the spell needs
(after training: one pip with no sources counts 8%, with three sources 50%, with eight 97%). It changes nothing a
human would have hand-set, generalises to any set, and is what makes a three- or five-colour deck's mana base
matter to the builder. Retrained on the exact games that produced the broken network, without castability the
collapse repeats (58 of 60 val decks with spells the deck cannot cast, 8.5 such cards a deck); with it, 2 of 60 (0.2
a deck), and the held-out loss is slightly better (0.6687 against 0.6693). Phase B uses it from generation 5.

### 5.8b A colour-diverse league of opponents (experiment E5): no effect; bot games keep misleading

The builder's own decks are white-green 57% of the time, so it trains mostly against white-green creature decks; the
humans' decks spread over all ten colour pairs. E5 added a fifth deck per pool, the best deck the builder can make in
a randomly chosen colour pair, so its training field covers every pair (no human data needed). Two runs, identical
except for those decks, each three bot-level generations from phase A's generation 2, judged at 100 simulations:

| Run | Won against the human decks, 100 simulations |
|---|---|
| With the colour league | 48.7% of 1,000 |
| Without (control) | 50.2% of 1,000 |
| Their starting point, phase A generation 2 | 60.9% of 1,000 |

The league made no difference, but both runs lost ten points: three more generations of games played by gorge's
heuristic bot took the builder from 61% to a coin flip against the humans. (The league run also collapsed for one
generation into decks with spells they could not cast, the episode that led to castability, §5.8.) This is the
strongest evidence for Dan's point that the play bot's search depth is critical: the weak bot is a good teacher for
two generations and a harmful one after.

### 5.9 Draft: the builder on pools it never trained on

The same builders on 500 Premier Draft val pools (about 41 cards each), against the human draft decks, at 100
simulations:

| Builder | Cards shared (of 40) | Main colours match | Won against the human decks |
|---|---|---|---|
| **Self-play, phase A generation 2** (trained on sealed pools only) | 30.8 | 80% | **54.4%** of 1,000 |
| 17lands heuristic | 33.5 | 88% | 50.3% of 1,000 |

A drafted pool is mostly two colours already, so there is less room to improve on the human deck than in sealed,
but the self-play builder still does, and transfers better than the heuristic.

## 6. Prior work

No published builder reports either of our measures (agreement with human decks from the same pool, or win rate
against them), so there is nothing to compare these numbers with directly. The closest work:
- **Ryan Saxe's builder** ([RyanSaxe/mtg](https://github.com/RyanSaxe/mtg)) adds cards one at a time with a network
  trained on 17lands human decks. That is the approach of our imitation reference (§5, to come), not of the
  self-play model.
- **Heuristic builders** (Draftsim, the MTGA 17lands assistant, Heaton's "Lucille") take the best ~23 cards in two
  colours by a rating, as our 17lands baseline does.
- **Hearthstone deck building from simulated games:** Q-DeckRec (Chen et al. 2018) and DSA-ME (Zhang et al. 2022)
  learn a deck win-rate model and search over one-card swaps. Q-DeckRec found that search on a fixed learned model
  exploits its errors; DSA-ME retrains the model on the decks the search finds, as this loop does.
- **npiguet/price-predictor** scores sealed decks with a Bradley-Terry model of Forge games and builds by simulated
  annealing, the same structure as here with a fixed bot.
- **Legends of Code and Magic drafting** (Kowalski & Miernik 2020; Vieira et al. 2020) learns card values from
  games played by fixed agents, noting how few games each candidate gets.
