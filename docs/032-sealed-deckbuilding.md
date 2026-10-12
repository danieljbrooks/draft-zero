# Building sealed decks from self-play: FDN on gorge

*Status (Monday 12 October 2026, 1:10 AM PT): a builder trained only on its own games beats the human decks 61% at
100 simulations after two cheap generations, ahead of the 17lands heuristic (57%) (§5.1); search-level self-play
(phase B) is running. The code is on the
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

### 5.1 Headline so far (Monday 12 October, 12:00 AM PT)

**A builder that learned only from games played by gorge's heuristic bot, two generations in, wins 60.9% of 1,000
games against the human decks from the same val pools when the full search plays them at 100 simulations** (71.1%
when the heuristic bot plays). Its decks share 18.6 of 40 cards with the humans'. In the same test, on the same
deals, the 17lands heuristic (which uses human statistics) wins 56.6%, and a builder trained to imitate the human
decks wins 54.3%: the deck that agrees most with the humans is not the one that beats them.

![Win rate against the human decks at 100 simulations, per builder](img/032-builders-light.png)

| Builder | Trained on | Cards shared with the human deck (of 40) | Main colours match | Won against the human decks, search at 100 simulations |
|---|---|---|---|---|
| Random deck-shaped build | nothing | 14.3 | 11% | 35.8% of 400 |
| 17lands heuristic | human games (17lands statistics) | 20.2 | 29% | 56.6% of 1,000 |
| Imitation of human decks (same network and builder) | human decks | **23.8** | **36%** | 54.3% of 1,000 |
| Self-play, phase A, generation 1 | its own games only (gorge's heuristic bot) | 18.2 | 18% | 52.3% of 1,000 |
| **Self-play, phase A, generation 2** | **its own games only** (gorge's heuristic bot) | 18.6 | 21% | **60.9% of 1,000** |

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

### 5.5 The builder found a rules bug in gorge

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
