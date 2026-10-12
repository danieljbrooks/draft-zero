# Building sealed decks from self-play: FDN on gorge

*Status (Sunday 11 October 2026, 10:40 PM PT): the data, the game harness and the first self-play loop are running;
results so far are in §5. The code is on the
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

### 5.1 Baselines

| Builder | Trained on | Cards shared with the human deck | Non-land cards shared | Main colours match |
|---|---|---|---|---|
| Random deck-shaped build | nothing | 14.0 / 40 | | |
| 17lands heuristic: best two colours by games-in-hand win rate, top 23 spells, 17 lands | human games (17lands statistics) | 20.3 / 40 | 11.3 / 23 | 28% |

*300 val pools for the random build, 1,000 for the heuristic.*

### 5.2 Does the play bot's search depth change which deck wins?

Human reference decks against two kinds of decks from other val pools (4 games per pool pair):

| Play bot | Human decks vs random builds | Human decks vs the 17lands heuristic | Games each |
|---|---|---|---|
| gorge's heuristic bot | 65.4% | 46.5% | 2,000 |
| search, 25 simulations | 67.6% | 47.4% | 800 |
| search, 100 simulations | *running* | | 400 |
| search, 300 simulations | *running* | | 400 |

- Deck quality shows clearly at every level: human decks beat random deck-shaped builds two games in three.
- **The 17lands heuristic beats the human decks** (53% for the heuristic), at both levels so far. The humans'
  decks are a modest bar in gorge: many 17lands players are average, and the bot may value raw card quality more
  than a human opponent would punish it.

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
