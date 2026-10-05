# Learning to play Magic: The Gathering limited from human games

*Experiment #4's report, October 2026. The plan is [docs/017](017-experiment4-scaling-up-imitation-learning.md) and
the detailed run log is [docs/018](018-experiment4-run-log.md).*

*Status, 5 October: final. The MLP's games at 1,000 and 3,000 simulations finished early on 5 October (§4.2); §4.6
added (PIMC), and §4.2 corrected (the bots knew the opponent's decklist).*

## Abstract

The goal of this project is to develop an agent that plays Magic: The Gathering limited at a high level. Most
game-playing systems for Magic: The Gathering are built on tree search. The most powerful recipe of this kind,
AlphaZero's, learns by playing against itself, and that takes millions of games. On the open-source XMage engine,
one such game costs about $0.055 at 1,000 search simulations per decision. AlphaZero's 44 million games of chess
would cost us about $2.4 million.

So we learn from expert human players instead. 17lands publishes replays of Magic Arena games. We rebuilt **161,206
Foundations (FDN) games by players with a 60%+ win rate** inside XMage, which gave **12.1 million decisions** to learn
from. On those decisions we trained two neural networks, a small transformer and a cheaper MLP. Each learns to
**predict the move the human made** (imitation learning, or behaviour cloning) and to **predict who will win**.

We then let each network guide a tree search and played it against MageZero's hand-written heuristic bot:

- **Search makes the networks strong.** On its own, the network's policy won 40% of games against the heuristic
  bot searching 100 simulations. With search of its own, the network won 55–56% at 100 simulations, 61–64% at 300,
  62–64% at 1,000 and 69% at 3,000.
- **With equal search on both sides, the network bot won 56% of 314 paired games.**
- **A faster search does better still.** With PIMC, which searches one dealt version of the hidden cards instead of
  re-dealing them for every simulation, the network won 74–76% at 1,000 simulations against IS-MCTS's 64%, and its
  games were about twice as fast (§4.6).
- **Playing its policy alone against itself, the transformer ranks cards somewhat like humans do.** Card win rates
  from 10,000 such games correlate with 17lands' at 0.41 on commons, about half of what sampling noise allows. The
  MLP's correlate at 0.34.

The next step is to keep improving these networks by self-play, starting from what they learned from people.

**Terms used throughout:**

- **Policy:** the network's probability for each legal move.
- **Value:** the network's estimate of the chance of winning from a position.
- **Search and simulations:** before each move, the bot plays out many short look-aheads ("simulations") inside the
  game engine. The policy suggests which moves to explore and the value judges where they lead. More simulations
  look further ahead and cost more.
- **Paired games:** every pair of decks is played twice, and the two bots swap decks and seats for the second game.

## 1. Introduction

### 1.1 The goal

Magic: The Gathering is a two-player card game with hidden information. In **limited** formats such as draft,
players build a 40-card deck from cards they open during the event. Almost every game is played with a deck nobody
has played before, so playing well means understanding a whole set of cards, not memorising one deck. A game lasts
about 15–20 turns. Each player makes dozens of decisions: which spell to cast and when, what to attack with, how to
block, what to target. Each player's hand and the order of both decks are hidden. The aim of this project
(DraftZero) is an agent that plays limited as well as a strong human.

### 1.2 Why not learn from self-play alone?

The usual recipe for a game-playing agent comes from AlphaZero (Silver et al., 2018):

- A neural network proposes moves (the policy) and judges positions (the value).
- A tree search uses both to look ahead before each move.
- The games the agent plays against itself become its next training data.

The recipe is general, but it needs a lot of games. AlphaGo Zero's first strong run played 4.9 million games,
AlphaZero played 44 million games of chess, and DeepNash played 5.5 billion games of Stratego (Figure 8).

Magic engines are slow to search. Ours is XMage, a full implementation of Magic's rules. On the most
cost-effective rented machine we measured, one game in which both players search costs
([docs/020](020-cpu-benchmarks.md)):

| Simulations per decision | Games per dollar | Cost per game |
|---|---|---|
| 100 | 183 | $0.0055 |
| 1,000 | 18 | $0.055 |
| 10,000 | 1.8 | $0.56 |

At 1,000 simulations, AlphaZero's 44 million games would cost about $2.4 million. Our own small attempts showed the
problem. Experiment #2a trained by self-play for 18 generations (43.6 pod-hours, 2,673 games), and its network
ended up judging positions no better than the hand-written heuristic it was meant to replace
([docs/014](014-exp2-run1-report.md), [docs/016](016-search-benchmark-results.md)).

### 1.3 Learning from experts instead

Strong human games already exist in large numbers, for free. 17lands is a community site whose app records Magic
Arena games, and it publishes them. For FDN alone that is 791,159 games, about 166,000 of them by players who win
60% or more. That is roughly the amount of data the original AlphaGo learned from: 160,000 human games of Go, before
any self-play (Silver et al., 2016).

So this experiment asks how far top players' games can take us:

- **imitate their moves** to learn a policy;
- **learn from their games' results** to learn a value;
- **search on top:** use both inside a tree search, and play against the hand-written heuristic bot.

### 1.4 What we found

- **The data maps onto the engine.** About 85% of the turns in top players' games replay in XMage to the state
  17lands recorded next. That yields about 75 decisions to learn from per game (§2).
- **Two small networks imitate top players well.** On games they never saw, both networks pick the move the human
  made 83% of the time when the human acted (guessing would get 47%). They predict attacks 86–87% of the time (§4.1).
- **Search turns imitation into strength.** The policy alone loses to the heuristic bot, but with search the network
  bot wins 55–69% of games depending on how much it searches (§4.2).
- **The MLP plays as well as the transformer** at a tenth of the computing cost per move (§4.2).
- **Self-play with each model's policy alone ranks cards partly like 17lands does** (0.41 for the transformer, 0.34
  for the MLP, on commons), but both undervalue burn and removal (§4.4).

The whole experiment cost about $75 of rented compute, nearly three quarters of it for evaluation games (§4.5).

## 2. Data: from 17lands replays to engine positions

### 2.1 Engine and framework

- **[XMage](https://github.com/magefree/mage)** is an open-source Magic engine with essentially every card
  implemented. It plays full games under the real rules.
- **[MageZero](https://github.com/WillWroble/MageZero)** is Will Wroble's AlphaZero-style framework for XMage. It
  turns a game position into input for a network and runs tree search through the engine.
- DraftZero adds the parts a whole format needs:
  - one agent for any deck in the set;
  - data from 17lands;
  - a fair search that can't see hidden cards;
  - comparisons with human play.

### 2.2 The set: Foundations (FDN) as a case study

FDN, released in November 2024, is our first case study. Nothing in the method is specific to it. 17lands publishes
the same kind of replay data for 27 other sets, and the same pipeline applies to them. What changes per set is the
mapping from 17lands' card ids to XMage and the network's list of possible moves.

### 2.3 The 17lands data

17lands' public FDN Premier Draft replay file ([17lands.com/public_datasets](https://www.17lands.com/public_datasets),
CC BY 4.0) holds 791,159 games from 135,418 drafts, played between 12 November and 15 December 2024. We are grateful
to 17lands for publishing it.

Each row is one game from one player's point of view. For every turn it records:

- the cards drawn, played and cast;
- the attackers and blockers;
- what died;
- a snapshot at the end of the turn: hands, boards and life totals.

The opponent's hand appears only as a count. So the data shows exactly what the player could see, which is also what
a fair agent should see.

**Top players only.** We keep games by players in 17lands' 60%-and-up win-rate group: **166,763 games from 25,328
drafts**. Those players won 64.0% of them, against 54.7% across the whole file. (The win-rate group is computed over
games that include the game itself, so it slightly favours wins. That hardly affects imitating moves, but it biases
the value's targets a little: [docs/008](008-gameplay-data.md) §3.5.)

**Training, validation and test games are split by draft.** A deck never appears in two splits, and games recorded
by both players stay together. The drafts of an earlier benchmark (5,557 games) are held out of everything.

| Split | Drafts | Games | Decisions |
|---|---|---|---|
| Training | 22,230 | 145,898 | 10,947,614 |
| Validation | 1,262 | 8,248 | 602,555 |
| Test | 1,081 | 7,052 | 531,406 |

### 2.4 Rebuilding the games inside the engine

17lands records what happened in each turn, but not the order of events within it, the targets, or which moves were
available. To learn from a decision we need three things: the position as the engine sees it, the legal moves, and
the move the human chose. Figure 1 shows how we recover them; [docs/008](008-gameplay-data.md) §4 has the details.

![Five boxes left to right: a 17lands replay; rebuild the start of the turn from the last snapshot plus the card drawn, inferring tapped lands, counters and attachments; load it into XMage by placing every card in its zone; replay the turn with both players' recorded plays, trying up to 12 orderings of what isn't recorded; and, if the result matches the next snapshot (about 85% of turns), training examples: the position, the legal moves and the move the player made. Turns that don't match are dropped.](img/019-pipeline-light.png)

*Figure 1. From a 17lands replay to training examples.*

1. **Every card, token and ability** in the file maps to its XMage object. Every spell cast and land played maps to
   one of MageZero's moves.
2. **The start of the turn** is the previous end-of-turn snapshot plus the card drawn. Which lands are tapped, and
   counters and attachments, are inferred from what was played. Every visible card is pinned down in 88.7% of
   positions.
3. **XMage loads the position** by placing every card directly in its zone, then resumes the game.
4. **The turn is replayed.** The engine plays both players' recorded moves. It tries up to 12 variants of the
   details 17lands doesn't record: the order of spells, the first or second main phase, when an instant was cast,
   how blockers were assigned, and modes. A spell's target is chosen by its fate: the creature the snapshot says
   died.
5. **A turn counts only if it matches the next snapshot** (life totals, the player's hand and both boards), with
   every recorded play, attack and block actually happening. **86.8% of the players' own turns and 83.5% of their
   opponents' turns matched.** Every decision in a matching turn becomes a training example.

Some labels are certain, such as an attack or a block. Others are inferred, such as the order of two spells cast
in the same turn. Certain examples count fully in training and inferred ones count half.

Rebuilding all 161,206 games took 3.6 hours on one rented 28-core machine and cost $1.82.

### 2.5 What the networks learn from

About 75 decisions per game, of six kinds. Most of them are about timing: when to act and when to wait.

| Decision | Training examples | Test examples |
|---|---|---|
| What to play at the start of the player's turn | 1,239,106 | 60,391 |
| Later in the player's own turn: the second main phase, combat, the end of the turn | 5,573,723 | 269,685 |
| In the opponent's turn: instants, flash creatures, answers to a spell | 2,218,297 | 106,215 |
| "Attack with this creature?" | 1,391,893 | 68,980 |
| Blocks: which attacker each creature blocks | 314,690 | 15,816 |
| Spell targets the outcome reveals | 209,905 | 10,319 |
| **All** | **10,947,614** | **531,406** |

**What the value learns from.** The target for each position is the game's result, smoothed by TD(λ = 0.99): a
blend of the final result and the network's own later estimates. At λ = 0.99 the targets stay close to the result
but are a little less noisy. We use 16 positions per game in each pass over the data. All the positions of a game
share one result, and with more of them the network starts memorising games instead of learning about positions.

## 3. Models and training

### 3.1 Two networks

Both networks see the same input. MageZero describes a position, from the acting player's seat, as a set of about
800–1,500 features: each card in each zone with its status, the life totals, the phase of the turn, and so on. The
opponent's hand is only a count. Both networks have four outputs:

- **what to do with priority:** cast this, play that, or pass;
- **yes/no questions**, such as "attack with this creature?";
- **choices of target**, including which attacker to block;
- **the value:** the chance of winning from here.

| | Transformer | MLP |
|---|---|---|
| Design | one transformer layer, width 512: a tuned version of MageZero's own network | a cheaper design: the features are pooled into one vector, then 2 feed-forward blocks of width 1,024 |
| Size | 40.6 million parameters | 115 million parameters, mostly the feature table |
| Training | 3 passes over the data, 4.3 hours on one GPU | 1 pass, 39 minutes on the same GPU |
| Cost to evaluate one position on one CPU core | ~36 ms | ~3.7 ms |

We trained the MLP because the transformer is too slow for searching on ordinary CPUs.

### 3.2 Training

- **Moves:** each network learns to put high probability on the move the human made. When 17lands only tells us the
  set of things a player did that turn, it learns to put high probability on that set.
- **Value:** it learns to predict the TD targets above.
- **Passing.** At first every network passed far too often in the opponent's turn: 97–99% of the time, against
  93.4% for the humans. The networks usually knew *what* the human cast, but not *when* to act. Counting the
  examples where the human acted three times fixed this. The final networks pass 92.8–93.1% of the time, against
  93.5% for the humans.

### 3.3 Hyperparameter search

We tuned on 10% of the training games, then checked the best settings on 30% before the full runs: about 65
transformer runs and 100 MLP runs. Appendix A summarises what mattered. In short:

- MageZero's default network trained unstably until its learning rate and initialisation were fixed.
- More data helped more than any change to the network.
- The MLP's feature table didn't learn at all until we raised its learning rate 30×.

## 4. Results

### 4.1 Imitation and value, on held-out games

![Four panels of validation curves against training examples seen, in millions. Policy loss: the transformer (blue) falls to 0.234 over 33 million examples (3 passes); the MLP (green) falls to 0.227 in 11 million (1 pass), below the transformer throughout. Picking the player's move: MLP 0.832, transformer 0.829. Attack accuracy: MLP 0.870, transformer 0.856. Value AUC: MLP 0.787, transformer 0.781.](img/019-curves-light.png)

*Figure 2. How the two final networks learned, measured on validation games. The transformer trained for three
passes over the data, the MLP for one. Most of the transformer's progress came in its first pass.*

On the test games, scored once after training (up to 20,000 examples of each kind):

| Test games | Transformer | MLP | For reference |
|---|---|---|---|
| Policy loss (lower is better) | 0.234 | **0.229** | |
| Picks the move the player made, when they acted | 0.825 | **0.829** | guessing: 0.468 |
| Attacks: right decision | 0.863 | **0.873** | |
| Blocks: right attacker | 0.748 | **0.749** | |
| Spell targets: right target | 0.709 | **0.730** | |
| Value: predicts the winner (AUC), all positions | 0.781 | **0.784** | |
| Value: AUC at the start of a turn | 0.752 | **0.756** | the heuristic: 0.661 on earlier games' turn starts |
| Value: AUC in turns 1–2 | 0.632 | **0.641** | |
| Value: AUC in turns 7–9 | 0.837 | **0.842** | |
| Passes in the opponent's turn | 0.928 | **0.931** | the humans: 0.935 |

AUC is the chance that the value rates the eventual winner's position above the loser's: 0.5 is a coin flip and 1.0
is perfect. The heuristic's figure comes from an earlier set of held-out games (docs/017 §4.1), so the comparison is
only indicative.

- **Both networks imitate top players closely.** When a human acted, both pick that play more than four times in
  five, and they pass about as often as the humans do.
- **The value improves as the game goes on.** Early in a game a position is mostly the two decks, and the value
  predicts the winner only modestly. By turns 7–9 it does well.
- **One pass of the MLP beats three passes of the transformer,** at a seventh of the training time. A second MLP
  pass made it worse: at this size, accuracy comes from new games, not from seeing the same ones again.

### 4.2 Strength: games against the heuristic bot

**The bots:**

- **The baseline, `heuristic@100`:** MageZero's hand-written evaluator, which scores life, cards and the board,
  searching 100 simulations per decision. That looks about a quarter of a turn ahead.
- **Policy alone:** the network's most likely move at every decision, with no search.
- **`network@N`:** search with N simulations per decision, guided by the network's policy and judged by its value.

**Search with hidden cards.** Neither bot sees the other's hand or the order of either library. Every simulation
re-deals the hidden cards consistently with what has been seen: information-set search (IS-MCTS; Cowling et al.,
2012). Both bots do know the 40 cards in the opponent's deck. We meant them to guess it from real 17lands decks that
fit the cards revealed so far, but the service that makes the guess failed on every call (a data file was missing
from the game machines), and each search fell back to the opponent's real decklist. We found this afterwards, in
§4.6, which also plays the guessed version.

**Paired games.** The decks are 3,150 top players' decks held out of experiment #1's training. (Most of their drafts
are among this experiment's training games, so the networks have seen these players' games with these decks, but
never these deals or this opponent.) Each pair of decks is
played twice with the bots' roles reversed, so each bot plays each deck once and goes first once. The luck of the
decks and of going first cancels within a pair. In our games this made the results 30–60% less variable than the
same number of independent games.

Every network and every search budget replays the same deck pairs with the same shuffles. So "MLP against
transformer" compares the same games, differing only in the network. A game that reaches 50 turns counts as half a
win. Games that crash the engine (1–5%) are left out.

![Line chart of games won against the baseline (the heuristic bot at 100 simulations) by the network bot's simulations per decision. The transformer (blue) won 40% of 2,435 paired games with its policy alone, 55% of 116 at 100 simulations, 61% of 99 at 300 and 62% of 98 at 1,000. The MLP (green) won 38% of 104, 56% of 103, 64% of 103, 64% of 100 at 1,000 and 69% of 100 at 3,000. A dashed line marks 50%.](img/019-ladder-light.png)

*Figure 3. More search, more wins. Each point is one set of paired games against the heuristic bot at 100
simulations. The transformer was not run at 3,000 simulations, and neither network at 10,000.*

**With equal search on both sides:**

| Match | Result |
|---|---|
| Transformer@100 against heuristic@100 | won 55% of 116 paired games |
| MLP@100 against heuristic@100 | won 56% of 103 paired games |
| Transformer@1000 against heuristic@1000 | won 56% of 95 paired games |
| **All three** | **won 56% of 314 paired games** |

- **The policy alone is not enough.** Even playing its best move every time, it wins only 4 games in 10 against a
  heuristic searching 100 simulations. It plays like a human but can't look ahead, so it walks into tactical
  mistakes that a shallow search catches. Picking moves at random in proportion to the policy, as a more human-like
  opponent would, loses another 3 points.
- **Search is where the strength comes from.** 100 simulations add 15–18 points and 300 add 20–26. Past 300 the gains
  slow down: the MLP wins 64% at both 300 and 1,000 and 69% at 3,000, all within the ±9 points that about 100 games
  allow. That fits experiment #3's finding that search is limited by how well its value judges positions.
- **With equal search, the network bot has a small edge,** 55–56% in each of the three matchups. Moves imitated from
  people plus a value learned from their results are at least as good as the hand-written heuristic searching just
  as hard.
- **The MLP plays as well as the transformer.** On the same games, the two are within 3 points of each other at
  every budget. The MLP's games also finished in about half the time, though they ran on a faster machine; the
  clean measure of its speed is the tenfold cheaper evaluation (§3.1).

### 4.3 Search makes the bot stronger, and less human

Winning and imitating pull apart. We took 1,000 held-out decisions by top players
([docs/016](016-search-benchmark-results.md)'s benchmark, rebuilt from this experiment's test games) and measured how
often each bot makes the human's choice. The score averages three kinds of decision (cast or pass, attack, block);
always giving the same answer scores 0.50.

| Bot | Policy alone | 100 simulations | 300 | 1,000 | 3,000 |
|---|---|---|---|---|---|
| Transformer | **0.775** | 0.748 | 0.732 | 0.729 | 0.719 |
| Heuristic | – | 0.641 | 0.648 | 0.642 | 0.658 |

The policy agrees with top players most when it doesn't search. Search moves its choices away from theirs, mostly
on casting and attacking, while the same search wins more games (§4.2). Search fixes some human-style mistakes, but
it also makes errors of its own, judged by a value learned from results. The heuristic never reaches the policy's
agreement, however much it searches. Keeping the human sense of the game while adding search is the main problem
for the next stage.

### 4.4 Do the models value cards like humans do?

17lands' best-known statistic is a card's **games-in-hand win rate:** how often a deck wins the games in which that
card was drawn. Players use it to rank cards. If a model plays like people do, the same statistic computed from its
own games should rank the cards the same way.

We played **10,000 games of each model's policy against itself**, the transformer's and the MLP's separately (its
most likely move every time, no search, about $0.00026 a game). We then compared each card's win rate in each
model's games with 17lands' (791,000 games), using the Spearman rank correlation.

**A caveat: these are cheap games played with no search.** Every move is the policy's first instinct, with no
look-ahead. That is exactly where the policy is weakest (§4.2), and removal suffers most: when to use it, and on
what, are the decisions a look-ahead helps with. We expect a large number of self-play games *with* search to give
more human-like card ratings.

**How high could it be?** Even a perfect agent wouldn't reach 1.0 from 10,000 games, because of sampling noise. We
estimate this ceiling by drawing 17lands' own games down to the same size (~17,700 player-games) and correlating them
with the full file. The range is the 5th to 95th percentile over repeated draws.

| Rank correlation with 17lands | Cards | Transformer | MLP | Transformer, random moves in proportion to its policy | Noise ceiling [5–95%] |
|---|---|---|---|---|---|
| **Commons** | 90 | **0.41** | 0.34 | 0.27 | 0.82 [0.74, 0.86] |
| Every non-basic card | 267 | 0.43 | 0.40 | **0.45** | 0.79 [0.74, 0.82] |

**Card by card.** The tables below rank cards among the 90 commons that both models drew at least 30 times in their
self-play. Each win rate is shown relative to its source's average over those 90 commons: 54.0% for 17lands, 49.7%
for the transformer's self-play and 49.8% for the MLP's. Self-play averages about 50% because every game has exactly
one winner. So "+4.0" means a deck wins 4 points more often than average when it draws the card.

*17lands' six best commons, and where each model ranks them (of 90). Win rates in points above or below each source's
average:*

| 17lands rank | Card | 17lands | Transformer (rank) | MLP (rank) |
|---|---|---|---|---|
| 1 | Bake into a Pie | +4.0 | +0.4 (44) | +1.9 (27) |
| 2 | Burst Lightning | +3.9 | −7.3 (83) | −5.4 (77) |
| 3 | Stab | +3.9 | +0.5 (43) | +1.3 (37) |
| 4 | Dazzling Angel | +3.8 | +10.9 (1) | +11.1 (2) |
| 5 | Luminous Rebuke | +3.7 | +5.9 (10) | +6.1 (9) |
| 6 | Refute | +3.6 | +2.9 (21) | +2.2 (24) |

*The other way round: the transformer's six best commons, and where 17lands and the MLP rank them (of 90):*

| Transformer rank | Card | Transformer | 17lands (rank) | MLP rank |
|---|---|---|---|---|
| 1 | Dazzling Angel | +10.9 | +3.8 (4) | 2 |
| 2 | Vanguard Seraph | +10.7 | +0.5 (41) | 1 |
| 3 | Banishing Light | +9.5 | +3.5 (8) | 5 |
| 4 | Tranquil Cove | +9.2 | +1.6 (24) | 8 |
| 5 | Healer's Hawk | +8.7 | +3.5 (7) | 4 |
| 6 | Felidar Savior | +7.7 | +3.4 (10) | 7 |

![The transformer's self-play: each card's win rate when drawn (vertical) against 17lands' (horizontal), commons on the left (Spearman 0.41) and every non-basic card on the right (0.43). Burst Lightning, Gorehorn Raider and Involuntary Employment sit far below the trend; Hare Apparent and Gleaming Barrier sit above it.](img/019-gih-transformer-light.png)

*Figure 4. The transformer: each card's win rate when drawn in its self-play, against 17lands. Simulated win rates
sit lower overall because self-play is zero-sum.*

![The MLP's self-play: each card's win rate when drawn (vertical) against 17lands' (horizontal), commons on the left (Spearman 0.34) and every non-basic card on the right (0.40). Burst Lightning, Involuntary Employment and Gorehorn Raider sit far below the trend; Gleaming Barrier sits far above it.](img/019-gih-mlp-light.png)

*Figure 5. The MLP: the same comparison for its self-play.*

- **The transformer's policy, which never searches, captures a real part of card quality.** Its 0.41 on commons is
  the best agreement we have measured at a sample size where noise doesn't dominate. Earlier agents scored
  0.20–0.46, but on only 1,000–3,000 games, where even 17lands' own games would reach only about 0.35–0.5. The MLP
  reaches 0.34.
- **At the top, the transformer agrees with 17lands quite well.** Four of its six best commons are in 17lands' top
  ten. The MLP's six best commons include three of 17lands' top ten, but also Gleaming Barrier, the lowest-rated of
  these 90 commons on 17lands.
- **Removal played without search is the weak spot, and Burst Lightning most of all.** Bake into a Pie and Stab, two
  of 17lands' three best commons, are only about average for both models: 27th to 44th. Burst Lightning, 17lands'
  second-best common, falls to 83rd for the transformer and 77th for the MLP, 5–7 points below average. It has more
  ways to be played badly than Stab. Stab gives one creature −2/−2. Burst Lightning can hit any target, a creature or
  the opponent, and it can be kicked for 4 more mana to deal 4 damage instead of 2. So the bot must decide whether to
  spend it now for 2 damage or wait to kick it, and whether to aim at a creature or the opponent. Each choice is
  another chance to waste it. Banishing Light, an enchantment that exiles an opponent's permanent when it enters, is
  the exception: 3rd for the transformer and 5th for the MLP. Both models play creatures much like the humans do.
- **Self-play spreads card win rates about twice as wide as humans do.** Across the 90 commons the standard
  deviation is about 5 points in each model's self-play, against 2.5 for 17lands, and the best commons sit 9–11
  points above average against about 4. Colour pairs show the same pattern, even more strongly (below).
- **The transformer's best move beats a random draw from its policy** on commons (0.41 against 0.27). Random draws
  waste its better timing with spells: Refute rises from 51st to 21st, and Luminous Rebuke from 23rd to 10th.
- **The two models agree with each other far more than with 17lands** (0.91 on commons). Most of the gap comes from
  what they share, the engine and the way they were trained, rather than from either network.

![Deck win rate by colour pair, in points above or below each source's average, for the transformer's self-play, the MLP's self-play and 17lands' top players. Both models put white-blue on top (about +10 points) and black-red last (−10 for the transformer, −7 for the MLP); blue-black sits near the average for both models and for top players; blue-red is about 4 points below average for both models and even for top players. Top players stay within about ±3 points.](img/019-colours-light.png)

*Figure 6. Deck win rate by colour pair in the transformer's and the MLP's self-play, and for 17lands' top players,
each relative to its own average.*

**Colour pairs.** Both models spread the colour pairs about four times wider than top players do. Deck win rates run
from 40% to 59% in the transformer's self-play and from 43% to 60% in the MLP's, against 61% to 66% for top players.

Historically, simple and self-play bots have struggled with slower, controlling decks, which win with removal, card
draw and well-timed instants. In Forge's AI self-play, cards win least often in blue (47.3% of their games, against
50.9% in white), and the AI "times instants and card advantage poorly" (Lord of the Pigs' limited simulations:
[draft-agent behaviour](https://github.com/npiguet/price-predictor/blob/master/experiments/2026-08-29-draft-agent-behaviour.md),
[scorer preferences](https://github.com/npiguet/price-predictor/blob/master/experiments/2026-08-27-scorer-preferences.md)).
Our own experiment #1 underrated blue commons more than any other colour ([docs/003](003-fdn-generalist-report.md)).

So it is encouraging to see the blue control pairs hold up here. Blue-black wins at its average rate for both
models, just as it does for top players. Blue-red, though about 4 points below average for both models, finishes
ahead of black-red. White-blue is both models' best pair. The clearest weakness is black-red, last for both models (−10
points for the transformer, −7 for the MLP), in line with their trouble with burn.

### 4.5 What it cost

| Step | Compute | Cost |
|---|---|---|
| Rebuilding 161,206 games into 12.1M decisions | 3.6 hours on one 28-core machine | $1.82 |
| Hyperparameter search | 25 hours of a rented RTX 3090, then Dama's machine | $12.55 |
| Choosing a GPU | 1.5 hours | $1.30 |
| The final transformer | 4.3 hours on Dama's machine | $0 (~$5 at rented RTX 3090 rates) |
| The final MLP | 39 minutes on Dama's machine | $0 (~$0.50) |
| Evaluation games and benchmarks | ~130 rented machine-hours, plus Dama's machine | ~$54 |
| PIMC follow-up (§4.6) | 21 hours on four rented RTX 3090-class machines | $5.31 |
| **All** | | **~$75** |

Many thanks to **Dama**, who lent the project a machine with two RTX PRO 6000 GPUs. It trained both final networks,
ran most of the hyperparameter search, and played many of the evaluation games, all at no cost to the project.

Learning from 146,000 human games cost less than evaluating the result. Playing 146,000 games of self-play at 1,000
simulations would cost about $8,000 on our engine, and those games would be played by a far weaker player than
17lands' top players.

### 4.6 Follow-up: PIMC, a faster and simpler search for hidden cards

**Motivation.** PIMC (perfect-information Monte Carlo) is a faster and simpler way to handle hidden information than
the information-set search of §4.2. It deals the hidden cards once, into one "world" that fits everything the bot
has seen, and runs MageZero's ordinary tree search on that world. IS-MCTS re-deals the hidden cards for every
simulation and shares one tree across all the deals, so the engine does 6 to 11 times as much work per simulation.
In experiment #3's benchmark ([docs/016](016-search-benchmark-results.md)), PIMC with one world agreed with top
players as often as IS-MCTS did, at less cost. Here are results where both bots use PIMC instead of IS-MCTS: the
MLP against the heuristic bot at 100 simulations, on the same deck pairs and shuffles as §4.2.

**Two versions.** Setting this up, we found that §4.2's games never guessed the opponent's deck (§4.2, "Search with
hidden cards"): every search knew the opponent's real decklist, though never its hand or library order. So PIMC ran
two ways:

- **Real decklist:** the world is dealt from the opponent's real decklist. This is the information §4.2's IS-MCTS
  games had, so the search method is the only change.
- **Guessed deck:** the world's opponent deck is drawn from real 17lands decks that fit the cards seen so far, and
  includes those cards. Neither bot knows anything it couldn't know in a real game.

**Results.**

| MLP's simulations | IS-MCTS, real decklist (§4.2) | PIMC, real decklist | PIMC, guessed deck |
|---|---|---|---|
| 0: the policy alone | 38% of 104 | 35% of 100 | 40% of 100 |
| 100 | 56% of 103 | 63% of 103 | 57% of 100 |
| 300 | 64% of 103 | 67% of 104 | 60% of 103 |
| 1,000 | 64% of 100 | **76%** of 102 | 74% of 103 |

*In the "policy alone" row, only the heuristic bot searches, so only its method changes.*

![Line chart of games won against the baseline by the MLP's simulations per decision, with both bots using the same search method. IS-MCTS with the real decklist (green): 38%, 56%, 64%, 64%. PIMC with the real decklist (blue): 35%, 63%, 67%, 76%. PIMC with a guessed deck (orange, dashed): 40%, 57%, 60%, 74%. A dashed line marks 50%.](img/019-pimc-light.png)

*Figure 7. PIMC against IS-MCTS. Each point is about 100 paired games against the heuristic bot at 100
simulations, with both bots using the method named.*

- **PIMC plays at least as well as IS-MCTS.** With the same information, PIMC scores 4–6 points more at 100 and 300
  simulations on the same games, and 12 points more at 1,000. Only the gap at 1,000 is clearly bigger than chance
  at 100 games.
- **PIMC keeps improving at 1,000 simulations,** where IS-MCTS had stalled at 300. One possible reason: PIMC's
  simulations all build one tree on one deal, so they can look further ahead than IS-MCTS's, which are spread over
  many deals.
- **Guessing the deck costs a few points:** about 3 on the same games, and 7 at 300 simulations. The guess almost
  never fails: 0.3% of searches had to fall back to the real decklist.

**Benchmark: what a game costs.** We timed both methods on one rented RTX 3090 with the same settings. At 100
simulations each played 16 deck pairs with the real decklist. At 300, IS-MCTS played 16 deck pairs, and PIMC's
figures come from its guessed-deck games on the same machine:

| Simulations | Median game: IS-MCTS | PIMC | Network bot's move: IS-MCTS | PIMC | Heuristic's move: IS-MCTS | PIMC |
|---|---|---|---|---|---|---|
| 100 | 6.1 min | **2.8 min** | 3.3 s | 2.4 s | 1.5 s | 0.3 s |
| 300 | 16.5 min | **8.8 min** | 9.9 s | 7.9 s | 1.5 s | 0.3 s |

- **PIMC games are about twice as fast** (2.2 times at 100 simulations, 1.9 at 300). The engine does one step per
  simulation against IS-MCTS's 6–11.
- **The heuristic bot's moves are 5 times faster, the network bot's only 1.2–1.4 times,** because evaluating the
  network, not running the engine, takes most of its time. Cheaper network serving is the next speed-up.
- **At 1,000 simulations** the ladders' games took a median 30 minutes with PIMC and 74 with IS-MCTS, but on busier
  machines for IS-MCTS. A separate benchmark of single decisions on one RTX 3090 found PIMC 1.9 times faster at
  1,000 simulations.

**PIMC is the better way for us to handle hidden information.** It plays as well as IS-MCTS or better, 12 points
better at 1,000 simulations. Its games are about twice as fast. It is also simpler: MageZero's own tree search on one
dealt world. With a guessed deck, it gives up only a few points. Self-play from the imitation start (§5.3) will use
it.

## 5. Discussion

### 5.1 Imitation learning is a strong, cheap start

A few dollars of compute turned public replays into a policy that plays like top players. It passes when they pass,
attacks when they attack, and partly ranks cards like them. With a little search, it at least matches a hand-written
bot searching just as hard. Our self-play agents did no better after tens of machine-hours. Experiment #1's agent
reached a similar edge over plain search (55.8%, [docs/003](003-fdn-generalist-report.md)), but its search could see
the opponent's hidden cards. The experiment #2 agents scored 45–50%. When every self-play game is expensive,
learning from experts first is the AlphaGo route, and it fits our budget.

Imitation is not the whole answer. The policy alone loses to a shallow search. The value's contribution flattens past
300 simulations. And the card statistics show blind spots, mostly around burn and removal. These are the limits of
a policy that copies moves without looking ahead, and of a value learned from one result per game.

### 5.2 Game simulation is the bottleneck

![Horizontal bars on a log scale of games each agent learned from: DraftZero #2a's training run from scratch, 2.7 thousand games; this work's top players' training games, 146 thousand; all 17lands FDN Premier Draft games, 791 thousand; then, for scale, AlphaGo's human games, 160 thousand; AlphaGo Zero's 3-day run, 4.9 million self-play games; AlphaZero's chess, 44 million; and DeepNash's Stratego, 5.5 billion.](img/019-games-light.png)

*Figure 8. The games behind each agent. Self-play systems that reached expert or superhuman play used millions to
billions of games. Human data gives us about AlphaGo's amount for free.*

| System | Games | Search in self-play | The same number of games on our engine |
|---|---|---|---|
| AlphaGo (2016) | 160k human games for its policy; 30M self-play games for its value | none | ~$8k for the 30M games, with no search |
| AlphaGo Zero (2017), 3-day run | 4.9M self-play games | 1,600 simulations | ~$270k at 1,000 simulations |
| AlphaZero (2018), chess | 44M self-play games | 800 simulations | ~$2.4M at 1,000 simulations |
| DeepNash (2022), Stratego | 5.5B self-play games | none | ~$1.4M even with no search ($0.00026 a game) |
| **This work** | **146k human games to learn from; about 38,000 games played to evaluate (about 800 of them with search)** | – | $1.82 to rebuild the human games |

The comparison is rough: a chess game is shorter than a Magic game, and chess engines are far faster. But the
conclusion holds. Self-play at the scale that made those systems strong would cost us six to seven figures.

Five things can change that:

- **More efficient self-play.** KataGo (Wu, 2019) showed how much the AlphaZero recipe itself can be tightened. With
  changes to its training process and network, it cut the compute needed about 50-fold, passing ELF OpenGo, a strong
  open replication of AlphaZero, after 19 days on fewer than 30 GPUs. Several of its ideas, such as varying the
  amount of search per move and predicting more than the final result, are not specific to Go.
- **Faster engines.** XMage is accurate but slow to search. Engines built for speed play random FDN games 16–32×
  faster (gorge, ManaBrew), and mtg-kernel is about 470× faster on a simple test deck, with FDN support in progress
  ([docs/015](015-rules-engine-comparison.md)). A 30–70× cheaper search step would bring a 1,000-simulation game
  from about $0.055 to a fraction of a cent, if the network can keep up.
- **Cheaper search.** PIMC searches one dealt world instead of re-dealing the hidden cards for every simulation.
  Its games are about twice as fast as IS-MCTS's, and no weaker (§4.6).
- **Cheaper networks.** With a fast engine, the network becomes the bottleneck. The MLP costs a tenth of the
  transformer per evaluation and plays as well, so it suits search on ordinary CPUs.
- **More compute.** Self-play is almost entirely CPU work. A rented RTX 3090 machine plays 183 games a dollar at 100
  simulations while its GPU sits mostly idle ([docs/020](020-cpu-benchmarks.md)).

### 5.3 Next: self-play from the imitation start

The next step is to let these networks keep learning from their own games, as AlphaGo did. Our first attempt at
this (experiment #2b, [docs/013](013-exp2-imitation-report.md)) failed in an instructive way. A network pretrained
heavily on human data lost its ability to keep learning, and self-play overwrote its human policy within one pass.
The follow-up is designed and built (docs/017 §6.6):

- **Start from these networks.** Before any training, check the value on positions from the bot's own games, which
  are the positions search actually reaches.
- **Stay close to the human policy.** Penalise the policy for drifting from the imitation policy, less and less over
  training, as in piKL, Diplodocus and AlphaStar. Mix human decisions into every training batch.
- **Train the value alone in one variant,** with the policy frozen, since the value is what search most needs
  improved.
- **Detect collapse cheaply,** on held-out human decisions, before paying for games.

The MLP makes this affordable: at 100 simulations a game of self-play costs about half a cent.

## Acknowledgements

Thanks to 17lands for publishing its replay data; to Dama for lending the project two RTX PRO 6000 GPUs; to Will
Wroble for MageZero; and to the XMage developers for an engine that implements every card.

## Appendix A: Hyperparameter search

**Protocol.** Each run trained one pass over the same 10% of the training games (1.1M decisions) and was scored on
the same validation examples, 20,000 of each kind. Two runs of the same settings with different random seeds showed
how much results vary by chance: about 0.006 in policy loss for the transformer and 0.0025 for the MLP. Promising
settings were rechecked on 30% of the games. Most of the search was a hill climb, each result suggesting the next run.
The decisions that shaped the final runs (two settings for the final transformer, the MLP's number of passes)
followed rules fixed before their results came in. Details: docs/018, "Stage 2" and "The MLP".

**The transformer (~65 runs).** It started from MageZero's default: 2 layers, width 512, learning rate 3e-4.

| Finding | Evidence (validation, 10% of the games unless noted) |
|---|---|
| The default trains unstably | at learning rate 3e-4, 4 layers and width 768 learned no attack or value head at all |
| Three fixes stabilise it | learning rate 1e-4 (every head better); a smaller embedding initialisation (attack +0.026, value AUC +0.05); pre-LN (policy loss −0.040) |
| Low learning rates, then cosine decay | 5e-5 to 1e-4 best; warm-up then cosine decay from 1e-4 balanced the policy and the value |
| Depth and width don't pay | 2 layers lost at 10% and 30% of the data (policy loss +0.006 at 30%) at half the speed; width 1,024 tied width 512 |
| 16 value positions per game | 16 instead of 4 helped the shared network most (policy loss −0.020); using every position let the value memorise games (validation log-loss 0.61 → 0.85) |
| TD(λ) trades the heads | lower λ helped the policy and higher λ the value; 0.99 chosen (value log-loss 0.583 against 0.599 at 0.95) |
| The passing fix | counting the examples where the human acted three times: passing in the opponent's turn 97.1% → 93.8% (humans 93.4%), policy loss −0.007; ten times overshot |
| A separate value tower | best policy at 10% (−0.012) but only −0.005 at 30%, for 2.3× the training time: not adopted |
| Data beats everything | 30% of the games for one pass beat every recipe change (policy loss 0.262 against 0.287); three passes over 10% recovered most of the policy's gain but none of the value's |

**The MLP (~100 runs).** It started from MageZero's MLP: the average of the feature embeddings, then ReLU blocks of
width 512.

| Finding | Evidence |
|---|---|
| How features are pooled matters most | the average of ~1,000 feature embeddings loses counts (how many creatures, how many cards in hand); taking the maximum instead: policy loss 0.327 → 0.291 |
| But max pooling freezes the embeddings | they moved 1–4% from their random start (23–38% with averaging) |
| Making them learn helps | the feature table at 30× the learning rate, max plus mean pooling, features seen in more than 3 positions, wider output heads: policy loss 0.258 against 0.267 (10%), 0.241 against 0.247 (30%) |
| SwiGLU, no dropout, 10% of features dropped | each −0.005 to −0.011 policy loss; heavy dropout and BatchNorm hurt |
| Repeats overfit | a second pass let the feature table memorise games; weight decay stopped that but cost what the repeats added, so the final MLP makes one pass over all the games |
| Rare moves need data | moves chosen fewer than 1,000 times in training: picked correctly 0.54 (10% of the data), 0.65 (30%), 0.71 (all) |

**Final settings:** `configs/exp4_train.yml` (transformer) and `configs/exp4_train_mlp_1ep.yml` (MLP).

## Appendix B: Reproducing

- **Weights:** Hugging Face `danbrooks/draftzero-checkpoints` (private): `exp4/stage3/` (transformer) and
  `exp4/mlp_1ep/` (MLP), each with a model card. Load either with
  `draftzero.gameplay.imitation_net.ImitationNet.load("hf://danbrooks/draftzero-checkpoints/exp4/mlp_1ep/best_policy.pt.gz")`.
- **Data:** `tools/imitation_scale/build.py splits | build | tables` on the 17lands file (docs/017 §6.8).
- **Training:** `python -m draftzero.gameplay.supervised train --config configs/exp4_train.yml` (or
  `exp4_train_mlp_1ep.yml`) `--tables-dir data/imitation_scale/h5`.
- **Games:** `tools/imitation_scale/play.py` (bots `policy_greedy`, `il_bc` with `--budget N`, `heuristic`;
  `--method ismcts` or `pimc`), with the belief service `tools/imitation_scale/belief_server.py`, which needs the
  deck pool's cache (HF `exp4/games/deckpool_FDN_PremierDraft.npz`, fetched by `deploy/exp4_games_setup.sh`). Game
  records: HF `exp4/games/runs/`. §4.6's ladders: `deploy/pimc_games.sh`, tables and Figure 7 from
  `tools/imitation_scale/pimc_games.py --figure`.
- **Card statistics:** `tools/imitation_scale/gih.py` and `gih_ceiling.py`.
- **This document's figures:** `python tools/imitation_scale/fig_doc019.py --running 1000,3000` (drop `--running`
  once the MLP's games finish). Figures 4 and 5 come from `tools/imitation_scale/fig_gih.py` on each model's greedy
  self-play games, and the card tables from `tools/imitation_scale/card_stats.py`.

## References

- Bakhtin et al. (2023). *Mastering the Game of No-Press Diplomacy via Human-Regularized Reinforcement Learning and
  Planning* (Diplodocus). [arXiv 2210.05492](https://arxiv.org/abs/2210.05492).
- Cowling, Powley and Whitehouse (2012). *Information Set Monte Carlo Tree Search.* IEEE Transactions on
  Computational Intelligence and AI in Games 4(2).
- Jacob et al. (2022). *Modeling Strong and Human-Like Gameplay with KL-Regularized Search* (piKL).
  [arXiv 2112.07544](https://arxiv.org/abs/2112.07544).
- Perolat et al. (2022). *Mastering the game of Stratego with model-free multiagent reinforcement learning*
  (DeepNash). Science 378:990. [arXiv 2206.15378](https://arxiv.org/abs/2206.15378).
- Silver et al. (2016). *Mastering the game of Go with deep neural networks and tree search* (AlphaGo). Nature
  529:484.
- Silver et al. (2017). *Mastering the game of Go without human knowledge* (AlphaGo Zero). Nature 550:354.
- Silver et al. (2018). *A general reinforcement learning algorithm that masters chess, shogi, and Go through
  self-play* (AlphaZero). Science 362:1140. [arXiv 1712.01815](https://arxiv.org/abs/1712.01815).
- Vinyals et al. (2019). *Grandmaster level in StarCraft II using multi-agent reinforcement learning* (AlphaStar).
  Nature 575:350.
- Wu (2019). *Accelerating Self-Play Learning in Go* (KataGo). [arXiv 1902.10565](https://arxiv.org/abs/1902.10565).
- 17lands. *Public datasets*, FDN Premier Draft replay data. CC BY 4.0.
  [17lands.com/public_datasets](https://www.17lands.com/public_datasets).
