# Searching for a self-play recipe that scales: laptop experiments on gorge

*Status (Sunday 11 October 2026, 6:00 AM PT): stage 0, E1 and E2 done (§6); E3's loops running. Everything runs on
Dan's laptop (an 8-core M1 Pro); r1 is down. The code is on the
[`gorge-train`](https://github.com/danieljbrooks/draft-zero/tree/gorge-train) branch (`gorge/recipe/`).*

**The goal (Dan, 11 October): find a recipe that keeps getting better on millions of games,** not just the best bot
a laptop can train in a day. A laptop plays about 10^5 cheap games a day, so each experiment here measures a *slope*:
how strength and value quality grow with games, per recipe, at equal compute. The recipe whose curve is still rising
gets the long run on r1 or pods, as the draft plan [docs/030](030-gorge-rl-plan.md) (uncommitted, another session's)
lays out.

Terms, as in docs/027:
- A **search** at N **simulations** plays N short look-ahead lines before each decision.
- The **full search** is gorge's search with our patches 0002-0007: casts, modes, choices, multiple targets and the
  opponent's decisions in the tree.
- A **policy** says which moves to try first, a **value** the chance of winning.
- **Cheap games** run the search at 25 simulations and record one searched decision in four.
- **Paired games** play each deck pair twice with the seats swapped.
- A **generation** is one round of self-play followed by training the next network.

## 1. What we start from

| Result | Source |
|---|---|
| Best bot so far: the full search at 100 simulations with a width-64 MLP (0.6M parameters) inside the engine, trained on 50 minutes of its own cheap games: 58.2% of 600 paired games against the full search without a network, 62.2% of 1,400 against the original plain search | docs/027 §4.5 |
| Its loop of cheap games (C1, C2) gained a little per generation; generation 3 never ran (r1 closed). The networks were lost with the machines | docs/027 §4.7 |
| Games, not positions, limit the value; the value's held-out loss stopped improving near 30,000 games, for every network tried | docs/027 §6 |
| Visit counts at 100 simulations over ≤6 candidates teach the policy little; completed-Q targets with a blended value target (loop B) kept improving where visits with the result (loop A) stalled | docs/027 §4.2 |
| The value carries most of the gain with search; λ ≈ 0.99 blends; judge a recipe over at least two generations | docs/029 |

## 2. How a recipe is judged

Every network is scored three ways, at fixed settings, so curves are comparable across experiments:

| Measure | How | Cost on the laptop |
|---|---|---|
| **Strength** | the full search at 100 simulations with the network, against the full search without one (the **zero point**), 500 deck pairs (1,000 paired games) on the eval decks, seed 3131 | ~20-30 minutes |
| **Value quality** | held-out log-loss and AUC on 600 cheap games on the eval decks (never trained on) | seconds |
| **The policy alone** | the network's top move, no search, against gorge's bot | ~5 minutes |
| **Against a fixed baseline** (added Sunday 8 AM, Dan) | the full search at 100 simulations with the network against gorge's heuristic bot, which never changes: 500 deck pairs (1,000 games), every generation | ~20-30 minutes |
| **The search ladder** (added Sunday 8 AM, Dan) | the best kind of network inside the full search at 100, 500 and 3,000 simulations against gorge's bot, beside the search without a network at the same simulations: 200 games a level (100 deck pairs; one standard error about 3 points), enough to see whether strength still rises | one level between generations |

The zero point moves with the engine's search; the bot is the absolute yardstick, so every generation's number against it is comparable across experiments and days. The ladder shows how a recipe's network scales with search: whether its gain over the search without it grows, holds or vanishes as simulations rise, and where more search stops helping.

Promising networks also play the previous best head to head. At 1,000 paired games about 3 points is within chance.
Single matches mislead, so a recipe's verdict rests on its *curve*: several points along games or generations.

## 3. The experiments

Ordered, each answering one question. After each, the run log (§6) records the hypothesis, the result and the next
step: extend a run, or start a new one.

**E1. Does a bigger network keep learning from more games?** *(offline: minutes per arm)*
- **Data:** stage 0's games, the full search without a network, cheap: 40,000 games, extended later.
- **Arms:** MLPs of four sizes (width 64 / 128 / 256 / 384, 0.6M to ~5M parameters), each trained on 5k, 10k, 20k
  and 40k games.
- **Read:** held-out value loss and policy cross-entropy against games, one line per size. If the wider networks
  keep falling where width 64 flattens, the 30,000-game plateau was capacity; if every size flattens together, it is
  the game's luck, and gains must come from search and targets.
- **Then:** the best two sizes play the strength match. The engine's cost of a wider network is part of the verdict:
  a network has to beat the zero point at equal time, not only at equal simulations.

**E2. Which targets?** *(offline on the same games)*
- **Arms:**
  - policy target: visit counts, or completed-Q at two strengths (gorge's `cq`, the counterpart of docs/029's
    Q-tilted target);
  - value target: the result alone, or blended with the search's root value (0.3, 0.5).
- **Read:** held-out losses, then the strength match for the best two.

**E3. The loop** *(the main event)*
- **Each generation:**
  - cheap self-play with the previous network inside the search (in-engine MLP, the prior choosing the candidates);
  - the next network trains on a window of recent games;
  - every generation gets the strength match and a head-to-head against the previous one.
- **Recipes, at equal laptop compute:**

| Loop | Network | Training | Window |
|---|---|---|---|
| **A** (docs/027's cheap-games loop, the control) | width 64, fixed | from scratch each generation | every game so far |
| **B** (grow) | width chosen by E1, grown when E1's curves say so | warm start from the last generation | every game so far |
| **C** (more search in self-play) | as B | as B | as B, with self-play at 50 simulations instead of 25 (fewer, better games) |

  The targets are E2's winners for all three.
- **Read:**
  - strength and value loss against total games, per loop;
  - each generation's step over the last.

  The recipe for millions of games is the loop whose curve is still rising at the end, not the one with the
  highest single point.

**E4. Extend the best loop** for as long as it keeps rising, with paired games every generation. Re-check E1's
question on the bigger data as it accumulates: does a wider network now pull ahead?

**Ideas held back for later rounds,** run if the loops stall:
- policy-surprise weighting (KataGo);
- auxiliary targets (life totals, turns left);
- Gumbel root selection at low simulations (docs/030's stage 0, engine work);
- a value-only network beside the policy;
- more candidates (`topk`) as the prior improves.

## 4. Laptop budget

Measured on the laptop (8 cores, M1 Pro):

| Work | Speed |
|---|---|
| Cheap self-play, full search, no network | ~22,000 games an hour (7 workers) |
| The 100-simulation strength match, no network on either side | ~4,500 games an hour |
| With the in-engine MLP | to measure (docs/027: about 2× slower) |
| Training a width-64 MLP | to measure (MPS; docs/027's took minutes on a GPU) |

A day is roughly:
- 100,000-200,000 cheap games;
- a few dozen trainings;
- 20-30 strength matches.

Every job runs under `caffeinate` and `nohup` with a hard time limit, so it survives the app closing (the laptop
sleeping paused docs/029's watchers for hours). The disk has ~15 GB free; a cheap game's records take ~9 KB.

## 5. What would change the plan

| If | Then |
|---|---|
| Every network size flattens at the same number of games (E1) | Scale games and search, not network size; put E3's compute into more games per generation |
| Wider networks keep improving but cost too much inside the engine | The recipe for millions of games includes a faster network path (docs/030 stage 0: incremental encoding, batching) |
| Loop A keeps up with B and C | The simplest recipe scales; prefer it |
| A loop beats its predecessor but not the zero point | Self-play overfitting to itself: add past networks as opponents |

## 6. Run log

*Appended as results come in. Times are Pacific.*

### Stage 0: the games (Sunday 12:10-2:15 AM)

- **Held-out games:** 600 cheap games on the eval decks, 9,392 recorded decisions. Every value and policy number below
  is measured on these (never trained on).
- **Training games:** 40,000 cheap games on the train decks, the full search without a network, in 1 hour 50 minutes
  on 7 cores (about 22,000 an hour). 641,027 recorded decisions, 16 per game.
- **Storage:** a packed game took 112 KB, almost all of it the per-card features in 32-bit floats. Stored as 16-bit
  (`gorge/recipe/shrink.py`; training reads them back unchanged, the same losses to four decimals) it takes 72 KB. The
  laptop has about 12 GB free, so a loop keeps at most ~150,000 games packed at once.
- **Training is cheap; games are not.** An MLP trains at about 10,000 decisions a second on the laptop's GPU. Every
  E1 arm took 1-6 minutes, against 2 hours for the games, so the recipe question is how much each game teaches.

### E1: network width against games (Sunday 2:15-3:10 AM)

*Hypothesis: docs/027's value plateau near 30,000 games was the width-64 network's capacity, and wider networks keep
improving.*

Sixteen MLPs (widths 64, 128, 256 and 384; 0.6M to 6.3M parameters), each trained on stage 0's first 5k, 10k, 20k
or 40k games with loop B's targets (completed-Q at 0.03, value half result and half the search's estimate), up to 8
passes with early stopping.

![](img/031-e1-light.png)

| Games | Value log-loss, width 64 / 128 / 256 / 384 | Value AUC, best width |
|---|---|---|
| 5k | 0.511 / 0.507 / 0.511 / 0.511 | 0.829 |
| 10k | 0.497 / 0.498 / 0.496 / 0.501 | 0.837 |
| 20k | 0.491 / 0.492 / 0.493 / 0.497 | 0.839 |
| 40k | 0.489 / 0.493 / 0.485 / 0.488 | 0.843 |

*The value's log-loss is against the game's result on the held-out games; predicting the base rate alone scores
0.691.*

- **Width doesn't matter at these sizes.** At every game count the four widths are within about 0.005, close to the
  held-out set's noise (600 games).
- **The value still improves from 20k to 40k games, but slowly:** about 0.003-0.008 per doubling, against 0.014 from
  5k to 10k. The policy's fit keeps falling evenly with each doubling.
- **Every arm stopped early,** most after 3-5 passes: a network memorises the games it has seen. Small networks
  overfit as fast as big ones.

- **Strength:** the width-64 network on 40k games won only **51.7% ± 1.1 of 1,000** paired games against the zero
  point, short of docs/027's 56-58%. E2 found why: the policy target.

*Reading (E1 used loop B's targets, which E2 replaces; the 80k point in E2 is with the better ones): the plateau is
the data's, not capacity. A cheap game's value labels are mostly luck (one result per
16 decisions), so more of the same games buys little. What remains to test: whether better games (a network in the
search, E3) or better labels (E2) move the curve more than more games do.*

### E2: which targets (Sunday 3:10-5:45 AM)

*Hypothesis: loop B's targets (docs/027 §4.2) are the right default: completed-Q for the policy, the value half
result and half the search's estimate.*

Width-128 MLPs on stage 0's first 40k games, with six target pairs; then the strength match (§2) for three of them.

| Policy target | Value target | Value log-loss | Value AUC | Strength (1,000 paired games) |
|---|---|---|---|---|
| Visit counts | result | **0.473** | **0.853** | **55.2%** |
| Visit counts | half result, half the search's estimate | 0.487 | 0.842 | |
| Completed-Q 0.03 | result | 0.473 | 0.851 | 51.7% |
| Completed-Q 0.03 | 30% the search's estimate | 0.476 | 0.850 | |
| Completed-Q 0.03 | half and half (width 64, E1) | 0.489 | 0.841 | 51.7% |
| Completed-Q 0.1 | half and half | 0.490 | 0.840 | |
| Completed-Q 0.01 | half and half | 0.489 | 0.842 | |

*Strength: the full search at 100 simulations with the network inside the engine against the same search without
one (the zero point), 500 deck pairs on the eval decks. One standard error is about 1.1 points; the three matches
share their deals.*

- **Loop B's targets are not the right default here.** Docs/027's best network (56.3% of 1,400 in its final check)
  used visit counts and the result; with completed-Q the same games give 51.7%, whichever the value target.
- **The value's label: the result alone.** Blending in the search's root estimate makes a worse value (log-loss
  0.487-0.490 against 0.473; AUC 0.842 against 0.853). Without a network that estimate is the heuristic's, weaker
  than the game's result.
- **The policy's label: visit counts, in games.** Completed-Q fits its own target more closely (cross-entropy 0.93
  against 1.04 for visits; the uniform guess is 1.05), and the values are equal, yet it plays 3.5 points worse. The
  prior chooses which 6 moves the search considers (`topk=6`), and at 25 simulations completed-Q follows noisy
  estimates; a prior that is sharp and wrong drops good moves before the search sees them. Visit counts at 25
  simulations stay close to the heuristic's spread.
- **Regularising the value** (width 128, visits + result): weight decay 0.1 gives 0.4735, dropout 0.2 0.4747, a
  learning rate of 3e-4 instead of 1e-3 **0.4684** (against 0.4726). The loops use 3e-4.
- **The value overfits after about one pass** with the result as its label (best after 0.75-1.75 passes, against
  3-6 with the blended label): the networks learn each game's luck once they see it twice. More games is the remedy,
  which suits a recipe for millions of games.
- **80,000 games** (stage 0 extended by 40,000 games on the same decks, Sunday 2:15-5:25 AM), width 64, visits +
  result: value log-loss **0.4636**, AUC 0.859, against 0.4700 and 0.853 at 40,000. Still improving with each doubling
  (about 0.006), unlike docs/027's plateau.

*Next: E3, the loops, with visit counts and the result as targets, starting from the 55.2% network.*
