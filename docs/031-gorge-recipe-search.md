# Searching for a self-play recipe that scales: laptop experiments on gorge

*Status (Sunday 11 October 2026, 12:30 AM PT): plan written; stage 0's games are being played. Results are appended
below as they come in (§6, the run log). Everything runs on Dan's laptop (an 8-core M1 Pro); r1 is down. The code is
on the [`gorge-train`](https://github.com/danieljbrooks/draft-zero/tree/gorge-train) branch.*

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

*Appended as results come in.*
