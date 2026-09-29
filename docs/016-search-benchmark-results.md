# Search benchmark, first experiment: results

**Status: in progress, draft.** Written 2026-09-29 while the runs finish. The plan is
[docs/012](012-search-benchmark.md) §2; this report follows its order. Numbers marked
*(pending)* are filled in as runs complete.

## Summary

*(pending)*

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

*(pending: pods, throughput, the offline re-timing on the reference pod)*

## 2. E0: references and noise

*(pending)*

## 3. E1: the hidden-information test

*(pending)*

## 4. E2: method × budget

*(pending)*

## 5. E2b: the backprop discount

*(pending)*

## 6. Discussion

*(pending)*

## Appendix: reproducing

*(pending)*
