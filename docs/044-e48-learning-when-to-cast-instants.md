# E48: learning when to cast instants

October 2026. A follow-up to [037-instants.md](037-instants.md): the network plays instants at the wrong time. E48
is the first training run whose network learns the reactive timing a human uses, with no rule at play. The timing
itself was never the hard part. A rule can enforce it in self-play. The hard part was getting the rule into the
network, and the fix is one line of bookkeeping in the training record.

## Summary

- **The rule (react).** In self-play, an instant-speed cast is offered only in reaction: when the opponent has done
  something this turn, or at the card's last window. Firing a burn spell in your own upkeep "because you can" is not
  offered.
- **Why the rule alone teaches nothing.** The policy loss is a softmax over the candidates in each training record.
  A cast the rule removed isn't a candidate, so it isn't in the record, and its prior gets no gradient. A network
  trained with the rule in self-play (E47) plays exactly as before once the rule is off.
- **The fix (the held-cast target).** Record the casts the rule held back as extra candidates with zero visits. The
  softmax then pushes their logits down, so the network learns that they were held.
- **Result.** With no rule at play, E48's Burst Lightning goes in its own upkeep or draw step **0.2%** of the time
  (its parent: 31.6%; the base network: 44.5%). Its card ratings agree with 17Lands better than any earlier network
  (Spearman ρ 0.43 against 0.31 for the base). Strength is unchanged to slightly up: 51.9% [50.0, 53.7] against its
  parent, head to head.

## Background

**The networks.**

- **v36** is E1's network (docs/runs.md), the base.
- **E42** is v36 continued for 12,000 self-play games with a patience rule: when passing is within 0.04 of the best
  move's value, pass until the next untap. The waited move is recorded as the target. The value head trains on the
  next own turn's value as well as the outcome.
- **E47, E48 and E50** each continue E42 (version 12) for 6,000 more games:

| run | self-play | what's recorded |
|---|---|---|
| E50 (control) | E42's recipe | as E42 |
| E47 | E42's recipe + react | as E42 (the held casts are absent) |
| **E48** | E42's recipe + react | **as E47, plus the held casts at zero visits** |

The only difference between E47 and E48 is what goes into the record. The games they play are the same kind.

**The search.** Single-observer information-set MCTS at 300 simulations per move. The opponent's decklist is
hidden: it's sampled from a corpus of decks that cover the colours seen so far.

**The measurements.** All of them use the network alone, with no rule at play.

- **The litmus:** the network against itself on 2,372 deck pairings of real FDN decks. From these games come
  per-card GIH WR (the win rate in games where the card was drawn or in the opening hand), and how and when each
  card is cast.
- **The class gap:** each card's IWD (GIH WR minus the win rate when not seen) relative to the average card's IWD,
  minus the same quantity in 17Lands Premier Draft. The gap is the mean of that residual over the instant-speed
  commons minus the mean over the others. A negative gap means the network gets less out of drawing its instants
  than humans do.
- **Head to head:** 593 deck pairings, both seat orders, 1,186 games, with paired 95% intervals.

## The react rule

At a decision point with an empty stack, an instant-speed cast that would also be offered at a later window this
turn cycle is not a candidate, unless the opponent did something this turn since the seat's previous decision.
"Did something" means a new opponent event:

- a new permanent of theirs (an object, a card, a +1/+1 counter);
- an attack or block mark, so the seat's own blocks count as a combat event;
- a card going to their graveyard or exile;
- a spell or ability of theirs on the stack.

Responses (a non-empty stack) and the card's last window are untouched. The rule reads nothing about the card but
its speed.

Two earlier definitions failed, and both failures matter for any learned version:

1. **Counting since the seat's previous decision.** This treated the seat's own actions as events.
2. **Comparing a state hash with the previous decision.** This fired when attacks ended and damage wore off. Worse,
   the engine passes automatically for a seat with nothing castable, so "the previous decision" could be turns old:
   an opponent's play from two turns ago licensed an own-upkeep cast.

With either, a third of the own-turn pre-combat casts got through. Only this turn's events count now.

**React as a rule at play doesn't make the network stronger.** v36 with react at play scores 49.1% [47.3, 50.8]
against plain v36 at 300 simulations, and 48.3% [46.5, 50.1] at 1,000. It gives up some good early plays, such as
removing a blocker before combat, for timing the network then misuses.

## Why the rule alone teaches nothing

DraftZero's policy target at a recorded decision is the visit distribution over that decision's candidates. The
loss is a softmax cross-entropy over those candidates only. A move that isn't a candidate isn't in the softmax, so
neither its logit nor anything that feeds it gets a gradient from that record.

React removes the held casts from the root's candidates. So E47's records say, in effect, "among the moves offered
(pass, play a land, cast a creature), the search preferred these". They never say "and the instant was not cast".
The network learns the distribution among the moves it was offered, and its prior for the held cast stays where it
was. When the rule is off at play, the cast is offered again and the old prior picks it.

E47 confirms this. With no rule at play, its Burst Lightning goes in its own upkeep 39.1% of the time, *worse* than
E42's 31.6%. Head to head it scores 50.0% [48.3, 51.7] against E42. React in self-play without the record taught
nothing, and it slightly blurred what E42 had learned.

The same thing applies to any rule that prunes root candidates in self-play. Unless the pruned moves reach the loss,
the rule shapes the games but not the network.

## The fix: record the held casts at zero visits

At a root where react held casts back, the training record carries those casts as extra candidates with zero visits
(and zero prior in the record). The policy target puts probability 0 on them, so the softmax lowers their logits
relative to the moves that were searched. The search and the game are unchanged; only the record grows. In a smoke
test, 78 of 262 records of the reacting seat carried held casts.

This is the same move as recording the patience rule's waited move as the target: make the loss see what the rule
decided.

## Results

### Timing (the litmus, no rule at play)

| | v36 | E42 | E47 | **E48** |
|---|---|---|---|---|
| Burst Lightning in its own upkeep or draw step | 44.5% | 31.6% | 39.1% | **0.2%** |
| Burst Lightning in its own combat | | 13.3% | 11.7% | **19.4%** |
| Burst Lightning on the opponent's turn | | 31.6% | | **41.7%** |
| Burst Lightning to the face, per 100 casts | | 5.8 | 8.8 | 6.7 |
| Instant-speed casts on the opponent's turn, per seat-game | | 0.84 | | **1.11** |

E48 casts its instants reactively without being told to: in combat, or on the opponent's turn. The only change from
E47 is the record of the held casts, so the missing gradient was the whole reason E47 learned nothing.

### Card ratings against 17Lands

These use the 78 FDN commons in the litmus, compared with 17Lands Premier Draft GIH WR. Self-play centres the
networks near 50%, against about 54.6% for 17Lands' average common, so each card is compared relative to its
source's average common. The litmus with the react rule on at play (v36 + react) shows how much of the effect a
rule gives without learning.

| agreement with 17Lands | v36 | v36 + react at play | E42 | **E48** |
|---|---|---|---|---|
| Pearson r, GIH WR | 0.34 | 0.40 | 0.41 | **0.42** |
| Spearman ρ, GIH WR | 0.31 | 0.35 | 0.36 | **0.43** |
| Pearson r, IWD | 0.19 | **0.36** | 0.31 | 0.31 |
| mean absolute gap per card (points) | 3.68 | 3.77 | 3.2 | **2.85** |
| instant-speed commons (20), mean gap | −3.2 | −2.8 | −2.0 | **−1.6** |
| other commons (58), mean gap | +1.1 | +1.0 | +0.7 | +0.6 |

- The rule at play closes about a tenth of the instants' gap (−3.2 → −2.8). Learning it closes about half
  (−3.2 → −1.6).
- The cards whose value is the window gain most:
  - Fake Your Own Death: −10.1 → −1.7.
  - Fleeting Flight: −7.6 → −2.3.
  - Refute: −8.1 → −4.8.
- Some instants don't move: Burst Lightning (−5.7 → −6.3) and Giant Growth (−7.0 → −5.1). They lose value to *how*
  the search uses them, as a pump in a race or as face damage, not to *when*.

### Strength (head to head, no rule at play, 300 simulations)

| E48 against | win % [95%] |
|---|---|
| E42 (its parent) | 51.9 [50.0, 53.7] |
| E47 (the same games, no held-cast record) | 50.3 [48.5, 52.0] |
| E50 (the plain continuation) | 51.3 [49.6, 52.9] |

The record teaches the reactive timing at no cost in strength, with at most a small gain.

### What it didn't fix

- **The class gap on the litmus stays at E42's** (−2.2; E48 against E47 −0.2 [−2.1, +1.7]). The network now plays
  its instants at sensible times, but games where it holds one aren't won more often yet.
- **Refute, a counterspell, is still under-rated** (about 49.5% GIH WR against 17Lands' 57.6%). The main cause is
  tapping out: in about a fifth of the games where Refute is drawn, the seat never has it castable when the
  opponent casts a spell, because it spent the mana on its own turn. E48 plays it no differently from its control. That is a question of how the network values open mana, which react
  doesn't address.

## Takeaways

1. **A rule used only in self-play shapes the games, not the network.** It reaches the network only if the moves
   it changed reach the loss. With a candidate-softmax policy loss, a pruned move has to be recorded as a candidate
   at zero visits.
2. **Learned timing beats the rule at play.** The network trained with the record agrees with 17Lands more
   (ρ 0.43 against 0.35), plays at least as strongly, and needs no rule once trained.
3. **Timing isn't the whole instant problem.** What remains is the value of holding mana and of using a trick
   well (the target, the race), not just the moment.

E48 (version 11) is the starting point for the next round of training runs.
