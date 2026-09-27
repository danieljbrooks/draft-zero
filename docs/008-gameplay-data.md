# Human gameplay data: imitation learning and MCTS coaching

A feasibility study, written 2026-09-26 on branch `gameplay-data`. So far this repo has asked
whether MageZero can play FDN limited well enough to generate trustworthy statistics from self-play.
This study asks the reverse question: **how far can existing human gameplay data be used?** Two
applications:

1. **Imitation learning.** Use human games (17lands, expert players) to bootstrap the policy and
   value networks.
2. **Coaching.** Given a human's game log or position, rate the actions available to them with the
   MCTS agent.

Both depend on one hard step: **mapping a log onto an XMage game state**. Most of the work went
there.

Everything here was measured on this repo's stack (experiment #1's engine: MageZero `bcc76de`, XMage
fork `5a32441c`, gen-33 checkpoint) on a local M1 Pro. "n" and 95% confidence intervals are given
where they matter.

## Summary

**Mapping works, and better than expected.**

- **Identities map completely.** Every card, token and ability id in the 791,159 FDN games resolves
  to an XMage card, token class or ability. The only exceptions are a `-1` placeholder and two stray
  ids, 589 occurrences in all.
- **Actions map completely.** Every recorded cast and land play maps to an exact MageZero action
  key. Flashback casts (2.9%) are recorded as ordinary casts; graveyard tracking gives them their
  `Flashback` key.
- **Turn-start states rebuild well.**
  - Every visible card is pinned down in 88.7% of turn-start states.
  - All 168,606 states rebuilt from a 20k-game sample pass validation.
  - A separate sample of 2,004 specs builds in XMage without a failure, about 4 ms each.
- **Whole turns replay.** A human's recorded turn is replayed through the engine with both players
  scripted. The engine lands on the next 17lands snapshot in **about 88% of turns** (87.7%, CI
  87.3–88.2, of 20,596 random turns), and 98% in the cleanest tier. Each reproduced turn yields
  about 4.8 labelled decisions, in a median 13 ms. This is an upper bound on fidelity: the
  opponent's recorded cards are put into its hand.
- **Arena logs are near-exact from the player's own seat.** The only log so far is Dan's: 2 Powered
  Cube games, not FDN.
  - The parser pairs 128 of 128 decision requests with the player's answer.
  - With 7 cards XMage lacks substituted, all 105 decisions build.

**Imitation is feasible on the data side; whether it helps play is still untested.**

- **Volume.** 17lands' public FDN file has 7.16M user turns, 1.51M of them by players at ≥ 60% win
  rate. Replay data exists for 27 other sets and one cube.
- **Labels.** Attacks and blocks map almost one-to-one onto MageZero's decisions. Priority actions
  ("cast X", "play land") are known per turn as an unordered set; turn replay turns them into
  per-decision labels with an imputed order.
- **Results** (§7.3), on 16,750 held-out human turn-start decisions. Scoring is lenient: any of the
  turn's plays counts as a correct first move.

  | Model | Top-1 |
  |---|---|
  | gen 33's priority head (at chance with 4+ options) | 44% |
  | gen 33's priority head, scored as the search would use it | 52% |
  | "Play a land, else the biggest spell" heuristic | 65% |
  | **gen 33's trunk with heads retrained on 30k human decisions** | **73%** |

  - The retrained heads also win on real spell choices: 58% against the heuristic's 40%.
  - Accuracy is still rising with more data.
- **Attacks and value.**
  - Gen 33's attack head (the one policy prior exp #1 used) predicts human attacks worse than a
    two-line power/toughness rule: 70% against 74%, n = 2,484.
  - A human-trained attack head ties the rule on accuracy and ranks better: AUC 0.83, against 0.76
    for gen 33.
  - Human outcomes improve the value head: AUC 0.685 against 0.654, paired gain +0.031 (CI
    0.018–0.042).
- **A human priority head can't help as exp #1 was configured.** Exp #1 ran with the priority prior
  off, so the search never reads that head. The attack prior and the value head were in use, so
  human-trained versions of those would matter at once. Whether any of this makes MageZero stronger
  needs a pod A/B (§7.5).

**Coaching works end to end, but it isn't valid yet.**

- **The pipeline runs.**
  1. A position from a 17lands row or an Arena log is rebuilt in XMage.
  2. The opponent's hidden hand is sampled from a belief model built on 17lands decks.
  3. MCTS scores each option.
  4. The human's choice is graded by a nominal win-probability loss.
- **Neither evaluator tells good players from bad.** The coach's regret does not fall with player
  skill, either with the offline heuristic (Spearman ρ = +0.09, CI −0.05 to +0.22, n = 242) or with
  gen 33 (ρ = −0.03, CI −0.18 to +0.13, n = 143).
- **The evaluator dominates the verdict.** The two evaluators pick the same best action only **51%**
  of the time. Hidden information matters much less: guessing the opponent's hand instead of knowing
  it changes the best action in 19% of decisions (n = 119), against 11% between two independent
  searches.
- **The test itself is weak.** Detecting a realistic skill effect needs about 800–3,000 decisions.
- **Search noise looks like mistakes.** Most "mistakes" at 300 simulations were artefacts of rarely
  visited options. The offline coach now re-searches such faults at 3,000 simulations before
  reporting them.

**Engine problems found on the way matter for experiment #2 regardless of human data** (§6):

- MageZero's search is clairvoyant: it searches the real game, hidden cards included.
- Exp #1's network saw the opponent's hand, and the YAML key meant to turn that off is silently
  ignored.
- The attack target is an arbitrary HashSet element when the defender controls a planeswalker.
- Logged seeds don't reproduce games.
- Timed-out games give player B a win.

**Recommended next steps** (§11):

1. Decide whether exp #2's network sees the opponent's hand. Both goals need a hidden-information
   network.
2. Pretrain the attack and value heads on human data, and A/B a human priority prior on a pod.
3. Validate coaching with a better value network on a GPU before showing grades to anyone.
4. Ask 17lands for research access to the raw game histories they hold.
5. Keep your own FDN Arena logs with detailed logging on.

---

## 1. How this was studied

Four phases, each run by parallel agents with adversarial reviewers. Between phases I integrated the
results, spot-checked the claims that carried most weight, and committed:

1. **Research.**
   - Web: 17lands semantics, other data sources, prior art.
   - XMage and MageZero code.
   - An empirical study of the 438 MB FDN replay file.
   - Dan's own Arena log.
   - The id mapping.
   - A completeness critic listed 14 contradictions between these and resolved 13. One stays open:
     whether 17lands' stored game histories keep the annotations that player actions could be
     recovered from.
2. **Build.** A shared state format (StateSpec) plus these modules:
   - the 17lands parser and reconstructor;
   - the Arena parser;
   - a Java XMage bridge;
   - an opponent belief model;
   - behavioural fingerprints.

   Each module got a reviewer who tried to break it. Across all phases the reviewers logged 106
   findings (1 critical, 29 major) and fixed 86 of them in place, most with regression tests (some
   fixes were to documentation or comments). The other 20, 4 of them major, were handed on, and the
   integration phase fixed or worked around several. Many were of the kind "the state validates but
   describes a different game".
3. **Integration.** Fixes across modules, turn replay and the coaching driver.
4. **Experiments.** Imitation, and coaching validity. Each was re-checked by an independent verifier
   that recomputed the headline numbers from the recorded outputs, without re-running the searches
   or the full training. The headline numbers reproduced, a few secondary ones were corrected (for
   example the per-option Q correlation, 0.485 to 0.498), and 10 analysis fixes followed.

## 2. Where human gameplay data exists

| Source | Fidelity | Volume and access | Best use |
|---|---|---|---|
| **17lands `replay_data` (public)** | Per turn: which cards were drawn, played and cast, attackers, blockers, deaths; end-of-turn hands (own), boards, life. No order, targets or tapped state. | 791k FDN Premier games; 108 files, 13.2 GB, 28 sets plus Powered Cube, STX to MSH. **CC BY 4.0.** | Imitation at scale, value pretraining, opponent priors |
| **17lands raw histories (private)** | Every Arena game-state message of each uploaded game. The 17lands client uploads them, but not the player's choices (only SelectN answers). A stored 2021 replay also lacked the action annotations the choices would be inferred from, so whether these histories support imitation is open. | Millions of games since early 2021. Viewable per game on the site; no export or API; site terms forbid outside use. | Imitation at scale, if 17lands grants research access |
| **Arena `Player.log` with detailed logs** | Per action: full state from the player's seat, the offered options and the chosen answer. The opponent's hand is never sent. | Only the player's own games. Overwritten on every launch (one previous copy kept). | **Coaching** (exact), a gold-standard evaluation set |
| MTGO `GameLog` files | Per action, with targets; hands not named | Own games only; EULA restricts reverse engineering | Coaching for MTGO players |
| Penny Dreadful bot logs (MTGO) | Per action, spectator view | Public, constructed format | Not limited |
| XMage replays, Forge, Cockatrice | XMage's replay feature is dead code; Forge has 374 puzzle states; Cockatrice has no rules engine | – | Forge puzzles as a coaching test set |
| Kaggle, HF, papers | Draft data, and card images on Hugging Face | – | – |

No public per-action corpus of human limited games exists. The only public per-action logs are Penny
Dreadful's spectator logs of constructed MTGO games, with no hands. The closest prior art is
`NMaass/mtg-rl-tools` (MIT, July 2026), which turns Arena logs into decision records and mirrors
Arena snapshots into XMage with face-down placeholder cards.

**Terms.** Reading your own Arena log is accepted practice (trackers are tolerated, and the setting
is called "Plugin Support"). Wizards' terms ban software that "grants any user an advantage", so
coaching should run **after** a game, never live.

## 3. What a 17lands replay row records

### 3.1 Size and coverage

The FDN PremierDraft file has 791,159 games from 135,418 drafts, played 2024-11-12 to 2024-12-15. It
covers the first 5 weeks of FDN and was never refreshed.

| | Games | User turns |
|---|---|---|
| All | 791,159 | 7,162,148 (6.86M with an action) |
| Win-rate bucket ≥ 0.60 | 166,763 | 1,509,873 |
| Win-rate bucket ≥ 0.64 | 63,289 | 571,962 |

FDN also has TradDraft (137k Bo3 games, 73 MB), Sealed (25 MB) and TradSealed (3 MB) files, not used
here.

### 3.2 Semantics, verified against the data

17lands publishes no column dictionary. These were worked out from the data:

- **Rows.** One row per game, not per turn as the datasets page says.
- **Turns** are counted per player. The order is U1, O1, U2, … when the user is on the play. The
  `num_turns` column is the per-player count. Columns stop at turn 30, which cuts off only 23 games.
- **Prefixes.** The `user_turn_`/`oppo_turn_` prefix says whose turn it was; a `user_`/`oppo_` infix
  says whose cards.
  - Un-prefixed per-turn fields belong to the active player: draws, discards, land plays, creature
    and non-creature casts, attackers.
  - `creatures_blocking` belongs to the defender.
- **Cast columns.**
  - `non_creatures_cast` holds only non-creature *permanents*.
  - Instants and sorceries go to `*_instants_sorceries_cast`, for both players.
  - Nothing the non-active player does is recorded except instants, sorceries and abilities: its
    draws, discards and permanent spells (flash creatures and the like) are missing.
- **Hidden information follows the user's view.**
  - The user's hand is named.
  - The opponent's hand is a count.
  - The opponent's face-down cards appear as id 3.
- **Tokens** have their own ids (expansion `TFDN`). They appear in the board columns but never in
  the `killed` columns.
- **`cards_drawn`** is chronological: 91% of self-drawing turns agree, against 50% by chance. So its
  first entry is the draw-step card, unless an upkeep trigger (Phyrexian Arena, Scrawling Crawler)
  drew first.
- **End-of-turn snapshots** are taken before the cleanup discard (a hand of 8 shows in 0.1% of
  turns).

### 3.3 What is missing

These are never recorded:

- the order of events inside a turn, and main phase 1 versus main phase 2;
- targets (about 19% of turns cast a targeted spell);
- modes and X values;
- tapped state, counters and attachments;
- the legal options the player had.

Some things can only be recovered in part:

- **Blocks.** The pairing of blocker to attacker is unique in 87.6% of combats with blocks, 81.6
  points of it trivially (only one attacker was blocked). With flying, reach and menace in the
  pairing model (added in the cleanup), it is about 88%.
- **The opponent's untapped lands.** Their number is known in 97.7% of states, but which lands they
  are is ambiguous in 32% by the research's coarse test, and in 25% of specs after the reconstructor
  matches colour pips.

### 3.4 Mirrored games

**13.7% of rows are the same game recorded by both players.** They can be matched on the board after
the on-the-play player's third turn plus `game_time`. This gives 54,138 pairs with 0 ambiguous
matches, and the hand sizes and turn-5 creature casts agree in 99.994% of the 53,328 pairs that
reach turn 5.

For these games both hands, both decklists and both draw sequences are known. That provides:

- ground truth for the opponent-hand model;
- hindsight comparisons for coaching;
- perfect-information encodings.

The pair lists stay in `data/` (gitignored) and should never be published. They link two anonymized
records.

### 3.5 The skill bucket includes the game itself

`user_game_win_rate_bucket` is computed over games that include those in the file. Users with ≤ 10
games in bucket 0.40 win 40.5% in-sample, with no regression toward the 54.6% mean. So filtering on
the bucket selects on outcome, most strongly for users with few games.

**Skill contrasts in this study use only users with `n_games` ≥ 100.**

## 4. Mapping 17lands onto XMage

### 4.1 Identities

| What | Coverage |
|---|---|
| Card ids (all 791k games) | 1,166 of 1,169 map. That is **100% of 59.5M event occurrences**; the misses are a `-1` placeholder and two ability ids that leaked into card columns. |
| Casts and land plays → MageZero keys (`Cast X` / `Play X`) | 100% (12.9M casts, 10.0M land plays) |
| Tokens | 36 of 37 Arena FDN tokens map to XMage token classes by id (Cat 1/1 and Cat 2/2 are different ids) |
| Ability ids → XMage abilities | 100%. 26 ids (17% of ability events) are in neither 17lands' table nor today's Arena database; their source cards were recovered by co-occurrence (right for 180 of the 189 ids that could be checked). |
| Activated non-mana abilities → exact vocab keys | 90.65%. Food and a granted Fishing Pole ability fall in the hashed tail because the vocab lists only abilities printed on set cards. |

One trap: flashback casts (2.9% of casts, mostly Think Twice) are recorded as ordinary casts, but
MageZero keys them `Flashback {cost}`. The labels detect this from graveyard tracking.

### 4.2 One state format: StateSpec

`src/draftzero/gameplay/statespec.py` defines a JSON state shared by every producer and by the Java
bridge. It holds:

- every zone for both players, with tapped state, summoning sickness, damage, counters, attachments,
  tokens and owner;
- the stack, attackers and blockers;
- the turn, phase and step, and how to enter it;
- the decklists, with `handUnknown` for hidden cards;
- provenance: source, a fidelity tier and flags;
- labels: what the human did.

It is encoder-independent. Human data is stored as specs plus labels, so it can be re-encoded for
MageZero v0.2, whose feature hash changes.

### 4.3 Rebuilding the state at the start of a user's turn

A turn-start state is the previous end-of-turn snapshot, plus the draw-step card, plus inferred
permanent status:

- which of the opponent's lands and creatures are tapped (from mana spent, attacks, and {T}-cost
  abilities);
- counters (from trigger counts, where exact);
- Aura and Equipment hosts;
- tokens by class;
- graveyards (best effort).

Two checks measure how well this works:

- **Conservation.** Does the previous snapshot plus the recorded events reproduce the next snapshot?
  It holds strictly across all zones of both players in 67.5% of half-turns. After the known
  unrecorded mechanisms are classified (token creation and death, off-turn draws and discards,
  flash, bounce, Evolving Wilds), 99.0% of half-turns have no unexplained difference; 92.8% if a
  card that left for an unknown zone (exile, bounce, library) also counts as unexplained.
- **Fidelity ladder.** How often is everything determined, apart from library order and the
  opponent's hand? Measured on 447,864 states:

| Tier | What is determined | Share |
|---|---|---|
| visible identities | every card on both boards, the user's hand, life | **88.7%** |
| + the opponent's untapped-land count | | 86.8% |
| + which opponent lands are tapped | | 59.5% |
| + no possible counter, ambiguous attachment, exile link or token copy | | 43.3% |
| + a clean graveyard history | | 36.7% |

The last rungs fall with the turn number: the counters-and-attachments rung from 99% on turn 1 to
16% by turn 10, the clean-graveyard rung from 98% to 6%. What remains ambiguous is mostly permanent
status (counters, attachments, which lands are tapped), not which cards exist.

The spec builder also grades each state, more strictly than the ladder: T0 means every visible
identity, tapped state, counter and attachment is known; T1 that tapped state or counters are
inferred; T2 that an attachment, exile link or copy is ambiguous; T3 that a visible identity is
ambiguous. §4.5 uses these tiers. On a 20k-game sample (168,606 states), the tiers came out T0
41.2%, T1 32.3%, T2 11.7% and T3 14.8%, and 0 states failed validation.

### 4.4 Building the state in XMage

`java/mzbridge` is a long-lived Java worker compiled against the unmodified XMage jars. It follows
the event-free injection recipe from the research proof of concept:

1. start a duel;
2. run only its setup;
3. move cards from each library into place with no events;
4. set tapped state, sickness, damage, counters, attachments, life and control;
5. set the turn and step;
6. re-anchor MageZero's search;
7. resume.

It answers `build`, `encode`, `coach` and `replay_turn` requests as JSON lines.

| Measured on 1,002 real 17lands states (2,004 specs) | |
|---|---|
| Builds | 2,004 / 2,004 |
| Exact round trip (spec → XMage → dump → compare) | 1,998 / 2,004 (6 differences: Giada's extra counters, which the spec flags) |
| End-of-turn entry rolled forward by the engine vs the direct main-phase build | 97.9% agree (most misses are flagged upkeep triggers or multi-draw turns) |
| `encode` features equal MageZero's own root encoding (a separate check on 30 real specs) | every perfect-information-off comparison; 55 of 74 overall, the 19 misses being perfect-information-on cases whose hidden hand was drawn with a different seed |
| Build / encode time | 2–3 ms / 3–4 ms per decision on the golden specs; the 2,004 real specs built, advanced and dumped in 8.4 s (about 4 ms each) |

Bugs the reviewers caught in the reconstructor and the bridge mostly produced states that passed
validation but described a different game:

- injected tokens firing their enter triggers;
- a Leyline in the opening seven being put onto the battlefield;
- the player on the play drawing on turn 1;
- the opponent's tap lands shown untapped;
- 0/0 Hydras dying without their counters;
- Heraldic Banner silently naming black;
- a stolen creature's static abilities still counting for its owner (a stolen Crusader of Odric
  built 4/4 instead of 3/3 in a synthetic probe).

Each has a regression test.

### 4.5 Replaying a whole turn

Turn replay goes past the turn start. The bridge's `replay_turn` op scripts both seats:

- The user plays the recorded lands, spells and abilities in some order, and declares the recorded
  attackers.
- The opponent declares the recorded blockers and casts its recorded instants in a plausible window.
- Targets are chosen by their fate: prefer the creature the snapshot says died.
- The engine's end-of-turn state is then compared with the next 17lands snapshot on life, the user's
  hand and both battlefields, by card name only: graveyards, exile, tapped state and counters are
  not compared. Since the review, a turn also counts as reproduced only if every recorded play,
  attack and block actually happened in the engine.

Up to 12 variants of the unrecorded details are tried: spell order, main phase 1 or 2, instant
timing, block pairing and modes.

Results on 2,400 turns, stratified 600 per tier and spread across the file (seed 1). An independent
seed-7 sample agrees overall within one point (84.1% stratified, 88.4% at the natural mix), though
single tiers differ by up to 2.5 points.

| | Reproduced |
|---|---|
| Stratified sample | 84.8% |
| **At the file's natural tier mix** | **89.0%** |
| By tier (T0 / T1 / T2 / T3) | 98.2% / 84.0% / 78.5% / 78.7% |
| By user turn (1 / 3 / 5 / 7 / 10+) | 99.4% / 97.6% / 84.3% / 77.7% / 72.6% |
| Matched on the first attempt | 91% of reproduced turns |
| Time per turn | median 13–14 ms, p90 about 90 ms |

The largest failure source is unknown counters: 76 of the 89 failures where only B's life total
differs (73 of 89 on seed 7) are in states flagged `counters_inexact`. Then come guessed targets and
graveyard-dependent plays after unknown mills.

A reproduced turn yields about 4.8 decision records, each with the legal options, the chosen label
and how certain that label is:

- `exact`: attacks, targets settled by what died, and a Pass once every recorded play is done (every
  exact PRIORITY label is such a Pass);
- `imputed_order`: priority plays whose order 17lands does not record;
- `guessed_target`.

Across the 2,400-turn run that came to:

- PRIORITY: 1,947 exact (all Passes) and 3,731 imputed;
- yes/no questions, mostly attacks: 3,014 exact, 19 imputed and 158 guessed;
- targets: 382 exact and 423 guessed; modes: 14 exact and 24 guessed.

**This is the key fidelity number for anything past the turn start.** For about 9 in 10 human turns,
a scripted replay in XMage lands on the same next state as the real game.

Read it as an upper bound, for three reasons:

- **The opponent's side uses hindsight.** Its recorded cards are put into its hand and decklist, and
  its lands are untapped to suit its recorded plays (flagged).
- **Guesses can land by luck.** A guessed target or mode can still reproduce the end state.
- **The opponent's deck is a placeholder.** The measurement used revealed cards plus basics, not the
  belief model.

So what the rate measures is whether the *user's* recorded plays can be realised in XMage to the
same end state.

## 5. Arena logs

With **Detailed Logs (Plugin Support)** on, `Player.log` holds the Game Rules Engine's messages:
full and diff game states, requests to the player, and the player's responses.

What Dan's local log (2 Powered Cube games) showed:

- **Pairing is exact.** 128 of 128 requests pair with the client's answer (`respId == msgId`),
  merging into 105 logical decisions.
- **The state is complete from the local seat.** It includes tapped state, damage and sickness, plus
  counters, attachments and targets as persistent annotations, and both players' committed actions.
- **Hidden information stays hidden.** The opponent's hand and library never appear; only revealed
  cards do.
- **Most priority windows are auto-passed.** Only 17% of the local player's priority windows
  produced an option menu; Arena passed the rest by the client's stop settings. So imitation labels
  exist only for the player's non-trivial stops.
- **Large messages are dropped.** Arena replaces any event whose game state has more than 50 game
  objects or 50 annotations, including any request in the same batch, by a one-line summary. The
  parser detects the gap (it hit once here).
- **The format is stable.** Across 52 versions of the reverse-engineered protocol from 2020 to 2026,
  the core game-state and request messages almost only gained fields (GameObjectInfo lost
  `abilities` and DeclareAttackersReq lost `autoAttackers`). Breakages were in the log wrapper
  (2019, 2020, 2021, 2023).

`src/draftzero/gameplay/arena.py` rebuilds a StateSpec at every decision, with the options Arena
offered and the player's answer. A privacy scrub fails loudly if any player name or id would be
written. With the 7 cards XMage 1.4.58 lacks (2026 cube cards) substituted:

- all 105 decisions build. 46 of them hold a creature reanimated by Animate Dead, which is now
  listed under its controller with `owner` set, so none gets a control warning. Tiers are 76 at T0,
  4 at T1 and 25 at T2, and 36 of the 105 are mid-action states (targets, costs, searches) rather
  than real decision points;
- 42 of 52 priority decisions have exactly Arena's legal set in XMage;
- 154 of 162 Arena options appear in XMage, and every miss involves a substituted card.

**No FDN Arena log exists yet.** Every number above comes from a cube log that is out of
distribution for the FDN network.

## 6. Engine findings that matter for experiment #2

These came out of reading and exercising MageZero to build the bridge. Most matter even without
human data.

| Finding | Evidence | Consequence |
|---|---|---|
| **The search is clairvoyant.** It copies the real game, hidden cards included, and never re-deals them. | `ComputerPlayerMCTS.java:378` ("dont shuffle here"); `shuffleUnknowns` has no callers; MageZero issue #4 | Self-play value targets leak hidden information. Coaching must sample the opponent's hand itself (the bridge does). |
| **Exp #1's network saw the opponent's hand.** | `configs/game.yml` writes `hidden_info:`, `Config.java:79` reads `hiddenInfo`, and the default is true | Harmless in exp #1, since true was intended, but setting it false does nothing. Human data includes the opponent's hand only for mirrored 17lands pairs (13.7% of rows). |
| **The priority and target priors were off in exp #1.** | `configs/curriculum.yml`; docs/003 §3.5 | A human-trained priority head changes nothing unless the prior is switched on. |
| **The attack target is an arbitrary HashSet element.** | `ComputerPlayer.java:1048`, `Combat.java:66` | When the defending player controls a planeswalker (under 1.4% of FDN states; 1.4% have a planeswalker on either side), every attacker may go at the planeswalker. |
| **Seeds don't reproduce games.** The MCTS2 constructor reseeds the RNG to a constant, shuffles use a separate RNG, deck order comes from an identity-hashed set, and UUIDs are random. | Bridge determinism work | Self-play games can't be replayed from their logged seed. The bridge seeds all four. |
| **A timed-out game gives player B +1.** | `ParallelDataGenerator.java:374` (`!playerAWon`) | A small value-label bias |
| **Duplicate cards in hand become duplicate root children.** | 15.2% of decisions in 12 offline self-play games at budget 100 (561 decisions) | Visits are split and policy targets inflated |
| **The budget counts reused visits.** | Median 32 new simulations per decision at budget 100; 39% of decisions got fewer than 10 (561 decisions, 12 offline self-play games) | Exp #1's moves probably got little fresh search too; this was not measured on its own games at budget 96 |
| **Thinly visited children have biased mean values.** | Coach review: 13 of 21 "faults" vanished at 3,000 simulations | Don't read low-visit Q values as verdicts |
| **Offline combat decisions at 300 simulations are noise.** | On one test block position (K=2), visit shares were flat (0.24–0.26) at 300 and Stop Choosing got 0.67 at 1,000; an attack question flipped from no to yes between the two | Use ≥ 1,000 simulations for attacks and blocks |
| `magezero/metrics.py` undercounts missed land drops | Turns where the player only passed never reach its parser | The dashboard under-reports the behaviour §9 finds |

## 7. Imitation learning

### 7.1 What labels exist, per MageZero decision type

MageZero routes each decision to a head:

- **PRIORITY** (cast, play, activate, pass) → the priority head;
- **CHOOSE_USE** (yes/no, including "attack with X?" per creature) → the binary head;
- **CHOOSE_TARGET** (targets, and blocks as "which attacker does this blocker block") → the target
  head.

Modes and X values are searched with uniform priors and are never labelled.

| Decision | From 17lands at turn start | From turn replay | From Arena logs |
|---|---|---|---|
| Land play / cast / activate | set-valued: the turn's plays, unordered | per decision, order imputed | exact |
| Pass | only "did nothing this turn" | exact once nothing is left to do | exact at shown stops (17% of windows) |
| Attack yes/no per creature | exact attacker set | exact | exact |
| Block per blocker | pairing unique in about 88% | from the recorded pairing | exact |
| Spell targets | – | exact when settled by fate, else guessed | exact |
| Mulligan, London bottoms | exact (bottoms 99.6%) | – | exact |
| Value target | game outcome | game outcome | game outcome |

Mulligans are off in this repo's self-play (`mulligans_enabled: false` in `configs/game.yml`). The
mulligan and bottom labels therefore have no MageZero decision to train until mulligans are turned
on.

### 7.2 Volume and throughput

| | |
|---|---|
| Parse the whole file | 185 s, one process (4,285 games/s) |
| Reconstruct turn-start specs | about 2,500 specs/s after parsing; about 1,900 specs/s including decompression and parsing |
| Encode a decision through the bridge | about 4 ms |
| Turn-start dataset end to end (rebuild, bridge, encode), as built for §7.3 | 165 decisions/s with 2 JVMs; the whole file (~6.7M decisions) in about 11 h on this Mac |
| gen 33 trunk embeddings for head training | about 80 rows/s on the Mac's shared GPU: the slowest stage, and the one that most needs a pod |
| Replay a turn (build, script, compare) | median 13 ms of worker time per reproduced turn (mean 34 ms over all turns, failures included); 25.6 turns/s per JVM end to end |
| All ~6.7M decision turns of the file through turn replay | roughly 40 hours with 2 JVMs on this Mac, about 75 JVM-hours (extrapolated) |

On a pod with 7–14 JVMs and a free GPU, the whole-file corpus drops from tens of hours to a few
(extrapolated, not measured).

### 7.3 Experiment: how predictable is human play, and is it a better prior than gen 33?

**Setup.**

- **Data.** Every 48th game of the FDN file (16,483 games) was split by event into train, validation
  and test, with both sides of a mirrored game kept in one split.
- **Decisions.** Each non-terminal user turn became one decision: the start-of-turn state rebuilt in
  XMage, the bridge advanced to the user's first main-phase priority, and the state encoded without
  the opponent's hand.
- **Label.** The set S of that turn's land plays, casts and activated abilities that are legal at
  that moment, or {Pass} if the user did nothing.
- **Scale.** 141,047 turn-start requests, built together with the replayed turns in 856 s with 2
  JVMs (165 requests/s). 99.5% reached the labelled decision.
- **Replayed turns.** A second set of 20,596 turns went through `replay_turn` and 87.7% reproduced.
  They supplied exact attack labels and per-decision priority labels.

Every model's choice is masked to the legal options. CIs are cluster bootstraps over games.

**What the decisions look like.**

- 88% are non-trivial (at least 2 legal non-Pass options), with a mean of 3.9 legal options.
- The label is Pass in 5%. Only 0.3% are unreachable: the user acted, but nothing they did is legal
  at the first main-phase decision.
- |S| is 1 in 68% of the non-Pass labels (mean 1.39).
- 47% of test labels are "a land only". Because 17lands has no order, any of the turn's plays counts
  as a correct first move, so the "no land in S" column is the harder test.

**Priority, held-out test turns (n = 16,750 decisions, 1,980 games):**

| Model | Top-1 in S | Top-3 | Set NLL | Top-1, ≥ 4 options (n = 9,277) | Top-1, no land in S |
|---|---|---|---|---|---|
| Uniform over legal | 36.5% | 88.2% | 1.072 | 33.3% | 32.6% |
| Heuristic: land, else highest-MV spell, else Pass | 65.0% | 97.5% | – | 63.1% | 39.6% |
| **gen 33 as is** | 43.8% | 91.1% | 1.452 | **34.0% (chance)** | **16.2%** |
| gen 33 as search would use it with the priority prior on (temperature 1.5, +0.1 non-Pass bonus; off in exp #1) | 51.5% | – | 1.040 | – | – |
| **gen 33 trunk + heads retrained on 30k human rows** | **73.1% [72.4, 73.8]** | 98.3% | 0.563 | **70.2%** | **57.9%** |
| same, with replayed-turn rows mixed in | 72.9% | 98.2% | 0.570 | 70.6% | 57.1% |
| gen 33 fine-tuned end to end (compute-starved: about 3.7k samples) | 61.8% | 95.2% | 0.810 | 59.6% | 23.4% |
| same architecture from scratch (compute-starved) | 64.5% | 96.9% | 0.742 | 62.9% | 41.2% |

- **Heads versus the heuristic:** +8.1 points [7.4, 8.7], paired.
- **Learning curve** (heads only): 70.5% at 5k rows, 72.0% at 15k, 73.1% at 30k. Each step is
  significant, and the curve is not saturated.
- **The end-to-end runs got 3–8 minutes each** on a GPU shared with the coaching experiment's
  inference server. They say nothing about the architecture at convergence.

**Where gen 33 disagrees with humans.**

- **It passes when humans cast.** When the human's turn was casts only, gen 33's first choice is
  Pass 80% of the time, with or without an attack that turn.
- **It shuns rares.** The probability it gives the human's cast falls with rarity: 0.11 for commons,
  0.04 for uncommons, 0.009 for rares and 0.005 for mythics. The human-trained head shows the
  opposite gradient.
- **It puts Pass before the land.** Its first choice is Pass on 14% of land-only turns (1,128 of
  7,861): 9.4% when the user did not attack (n = 5,677) and 27.2% when they did (n = 2,184), where
  the land may have come after combat. A precombat Pass does not by itself forfeit the land drop.
- **The encoder mismatch is not the cause.** On mirrored games, re-encoding with the opponent's
  **true** hand visible, as gen 33 was trained, changes its top-1 by −0.3 points [−0.9, +0.3].

The fingerprint in §9 found a similar pattern in play, in a different network (the exp #2 pilot's
gen 1, 32 game-sides): missed land drops and fewer spells.

**Skill.** Top-1 is flat across win-rate bands for every model (users with ≥ 100 games; strongest
minus weakest band for the heads model: +0.5 points [−2.3, +3.1]). At turn-level set labels, strong
players are not more predictable, so a per-skill (Maia-style) model is not warranted yet.

**Value (n = 16,797):**

| | AUC | Log-loss | Mean P(win) (base rate 0.518) |
|---|---|---|---|
| Constant | – | 0.693 | – |
| gen 33 value head | 0.654 [0.638, 0.673] | 0.663 | 0.446 (pessimistic) |
| Value head retrained on human outcomes | 0.685 [0.669, 0.704] | 0.637 | 0.540 (the train base rate is 0.538) |

- The paired AUC gain is +0.031 [0.018, 0.042].
- Gen 33's value head is uninformative early (AUC 0.54 on turns 1–2) and decent late (0.73 from turn
  7).

**Attacks** (replayed test turns, exact per-creature labels, n = 2,484; humans attacked with 49%):

| Model | Accuracy | AUC |
|---|---|---|
| Always / never attack | 49.0% / 51.0% | – |
| Power ≥ best untapped blocker's toughness | 74.4% [71.5, 77.1] | 0.745 [0.715, 0.771] (as 0/1) |
| **gen 33 binary head** (the prior exp #1 used in search) | 70.1% [67.2, 73.3] | 0.760 [0.726, 0.795] |
| Binary head retrained on 17,520 human attack decisions | 75.3% [72.7, 77.9] | 0.830 [0.801, 0.857] |

- The heuristic beats gen 33 by +4.3 points [1.5, 7.1].
- The human head beats gen 33 by +5.2 points [3.2, 7.0].
- The human head and the heuristic tie on accuracy (+0.9 points [−1.9, +3.6]), but the human head
  ranks and calibrates better.

**Replayed-turn priority decisions** (n = 5,589, mostly imputed order):

- Top-1 is 72.5% with replayed rows mixed into training, against 64.5% for heads trained on
  turn-start rows alone.
- **Training on turn-start rows alone teaches "always act"**, because only 5% of those labels are
  Pass. Mixing in replayed rows raises the heads' accuracy:

  | Decisions | Turn-start rows only | Replayed rows mixed in | gen 33 |
  |---|---|---|---|
  | Exact "stop acting" Pass decisions | 48% | 76.6% | 72.9% |
  | Turn-start rows where the human did nothing (n = 805) | 26.3% | 40.6% | – |

**Reading.**

- **Human play is predictable enough to be a useful prior.** Retraining only the heads, on 0.4% of
  the available data, beats every baseline. The data volume is there to go much further.
- **Gen 33's policy is a poor model of human play.** It predicts human choices less accurately than
  simple heuristics on both priority and attacks. The 17lands comparison exposes that directly, with
  no games played.
- **The priority margin depends on the label.** Under a looser label that also accepts Pass when the
  user attacked (the coach's convention), gen 33 edges the heuristic: 66.0% against 65.1%. Two
  findings hold under either label: gen 33 is at chance with 4 or more options, and it passes on 80%
  of cast-only turns.
- **Whether a better human prior makes MageZero stronger is still open.** That takes the search A/B
  in §7.5.

### 7.4 How human data should enter training

From the literature (full review in the research notes):

- **Human data mostly speeds up the start.**
  - AlphaGo Zero's supervised network was overtaken by self-play within 24 hours.
  - It helps most where exploration is hard, as in AlphaStar: Test Elo 149 with no human data
    against 1,540 with human initialisation, a KL term toward the human policy, and a human
    statistic.
  - Limited Magic has AlphaStar's traits: a huge action space, sparse reward and hidden information.
- **Anchor on the human policy; don't just clone it.** AlphaStar, VPT, piKL and Cicero all keep a KL
  term toward the human policy while RL (AlphaStar, VPT) or search (piKL, Cicero) improves on it.
- **Relabel human states with MCTS rather than trusting human actions.** MuZero Reanalyse does this,
  and AlphaStar Unplugged's MCTS on a cloned prior was its best offline agent. Turn replay makes
  this possible here: human states, MageZero targets.
- **Train on everything, then fine-tune on strong players.** Filtering to elite games first hurt in
  AlphaStar Unplugged: 84% win rate against the very_hard bot with all MMR>3500 data, 65%
  elite-only, 89% after fine-tuning.
- **Keep the value weight low on human outcomes, and use few states per game.** AlphaGo Zero's
  supervised run on human games used a value-loss weight of 0.01; AlphaGo's value net overfit on
  correlated positions from the same game.

Recommended use, in order of cost:

1. **Binary (attack) head.** It was the one prior exp #1 had on. Labels from turn replay are exact.
2. **Value head.** Pretrain it on turn-start states with the game outcome, at low weight.
3. **Priority head, used as a prior.** Only worth it with the priority prior switched on, and only
   if the pod A/B wins.
4. **Relabel human states with search** (Reanalyse), rather than imitating human priority choices
   directly.

### 7.5 The A/B that decides it

Offline accuracy is not evidence of strength; AlphaGo Zero's best human predictor lost to self-play.
The deciding test is play:

- **Arms:** on the same human-fine-tuned weights, {binary} only as the control (exp #1's setting)
  against {priority, binary}, plus {priority, target, binary}; gen 33 against the fine-tuned net,
  both at {binary}, for the value effect. priorTemp 1.5 and priorBonus 0.1 stay fixed.
- **Setup:** budgets 96 and 300, ≥ 400 games per arm (200 deck pairs, seats swapped, same seeds
  across arms), gen 33 at {binary} and budget 96 as the fixed yardstick.
- **Where:** a pod; network search needs a GPU.

**Power.** At 400 games per arm the standard error of a difference is about 3.5 points. About 7
points is the smallest significant gain, and about 10 points has 80% power. For comparison, exp #1's
40-game evals had ±15 points of noise (docs/003 §6.1).

**Rules.**

- Give each network the input encoding it was trained on, and fix the `hiddenInfo` key first.
- Adopt the human prior only if two things hold: the difference's 95% CI stays above −2 points, and
  the rate of passing when a spell is castable falls.

## 8. Coaching

### 8.1 The pipeline

`src/draftzero/gameplay/coach.py` runs these steps:

1. Build the position (17lands row and turn, or an Arena decision) as a StateSpec.
2. If the opponent's hand or deck is unknown, draw K samples of it from the belief model (§8.2).
3. Run one fresh MCTS search per sample on the bridge.
4. Aggregate each option's mean Q and its spread across samples.
5. Grade the human's choice.

Grading follows chess, backgammon and poker review tools:

- **By win-probability loss.** Q on a ±1 scale is converted as (1 + Q) / 2; this is nominal, since
  the heuristic is uncalibrated.
- **Only non-trivial decisions** are graded.
- **Near-ties count as fine.**
- **Significance.** A fault must beat twice the paired standard error across samples, and twice the
  pooled spread, before it is called a mistake.
- **Thin options are re-searched.** A fault on an option searched fewer than 100 times per sample is
  re-searched at 3,000 simulations. This applies to the offline evaluator only. Network grades on
  thin options are only flagged thin_search.
- **Arena decisions are graded against what Arena offered.**
- **No hindsight.** The opponent's colours come from cards seen so far.
- **Luck is reported separately.**

Cost, offline evaluator, one worker: median 3.9 s per decision at 8 samples × 300 simulations, and
about 50 s when a fault is re-searched.

### 8.2 Guessing the opponent's hand

`belief.py` picks a real 17lands deck of the opponent's colours, weighted by overlap with the cards
seen so far, and forces the seen cards into it. It then draws the hand from that deck minus every
card in a known zone. Two optional weightings apply to the draw:

- **Hand retention:** how often each kind of card is still in hand at that turn.
- **Unknown destinations:** cards seen but now in no known zone (bounced, most likely) are weighted
  as 84% likely to be in hand.

The final cleanup made the second weighting possible. It barely moves the averages (deck-model
recall 0.132 to 0.134 on 300 pairs), but it lifts recall from 0.08 to 0.21 on the 2% of views that
have such cards. The table below comes from the earlier draw.

It was evaluated on 2,500 held-out mirrored pairs (13,591 views at user turns 3, 5 and 7, with both
drafts held out of the deck pool; metrics cover the 13,345 views with a non-empty hand). The model
rows below use 17lands' whole-game opponent colours, which is mild hindsight. With colours from the
cards seen so far, which is what the coach uses, the deck model gets recall 0.126, non-land recall
0.044, log-lik −4.34 and overlap 0.55, and the frequency model gets 0.089:

| Model | Recall@16 of true hand cards | Non-land recall | Log-lik per card | Deck overlap |
|---|---|---|---|---|
| Uniform over the colour pool | 0.035 | 0.036 | −5.00 | 0.30 |
| Colour-conditioned card frequency | 0.112 | 0.046 | −4.55 | 0.53 |
| **Deck model** | **0.134** | 0.047 | −4.26 | 0.57 |
| Deck model + hand retention | 0.115 | 0.068 | −4.10 | 0.57 |
| Oracle (true deck, uniform hand) | 0.232 | 0.164 | −3.04 | 1.0 |

The deck model's gain is mostly lands. Uniform hand draws put twice too many lands in hand (1.53
against a true 0.79), and retention weighting fixes that (0.75).

Non-land beliefs are weak: by turn 7 only about five non-land cards have been revealed, and draft
decks are idiosyncratic. Better conditioning ("held up mana", "missed a land drop") is future work,
trainable on the mirrored pairs.

### 8.3 Is the coach better than the student?

The test from the literature: if the coach is valid, its average regret on human decisions should
fall as the players get stronger.

- **Pilot (offline heuristic evaluator).** 150 decisions from users with ≥ 100 games gave Spearman
  ρ(win-rate bucket, loss) = **+0.08** (95% CI −0.08 to +0.24, n = 149). **No skill signal.**
- **Search noise at 300 simulations.** Before the re-search rule, 13 of 21 "faults" were artefacts
  of rarely visited options, and they disappeared at 3,000 simulations.

**The full experiment** (`tools/gameplay/coach_validity.py`):

- **Decisions:** 17lands main-phase decisions from users with ≥ 100 games, stratified over four
  win-rate bands. Only turns 3–8, tiers T0/T1, and non-trivial decisions.
- **Grading:** the human's action set graded best-of-set.
- **Hidden cards:** belief determinization with the game's own draft held out, K = 8 × 300
  simulations.
- **Network evaluator:** gen 33 served on the Mac (MPS, plus a CPU server), with the binary prior
  only, as in exp #1, and perfectInfo on: inside each sample the network sees the sampled opponent
  hand, as gen 33 was trained.

| | Offline heuristic (n = 242) | gen 33 network (n = 143) |
|---|---|---|
| ρ(win-rate bucket, loss) | +0.087 [−0.048, +0.216] | −0.032 [−0.181, +0.127] |
| Engine's best is in the human's set (chance: about 50%) | 72.7% (71.3% on the network's 143) | 57.3% |
| Mean win-probability loss | 0.51% | 0.72% |
| Graded inaccuracy or worse | 6.6% | 9.8% (all 14 cautioned as thin search) |
| Median Q spread between the best and worst option | 0.21 | 0.07 (58% near-ties) |

**Neither evaluator shows a skill signal.** That still holds when cautioned grades are dropped. The
loss is not higher in lost games either (ρ(won, loss) −0.02 for the network, +0.03 offline). The
offline coach even calls more faults in won games than in lost ones (12 of 126 against 4 of 116,
Fisher p = 0.07), and the pilot showed the same pattern (mean loss 0.9% against 0.25%).

The network run was cut back by contention on the shared Mac: 143 decisions against a target of
200+, and 20 hindsight decisions against about 100. It also changed device mid-run (48 decisions on
shared MPS, then MPS plus a CPU server). Network search is non-deterministic, with 4 evaluations in
flight. Absolute losses are nominal and not comparable across evaluators; ranks and agreement are.

**The evaluator dominates the verdict.**

- On the same 143 decisions, the two evaluators choose the same best action only **51%** [43, 59] of
  the time.
- Their per-option rankings correlate at only 0.50.
- Gen 33's values barely separate the options.

**Hidden information matters, but much less.** Mirrored games were coached twice, once with belief
samples of the opponent's hand and once with the true hand and deck (offline, n = 119):

- the best action is the same in **81%** [73, 87] of decisions;
- per-option Q values correlate at 0.88;
- with the network (n = 20), the best action agrees in 85% [64, 95]. On the same 20 belief specs,
  though, the network and the offline heuristic agree on only 50% [30, 70].

For scale, two independent K = 8 searches agree in 89% of decisions. So hidden information changes
the best action in about 19% of decisions, against 11% between two K = 8 searches and 49% between
the two evaluators. These rates come from different decision sets (119 mirrored pairs, 60 and 143
skill decisions), each includes sampling noise, and the 81% and 89% CIs overlap. Read them as rough
sizes, not additive shares.

**Stability** (offline, 60 decisions). A random K = 8 subset matches K = 16's best action in 94.5%
of decisions, but it shares half of K = 16's samples, so that figure is inflated. The fair measure:
split-half K = 8 against K = 8 gives 89%. **K = 8 is adequate** for user-facing verdicts, and K = 4
for screening.

**The test itself is weak.**

- Among users with ≥ 100 games, the win-rate bucket predicts the game result at only ρ = 0.13, and
  an engine-free check, mana efficiency, has ρ = 0.00 with the bucket (n = 2,364). Mana efficiency
  mostly tracks draws and board state rather than skill (ρ = +0.18 with the result over turns 3–8),
  so it is weak evidence. The power arithmetic below carries the point.
- The smallest detectable ρ at 80% power is about 0.23 at n = 143. Detecting ρ = 0.1 needs about 780
  decisions; ρ = 0.05 needs about 3,100.
- So the null result is not proof of invalidity by itself. The damning part is that the two
  evaluators disagree about half the time.

**Cost.**

| Setting | Per decision |
|---|---|
| Offline on this Mac | median 4.2 s |
| gen 33 network on this Mac, 2 JVMs sharing the GPU (median 40 simulations/s per search: about 50 on MPS, 18 on a CPU server) | median 61 s per JVM, 39 s per decision overall |
| Extrapolated to a 3090 pod with a shared fp16 server | about 11 s and $0.0016, order of magnitude only: docs/006's own table implies about 3× less, and whether coaching keeps 28 search threads busy is untested |

A 2,000–3,000-decision validity run is a few dollars.

### 8.4 Arena coaching on Dan's log

The whole pipeline ran on the local cube log. Of 66 decisions:

- **Grades:** 26 best, 18 fine, 6 inaccuracies, 4 mistakes, 3 blunders; 8 ungraded, 1 trivial.
- **All 13 faults carry caution flags.** 65 decisions involved substituted cards. The belief model
  can't help with cube decks, so hidden hands were redrawn from a placeholder library.

It shows the mechanism works end to end. It says nothing about coaching quality on FDN. The run also
predates the arena.py owner fix: in 28 of the 66 decisions a reanimated creature was listed under
its owner. Re-run `dz gameplay coach arena` to refresh it.

## 9. Behavioural fingerprints: MageZero against humans

`fingerprints.py` computes turn-level behaviour straight from 17lands rows, with no state mapping,
and the same metrics from MageZero's JVM logs.

Humans, full file (top = bucket ≥ 0.60 with ≥ 100 games, 90,719 games; low = bucket < 0.50 with ≥
100 games, 91,381 games):

| | Top | Low |
|---|---|---|
| Spells per own turn, turns 1–5 | 0.21 / 0.66 / 0.77 / 0.89 / 0.98 | 0.21 / 0.67 / 0.80 / 0.91 / 0.99 |
| Attackers / eligible creatures | 0.506 | 0.463 |
| Instant-speed casts on the opponent's turn, per game | 1.38 | 1.13 |
| Mulligan rate | 0.098 | 0.133 |

Spell volume is a format norm, identical across skill. Stronger players attack more, act more at
instant speed and mulligan less.

MageZero, directional only (the exp #2 pilot logs, 32 game-sides per arm, no confidence intervals;
each per-turn cell rests on about 10–30 turns):

- **Gen 1 network (priors off; trained on 16 bootstrap games, not gen 33).**
  - It missed a land drop while holding a land on 16–25% of turns 1–3 (8, 5 and 5 of about 30
    turns), against about 0% for humans. Gen 0 offline search also missed 6–13%, so this is not only
    a network problem.
  - It cast 0.70 spells per own turn, against 0.82.
  - 6% of game-sides cast no spell at all, against 0.4%.
- **Gen 0 offline search.** It matches human spell volume, but attacks far more: 0.81 of eligible
  creatures, against 0.49. MageZero's denominator counts only the creatures XMage asks about, which
  inflates its rate. It also attacked on 91% of turns with an eligible creature, against 68%.

These are exactly the regressions a per-generation fingerprint would catch. It is cheap enough to
run after every generation, next to GIH ρ.

## 10. Code map

| Path | What |
|---|---|
| `src/draftzero/gameplay/statespec.py` | StateSpec v1, the shared state format |
| `ids.py`, `replay.py`, `reconstruct.py`, `labels.py`, `seventeenlands.py` | 17lands: id tables, row parser, turn-start / after-turn / block specs, turn labels, CLI |
| `pairs.py`, `belief.py` | Mirrored-game matcher; opponent deck and hand model |
| `arena.py` | Arena detailed-log parser with a privacy scrub |
| `bridge.py` + `java/mzbridge/` | The XMage worker (`build`, `encode`, `coach`, `replay_turn`) and its Python client |
| `turnreplay.py` | Replays recorded turns and measures reproduction |
| `coach.py` | Coaching driver and reports |
| `fingerprints.py` | Human and MageZero behavioural fingerprints |
| `imitation.py` | Imitation dataset builder (MageZero HDF5 layout plus legal and set labels), models, report |
| `tools/gameplay/coach_validity.py` | The coaching-validity experiment |
| `assets/gameplay/` | Derived FDN id tables (tokens, abilities, card facts), from 17lands data (CC BY 4.0) and XMage (MIT) |
| `tests/test_gameplay_*.py` | 255 tests (306 in the whole suite). Without Java, XMage and the 17lands download, 193 run and 62 skip cleanly. |

The modules run through `dz gameplay <tool>`. The validity experiment runs as `python
tools/gameplay/coach_validity.py {sample,run,analyze}`:

```bash
dz gameplay 17lands fetch                                  # cards, abilities, replay and game data
dz gameplay 17lands stats --every 40                        # conservation and fidelity rates
dz gameplay reconstruct --row 4 --turn 5                    # one turn-start StateSpec
dz gameplay bridge build-jar                                # compile java/mzbridge (locked, atomic)
dz gameplay bridge build java/mzbridge/specs/main_phase.json
dz gameplay turnreplay measure --turns 2400 --workers 1     # reproduction rate
dz gameplay coach 17lands --row 4 --turn 5                  # grade one human decision
dz gameplay arena                                           # parse your local Arena log
# coach it; the flags stand in for cards XMage lacks (report in data/gameplay/coach/)
dz gameplay coach arena --substitute-missing Plains --substitute-token GoblinToken
dz gameplay fingerprints --every 20 --out /tmp/fp.json     # human behaviour by skill (default --out is the full-file result)
```

Several of these write to fixed paths under `data/gameplay/` and overwrite earlier results:
turnreplay `measure`, `arena`, `coach arena` and `fingerprints`. Bridge workers default to a 3 GB
heap; set a smaller one on a 16 GB machine running several.

## 11. Recommendations

**For experiment #2, whether or not human data is used:**

1. **Decide whether the network sees the opponent's hand.** Fix the `hidden_info` / `hiddenInfo` key
   first. Both goals here need a network that doesn't. Human data has the opponent's hand only for
   the 13.7% mirrored 17lands games, and coaching must not assume it.
2. **Ask Will** which of these v0.2 fixes:
   - the config key;
   - seeding;
   - the timeout label;
   - duplicate root children;
   - the attack-defender choice;
   - determinization in the search (`shuffleUnknowns`).
3. **Add the fingerprint monitor to the loop.** It would have flagged the pilot's missed land drops
   immediately.

**Imitation:**

4. Build the human corpus: turn-start states with outcomes, and replayed-turn decisions. Store it as
   specs and labels, so it survives the v0.2 encoder change.
5. Pretrain the binary and value heads first. Human-trained heads on gen 33's trunk beat gen 33 on
   attacks (75.3% against 70.1%, AUC 0.83 against 0.76; on accuracy they only tie the
   power/toughness rule) and on value (AUC 0.685 against 0.654). The priority head beats every
   baseline after retraining on 30k rows and is still rising. Next:
   - the full corpus: about 6.7M turn-start decisions, about 11 hours with 2 JVMs;
   - replayed-turn rows, which carry the exact Pass and attack labels;
   - training on a pod, not on a laptop GPU shared with an inference server.
6. Run the pod A/B of §7.5 before trusting any human prior.
7. Write to 17lands asking for research access to raw game histories. Their open-source client uploads full
   state logs, so they likely hold them for millions of games (retention and access are unconfirmed);
   one yes would change the data picture. Mention that the public data is
   already credited under CC BY 4.0.

**Coaching:**

8. Don't show grades to anyone until the validity test passes: regret must fall with skill. Today
   neither evaluator passes, and the blocker is the evaluator: gen 33's Q values barely separate
   options (median spread 0.07; 58% near-ties). Needed, in order:
   - a value network that separates actions, trained on hidden-information inputs and pretrained on
     human outcomes;
   - network search on a GPU, with thin options re-searched;
   - a validity run of ≥ 2,000 decisions (about $3–5 on a 3090 pod) with a sharper criterion, such
     as per-game summed regret against the result, or agreement with top players on replayed,
     order-resolved decisions.
9. Save FDN Arena logs:
   - Turn on Options → Account → Detailed Logs (Plugin Support).
   - Copy `~/Library/Logs/Wizards Of The Coast/MTGA/Player.log` after each session; Arena overwrites
     it at every launch.

   Twenty FDN games would give a real, exact coaching and evaluation set.
10. Coach after the game, never live (Wizards' terms).

**Worth doing later:**

- Use turn replay to relabel human states with MCTS targets (Reanalyse).
- Learn the opponent-hand model from the mirrored pairs ("held up mana").
- Use FDN TradDraft (Bo3, possibly stronger players) and other sets.
- Add FDN token abilities (Food) to the vocab.
- Pass the unknown-destination weighting into coaching and imitation (`determinize(lost=...)`); only
  the belief evaluation uses it so far.
- Re-run the Arena coaching report now that `arena.py` handles control changes.

## 12. Caveats, licensing and privacy

- **17lands data** is CC BY 4.0. Small excerpts are committed as test fixtures with attribution, and
  derived id tables live in `assets/gameplay/`. 17lands' site terms restrict outside use of anything
  beyond the public datasets.
- **Mirrored-pair lists** stay in `data/gameplay/` (gitignored) and must not be published.
- **Dan's Arena log** was used only locally. No identifiers from it are in any committed file: a
  scan over every committed file found 0. The Arena outputs live under `data/gameplay/`
  (gitignored).
- **Version.** Everything was measured on exp #1's v0.1 engine. v0.2 changes the feature hash and
  encoder, so encoded features must be regenerated; the specs and labels carry over.
- **Hardware.** Timings come from a shared M1 Pro under load; pod numbers will differ.
- **What the benchmarks leave out.** Turn replay compares life, the user's hand and both boards by
  name. It does not compare graveyards, tapped state or counters. The coaching grades rest on an
  evaluator that has not been shown to be valid.

## Credits

- **[17lands](https://www.17lands.com/)** supplies the public replay, game and card data (CC BY
  4.0).
- **[MageZero](https://github.com/WillWroble/MageZero)** by Will Wroble supplies the search,
  encoders and trainer.
- **[XMage](https://github.com/magefree/mage)** is the rules engine.
- **[riQQ/MtgaProto](https://github.com/riQQ/MtgaProto)** reverse-engineered the Arena protocol
  definitions.
- **[rconroy293/mtga-log-client](https://github.com/rconroy293/mtga-log-client)** is the 17lands
  client, which shows what 17lands collects.
