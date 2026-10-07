# Rules engines for a fast Limited agent: speed and MCTS support

Which rules engine should a high-level Limited agent train on? This compares five candidates on
speed, FDN cards, and support for MCTS-style learning:

- **XMage** as MageZero uses it: the MageZero v0.2.0 bundle, XMage 1.4.58, WillWroble/mage
  `cb7e9c6f`. It's our current engine.
- **[Forge](https://github.com/Card-Forge/forge)** at `ddb27fd4939` (2026-09-13).
- **[gorge](https://github.com/adams-shaun/gorge)** at `a4af596` (2026-09-27). It's a Go engine
  that compiles Forge's card scripts.
- **[mtg-kernel](https://github.com/jackmaiorino/mtg-kernel)** at `2c5e72f` (2026-09-27). It's a
  Rust engine with a Python training stack, and the engine behind Spellbench.
- **[ManaBrew](https://github.com/TrevorS/manabrew/tree/fork/parity-loop)**: the fork
  `TrevorS/manabrew`, branch `fork/parity-loop`, at `6bc6d4c` (2026-09-30). It's a Rust port of
  Forge's rules engine, checked against Java Forge. The fork adds a reinforcement-learning
  environment to [upstream](https://github.com/witchesofthehill/manabrew).

**Setup:**
- Each engine was built and timed on one laptop (M1 Pro, 16 GB): the first four on 2026-09-28,
  ManaBrew on 2026-09-30.
- They ran one engine at a time, one core each, with the same test (§7).
- The workloads were experiment #1's first two eval pairs and a Pauper Burn mirror, the one deck all
  five can play today.

## Summary

**gorge and ManaBrew are the two fast engines with the FDN cards today.** gorge is the easier of the
two to search on; ManaBrew follows Forge's rules more closely.
- **ManaBrew is a Rust port of Forge, not Forge itself** (§3).
  - It plays all 286 FDN cards.
  - It matched Java Forge decision for decision in 267 of 280 seeded FDN games.
  - It plays random games about as fast as gorge.
  - But a game can be resumed only at the start of a turn, so one search step costs about 0.5 ms,
    against gorge's 46 µs.
- **gorge** has the cheapest search step of the engines with FDN, and a step API. Its card behaviour
  hasn't been audited: 12 FDN cards are flagged.
- **mtg-kernel is the fastest and the best built for search, and FDN support is now in progress**
  (§4).
  - The maintainer has a seven-milestone plan in
    [mtg-kernel#110](https://github.com/jackmaiorino/mtg-kernel/issues/110), with three milestones
    in open pull requests.
  - Nothing is merged yet, and no date is promised.
- **Forge and XMage have every card but are much slower.** gorge and ManaBrew play random FDN games
  16–32× faster than XMage and 31–72× faster than Forge.
- **MageZero is the only one with a working AlphaZero training loop.**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/015-engine-speed-dark.png">
  <img alt="Horizontal bars on a log scale showing each engine's speed relative to XMage, whose speed is a vertical line at 1x. Three bars per engine: random play in turns per second on the Pauper Burn mirror, copying a mid-game state, and one search step. mtg-kernel: 468x, 25x and 72x. gorge: 49x, 11x and 27x. ManaBrew: 23x, 77x and 2.4x. Forge is slower than XMage on all three: 0.56x, 0.071x and 0.22x." src="img/015-engine-speed-light.png">
</picture>

- **gorge against XMage:** 16–50× faster per turn of random play. Copies are 11× faster, and a
  search step is 27× faster.
- **ManaBrew against XMage:** 23–31× faster per turn of random play.
  - Its copies are copy-on-write: cards are duplicated only when first changed. That makes a copy
    77× faster than XMage's (a full copy is 2.3× faster).
  - But a search step is only 2.4× faster, because a game resumes only at the start of a turn.
- **mtg-kernel against XMage:** about 470× faster per turn on Burn. Copies are 25× faster, and a
  search step is about 70× faster.
- **Forge against XMage:** about half XMage's speed in play. Copies are 14× slower, and a search
  step is about 4.5× slower.
- **With a faster engine, the network becomes the bottleneck.** On gorge or mtg-kernel, inference
  would limit search, not the engine. MageZero's inference server tops out around 200 requests/s
  (docs/003), and it was already the bottleneck in experiment #2a (docs/014). A faster engine only
  pays off with batched GPU inference across many games.

**Three fast routes.** Each needs its own extra step before it can train a Limited agent.
- **gorge has the cards and cheap search steps today.** We'd build the tree search, a batched
  network bridge, and a sampler for the opponent's unknown 40 cards.
  - Pin a commit, because it changes hundreds of times a day.
  - Audit its card behaviour. For example, Nine-Lives Familiar comes back with 1 counter instead
    of 7, and Kiora's Threshold condition is ignored.
- **ManaBrew has the cards, checked against Forge, and its fork has a learning environment.** We'd
  need the same search, network bridge and sampler, plus one of two things:
  - a way to resume a game at any decision, not just at turn start;
  - or a search that pays about 0.5 ms per step.

  It's AGPL-3.0 (§3).
- **mtg-kernel has the search and learning scaffolding today**, and about 10× gorge's raw speed. Its
  extra step is FDN support, which the maintainer is now building (§4).

**The other two engines and Manafold:**
- **XMage stays the reference.** It has every card and the only working training loop. But its
  search sees hidden cards, and runs can't be reproduced from a seed.
- **Forge has the best card pipeline**, which gorge and ManaBrew reuse, but it is the slowest engine
  to search on.
- **Manafold**, the other engine Spellbench lists, isn't playable yet: only Mountain and Plains.

## 1. Speed

One core each, medians. FDN cells show pair A / pair B.

| | XMage (MageZero v0.2) | Forge | gorge | mtg-kernel | ManaBrew |
|---|---:|---:|---:|---:|---:|
| Random play on FDN, turns/s | 45 / 45 | 23 / 20 | 731 / 1,444 | not yet (FDN in progress) | 1,362 / 1,400 |
| Random play on Burn, turns/s | 55 | 31 | 2,703 | 25,560 | 1,240 |
| Built-in bot on FDN, games/s | 0.61 / 0.32 | 0.30 / 0.60 | 38 / 58 | no heuristic bot | 17 / 18 |
| Copying a mid-game state | 164 µs, 1.3 MB | 2.3 ms, ~1.9 MB | 14 µs, 100 KB | 6.6 µs, 45 KB | 2.1 µs, 11 KB copy-on-write; 72 µs, ~600 KB full |
| One search step | 1.2–2.4 ms | 5.5 ms | ~50 µs | ~17 µs | ~0.5 ms |
| Built-in search, per thread | 424–806 sims/s | 30 sims/s | ~40 rollouts/s | 1,900–2,600 sims/s | none; ~50 random rollouts/s from a copy |

**What each row means:**
- **Random play:** uniform over each engine's legal options. The Burn mirror is the one workload
  all five run.
- **Built-in bot:** each engine's own cheapest real bot on both seats.
  - XMage: CP7 ("Computer - mad") at skill 2, the client's default.
  - Forge: its default AI.
  - gorge: its heuristic bot.
  - mtg-kernel's only built-in policy is uniform random. On Burn it plays 684 games/s through its
    learning interface.
  - ManaBrew: `manabot`'s SimpleAi, a small rule bot that "casts spells when possible, otherwise
    passes priority". Forge's AI isn't ported.
- **Copy:** a copy of a mid-game state (turns 5–8, permanents on both sides).
  - XMage: `Game.copy()`.
  - Forge: `GameCopier.makeCopy()`.
  - gorge: `Engine.Clone()`.
  - mtg-kernel: a `GameState` clone.
  - ManaBrew: `GameState::clone`, which shares cards until they change. Its snapshot
    (`make_snapshot`, 2.2 µs) uses the same copy.
  - mtg-kernel is timed on a Burn state; the others on FDN pair A.
- **One search step:** what one node expansion costs before any network.
  - XMage: a MageZero MCTS simulation (copy the parent state, run to the next decision, compute
    features, score), from 1.2 ms at turn 5 to 2.4 ms at turn 7.
  - Forge: copy, one random decision, advance: 5.5 ms.
  - gorge: `Engine.Clone` plus one `Submit`: 46 µs.
  - mtg-kernel: a clone plus one step through its learning interface: about 17 µs. This is derived,
    not timed: a 6.6 µs clone plus 10.6 µs per step (94k steps/s).
  - ManaBrew: restore the turn-start snapshot and replay the turn up to the decision. That's 0.48–0.52
    ms on FDN, because a game can't resume mid-turn (§3).
- **Built-in search:** each engine's own search, so the rows differ in what a simulation is (§2).

## 2. MCTS support

| | XMage / MageZero | Forge | gorge | mtg-kernel | ManaBrew |
|---|---|---|---|---|---|
| Existing search | AlphaZero-style tree search with a network, tree reuse | Depth-3 search over its own plays; not MCTS | Sampled worlds, each played to the end by its bot; no tree | Information-set MCTS with a static evaluator. A network-guided version exists but runs only on x86 and isn't used in training | None |
| Learning stack | Full loop: inference server, trainer, self-play | None | Small CPU-only neural net written in Go | Python/Torch over JSON lines, native Rust forward pass, CUDA; policy-gradient self-play | The fork adds a Rust learning environment: games on parallel threads, observation encoding, candidate tables. No Python binding, network or trainer yet |
| Stepping a game from outside | No step API: the engine calls into player objects, and MageZero pauses it with scripted puppets | No step API: 109 controller callbacks to implement | Yes, in-process | Yes, in-process, with snapshot and restore | No step API: Forge's shape, 108 callbacks on the game's own thread. The learning environment steps it through channels. A copy resumes only at the start of a turn |
| Reproducible from a seed | No, because card IDs are random: 5 of 6 same-seed games differed | Yes | Yes | Yes | Yes, once a seeded random-number generator is installed |
| Hidden information | Search sees the opponent's hand and library order | Same | Per-player redacted views. Its sampler assumes both decklists are known, so Limited needs a sampler over the opponent's pool | Re-deals unseen cards from what the searcher knows. For Limited it would deal from a guess at the opponent's pool instead of their real cards | Copies hold both hands and libraries, and card IDs follow decklist order. The learning environment's encoder hides them |
| Action identity | Hashed into 128 slots; targets by card name. Our fork adds a set-wide vocabulary | Ability text | Positional options | Stable hashed IDs | Positional candidates; card and ability IDs stable across copies |
| Parallelism | 4 search threads in one JVM matched 1 thread's total | One process per core: the RNG and ID counters are JVM-wide | Many games per process | Many threads | Many games per process, about 8 MB each |
| FDN cards | 286/286 | 286/286 | 282/286, plus 12 flagged as partly wrong | 7/286 today; FDN in progress (§4) | 286/286; matched Java Forge in 267 of 280 FDN games |
| New sets | Upstream takes weeks to months; MageZero's fork must rebase | Scripted about 2–3 weeks before release | Reuses Forge's card scripts; new mechanics need Go code | Each card coded in Rust, mostly by agents (670–3,100 lines per card, including tests) | Reuses Forge's card scripts; new mechanics are ported from Forge's Java |
| Maturity | Since 2010; MIT | Since ~2007; 130 authors in the last year; GPL-3.0 | Created 2026-09-04; 6,844 commits by 2026-09-28 (515 on 2026-09-27), mostly by AI agents; README says "not ready for parity or production use"; Apache-2.0, with card scripts fetched from Forge | Created 2026-07 from work inside an XMage fork; most commits by an automated identity; MIT | Upstream since 2026-05 (18 contributors; "pre-release"). The fork: about 1,600 commits in two weeks under one author; AGPL-3.0 |

## 3. ManaBrew: a Rust port of Forge

**It isn't Forge: it's a separate program that aims to behave exactly like Forge.**
- It ports Forge's rules engine (`forge-game`) file by file to Rust, keeping Forge's file, module
  and method names. The goal is "1:1 behavioral parity", with Java Forge as "the source of truth".
- File counts line up with Forge's: 224 effect files against Forge's 206, 142 triggers against 142,
  and 46 replacement effects against 47.
- It reads Forge's own card scripts from a pinned Forge checkout, so its card pool follows Forge's.
- **Deliberate differences:**
  - Card-script parameters are compiled into typed data.
  - Cards are shared between copies until changed.
  - A `mirror_forge_bugs` switch can reproduce Forge's known bugs. Parity runs turn it on; the
    learning environment leaves it off.
- **Forge's AI is not ported.** The built-in bot is new and minimal.
- **Compared with gorge:** gorge also reads Forge's scripts, but it is its own Go design with no
  Java check.

**Fidelity, measured.** We ran ManaBrew's parity harness on our decks. It plays the same decks and
seed in Rust and in Java Forge, with the same seeded random agent on both seats. It compares 42
state fields at every turn and every decision.

| Workload | Games | Identical to Java Forge to the end |
|---|---:|---:|
| FDN pair A, both seat orders, 70 seeds, up to 40 turns | 140 | 133 (95%) |
| FDN pair B, the same | 140 | 134 (96%) |
| Pauper Burn mirror | 70 | 70 (100%) |

- **The 13 FDN mismatches** came between turns 7 and 24.
  - 6 are one bug: a creature with a +1/+1 counter keeps trample in Rust but not in Java.
  - The rest are 2 life-total differences around mana payment, 3 differences in the actions
    offered, 1 land tapped differently and 1 graveyard difference.
- **Card by card:** all 286 FDN cards passed the harness's per-card probe against Java (6 seeds × 30
  turns).
- **Parity runs switch on Forge-bug mirroring and Forge's mana check**, which the learning
  environment doesn't use.

**For training.**
- **Speed:** random games play about as fast as on gorge: 54–57 FDN games/s on one core. The fork's
  learning environment handles one learner decision every ~240 µs, of which 51 µs builds the
  observation.
- **Cheap, exact copies:** copying, restoring and resuming a copy reproduced the original game
  every time (16 of 16).
- **Fully reproducible from a seed**, in one process and across processes. The engine's default
  random source isn't seeded, though, so a seeded one has to be installed.
- **Hard to search:**
  - The engine calls into player objects from its own game loop, as Forge does.
  - A resumed game always restarts its turn at Untap.
  - So a search step has to replay the turn up to the decision, about 0.5 ms.
  - Resuming a game at any decision would bring this down to roughly gorge's cost. We'd write that
    change ourselves or ask for it.
- **Copies are omniscient**, and card IDs are assigned in decklist order.
- **Limited:** London mulligan and 40-card decks work. A port of Forge's draft and sealed modes is
  included (`forge-limited`).
- **Licence:** AGPL-3.0, as a derivative of GPL Forge. These points are our reading, not legal
  advice:
  - Code linked into the engine, such as a Python module or a Rust trainer, becomes part of the AGPL
    work.
  - A trainer in a separate process limits that.
  - Offering an agent on a modified engine over a network means offering its source.
- **The fork moves fast:** about 1,600 commits in two weeks under one author, at all hours, mostly
  `fix(engine)`. Pin a commit.

## 4. mtg-kernel's Limited support

*Update, 6 October: milestone 4 is done (PR #140 merged). Both fixture decks play complete games with London
mulligans, and 38 FDN cards plus the basics are playable. [docs/025](025-mtg-kernel-fdn-self-play.md) runs
self-play and training on them.*

FDN support is an extra step, not a blocker, and it's now under way. The maintainer took on
[mtg-kernel#110](https://github.com/jackmaiorino/mtg-kernel/issues/110) on 2026-09-30, with a
seven-milestone plan. It uses our FDN pair A decks and the 286-card FDN list as test fixtures.

Status on 2026-10-01:

| Milestone | Status |
|---|---|
| 1. Load `.dck` decks and check card coverage | Open: [#113](https://github.com/jackmaiorino/mtg-kernel/pull/113) |
| 2. A reset/step interface for custom 40-card decks | Open: [#115](https://github.com/jackmaiorino/mtg-kernel/pull/115), on top of #113 |
| 3. Limited rules: full priority windows, London mulligan, combat damage choices with trample, planeswalkers | Priority windows (opt-in) open in [#116](https://github.com/jackmaiorino/mtg-kernel/pull/116), on top of #115; the rest to come |
| 4. First FDN games: the 36 cards the two fixture decks need | Split into seven mechanic batches (#116); a batch of six simple cards is next |
| 5. The full FDN pool | Not started |
| 6. Fair Limited search: sample the opponent's deck from public information | Not started |
| 7. DraftZero integration: our observations, actions and decks; then search and training throughput | Not started |

**Where that leaves it:**
- None of the three PRs is merged.
- Today 7 of the 286 FDN names resolve.
- The plan promises no delivery date before its mechanic inventory and first complete-game
  comparison.
- Each later set would be its own card step. That matters for simulating a set soon after its
  spoilers: Forge scripts arrive 2–3 weeks before release, and gorge and ManaBrew inherit them.

## 5. mtg-kernel's speed claims

Our roadmap quoted mtg-kernel as "about 40× faster than XMage overall, and 1,000× for training".
- **"1,000×"** is a design target and was never measured. The XMage baseline behind it has been
  withdrawn.
- **"40×"** isn't in the repo. The nearest figure is 53×: 13.65 against 0.2588 games/s.
  - It compares network-against-network games with games against XMage's CP7 bot, which thinks for
    about 14 s per move.
  - So it measures bot cost, not engine speed.
- **Measured engine against engine on Burn:** about 470× XMage per turn of random play, and 25× on
  state copies.

## 6. Side findings for current runs

- **XMage's search threads don't add up.**
  - One search thread kept 1.6–2.2 cores busy.
  - XMage copies the whole game every time it lists legal moves: the in-place shortcut is switched
    off (`if(false && simulation)`, §8).
  - Four search threads in one JVM matched one thread's total: about 425 sims/s either way.
  - On the pods, one laptop thread's 424–806 sims/s on turn 5–7 boards compares with about 20 per
    game thread (docs/006). The two aren't directly comparable, but the gap is worth a profile.
- **Forge calls `System.gc()` after every game** (`Match.java:103`). That costs about 0.36 s here.
  Headless Forge sims can skip it with `-XX:+DisableExplicitGC`.

## 7. Method and caveats

**Workloads:**
- **FDN pair A:** `FDN_top_04956_UG` vs `FDN_top_20626_WG`.
- **FDN pair B:** `FDN_top_07961_WR` vs `FDN_top_02581_UR`. Both pairs are from `assets/sample/`.
  Seats swap every game.
- **Burn mirror:** mtg-kernel's Burn list, run in every engine.
- **FDN card list:** the 286 names in the 17lands FDN reference (`assets/reference/FDN_gih.json`).

**Timing:**
- Each engine ran alone, single-threaded.
- The two JVM engines warmed up for 10–67 s first.
- Game-playing measurements ran for 20–90 s each, and each copy method was timed 1,000–5,000 times.
- Versions:
  - XMage and Forge: JDK 21.
  - gorge: Go 1.27.1.
  - mtg-kernel: Rust 1.94.1.
  - ManaBrew: Rust 1.98.1 (it needs 1.95 or newer). Its parity runs used Java Forge from its pinned
    submodule (`e7d2b93`) on JDK 21.

**Turns:** each game's own turn counter, which counts both players' turns.
- Compare turns per second, not decisions per second.
- The random players differ in what they count as a decision. XMage's and ManaBrew's pay mana
  automatically; gorge's taps lands one at a time.
- Random games also run to different lengths: 22–29 turns in XMage, Forge and ManaBrew, 36–43 in
  gorge.

**Caveats:**
- MTG Arena was running during the first four engines' timing, using about half a core.
- Another process started near the end of ManaBrew's run. It overlapped only the last measurement
  (the learning environment on Burn), which isn't reported.
- gorge's FDN behaviour rests on its own tests and over 12,000 games here, none of which crashed or
  stalled. It hasn't been independently audited. Of its 12 flagged cards, only Nine-Lives Familiar
  was confirmed wrong at runtime.
- ManaBrew's agreement with Java Forge was measured with random agents, which reach fewer lines of
  play than strong ones.
- mtg-kernel's speed was measured on Burn and Rally. FDN boards may cost somewhat more per step.
- The benchmark harnesses aren't in this repo yet.

## 8. Where the claims come from

**XMage** (WillWroble/mage `cb7e9c6f`):
- `Mage/src/main/java/mage/game/GameImpl.java:357`: the search starts from `this.copy()`, hidden
  cards included.
- `GameImpl.java:368`: `if(false && simulation)`, so listing legal moves copies the game.
- `Mage.Server.Plugins/Mage.Player.AIMCTS/src/mage/player/ai/ComputerPlayerMCTS.java:384`:
  `//dont shuffle here`.
- `Mage.Server.Plugins/Mage.Player.AI/src/main/java/mage/player/ai/encoder/ActionEncoder.java:23`
  and `:49`: Pass is slot 0, and every other action is hashed into slots 1–127.

**Forge** (`ddb27fd4939`):
- `forge-ai/src/main/java/forge/ai/simulation/GameCopier.java:37-45`: every zone is copied, hand
  and library included.
- `GameCopier.java:317`: rebuilding each card from its script "accounts for the vast majority of
  GameCopier execution time".
- `forge-game/src/main/java/forge/game/player/PlayerController.java`: 109 abstract callbacks.
- `forge-core/src/main/java/forge/util/MyRandom.java:34`: one static RNG for the JVM.
- `forge-game/src/main/java/forge/game/Match.java:103`: `System.gc()` after every game.

**gorge** (`a4af596`):
- `rules/clone.go:51` and `state/game.go:645`: `Engine.Clone` and `Game.Clone`.
- `view/visibility.go:14-35`: Seat, Public and Omniscient views.
- `internal/searchprobe/sample.go:24`: the sampler's `PublicGame` holds both decklists.
- `sample.go:358`: opponent actions are regenerated with the built-in bot.
- `effects/zone.go:2474`: `WithCountersAmount$ X` falls back to 1, which breaks Nine-Lives Familiar.
- `AGENTS.md:188`: the engine's known approximations.

**mtg-kernel** (`2c5e72f`):
- `mtg-kernel/src/engine.rs:10342`: "no-trample simplification".
- `mtg-kernel/src/surface.rs:303`: the priority windows its learning interface skips (#116 adds
  them as an option).
- `mtg-kernel/src/rl.rs:1383`: seven cards dealt, no mulligan.
- `mtg-kernel/src/kernel_native_search_opponent_v1.rs:1036`: re-dealing hidden zones for each
  simulation.
- `ROADMAP.md:46`: the withdrawn XMage figure.
- `docs/native_scaled_selfplay_population_program_v1.md:174-176`: 13.65 against 0.2588 games/s.
- `docs/research/regime_review_v1.md:13-16`: CP7's think time.
- The FDN plan: `docs/design/fdn_limited_implementation_v1.md` and
  `docs/design/fdn_fixture_mechanics_v1.md` on #116's branch.

**ManaBrew** (`6bc6d4c`; crate paths under `manabrew-rs/crates/`):
- `docs/agents/PARITY_PHILOSOPHY.md:3-10`: the 1:1 parity goal.
- `manabrew-engine/src/agent/mod.rs:32`: the `PlayerAgent` callbacks.
- `manabrew-engine/src/game_loop/phase_handler.rs:38`: every turn the loop runs starts at Untap.
- `manabrew-engine/src/game.rs:261` and `parity/src/runner.rs:1260`: `mirror_forge_bugs`, on in
  parity runs.
- `manabot/src/agent/simple_ai.rs:30`: the built-in bot.
- `manabrew-gym/AGENTS.md:3`: the learning environment; a Python binding is planned but doesn't
  exist.

**Plot:** `tools/engine_bench/speed_plot.py`.

## Follow-up ideas

- **Follow mtg-kernel's FDN work (#110)** and test training on FDN when its games run (ROADMAP).
- **Commit the five benchmark harnesses**, so these numbers can be rerun.
- **Audit gorge's FDN behaviour against XMage**, starting with the 12 flagged cards. docs/008's turn
  replay could compare the two engines on recorded 17lands turns.
- **If we take the ManaBrew route:** resuming a game at any decision, not just at turn start, is the
  change that makes search cheap.
- **Profile XMage's per-thread search rate on a pod** (§6).
