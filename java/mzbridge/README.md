# mzbridge: StateSpec -> XMage bridge

A long-lived Java worker that builds XMage games from StateSpec v1 JSON
(`src/draftzero/gameplay/statespec.py`) and answers questions about the first decision a chosen
player faces there. It is compiled against the draft-zero XMage build (`xmage/lib/*.jar`, the
MageZero fork of experiment #1) and does not change the fork. The Python client is
`draftzero.gameplay.bridge`.

It promotes the research proof of concept (`research/java_poc`, see `xmage_state.md`) and fixes
the flaws the completeness critique lists (§3.2, C3, C4 and WP2). The main fixes:

- one builder, the proof of concept's event-free injector;
- fresh search trees, so reused visits no longer count toward the budget;
- the opponent seat is a puppet;
- per-request seeds for every source of randomness, UUIDs included;
- one database copy per worker;
- the BEGIN_STEP anchor case (3b) now searches the right state.

```
src/org/draftzero/mzbridge/
  Worker.java           JSONL loop, ops build / encode / ping / quit, decision capture
  Coach.java            coach op: K determinizations x MCTS, aggregation, human rank / regret
  StateInjector.java    StateSpec -> Game (the injection recipe), anchor, hand resampling, checks
  BridgePlayer.java     ComputerPlayerMCTS2 for both seats: PUPPET or DECIDER (capture / search)
  Spec.java             Gson mirror of StateSpec v1 + validation
  Dumper.java           StateSpec-shaped dump of a live game
  Decision.java         a decision: type, text, legal options, root children
  DeterministicIds.java reproducible UUIDs inside a request
  Reflect.java          the few engine fields without setters
build.sh, run.sh, log4j.properties
specs/*.json            golden scenarios (the 7 research scenarios in StateSpec v1)
```

## Build and run

```sh
java/mzbridge/build.sh                        # -> java/mzbridge/build/mzbridge.jar (~2 s; git-ignored)
echo '{"id":1,"op":"ping"}' | java/mzbridge/run.sh w0
python -m draftzero.gameplay.bridge build main_phase          # a golden spec by name, or a path
python -m draftzero.gameplay.bridge coach java/mzbridge/specs/respond.json -K 4 --budget 300 --human-action Pass
```

`run.sh <name>` runs a worker in `data/mzbridge/runtime/<name>/` (git-ignored), which holds:

- its own copy of `xmage/db`. H2 opens `./db/cards.h2` by a relative path, and two JVMs on one
  `db/` crash (critique C3). The copy is an APFS clone (`cp -c`) or a reflink when the
  filesystem allows it; a clone measured about 16 KB of extra disk here.
- a pid file, so two workers cannot share a name.
- `worker.log`, which gets stderr when the worker is started from Python.

The JVM flags are:

| flag | why |
|---|---|
| `-Xmx3g` | heap cap |
| `-XX:+UseG1GC -XX:G1PeriodicGCInterval=10000` | an idle worker returns its search memory to the OS |
| `--enable-final-field-mutation=ALL-UNNAMED` | JDK 26+ only (G12) |
| quiet log4j | warnings and errors, to stderr |

| env | default |
|---|---|
| `MZB_HEAP` | `3g` |
| `MZB_LOG_LEVEL` | `warn` (`info` shows MageZero's search logging) |
| `MZB_JAVA_OPTS` | extra JVM flags |
| `MZB_RUNTIME_ROOT` | `data/mzbridge/runtime` |
| `MZ_XMAGE_DIR` | `<repo>/xmage` |
| `MZ_ACTION_VOCAB` | `<repo>/assets/vocab/FDN_SPG.tsv` (read by `ActionEncoder`) |
| `JAVA`, `JAVAC`, `JAR` | `java`, `javac`, `jar` |

Delete a runtime dir when you no longer need it. `run.sh` replaces a database copy that is
smaller than `xmage/db` (damaged) or older than it (xmage rebuilt). The Python `Bridge` rebuilds
the jar when any source file is newer than it. A request that outlives the client's timeout
(default 900 s) kills its worker, since a hung engine would keep a core busy and its late answer
would queue ahead of the next request; `BridgePool` starts a fresh worker on the next request.
The tests run their two workers under a pytest temporary directory, so concurrent test sessions
on a shared machine do not collide on worker names.

## Protocol

JSON lines. The worker reads one request per line on stdin and writes one response per line on
stdout. Logs go to stderr only, and stray `System.out` prints from XMage are redirected there.
On start the worker prints `{"event": "ready", "ok": true, "startup_ms", "db_ms",
"deterministicIds", "actionVocab", "actionDim"}` once the card database is open. If the working
directory has no `db/`, it prints `ok: false` with the reason and exits.

```
request   {"id": 7, "op": "build" | "encode" | "coach" | "ping" | "quit", "spec": {...}, "options": {...}}
response  {"id": 7, "op": "build", "ok": true, ..., "timing_ms": {...}}
error     {"id": 7, "op": "build", "ok": false, "error": {"type", "message", "problems": [...], "trace"}}
```

The worker survives a failed request. `problems` lists every spec problem found, not just the
first one. Invalid specs fail validation. Built states that do not read back as the spec fail
verification: zones, tapped, sick, damage, counters, attachments, library top and size, stack.
With `options.lenient` both become `warnings` instead.

### The decision

Every op builds the state and resumes the engine until the **decision player** faces its first
**non-trivial decision** (two or more options). The decision player is chosen by
`options.decisionPlayer` ("A"/"B"). Without it, it is `spec.priorityPlayer`, and failing that
`spec.activePlayer`.

**That default is wrong for end-of-turn snapshots of the other seat's turn**, which is what
`reconstruct.py`'s recommended 17lands entry (`eot_rollover`) produces: B's END_TURN, meaning
"the start of A's turn". There the default decides for B, who almost never has a decision in
A's turn (measured on the first 40 FDN games: 315 of the 385 specs requested as
`eot_rollover` gave no decision). Even with `decisionPlayer: "A"`, A's first decision was an
instant in B's end step or A's upkeep/draw step in 194 of 357 END_TURN specs, i.e. not the
main-phase decision the 17lands labels describe. Use
`bridge.turn_start_options(spec)` (decision player A, `decideFrom` A's next main phase), and
check `decision.where` before using a decision as the label of a logged action: the bridge moves
on past trivial windows, so the decision can be later than the spec's own position. `where.passedBefore`
counts the priority windows the decision player passed on the way (0 = the spec's own window).

- **Trivial decisions** are taken without search, and recorded the way MageZero records them:
  - a priority window where only Pass is legal (also at MageZero's combat checkpoints);
  - a target with one option.
- **The other seat is a puppet.** It passes priority, answers "no" to attack questions and
  declares no blocks. It resolves other forced choices with XMage's plain heuristics.

  The puppet never acts on its own, so a replayed human turn cannot diverge. Its answers go into
  the player history in the form `MCTSPlayer` replays, so a search anchored earlier rebuilds the
  same state. Inside a search both seats are MageZero's simulation players, as usual.

`decision` = `{player, type, text, where: {turn, phase, step, activePlayer, stack, passedBefore}, legal: [{label, idx, n?, source?}]}`

| type | when | labels | idx |
|---|---|---|---|
| `PRIORITY` | cast / play / activate / pass | `ability.toString()`: `Cast Stab`, `Play Plains`, `Pass`, rule text | action vocabulary (`ActionEncoder.getActionIndex`, isPlayer) |
| `CHOOSE_USE` | "attack with: X?" (one per available attacker, UUID order), "may" abilities | `no`, `yes` | 0, 1 (binary head) |
| `CHOOSE_TARGET` | targets, blocks (one per available blocker: "choose which creature to block for X"), discards | entity names; `Stop Choosing`; `PlayerA` = the decision player, `PlayerB` = the other | `ActionEncoder.getTargetIndex` (tokens hash into the tail) |
| `MAKE_CHOICE` | named choices | choice keys | -1 (no head) |
| `CHOOSE_NUM` | X, amounts, modes | `min`..`max` (at most 65) | -1 (no head) |

`text` is the exact `decisionText` MageZero feeds its StateEncoder:

- `priority`;
- the chooseUse message;
- `<rule>:Choose a target:<target name>`;
- the choice message;
- `choose num for <source>`.

Duplicate labels, such as two copies of a card in hand, are merged into one option with a count
`n`.

If the game ends first, or the end of turn `spec.turn + 1` comes before a decision, `decision` is
null and `noDecision` says why.

### ops

**`ping`** returns `{version}`.

**`build`**:

| option | default | |
|---|---|---|
| `seed` | 0 | library order below the known top cards, hidden (`handUnknown`) cards, engine RNG, UUIDs |
| `idSeed` | `seed` | UUIDs only (see Determinism) |
| `decisionPlayer` | see above | |
| `decideFrom` | none | `{"turn": 7, "step": "PRECOMBAT_MAIN"}`: the decision window opens there; before it the decider acts like the puppet (passes, no attacks). For end-of-turn snapshots whose first real decision would otherwise be an upkeep or draw-step instant. `turn` must be at most `spec.turn + 1` |
| `advance` | true | false = build and dump only |
| `dumpDecisionState` | false | also dump the game at the decision (`decisionState`) |
| `dumpLibrary` | false | full library order under `players.X.x.library` |
| `lenient` | false | warn instead of failing on accounting or read-back mismatches |

Returns `{seed, dump, decision, noDecision?, decisionState?, warnings, timing_ms: {build, advance, total}}`.

`dump` is StateSpec-shaped:

- `turn`, `activePlayer`, `phase`, `step`, `enterMode`, `priorityPlayer`, `passedPlayers`;
- per player: `life`, `decklist`, `landsPlayed`, `hand`, `graveyard`, `exile`, `libraryTop`
  (as many cards as the spec gave), `librarySize`, `manaPool`, and `battlefield` with one entry
  per permanent: `name` or `tokenClass`, `id` alias, `tapped`, `sick`, `damage`, `counters`,
  `attachTo`;
- `stack`, `attackers`, `blockers`.

Engine details go under `x`, which `StateSpec.from_dict` ignores:

- per permanent: `power`, `toughness`, `canAttack`, `canBlock`, `attacking`, `blocking`,
  `attachments`, `uuid`;
- per game: `engineStep`, `stepPart`, `paused`.

`bridge.diff_dump(spec, dump)` compares a spec with its dump.

**`encode`** takes `seed`, `idSeed`, `decisionPlayer`, `decideFrom`, `perfectInfo` (default **false**: the
opponent's hand is encoded as a count only) and `lenient`.

It returns `{decision, features, nFeatures, perfectInfo, timing_ms: {build, advance, encode, total}}`.
`features` are MageZero's StateEncoder feature ids, sorted. They come from the decision
player's perspective, with the decision's type and text, taken at the decision point: for
priority decisions, after the micro-decision history is cleared, as MageZero's own encoding of
that point has it. No search is run.

**`coach`**:

| option | default | |
|---|---|---|
| `determinizations` (or `K`) | 4 | |
| `budget` | 300 | new simulations per determinization (fresh tree each time) |
| `seed` | 0 | determinization k uses `mix(seed, 100 + k)`; UUIDs use `seed` for all k |
| `evaluator` | `{"type": "offline"}` | offline = the fork's heuristic (`GameStateEvaluator3`, uniform priors); `{"type": "remote", "port": 50052, "host": "127.0.0.1"}` = a MageZero inference server (`RemoteModelEvaluator`) |
| `priors` | all off | `{priority, target, binary, opponent: bool, temperature}`; only used with a remote evaluator (exp #1 ran with only `binary` on) |
| `resample` | the non-decision seat | seats whose whole hand is re-drawn from their library (below the known top cards) per determinization |
| `humanAction` | none | a label (`Play Plains`), `yes`/`no`/`true`/`false` for CHOOSE_USE, or 17lands-style ability text without the cost |
| `perfectInfo` | **true** | the search's encoder setting, as experiment #1's network was trained; matters for remote evaluation only |
| `rootFeatures` | false | also return each determinization's root features as MageZero's search encoded them (`MCTSNode.stateVector`); the tests check that `encode` reproduces them exactly |
| `timeoutSec` | 120 | per search |
| `decisionPlayer`, `decideFrom`, `lenient` | | as above |

A coach response contains:

- `decision` (the legal list);
- `consistent`: every determinization reached the same decision;
- `settings`;
- `determinizations`: one entry per k, with `k`, `seed`, `sampledHands`, `type`, `text`,
  `best`, `rootVisits`, `rootQ`, `value`, `seconds` and `children`
  (`[{label, idx, N, Q, prior}]`);
- `aggregate`: one entry per label, sorted by rank, with `label`, `idx`, `meanQ`, `sdQ`,
  `visitShare`, `N`, `nDet` and `rank`;
- `best`;
- `human`, when `humanAction` was given: `found`, `label`, `rank`, `meanQ`, `sdQ`,
  `visitShare`, `regret` (best meanQ minus the human action's meanQ);
- `timing_ms`: `build`, `search`, `total`, `simsPerSec`.

How the numbers are computed:

- **Q** is the child's mean backed-up value from the decision player's view, in [-1, 1],
  discounted 0.99 per ply; unvisited children have Q null.
- **Duplicate children**, such as two copies of a card, are merged per determinization:
  N summed, Q visit-weighted.
- **Aggregates across determinizations**: the mean and sd of Q (sd over the determinizations
  where the label was visited), and the mean visit share.
- **Rank** is by mean Q, ties broken by visit share.

Report a regret as a mistake only when it clearly exceeds `sdQ` (`magezero_mcts.md` §4.8).

**`quit`** answers, closes the database without writing to it, and exits.

A spec whose `startingPlayer` contradicts the turn parity (the player on the play takes the odd
turns) gets a warning: legal after an extra turn, but usually a per-player vs global turn
mix-up. On turn 1 it is an error.

### Remote evaluator (a MageZero network)

`coach` with `evaluator: {"type": "remote", "port": P}` searches with MageZero's network through
the fork's `RemoteModelEvaluator`, as self-play does: msgpack over HTTP, `/healthz` checked
once per worker. Serve a checkpoint with the unmodified mz-engine server, from a directory
holding `models/<deck>/ver<N>/<checkpoint>.pt.gz`:

```sh
mkdir -p nn/models/FDN_generalist/ver1
ln -s "$PWD/models/FDN_generalist/ver1/gen33.pt.gz" nn/models/FDN_generalist/ver1/
cd nn && MZ_ACTION_VOCAB=../assets/vocab/FDN_SPG.tsv MZ_DEVICE=cpu \
  ../.venv/bin/python ~/Desktop/Code/mz-engine/src/magezero/server.py \
  --deck FDN_generalist --version 1 --port 50091 --checkpoint gen33
python -m draftzero.gameplay.bridge coach main_phase --evaluator remote --port 50091 --priors binary
```

This was tested on 2026-09-26 with gen33 on CPU:

- search ran at about 20 simulations per second;
- `priors` changed the root children's priors;
- an unreachable port gives a clear error.

Remote search is not reproducible (see Determinism). Keep `perfectInfo` true for gen33, which
was trained with the opponent's hand visible.

## Entry modes

The semantics are those of `statespec.py`. How the bridge implements each mode:

- **PRIORITY_FRESH**: step part `PRE` and the game paused. `resume()` takes `resumeBeginStep`
  (no turn-based actions), then priority from the active player. Inside the combat phase the
  injector does rule 507.1's combat set-up itself.
- **PRIORITY_HELD**: step part `PRIORITY`. Priority starts at `priorityPlayer`, and
  `passedPlayers` keep their passed flag.
- With injected `attackers`, the two differ at DECLARE_ATTACKERS: PRIORITY_FRESH goes through
  `DeclareAttackersStep.resumeBeginStep`, which finishes the declaration (the attackers are
  tapped with events and "whenever ... attacks" triggers fire and go on the stack), while
  PRIORITY_HELD assumes that already happened (those triggers are lost: the stack holds spells
  only). Likewise at DECLARE_BLOCKERS with `blockers`.
- **BEGIN_STEP**: the engine is put at the end of the *previous* step (step part `POST`).
  Both the live game and every paused MCTS copy then run `postPriority` and play the target
  step from `beginStep`. This covers the draw, combat set-up, attacker and blocker
  declaration, and cleanup's discard.

  The proof of concept's `PRE` without a pause behaved differently in the live game and in the
  (always paused) search copies. That was the 3b failure: an NPE with a stale anchor, and no
  blocks with a fresh one. The same spec now reaches the same block decision, and the same
  search statistics, as the PRIORITY_HELD entry (`tests/test_gameplay_bridge.py`).

  BEGIN_STEP at UNTAP is rejected: use the previous turn's END_TURN with PRIORITY_FRESH. A
  CLEANUP entry must use BEGIN_STEP.

Conventions the contract leaves open, and how the bridge reads them:

- `stack` is bottom to top, in cast order.
- Attackers' tapped state comes from the engine: declaring an attack taps a creature without
  vigilance.
- `id` aliases of a `count > 1` entry are `<seat>:<id>#k`, and the bare alias is the first copy.
- A `token` + `set` that maps to more than one class is an error: FDN's Beast is `BeastToken`
  (3/3) or `BeastToken2` (4/4), and Cat is `CatToken`, `CatToken2` or `CatToken3`. Use
  `tokenClass`.
- `counters` set the count of each listed type; a type the spec does not list keeps what the
  card entered with (a planeswalker's printed loyalty, "enters with" counters).
- The dump's `name` is the card's own name. Effects can rename a permanent: Witness Protection
  turns a creature into "Legitimate Businessperson". The effective name is `x.name`.
- `hand` cards, `libraryTop` and spells on the stack are taken out of the decklist by name.
  With `handUnknown > 0`, that many cards are drawn from the shuffled remainder with the request
  seed. `librarySize` then trims the library from the bottom.
- `provenance` and `labels` are not read. The golden specs keep the decision player in
  `labels.decisionPlayer` for the tests.

## Determinism

The same spec, seed and options give the same build, decision, dump, features and offline
search statistics (N and Q). That holds for two requests in one worker and across workers. It
needed all of the following:

| source | what the bridge does |
|---|---|
| `RandomUtil` (thread-local; the MCTS2 constructor reseeds it with a constant, G8) | seeded after the players are constructed and again before resuming |
| `GameState.localRandom` (in-game shuffles; carried by copies, C4) | seeded before `init()`, after injection and before the anchor copy |
| library order | `Deck.getMaindeckCards()` is a HashSet of cards hashed by identity, so copies of a card land in a different order every build; the bridge restores the decklist order |
| UUIDs | `UUID.randomUUID()` is made reproducible by a SecureRandom provider installed first in the worker (`DeterministicIds`); card and token classes are loaded before the stream is reset. Attack and block questions come in UUID order, and MCTS children sort by a string that contains UUIDs |
| token ids | the fork draws `PermanentToken` ids from `game.getLocalRandom()`; the bridge seeds that per token from `idSeed` |

In a `coach` request the determinizations share `idSeed`, so they ask the same question. They
differ in the library order, the hidden cards and the engine RNG. Network (remote) search is not
deterministic: up to 4 evaluations are in flight at once.

## Gotchas from the research, as handled here

| # | gotcha | handling |
|---|---|---|
| G1 | `stepPart == null` ends the game; the step must be the instance inside `Phase.steps` | set by reflection on that instance; `verify()` checks it |
| G2 | `TwoPlayerDuel.init` adds "starting player skips DRAW" | TurnMods cleared, except that one when the spec is entered before turn 1's draw; checked |
| G3 | `GameImpl.drawHand` is a JVM-wide static | untouched; opening hands go back to the library |
| G4 | MCTS anchor is the empty `init()` copy | `pause()` + `setLastPriority` after injection (and after resampling); checked to be at the injected step; re-anchored at MageZero's combat checkpoints on the way to the decision |
| G5 | paused copies resume via `resumeBeginStep` | BEGIN_STEP entered as the previous step's `POST` (above) |
| G6 | stop options are not checked inside the resumed phase | the decider pauses the game; `stopOnTurn = turn + 1` is only a safety net |
| G7 | tokens, ATTACH, casts fire events | event-free variants; pending triggers cleared and checked; watchers reset |
| G8 | RNG | see Determinism |
| G9 | sickness, damage, passed have no setters | reflection (`Reflect`), read back by `verify()` |
| G10 | owner = controller | control changes are not supported (not in StateSpec v1) |
| G11 | decks >= 40, match wired | validated; `actionEncoder` preset so `printAllActionsFromDeck` is skipped |
| G12 | JDK 26 final-field warning | flag added by `run.sh` when the JDK knows it |
| G13 | `GameState` does not serialize | specs are the interchange format; rebuilds take a few ms |
| G14 | `db/` in the working directory | one runtime dir and one DB clone per worker (the report's "AUTO_SERVER lets JVMs share" is wrong, C3) |
| G15 | players must be `PlayerA` / `PlayerB` | always |

New gotchas found while building the bridge:

- **Library order is not reproducible.** Identity-hashed deck sets make same-name copies land
  in a different order every build.
- **Token ids come from the local RNG.** `PermanentToken` draws its id from the game's local
  RNG, not from `UUID.randomUUID()`.
- **Warm-up changes the ids.** First-time database lookups and card-class initialisation
  consume UUIDs.
- **Search trees are reused.** `ComputerPlayerMCTS2.getNextAction` re-roots on the previous
  tree. The bridge clears its package-private `root` by reflection before every search, so the
  budget is new simulations.
- **Opening-hand actions.** `init()` draws the first seven decklist cards and offers their
  opening-hand actions. A Leyline among them was put onto the battlefield (an extra permanent,
  or "no 'Leyline Axe' left in the library"): 14 of the first 770 17lands specs failed this
  way. Both seats now decline every yes/no question during `init()` (`BridgePlayer.setup`), and
  the build fails if `init()` leaves anything on the battlefield.
- **The turn-1 draw.** Clearing the TurnMods (G2) also dropped "the player on the play skips its
  first draw". A spec entered before turn 1's draw (turn 1 UPKEEP, or DRAW with BEGIN_STEP: the
  Arena mulligan and starting-player decisions) drew an 8th card. The TurnMod is put back there.
- **Checkpoints.** MageZero re-anchors its search at BEGIN_COMBAT, DECLARE_ATTACKERS and
  DECLARE_BLOCKERS (`GameImpl.isCheckPoint`) even when only Pass is legal, which also clears
  both players' micro-decision histories (encoded as ChosenTargets / UseChoices). The puppet and
  the decider do the same on their way to the decision, so a combat decision is encoded as in
  self-play and a search replays a short prefix. Other windows where MageZero would re-anchor
  (a non-trivial priority passed by the puppet, or by the decider before `decideFrom`) are not
  mirrored: computing their playable list would cost more than the encode itself.

## Timings and memory

Measured on 2026-09-26 on the M1 Pro, lightly loaded (load average 3–5 of 8 cores), with the
method of `python -m draftzero.gameplay.bridge bench`: the 7 golden specs, 30 seeds each, with
the first round dropped (it pays for class loading).

| | measured |
|---|---|
| worker start (spawn to ready, Python `Bridge`) | 1.13–1.21 s: JVM to `main` ~52 ms, card DB open 960–1020 ms |
| `build` (inject only, warm) | median 2 ms, p90 3 ms (the first build in a JVM: ~180 ms) |
| `build` + advance to the decision (capture) | median 3 ms, p90 6–7 ms |
| `encode` request (build + advance + encode) | median 3 ms, p90 5 ms; the StateEncoder call itself median 0.7 ms |
| `coach`, K=4 x 300, offline, one worker | 2.2–2.9 s per decision (median 2.4–2.5 s); 416–535 simulations/s |
| `coach`, K=4 x 300, offline, two workers in parallel | 14 decisions in 21.9 s = 1.56 s per decision overall |
| `coach` with gen33 on CPU (remote, mz-engine server) | 19–21 simulations/s, measured at K=1 x 30–40; so about 60 s for K=4 x 300 (extrapolated) |
| worker RSS | 0.38–0.42 GB after start; 0.9 GB after 400 builds/encodes; 1.9 GB after coach requests (1.6 GB with `MZB_HEAP=1500m`); back to 0.8 GB after 10 s idle |

Re-measured in the review on 2026-09-26 (load average about 3.6, after the fixes): bench with 20
seeds gave the same build / capture / encode medians (2 / 3 / 4 ms, StateEncoder 0.8 ms), coach
K=4 x 300 median 2.45 s on the golden specs (412–531 simulations/s), RSS 0.41 GB at start, 0.9 GB
after encodes, 1.9 GB after coach. On 16 real 17lands `main1` states coach K=4 x 300 took median
1.64 s. All 770 specs reconstructed from the first 40 FDN games built and round-tripped
(6 s for the 770 builds in one worker); gen33 remote search ran at 18 simulations/s.

The research reported builds of 5–18 ms and encodes of 10–20 ms under a load of 27–49. Expect
those numbers on a busy machine.

## Limitations

- **One decision per request.** Later decisions of the same turn (the second attack question,
  the target of the spell just chosen, a replayed human turn) need the `replay_turn` op of WP2
  with a scripted decider, which is not built yet.
- **Not reproducible yet**: per-turn watcher history (spells cast or life gained this turn),
  until-end-of-turn effects, linked exile (Banishing Light), control changes, face-down
  creatures, loyalty activations used, and activated or triggered abilities on the stack (the
  stack holds spells only).
- **Decisions the engine never asks.** A spec for an Arena mulligan or starting-player decision
  (turn 1 UPKEEP) reaches the next priority decision instead: `init()` has already run, and the
  MCTS players never mulligan. A spec taken mid-action (Arena `SelectTargetsReq`, flagged
  `mid_action`) cannot hold a half-cast spell, so it reaches a priority decision too. Compare
  `decision.type` with the logged decision type.
- **Attack defenders.** The attack defender is `HashSet` order in the fork (C5). Coaching and
  attack labels are unreliable when the opponent controls a planeswalker.
  `test_planeswalker_defender_is_not_a_decision` documents it: the attack question stays yes/no
  with a planeswalker in play, while a spec can still state an attack on the planeswalker.
- **Search sees everything.** Inside one determinization the search is clairvoyant for both
  seats (PIMC). `resample` only re-draws hands from the library as built. For real logs, the
  opponent's library must come from a belief model (WP3), not the true deck.
- **Shallow micro-decision search.** Offline search values attack and block questions only at
  the next priority node, which lies below every later attacker's or blocker's question. With
  fresh trees the default budget is not enough for combat: measured on the golden specs
  (K=2), the first of three block questions came out flat at 300 simulations (visit shares
  0.24–0.26, mean Q within 0.005) and separated only at 1,000 (Stop Choosing 0.67); the first
  attack question ranked "no" first at 300 and "yes" first at 1,000. The research's peaked block
  statistics came from reused trees. Use budgets of 1,000+ for combat decisions, and treat
  offline combat verdicts at 300 as noise.

## Data attribution

The golden decklists come from 17lands FDN Premier Draft top-player decks (deck 26941 with
Felidar Savior and Pacifism swapped in, and deck 28740), via the research proof of concept.
17lands data is licensed CC BY 4.0 (https://www.17lands.com/public_datasets).
