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
  Coach.java            coach op: K determinizations x MCTS (re-drawn or given), aggregation, human rank / regret
  StateInjector.java    StateSpec -> Game (the injection recipe), anchor, hand resampling, checks
  ReanimatedAura.java   Animate Dead & co.: the enchant swap and sacrifice link their ETB trigger would make
  Substitutions.java    options.substitute: stand-ins for cards / tokens this XMage build lacks
  BridgePlayer.java     ComputerPlayerMCTS2 for both seats: PUPPET or DECIDER (capture / search)
  Spec.java             Gson mirror of StateSpec v1 + validation
  Dumper.java           StateSpec-shaped dump of a live game
  Decision.java         a decision: type, text, legal options, root children
  DeterministicIds.java reproducible UUIDs inside a request
  Reflect.java          the few engine fields without setters
  TurnReplay.java       replay_turn op: one recorded turn replayed with both seats scripted, compared with its end snapshot
  TurnScript.java       replay_turn's script (A's lands/spells/abilities/attacks, B's blocks and plays), policies, snapshot
  ReplayRun.java        one replay attempt: windows, mana reservation, attacks/blocks, modes, recorded decisions
  ReplayPlayer.java     BridgePlayer that follows a ReplayRun (both seats)
  TargetResolver.java   targets from the snapshot ("fate": what died, left, was needed), else a heuristic, flagged
  ReplayWatcher.java    deaths during the replayed turn, the hand as its cleanup began
build.sh, run.sh, log4j.properties
specs/*.json            golden scenarios (the 7 research scenarios in StateSpec v1)
```

## Build and run

```sh
java/mzbridge/build.sh                        # -> java/mzbridge/build/mzbridge.jar (~2 s; git-ignored)
echo '{"id":1,"op":"ping"}' | java/mzbridge/run.sh w0
python -m draftzero.gameplay.bridge build main_phase          # a golden spec by name, or a path
python -m draftzero.gameplay.bridge coach java/mzbridge/specs/respond.json -K 4 --budget 300 --human-action Pass
python -m draftzero.gameplay.bridge coach --specs samples.jsonl --budget 300 --decision-player A \
    --human-set "Cast Stab" --human-set "Play Swamp"            # K given determinizations, a set of human actions
python -m draftzero.gameplay.bridge build arena_spec.json --substitute-missing Plains --substitute-token BooToken
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
The tests run one worker (the shared machine allows one bridge JVM per engineer) under a pytest
temporary directory, so concurrent test sessions do not collide on worker names; cross-JVM
determinism is checked by restarting that worker.

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
first one. Invalid specs fail validation (`Spec.validate()` applies the same rules as
`statespec.StateSpec.validate()`; change both together). A spec naming cards the XMage database
does not have fails with one error listing all of them, unless `options.substitute` is given
(below). Built states that do not read back as the spec fail verification: zones, tapped, sick,
damage, counters, attachments, controller and owner, library top and size, stack, and triggers
left pending by the injection. With `options.lenient` both become `warnings` instead.

Two kinds of state build and read back exactly, yet are not what the game will play from, and
come back as `warnings`:

- **Choices made as a permanent entered**, which the spec has no field for: Heraldic Banner's
  color, Adaptive Automaton's or Secluded Courtyard's creature type, a target chosen on entry.
  Each is answered as XMage's AI would, except a color, which is the chooser's main color (the
  color of most cards it owns) for a benefit and its opponent's for a drawback, and reported:
  `B: Heraldic Banner asked "Choose color" as it was injected; answered Green (the spec cannot say
  what was chosen)`. Yes/no questions ("pay 2 life?" of a shock land) are answered no and not
  reported: the spec's own state wins.
- **Permanents the state-based actions remove on resume**: a creature with toughness 0 or
  lethal damage, a planeswalker without loyalty, an Aura attached to nothing, a second legend of
  one name: `A: Pacifism is an Aura attached to nothing: the state-based actions remove it when
  the game resumes`.

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
| `substitute` | none (strict) | stand-ins for cards XMage lacks, see below |

Returns `{seed, dump, decision, noDecision?, decisionState?, warnings, substitutions?, timing_ms: {build, advance, total}}`.

**`options.substitute`** (build, encode, coach) builds a spec that names cards this XMage build
does not have, such as the 2026 cube cards of an Arena log against XMage 1.4.58:

| key | effect |
|---|---|
| `"missing": "Plains"` | every card the XMage database does not know becomes a Plains |
| `"missingToken": "BooToken"` | every token that resolves to no class becomes this class (an ambiguous `token`+`set` still fails) |
| `"<card>": "<card>"` | an explicit replacement, applied whether or not XMage knows the card |

A name is replaced in every zone and in the decklist, so the accounting still holds. What a
stand-in cannot do is dropped, each with a warning: a substituted spell on the stack when the
stand-in is a land (that copy stays in the library), the counters of a substituted card (a token
stand-in keeps them), and attachments, attacks, blocks and stack targets involving a substituted
permanent (a stack item keeps its targets up to the first lost one, so none slides into the
wrong slot). An Aura left without its host is then put into the graveyard when the game resumes
(a state-based-action warning says so).
`warnings` lists every substitution with the zones it touched, e.g. `substitute: 'Sear' ->
'Plains' (not in the XMage card database): {decklist=1, stack=1}`, and `substitutions` maps
original to stand-in. The result is an approximation of the logged state, never a silent one.

`dump` is StateSpec-shaped:

- `turn`, `activePlayer`, `phase`, `step`, `enterMode`, `priorityPlayer`, `passedPlayers`;
- per player: `life`, `decklist`, `landsPlayed`, `hand`, `graveyard`, `exile`, `libraryTop`
  (as many cards as the spec gave), `librarySize`, `manaPool`, and `battlefield` with one entry
  per permanent (listed under its controller): `name` or `tokenClass`, `id` alias, `owner`
  (only when another seat owns it), `tapped`, `sick`, `damage`, `counters`, `attachTo`;
- `stack`, `attackers`, `blockers`.

Engine details go under `x`, which `StateSpec.from_dict` ignores:

- per permanent: `power`, `toughness`, `canAttack`, `canBlock`, `attacking`, `blocking`,
  `attachments`, `uuid`;
- per game: `engineStep`, `stepPart`, `paused`.

`bridge.diff_dump(spec, dump)` compares a spec with its dump.

**`encode`** takes `seed`, `idSeed`, `decisionPlayer`, `decideFrom`, `perfectInfo` (default **false**: the
opponent's hand is encoded as a count only), `lenient` and `substitute`.

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
| `humanActions` | none | a set-valued human action, e.g. every spell a 17lands player cast that turn (only one of them answers this decision); reported as `humanSet` |
| `specs` | none | K pre-determinized specs of the same decision (e.g. `belief.determinize` samples), used instead of the request spec (which may then be omitted) and instead of `resample`; `determinizations` is K |
| `idSeed` | `seed` | UUIDs, shared by every determinization so they ask the same question |
| `perfectInfo` | **true** | the search's encoder setting, as experiment #1's network was trained; matters for remote evaluation only |
| `rootFeatures` | false | also return each determinization's root features as MageZero's search encoded them (`MCTSNode.stateVector`); the tests check that `encode` reproduces them exactly |
| `timeoutSec` | 120 | per search |
| `decisionPlayer`, `decideFrom`, `lenient`, `substitute` | | as above |

A coach response contains:

- `decision` (the legal list);
- `consistent`: every determinization reached the same decision;
- `settings`;
- `determinizations`: one entry per k, with `k`, `seed`, `sampledHands` (or, for given
  `specs`, `hands`: the other seat's hand as built), `type`, `text`, `best`, `rootVisits`,
  `rootQ`, `value`, `seconds` and `children` (`[{label, idx, N, Q, prior}]`);
- `aggregate`: one entry per label, sorted by rank, with `label`, `idx`, `meanQ`, `sdQ`,
  `visitShare`, `N`, `nDet` and `rank`;
- `best`;
- `human`, when `humanAction` was given: `found`, `label`, `rank`, `meanQ`, `sdQ`,
  `visitShare`, `regret` (best meanQ minus the human action's meanQ);
- `humanSet`, when `humanActions` was given: `members` (one `human` entry per action),
  `nLegal` (members matching a visited label, duplicates once), `bestLabel`, `setBestLabel`,
  `regret_best_of_set` = best meanQ minus the best member's meanQ (the charitable reading: the
  player's answer here was its best cast), `regret_mean_of_set` = best meanQ minus the members'
  mean meanQ; both null when no member matched;
- `warnings` (e.g. substitutions), `substitutions`;
- `settings.specs` (K when the determinizations were given) and `settings.idSeed`;
- `timing_ms`: `build`, `search`, `total`, `simsPerSec`.

Coaching over the belief model's determinizations (the Python client):

```python
samples = [belief.determinize(...) for k in range(4)]          # K concrete specs, one decision
r = b.coach_specs(samples, budget=300, seed=1, decisionPlayer="A",
                  humanActions=["Cast Stab", "Play Swamp"])       # a 17lands turn's casts
r["aggregate"], r["humanSet"]["regret_best_of_set"], r["consistent"]
```

Each spec gets a fresh tree; all are built with the request's `idSeed`, and card ids depend on
(seat, card name, copy) only (Determinism), so specs whose unseen cards differ still ask the same
question. `consistent` is false if they did not.

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
  card entered with (a planeswalker's printed loyalty, a saga's first lore counter, "enters
  with" counters). Checked on the Arena log: 19 of 19 saga LORE and 10 of 10 planeswalker
  LOYALTY counts built exactly (`test_counters_set_the_count`). The arena build report's "spec
  loyalty 4 built as 7" does not reproduce with this code.
- Battlefield entries are injected in list order, and "enters with additional counters"
  replacement effects of permanents already injected apply to later ones (Giada, Font of Hope:
  an Angel listed after Giada gets its +1/+1 counters, one listed before does not), as if they
  had entered in that order. List the counters to pin them.
- A permanent is listed under its **controller**. `owner` names the other seat when control
  changed (a reanimated or stolen creature): its card comes out of the owner's decklist and
  library, it enters under the controller's control with the controller as its original
  controller (what a reanimation does in XMage), so control does not revert when effects are
  reapplied. It dies into its owner's graveyard. Tokens cannot carry `owner`. Animate Dead,
  Dance of the Dead and Necromancy get the two lasting parts of their ETB trigger rebuilt
  (`ReanimatedAura`): the "enchant creature card in a graveyard" -> "enchant the creature it
  returned" swap (XMage refuses the attachment without it, and the state-based actions would
  bury the Aura) and "when this leaves the battlefield, that creature's controller sacrifices
  it".
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
| UUIDs | `UUID.randomUUID()` is made reproducible by a SecureRandom provider installed first in the worker (`DeterministicIds`); card and token classes are loaded before the stream is reset. Each decklist card draws from a stream seeded by (idSeed, seat, name, copy), so its ids do not depend on the rest of either decklist. Attack and block questions come in UUID order, and MCTS children sort by a string that contains UUIDs |
| token ids | the fork draws `PermanentToken` ids from `game.getLocalRandom()`; the bridge seeds that per token from `idSeed` |

In a `coach` request the determinizations share `idSeed`, so they ask the same question. They
differ in the library order, the hidden cards and the engine RNG (and, with `specs`, in whatever
the caller's determinizations differ in). Network (remote) search is not deterministic: up to 4
evaluations are in flight at once.

A spec entered before any decision can exist gives the same answer every time. The Arena
ChooseStartingPlayerReq spec (turn 1 upkeep, both hands empty) has no decision for A before the
safety stop: A skips its first draw and has nothing to do in B's turn 2. Three fresh JVMs x 2
repeats gave "no decision" 6 of 6 times (`test_pregame_spec_is_deterministic` checks both test
workers). The arena review's one-off PRIORITY decision for it is the behaviour of the jar
before the turn-1 draw fix, which landed during that review: with that fix reverted, the spec
gives a PRIORITY decision in turn 1's main phase with the one card A wrongly drew (a land).

## Gotchas from the research, as handled here

| # | gotcha | handling |
|---|---|---|
| G1 | `stepPart == null` ends the game; the step must be the instance inside `Phase.steps` | set by reflection on that instance; `verify()` checks it |
| G2 | `TwoPlayerDuel.init` adds "starting player skips DRAW" | TurnMods cleared, except that one when the spec is entered before turn 1's draw; checked |
| G3 | `GameImpl.drawHand` is a JVM-wide static | untouched; opening hands go back to the library |
| G4 | MCTS anchor is the empty `init()` copy | `pause()` + `setLastPriority` after injection (and after resampling); checked to be at the injected step; re-anchored at MageZero's combat checkpoints on the way to the decision |
| G5 | paused copies resume via `resumeBeginStep` | BEGIN_STEP entered as the previous step's `POST` (above) |
| G6 | stop options are not checked inside the resumed phase | the decider pauses the game; `stopOnTurn = turn + 1` is only a safety net |
| G7 | tokens, ATTACH, casts fire events | event-free variants (tokens too, see below); "as this enters" yes/no questions declined, named choices reported as warnings; pending triggers cleared (checked) and queued simultaneous events dropped; watchers reset |
| G8 | RNG | see Determinism |
| G9 | sickness, damage, passed have no setters | reflection (`Reflect`), read back by `verify()` |
| G10 | owner = controller | `Perm.owner`: the card comes from the owner's library and enters under the controller, with its static abilities re-pointed at the controller (above) |
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
- **"As this enters" questions during injection.** Injected permanents still run their
  enters-the-battlefield replacement effects, and some ask: a shock land's "pay 2 life?" was
  answered yes by the puppet (39 of 105 Arena specs). The life total was reset afterwards, but
  the queued life-loss event triggered "whenever an opponent loses life" permanents on resume
  (Bloodthirsty Conqueror). `BridgePlayer.setup` now covers the whole build, so injection
  declines every such question (the spec's tapped flag wins anyway), and events any injection
  step still queues (Authority of the Consuls tapping entering creatures: 12 of 2,004 real
  17lands specs) are dropped before the game resumes.
- **The turn-1 draw.** Clearing the TurnMods (G2) also dropped "the player on the play skips its
  first draw". A spec entered before turn 1's draw (turn 1 UPKEEP, or DRAW with BEGIN_STEP: the
  Arena mulligan and starting-player decisions) drew an 8th card. The TurnMod is put back there.
- **Tokens fired their ETB triggers after the resume.** `Token.putOntoBattlefield` queues
  ZONE_CHANGE, ENTERS_THE_BATTLEFIELD and CREATED_TOKEN events on GameState's simultaneous-event
  list, which is handled only when the game resumes, after the injector had cleared the pending
  triggers. Every "whenever a creature enters" permanent then triggered once per injected token
  (17lands row 560303 user turn 9: 6 Authority of the Consuls triggers on the stack at the
  decision, and +6 life after an END_TURN rollover). It also ran CREATE_TOKEN replacement effects:
  a token doubler made a spec token "made 2 permanents" (1 of 1,002 real states). Tokens now
  enter like cards (a PermanentToken added directly, no events), and the hygiene step also
  empties the simultaneous-event list.
- **Static abilities of a control-changed permanent kept its owner as controller.** A card's
  static abilities are registered with its owner as controller when the deck is loaded; XMage
  re-points them when a permanent enters (`ZonesHandler`: `ContinuousEffects.setController`).
  The injector did not, so a Crusader of Odric that A had taken counted B's creatures (4/4
  instead of 3/3), and a stolen anthem would have pumped its owner's team. Found in the review.
- **Token ids collided in large token counts.** Token streams were salted
  `1000*seat + 31*entry + copy`, so the 32nd token of an entry drew the next entry's first
  token's ids: one UUID, two permanents, a failed build. Each (seat, entry, copy) now has its own
  mixed stream. Found in the review.
- **Card ids were one sequential stream over both decklists.** Changing one card of A's
  library (a belief sample) renumbered every B card, and with it the order of attack and block
  questions. Each decklist card now draws its ids from a stream seeded by (idSeed, seat, name,
  copy number).
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

Re-measured after the second phase (2026-09-26, load average 3.8, bench with 12 seeds): build
median 2 ms (p90 3), build + capture 3 ms (p90 6), encode request 4 ms (p90 7; StateEncoder
0.9 ms), coach K=4 x 300 median 2.44 s (418–555 simulations/s), RSS 0.42 GB at start, 0.8 GB
after encodes, 2.0 GB after coach. Tokens entering without events and per-card id streams
cost nothing measurable. The 2,004 real 17lands specs above built, advanced and dumped in 8.4 s
in one worker (about 4 ms each). Coach K=4 x 300 on 10 of those main1 states: median 1.71 s over
4 given `specs`, 2.01 s with the bridge's own re-drawing.

The research reported builds of 5–18 ms and encodes of 10–20 ms under a load of 27–49. Expect
those numbers on a busy machine.

## Fixed in the second phase (2026-09-26), measured

On 1,002 real 17lands FDN decision states (every 790th game of the replay file from row 193,
one seeded-random user turn each, row 560303 turn 9 included; `reconstruct.state_at_user_turn`
in both entries, 2,004 specs; seed 1, decision player A, the main1 decision window), the
committed jar against this one:

| | before | after |
|---|---|---|
| specs that build | 2,002 / 2,004 (a token doubler: "Knight33Token made 2 permanents") | 2,004 / 2,004 |
| exact round trip (`diff_dump`) | 1,994 / 2,002 | 1,998 / 2,004 |
| main1 decisions with triggers on the stack (injected tokens) | 14 / 1,001 | 0 / 1,002 |
| END_TURN rollover reaching the main1 spec (hand, life, battlefields) | 954 / 993 (96.1%); 21 of the 39 misses unexplained by the spec's flags | 973 / 994 (97.9%); 4 of the 21 misses unexplained |
| row 560303 user turn 9 | 6 Authority of the Consuls triggers at the decision; rollover life off | 0; rollover matches |

What is left, after:

- Round-trip differences (6 specs, 3 states x 2 entries): all are +1/+1 counters that Giada,
  Font of Hope puts on Angels injected after it. The specs list none (17lands does not record
  them; reconstruct flags `counters_inexact`).
- Rollover misses (21): 17 are flagged by reconstruct (`main1_turn_start_triggers_skipped`,
  multiple draws, unknown draw): upkeep and draw triggers that main1 skips and the engine plays.
  The 4 others: 2 Giada counters, a Clinquant Skymage counter from its draw-step trigger (the
  engine is right; main1 is not flagged), and an Aura with no host (`attach_no_host`) that the
  first state-based-action check puts into the graveyard.
- In 6 states of 1,002 A's first decision in both entries is the attack question: A had
  nothing to do in its main phase. In 2, A has no decision before the safety stop. Check
  `decision.where`.
- On the committed fixture (67 states), `(4, 7)` (Dazzling Angel + 2 Faerie tokens) no longer
  differs; `(0, 7)`, `(1, 8)`, `(1, 9)`, `(198, 4)` and `(198, 5)` still do, for data reasons
  (a blind draw, Clinquant Skymage, Giada, Phyrexian Arena's upkeep).

On the local Arena log (105 specs of a Powered Cube session, `data/gameplay/arena/decisions.jsonl`),
with `substitute {"missing": "Plains", "missingToken": "BooToken"}` (7 cards and the Boo token
are not in XMage 1.4.58):

| | before | after |
|---|---|---|
| strict build | 0 / 105 (unknown cards) | 0 / 105, with one error listing all 4-6 unknown cards |
| with substitution | 56 / 105 (Python-side substitution; 46 Animate Dead attachments refused, 3 NPEs) | 105 / 105; the 46 specs that list the reanimated creature under its owner (as arena.py does today) build with the wrong controller and a warning |
| with substitution, the 46 reanimated creatures moved under their controller with `owner` | not expressible | 105 / 105, no control warning |
| saga LORE / planeswalker LOYALTY counts after the build | | 19 / 19 and 10 / 10 exact |
| priority decisions at the logged position: exact legal-set match | | 42 / 52; 154 / 162 Arena options offered by XMage; every miss is a substituted card |

The other two arena-review findings are not bridge bugs. The "land plays offered during B's main
phase" (game 2 decisions 6-7) were decisions of A's next turn (`where`: turn 6, A active, 8-9
windows passed): A's only instant there was substituted by a Plains, so A had nothing to do in
B's turn and the bridge moved on. XMage offers no land play to a non-active player
(`test_no_land_play_in_the_opponents_turn`). The 3 stack NPEs were those substituted Plains on the
stack; a land on the stack is now a clear SpecException, and `substitute` drops such items.

Found and fixed in the review of this phase (same day; independent sample: every 800th game
from row 11, one random user turn each, plus row 560303 turn 9: 990 states, 990 `main1` + 990
`eot_rollover` + 469 `declare_attackers` block specs, each built with its `labels.bridge`):

| | jar before this phase (20c8924) | committed fixes (5e03155) | after the review |
|---|---|---|---|
| specs that build | 2,445 / 2,449 (2 token doublers, 2 control changes) | 2,449 / 2,449 | 2,449 / 2,449 |
| exact round trip | 2,432 / 2,445 | 2,439 / 2,449 (Giada counters) | 2,439 / 2,449 (same) |
| main1 decisions with injected-token triggers on the stack | 14 / 987 (row 560303: 6) | 0 / 988 | 0 / 988 |
| END_TURN rollover reaching the main1 spec | 939 / 981 (28 misses unflagged) | 962 / 982 (8 unflagged: Clinquant Skymage, Giada) | 962 / 982 (same) |
| builds whose queued events were dropped (all TAPPED_BATCH, Authority of the Consuls) | | 16 / 2,449 | 16 / 2,449 |
| warnings about a choice made on entry (Heraldic Banner 68, Secluded Courtyard 3) | none (silent guesses) | none (silent guesses) | 71 in 2,449 builds |
| warnings about permanents the state-based actions remove on resume | none | none | 0 in 2,449 builds |

- Control-changed static abilities (above) and the token-id collision (above) are fixed, each
  with a regression test that fails when the fix is reverted.
- The Python client could hang for ever on a restart. A worker's stdout pump looked its queue up
  when the output ended, so a worker restarted (`BridgePool` after a timeout or a crash, or
  `close()` + `start()`) before its old pump saw EOF got the old end-of-output sentinel: `start()`
  read "the worker exited" and blocked in `proc.wait()` on the live new JVM. It hung a full test
  run for 10 minutes on the loaded machine. Each pump now writes to the queue it was started
  with, and an end of output waits at most 10 s for the process before killing it.
- `substitute` dropped the counters of a substituted card without a warning; it warns now.
- Heraldic Banner's color used to be the alphabetical first option (Black) whatever the deck:
  now the owner's main color, with a warning. 26 of the 990 `main1` states have a Banner.
- On the Arena log: unchanged 105 / 105 builds with substitution, 0 choice or state-based-action
  warnings.
- Checked in the review, no change needed: Java's and Python's `validate()` gave the same verdict
  on 658 mutated real specs (17 kinds of mutation, `owner` included); builds, decisions,
  decision-point dumps and offline coach statistics were identical on two fresh JVMs serving 77
  real requests (30 of them coach) in opposite orders; `coach` over `belief.determinize` samples
  (options.specs, K=4) asked one question in 20 / 20 priority, 25 / 25 attack and 25 / 25 block
  decisions; game 2 decisions 6-7 of the Arena log decide in B's turn when Sear is replaced by an
  instant (Stab) instead of a Plains.

## replay_turn: one recorded turn through the engine

`python -m draftzero.gameplay.turnreplay --row 4 --turn 3` (driver: `src/draftzero/gameplay/turnreplay.py`).
The request is a turn-start spec (17lands: `eot_rollover` of user turn n with the turn's later draws
on A's library, `future_draws=True`) plus `options.script` and `options.expected`:

```
script    {seat: "A", turn: <global turn>, lands[{key, name}], casts[{key, name, family}],
           activations[{key, name: source}], attacks{"A:<alias>": bool, "new:<name>": true},
           attackGuess[names], blocks[[[blocker, attacker|null], ...], ...] (alternatives),
           blockPairing, opp[{kind, key, name, mv?, window?}], unscripted (A's recorded actions
           the request could not script, e.g. an ability with no XMage key)}
expected  {life{A, B}, hand{A[names], unknownA}, battlefield{A[names], B[names]} (tokens without
           " Token"), deaths{A|B: {combat[], noncombat[]}} (reported, not compared)}
options   seed, idSeed, maxAttempts (12), policies[] (explicit attempts), encode, perfectInfo,
          lenient, substitute
```

Both seats are `ReplayPlayer` puppets. A takes each scripted item at the first priority where it
is due (its window) and legal (XMage's `getPlayable`), so an item not legal yet (the card is still
to be drawn or bounced back, the mana is missing) carries over; it declares exactly the recorded
attackers (MageZero's UUID order, one CHOOSE_USE per available creature). B blocks per the chosen
pairing and casts its plays in a window read off the card: counterspells when the spell the
snapshot says did not resolve is on the stack, flash blockers after attackers, tricks after
blockers, removal after attackers, the rest in the end step. Auto-tap is steered: producers of the
colours later items need, Treasure-style producers and creatures about to attack are hidden from a
payment while the rest can pay. Targets, modes and colours come from the snapshot where it decides
them (`TargetResolver`), else a documented heuristic, flagged (`guessed_target`, `guessed_mode`, ...).
A harmful effect that could hit either seat prefers the other seat's doomed permanent: the
chooser's own doomed ones rank below every candidate of the other seat (they usually left by the
other seat's hand or in combat). Where labels could not tell which copy of a card attacked
(`attackGuess`), the recorded number of copies attacks, the planned one first. A recorded "Cast X"
with no castable copy is replayed as X's flashback from the graveyard when there is one
(`cast_as_flashback`: flashback granted by another card, or a copy flashed back after being cast
from hand the same turn, which 17lands logs as casts).

Attempts differ only in what 17lands does not record: spell order (mana value up / down),
main 1 vs main 2, A's instants in combat, B's windows, the land drop last, the block pairing,
and (added after an attempt that guessed them) the mode, "may" answers, target tie order,
Treasure use and a planeswalker defender. The first attempt that reproduces the turn wins;
otherwise the closest one is reported. Reproduced means: the end state matches (life, A's hand as
cleanup began, both battlefields by name per controller) **and** the replay did what was recorded:
every recorded play of A's happened (no `undone:A`, no `unscripted:A`), every recorded attacker
that is a spec permanent was declared and no creature recorded as not attacking was
(`attack:A:unrealised` / `attack:A:extra`, by count for `attackGuess` copies), and every recorded
block between two spec permanents was declared (`block:B:unrealised`), read from the engine's own
ATTACKER_DECLARED / BLOCKER_DECLARED events. An end state reached without them is a coincidence
whose decisions would teach the wrong play (a Stab on the user's own creature that the opponent
killed, a blocker removed before combat instead of during it). Attackers and blockers that entered
during the turn (`new:` refs) are only flagged (`attack_new_unrealised`, `block_new_unrealised`):
17lands lists a turn's attackers twice in about 3% of attacking turns (exactly doubled lists;
0.4% with 4 or more entries), which labels turns into `new:` attackers nothing can realise. B's
undone plays (`undone:B`: its hand is hidden) and deaths that differ from the record
(`deaths:A|B`; attributed to the owner, as 17lands does) are reported, not compared.

Every attempt rebuilds the spec with the same seeds, so the op is deterministic: same decisions,
features and diff across calls and JVMs (tested; in the review, 150 real turns with 795 encoded
decisions came back identical from two fresh JVMs serving them in opposite orders).

Response: `reproduced` (the verdict; the envelope's `ok` only says the request ran), `attempt`,
`attempts`, `policy`, `diff` {life, hand, battlefield, undone, divergence, deathsInfo}, `diffKeys`, `items`
(done / due / at), `targets`, `flags`, `notes`, `end`, `attemptLog`, `warnings`, `timing_ms`, and
`decisions`: every decision of A's reached in its main phases, combat and target choices,
`{type, text, where, legal[{label, idx}], chosen, label_kind, evidence, set?, features?}` with
label_kind `exact` (recorded: attacks, a Pass once nothing is left to do, a target the snapshot
settles), `imputed_order` (a recorded play whose moment is the policy's; `set` lists the recorded
plays legal there; a "no" for a creature labels left out of the attack plan; the attacking copy
of an `attackGuess` card) or `guessed_target` (targets, modes, "may" answers from the heuristic).
A target question whose options all carry one label (two copies of a card) is not recorded.

Measured 2026-09-26 after the review (one JVM; 2,400 FDN user turns, 600 per tier, spread over
the whole file): seed 7 reproduced 84.1% of the stratified sample (T0 98.0%, T1 81.5%, T2 76.5%,
T3 80.5%), 88.4% at the file's natural tier mix, 91% of the reproduced turns on the first attempt;
seed 1 (the build's sample) 84.8% (T0 98.2%, T1 84.0%, T2 78.5%, T3 78.7%; natural mix 89.0%).
Worker time per turn median 14 ms, p90 85-91 ms, max 841 ms (reproduced turns median 13 ms, p90
27 ms); 25.5 turns/s end to end with one JVM (Python request building included); 4.8 decision
records per reproduced turn. `python -m draftzero.gameplay.turnreplay measure --turns 2400
--seed 7 --workers 1`. The stricter verdict turned 17 of the 4,029 turns the build had reproduced
on these two samples into failures (undone recorded plays, unrealised recorded attacks and
blocks), and the target, flashback and attacking-copy fixes reproduced 43 others; before the
review, seed 1 measured 84.4% and seed 7 83.5%. Top failure keys (seed 7, 381 failures): life:B
89 (73 of them in specs flagged counters_inexact; B mostly takes 1-2 less damage than recorded:
counters the spec does not know), undone:A 60, bf:A:missing 49, bf:B:missing
47, bf:B:extra 27, bf:A:extra 23, hand:A:extra 22, life:A 22, hand:A:missing 17,
block:B:unrealised 12, attack:A:unrealised 8, game_over 5.

Not modelled: what 17lands leaves out and the spec cannot carry (counters on the user's creatures,
cards milled or surveiled, graveyard contents for cost reductions), B's plays with no record
(flash creatures are inferred), which copy of a card was the target, triggers that need an
ordering, choices inside effects other than targets/modes/colours (they take XMage's defaults).

## Limitations

- **One decision per request** for `build`, `encode` and `coach`. The later decisions of a
  recorded human turn come from `replay_turn` (above), which records every decision of A's on
  the way; coaching a decision past the first of a state is not built yet.
- **Not reproducible yet**: per-turn watcher history (spells cast or life gained this turn),
  until-end-of-turn effects, linked exile (Banishing Light), face-down creatures, loyalty
  activations used, transformed faces, and activated or triggered abilities on the stack (the
  stack holds spells only).
- **Choices made on entry and spell modes are guesses.** The spec has no field for a choice a
  permanent made as it entered (a color, a creature type; reported as warnings) or for the mode
  of a modal spell on the stack (`Card.cast` takes the first mode: Abrade on the stack deals 3
  damage to its target creature, never "destroy target artifact").
- **Control changes are permanent.** An `owner` permanent's original controller is its
  controller, which is right for reanimation. A creature taken with an Aura (Mind Control) stays
  with the taker if the Aura leaves, and one taken until end of turn (Threaten) stays past the
  turn. Auras that grant control still apply while attached.
- **Substitution changes the game.** A stand-in Plains is not the card it replaces: legal sets
  and search values around substituted cards are artefacts (on the Arena log, all 9
  XMage-only and 8 Arena-only legal options are substituted cards). Filter coaching and imitation
  rows on `substitutions`.
- **Decisions the engine never asks.** A spec for an Arena mulligan or starting-player decision
  (turn 1 UPKEEP) reaches the next priority decision, or none (Determinism): `init()` has
  already run, and the MCTS players never mulligan. A spec taken mid-action (Arena `SelectTargetsReq`, flagged
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
