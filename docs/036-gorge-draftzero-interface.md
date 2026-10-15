# The gorge–DraftZero interface

*Started 2026-10-14 · Applies to v0.2.0 · Plan*

**DraftZero runs gorge through a fork, and would rather not.** The fork is `danieljbrooks/gorge`, branch
`draftzero`: gorge main `36a6c1c36` plus eight DraftZero commits (tip `27ff42e4c`, about 2,700 lines, 40% of them
tests). The commits add:
- self-play training options to the search;
- two driver additions;
- one rules fix.

Every option is opt-in: with its keys off, gorge plays exactly as upstream does. The simplest way to drop the fork
is for gorge to take the eight commits upstream (§4, option A). The alternative is a small set of public seams in
gorge's search (option B). gorge's existing external-agent protocol already covers DraftZero's deck builder and a
future drafter (option C).

## 1. How DraftZero uses gorge today

DraftZero is Python; gorge is Go. They meet at a process boundary: DraftZero builds gorge's `dzgorge` driver at a
pinned commit and runs it as a subprocess. Files carry everything else. gorge main already owns `cmd/dzgorge`:
dama took in the driver, most of v0.1.0's patches, the hidden-decklist worlds (`:closedlist`), `draftlab` and the
`gorge-agent/1` protocol.

| What crosses | Direction | Form | Rigid interface ([architecture.md](architecture.md)) |
|---|---|---|---|
| The engine | gorge → DraftZero | `dzgorge`, built by `engine/gorge/build.sh` at `GORGE_REF`, with a build record (`bin/dzgorge.build.json`: commit, Go version, binary hash, card lock) | 1 |
| Card behaviour | gorge | `.cards`, compiled from Forge's card scripts at gorge's `FORGE_REF` (`cards.lock`) | 2 |
| Commands | DraftZero → gorge | `dzgorge play`, `pack`, `version`, `coverage`, `sealed-pools`; policy specs such as `az:sims=200:autopay:oppnodes:oppfull:morekinds:closedlist:topk=6:gonet=model.dzgw` (`src/draftzero/engine/specs.py`) | 3 |
| Networks | DraftZero → gorge | `.dzgw` weights (`dzgw-1`, read by gorge's `internal/gonet`), or a scorer over a socket (`policynet.External`, wire `GZB1`/`GZR1`) | 4, 7 |
| Position encoding | gorge → networks | gorge's entity features (`internal/policynet`) | 5 |
| Games and training data | gorge → DraftZero | `games.jsonl`; the visit corpus (`corpus.jsonl.gz`), packed by `dzgorge pack` into `dzg-pack-1` shards | 6 |
| Inputs | DraftZero → gorge | the deck pool (`decks/`, `pool.tsv`), the belief corpus for `:closedlist`, deck and pairing files for builder games, sealed pools | 3 |

## 2. The fork, and why each commit exists

**History.** v0.1.0 pinned `adams-shaun/gorge@c9a422b` and applied nine patches at build time. Patches 0001-0007 and
the driver have since gone upstream. On 14 October the remaining work became commits on our fork, rebased onto gorge
main. The latest rebase, on 14 October evening, took in 102 upstream commits with two conflicts, both additions on
each side of `internal/azmcts/seat.go`. The previous tips are kept as branches (`draftzero-8aa1f37dc`,
`draftzero-c9a422b`), so every earlier pin still builds.

| # | Commit | Package | What it adds | Why DraftZero needs it | Lines |
|---|---|---|---|---|---|
| 1 | `17f1558` topk-max-options | `internal/azmcts` | `PriorRankMaxOptions`: above N options, the search keeps the bot's answer and a uniform prior instead of ranking every option with the network | One decision with 388 options cost minutes per searched move and held a game for hours | 14 |
| 2 | `65adfe1` sims-by-kind | `internal/azmcts` | `SimsByKind`: a simulation budget per decision kind (priority, attackers, blockers, target, ...) | Spend search where it pays: casting decisions more, blocks less | 7 |
| 3 | `2d5c18f` mayplay fix | `rules` | A card's own static ability grants nothing off the battlefield unless it names `EffectZone$` (CR 113.6) | Omniscience in hand gave itself a free cast, then every other spell. The self-play deck builder found it: decks that drew it won about 80% | 37 |
| 4 | `719019e` driver flags | `cmd/dzgorge` | `-deckfile`, `-pairs-file`, `ksims=`, `-topk-max-options`, `-audit-oracle`/`-audit-rate`/`-audit-out`, and a flush of `games.jsonl` after every game | Play given decklists and pairings (the deck builder's games and evaluations); expose 1 and 2; audit the search against an oracle; resume a match after a hard kill without replaying finished games | 281 |
| 5 | `274e1b0` playout-cap randomisation | `internal/azmcts` | `pcr=`, `cheap=`, `pcrk=`: at each decision a seeded coin picks a full search (recorded) or a cheap one (not recorded) | KataGo's way to play more games per CPU-hour. Measured in E2b and E2c (docs/034 §7): no gain at our scale, so it stays off by default | 669 |
| 6 | `87e5a51` decided-game search | `internal/azmcts` | `decided=`, `decidedn=`, `decidedsims=`, `decidedw=`: fewer simulations once a game is decided, those positions recorded at a lower training weight | 18.4% of recorded positions come after the game is decided; it saves 12.5% of simulations per searched decision (docs/034 §5-6) | 783 |
| 7 | `bb669f9` forced playouts | `internal/azmcts` | `forced=`: KataGo's forced playouts at the root, with the forced visits pruned back out of the policy target | Root exploration that doesn't pollute the training target (4.5% of visits pruned, docs/034 §6) | 602 |
| 8 | `27ff42e` sealed-pools | `cmd/dzgorge` | `dzgorge sealed-pools`: deterministic `draftlab` sealed pools as JSON lines, optionally each pool's `best23@1` deck | The deck builder's training and evaluation pools, reproducible from (set, collation, seed, packs) | 379 |

**What the commits promise:**
- **Opt-in.** An absent or zero key leaves gorge as upstream. For 5-7 this is tested: with their keys off, games are
  byte-identical to the commit before.
- **Deterministic.** Each draw is a pure function of the seat's seed and the decision's number, so results never
  depend on the worker count.
- **Validated.** `NewSeat` refuses out-of-range values.
- **Tested.** `go test` covers `internal/azmcts`, `cmd/dzgorge` and `rules` in the fork, and DraftZero runs its
  engine tests and a smoke run on each built binary.

One known issue: with `-audit-oracle`, the audited side records no visits.

**What the fork costs:**
- a rebase whenever gorge main moves (minutes while conflicts stay additive);
- a card recompile when gorge's card compiler changes, as it did on 14 October;
- the risk that a change upstream in the search internals (`seat.go`, `tree.go`, `search.go`) changes what our
  options mean without a conflict to flag it;
- results pinned to fork commits that gorge's own history never contained.

## 3. Why DraftZero can't simply import gorge

- **Go's `internal/` rule.** `dzgorge` imports seven `internal/` packages: `azmcts`, `gonet`, `policynet`,
  `searchprobe`, `searchseat`, `bench` and `spellbench/builtins`. Only code inside the gorge module may import them,
  so a separate DraftZero Go module can't build its own driver.
- **The options live inside the search loop.** Items 1, 2 and 5-7 change per-decision budgets, root selection and
  what gets recorded. Public packages alone wouldn't be enough: they need hooks inside `azmcts`, not a caller
  outside it.
- **What is already public is the rest:** `cards`, `rules`, `state`, `deck`, `decision`, `seat`, `view`, `bots`,
  `limited` and `draftlab`, with `draftlab/extern` for out-of-process agents.

## 4. Options for a clean interface

### A. Upstream the eight commits (recommended)

gorge takes them as they are, as opt-in keys. DraftZero then pins gorge main itself (`GORGE_REPO` =
`adams-shaun/gorge`, `GORGE_REF` = a main commit), with no fork and no Go code of its own. The interface stays what §1
lists. What gorge would take on: reviewing about 2,700 lines and keeping the keys working. Why it fits:
- the options are the standard KataGo self-play set, off by default;
- gorge's own draft-bot RL program (`docs/superpowers/specs/2026-10-14-draft-rl-design.md`) self-plays with the same
  az search;
- item 3 is a rules bug fix regardless (Dan's pull request).

### B. Seams in gorge; the options stay in DraftZero

If gorge would rather keep training options out of its core, this is the smallest API that lets DraftZero carry them
in a Go module of its own. The pin then moves into a `go.mod` line, `require github.com/adams-shaun/gorge
v0.0.0-<date>-<commit>`, with `go.sum` checking it.

| Seam | Sketch | Covers |
|---|---|---|
| A public search package | `azmcts`, or a facade such as `gorge/az`, exporting `Options`, `SeatConfig`, `NewSeat`, `Result`, `Stats` and `Diag` | Importing the search at all |
| A per-decision budget hook | `SeatConfig.Budget func(DecisionInfo) Budget`. `DecisionInfo` = kind, sequence number, turn, number of options, recent root values. `Budget` = simulations, root noise on/off, sampling on/off, record on/off, record weight, rank the prior on/off | 1, 2, 5, 6 |
| Root selection and target hooks | `Options.RootSelect func(RootView) (child int, ok bool)`, asked before PUCT at the searching seat's root; `Options.Target func(RootView) []float32`, the policy target after the search | 7 |
| A record sink | A public visit-record type with a weight field, and a `Recorder` interface | The corpus DraftZero trains on |
| The driver as a library | `dzgorge.Main(args []string, extra ...Command)` | 4 and 8: DraftZero's own binary, with its flags and commands added |
| Public loaders | The belief corpus (`searchprobe.LoadBeliefDir`), `.dzgw` weights (`gonet`), the deck pool | Building that binary |

What gorge would take on: designing these within its package rules (`archtest`, `pkgmap`), then keeping them stable.
The hooks are a narrower surface than the options themselves, but a harder one to change later.

### C. Out of process, over `gorge-agent/1`

gorge already lets an external process act as a drafter, a builder or a player. Each sees only its own seat's view,
and gorge batches the requests (`draftlab/extern`). This is the right interface for DraftZero's deck builder and a
future drafter: their models stay in Python, and gorge needs no change. It is the wrong one for the player, whose
search must run in Go beside the engine; an out-of-process player would send every simulation's question across the
wire.

**Recommendation:** A for all eight commits. If gorge prefers training options out of core, A for 3, 4 and 8 and
B's budget and target hooks for the rest. C for the builder and drafter from now on.

## 5. Contracts DraftZero needs kept, under any option

| Contract | Current version or form | Why it matters |
|---|---|---|
| Policy spec grammar and `dzgorge` flags | `az:...` keys; the flags in §1 | Every config and script speaks them |
| Game records | `games.jsonl` fields: pair, leg, seed, decks, policies, result, turns | Evaluation and paired scoring |
| Training records | the visit corpus and `dzg-pack-1` | Shards on disk and on Hugging Face must stay readable |
| Network weights | `dzgw-1` | Published models must keep loading |
| Position encoding | gorge's entity feature set | A change invalidates every trained network |
| Build record | `bin/dzgorge.build.json` | A run's manifest records which engine played it |
| Sealed pools | the `sealed-pools` JSON lines | The builder's splits are defined by them |
| Belief corpus | the deck directory and `pool.tsv` split | `:closedlist` worlds are sampled from it |

The ask: a format or version field in each output, the entity feature set's version recorded in every record, and a
line in gorge's history when one of these changes.

## 6. Questions for dama

- Would gorge take items 1, 2 and 4-8 upstream as opt-in keys? Item 3 is already Dan's pull request.
- If not, would a public search facade with the budget and target hooks fit gorge's package rules?
- Can the entity feature set carry a version number in the records it writes?
- Would gorge's draft-bot RL program use playout-cap randomisation or the decided-game search? Shared use argues for
  upstream.
- A naming clash: gorge's draft-RL spec calls its 17Lands-rated drafter "B17"; DraftZero's B17 is docs/031's network.
