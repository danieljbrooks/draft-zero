# DraftZero architecture (v0.2.0)

*Applies to v0.2.0. Kept current with the code; the plan behind it is [docs/033](033-v0.2.0-refactor-plan.md).*

DraftZero trains agents for Magic limited by self-play on gorge, a Go rules engine with a tree search. Two repositories do the work: DraftZero (Python) trains the networks and runs the loop, and our fork of gorge (Go) plays the games. Actors play games and write shards to a shared store; learners train on recent shards and publish new versions; an evaluator rates each version; a controller decides what plays. Ten interfaces between these parts are rigid: changing one needs Dan's review.

## Repositories and what each owns

![Architecture: repositories and the ten interfaces](img/033-architecture-light.png)

*The diagram predates gorge main's update of 14 October: the `dzgorge` driver is now upstream's `cmd/dzgorge` on the fork (not built from DraftZero source), and `engine/gorge/` holds only the pin and the build script.*

| Repository or service | Language, licence | Owns | How DraftZero uses it |
| --- | --- | --- | --- |
| [danieljbrooks/draft-zero](https://github.com/danieljbrooks/draft-zero) | Python, MIT | The actor-learner loop, network training and export, run configs, manifests, metrics, evaluation, plots, deck data, docs | This repository |
| [danieljbrooks/gorge](https://github.com/danieljbrooks/gorge), branch `draftzero` | Go, Apache-2.0 | gorge main `36a6c1c36` plus eight DraftZero commits: 0008 topk-max-options, 0009 sims-by-kind, the 0009 mayplay fix, the driver deltas (`-deckfile`, `-pairs-file`, `ksims=`, `-topk-max-options`, `-audit-*`), playout-cap randomization (`pcr=`, `cheap=`, `pcrk=`; `5fd039e`), the decided-game cheap search (`decided=`, `decidedn=`, `decidedsims=`, `decidedw=`; `67d2cd1`) forced playouts with policy target pruning (`forced=`; `a7afdc1`) and `dzgorge sealed-pools` (`8aa1f37`); rebased on 14 October evening to `27ff42e` | Pinned by `engine/gorge/GORGE_REF`; `engine/gorge/build.sh` (`dz engine build`) builds its `cmd/dzgorge` |
| [adams-shaun/gorge](https://github.com/adams-shaun/gorge) (dama's upstream) | Go, Apache-2.0 | The rules engine, the search (`internal/azmcts`), the position encoding (`internal/policynet`), the in-engine MLP (`internal/gonet`), `cmd/dzgorge`, closed-list worlds, `draftlab` and `gorge-agent/1` | The fork rebases onto it. Once it takes our five commits, DraftZero pins it directly and the fork retires |
| [adams-shaun/forge](https://github.com/adams-shaun/forge) (dama's fork of Forge) | Java, GPL-3.0 | The card scripts, at gorge's `FORGE_REF` (`098b0553`) | gorge downloads and compiles them into `.cards/`. Never committed or published |
| [Hugging Face](https://huggingface.co/danbrooks) (`danbrooks`) | — | Private Storage Buckets for runs in progress; public model and dataset repos for published runs; the private `draftzero-checkpoints` archive | Storage only; compute comes from elsewhere. See [runs.md](runs.md) |
| [17lands public data](https://www.17lands.com/public_datasets) | CC BY 4.0 | FDN decks, sealed pools, games-in-hand win rates | `assets/decks.tsv`, `assets/sealed/`, `assets/reference/`, with attribution |

mtg-kernel may join later behind the same Python engine interface. XMage and MageZero were removed in v0.2.0; they remain at the v0.1.0 tag.

## The ten rigid interfaces

Every result, checkpoint and comparison between runs depends on these. Everything else can change freely, by Dan or an agent, as long as the tests pass.

| # | Interface | Where it lives | Between | Why it is rigid |
| --- | --- | --- | --- | --- |
| 1 | Engine pin | `engine/gorge/GORGE_REF` (40-hex fork commit) and `engine/gorge/GORGE_REPO` | DraftZero ↔ gorge | It sets the game rules and the search. A rules change upstream (such as census-on) changes results |
| 2 | Card corpus pin | gorge's `FORGE_REF`, recorded in `.cards/cards.lock` | gorge ↔ Forge fork | It decides what every card does |
| 3 | Driver CLI and policy strings | `dzgorge play` and `dzgorge pack` flags: upstream's `cmd/dzgorge` plus DraftZero's deltas. Specs such as `az:sims=100:autopay:oppnodes:oppfull:morekinds:closedlist:topk=6:gonet=…`, built by `src/draftzero/engine/specs.py` | Python ↔ Go driver | Every config and script speaks it |
| 4 | Scorer hook | `policynet.External` and `internal/gonet` in gorge; later a learned opponent-deck model behind `searchprobe.Belief` | gorge's search ↔ our networks | The only code seam between the engine and our models |
| 5 | Position encoding | gorge's entity features (`internal/policynet/entity.go`, `features.go`, `option.go`) | gorge ↔ networks | A change invalidates every trained network |
| 6 | Training data and socket format | `src/draftzero/nets/SPEC.md`, `nets/packfmt.py`: shards `dzg-pack-1`, wire `GZB1`/`GZR1` | Go ↔ Python | Shards on disk and on Hugging Face must stay readable |
| 7 | Checkpoint formats | `.pt` (architecture, config, weights) and `.dzgw` (`dzgw-1`, the in-engine MLP), written by `src/draftzero/nets/export.py` | trainer ↔ engine, published models ↔ users | Published models must keep loading |
| 8 | Run config schema | `src/draftzero/config.py` | people ↔ pipeline | Old configs must keep running, or fail loudly |
| 9 | Run manifest and metrics schema | `src/draftzero/manifest.py` (`draftzero-manifest/1`), `src/draftzero/metrics.py` (`draftzero-metrics/1`) | runs ↔ reports and papers | Reports and plots read every run the same way |
| 10 | Evaluation protocol and data splits | `src/draftzero/eval/` (opponents, simulations, paired games, seeds); `assets/decks.tsv` and `assets/sealed/` (train, val and test splits) | runs ↔ runs | Changing either breaks comparisons with earlier runs; the test split is used once |

Interfaces 2, 4 and 5 live inside gorge. They reach DraftZero only through a new `GORGE_REF`, so a change to them is also a change to interface 1. [docs/036](036-gorge-draftzero-interface.md) lists what our fork of gorge changes and why, and the interface that would let DraftZero drop it.

**Free to change, with green tests:** new network architectures (added beside the old ones, never edited once published), trainer internals, actor, learner and controller internals, plots, experiment scripts, and docs other than published results.

### How to change a rigid interface

1. Open a pull request that Dan reviews. `CODEOWNERS` lists the DraftZero paths of these interfaces, so GitHub asks for his review.
2. Bump that interface's version, for example `dzg-pack-1` → `dzg-pack-2`, `draftzero-manifest/1` → `/2`, or a new `GORGE_REF`. Keep a reader for the old version while published data uses it.
3. Update its contract test in `tests/contract/` and add a line under **Interfaces** in `CHANGELOG.md`.

`AGENTS.md` lists the same ten, so an agent stops and asks before touching one. A change on the gorge side lands as a commit on the fork (or upstream) first, then reaches DraftZero through steps 1–3 for interface 1.

## Package layout

```
draft-zero/
  README.md CHANGELOG.md ROADMAP.md AGENTS.md CODEOWNERS LICENSE pyproject.toml
  engine/gorge/       GORGE_REPO, GORGE_REF, build.sh          (interface 1)
  src/draftzero/
    cli.py            the dz command
    config.py         run config schema                         (interface 8)
    manifest.py       run provenance                            (interface 9)
    metrics.py        metrics log and schema; wilson intervals   (interface 9)
    tracking.py       mirrors metrics into Trackio when enabled
    hub.py            hf://owner/repo[@rev]/path model files
    engine/           gorge.py (the Gorge class), specs.py (policy strings, interface 3)
    nets/             player networks: packfmt, data, models, targets, train, serve, client, export,
                      gonet_parity, bench, SPEC.md                (interfaces 6, 7)
    loop/             store, control, actor, learner, evaluator, controller, local, relay, cli, builder (the
                      deck builder's actor, learner and evaluator)
    eval/             paired games, ratings, games-in-hand statistics, 17lands reference (interface 10)
    plots/            the standard figures
    data/             17lands decks, pools and splits
    build/            the sealed deck builder (docs/032): cards, model, build, kinds, train, pools (draftlab)
    oppdeck/          the opponent-deck belief corpus (v0: gorge's :closedlist)
    workers/          local, ssh and RunPod machines
    alerts.py secrets.py resources.py watchdog.py
  configs/            base/player.yml, smoke.yml, smoke-distributed.yml, smoke-sealed.yml, runs/*.yml
  assets/             decks.tsv, sealed/, reference/, cards/, sample/
  docs/               numbered docs, README.md (index), architecture.md, runs.md, img/
  tests/              unit/, contract/, engine/, smoke/
```

| Package | Was in v0.1.0 |
| --- | --- |
| `engine/gorge/` | `gorge/GORGE_REF`, `gorge/build.sh`; `gorge/cmd/dzgorge/` and `gorge/patches/` moved to the fork |
| `draftzero.nets` | `gorge/dzg/` |
| `draftzero.build` | `gorge/sealed/` |
| `draftzero.loop` | `draftzero.selfplay` (store, control, controller) |
| `draftzero.data` | `gorge/decks.py`, `draftzero.paths`, `tools/extract_decks.py` |
| `draftzero.eval` | `gorge/analyze.py`, `gorge/recipe/score.py`, `stats.py`, `reference.py` |
| `draftzero.plots` | `gorge/figures.py`, `gorge/dzg_figures.py`, `gorge/recipe/figs.py` |

Tests sit in four tiers, selected by pytest markers. `tests/unit/` covers one module at a time on synthetic data; `tests/contract/` has one test per rigid interface. Both run with plain `pytest` (the default markers leave out the other two), and CI (`.github/workflows/ci.yml`) runs them on every push and pull request. `tests/engine/` needs a built gorge (`pytest -m engine`), and `tests/smoke/` runs a whole tiny run on the sample decks (`pytest -m smoke`); both run locally.

### The engine from Python

`dz engine build` runs `engine/gorge/build.sh`: it fetches the fork at `GORGE_REF`, downloads and compiles the cards if needed, builds `cmd/dzgorge` and writes `bin/dzgorge.build.json`. `draftzero.engine.Gorge` then wraps the binary:

| Call | What it does |
| --- | --- |
| `version()` | Repository, fork commit, dirty flag, binary SHA-256, build record, card lock and census setting, for the manifest |
| `coverage()` | Which decks the engine fully supports |
| `play(a, b, pairs=…, seed=…)` | Runs `dzgorge play` between two policy specs; returns complete games, the summary and, with `record=True`, the visit corpus. A hung chunk is killed at its timeout and its first missing pair replayed once, then skipped |
| `pack(corpus, out_dir)` | Runs `dzgorge pack` into `dzg-pack-1` shards (with the optional `weight.npy` when the corpus holds decided-game records), shrinks card features to float16 and prefixes game ids per actor |

`PolicySpec` builds and parses policy strings: kind (`az`, `prior`, `bot`, `sbtactical`, `random`), simulations, `cpuct`, `topk`, `explore`, `closedlist`, playout-cap randomization (`pcr`, `cheap`, `pcrk`), the decided-game cheap search (`decided`, `decidedn`, `decidedsims`, `decidedw`), forced playouts (`forced`) and `gonet=`. From the shell, `dz engine version` prints the build record, `dz engine coverage` the decks the engine fully supports, and `dz play --a SPEC --b SPEC --pairs N` plays two specs against each other.

## The actor-learner loop

![The actor-learner loop](img/033-actor-learner-light.png)

Actors write shards and read new versions; learners read a window of shards and write versions. Nothing else connects them, so actors can run on any machine. Each role is `dz <role> --config X --store S [--machine M]` (the same as `python -m draftzero.loop.cli <role> ...`), and `dz run CONFIG` starts all of them on one machine.

| Role | How many | What it does |
| --- | --- | --- |
| Actor | many, on any machine | Reads `control.json`; fetches the current player version (`.dzgw`); plays a chunk of self-play games through `Gorge.play` with exploration, `:closedlist` and the network in the search; uploads the shard (corpus, games, meta), a heartbeat and metrics in one batch. Version 0 has no network |
| Learner | one per learned model, holding a lease | Ingests new shards and packs them; trains on the newest window, warm-started from the current version; exports `.dzgw`; publishes `versions/player/v{N+1}` and `latest.json`; logs `learn.*` metrics |
| Evaluator | a few | Plays the evaluation shards the controller assigns to it: paired games of a version against one opponent of `eval.opponents` (`prev`, `sbtactical`, `bot`, `nonet`, `v<N>`, `file:…`) on the eval split, without exploration; uploads each shard's games, and whichever evaluator uploads a job's last shard writes `evals/<job>/summary.json` and the `eval.score` metric |
| Controller | one | Writes `config.yml` and `manifest.json` at the start; applies plan requests (`paused` and `budget.*` at once, the rest at the next version), logs them to `events.jsonl` and moves rejected ones to `plan/rejected/`; keeps `control.json` current; opens an evaluation job per evaluated version and opponent, split into shards of 10 pairs, assigns them to live evaluators and moves a shard whose evaluator went quiet; writes `status.md` and `status.json`; stops the run on `stop.request` or the budget |

**Two modes.** Async is the default: actors never wait, and each shard records the version that produced it. Runs are fully audited but not bit-for-bit repeatable. In sync mode the controller fixes the version, actors play until `sync.games_per_version` games exist for it, and the learner trains exactly once per version, so the run is built to replay exactly from its manifest (no byte-identical replay test yet: docs/033 step 4).

**Seeds.** An actor's chunk seed is `int(sha256(f"{run_seed}|{machine}|{seq}")) % 2**31`. The evaluator uses `eval.seed` for every version, so versions meet the same deals.

### Store layout

The store is a local folder or a Hugging Face Storage Bucket (`hf://buckets/<owner>/<bucket>`); machines without a token run against a local folder of their own that a relay keeps in step over ssh. The layout is the same everywhere:

```
config.yml, manifest.json                  the run as launched (controller, at start)
control.json                               {"format": "dz-control/1", "version": {"player": N}, "play": {…},
                                            "evals": [jobs with their shard assignments], "stop": bool,
                                            "paused": bool, "updated": …}
status.md, status.json                     the controller's status page (`dz status STORE [--json]`)
events.jsonl                               plan changes {time, kind: "set", path, old, new, by, at_version}
plan/requests/<uuid>.json                  a requested change {path, value, by, time}, from `dz plan … set`
plan/rejected/<uuid>.json                  a refused request and its reason
versions/player/v0001/                     model.pt, model.dzgw, meta.json {version, parent, sha256, shards,
                                            positions, games, metrics, time}
versions/player/latest.json                {version, path}
shards/<actor>/000001/                     corpus.jsonl.gz, games.jsonl, meta.json {actor, seq, seed, version,
                                            games, records, sims, spec, time, wall_s}
evals/<job>/<shard>.games.jsonl            evaluation games
evals/<job>/summary.json                   paired score per opponent
heartbeats/<machine>.json                  liveness, one file per machine
metrics/<component>/<machine>/000001.jsonl metrics in the draftzero-metrics/1 schema
learner/lease.json, ingested.json, status.json
relay/status.json, relay/uploaded.json     the relay's last sync and upload ledger (relay runs only)
stop.request                               present = stop the run
```

A local store lists no dot-files: rsync's and the relay's temporary files (`.relay-tmp/`) are never read as store
files.

**One writer per file.** Each actor writes only under `shards/<actor>/`, the learner only `versions/` and `learner/`, the evaluators only `evals/`, and the controller only `config.yml`, `manifest.json`, `control.json`, `events.jsonl`, `plan/rejected/` and the status page. `dz plan` writes a request file, and the controller applies it. On Hugging Face this keeps API calls low: an actor uploads one batch per chunk.

**The relay.** A machine that holds no token hands its shards over ssh to a relay (`dz relay`), the only process with the token. The relay pulls complete shards (the player's `shards/` and the deck builder's `builder/shards/`), `heartbeats/` and `metrics/` from remote actor folders into its store and pushes `config.yml`, `control.json` and the newest versions of each kind back; with a bucket store it mirrors through a local folder.

### Changing the plan mid-run

Fields are live or fixed. Live fields (`actor.sims`, `actor.pcr.*`, `actor.decided.*`, `actor.forced.*`, `actor.record_every`, `actor.chunk_games`, `learner.lr`, `learner.window_positions`, `learner.min_new_positions`, `learner.steps_per_position`, `learner.epochs`, `eval.every_versions`, `eval.pairs`, `eval.sims`, `budget.*`, `paused`) change with `dz plan STORE set learner.lr=3e-4`. The controller validates the change; a rejected one moves to `plan/rejected/` with the reason. It applies `paused` and `budget.*` at once (no version arrives while actors are paused, so a pause must lift without one) and every other change at the next version, and appends an event either way. The effective config at any version is the launch config plus the events up to it. Any other field is fixed: change it by starting a new run (`dz run --set` sets any field at launch).

## Where the parts plug together

One game's path through the system:

1. An actor builds a policy spec from the config and the current version, for example `az:sims=100:explore:autopay:oppnodes:oppfull:morekinds:closedlist:cpuct=0.25:topk=6:gonet=…/v0012/model.dzgw` (interface 3).
2. `dzgorge play` runs the game at `GORGE_REF` (1) with the cards at `FORGE_REF` (2). The search encodes positions (5) and scores them with the MLP through `gonet` (4, 7). With `:closedlist`, each simulation deals the opponent's hidden cards from a belief corpus of the run's own decks.
3. The recorded decisions (visit counts and root values) go to the store as a shard; the learner packs them into `dzg-pack-1` (6).
4. The learner trains the next version and exports `.pt` and `.dzgw` (7).
5. The evaluator plays it in paired games under the evaluation protocol (10). Every step logs metrics, and the manifest records the code, engine, cards and config (8, 9).

Deckbuilding and drafting sit beside play. The deck builder (`draftzero.build`, docs/032) plays its decks through `dzgorge play -deckfile -pairs-file`; with `builder.learn` it trains on the loop (`draftzero.loop.builder`: builder actors, a builder learner and builder evaluators on the same store, versions under `versions/builder/`), on draftlab sealed pools from `dzgorge sealed-pools`. gorge's `draftlab` (packs, sealed pools, drafts, events) and `gorge-agent/1` (external `pick` and `build` agents) are the environment for the learned builder and drafter. Search-based play stays in Go.
