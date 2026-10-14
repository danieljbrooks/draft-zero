# Changelog

All notable changes to DraftZero. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [semantic versions](https://semver.org/). While in 0.x, a minor bump (0.2.0, 0.3.0) marks a change to a rigid interface or to how the pipeline runs; a patch bump (0.2.1) covers fixes and additions that keep every interface. An **Interfaces** section lists every change to one of the ten rigid interfaces ([docs/architecture.md](docs/architecture.md#the-ten-rigid-interfaces)).

## [Unreleased]

Version 0.2.0, in progress (`0.2.0.dev0`). One reproducible, config-driven training pipeline on gorge; XMage and imitation learning are removed. The plan is [docs/033](docs/033-v0.2.0-refactor-plan.md).

### Added

- The actor-learner loop (`draftzero.loop`): actors, one learner per learned model, an evaluator and a controller that share nothing but a store. Async mode by default; a sync mode that replays exactly from the manifest.
- Stores: a local folder, a Hugging Face Storage Bucket (`hf://buckets/<owner>/<bucket>`), and a relay over ssh for machines without a token.
- Run configs (`draftzero.config`): one YAML file per run, checked against a schema, with `extends:` for shared bases (`configs/base/player.yml`). Live fields change mid-run with `dz plan`; each change is an event in `events.jsonl`.
- Run manifests (`draftzero.manifest`, `draftzero-manifest/1`): code, engine build, card lock, census setting, config and its hash, machine and seed.
- Metrics (`draftzero.metrics`, `draftzero-metrics/1`): one JSON line per value, with sample size and 95% interval where they apply. `draftzero.tracking` mirrors them into Trackio when enabled.
- The engine wrapper (`draftzero.engine`): `Gorge` (`version`, `coverage`, `play` with recovery from hung games, `pack`) and `PolicySpec` for policy strings.
- `engine/gorge/`: `GORGE_REPO`, `GORGE_REF` and `build.sh`, which builds the fork's `dzgorge` at the pin and writes a build record.
- Evaluation (`draftzero.eval`): paired-game scoring, SPRT, Bradley–Terry ratings; `sbtactical` and `nonet` as standard opponents. Standard figures in `draftzero.plots`.
- `draftzero.oppdeck`: a belief corpus of the run's own decks for gorge's `:closedlist`, the version-0 opponent-deck model.
- `draftzero.hub`: `hf://owner/repo[@rev]/path` model files.
- The `dz` commands `version`, `manifest` (alias `provenance`), `run`, `actor`, `learner`, `evaluator`, `controller`, `plan`, `status`, `stop`, `report`, `models pull`, `play`, `eval`, `engine build|version|coverage`, `pod up|down|list|balance|offers` (RunPod pods for relay actors; `up` requires `--max-price` and `--max-hours`) and `relay`.
- `assets/sample/`: 80 FDN decks in gorge's format (40 train, 40 eval), which the smoke configs play.
- CI: `.github/workflows/ci.yml` runs the unit and contract tests on Python 3.11 and 3.12.
- Four test tiers with pytest markers: `tests/unit/`, `tests/contract/` (one test per rigid interface), `tests/engine/` and `tests/smoke/`.
- `AGENTS.md`, `CODEOWNERS` (the rigid-interface paths) and this changelog.
- Docs: a start-date line on every doc, the [docs index](docs/README.md), [architecture.md](docs/architecture.md), [runs.md](docs/runs.md) with the experiment register, and docs/033, the refactor plan.

### Changed

- **The engine is gorge only,** pinned through the fork `danieljbrooks/gorge`, branch `draftzero`: gorge main `813dcf3b4` plus five DraftZero commits (0008 topk-max-options, 0009 sims-by-kind, the 0009 mayplay fix, the driver deltas, playout-cap randomization). v0.1.0 pinned `adams-shaun/gorge` at `c9a422b` and applied patches 0001–0009 at build time; 0001–0007 and the driver itself are now upstream.
- gorge's paid-mana census is on by default, and the search hides the opponent's decklist (`:closedlist`). Results and networks from v0.1.0 are not comparable with 0.2.0's.
- Packages moved: `gorge/dzg/` → `draftzero.nets`, `gorge/sealed/` → `draftzero.build`, `draftzero.selfplay` → `draftzero.loop`, `gorge/decks.py` → `draftzero.data`, `gorge/analyze.py`, `stats.py` and `reference.py` → `draftzero.eval`, figure scripts → `draftzero.plots`, `tools/compute_bench/pods.py` → `draftzero.workers.runpod`.
- `tools/extract_decks.py` no longer needs an XMage jar.
- Dependencies: the core is numpy, torch, pyyaml and huggingface_hub; extras `tracking`, `plots`, `sealed` and `dev`.
- `README.md` and `ROADMAP.md` rewritten; the roadmap is now short (phases 1–4).

### Removed

- XMage and MageZero: `java/`, the MageZero dependency, `engine.py`, `loop.py`, `final_eval.py`, `report.py`, `dashboard.py`, `pools.py`, `gameplay/`, `decks/`, and the XMage parts of `selfplay/` (worker, trainer, tables, GNN training, records). They remain at v0.1.0.
- Imitation learning, deprecated: it depended on turning 17lands games into XMage states.
- `gorge/cmd/dzgorge/` (now upstream's), `gorge/patches/`, `gorge/build.sh`, the `gorge/recipe/` and `gorge/*.sh` experiment scripts, `gorge/sortcorpus.py`, `gorge/figures.py`, `gorge/dzg_figures.py`.
- All 76 v0.1.0 configs, the deploy scripts except `runpod_arm.sh`, the tools except `extract_decks.py` and `engine_bench/speed_plot.py`, `assets/vocab/`, `assets/gameplay/FDN_{abilities,tokens}.tsv`, and the XMage-era tests.

### Interfaces

All ten rigid interfaces are named and versioned for the first time.

1. **Engine pin:** `engine/gorge/GORGE_REF` = `5fd039efe87fc601773df51f615fe45bf66a47e1` on `danieljbrooks/gorge` (was `gorge/GORGE_REF` = `c9a422b703f40acec744cb8c947b724f057305b3` on `adams-shaun/gorge` plus patches).
2. **Card corpus pin:** gorge's `FORGE_REF` = `098b0553` on `adams-shaun/forge` (was `ab2a79bd`).
3. **Driver CLI and policy strings:** upstream's `cmd/dzgorge` plus DraftZero's deltas. New: the `:closedlist` key with `-belief-decks`, `-belief-pool` and `-belief-split`; the `sbtactical` policy; `-no-paid-mana-census`, since the census is now on by default; playout-cap randomization (`pcr=`, `cheap=`, `pcrk=`). Kept from v0.1.0: `-deckfile`, `-pairs-file`, `ksims=`, `-topk-max-options`, `-audit-*`.
4. **Scorer hook:** `policynet.External` unchanged, now upstream; the in-engine MLP moved to gorge's `internal/gonet`.
5. **Position encoding:** gorge main's, which may differ from `c9a422b`'s. Check v0.1.0 networks with `dzgorge gonet-check` and retrain rather than reuse them.
6. **Training data and socket format:** `dzg-pack-1` and `GZB1`/`GZR1`, unchanged; the spec moved to `src/draftzero/nets/SPEC.md`.
7. **Checkpoint formats:** `.pt` and `.dzgw` (`dzgw-1`), unchanged.
8. **Run config schema:** new (`src/draftzero/config.py`). v0.1.0 configs do not load.
9. **Run manifest and metrics schema:** new: `draftzero-manifest/1` and `draftzero-metrics/1`. The manifest's `belief` key records the `:closedlist` belief corpus (decks, pool, split and `corpus_hash`), or null without `:closedlist`.
10. **Evaluation protocol and data splits:** paired games on the eval split at seed 3131 against `prev` and `sbtactical` by default; the splits in `assets/decks.tsv` and `assets/sealed/` are unchanged.

## [0.1.0] - 2026-10-14

Everything before the 0.2.0 refactor, kept reproducible. A pre-release on GitHub.

### Added

| Area | Docs | Code |
| --- | --- | --- |
| XMage pipeline: experiments 1–4, imitation learning, the GNN, the search benchmark | docs/001–024, 028–029 | `src/draftzero/`, `java/mzbridge/`, `tools/`, `configs/` |
| mtg-kernel comparison | docs/025 | tag `archive/test-mtg-kernel` |
| Self-play on gorge | docs/026–027 | `gorge/`: the `dzgorge` driver, the `dzg` networks, the patches |
| Recipe search | docs/031 | `gorge/recipe/` |
| Sealed deckbuilding | docs/032 | `gorge/sealed/`, `assets/sealed/` |

- **Pins:** gorge `adams-shaun/gorge` at `c9a422b` (`gorge/GORGE_REF`) with `gorge/patches/` applied in file-name order (two files carry the number 0009; all ten apply cleanly); Forge card scripts from `adams-shaun/forge` at `ab2a79bd`; MageZero `danieljbrooks/MageZero@521a8bd` with the v0.2 XMage bundle.
- **Checked:** `gorge-train` and `sealed-deckbuilding` merged into `main` with no file changed on both; `dzgorge` builds and passes `go vet` with all ten patches; gorge's search tests pass; `gorge/dzg` tests: 55 passed.
- **Tags:** `exp1-fdn-generalist` (experiment 1), `sealed-builder-v1` (docs/032's builder), and two archive tags for branches deleted after this release: `archive/selfplay-transition` (docs/028–029's XMage code) and `archive/test-mtg-kernel` (docs/025's harness). All other pre-0.2.0 branches are contained in `main`.
- **Models:** [`danbrooks/draftzero-fdn-exp1`](https://huggingface.co/danbrooks/draftzero-fdn-exp1), [`danbrooks/draftzero-fdn-gnn`](https://huggingface.co/danbrooks/draftzero-fdn-gnn) and [`danbrooks/draftzero-fdn-sealed-builder`](https://huggingface.co/danbrooks/draftzero-fdn-sealed-builder). docs/031's latest play network is still to be added.

[Unreleased]: https://github.com/danieljbrooks/draft-zero/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/danieljbrooks/draft-zero/releases/tag/v0.1.0
