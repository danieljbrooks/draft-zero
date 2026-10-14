# Rules for agents working in this repository

DraftZero trains agents for Magic limited by self-play on gorge. Read [docs/architecture.md](docs/architecture.md) before changing code, and [docs/runs.md](docs/runs.md) before launching a run. These rules apply to every agent session, Claude or otherwise.

## Stop and ask before changing a rigid interface

Ten interfaces are rigid: every result, checkpoint and comparison between runs depends on them. Do not change one without Dan's review. Stop and ask first, even for a change that looks small.

| # | Interface | Paths |
| --- | --- | --- |
| 1 | Engine pin | `engine/gorge/` (`GORGE_REPO`, `GORGE_REF`, `build.sh`) |
| 2 | Card corpus pin | gorge's `FORGE_REF` and `.cards/cards.lock`; changes only with a new `GORGE_REF` |
| 3 | Driver CLI and policy strings | `dzgorge` flags and policy keys on the gorge fork; `src/draftzero/engine/specs.py` |
| 4 | Scorer hook | gorge's `policynet.External`, `internal/gonet`, `searchprobe.Belief` |
| 5 | Position encoding | gorge's `internal/policynet/` features |
| 6 | Training data and socket format | `src/draftzero/nets/SPEC.md`, `src/draftzero/nets/packfmt.py` |
| 7 | Checkpoint formats | `src/draftzero/nets/export.py` (`.pt`, `.dzgw`) |
| 8 | Run config schema | `src/draftzero/config.py` |
| 9 | Run manifest and metrics schema | `src/draftzero/manifest.py`, `src/draftzero/metrics.py` |
| 10 | Evaluation protocol and data splits | `src/draftzero/eval/`, `assets/decks.tsv`, `assets/sealed/` |

`CODEOWNERS` lists the paths in this repository, so GitHub asks for Dan's review. When Dan approves a change:

1. Bump the interface's version (for example `dzg-pack-1` → `dzg-pack-2`), and keep a reader for the old version while published data uses it.
2. Update its contract test in `tests/contract/`.
3. Add a line under **Interfaces** in `CHANGELOG.md`.

Free to change with green tests: new network architectures (added beside the old ones; a published architecture is never edited), trainer internals, actor, learner and controller internals, plots, experiment configs and scripts, and docs other than published results.

## Never commit

- **Forge card scripts, gorge's `.cards/` directory or anything compiled from them.** They are GPL-3.0 and are downloaded at build time only.
- **Secrets:** Hugging Face tokens, RunPod keys, alert tokens, `.env` files, ssh keys. Load them at run time (`draftzero.secrets`), and never print them in logs or put them in URLs.
- **Generated data:** `data/`, `runs/`, 17lands downloads, game corpora, shards, checkpoints. Small, reviewable inputs go in `assets/`.

## Tests

- Before every commit, run the unit and contract tests: `pytest -m 'not engine and not smoke'`.
- When you change the engine wrapper, the driver flags or the loop, also run `pytest -m engine` and `pytest -m smoke` against a built gorge (`dz engine build`).
- Every bug fix comes with a test. Every file in `configs/` must pass the schema test.
- Keep tests small: synthetic shards, the sample decks, a few games. Put a hard timeout on anything long.

## Commits, runs and anything public

- End every commit message written in an agent session with a `Co-Authored-By:` line naming the model, for example `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Do not push to `main`, create tags or releases, upload to Hugging Face, start paid machines or delete remote data unless Dan asked for that action. Publishing a run always stops for his confirmation.
- Register an experiment in [docs/runs.md](docs/runs.md#experiment-register), with its question and success criterion, before launching it. A publishable run starts from a release tag and a clean tree.

## Docs

- Docs are numbered in the order they are written; a new doc takes the next free number. Numbers never change.
- Every doc starts with its title and one line under it: `*Started YYYY-MM-DD · Applies to vX.Y.Z · Kind*`, where Kind is Plan, Report, Run log, Reference or Review. Add it to the [docs index](docs/README.md).
- Write plainly: lead with the point, short sentences, numbers with units, tables for items against attributes, no marketing language. Spelling is British-leaning (colour, behaviour).
- Keep machine-specific notes (hosts, containers, paths on one computer) out of the repository.
