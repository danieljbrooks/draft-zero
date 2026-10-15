# DraftZero docs

The docs are numbered in the order they were written, and the numbers never change, so links keep working. Each doc carries one line under its title: the date it was started (the commit that first added it), the version of the code it applies to, and its kind. A **Plan** proposes work before it runs, a **Report** gives results, a **Run log** records runs as they happened, a **Reference** is a guide or survey to come back to, and a **Review** reads someone else's work. Docs 001–032 describe v0.1.0: the XMage pipeline, imitation learning and the first gorge work. Their code is at the [v0.1.0 tag](https://github.com/danieljbrooks/draft-zero/tree/v0.1.0), except docs/025's (tag `archive/test-mtg-kernel`) and docs/028–029's (tag `archive/selfplay-transition`). Number 030 is reserved for a plan from another session that was never committed. A new doc takes the next number and starts with its own line, for example `*Started 2026-10-20 · Applies to v0.2.0 · Report*`.

Two pages are not numbered, because they track the current code:

- [architecture.md](architecture.md): the v0.2.0 repositories, the ten rigid interfaces, the package layout and the actor-learner roles.
- [runs.md](runs.md): how runs are organised, where their data lives, how to load, reproduce and publish one, and the experiment register.

| Number | Title | Started | Applies to | Kind | Superseded by |
| --- | --- | --- | --- | --- | --- |
| 035 | [RunPod network volumes as a run's store](035-runpod-network-volumes.md) | 2026-10-14 | v0.2.0 | Report |  |
| 034 | [The first experiments on the 0.2.0 pipeline](034-v0.2.0-first-experiments.md) | 2026-10-14 | v0.2.0 | Report |  |
| 033 | [DraftZero 0.2.0 refactor plan](033-v0.2.0-refactor-plan.md) | 2026-10-14 | v0.2.0 | Plan |  |
| 032 | [Building sealed decks from self-play: FDN on gorge](032-sealed-deckbuilding.md) | 2026-10-11 | v0.1.0 | Report |  |
| 031 | [Searching for a self-play recipe that scales: laptop experiments on gorge](031-gorge-recipe-search.md) | 2026-10-11 | v0.1.0 | Run log |  |
| 029 | [From imitation to self-play: results](029-imitation-to-self-play-results.md) | 2026-10-08 | v0.1.0 | Report |  |
| 028 | [From imitation to self-play: what the literature suggests, and a plan of runs](028-imitation-to-self-play-plan.md) | 2026-10-07 | v0.1.0 | Plan |  |
| 027 | [Scaling up self-play on gorge: DraftZero's networks, games per dollar, and what limits learning](027-gorge-scaling-up.md) | 2026-10-07 | v0.1.0 | Report |  |
| 026 | [Self-play on gorge: FDN games, search, a first network, and 17lands statistics](026-gorge-fdn-self-play.md) | 2026-10-06 | v0.1.0 | Report |  |
| 025 | [Self-play on mtg-kernel with FDN limited decks](025-mtg-kernel-fdn-self-play.md) | 2026-10-06 | v0.1.0 | Report |  |
| 024 | [MageZero's graph network, part 2: local layers, a full run on Dama's GPUs, and games](024-gnn-local-layers-and-games.md) | 2026-10-05 | v0.1.0 | Report |  |
| 023 | [MageZero's graph network: the run log](023-gnn-run-log.md) | 2026-10-04 | v0.1.0 | Run log |  |
| 022 | [Testing MageZero's graph network for imitation learning](022-gnn-imitation-test-plan.md) | 2026-10-04 | v0.1.0 | Plan |  |
| 021 | [Self-play from the imitation start: architecture and plan](021-self-play-plan.md) | 2026-10-05 | v0.1.0 | Plan | [033](033-v0.2.0-refactor-plan.md) |
| 020 | [CPU benchmarks: where the MLP's compute should go](020-cpu-benchmarks.md) | 2026-10-03 | v0.1.0 | Report |  |
| 019 | [Learning to play Magic: The Gathering limited from human games](019-imitation-learning-report.md) | 2026-10-04 | v0.1.0 | Report |  |
| 018 | [Experiment #4: detailed run log](018-experiment4-run-log.md) | 2026-10-01 | v0.1.0 | Run log |  |
| 017 | [Experiment #4: Scaling up imitation learning](017-experiment4-scaling-up-imitation-learning.md) | 2026-09-30 | v0.1.0 | Plan |  |
| 016 | [Experiment #3: MCTS Method Benchmark](016-search-benchmark-results.md) | 2026-09-28 | v0.1.0 | Report |  |
| 015 | [Rules engines for a fast Limited agent: speed and MCTS support](015-rules-engine-comparison.md) | 2026-09-28 | v0.1.0 | Report |  |
| 014 | [Experiment #2a: self-play from a heuristic start (final report)](014-exp2-run1-report.md) | 2026-09-28 | v0.1.0 | Report |  |
| 013 | [Experiment #2b – Imitation: does an imitation start help? (#2a against #2b)](013-exp2-imitation-report.md) | 2026-09-27 | v0.1.0 | Report |  |
| 012 | [Search benchmark: decision quality, hidden information and compute](012-search-benchmark.md) | 2026-09-27 | v0.1.0 | Plan |  |
| 011 | [Experiment #2b – Imitation (run 2): starting self-play from human decisions](011-exp2-run2-imitation.md) | 2026-09-26 | v0.1.0 | Plan |  |
| 010 | [Moving to Will's v0.2 engine, and its pilot](010-v02-migration-pilot.md) | 2026-09-26 | v0.1.0 | Report |  |
| 009 | [Hidden information: what MageZero sees, what its search sees, and how other games do it](009-hidden-information.md) | 2026-09-26 | v0.1.0 | Report |  |
| 008 | [Human gameplay data: imitation learning and MCTS coaching](008-gameplay-data.md) | 2026-09-26 | v0.1.0 | Report |  |
| 007 | [AlphaZero-style distributed training](007-alphazero-distributed-training.md) | 2026-09-25 | v0.1.0 | Reference |  |
| 006 | [Experiment #2 pilot: settings, single-JVM tuning, parallel JVMs](006-exp2-pilot.md) | 2026-09-25 | v0.1.0 | Report |  |
| 005 | [Reserving RunPod instances: what worked and what didn't](005-runpod-tips.md) | 2026-09-25 | v0.1.0 | Reference |  |
| 004 | [lordofthepigs' Forge simulations: notes for DraftZero](004-price-predictor-review.md) | 2026-09-25 | v0.1.0 | Review |  |
| 003 | [FDN Generalist: one MageZero agent for a whole limited format](003-fdn-generalist-report.md) | 2026-09-25 | v0.1.0 | Report |  |
| 002 | [FDN long run — runbook](002-RUNBOOK.md) | 2026-09-21 | v0.1.0 | Reference |  |
| 001 | [FDN generalist](001-generalist-original-README.md) | 2026-09-21 | v0.1.0 | Reference |  |

Figures are in [`img/`](img/), named after the doc that uses them (`026-ladder-light.png`). Small data tables behind a doc are in [`data/`](data/).
