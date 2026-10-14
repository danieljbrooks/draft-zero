# Roadmap

*Updated 2026-10-14. The plan behind phase 1 is [docs/033](docs/033-v0.2.0-refactor-plan.md); experiments are registered in [docs/runs.md](docs/runs.md#experiment-register). The XMage-era roadmap is at the [v0.1.0 tag](https://github.com/danieljbrooks/draft-zero/blob/v0.1.0/ROADMAP.md).*

The goal: agents that play Magic limited tournaments end to end (drafting, building and playing) learned only from self-play, trained at a scale where their card and deck statistics can be trusted. Four phases get there.

| Phase | Name | What it delivers | Status |
| --- | --- | --- | --- |
| 1 | Infrastructure | DraftZero 0.2.0: one reproducible actor-learner pipeline on gorge, and phase 1's experiments run on it | in progress |
| 2 | Experiments | The recipe: budgets, architecture and the order of components, found on distributed actors | later |
| 3 | Full-scale training | The end-to-end model on the full set | later; may need compute funding |
| 4 | Storytelling | Write-ups, human games, gorge as a product, the community | later |

## Phase 1: Infrastructure

### The 0.2.0 refactor

docs/033's work plan. Steps 0–4 run in order; after step 4, steps 5–7 can run in parallel. Contract tests land with each interface.

| Step | What | Done when | Status |
| --- | --- | --- | --- |
| 0 | Freeze and tag v0.1.0 | GitHub shows only `main`, release v0.1.0 and the two archive tags | done 14 October; docs/031's play network still to add to the release |
| 1 | Pin gorge through our fork | The fork's `dzgorge` builds and plays a fixed-seed smoke match | done 14 October: gorge main `813dcf3b4` plus four commits; the GitHub push waits for a token with the `workflow` permission |
| 2 | New layout, XMage out | A fresh `pip install -e .` pulls no MageZero, unit tests pass, no XMage import remains | open |
| 3 | Config, manifest, run directory, metrics schema | The smoke config writes its manifest and metrics; every file in `configs/` passes the schema test | open |
| 4 | The actor-learner on gorge | The smoke run passes in both modes; a mid-run change lands and is logged; a sync replay is byte-identical | open |
| 5 | Evaluator, ratings, plots, report, tracker | `dz report` draws the standard plots for the smoke run, and Trackio shows the same run | open |
| 6 | The deck builder on the loop | It reproduces docs/032's generation-1 result within its interval | open |
| 7 | Opponent-deck model | `:closedlist` on the run's own decks, then a learned model; held-out metrics logged; sampled worlds measured against true-deck worlds | open |
| 8 | Tests and CI | CI is green on `main`; a change to a contract file without a version bump fails CI | open |
| 9 | Docs | Every doc has its date line; the README has every section of the outline | drafted 14 October; the date-stamp script and a template for new docs remain |
| 10 | Reproduce a baseline, tag v0.2.0 | docs/031's best recipe, rerun on the new pipeline, matches its v0.1.0 strength within the interval | open |
| 11 | Walkthrough with Dan | Dan launches, changes and publishes a run without help | open |

### Phase 1's experiments

They run in this order on the 0.2.0 pipeline. Each needs only components 0.2.0 builds, except drafting, which uses gorge's `draftlab`.

1. **E1, baseline self-play.** The player (MLP, then GNN) trains from scratch on the actor-learner with decklists hidden (`:closedlist`); the deck builder is fixed. Does the async loop match docs/031's loop F at equal compute? Its result is the baseline for every later run.
2. **E2, actor and learner budgets.** Actor simulations (25, 50, 100), the share of decisions recorded, the replay ratio and the window, plus playout-cap randomisation: P1 (full 200 / cheap 40, full share 0.5) and P3 (500 / 100, 0.25) against the loop-F control. Which mix gives the most strength per CPU-hour? This seeds phase 2's recipe search.
3. **E3, learn deckbuilding.** docs/032's builder on the loop, first with E1's player frozen, then with the two alternating. Does its 63.8% against human decks hold with hidden decklists, and rise with a stronger player?
4. **E4, learn drafting.** A picker learns from the builder's valuation of the pools it drafts, in `draftlab` pods, as a `gorge-agent/1` `pick` agent.
5. **E5, joint training.** Player, builder and drafter learn together from the same games, against the sequential order E1, E3, E4.
6. **More, as candidates:** Gumbel root selection, the transformer or GNN inside the engine, auxiliary targets (life totals, turns left), an mtg-kernel cross-check.

### Asks of dama

- Take patch 0008 and both 0009s upstream, and our driver deltas into `cmd/dzgorge`, so the fork can retire.
- Make `searchprobe.Belief` an interface, so a learned opponent-deck model can sit behind it.
- Confirm that `cmd/dzgorge` is upstream-owned.

## Phase 2: Experiments

Runs on phase 1's infrastructure, at a larger scale:

- the recipe search: actor and learner budgets, targets and exploration;
- distributed actors with Hugging Face Storage Buckets as the store;
- a publishable experiment plan, with every run registered before launch;
- an architecture choice for the player (MLP, transformer or GNN, and what can run inside the engine);
- a draft prototype on `draftlab`;
- hardening: CI, contract tests, recovery from lost machines.

## Phase 3: Full-scale training

The end-to-end model (draft, build, play) on the full set, trained with phase 2's recipe. It may need compute funding. Every run starts from a tagged release and is published with its manifest.

## Phase 4: Storytelling

- Write-ups and papers from the published runs.
- Games against people.
- gorge as a product, with dama.
- Communication with the community: 17lands, Forge, MageZero and the gorge users.
