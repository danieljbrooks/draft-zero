# Roadmap

Where DraftZero is going next and what's in the way. **Read this before starting any work
session, and update it before ending one** — mark items done, record decisions in the log at
the bottom, and move anything you learned into the right phase. It's the single source of
truth for status. Chat history isn't.

*Last updated: 2026-09-25*

## Where things stand

Experiment #1 (run `2026-09-22_02-46-01`) is finished. See the
[report](docs/fdn-generalist-report.md).

- 2,507 games over 34 generations on one RunPod L40S, about $28 of pod time.
- The gen-33 checkpoint beat raw search **110/197 (55.8%, CI 49–63%)** but only tied gen 10
  at **96/196 (49.0%, CI 42–56%)**. Training plateaued around gen 10.
- Card valuations barely track 17lands: rank correlation **0.28** on commons (gens 10+).
  Premium removal underperforms: Stab 46%, Burst Lightning 46%, Refute 45%, against 58% for
  each on 17lands.
- Everything is saved in the private HF repo `danbrooks/draftzero-checkpoints`: 35
  checkpoints, 269 replay shards, all games and logs (report §5).

The next goal: **a player strong enough that its gameplay statistics can be trusted**, so
removal and control decks get played correctly and card-level stats mean something.

## Will's review of experiment #1 (2026-09-25)

Will named three main issues. The rest of this file is organized around them.

1. **A throughput bottleneck from the JVM layout.** Exp #1 ran **one JVM with 18 threads**.
   His advice: run **several JVMs in parallel** (one per opponent, as MageZero's runner does),
   with **4 threads, ZGC and a 48 GB heap per JVM** if RAM allows, and **don't cut search budget
   to buy throughput**.
   - His reference point: a 32-core, 128 GB machine plays **~1,600 games/day (~1/min) at a search
     budget of 1,000**. Exp #1 played ~1,500–2,200 games/day (64–90 games/hr) on ~24 cores, but at
     a budget of **96**. That's roughly 10× less search per game at a similar game rate, which
     points to a lot of headroom. His runs are single-deck and his hardware differs, so treat
     this as a direction, not a promise.
   - MageZero already supports this. Parallel JVMs arrived in `59ce546` (`max_jvms`), which *is*
     in exp #1's engine (`bcc76de`), but draft-zero's loop calls `launch_jvm` itself, one at a
     time, and never used it.
2. **λ too low.** This is the second time he's said it (after 2026-09-23). See D1.
3. **Too many games against itself.** Already planned: the 20 / 70 / 10 mix in Phase 4.

Two more notes:
- **Drop minimax as a baseline.** Offline MCTS with the heuristic value function beats XMage's
  minimax bot almost always, and a trained model beats both. Exp #1's evals already used only
  offline MCTS; the defaults and smaller configs still include minimax.
- **Expect trial and error.** "There are so many parameters. Deep RL is hard." That's an
  argument for making each iteration cheap and fast, which is one more reason to fix throughput
  first.

**What this puts in doubt.** The report concluded that throughput was "capped by inference, not
CPU" (§4.1). Every run behind that conclusion used a single JVM, including the 4× RTX 5090 box
with 96 threads in one JVM, so the cap may have been the JVM layout rather than inference. The
same caveat applies to every throughput number in this file and the report until Phase 3
re-measures them.

## Plan

```
Phase 1  Close exp #1 ─────────┐
Phase 2  v0.2.0 + repo lines ──┴──► Phase 3  Pilots ──► Phase 4  Exp #2 (Will reviews first)

Anytime: compute funding · perf investigation · study
```

Phases 1 and 2 can run in parallel. Phase 3 needs Phase 2. Phase 4 needs Phase 3 and the
open decisions below.

---

### Phase 1 — Close experiment #1

**Done when:** the code and report are committed and tagged, a public release exists, and the
conclusions are posted on Discord.

- [x] 17lands terms: the public datasets are **CC BY 4.0**, so the decks can be redistributed
      with credit. Checked 2026-09-25. Credit is in the README, the report and the release.
- [x] Rescue the deck builder. `extract_decks.py` existed only in the untracked
      `MageZero-Experiments/deckgen/`. It now lives at `tools/extract_decks.py`, and it
      regenerates all 31,516 `.dck` files and `assets/decks.tsv` byte for byte.
- [x] Sample decks for a fresh clone: `assets/sample/` (80 decks, including exp #1's 40 eval
      decks) and `configs/sample.yml`. The config loads and every deck resolves. A full game
      run from a fresh clone hasn't been tried.
- [x] Make the report, README and roadmap self-consistent and cross-linked. Stale README
      throughput numbers are fixed, and Will's λ feedback is added to the report as a dated
      addendum.
- [x] Kaggle: abandoned. Summarized in report Appendix A: 319 games, 17.5 games/hr, and
      6-game evals too small to show anything. Session 3 of each run stays unpulled.
- [x] WillWroble/MageZero#3: the correction was posted on 2026-09-21.
- [x] Commit the report (it had never been committed), merge `final-eval` into `main`, and tag
      `exp1-fdn-generalist`. Done 2026-09-25.
- [x] Publish [`danbrooks/draftzero-fdn-exp1`](https://huggingface.co/danbrooks/draftzero-fdn-exp1)
      on HF: public, CC BY 4.0. Published 2026-09-25 (commit `069f76c`). All 17 files are
      verified against the staged copy, with sha256 checks on the checkpoints and deck pool,
      and it's reachable without a token.
- [x] ~~Check whether v0.1 checkpoints load under MageZero v0.2.0~~: moved to Phase 2, where
      it's easiest to test. The release says it's untested.
- [ ] Post conclusions and the exp #2 direction on the MageZero Discord (promised in the thread).
- [x] Delete the `perf/feature-reset` branch (it was correct but measured no end-to-end gain).
      Deleted 2026-09-25. The next attempt starts from a clean branch.
- [x] Archive the pre-draft-zero workspace. It's renamed to
      `~/Desktop/Code/DEPRECATED_PREDRAFTZERO_MageZero-Experiments/`, and its `README_ARCHIVE.md`
      explains what it held. `_git_bundles/` keeps the 12 commits that exist on no remote,
      including the Kaggle harness, plus the XMage fork commit as a patch. Each was restored
      from scratch and checked. The clones were shallow, so the first bundles couldn't be
      restored; they were rebuilt after fetching full history.

### Phase 2 — MageZero v0.2.0 and repo boundaries

**Done when:** draft-zero is pinned to a v0.2.0-based engine, the fork fixes are upstream PRs,
draft-zero can run several JVMs in parallel, and a v0.2 smoke run passes.

- [ ] **Run several JVMs in parallel** (Will's issue #1). `loop.py` calls MageZero's `launch_jvm`
      one chunk at a time; make it run several concurrently, with 4 threads each and ZGC.
  - **Design question: inference servers.** Every exp #2 game is networked on both sides.
    MageZero's runner uses two fixed ports (`PRIMARY_PORT`, `OPPONENT_PORT`), and its own
    parallel path refuses networked opponents ("online mcts opponents need a server port each;
    not supported", `59ce546`). So either each JVM gets its own server and port, or one server
    takes several JVMs. Ask Will which he'd accept upstream before building it.
  - **RAM limits the JVM count.** At 48 GB of heap per JVM, exp #1's L40S pod (~116 GB usable)
    fits only 2. The Phase 3 pilot has to trade heap per JVM against JVM count.

- [ ] Rebase the fork onto `v0.2.0-alpha` (released 2026-09-22). Merge, don't copy files:
      copying from the pre-dense-vocab `fdn-generalist` branch silently reverts
      `require_encoding`, `initial_rows` and `map_csr`.
- [ ] Open upstream PRs for the seven fork commits (report §7.2). Lead with backward
      compatibility; Will won't accept changes that force retrains.
  - [ ] Two embedding bugs: a stale `num_embeddings` after the table grows, and trusting stale
        `embed_rows` metadata. These crash any run whose vocabulary grows.
  - [ ] Server threading: a configurable CPU thread count, and an HTTP pool sized to game threads.
  - [ ] Runner path resolution, a test import fix, and the metrics/runner module.
- [ ] Check whether experiment #1's v0.1 checkpoints load under v0.2.0 (moved from Phase 1).
      The HF release says it's untested; update its model card with the answer either way.
- [ ] Re-pin draft-zero to the v0.2.0-based engine. Start from a fresh clone of
      `danieljbrooks/MageZero`. The existing `~/Desktop/Code/mz-engine` checkout is a git
      worktree of the archived repo (see its `README_ARCHIVE.md`), so it shouldn't be used for
      new work.
- [ ] **Give the XMage fork source a real home.** draft-zero runs on "the fdn-generalist XMage
      build" (`deploy/bootstrap.sh`). That build's source is one commit on top of
      WillWroble/mage (`5a32441c`: set-wide action vocabulary, deck pools, per-game summaries),
      and it isn't on GitHub. Its only copy is the patch in the archive's `_git_bundles/`.
      Push it to a private `danieljbrooks/mage` fork, and rebase it along with the v0.2.0
      work above.

**Repo boundaries.** The rule is: a thing goes in MageZero only if it would make sense to
someone who has never heard of FDN draft. Machinery goes upstream; content gets published.

| Where | What |
|---|---|
| MageZero (upstream) | Deck-pool sampling for any format, eval harness, value-label histograms, report renderer |
| draft-zero (this repo) | FDN pools, 17lands stats, workers and pod layer. The worker layer is arguably generic, so it's a candidate for upstream later. |
| Published artifacts (HF) | Decks, checkpoints, reports. Link to them from MageZero; never commit them into it. |

- [ ] Ask Will whether he wants MageZero to link published baselines, for example from a
      "community baselines" section.

### Phase 3 — Pilots before the big run

The main reason to pilot is budget. Exp #1 cost about **$0.011 per game** ($28 / 2,507) at a
96-simulation search budget, and there's about **$61 left** of the $100 RunPod balance ($38.68
spent per report §4.4).

The earlier estimate for exp #2 was up to ~3× that per game at budget 300: roughly 1,700
games for $61, and ~$700 for the report's 20,000-game target. That estimate scaled exp #1's
*single-JVM* throughput. **Will's review suggests that layout was the bottleneck, so treat the
estimate as a pessimistic ceiling.** The JVM-layout pilot below replaces it with measurements.

Each pilot costs a few dollars. **Every pilot gets an external hard time limit.**

- [ ] **v0.2 smoke test.** Does the pipeline run end to end?
- [ ] **JVM layout × search budget** on the pod type you'd actually rent. Run exp #1's layout
      (1 JVM × 18 threads) against N JVMs × 4 threads, with heap per JVM as RAM allows, at budget
      300 and, if affordable, 1,000.
  - Measure games/hr, sims/s, cost per game, and the inference server's request rate, to see
    whether inference becomes the cap once the JVM stops being one.
  - This sizes exp #2 and the funding ask. Don't lower the budget to hit a throughput target.
- [ ] **Dedup off.** Will measured it at ~25% of each inference request. Verify the speedup,
      and check that outputs are identical.
- [ ] **λ sweep**, 2–3 generations per arm at budget 96. Which λ gives roughly flat value-label
      histograms? This shows early shape, not long-run stability. Arms depend on decision D1.

Not worth piloting:
- **Eval size.** Exp #1 already answered it: a 40-game eval is about ±15 points (report §6.1).
  Use 200 games per opponent and run evals less often.

### Phase 4 — Experiment #2

**Done when:** it's run, reported with the same structure as exp #1, and published.

**What changes:** Will's recommendations, taken as one bundle. Bundling means we can't tell
which change helped. That's acceptable: the goal is a better player, not attribution.

| Setting | Exp #1 | Exp #2 |
|---|---|---|
| Engine | 0.1.0-based fork | v0.2.0 |
| JVM layout | 1 JVM × 18 threads, 48 GB heap, ZGC | Several JVMs × 4 threads, ZGC, heap as RAM allows (Will: 48 GB each) |
| Search budget | 96 | At least 300. Higher if the pilot allows (Will runs 1,000); never lower to buy throughput |
| Heuristic baseline | Offline MCTS (defaults still list minimax) | Offline MCTS only; remove minimax from the defaults and configs |
| λ (TD discount) | 0.70 from gen 3 (schedule) | Fixed, value from the pilot |
| Opponent mix (self / previous / gen 0) | 70 / 20 / 10 | 20 / 70 / 10 |
| Priors | Yes/no prior on, others off | All off |
| Inference dedup | On | Off |
| Value-label histograms | Not every generation (§6.2) | Every generation |
| Strength eval | 40 games every 5 gens, plus a 400-game final eval | 200 games per opponent, less often |

**Success criteria.** Fix these before launch; don't choose them after seeing results.

- [ ] Rank correlation with 17lands on commons beats **0.28** (the exp #1 figure for gens 10+).
- [ ] Premium removal closes its gap to 17lands: Stab, Burst Lightning and Refute were 45–46%
      against 58%.
- [ ] Strength against a yardstick that stays fixed across experiments. Raw search at budget
      300 is stronger than at 96, so a raw-search win rate at 300 isn't comparable to exp #1's
      55.8%. Keep a raw-search-at-96 opponent, or play exp #1's gen 33 directly if it loads
      under v0.2.
- [ ] Qualitative: Dan plays it, and it doesn't make exp #1's blunders (like chumping a 2/2
      with a 1/1).

**Gate:**
- [ ] Send the exp #2 design to Will before launching (committed to on Discord, 2026-09-23).

---

## Open decisions

| ID | Decision | Context | Status |
|---|---|---|---|
| D1 | **Which direction to move λ** | The report (§7.3, §8.2) reads 0.70 as not too high and proposes sweeping 0.6 / 0.7 / 0.8. Will's rule is that bimodal histograms mean λ is too high and roughly flat is the target, so he recommends fixing it at ~0.95. Exp #1's labels narrowed to a single hump (§6.2), which by his rule is the *too-low* side. The report's sweep can't test his hypothesis. **Will has now named "λ too low" as one of three main issues twice** (2026-09-23 and 2026-09-25). Suggested: decide on a fixed ~0.95, and use the pilot only to check histogram shape at 0.9 / 0.95. | Open, strongly indicated |
| D2 | **Scale vs. money** | Depends on the JVM-layout pilot. If parallel JVMs recover the headroom Will's numbers suggest, the remaining balance buys far more than the ~1,700-game ceiling estimate. Decide after the pilot, and use its number for any funding ask. | Open |
| D3 | **What "num_trees" refers to** | Not found in draft-zero, MageZero, or upstream v0.2. If it means search budget, it's already the throughput pilot. | Open |
| D4 | **Kaggle** | Sticking with RunPod. Kaggle results abandoned; summarized in report Appendix A. | Decided 2026-09-25 |
| D5 | **Public or private draft-zero** | **Public, MIT**, since 2026-09-25. Deck data stays CC BY 4.0. Because everything pushed is now public, any experiment that should stay private needs a separate private repo. | Decided 2026-09-25 |

## Side tracks

These can happen anytime, and none blocks the main plan.

**Compute funding.** The pitch is the exp #1 public release plus the exp #2 design. Ask for
CPU hours, not GPU hours: the GPU ran at 11–14% during network self-play while the CPU
saturated at 93%. Rank RunPod offers by vCPU per dollar, **and now by RAM too**: parallel JVMs
want a lot of heap (Will: 48 GB each), so RAM per core limits the JVM count.
- [ ] SaladCloud benchmark (suggested on Discord). It fits CPU-only, interruptible, retryable
      work: gen-0 games and raw-search evals. It doesn't fit network self-play, which needs a
      GPU for inference. The minimum is €5.

**Performance (WillWroble/MageZero#3).** A good task for a background agent, with a revised
brief. With the network on, inference is about half the time per simulation (41 vs 84 sims/s,
report §4.1). The earlier JFR profile ran without the network, so it only saw the XMage half.
It found copy machinery at 37% of samples, `getPlayable` / `canActivate` / mana options at 15%,
and `stateRefresh` at 0.0%.
- [ ] Profile v0.2 **with the network on**. Split XMage time from inference time *before*
      changing any code. Do it **after** the parallel-JVM change: every measurement so far,
      including the local profile's ~1.9 s ZGC allocation stalls, came from a single JVM, and
      the layout change may move the bottleneck entirely.

**Study.** Read with a specific question in mind.
- [ ] KataGo paper (Wu, 2019): AlphaZero on a small compute budget. Its "playout cap
      randomization" (full search on a random fraction of moves, cheap search on the rest)
      bears directly on the 96-vs-300 tradeoff.
- [ ] Sutton & Barto, chapter 12 (eligibility traces): what λ does to the value targets. Relevant
      to D1.
- [ ] AlphaZero paper: the baseline the rest builds on.

## Working agreements

These apply to people and to Claude sessions, and each rule comes from an actual mistake.

- **One session per workstream.** Read this file first, and update it last.
- **Profile before optimizing.** A synthetic benchmark once put `stateRefresh` at ~32% of a
  simulation. A real profile showed 0.0%, and the 27× speedup changed nothing end to end.
- **Check how MageZero does it before building infrastructure.** Exp #1 ran one JVM with 18
  threads, which Will called "a very LLM thing to do", while MageZero already parallelized
  across JVMs (`max_jvms`). Read the upstream runner and ask Will before inventing a layout.
- **Every long job gets an external hard kill.** A config cap isn't enough: a "8-minute" run
  with `games: 500` ran for 5h43m because the game count bound before the time cap did.
- **Verify that the binary matches the commit.** Run `package`, not just `compile`, then check
  the build timestamp.
- **Win rates always come with n** and a confidence interval where it matters.
- **RunPod:** spend only the existing balance. Never add credits.
- **Upstream changes stay backward compatible.** A change that forces a retrain needs a
  migration story or it won't be merged.
- **Give background agents bounded tasks with a clear check**, for example "smoke test
  passes". Keep judgment calls interactive: exp #2 design, λ, what to publish, and anything
  that goes to Will.

## Decision log

Add dated entries, newest first.

- **2026-09-25** — Will's review of exp #1 added. His three main issues: the single-JVM layout,
  λ too low, and too much self-play. Parallel JVMs added to Phase 2, the Phase 3 budget estimate
  downgraded to a ceiling, minimax dropped as a baseline, and D1 strengthened. It also puts the
  report's "capped by inference" conclusion (§4.1) in doubt, because every run was a single JVM.
- **2026-09-25** — draft-zero made public under MIT (D5 revised). The `exp1-fdn-generalist` tag
  predates `LICENSE`. It was left in place rather than moved, because Will has repo access
  and may already have fetched it, and the README states that MIT covers every version.
- **2026-09-25** — Experiment #1 release published on HF: `danbrooks/draftzero-fdn-exp1`.
  The Discord wrap-up post can go out now that its link resolves.
- **2026-09-25** — draft-zero stays private for now (D5). HF release approved: public,
  CC BY 4.0. `final-eval` merged into `main` and tagged `exp1-fdn-generalist`.
  `perf/feature-reset` deleted. `MageZero-Experiments` archived as
  `DEPRECATED_PREDRAFTZERO_MageZero-Experiments`. Found that the XMage fork source exists
  only as a patch in that archive; it's tracked in Phase 2.
- **2026-09-25** — Phase 1 work: the deck builder was rescued and reproduces the pool byte for
  byte, the sample decks were added, and the docs were cross-linked. 17lands data confirmed
  CC BY 4.0. Kaggle abandoned. The HF release is staged and waits on a write token and D5.
- **2026-09-25** — Roadmap created. Staying on RunPod (D4). Exp #2 adopts Will's recommendation
  bundle (Phase 4), pending D1–D3.
