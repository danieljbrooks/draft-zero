# Roadmap

Where DraftZero is going next and what's in the way. **Read this before starting any work
session, and update it before ending one** — mark items done, record decisions in the log at
the bottom, and move anything you learned into the right phase. It's the single source of
truth for status. Chat history isn't.

*Last updated: 2026-09-27*

## Where things stand

Experiment #1 (run `2026-09-22_02-46-01`) is finished. See the
[report](docs/003-fdn-generalist-report.md).

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

## Blocked on Will (2026-09-25)

| ID | Waiting for | Blocks | Doesn't block |
|---|---|---|---|
| ~~**B1**~~ | **Resolved 2026-09-26:** Will pushed it as `cb7e9c6f` ("v0.2.0"), and built from it the release's three MageZero jars match byte for byte (see the decision log). Was: **the Java source behind the v0.2 bundle**, pushed to `WillWroble/mage` (asked on [#7](https://github.com/WillWroble/MageZero/issues/7)). The newest public commit is `2f35d9f7` (Sep 16), where `Features.TABLE_SIZE = 2_000_000`; the v0.2 jars have `2147483647`. | **Exp #2's engine.** Our training-only XMage additions (vocab, deck pools, per-game summaries) have to be rebuilt on v0.2's Java, so Phase 3 pilots and Phase 4 wait on this. | The Python migration, the vocab prototype on `2f35d9f7`, the XMage fork for exp #1, and the design docs |
| **B2** | **Will's answer on #7** (action vocab) | Upstream PRs, and our agents running through the normal `mz import` flow | Implementing it in our forks |
| **B3** | **Will's preference for inference servers with parallel JVMs** (one per JVM, or shared). Not asked yet; the parallel-JVMs design doc drafts the question. | An upstream-compatible parallel-JVM implementation | The design itself |

**Can run now, in parallel, without touching the same files:**
- **A. Action vocab in our forks:** Python off v0.2, Java off `cb7e9c6f` (v0.2), plus moving `VocabDump`
  into draft-zero.
- **B. XMage fork on GitHub:** makes exp #1 playable. Start it first, because A's Java half needs
  this fork.
- **C. Python migration** of draft-zero onto Will's v0.2.
- **D. Design docs:** parallel JVMs (including the B3 question) and exp #2.

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
- [x] Post conclusions and the exp #2 direction on the MageZero Discord (promised in the thread).
      Dan posted them, 2026-09-26.
- [ ] HF model card: replace "whether the checkpoints load under v0.2.0 is untested" with
      "v0.1-only". Will's v0.2 release notes say v0.1 models and data are no longer compatible.
- [x] Delete the `perf/feature-reset` branch (it was correct but measured no end-to-end gain).
      Deleted 2026-09-25. The next attempt starts from a clean branch.
- [x] Archive the pre-draft-zero workspace. It's renamed to
      `~/Desktop/Code/DEPRECATED_PREDRAFTZERO_MageZero-Experiments/`, and its `README_ARCHIVE.md`
      explains what it held. `_git_bundles/` keeps the 12 commits that exist on no remote,
      including the Kaggle harness, plus the XMage fork commit as a patch. Each was restored
      from scratch and checked. The clones were shallow, so the first bundles couldn't be
      restored; they were rebuilt after fetching full history.

### Phase 2 — Migrate to Will's v0.2 engine

**The principle (agreed 2026-09-25):** Will's engine stays clean.
- **MageZero and his XMage fork** get only generic, opt-in mechanisms he agrees to, as proposed
  in [#7](https://github.com/WillWroble/MageZero/issues/7).
- **draft-zero** keeps everything limited-specific: 17lands stats, the deck pools built from
  17lands, metrics and dashboards, and extra logging.
- **Our agents** ship as artifacts that plug into his ecosystem: `.mz` bundles with the action
  vocab inside the checkpoint, plus curated `.dck` decks, the way his 16-deck Standard pool ships
  in `xmage/decks/`.

**Done when:**
- draft-zero runs on Will's v0.2, with no MageZero fork in between
- the action vocab exists as PR-shaped branches in our forks
- draft-zero can run several JVMs in parallel
- a v0.2 smoke run passes

**What v0.2.0 changes** ([release notes](https://github.com/WillWroble/MageZero/releases/tag/v0.2.0-alpha)):
- It **ships its own XMage bundle**, built from Java source Will pushed on 2026-09-26 as `WillWroble/mage` `cb7e9c6f`.
- It **widens the feature hash from 2M to 2^31 bins**, and **upgrades the state encoder**: fewer
  redundant features, dynamic subtypes, and per-turn watchers.
- **"No longer compatible with models and data generated with v0.1.0"**, so exp #1's checkpoints
  can't be v0.2 opponents, and exp #2 starts from gen 0.
- It fixes parallel JVM launches colliding on the H2 card DB.
- It needs a MageZero runner from 2026-09-20 or later.
- New `game.yml` knobs: `prune_duplicate_states` (on by default), `backprop_discount` (default
  0.99), `max_minutes` and `prior_bonus`.

**Bytecode comparison (2026-09-25).** 39 classes compiled from the public source (`2f35d9f7`)
were compared with the v0.2 jars: 34 are identical. `Features` and `StateEncoder` differ where
the release notes say. For the vocab change:
- **Identical in v0.2, so the change applies cleanly:** `ActionEncoder`, `ComputerPlayerMCTS2`,
  `Config$PlayerConfig`.
- **Changed in v0.2, where the rebase will conflict:** `LabeledStateWriter` and
  `ParallelDataGenerator`.
- Each player already gets its own `ActionEncoder`, so a per-player vocab is plumbing, not a
  redesign.

- [x] ~~Check whether exp #1's checkpoints load under v0.2~~: answered by the release notes.
      They're incompatible.
- [x] **Python: pin Will's v0.2** (2026-09-26, branch `v02-migration`). draft-zero pins
      `danieljbrooks/MageZero` `draftzero`: Will's `main` (`11a5974`) plus three opt-in commits,
      namely the policy width (#7), a CPU fallback for train/test, and `MZ_SERVER_THREADS`.
      `draftzero/engine.py` runs his server/train/test scripts unmodified and does the rest.
      A 3-generation laptop smoke test passes end to end (`configs/smoke_v02.yml`).
      The original plan:
  - Move the fork-only modules (`metrics`, `report`, `resources`, `refresh_dashboard`) into
    draft-zero.
  - Check every MageZero call against v0.2. The names all exist upstream, but the fork rewrote
    ~480 lines of `runner.py`, so arguments and behavior need checking too.
  - Pin a runner commit from 2026-09-20 or later.
  - The four small fork fixes (two embedding-size bugs in the dense-vocab code, a test import,
    helper-script paths): re-test them on v0.2, and offer Will any that still reproduce as small
    fixes. Drop the two server-tuning commits.
  - `danieljbrooks/MageZero` `mz-engine` stays as exp #1's pinned engine. No new work there.
- [ ] **Action vocab in our forks, PR-shaped** (#7, D6).
  - Python: `danieljbrooks/MageZero`, branch `action-vocab`, off Will's v0.2 `main`. **Width
    done (2026-09-26):** `MZ_ACTION_VOCAB` sets the head width, and a width mismatch is refused
    with a clear error. Still open: storing the vocab in the checkpoint, and the checks below.
  - Java: `danieljbrooks/mage`, branch `action-vocab`, off v0.2's Java (`cb7e9c6f`). Exp #1's
    vocab commit already rebases onto it (the `v0.2-generalist` branch below).
  - A configurable width (default 128), plus an optional exact vocab stored in the checkpoint and
    set per player.
  - **Done when:** with no vocab, seeded games match upstream exactly; a vocab agent plays a
    128-slot agent in the same game; and the vocab survives `mz export` / `mz import`.
  - No PRs to Will until he answers #7.
- [x] **Move `VocabDump` into draft-zero** (2026-09-26): `tools/vocab/`, with
      `build_vocab.sh`. Rebuilt from the v0.2 bundle's jars, the FDN vocab is byte-identical to
      exp #1's `FDN_SPG.tsv`: no ability text changed.
- [x] **XMage fork on GitHub** (2026-09-26): [`danieljbrooks/mage`](https://github.com/danieljbrooks/mage),
      public, a fork of `WillWroble/mage`. It is for testing initially: we use Will's standard
      releases wherever possible, and its `DRAFTZERO.md` says so.
  - `exp1-fdn-generalist` is exp #1's exact engine (`5a32441c` on `2f35d9f7`). That makes exp
    #1's checkpoints playable, and ends the patch in the archive being the only copy.
  - `v0.2-generalist` is the same changes on v0.2.0, plus `build-generalist-bundle.sh`.
  - The built v0.2 bundle is in the private HF repo `danbrooks/draftzero-checkpoints`, as
    `xmage/generalist-xmage-v0.2-48e49184.tar.gz`, next to the pilot's v0.1 bundle.
  - [x] The HF model card links `exp1-fdn-generalist` (2026-09-26).
- [x] **Run several JVMs in parallel** (Will's issue #1), 2026-09-26: `jvm.jvms` JVMs at once,
      sharing one inference server per checkpoint, each server's HTTP pool sized to the game
      threads that query it (v0.2's fixed pool of 6 would cap every batch at 6). The eval runs
      on every JVM slot too. The original notes:
  - **Design question: inference servers.** Every exp #2 game is networked on both sides.
    MageZero's runner uses two fixed ports (`PRIMARY_PORT`, `OPPONENT_PORT`), and its own
    parallel path refuses networked opponents ("online mcts opponents need a server port each;
    not supported", `59ce546`). So either each JVM gets its own server and port, or one server
    takes several JVMs. Ask Will which he'd accept upstream before building it.
  - **RAM limits the JVM count.** At 48 GB of heap per JVM, exp #1's L40S pod (~116 GB usable)
    fits only 2. The Phase 3 pilot has to trade heap per JVM against JVM count.
  - v0.2 fixes "parallel JVM launches colliding on the H2 card DB", so build this on v0.2.
- [x] **Training-only XMage additions on v0.2** (2026-09-26): the vocab, deck pools and the
      per-game summary line.
  - Exp #1's commit is rebased onto `cb7e9c6f` as branch `v0.2-generalist` (`14c9228d`). It
    is local only for now, in `~/Desktop/Code/mage`, until the XMage fork above is on GitHub.
  - There was one conflict, in `ParallelDataGenerator`, where v0.2 added commander mode; both
    changes were kept.
  - `build-generalist-bundle.sh` (on that branch) rebuilds the three changed jars into Will's
    release bundle, and leaves every other file as Will shipped it.
  - An offline smoke test passed: 4 of 4 games, pool decks, `GAME_SUMMARY` lines, a
    1,024-wide policy with the FDN vocab, and v0.2's 2^31 feature hashes.
  - **Still needed before exp #2 runs on it:** MageZero's v0.2 Python has to train and serve a
    1,024-wide head (the Python half of the action-vocab item). The FDN vocab should also be
    re-checked against v0.2's ability text (the `VocabDump` item).

**Repo boundaries.**

| Where | What |
|---|---|
| Will's MageZero and XMage fork | Only generic, opt-in mechanisms he agrees to: the configurable action width and optional action vocab (#7), maybe deck pools. Plus bug fixes to code already there. |
| draft-zero (this repo) | The FDN pools and deck builder, `VocabDump` and the FDN vocab, 17lands stats, metrics, reports and resource sampling, orchestration, workers, and the per-game stats hook |
| Published artifacts | `.mz` bundles (vocab inside the checkpoint) plus curated `.dck` decks for his ecosystem; full HF releases for research |

- [ ] Ask Will whether MageZero should link published baselines, and whether FDN decks could
      ship the way his Standard pool does in `xmage/decks/`. Both belong in the #7 conversation.

### Phase 3 — Pilots before the big run

The main reason to pilot is budget. Exp #1 cost about **$0.011 per game** ($28 / 2,507) at a
96-simulation search budget, and there's about **$61 left** of the $100 RunPod balance ($38.68
spent per report §4.4).

The earlier estimate for exp #2 was up to ~3× that per game at budget 300: roughly 1,700
games for $61, and ~$700 for the report's 20,000-game target. That estimate scaled exp #1's
*single-JVM* throughput. **Will's review suggests that layout was the bottleneck, so treat the
estimate as a pessimistic ceiling.** The JVM-layout pilot below replaces it with measurements.

Each pilot costs a few dollars. **Every pilot gets an external hard time limit.**

- [x] **v0.2 smoke test** (2026-09-26, [docs/010](docs/010-v02-migration-pilot.md)). It runs end to
      end, on a laptop and on a 13.6-core pod:
  - 78 games over 3 generations, 0 failed games, 0 search timeouts in 8,556 searches.
  - Network self-play ran at 59 games/hr on 3 × 4 threads, CPU-bound, about $0.012 per game.
  - A sized server pool beat v0.2's fixed 6 threads by 16%. Training needed batch 128: 512 ran
    a 32 GB GPU out of memory.
  - Follow-ups:
    - [ ] Re-measure 7 × 4 with a sized shared server on a 31-core pod before sizing exp #2.
    - [ ] Smaller jobs per generation, to cut the straggler tail (CPU averaged 70–78% of the
          quota per generation, against 98% in steady state).
    - [ ] Self-play value labels sat closer to 0 under v0.2 than v0.1 at λ = 0.95 (median |v|
          0.10–0.37 vs 0.41–0.45). Confirm on a larger run before fixing λ.
- [x] **JVM layout × search budget** (2026-09-26, on v0.1, budget 300 only; see
      [docs/006](docs/006-exp2-pilot.md)). 7 JVMs × 4 threads beat 1 × 28 by 3.3× offline, and by 1.9× with
      one shared inference server. Re-measure on v0.2 before sizing exp #2.
      The original plan was: **JVM layout × search budget** on the pod type you'd actually rent. Run exp #1's layout
      (1 JVM × 18 threads) against N JVMs × 4 threads, with heap per JVM as RAM allows, at budget
      300 and, if affordable, 1,000.
  - Measure games/hr, sims/s, cost per game, and the inference server's request rate, to see
    whether inference becomes the cap once the JVM stops being one.
  - This sizes exp #2 and the funding ask. Don't lower the budget to hit a throughput target.
- [x] **Hide the opponent's hand from the network** (2026-09-26). `configs/game.yml` said
      `hidden_info:`, but the Java reads `hiddenInfo`, so exp #1 (and the v0.2 pilot) encoded both hands
      in every state (docs/009). The key is fixed, set to false, and set explicitly by the loop
      (`hidden_info` in its config). Checked in the JVM: the other player's hand now encodes as a
      card count (`CardsInHand`), which never appears with the switch on.
- [ ] **Measure what the search's leak is worth** (docs/009 §8.3). A throwaway flag that re-deals the
      opponent's hidden cards before each search (`shuffleUnknowns` in `createMCTSGame`) and turns off
      tree reuse. Then play the clairvoyant raw search against it, head to head: CPU only, a few hours.
      Measurement only: it ignores known cards and uses the global RNG, so don't train with it.
      This sizes the fix planned for exp #3.
- [x] **Dedup off** (checked 2026-09-26). Will measured the server's feature dedup at ~25% of each
      request. In v0.2 the dedup (`FeatureVocab._dedupe`) first checks whether each bag is already
      sorted and unique, which XMage guarantees, and skips the work when it is. Nothing to turn
      off.
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

**Goals** (Dan, 2026-09-26), in order:
1. **A high win rate against the baseline:** raw search at the same budget, 200-game evals every
   8 generations.
2. **High throughput:** games per hour and cost per game, logged every generation.
3. **A good rank correlation with 17lands on commons:** above exp #1's 0.28.

**Runs and budget:** one or two runs, **hard-capped at $29 per pod** (`deploy/exp2.sh`: the
watchdog stops at budget − 0.75 h, pushes the weights to HF and removes the pod, and a
self-destruct fires at the budget whatever happens). Run 1 uses Will's settings
(`configs/exp2.yml`). Run 2, if affordable, uses the same settings from a network pretrained
on human decisions, so the only difference is the starting point (imitation, D8).

**Run 1** (2026-09-27): `dz-exp2-run1`, RunPod pod `206haai9na81zy`, a Secure RTX 3090 (31.1 cores,
116 GB, 24 GB GPU) at $0.50/hr. It runs `deploy/exp2.sh` with a $28.90 cap: a graceful stop at 57.05 h
and hard removal at 57.8 h (about 2026-09-29 11:50 UTC). The layout is 7 × 4 threads, 112 games per
generation, training batch 64. Weights go to HF `danbrooks/draftzero-checkpoints` under
`2026-09-27_01-59-54/`. The session "Experiment 2 and performance - RUN 1" launched it and
owns it. **Other sessions: don't touch this pod or that HF prefix.**

**Run 2 handoff (imitation A/B).** It needs its own session, working in its own git worktree and
branch, never the shared `~/Desktop/Code/draft-zero` checkout.
- **Question:** at equal cost, does starting self-play from a network pretrained on human
  decisions do better on the three goals above than run 1's heuristic-search start?
- **Same as run 1 except the starting network:** same pod type (Secure RTX 3090, 31 cores,
  $0.50/hr; take one with `vcpuCount` ≥ 31, and remove a smaller one at once), same
  `configs/exp2.yml` through `deploy/exp2.sh`, same settings.
- **Budget:** whatever is left after run 1's $28.90, minus about a $1 margin. Check
  `runpodctl user` before creating anything. Never add credits.
- **Steps:**
  1. Rebuild the human-decision tables on v0.2 with the imitation study (docs/008;
     `python -m draftzero.gameplay.imitation splits / build / tables`), building the mzbridge
     against the v0.2 bundle, with the opponent's hand hidden.
  2. Pretrain a network on them with the study's from-scratch path (`new_net`,
     `load_checkpoint`; the policy fit to the human action sets, attack decisions, value from
     game results), saved in MageZero's checkpoint format (`model_state_dict` +
     `feature_vocab`), so `train.py --checkpoint` continues from it. The pod GPU is fine for
     this: minutes, inside the run's budget.
  3. Add an `init_checkpoint` option to `draftzero.loop`. Gen 0 then plays network self-play
     with that network instead of heuristic search, and it becomes `gen0.pt.gz` (the "gen 0"
     opponent in the mix). Smoke-test on the laptop (`configs/smoke_v02.yml`) first.
  4. Before launching, check that the pretrained network matches held-out human decisions at
     about docs/008's level (73% top-1), and that a few games look legal and sensible.
  5. Log the human-agreement rate every generation, to see whether self-play keeps the prior
     or washes it out.
- **Setup notes:**
  - The pod credential is the fine-grained HF token `draftzero-pod-write` in the laptop's HF
    token store (write access to `danbrooks/draftzero-checkpoints` only). Never print it.
  - It goes to the pod as `/root/.dz_env` (`HF_TOKEN=`, `HF_REPO=danbrooks/draftzero-checkpoints`)
    and `/root/.hf_token`, mode 600, written through the SSH proxy with a leading-space `echo <base64> | base64 -d` line.
  - Then `deploy/pilot_v02_setup.sh` and `deploy/exp2.sh`. Pass `DZ_COST_PER_HR` and `DZ_BUDGET_USD`.
  - Read `docs/005-runpod-tips.md` before reserving.
- **Done when:** run 2 is launched with the pretrained start, and step 4's checks are recorded
  in a doc. After both runs end, compare them at equal spend.

**Secondary checks**, carried over:

- [ ] Rank correlation with 17lands on commons beats **0.28** (the exp #1 figure for gens 10+).
- [ ] Premium removal closes its gap to 17lands: Stab, Burst Lightning and Refute were 45–46%
      against 58%.
- [ ] Strength against a yardstick that stays fixed across experiments. Raw search at budget
      300 is stronger than at 96, so a raw-search win rate at 300 isn't comparable to exp #1's
      55.8%. Keep a raw-search-at-96 opponent. Exp #1's gen 33 is incompatible with v0.2
      (per Will's release notes), so it can't be played directly. Not in `configs/exp2.yml` yet:
      the eval runs its baseline at the agent's budget, so this needs a per-baseline budget.
- [ ] Qualitative: Dan plays it, and it doesn't make exp #1's blunders (like chumping a 2/2
      with a 1/1).

**Gate:**
- [x] ~~Send the exp #2 design to Will before launching~~. Will has given a lot of feedback
      already; Dan communicates the parameters when the run starts (2026-09-26).

---

## Open decisions

| ID | Decision | Context | Status |
|---|---|---|---|
| D1 | **Which direction to move λ** | The report (§7.3, §8.2) reads 0.70 as not too high and proposes sweeping 0.6 / 0.7 / 0.8. Will's rule is that bimodal histograms mean λ is too high and roughly flat is the target, so he recommends fixing it at ~0.95. Exp #1's labels narrowed to a single hump (§6.2), which by his rule is the *too-low* side. The report's sweep can't test his hypothesis. **Will has now named "λ too low" as one of three main issues twice** (2026-09-23 and 2026-09-25). Suggested: decide on a fixed ~0.95, and use the pilot only to check histogram shape at 0.9 / 0.95. | Open, strongly indicated |
| D2 | **Scale vs. money** | Depends on the JVM-layout pilot. If parallel JVMs recover the headroom Will's numbers suggest, the remaining balance buys far more than the ~1,700-game ceiling estimate. Decide after the pilot, and use its number for any funding ask. | Open |
| D3 | **What "num_trees" refers to** | Not found in draft-zero, MageZero, or upstream v0.2. If it means search budget, it's already the throughput pilot. | Open |
| D4 | **Kaggle** | Sticking with RunPod. Kaggle results abandoned; summarized in report Appendix A. | Decided 2026-09-25 |
| D6 | **Will's answer on the action vocabulary** ([WillWroble/MageZero#7](https://github.com/WillWroble/MageZero/issues/7)) | This decides whether our FDN agents can run in the normal MageZero flow (`mz import`, add a deck, play). The proposal is two opt-in pieces: a configurable policy width (default 128), and an optional exact action vocabulary stored inside the checkpoint and set per player. MageZero gets only that generic mechanism. The FDN vocab, the `VocabDump` builder, deck pools, 17lands stats and logging stay in draft-zero and ship with our models. If Will says no, our agents stay runnable only through draft-zero. Implementation proceeds in our forks, PR-shaped, which is useful either way. No PRs to Will until he answers. | Waiting on Will (posted 2026-09-25) |
| D7 | **Does exp #2's network see the opponent's hand?** | Exp #1's did: `configs/game.yml` writes `hidden_info:` but the fork reads `hiddenInfo` (`Config.java:79`), so the key is ignored and the default (true) wins. Human data (17lands, Arena) never has the opponent's hand, and coaching must not assume it, so both gameplay-data goals want a hidden-information network ([docs/008](docs/008-gameplay-data.md) §6, §11). Fix the key either way. Fixing the key is not enough: the search itself is clairvoyant, and it flips a decision on a card the player can't see even with the network's input masked. v0.2.0 changes neither. Staged plan: [docs/009](docs/009-hidden-information.md) §5.4. Fair baselines and the order of work, where the first three steps need no GPU: §8.5. | **Decided 2026-09-26:** the key is fixed and the network doesn't see the opponent's hand. The search's leak is accepted for exp #2, and exp #3 should fix it (see Phase 3 and the decision log). |
| D8 | **A human-prior arm in exp #2?** | Gen 33's priority head ranks a human action first 44% of the time (chance on 4+ options); heads retrained on 30k human decisions reach 73%, and its attack (binary) head, the one prior exp #1 used, loses to a power/toughness rule (docs/008 §7.3). A human prior only matters with the prior switched on; the A/B protocol is in docs/008 §7.5 (≥ 400 games per arm). | Open |
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

**Human gameplay data** ([docs/008](docs/008-gameplay-data.md), branch `gameplay-data`). 17lands
replays and Arena logs now map onto XMage states; the next steps are:
- [ ] Build the full human corpus on a pod: turn-start decisions (~6.7M, ~11 h at 2 JVMs) plus
      replayed-turn decisions, stored as specs + labels so it survives v0.2's encoder.
- [ ] Pretrain the binary and value heads on it; run the D8 A/B.
- [ ] Coaching validity on a GPU with a better value net: ≥ 2,000 decisions (~$3–5 on a 3090).
- [ ] Ask 17lands for research access to raw game histories.
- [ ] Dan: keep FDN Arena logs with Detailed Logs on (copy `Player.log` after each session).
- [ ] Ask Will whether v0.2 fixes: the `hiddenInfo` key, seeding (the MCTS2 constructor reseeds a
      constant), the timeout label (`!playerAWon`), duplicate root children, the attack-defender
      HashSet, and determinization in search (`shuffleUnknowns` is never called).
- [ ] Add the behavioural fingerprint (`dz gameplay fingerprints`) to the loop as a per-generation
      monitor: the exp #2 pilot's gen 1 missed land drops with a land in hand on 16–25% of turns 1–3.

**Limited bots.** Research how we might get some Limited bots up. Not started: this is a
placeholder until Dan picks it up.
- Prompted by Jack Maiorino's [Spellbench](https://jackmaiorino.github.io/spellbench/)
  ([repo](https://github.com/jackmaiorino/spellbench)): one protocol for MTG bots on any rules
  engine, plus a public Elo leaderboard. XMage support is planned next, for FDN Limited and the
  Standard pool, with DraftZero and MageZero named as candidates.
- Worth knowing going in: Spellbench's draft protocol v2
  ([spec](https://github.com/jackmaiorino/spellbench/blob/main/spec/SPELLBENCH_PROTOCOL_V2.md))
  is a fair-play contract. A bot sees only its own seat's view and can't search a copy of the
  real game. Our search does exactly that today (D7, docs/009), so a v2 entry probably waits on
  exp #3's hidden-information work.
- [ ] Research: what "up" should mean (a leaderboard entry, something people can play against,
      or both), what we'd ship, and what's in the way. Write up the options before building
      anything.

**Search benchmark** ([docs/012](docs/012-search-benchmark.md), branch `search-benchmark`). A
proposal out for comment (2026-09-27), nothing built or run. Its first experiment (E0–E2):
- **Methods:** today's clairvoyant MCTS, PIMC with 1 and with 4 worlds, and particle IS-MCTS on the
  same 4 worlds. IS-MCTS was added after a reviewer who uses it for Magic called it the big one.
- **Budgets:** 100 to 3,000 simulations.
- **Evaluators:** offline search, and a trained exp #2 network.
- **Measures:** agreement with top 17lands players on 1,000 decisions, and a pass/fail leak test
  that colors the plot.

It costs ~23 pod-hours (~$11.50), with the offline runs on the laptop. The follow-ups (E3–E9, first
among them validation by play) are ideas for now.
- [ ] Collect reviewers' comments on the first experiment (its §2.11) and revise.
- [ ] Build particle IS-MCTS in the fork behind a flag (docs/009 §6.4), about one to two weeks:
      the first experiment's long pole.
- [ ] Decide funding for the network runs. The RunPod balance is committed to exp #2.
- [ ] Fix the search issues its §2.9 lists before the benchmark runs: whole-tree walks on every
      iteration, and the virtual loss's sign at opponent nodes.
- [ ] Dan: decide whether to raise the virtual-loss issue, and the race that can reset priors to
      uniform, with Will.

**Faster engines.** Explore Jack Maiorino's [mtg-kernel](https://github.com/jackmaiorino/mtg-kernel),
the engine behind Spellbench's first benchmark. It's reportedly about 40× faster than XMage
overall, and 1,000× for training (unverified). Not started: this is a backlog item.
- Why it matters: XMage's speed caps the search budget and the cost of every run. docs/012 records
  engine-independent counts so its results can be re-costed on a faster engine.
- [ ] Explore:
  - its card and rules coverage (FDN?);
  - what the 40× and 1,000× figures measure;
  - whether its search can be kept to one seat's view (Spellbench v2's contract);
  - what running MageZero's search and networks on it would take.

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

- **2026-09-26** — Hidden information: exp #2 accepts some leakage; exp #3 should do better. For
  exp #2 the network no longer sees the opponent's hand, but the search still runs on the real
  game, hidden cards included. Card statistics for counterspells and combat tricks carry that
  bias. Exp #3's direction is docs/009 §8: a world sampler with determinized search, a
  certification suite, and a fair gen-0 search. The head-to-head measurement above sizes it.
  The engine work goes on a `danieljbrooks/mage` branch `hidden-info`, stacked on
  `v0.2-generalist` with one commit per piece, so each can be offered upstream (it overlaps
  MageZero#4). The engine then reads: Will's v0.2.0, then `v0.2-generalist`, then `hidden-info`.

- **2026-09-26** — v0.2 pod pilot ([docs/010](docs/010-v02-migration-pilot.md)): the loop runs end to
  end on v0.2 at pod scale, and both watchdog fixes held on the pod. `v02-migration` stays off
  `main` for now: the gameplay-data work still runs on the v0.1 engine and exp #1's checkpoint.
- **2026-09-26** — Moved to Will's v0.2 engine (branch `v02-migration`). The v0.2 server's
  HTTP pool is fixed at 6 threads, which caps a shared server's batch at 6. That's a problem
  with several JVMs, so our MageZero branch makes it opt-in configurable. Exp #1's
  conclusions are posted on Discord, and Will has given feedback on the pilot numbers.

- **2026-09-26** — B1 resolved: Will pushed the v0.2 Java source (`WillWroble/mage` `cb7e9c6f`).
  - It is the release's source: built unmodified with javac, all 46 classes in the release's
    `mage-magezero`, `mage-player-ai` and `mage-player-ai-rl` jars match its bytecode.
  - Exp #1's XMage commit is rebased onto it as `v0.2-generalist`, and an offline smoke test
    passes.
  - Will's reply on #7's configurable width is pending: he wants to measure how much a
    1,024-logit head slows the network (B2).
- **2026-09-26** — Human gameplay data study ([docs/008](docs/008-gameplay-data.md), branch
  `gameplay-data`). Mapping works: 100% of 17lands card/token/ability ids map to XMage; turn-start
  states build 2,004/2,004; a scripted replay of a recorded turn reproduces the next snapshot in
  ~89% of turns; Arena logs pair 128/128 decisions with the player's answer. Imitation: human-trained
  heads on gen 33's trunk beat gen 33 and a heuristic on held-out human decisions (73% vs 44% / 65%);
  strength is untested (D8). Coaching works end to end but is not valid yet: no skill signal with
  either evaluator, which agree on the best action only 51% of the time. Opened D7 and D8.
- **2026-09-25** — Blocked on Will's v0.2 Java source (B1).
  - The v0.2 bundle's jars have the 2^31 hash, but the public `WillWroble/mage` source doesn't.
  - A bytecode comparison shows the vocab change's core classes are unchanged in v0.2, so it
    proceeds in our forks on `2f35d9f7`.
  - Phase 2 is rewritten around the agreed principle: Will's engine gets only generic, opt-in
    mechanisms (#7), and everything limited-specific stays in draft-zero. That drops the old
    plan to upstream all seven fork commits.
  - v0.1 checkpoints are confirmed incompatible with v0.2 (release notes).
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
