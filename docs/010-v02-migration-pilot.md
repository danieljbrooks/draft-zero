# Moving to Will's v0.2 engine, and its pilot

DraftZero now runs on Will's MageZero v0.2. This doc records:
- what changed, and why;
- a 3-generation pilot on a RunPod pod that checks it end to end.

This was done on 2026-09-26, on branch `v02-migration`, and follows the first exp #2 pilot on the v0.1 engine ([docs/006](006-exp2-pilot.md)).

## Summary

- **v0.2 runs end to end at pod scale.** The pilot played 78 games over 3 generations:
  - heuristic bootstrap;
  - network self-play;
  - self-play plus a frozen gen-0 opponent on its own server;
  - an eval.

  All 78 games completed, and none of 8,556 searches hit the 60 s timeout (longest: 12.8 s).
- **The engine is Will's, with small opt-in additions.** draft-zero launches his server, train and test scripts unmodified. Our MageZero branch adds four opt-in settings on top of his `main`; without them, everything behaves exactly as upstream:
  - the policy width (#7);
  - a CPU fallback for train and test;
  - the server's HTTP pool size;
  - the training batch size.
- **Two v0.2 defaults don't fit a format-wide agent, and both are now settings:**
  - *Training batch.* v0.2 trains in batches of 512 states. A format-wide state has about 1,700 features, and a batch of 512 ran a 32 GB GPU out of memory. The pilot trained at 128.
  - *Server HTTP pool.* v0.2's pool is fixed at 6 threads. Each waiting request holds a thread, so the pool also caps the batch. Sized to the game threads, the shared server ran **16% faster** (258 vs 222 sims/s, 10 min each).
- **The two pilot bugs from docs/006 are fixed and verified on the pod:**
  - When the loop finished its generations, the watchdog stopped cleanly instead of restarting it.
  - The final sync reached the persist directory: 6 checkpoints, 611 MB.
- **On this 13.6-core pod, CPU is the cap.** Both benchmark runs used 98% of the core quota.
  - With the network on, 3 JVMs × 4 threads played **59 games/hr**, about 4.3 per core-hour, at search budget 300.
  - That's **$0.012 per game** on this $0.72/hr pod. Exp #1 cost $0.011 per game at budget 96.
- **Open question: value-label size.** In network self-play, v0.2's value labels sit closer to 0 than v0.1's did at the same λ (details below). Watch it in the first larger run before fixing λ.

## What changed

**Engine: [`danieljbrooks/MageZero`](https://github.com/danieljbrooks/MageZero) `draftzero`.** These are opt-in commits on Will's `main` (`11a5974`). Without their variables, behaviour is unchanged.

| Commit | What |
|---|---|
| CPU fallback | train/test use CUDA when present, else CPU. Checkpoints map to CPU on load. |
| Policy width (#7) | `MZ_ACTION_VOCAB` sets the head width from the vocab's `dim` line. A width mismatch is refused with a clear error; `test.py` used to fall back to an untrained model. This commit is also on branch `action-vocab`, the PR-shaped one. |
| `MZ_SERVER_THREADS` | The server's HTTP pool (default 6). |
| `MZ_TRAIN_BATCH` | States per train/test batch (default 512). |

**XMage: [`danieljbrooks/mage`](https://github.com/danieljbrooks/mage) `v0.2-generalist`.**
- It is exp #1's XMage commit (vocab, deck pools, `GAME_SUMMARY`) rebased onto Will's `cb7e9c6f`.
- `build-generalist-bundle.sh` rebuilds three jars into his release bundle.
- Built from `cb7e9c6f` unmodified, all 46 classes match the release's bytecode.
- The FDN vocab regenerated from v0.2's card classes is byte-identical to exp #1's.

**draft-zero:**
- **`engine.py`** runs MageZero's scripts by their installed path. It stages frozen checkpoints as their own models, since v0.2's server loads only `model.pt.gz`. It also turns the train/test output into metrics rows.
- **Moved in from exp #1's fork:** `metrics`, `report` and `resources`. CPU is now read from the container's cgroup counter, where the fork used host-wide `psutil`.
- **Parallel JVMs (`jvm.jvms`):** they share one inference server per checkpoint. Each server's HTTP pool is sized to the game threads that query it. The eval is split across every JVM slot.
- **Watchdog:**
  - A run that finished its generations counts as done, not crashed.
  - A persist directory inside a synced directory is excluded from the sync.
- **Bad-target audit:** it separates prompts that aren't targets at all and prompts the card's own text restricts to one side. Before, 79% of targeting prompts were unclassifiable; now 20% are.
- **`tools/vocab/`:** `VocabDump`, moved from the archive, with a build script.
- **Tests:** 70 pass, including new ones for the engine adapter, the sync fix and the classifier.

## The pilot

- **Pod:** RunPod Secure RTX PRO 4500 (Blackwell, 32 GB), $0.72/hr. The real cgroup limits were 13.6 cores and 87.5 GB; `nproc` reported 128.
  - No 3090 was in stock.
  - An RTX A5000 came with only 9 vCPUs and was removed within minutes.
- **Layout:** sized from the cgroup: 3 JVMs × 4 threads, 20 GB heap each, generational ZGC. Each generation was 24 games (2 per game thread).
- **Settings:** exp #2's: search budget 300, 60 s timeout, λ = 0.95, all priors off, mix 20 / 70 / 10, training batch 128.
- **Run:** `deploy/pilot_v02_setup.sh`, then `deploy/pilot_v02.sh`.
  - Setup took about 4 minutes.
  - The first training step ran out of GPU memory at batch 512. The pilot resumed from that stage at 128.
- **Cost:** $1.14 in total, including the A5000's few minutes. The balance went from $59.29 to $58.15.

### Generations

| Gen | Games | Wall | Games/hr | Mean turns | Train + test |
|---|---|---|---|---|---|
| 0 (heuristic search, offline) | 24 | ~6 min | 243 | 17.7 | 2 epochs, 4,550 states, 28 s |
| eval: gen 0 vs offline search | 6 | 5.9 min | | 19.0 | |
| 1 (network self-play) | 24 | 24.5 min | 59 | 20.0 | 1 epoch, 9,369 states, 41 s |
| 2 (5 self-play, 19 vs frozen gen 0) | 24 | 24.1 min | 60 | 23.2 | 1 epoch, 12,972 states, 56 s |

- **The network costs about 4×:** offline 243 games/hr against 59 with the network on.
- **Straggler JVMs waste a slice of every generation.**
  - The slowest job took 24.4 min, against 18.0 and 18.8 for the other two, and a generation waits for its slowest job.
  - That tail is part of why CPU averaged 70–78% of the quota per generation, against 98% in steady state.
  - More, smaller jobs per generation would shrink it.
- **Training is cheap:** under a minute per generation at these sizes. Peak GPU memory, with training and the servers together, was 25.8 GB of 32.
- **League:** gen 1's net beat frozen gen 0 in 12 of 19 games (63%, n = 19).
- **Eval:** gen 0 against offline search went 3 of 6 (n = 6).

  Neither result says anything about strength yet.

### Search

| Gen | Searches | Median s | p95 s | Max s | Timeouts | Mean evals/search |
|---|---|---|---|---|---|---|
| 0 (offline, incl. eval) | 2,956 | 0.36 | 1.9 | 4.9 | 0 | 97 |
| 1 (network) | 2,497 | 2.31 | 5.0 | 12.8 | 0 | 125 |
| 2 (network) | 3,103 | 1.82 | 4.3 | 9.3 | 0 | 124 |

- **No search hit a tree limit or the forced timeout,** and no game failed.
- **The laptop failure didn't recur.** One laptop smoke game had died on a "too many nodes" search in v0.2's own MCTS code, which is unchanged since v0.1. It didn't happen in these 78 games.

### Shared inference server

- **Batch sizes:** they averaged 3.0 on the current-net server (p95 13, max 24), and 1.5 on the frozen gen-0 server (p95 4). These are averages over each server's whole run, including the eval and generation tails.
- **The pool benchmark:** two 10-minute runs, same layout, network on, one shared server on the pilot's own gen-1 checkpoint.

| HTTP pool | Sims/s | Searches | Games finished | Cores busy | Timeouts |
|---|---|---|---|---|---|
| 6 (v0.2 default) | 222 | 1,176 | 5 | 13.3 of 13.6 | 0 |
| 24 (sized: 2 × 12 game threads) | **258** | 1,336 | 10 | 13.3 of 13.6 | 0 |

- **The sized pool is 16% faster at the same CPU use.** The games-finished counts are too small to compare (n = 5 and 10).
- **On a 31-core pod,** with 28 game threads on one server, a 6-thread pool would cost more than this.

### Value labels at λ = 0.95

Median |label|, by shard (8 games each; B sides of league games aren't training data):

| | This pilot (v0.2) | v0.1 pilot (docs/006) |
|---|---|---|
| Gen 0, heuristic | 0.21–0.63 | 0.48–0.49 |
| Network self-play | 0.10–0.37 | 0.41–0.45 |
| vs frozen gen 0 (A side) | 0.33–0.49 | — |

- **v0.2's self-play labels sit closer to 0,** and up to 49% of a shard's labels are near 0. The v0.1 pilot's maximum was 17%.
- **It isn't game length.** The shards don't track turns (the 23.5-turn job isn't systematically lower), and gen 0 already varies from 0.21 to 0.63.
- **It's too small a sample to say more.** Check it on the first larger run, before λ is fixed for exp #2. If it holds, compare how v0.2 builds the labels (`generateLabeledStatesForGame`) with v0.1.

### Bad targets

Over all 78 games, 603 "choose target" prompts:

| Category | Prompts |
|---|---|
| Not a target at all (a discard, a fetch) | 122 |
| Restricted by the card's own text | 195 |
| Classified | 150, of which **30 bad (20%, n = 150)** |
| Unclassifiable | 118 |
| Owner unknown | 18 |

- **Examples:** Fleeting Flight on an opponent's creature, Stab on the caster's own Vampire Soulcaller, and Luminous Rebuke on the caster's own Savannah Lions.
- **Harmless cases are counted too.** "Fleeting Distraction" on your own creature is a cantrip with a small downside, so it's often fine, but the audit still counts it as bad.
- **Where to find them:** every case is listed in `bench_local/v02_pilot/runs/*/bad_targets.jsonl`, on the laptop only.

## Next

1. **Before exp #2, rerun the pilot's layout on a 31-core pod.** It should show whether 7 × 4 with the sized shared server stays CPU-bound or becomes server-bound. Cost-per-game estimates for 20,000 games depend on it.
2. **Cut the straggler tail with more, smaller jobs** (`chunk_games` of 4, so several waves per generation). A free JVM slot then picks up the next job instead of waiting for the slowest JVM.
3. **Value labels:** confirm or explain the lower self-play label size before fixing λ.
4. **Merging `v02-migration` into `main`.** The gameplay-data work (docs/008, 009) runs on the v0.1 engine and exp #1's gen-33 checkpoint, which v0.2 can't load, so merging is a decision for Dan.

## Reproducing

- **Laptop smoke test:** `configs/smoke_v02.yml`. It needs the v0.2 bundle in `MZ_XMAGE_DIR` and `MZ_ACTION_VOCAB=assets/vocab/FDN_SPG.tsv`.
- **Pod:**

  ```bash
  HF_TOKEN=<read token> bash deploy/pilot_v02_setup.sh && bash deploy/pilot_v02.sh
  ```

  Add `RESUME=1` to continue a stopped run.
- **Results:** `bench_local/v02_pilot/` on the laptop. It's gitignored and holds the metrics, games, dashboards, logs and benchmark results.
