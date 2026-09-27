# Experiment #2, run 2: starting self-play from human decisions

Run 2 is the imitation half of experiment #2's A/B (ROADMAP, "Run 2 handoff"; decision D8). It
uses run 1's settings exactly (`configs/exp2.yml` through `deploy/exp2.sh`, same pod type) except
for the starting network. Run 1 starts from gen 0's heuristic-search bootstrap. Run 2 starts from
a network pretrained on 17lands human decisions.

**Question:** at equal cost, does the pretrained start do better on exp #2's three goals than run 1?
The goals are win rate against raw search, throughput, and 17lands rank correlation.

This doc records how the start was built and the checks made before launch. It was written
2026-09-27 on branch `exp2-run2`.

## Summary

- **The pretrained network matches held-out human decisions at docs/008's level.**
  - Top-1 is **73.0%** on 16,081 held-out turn-start decisions. docs/008's best was 73.1%: gen 33's
    trunk with retrained heads.
  - Its value head is better than docs/008's (AUC 0.702 vs 0.685).
  - The attack head is close (AUC 0.795 vs 0.83; accuracy 71%, the power/toughness rule's 74%).
  - It was trained from scratch on MageZero v0.2's encoder with the opponent's hand hidden, as the
    loop now plays.
- **Its games look legal and sensible.** In 14 self-play games at run 2's settings, a side averaged:
  - 6.4 lands and 7.1 spells per game;
  - 0.14 missed land drops per game;
  - no passes with a play available, and no bad targets;
  - searches timed out 0.2% of the time.
- **The loop can now start from any checkpoint** (`init_checkpoint`), and it logs every
  generation's agreement with human decisions (`human_agreement`).
- **Launched 2026-09-27 03:42 UTC** on pod `xdslbqu0imovu0` (Secure RTX 3090, 31.1 cores,
  116 GB, $0.50/hr), capped at $28.20 in total including setup.

**One caveat for reading the A/B: exp #2 runs with every prior off, so the search never reads the
policy heads.** The pretrained start can only act through its value head and the trunk
representation it hands to self-play training. The human policy is still in the network, and its
agreement is logged, but the search doesn't use it. The handoff kept priors off on purpose, so
that the starting network is the only difference between the runs. A test of the human policy as a
prior would need the priority prior switched on (docs/008 §7.5).

## 1. Human-decision tables on v0.2

The tables are docs/008's pipeline, rebuilt with the mzbridge compiled against the v0.2 generalist
bundle (`generalist-xmage-v0.2-48e49184`):

```bash
python -m draftzero.gameplay.imitation splits
python -m draftzero.gameplay.imitation build --every 48 --workers 3
python -m draftzero.gameplay.imitation tables
```

- **Build:** 16,483 games and 141,047 turn-start decisions in 7.8 minutes on the laptop.
  140,337 of the decisions built.
- **Hidden hand:** the bridge encodes with `perfectInfo = false`. This is the same StateEncoder
  setting the loop's `hidden_info: {see_opponent_hand: false}` gives the JVM
  (`ParallelDataGenerator` sets `perfectInfo` from `see_opponent_hand`).
- **Tables:**

  | Table | Train | Val | Test |
  |---|---|---|---|
  | Turn-start decisions | 115,210 | 6,958 | 18,169 |
  | Replayed attack decisions | 17,393 | 1,017 | 2,691 |
  | Replayed priority decisions (not used) | 37,991 | 2,147 | 6,123 |

- **Split leak, checked.** This worktree lacked the mirrored-game pairs file
  (`data/gameplay/pairs_FDN_PremierDraft.jsonl`), so `splits` joined no mirrored games. docs/008's
  split joined 18,109.
  - As a result, 2,049 of the 18,169 test decisions (11%) have their mirrored half in train or val.
  - Every number below is on the **clean** test rows only (16,081 turn-start, 2,411 attack).
  - On all rows the result is the same to 0.1 point: 73.1% top-1.
  - The per-generation agreement uses a fixed 3,000-row subset of all test rows, leak included. It
    is a trend line, not a clean estimate.

## 2. Pretraining

`python -m draftzero.gameplay.pretrain train --out models/imitation/pretrained.pt.gz --budget 2400`
ran on the run's own pod GPU, before launch.

- **Model:** MageZero v0.2's NetTransformer (2 layers, d = 512) with 1,024-wide policy heads,
  trained from scratch.
- **Feature vocab:** MageZero's ignore-list rule (`kept_feature_ids`, k = 10) over the train rows,
  at v0.2's hash width (2³¹ − 1). That gives 14,496 rows and a mean state length of 758 tokens.
  - `train.py` later extends this vocab with self-play features, as it does between generations.
- **Losses, trained together:**
  - *Priority:* set NLL of the human's turn over the masked legal softmax (docs/008 §7.3).
  - *Binary:* attack yes/no cross-entropy (slot 1 = yes, MageZero's CHOOSE_USE layout).
  - *Value:* MSE to the game result z = ±1, weight 0.5.
  - The value weight is 0.5, not the study's 0.1, because with priors off the value head is the
    only part the search reads.
- **Training:**
  - Adam at lr 3e-4 with warm-up, batch 64 under a token budget, and token dropout 0.3.
  - 40 minutes, 32,714 steps, about 2.0M samples (17 epochs).
  - Early stopping on the validation loss. The best state came at 24 minutes, and validation was
    flat from 20 minutes on.
- **Checkpoint:** MageZero's format (`model_state_dict` + `feature_vocab`, gzip). The v0.2
  server loads it (`NetTransformer(len(vocab))`), and so does `train.py --checkpoint`. The laptop
  smoke run exercised both.

## 3. Checks before launch (handoff step 4)

**Human agreement**, on the clean held-out test rows:

| | Pretrained start | docs/008 reference |
|---|---|---|
| Priority top-1 in S (n = 16,081) | **73.0%** | 73.1% (gen 33 trunk + heads on 30k rows) |
| Top-3 | 98.6% | 98.3% |
| Set NLL | 0.543 | 0.563 |
| Top-1, ≥ 4 legal options | 70.9% | 70.2% |
| Value AUC vs game result | 0.702 | 0.685 (value head retrained on human outcomes) |
| Attack accuracy / AUC (n = 2,411) | 71% / 0.795 | rule 74%; human head AUC 0.83 |

- The comparison with docs/008 is approximate. The encoders differ (v0.2 against v0.1), and so do
  the splits (see §1).

**Games.** 16 self-play games from the pretrained start at run 2's settings: budget 300, hidden
hand, priors off, 8 JVMs × 2 threads (`configs/run2_precheck.yml`).

- 14 games finished. Per game side:

  | Metric | Value |
  |---|---|
  | Turns (both players) | 17.6 |
  | Lands played | 6.4 |
  | Spells cast | 7.1 |
  | Missed land drops | 0.14 |
  | Idle turns | 0.43 |
  | Pass with a play available | 0% |
  | Bad targets | 0 |

- Search ran at 44 sims/s with a 0.2% timeout rate. One game went past 20 turns.
- **The 16th game hit an engine pathology, not a network one.**
  - On a board with Koma, World-Eater, a v0.2 search found no legal node and ran to the engine's
    300 s hard cap (`ComputerPlayerMCTS2.applyMCTS`), counting the whole tree once per simulation.
    Then it started another one.
  - The check was stopped there; the JVM's `max_minutes: 50` bounds such a game in the run.
  - It can happen in run 1 too (same engine). Watch for it in both runs' search-time tails.
- The laptop smoke test (`configs/smoke_v02_init.yml`) covered 3 generations, 12 games:
  - gen 0 from the pretrained start;
  - gen 1 network self-play;
  - gen 2 against the frozen pretrained gen 0, with its league row;
  - the eval every generation;
  - human agreement for each generation's checkpoint.

## 4. What changed in the code

- **`draftzero.gameplay.pretrain`:** the `train` and `agreement` commands.
- **`draftzero.loop`:**
  - `init_checkpoint`: a fresh run copies the network to `model.pt.gz` and `gen0.pt.gz`, and gen 0
    plays network self-play (the `bootstrap_games` count) instead of heuristic search.
  - The pretrained network stays `gen0.pt.gz`. It is the "gen 0" opponent in the 20 / 70 / 10 mix,
    and gen 0's eval scores it against raw search.
  - The network trained on gen 0's games is kept as `gen0_trained.pt.gz`. It plays gen 1, as in
    run 1.
  - Gen 0 trains with `--checkpoint`, for `epochs` (1), not `epochs_bootstrap`.
  - `human_agreement: {rows: 3000}` runs `pretrain agreement` in a subprocess after every training
    step, and once on the start. It appends `kind: human_agreement` rows to `metrics.jsonl`, and a
    failure is only a warning.
- **`configs/exp2_run2.yml`:** `configs/exp2.yml` plus those two keys, and the model name
  `FDN_exp2_imit`.
- **`engine.run_train`** takes `keep_as` (the checkpoint's saved name).
- **`imitation.device()`** now prefers CUDA.
- Tests: `tests/test_init_checkpoint.py`.

## 5. The run

| | |
|---|---|
| Pod | `dz-exp2-run2`, `xdslbqu0imovu0`, Secure RTX 3090, 32 vCPU sold / 31.1-core cgroup, 125 GB sold / 116 GB cgroup, $0.50/hr, rented 02:18:22 UTC |
| Budget | **$28.20 in total**: run 1's committed $28.90 plus a $1 margin, subtracted from the $58.06 balance at 02:07 UTC. This is below the $29 target, because an overdrawn balance stops every pod on the account, run 1's included. |
| Setup and pretraining | 1.40 h ($0.70) before launch |
| `deploy/exp2.sh` | `DZ_BUDGET_USD=27.45`: graceful stop at 54.15 h, hard removal at 54.90 h (about 2026-09-29 10:36 UTC) |
| Backstop | armed at pod creation: removal 56.4 h after rental ($28.20), whatever happens |
| Layout | 7 JVMs × 4 threads, 11 GB heaps, 112 games per generation, training batch 64 |
| Run dir / HF prefix | `runs/2026-09-27_03-42-21` → `danbrooks/draftzero-checkpoints` under `2026-09-27_03-42-21/`. `gen0.pt.gz` there is the pretrained start. |

**Comparing the runs.** Compare at equal spend. Run 2 spent $0.70 of its budget before gen 0, and
its games take about as long as run 1's. So compare by dollars (or by games), not by
generation number.

- Run 1 logs no human agreement. Its checkpoints on HF can be scored afterwards with
  `python -m draftzero.gameplay.pretrain agreement --checkpoint <genN.pt.gz> --rows 3000`, which
  uses the same fixed rows.
