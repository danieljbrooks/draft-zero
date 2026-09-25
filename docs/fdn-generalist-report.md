# FDN Generalist: one MageZero agent for a whole limited format

**Run:** `2026-09-22_02-46-01` · **Written:** 2026-09-23 · training finished 12:31 UTC, final eval 14:50 UTC, pod deleted after the final push
**Author:** Daniel Brooks · **Engine:** MageZero (Will Wroble) · **Harness:** draft-zero · **Data:** [17lands](https://www.17lands.com/public_datasets) (CC BY 4.0)
**Code:** [`danieljbrooks/draft-zero` @ tag `exp1-fdn-generalist`](https://github.com/danieljbrooks/draft-zero/tree/exp1-fdn-generalist) · **Release:** [`danbrooks/draftzero-fdn-exp1`](https://huggingface.co/danbrooks/draftzero-fdn-exp1) on Hugging Face (§5) · **Next:** [ROADMAP.md](../ROADMAP.md)
*Revised 2026-09-25 to add links, data provenance, the public release, Appendix A and the Addendum. The results are unchanged.*

---

## Summary

- **Goal.** Train a single MageZero self-play agent that plays *any* Foundations (FDN) limited deck, using 28,366 top-player decks. Then ask two things: does play improve, and can self-play produce 17lands-style card and color statistics?
- **Scale.** 2,507 games over 34 generations on one RunPod L40S (~34 h of training, ~$28 of pod time). The budget was 96 MCTS simulations per decision, and the only prior enabled was the yes/no one.
- **Strength.** Against the same search without a network ("raw search"), the trained checkpoints won **137/238 (57.6%, 95% CI 51–64%)** pooled across all milestone evals. In the 400-game final eval, the gen-33 checkpoint beat raw search **110/197 (55.8%, CI 49–63%)** but only tied the **gen-10** checkpoint, **96/196 (49.0%, CI 42–56%)**. The network adds a modest edge over raw search, and training stopped improving by about gen 10.
- **17lands-style stats** work mechanically: GIH WR, GNS WR, IWD, GP WR, and color-pair win rates. But the agent's card valuations agree with 17lands only weakly: rank correlation **0.28** on commons from gens 10+. That agreement *declined* during training, from 0.43–0.45 in gens 1–14 to 0.05 in gens 15–24.
- **Consistent weaknesses: removal and blue.** Blue commons fall furthest short of their 17lands win rates, and red is close behind. The premium removal spells play at or below average: Stab 46% vs 58% on 17lands, Burst Lightning 46% vs 58%, Refute 45% vs 58%.
- **Conclusion.** 57% is progress, and it shows the network adds value over raw search. It isn't yet a strong enough player for its self-play statistics to mean much as card evaluations. The next run should adopt MageZero's own conventions (v0.2.0, priors off, value-label histograms to tune λ) and run bigger, with stronger evals.

---

## 1. Intention

MageZero trains an AlphaZero-style agent to play Magic: The Gathering through XMage. It uses self-play, MCTS guided by a neural network, and per-decision-type policy heads. It is built to train **one agent for one deck**.

This experiment asked whether the same machinery can train **one agent across an entire limited format**:

1. **Limited self-play at format scale.** Every game samples two decks from 28,366 top-player FDN decks. The agent never specializes, so it has to learn general play: curving out, combat, removal, and card evaluation across all ten color pairs.
2. **Measurable improvement.** Win rate against a fixed baseline should rise over generations.
3. **17lands-style statistics from self-play.** If a strong agent plays thousands of games with real decks, its per-card win rates (GIH WR, IWD, and so on) and per-archetype win rates should resemble 17lands'. Where they don't, that's either a flaw in the agent or a place where human play is off. An agent that played well enough could be used to evaluate cards and archetypes in a set with no human data yet.

## 2. What is new here

| | MageZero (upstream) | This experiment |
|---|---|---|
| Decks | One deck per agent | **28,366 train decks + 3,150 held-out eval decks**, one agent |
| Opponents | Self, past self, heuristic / minimax AIs | Self, league of past checkpoints, gen 0 |
| Output | A strong agent for that deck | An agent plus **format statistics**: card GIH WR / IWD, color-pair WR, correlation with 17lands |
| Evaluation | Per-deck win rates vs AIs | Paired, fixed-deck evals vs raw search; a 400-game final eval |

As far as I know, no one had tried a format-wide MageZero agent or 17lands-style analytics from self-play. MageZero's author called it "an interesting experiment" and warned that policy priors probably won't work with an action space that large (§7).

## 3. Setup

### 3.1 Code versions

| Component | Version |
|---|---|
| draft-zero (this harness) | `danieljbrooks/draft-zero`, tag **`exp1-fdn-generalist`**, which includes branch `final-eval` @ `661ad10` and this report (training ran on `main` @ `a65951e`; the pod's code was verified identical to the repo) |
| MageZero engine | `git+https://github.com/danieljbrooks/MageZero@bcc76de`: package 0.1.0, i.e. upstream v0.1.0-alpha plus 7 fork commits (§7, item 2); **does not include upstream v0.2.0-alpha** |
| PyTorch / CUDA | torch 2.9.1+cu128, driver 550.144.03 |
| JVM | OpenJDK 21.0.12, `-XX:+UseZGC` (non-generational), `-Xmx48g` |
| Python | 3.12.3 |

### 3.2 Data

- **Decks.** 31,516 top-player FDN decks in XMage `.dck` format, split by draft into **28,366 train** and **3,150 eval** decks (index and split: `assets/decks.tsv`). Each carries its main colors and the win-rate bucket of the player who built it. They're every deck in 17lands' public FDN Premier Draft game data whose player sits in the ≥60% win-rate bucket.
  - *Rebuild:* `tools/extract_decks.py` regenerates all 31,516 files byte for byte, and the committed split, from 17lands' public data (checked 2026-09-25; README, "Data").
  - *Sample:* the 40 fixed milestone-eval decks, plus 40 train decks, are committed in `assets/sample/`.
  - *Full pool:* published with the release (§5).
- **Human reference.** 17lands public FDN Premier Draft game data, 791,159 games. From it, per-card GIH WR (`assets/reference/FDN_gih.json`).
- **License.** 17lands publishes its public datasets under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The decks and reference statistics here derive from them and carry the same license.
- **Card rarity and colors** come from XMage's own card database, which was needed to restrict stats to commons.

### 3.3 Model

MageZero's `NetTransformer`, with **21.9M parameters**:

- **Input.** A sparse game-state feature embedding. It grew during the run as new features appeared, ending at **31,676 × 512**.
- **Body.** A 2-layer transformer encoder (d_model 512, 4 heads, feed-forward 1024), then an MLP 512→256. Dropout is 0.3 on input and 0.2 on the embedding.
- **Five output heads:**
  - player priority (what to cast, play, or activate, or pass; 128 slots)
  - opponent priority (a prediction of the opponent's action; 128 slots)
  - target choice (128 slots)
  - yes/no choice (2 options)
  - value

### 3.4 Training loop and parameters

Each generation plays a batch of games, trains on a replay buffer, and at every 5th generation runs a strength eval. Parameters from `configs/fdn_l40s.yml`:

| Parameter | Value |
|---|---|
| Target | 2,500 games (stopped at 2,507 after gen 33) |
| Games per generation | 72 (gens 0–1: 96 bootstrap games) |
| Opponent mix | 70% self, 20% a random past checkpoint ("league"), 10% gen 0 |
| Replay buffer | 150,000 states |
| MCTS search budget | **96 simulations per decision**, 12 s timeout, tree reuse across moves |
| JVM | 18 game threads, 24-game chunks, 48 GB heap, 25 min cap per job |
| Deck sampling | Random train decks for both sides each game |

### 3.5 Curriculum and priors (`configs/curriculum.yml`)

| Setting | Value |
|---|---|
| TD discount λ | 0.95 (gen 0), 0.92, 0.85, then **0.70 from gen 3** |
| Prior temperature | 1.5 |
| **Yes/no prior** | **On from gen 2** |
| Priority prior | **Off.** Enabled at gen 3 in an earlier FDN experiment, it halved the number of spells cast. |
| Target prior | **Off.** The target head never learned (§6.6). |
| Opponent prior | Off |

λ controls the value targets: `y_i = λ·y_{i+1} + (1−λ)·h_i`, where h_i is the MCTS root score and the final target is the game result. MageZero's README quotes 0.9–0.95, but its default `curriculum.yml` settles at 0.70, and the value-label histograms (§6.2) show why: in this format 0.95 already produced bimodal labels, the sign that λ is too high. At 0.70 the labels form one clean hump.

### 3.6 Evaluation strategy

1. **Milestone evals, every 5 generations.**
   - **Games:** the new checkpoint plays **20 fixed deck pairs, each played twice with sides swapped (40 games)**, against **raw search**.
   - **Raw search:** the same MCTS at the same 96-simulation budget, with XMage's hand-written evaluator instead of the network.
   - **Deck control:** the fixed decks make each milestone a paired comparison with the last.
   - **Windowed estimate:** a pooled win rate over the last 10 generations smooths the ±15-point noise of a single 40-game eval.
2. **League games.** 20% of every generation is the current checkpoint against an older one. Later checkpoints should beat earlier ones.
3. **Per-generation prediction checks.** Each new network is scored on the new data: accuracy per head, value-prediction correlation.
4. **Final eval (400 games), run after training stops and before the pod is torn down.**
   - The final checkpoint plays 100 pairs (200 games) against raw search and 100 pairs against the **gen-10** checkpoint.
   - Pairs 1–20 are exactly the milestone matchups (same fixed seed; verified against the gen 5–20 evals).
   - Play/draw is balanced (80 vs 78 in the first 158 games), and both sides get identical budgets and timeouts.
5. **17lands-style stats** (§6.5), counting **only the current-generation agent's side** of each game.

### 3.7 17lands-style statistics

`draftzero.stats` and the run's `format.html` compute:
- color-pair records
- per-card **GIH WR** (win rate when the card was in the opening hand or drawn)
- rank correlation with 17lands per generation

For this report I added:
- **GNS WR** (in the deck, never seen)
- **IWD** (GIH − GNS)
- **GP WR** (win rate of decks containing the card)

The decklists supply the last two. There is also a sortable page with all of these ("FDN Agent Card Stats").

One attempted metric failed: the engine's built-in **bad-target classifier** (targeting your own creature with removal) logged zero classified targets all run. It needs each card's rules text, which it never receives. I rebuilt it offline from the game logs instead (§6.6).

## 4. Infrastructure

### 4.1 The instance

| | |
|---|---|
| Provider | RunPod, **community cloud** |
| GPU | 1× NVIDIA L40S, 46 GB |
| CPU / RAM | 28 vCPU (≈24 usable under the container's cgroup), 125 GB |
| Disk | 80 GB container disk, **no network volume** (community cloud doesn't support them) |
| Price | **$0.79/hr** |
| Durability | Checkpoints and run files pushed to a private Hugging Face repo every generation; a watchdog verifies the push before allowing teardown |
| Throughput | ~64–90 network self-play games/hr; ~41 simulations/s with the network vs ~84 without |

**Throughput is capped by inference, not CPU.** MageZero's inference server is a single worker loop on one GPU, and it tops out around 200 requests/s. On a 4× RTX 5090 box (192 vCPU, 163 usable), only ~40 cores did work and throughput matched a 40-core box. So the best value is a single mid-range GPU with about 24 cores. More cores or GPUs don't help until inference scales out.

### 4.2 Getting a working instance

Finding a usable instance took more time than configuring the run.

- **The cheap machines are community-cloud only.** Those are the high-vCPU, small-GPU boxes this CPU-heavy workload wants. For example, an RTX 3090 with 25 vCPU at $0.22/hr. Network volumes exist only on secure cloud, so affordable meant no persistent disk. That's why the Hugging Face checkpoint push was built before anything else ran.
- **Community stock turned over faster than I could claim it.** It acted like an auction floor:
  - An RTX 3090 (114 vCPU/$) and an L4 sold out mid-survey.
  - A 2× RTX 4090 with 170 vCPU at $0.68/hr sold out within two minutes, twice.
  - 4× RTX 4090 and 2× RTX 6000 Ada listings were shown as available, but create calls failed with "no instances available".
  - I ended up writing a grab loop that polled candidates in order of vCPU per dollar.
- **Some hosts had broken CUDA.** A community 2× RTX 5090 host passed `nvidia-smi`, but `torch.cuda.is_available()` was False (its `nvidia-uvm` device was owned by `nobody:nogroup`). RunPod handed back **the same defective machine four times** (identical GPU UUID and host IP). The fix was a CUDA health check in the grab loop, a bad-host denylist, and automatic deletion of any pod that failed.
- **Orphaned pods kept billing.** Aborted grabs left pods running, including a forgotten A4000 test pod at $0.25/hr. The grab loop now cleans up after itself.
- **Dependency drift.** `pip install -e .` pulled a torch build too new for the host's CUDA 12.8 driver, and CUDA silently went unavailable. Bootstrap now pins torch.
- **The path to the final box:**
  1. RTX 4000 Ada, secure: smoke test ($1.85).
  2. RTX 5090, secure, $0.99/hr: 5.8 h, GPU only 11–14% utilized.
  3. 4× RTX 5090, secure, $3.96/hr: ~$1.13, which proved the inference bottleneck.
  4. **L40S, community, $0.79/hr**: the run in this report, best in games per dollar of everything tested.

### 4.3 Reliability during the run

| Incident | Effect | Fix |
|---|---|---|
| Embedding table grew, but the recorded row count didn't | A size mismatch crashed every restart at gen 2 | Engine `5cbf790` + `bcc76de` |
| The watchdog's restart command launched another watchdog | A cascade of 269 watchdogs overloaded the pod and the HF API | Restart the loop only; watchdog singleton guard (`a65951e`) |
| Resuming could discard a generation's games | 67 completed gen-1 games lost | `8abd674` |
| A credential file was executed as shell | A token was exposed (revoked and replaced) | Credentials are parsed, never sourced |
| Inference server exited before becoming ready (gen 16) | 1 crash, auto-restarted in 31 s | Retry added to the final eval |
| Every watchdog start sent a false "sync failing" alert | Most of the run's 419 `sync_failed` alerts | `2f22bad` |

After the fixes the run finished with one transient crash (gen 16) and no lost generations.

### 4.4 Cost

Total spend on the account was **$38.68 of $100**, including every failed or exploratory instance. The L40S run itself was about 35.5 h × $0.79 ≈ **$28** through the final eval.

## 5. What's saved

**Public release:** [`danbrooks/draftzero-fdn-exp1`](https://huggingface.co/danbrooks/draftzero-fdn-exp1) on Hugging Face (CC BY 4.0). It's the subset someone needs to reuse this run:

| Path | Contents |
|---|---|
| `checkpoints/` | Gens 0, 10 and 33. Gen 33 is final; gen 10 is where strength plateaued. |
| `decks/` | The full 31,516-deck pool, the train/eval split, and the fixed eval deck pairs |
| `games.jsonl`, `metrics.jsonl`, `final_eval.json`, `deck_records.tsv` | Every game, all metrics, the final eval |
| `dashboards/` | The run's `dashboard.html` and `format.html` |
| `report.md` | This report |

**Caveat:** the checkpoints are MageZero **0.1.0**-based (§3.1). Whether they load under v0.2.0 is untested.

**Full archive (private):** everything is in the HF repo `danbrooks/draftzero-checkpoints/2026-09-22_02-46-01/`:

| Path | Contents |
|---|---|
| `models/` | All 35 checkpoints, gens 0–33 plus the latest (6.75 GB) |
| `games.jsonl`, `metrics.jsonl`, `run.json`, `alerts.jsonl`, `deck_records.tsv` | Every game with both decks, colors, cards drawn and winner; all metrics |
| `replay/` | All 269 replay shards (6.9 GB): training states with search visit distributions and value targets |
| `logs.tar.gz` | Every per-game JVM log (casts, targets, per-decision search visits and scores) plus loop and watchdog logs. The complete version is pushed before teardown. |
| `extras.tar.gz` | The run's own dashboards, the exact deck pools, all 196 generated game configs, configs, the watchdog's final status |
| `final_eval.json` | Final eval results (pushed on completion) |

## 6. Results

### 6.1 Strength vs raw search over time

| Checkpoint | Record | Win rate |
|---|---|---|
| Gen 0 (untrained) | 20/39 | 51% |
| Gen 5 | 22/40 | 55% |
| Gen 10 | 27/39 | 69% |
| Gen 15 | 20/40 | 50% |
| Gen 20 | 19/39 | 49% |
| Gen 25 | 26/40 | 65% |
| Gen 30 | 23/40 | 58% |
| **All trained milestones (5–30)** | **137/238** | **57.6% (95% CI 51–64%)** |

**Variance dominates single milestones.** A 40-game eval is about ±15 points. The sequence 69 → 50 → 49 → 65 looks like a peak, a collapse, and a recovery. It's consistent with a flat ~57% plus noise, even though the decks were identical each time. That noise is why the milestone windows were pooled, and why the final eval uses 200 games per opponent. For about the same budget, a larger eval run less often is worth more than frequent small ones.

**Loss vs strength.** Value loss fell steadily, from 0.072 to 0.016, but strength was flat from gen 10 onward. Most value labels are partly the search's own estimates, so a falling loss says little about whether the agent plays better.

### 6.2 Value-label histograms (TD range check)

MageZero's `dataset_stats.py` plots value-label histograms. A **bimodal** histogram, with mass piled at ±1, means λ is too high and the value network suffers (Will Wroble's rule of thumb). draft-zero never produced these, so they were computed afterwards from all 579,920 replay states, with the same 21 bins over [−1, 1]:

| Generations | λ | States | Histogram (−1 → +1) | abs(v) > 0.8 | abs(v) < 0.2 | Bimodality coef. |
|---|---|---|---|---|---|---|
| 0 | 0.95 | 18,686 | `▇▄▄▅▄▅▄▃▄▄▆▄▅▅▄▅▅▆▆▆█` | 25.0% | 18.7% | **0.58** |
| 1 | 0.92 | 23,218 | `▃▃▄▆██▆▆▆▆▅▆▆▆▇█▇█▇▄▃` | 12.1% | 19.2% | 0.56 |
| 2 | 0.85 | 17,098 | `▂▃▆███████▇▅▆▇██▇▆▄▂▂` | 7.9% | 21.7% | 0.53 |
| 3–10 | 0.70 | 125,732 | `▁▁▂▂▃▄▅▇███▇▆▅▄▃▃▂▁ ▁` | 5.3% | 36.9% | 0.39 |
| 11–20 | 0.70 | 133,547 | `▁▁▁▂▂▃▄▆███▆▅▄▃▂▂▁▁ ▁` | 5.8% | 40.4% | 0.35 |
| 21–33 | 0.70 | 160,639 | `▁▁▁▁▂▂▃▅██▇▄▃▂▂▁▁▁▁ ▁` | 7.1% | 42.8% | 0.34 |

(A bimodality coefficient above about 0.555 suggests two humps.)

- **λ 0.95 is too high here.** Gen 0's labels are bimodal: a quarter of them have abs(v) above 0.8, and the coefficient is 0.58.
- **λ 0.70 is in range.** The labels form a single hump, which supports MageZero's default.
- **The hump narrows at fixed λ.** Labels with abs(v) below 0.2 rise from 37% to 43% over training. That's consistent with a value network growing less decisive, which fits the flat strength after gen 10. The histograms show λ isn't too high; whether 0.70 is too *low* needs a sweep.

### 6.3 Final eval (400 games)

The gen-33 checkpoint played 100 pairs against each opponent in 2.2 hours. Three games against raw search and four against gen 10 didn't finish.

| Opponent | Record | Win rate (95% CI) |
|---|---|---|
| Raw search | **110/197** | **55.8% (49–63%)** |
| Gen-10 checkpoint | **96/196** | **49.0% (42–56%)** |

- **Against raw search:** about the same margin as the pooled milestones (57.6%), so a stable, modest edge.
- **Against gen 10:** a coin flip. The final model is no stronger than gen 10, 23 generations and about 1,650 games earlier. It isn't clearly weaker either.
- **Early batches can mislead.** Midway, the final model trailed gen 10 at 46/116 (40%, CI 31–49%), which looked like a significant regression. The last 80 games went 50/80 and pulled it back to even. A 40-game checkpoint can mislead even when its interval seems to exclude 50%.
- **Taken together:** with the league games (§6.4), this says **training plateaued around gen 10**, rather than regressing.

### 6.4 Other signals of a plateau

- **League games** (snapshot at gen ~25): later checkpoints did *not* beat earlier ones. Gens 15+ against checkpoints 1–14 went **57/123 (46%)**; gens 20+ against gens 10–14 went **24/56 (43%)**.
- **Agreement with 17lands declined as training went on.** Rank agreement of commons with 17lands GIH WR, using equal ~500-game windows:

  | Window | Agreement (commons with ≥30 games) |
  |---|---|
  | Gens 1–10 | +0.43 |
  | Gens 5–14 | +0.45 |
  | Gens 10–19 | +0.22 |
  | Gens 15–24 | +0.05 |

  The windows are the same size, so smaller samples don't explain the drop. The disjoint windows (1–10 vs 15–24) differ at p ≈ 0.02.
- **Play style shifted around gen 12.** Attacks per turn rose from about 0.30 to 0.44. Activated abilities per turn roughly halved, from 0.14–0.20 to about 0.10. Games shortened from ~25 to ~20 turns. The rate of casting spells didn't change (~0.42 per turn).

### 6.5 17lands-style stats: the current-generation agent, gens 10–33

These cover 1,686 games and 2,880 agent deck results: self-play counting both sides, plus league games counting the agent's side only. Self-play win rates center on 50% by construction, while 17lands' human GIH averages about 55%, so **compare rankings, not levels**.

**Color pairs**

| Pair | Record | Win rate |
|---|---|---|
| Selesnya (WG) | 109/186 | 59% |
| Orzhov (WB) | 274/487 | 56% |
| Gruul (RG) | 98/182 | 54% |
| Azorius (WU) | 186/354 | 53% |
| Simic (UG) | 81/156 | 52% |
| Boros (WR) | 96/192 | 50% |
| Rakdos (BR) | 152/316 | 48% |
| Dimir (UB) | 186/422 | 44% |
| Golgari (BG) | 54/132 | 41% |
| Izzet (UR) | 94/235 | **40%** |

**Commons, sample.** Of 86 commons with 30+ games in hand:
- **Top by 95% lower bound:**

  | Card | Record | GIH WR | 17lands GIH |
  |---|---|---|---|
  | Beast-Kin Ranger | 113/190 | 59% | 55.2% |
  | Inspiring Paladin | 111/189 | 59% | 53.9% |
  | Dazzling Angel | 216/377 | 57% | 57.8% |
  | Felidar Savior | 202/355 | 57% | 57.4% |
  | Healer's Hawk | 212/375 | 57% | 57.5% |

- **Bottom:** Firebrand Archer 22/68 (32%), Gleaming Barrier 16/43 (37%), Sure Strike 35/93 (38%), Witness Protection 58/154 (38%).
- **Most negative IWD** (drawing the card hurt):
  - Fleeting Flight: −16.7 points (254 games in hand / 276 in deck but unseen)
  - Giant Growth: −14.9
  - Witness Protection: −12.0

  Two pump spells and an aura, all cards that depend on picking the right target.
- **Rank agreement with 17lands:** 0.28.

### 6.6 Removal and blue underperform

**By color.** Average agent GIH WR minus 17lands GIH WR, weighted by games, commons with 30+ games:

| Color | Commons | Gap vs 17lands |
|---|---|---|
| White | 15 | −1.5 |
| Green | 13 | −1.7 |
| Colorless | 15 | −4.5 |
| Black | 15 | −5.5 |
| Red | 14 | −7.3 |
| **Blue** | 14 | **−8.0** |

About −5 is the structural baseline (self-play vs a human average near 55%). White and green overperform relative to that baseline, and blue and red underperform. Blue-based pairs are among the worst colors: Izzet 40%, Dimir 44%.

**Removal.** The premium removal and interaction spells, the best commons for humans, play at or below the agent's average:

| Card | Color | Agent GIH | 17lands GIH |
|---|---|---|---|
| Stab | B | 224/486 (46.1%) | 57.9% |
| Burst Lightning | R | 178/388 (45.9%) | 57.9% |
| Refute | U | 187/413 (45.3%) | 57.6% |
| Fleeting Distraction | U | 138/302 (45.7%) | 54.8% |
| Bake into a Pie | B | 207/411 (50.4%) | 58.0% |
| Eaten Alive | B | 206/421 (48.9%) | 56.1% |
| Luminous Rebuke | W | 181/340 (53.2%) | 57.7% |
| Involuntary Employment | R | 58/136 (42.6%) | 55.6% |
| Witness Protection | U | 58/154 (37.7%) | 52.8% |

**Why.** The diagnostics point at targeting, timing and evaluation, not at misreading the board.

- **Aiming removal at its own creatures is rare but real** (from the gens 0–25 game logs). With a curated list that excludes sacrifice costs and cantrips, removal hit the agent's own creature in **5.8% of cases (48/824)**. That rate was flat across generations and matched the untrained gen 0 (4/55), and there was always another legal target. In decisions where both an own and an opponent's creature were legal, the rate was 19/654 (2.9%). In 15 of those 19, the search's own scores rated the wrong target best. That's an evaluation error, which more simulations wouldn't fix.
- **The target head never learned.** Its accuracy stayed around 0.32 against chance of about 0.31 (~3.2 legal targets). With the target prior off, search spreads its visits evenly across targets at the start. The yes/no head, by contrast, went from 37% to 88%.
- **Search isn't guessing blindly.** In 1,138 removal targeting decisions, the chosen target got a median 54% of visits (25% would be uniform). Own creatures got 21% of visits while being 45% of legal targets. The weakness is *which opponent creature* to kill, and when, which needs a better evaluator. Blue's instant-speed interaction (Refute, Fleeting Distraction, Run Away Together) depends on exactly those timing decisions.
- **Pump spells land on the wrong creature too.** Giant Growth and Sure Strike landed on an opponent's creature 10/44 times (23%, gens 1–25).

### 6.7 Conclusion

- **57% is progress.** Against search with an identical budget and no network, trained checkpoints win 137/238 (57.6%, CI 51–64%), and the final checkpoint wins 110/197 (55.8%). The network adds value.
- **But it plateaued early.** The final checkpoint only ties gen 10, 96/196 (49%). The last ~1,650 games of self-play bought no measurable strength, while the value labels narrowed (§6.2) and card valuations drifted away from human ones (§6.4).
- **It isn't yet a player whose statistics mean much.** Card valuations agree with humans at only 0.28. They drifted *away* from human values while strength stayed flat. The failures (removal, blue, pump spells, auras) cluster around target selection and timing, where the agent is weakest.
- **Self-play statistics are only as good as the self-play.** A generalist's 17lands-style numbers describe how *this agent* uses cards. They estimate card quality only once the agent plays near human level. The pipeline works end to end: stats, fixed paired evals, decklist-based GNS and IWD, rank agreement with 17lands. The limit is the agent, not the measurement.

## 7. Aligning with MageZero conventions

Several choices here drifted from MageZero's own, and at least two likely cost strength:

1. **Engine version.** The pin (`bcc76de`) is 0.1.0-based. Upstream v0.2.0-alpha adds resume that tracks partial HDF5 progress, `.txt` deck support, and removes the `full_table` path. The next run should be on 0.2.0.
2. **Upstream the fixes.** Seven fork commits aren't upstream:
   - **Two embedding bugs:** a stale `num_embeddings` after the table grows, and trusting stale `embed_rows` metadata. These crash any run whose vocabulary grows.
   - **Server threading:** a configurable CPU thread count, and an HTTP pool sized to the game threads instead of a fixed 6.
   - **Three smaller commits:** runner path resolution, a test import fix, and a metrics/runner module.

   These should go to Will as PRs rebased on his `main`, with the encoding kept backward compatible, the same way the dense-vocab change was merged.
3. **TD discount and value histograms.** The run used MageZero's default λ schedule (0.70 from gen 3). The value-label histograms (§6.2) confirm 0.70 isn't too high, and show the README's 0.9–0.95 is too high for this format. Future runs should produce these histograms every generation, as MageZero's runner does, and tune λ from them.
4. **Priors.** Will's advice for a format-wide action space is to turn priors off. This run had only the yes/no prior on. Next time, run it all-off, with the yes/no prior as an A/B arm.
5. **Reporting.** Use MageZero's own measures (games per generation, simulations per second, per-head accuracy) alongside the strength evals, so results compare directly with per-deck runs.
6. **Encoding stability.** Keep feature and action encodings compatible with MageZero's export/import tooling, which still assumes `ignore.roar`. Anything the generalist changes about its growing vocabulary has to stay backward compatible with that.

## 8. Next steps: a bigger run on MageZero 0.2.0

1. **Re-base.** Rebase the fork onto v0.2.0-alpha, upstream the fixes (§7, item 2), and re-pin draft-zero.
2. **Training signal.** Plot value-label histograms every generation and run a small λ sweep (0.6 / 0.7 / 0.8) to see whether the narrowing hump (§6.2) limits strength. Priors off, with the yes/no prior as an A/B arm. Generational ZGC.
3. **Scale inference, then scale out.** Run one inference worker per GPU so multi-GPU machines actually pay off, or run several single-L40S workers through draft-zero's worker abstraction. Target the original **~20,000 games** (≈250 L40S-hours, ~$200; about 2.5 days on four workers).
4. **Stronger evals.**
   - Run a larger fixed eval (100 pairs) less often, instead of 40 games every 5 generations.
   - Add head-to-head matches against earlier milestone checkpoints to build a rating ladder.
   - Train **single-deck specialists** for a few archetypes, to measure what generality costs.
5. **Targeted fixes for the observed weaknesses.**
   - Allow a separate, larger search budget for targeting decisions (about 9% of all decisions; roughly +20% cost). This needs an engine change.
   - Consider a rules-based targeting prior (for harmful effects, prefer the opponent's permanents).
   - Test 96 vs 300 simulations as a strength check.
6. **Instrumentation.**
   - Give the bad-target classifier each card's rules text, and log the targeted player unambiguously (player targets currently come out in the logs under the wrong player's name).
   - Log per-card usage: cast rate when castable, timing.
   - Keep saving replay data and logs by default (now automated).
7. **Research questions.**
   - Does rank agreement with 17lands rise as strength rises? (That's the test of whether self-play stats become meaningful.)
   - Does a λ sweep change strength or the narrowing of the value labels?
   - How does a generalist compare with specialists?
   - Does the removal and blue gap close with better targeting?

---

## Addendum (2026-09-25): feedback since this was written

- **λ direction is disputed.** §7 item 3 and §8 item 2 read 0.70 as not too high and propose sweeping 0.6 / 0.7 / 0.8. After reading the §6.2 histograms, Will (MageZero's author) disagreed:
  - Bimodal value labels mean λ is too **high**; roughly flat is the target.
  - For multi-deck self-play with small generations, λ should stay fixed at about **0.95**.
  - The single narrowing hump here is the *too-low* side, and together with the 96-simulation budget it likely explains the early plateau.

  The proposed sweep can't test that hypothesis, so the next run should sweep upward.
- **Other advice for the next run:** search budget 300, an opponent mix of 20% self / 70% previous / 10% gen 0 (this run used 70 / 20 / 10), larger evals, and dropping the inference server's feature dedup step (~25% of each request in Will's tests).
- **Where the plan lives now:** §8 is superseded by [ROADMAP.md](../ROADMAP.md), which tracks these as open decisions.

## Appendix A: before this run

Two earlier setups preceded the RunPod generalist. Both trained per-deck agents in MageZero's own style, not a format-wide one.

- **Laptop (M1 Pro, 16 GB).** FDN RG vs UB co-training ran for 8 generations × 48 games (run `2026-09-17_06-55-47`). Throughput was about 85–100 games/hr without the network and 14–27 games/hr with it. That was too slow for the generalist.
- **Kaggle GPU notebooks** (2× T4, 4 vCPU, 31 GB, free tier), suggested by chrismaghuhn on the MageZero Discord. FDN UW vs BR co-training ran in two variants: standard 128-slot action heads, and an FDN set-wide action vocabulary.
  - **Scale.** 319 games completed (1 failed, 5 timed out) in 18.2 h of self-play, across 24.9 notebook-hours in four sessions: **17.5 games/hr**.
  - **Progress.** The standard variant reached 4 of 8 generations and the vocabulary variant 6 of 8. A third session was pushed for each but never pulled.
  - **Strength.** Evals were 6 games per matchup, with 95% intervals about ±30 points, so they can't show learning either way.
  - **Why it was abandoned.** A notebook has only 2 physical cores, while XMage self-play is CPU-bound. Kaggle's advantage is free hours and parallel notebooks, not per-hour speed, and a single RunPod pod with ~24 cores was simpler and ~4× faster.

## Credits

- **[17lands](https://www.17lands.com/)** for the public game data (CC BY 4.0) behind every deck and every human reference number in this report.
- **Will Wroble** for [MageZero](https://github.com/WillWroble/MageZero), and for the review and advice summarized in the Addendum.
- **[chrismaghuhn](https://github.com/chrismaghuhn)** for advice on compute (the suggestion to try Kaggle notebooks, Appendix A) and on performance ([WillWroble/MageZero#3](https://github.com/WillWroble/MageZero/issues/3)).
- The **[XMage](https://github.com/magefree/mage)** project for the rules engine.
