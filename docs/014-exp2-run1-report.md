# Experiment #2, run 1: final report

Run 1 is experiment #2's baseline arm: MageZero v0.2 self-play on FDN from a heuristic-search start
(gen 0 plays raw search; the first network trains on those games). Run 2, the imitation-start arm,
and the comparison between the two through gen 10 are in [docs/013](013-exp2-imitation-report.md).
This report covers run 1 to the end: gens 0–18, stopped on 2026-09-28 at 21:36 UTC.

Experiment #2 had three goals:
1. a high win rate against the baseline (raw search at the same budget);
2. high throughput;
3. good agreement with 17lands on commons.

## Summary

- **Win rate against raw search: no gain.** Run 1 started at 47% and was flat from gen 8.

  | Network | Eval games won | Share | 95% CI |
  |---|---|---|---|
  | Gen 0 | 92 of 194 | 47.4% | 41–54% |
  | Gen 8 | 98 of 196 | 50.0% | 43–57% |
  | Gen 16 | 95 of 191 | 49.7% | 43–57% |

- **The league shows learning that stopped around gen 8.**
  - In gens 1–8, new networks beat older ones **58.7% (357/608)**.
  - In gens 9–18 that fell to **51.7% (449/869)**, and to 50.4% (262/520) in gens 13–18.
  - Later networks still beat the gen-0 network (61.5%, 67/109, gens 9–18), but not each other.
- **The policy heads kept learning after strength stopped; the value head didn't.**
  - Held-out loss (the previous network on each new generation's games) fell from 0.907 to 0.553.
    Almost all of that is policy.
  - Held-out value loss was 0.075–0.087 in gens 1–9 and 0.090–0.102 in gens 14–18, while
    training value loss kept falling (0.089 → 0.049). That's overfitting.
  - With priors off, the search reads only the value head. So the part of the network that
    improved never reached play, and the part that reached play stopped improving at about the
    time the league flattened.
- **Throughput: 65–74 games/hr in steady state,** 2,673 games in 43.6 hours. The single shared
  inference server was the bottleneck: the pod's CPU quota was only 59–72% used and the GPU
  16–41% busy.
- **17lands agreement on commons: GIH ρ +0.20** (84 commons, 1,964 network games). Experiment #1
  reached 0.28.
  - It declined as the run went on: +0.22 in gens 1–9, +0.06 in gens 10–18.
  - Deck-level stats agree much better (GNS +0.51, GP +0.48) than the card's own effect
    (IWD +0.04). The bot mostly agrees with humans on which *decks* are good, not on what a card
    adds when drawn.
- **Bad targets: 6.8% of permanent targets were real mistakes** (304/4,455).
  - The network made fewer than raw search: in the same eval games, 6.0% (25/420) against 9.0%
    (41/458).
  - The rate was flat across generations.
  - The raw audit reads 16.6%, but that includes a logging artifact and harmless targets (§2.4).
    The in-loop metric recorded almost nothing, so this is from a post-run audit of every JVM log.
- **Human agreement:** among non-Pass plays, the network's top choice matched a human's in
  **62.5%** of held-out decisions at gen 18 (chance 48%). That's the same level as run 2 (61.5%),
  which answers docs/013's open question: a network that never saw human data gets there on its
  own.
- **Cost: about $22.10** of the $29 cap (43.7 pod-hours at $0.506/hr). Stopping 13.5 hours early
  saved about $7 and lost nothing measurable: gen 16's eval was flat and the league was at 50%.

## 1. Setup and run

| | |
|---|---|
| Engine | MageZero v0.2 (`danieljbrooks/MageZero` branch `draftzero`), XMage `v0.2-generalist` bundle |
| Config | `configs/exp2.yml`: budget 300, λ 0.95, every prior off, opponent mix 20% self / 70% past / 10% gen 0, opponent's hand hidden |
| Layout | 7 JVMs × 4 game threads, 112 games per generation (jobs of 4), one shared inference server, training batch 64 |
| Evals | every 8 generations: 100 paired games (200) against raw search at budget 300 |
| Pod | `206haai9na81zy`, Secure RTX 3090, 31.1 cores, $0.506/hr, 40 GB container disk |
| Run | started 2026-09-27 01:59:54 UTC; stopped 2026-09-28 21:36 UTC by hand, during gen 19's self-play (15 games in). Gens 0–18 complete. |
| Games | 2,673: 109 heuristic (gen 0), 1,964 network self-play and league (gens 1–18), 581 eval, 19 from gen 19 |
| Cost | pod uptime 43.7 h ≈ $22.10; balance afterwards $21.99 |

**Why it stopped early.** The budget stop was due at about 11:00 UTC on Sep 29, after gen 24's eval.
By gen 18 the evals, the league and the 17lands trend were all flat or declining (§2), so another
six generations were unlikely to change any goal.

## 2. Results

### 2.1 Strength

| | Games | Win rate | 95% CI |
|---|---|---|---|
| Eval, gen 0 (heuristic-trained net) against raw search | 92/194 | 47.4% | 41–54% |
| Eval, gen 8 | 98/196 | 50.0% | 43–57% |
| Eval, gen 16 | 95/191 | 49.7% | 43–57% |
| League, gens 1–8 (new network against an older one) | 357/608 | **58.7%** | 55–63% |
| League, gens 9–18 | 449/869 | **51.7%** | 48–55% |
| League, gens 13–18 | 262/520 | 50.4% | 46–55% |
| League, against the gen-0 network, all gens | 159/258 | 61.6% | 56–67% |
| League, against the gen-0 network, gens 9–18 | 67/109 | 61.5% | 52–70% |

- **Evals never moved beyond noise.** A 200-game eval resolves about ±7 points, and the three
  evals span 2.6 points.
- **The league did see learning,** then stopped seeing it. Each network in gens 2–8 beat its
  predecessors about 59% of the time. From gen 9 on that fell to about 50%: new networks were no
  better than the ones they replaced.
- The gen-0 network stays beatable (about 62% throughout), so the later networks did keep what
  they'd learned; they just stopped adding to it.

### 2.2 Training curves

![Run 1 learning curves](img/014-run1-learning-curves.png)

*Top row: training loss (last epoch of each generation); held-out loss and accuracy (the previous
network on each generation's new games, before training on them); strength. Bottom row: cumulative
17lands GIH ρ on commons (labels: commons counted); training targets; throughput and search
timeouts.*

| Gen | 1 | 2 | 4 | 8 | 10 | 12 | 14 | 16 | 18 |
|---|---|---|---|---|---|---|---|---|---|
| Held-out total loss | 0.907 | 0.834 | 0.788 | 0.679 | 0.637 | 0.598 | 0.588 | 0.550 | **0.553** |
| Held-out value loss | 0.087 | 0.078 | 0.094 | 0.079 | 0.083 | 0.086 | 0.095 | 0.093 | 0.090 |
| Training value loss | 0.082 | 0.074 | 0.066 | 0.054 | 0.052 | 0.051 | 0.050 | 0.049 | 0.049 |
| Priority accuracy (agrees with the search's choice) | 77.9% | 78.0% | 79.1% | 82.5% | 83.1% | 82.3% | 81.9% | 81.7% | 83.1% |
| Target accuracy | 26.5% | 26.6% | 29.6% | 34.0% | 30.8% | 36.2% | 38.4% | 40.8% | 38.1% |
| Yes/no accuracy (attack, block, kicker…) | 63.8% | 61.5% | 67.0% | 72.5% | 73.3% | 73.6% | 74.0% | 70.1% | 73.7% |

- **Policy heads:** the target head was still improving at the end (training loss 0.50 → 0.27;
  held-out accuracy 27% → 38–41%). The priority head plateaued at about 82–83% from gen 8.
- **Value head: the gap between held-out and training loss grew steadily from gen 5,** from about
  0.01 to 0.04–0.05. Held-out value loss was lowest at gen 5 (0.075) and reached 0.090–0.102 in
  gens 14–18, while training value loss fell to 0.049. The network was memorising each
  generation's positions rather than learning to evaluate new ones. The held-out series is noisy
  (it's one generation's games each time), but its trend matches the league flattening.
- **Why it matters here:** with every prior off, MageZero's search starts each option with equal
  prior and never reads the policy heads. The value head is the network's only influence on play.
  Better policy heads couldn't make better play, and a value head that had stopped generalising is
  the most likely reason the league flattened. That's a correlation in time, not a tested cause;
  §4 item 1 is the test.

### 2.3 17lands correlations

![Run 1 against 17lands](img/014-run1-17lands.png)

*Left: every common with at least 30 games in hand, each source's win rate relative to its own
mean. Middle: rank correlation with 17lands by stat and window (`tools/exp2_analysis.py stats`).
Right: the in-hand-weighted gap between bot and 17lands by color.*

**Method** (same as docs/013 §2.5):
- per-card win rates from both sides of self-play games and the current network's side of league
  games;
- 17lands figures from all 791,159 FDN Premier Draft games;
- Spearman ρ on commons with at least 30 games in hand (and 30 unseen, for GNS and IWD).

| ρ with 17lands, commons | Gens 1–9 (972 games) | Gens 10–18 (992 games) | Gens 1–18 (1,964 games) |
|---|---|---|---|
| GIH WR (in hand) | +0.22 (76) | +0.06 (76) | **+0.20** (84) |
| GNS WR (in deck, never seen) | +0.38 (72) | +0.45 (75) | +0.51 (84) |
| IWD (GIH − GNS) | +0.05 (72) | −0.05 (75) | +0.04 (84) |
| GP WR (in deck) | +0.43 (76) | +0.42 (76) | +0.48 (84) |

*Commons counted in brackets. Experiment #1's commons GIH figure was 0.28.*

- **GIH agreement fell in the second half,** +0.22 → +0.06 over the same number of commons. That's
  the same decline experiment #1 and run 2 showed.
- **What agreement there is comes from the decks.** GNS (the card was in the deck but never drawn)
  and GP track the whole deck's strength, and they correlate at 0.4–0.5 in both halves. IWD, the
  lift from actually drawing the card, is about zero. The bot has learned which of the 17lands
  decks win, but not which cards win games when drawn.
- **Colors show the same deck effect.** Relative to 17lands, the bot's blue commons run 3.8 points
  low and red 3.5 low, while white runs 2.3 high. Gain lands from particular two-color decks
  (Scoured Barrens +13.7 points above the bot's mean, Blossoming Sands +12.6) sit near the top for
  the same reason.
- **Absolute levels differ,** so compare relative to each source's mean: the bot's commons average
  53.0% in hand and 17lands' 54.2%. Rank correlations are unaffected.

**Where the bot agrees and disagrees** (points relative to each source's mean; full table in the
appendix):

| Card | 17lands rank of 91 commons | 17lands | Bot | Bot games in hand |
|---|---|---|---|---|
| Dazzling Angel | 4 | +3.6 | +8.9 | 315 |
| Luminous Rebuke | 5 | +3.5 | +9.3 | 263 |
| Helpful Hunter | 8 | +3.2 | +8.1 | 314 |
| Bake into a Pie | 1 | +3.8 | +4.5 | 372 |
| Burst Lightning | 2 | +3.7 | −0.3 | 294 |
| Refute | 6 | +3.3 | −4.7 | 323 |
| Gorehorn Raider | 14 | +2.4 | −8.4 | 175 |
| Fleeting Flight | 16 | +2.2 | −6.1 | 211 |
| Marauding Blight-Priest | 66 | −1.5 | +10.1 | 84 |

- Counterspells and combat tricks (Refute, Fleeting Flight) are where the bot is furthest below
  humans. Both pay off only when mana is held and a response is timed well. A plausible, untested
  reason is that the search handles instant-speed play poorly.

### 2.4 Bad targets

**What's counted.** A target is bad when a harmful effect hits the caster's own permanent, or a
beneficial one hits the opponent's (`tools/audit_targets.py`, as in docs/006 and docs/010).

**The in-loop metric didn't work.** `metrics.jsonl` classified 0 targets in 18 of 19 generations,
so these numbers come from auditing every JVM log after the run: 711 logs, 20,170 targeting prompts.

**Two corrections to the raw audit:**

1. **Player targets are mislabelled in v0.2's logs.** For example, PlayerB cast Burst Lightning
   with only the two players as targets and the log says `Targeting PlayerB`, yet PlayerA's life
   went 20 → 18. So "PlayerB burned its own face" is really an ordinary shot at the opponent.
   The raw audit counts 181 such false bad targets, 23% of its total. All player targets (330) are
   excluded below.
2. **Some "bad" targets are harmless:**
   - a tap ability giving an opponent's creature haste during your own turn (159);
   - Fleeting Distraction, a −1/−0 cantrip, on your own creature (83);
   - "target creature can't block" on your own creature during your attack (65).

   These 307 are counted separately below.

| Category (all agents, all generations) | Targets |
|---|---|
| Classified | 4,785 |
| of which player targets (label artifact, excluded) | 330 |
| Permanent targets | **4,455** |
| of which fine | 3,844 |
| harmless "bad" | 307 |
| **real mistakes** | **304 (6.8%, 95% CI 6–8%)** |
| Raw audit, for comparison: bad / classified | 792 / 4,785 = 16.6% |

(Of the 20,170 prompts, the rest weren't classifiable: 6,299 the card's own text restricts to one
side, 3,475 aren't targets at all, 4,607 effects the classifier can't read, and 1,004 targets with
unknown owners, mostly tokens.)

**Mistakes by who made them** (permanent targets):

| Agent | Mistakes | Rate | 95% CI |
|---|---|---|---|
| Raw search, gen-0 self-play | 20/179 | 11.2% | 7–17% |
| Raw search, in evals | 41/458 | 9.0% | 7–12% |
| Network, in evals | 25/420 | **6.0%** | 4–9% |
| Network, self-play and league | 130/2,113 | 6.2% | 5–7% |
| Older network (league opponent) | 88/1,285 | 6.8% | 6–8% |

| Eval | Network | Raw search |
|---|---|---|
| Gen 0 | 12/144 = 8.3% | 13/152 = 8.6% |
| Gen 8 | 7/131 = 5.3% | 16/158 = 10.1% |
| Gen 16 | 6/145 = 4.1% | 12/148 = 8.1% |

| Network's own choices | Gens 1–4 | 5–9 | 10–14 | 15–19 |
|---|---|---|---|---|
| Mistakes | 36/577 = 6.2% | 32/565 = 5.7% | 34/529 = 6.4% | 28/442 = 6.3% |

- **The network targets better than raw search.** In the same eval games it made about two thirds
  as many mistakes (6.0% against 9.0%, a two-sided p of about 0.09). In the gen-8 and gen-16
  evals, the gap was roughly 2 to 1. That suggests the value head learned something about
  targeting that raw search's built-in evaluation lacks.
- **There was no improvement across generations.** The network's rate was about 6% from gen 1 to
  the end.
- **The commonest mistakes:**
  - removal on its own creature: Stab 47, Witness Protection 41, Luminous Rebuke 20, Bake into a
    Pie 19, Goblin Negotiation 12;
  - help for the opponent's creature: Fake Your Own Death 31, Fleeting Flight 30, Sure Strike 24,
    Giant Growth 22.
- **Mistakes don't predict losing.** Half were made by the eventual winner (151 of 302 decided
  games). Many are low-stakes, for example when the only legal targets were the wrong side's.
  So this measures play quality, not game outcomes.

### 2.5 Human agreement

The measure is the network's top-ranked non-Pass play, and whether it's among the plays the human
made that turn. It uses the same 2,533 held-out 17lands decisions as docs/013 §2.4; chance is
48.1%.

| Network | Run 1 | Run 2 (docs/013) |
|---|---|---|
| Gen 0 | 59.3% | 60.7% (after gen 0's training) |
| Gen 1 | 58.7% | 58.4% |
| Gen 3 | 61.0% | 59.7% |
| Gen 8 | 62.7% | – |
| Gen 12 | 63.5% | 61.5% |
| Gen 18 | 62.5% | – |

- Self-play alone reached the level run 2 kept from its human pretraining. So docs/013's 61% is
  about what any network that plays Magic learns, not a lasting benefit of the human data.

### 2.6 Training targets and policy shape

- **Value labels were flat,** as Will's λ rule asks: median |v| 0.42–0.52 every generation, with
  9–15% of labels near zero (|v| < 0.1). λ 0.95 held up over the whole run.
- **Search policy targets were one-hot in 46–51% of states.**
- **The policy is Pass-heavy and over-confident.** This was measured on gens 1–4 only (from the
  training shards, 1,500 priority states per generation):
  - about 76% of the network's policy mass is on Pass;
  - the network puts 93–99% on its top action, against 83–85% in the search's own visit
    distribution.

  With priors off this doesn't affect play, but it matters for any future run that turns the
  priority prior on.

### 2.7 Operations

- **Throughput:** 39 games/hr in gen 1, rising to 65–74 from gen 11.
  - An ordinary generation took about 107 minutes: 112 games plus training.
  - An eval generation took about 250 minutes, of which the eval was 130–143.
- **The bottleneck was the inference server:**
  - one Python process at about 130% CPU served all 28 game threads;
  - the CPU quota averaged 59–72% used and the GPU 16–41% busy;
  - search timeouts stayed under 0.7% of evaluations.

  A replica-server fix (`jvm.servers`, commit c40714a) was built during the run but not applied
  to it.
- **Dropped games:** 52 of 2,016 scheduled self-play games (2.6%) and 19 of 600 eval games (3.2%).
  That matches run 2's XMage "Error in unit tests" failures: 242 such errors in self-play logs and
  95 in eval logs.
- **Disk:** at the initial rate the 40 GB disk would have filled around gen 14. A guard script
  (every 20 minutes) fixed it and held the disk at 28–30 GB:
  - capped server logs at 50 MB;
  - gzipped finished generations' JVM logs;
  - pruned the archived replay shards.
- **GPU memory** peaked at 23.5 of 24 GB in gen 14, at training batch 64. A batch-32 relaunch was
  staged but never needed; later generations peaked at 15–20 GB.
- **Alerts:** one, a harmless rsync "file vanished" on a scratch shard (gen 2). There were no
  crashes and no loop restarts.

## 3. Lessons learned

1. **With priors off, only the value head matters, so watch its held-out loss.** The policy heads
   improved to the end and none of it reached play. Held-out value loss started rising at about
   the same time the league flattened. It is the cheapest early warning we have.
2. **The value head overfits at 112 games per generation.** Training value loss fell from 0.089 to
   0.049 while held-out value loss rose from its gen-5 low of 0.075 to 0.090–0.102. More data per update, fewer passes, or regularisation is the first
   thing to fix before paying for longer runs.
3. **The league is the sensitive strength signal; 200-game evals every 8 generations are not.**
   The league showed learning by gen 3 and the plateau by gen 9. The evals couldn't separate any
   of the three networks. (docs/013 lesson 7, confirmed over the full run.)
4. **GIH agreement with 17lands is mostly deck agreement.** GNS and GP correlate at about 0.5 and
   IWD at about 0. A card metric that controls for the deck (IWD, or a per-deck baseline) is the
   honest measure of whether the bot values cards like humans do.
5. **Audit every metric against the raw logs at least once.**
   - The in-loop bad-target metric was dead all run.
   - The post-run audit's biggest "mistake" category (Burst Lightning to its own face) was a
     logging artifact, found only by reading life totals.
6. **The network makes fewer bad targets than raw search,** about 6% against 9%, even though it
   doesn't win more. That's evidence the value head learned something, and a cheap behavior
   metric worth logging every generation once fixed.
7. **Stop when the curves are flat.** Stopping at gen 18 saved about $7 (a third of the remaining
   budget) and lost nothing the report needed.

## 4. Recommended next steps

These add to docs/013 §4, which still applies: a fixed yardstick every 2–4 generations,
head-to-head at equal spend, an Elo ladder from league games, and the engine fixes.

1. **Fix the value head's overfitting before any longer run.** In rough order of cost:
   - train fewer epochs per generation, or stop on held-out value loss;
   - enlarge the replay window, so each update sees more distinct positions;
   - add weight decay or dropout on the value head;
   - log held-out value loss prominently on the dashboard.

   Test offline first on this run's saved replay shards (§5), in the style of docs/013 §2.3.
   That costs about $1 of GPU, against $20+ for a run.
2. **Apply the replica inference servers** (`jvm.servers`, c40714a). The CPU was 30–40% idle,
   waiting on one server. Measure games/hr with 2–3 replicas in a short smoke run before
   committing.
3. **Make the bad-target metric work in the loop:**
   - attribute each target to its own game's decks, as the audit does;
   - drop player targets until the log names them correctly;
   - tag the harmless categories;
   - fix the player-name logging in the MageZero fork, so face targets can be judged.
4. **Report IWD (or a deck-controlled GIH) as the 17lands headline,** alongside GIH, so deck
   strength doesn't masquerade as card judgement.
5. **Repo fixes from this run:**
   - skip the persist mirror on a same-disk pod;
   - quiet the server logs;
   - prune archived shards;
   - gzip finished JVM logs;
   - exclude `scratch/` from the rsync (the vanished-file alert);
   - stop uploading duplicate frozen checkpoints to HF (`models/FDN_exp2.genN/…`).

## 5. Data

| What | Where |
|---|---|
| Every checkpoint, gens 0–18 (`gen18.pt.gz` = `model.pt.gz` is the final network; SHA-256 `ac9b2d4f…bb59`) | HF `danbrooks/draftzero-checkpoints` under `2026-09-27_01-59-54/models/FDN_exp2/ver1/`; verified byte-for-byte against the pod before removal |
| Final network, local copy | `models/FDN_exp2/ver1/gen18.pt.gz` in the main checkout (gitignored) |
| `games.jsonl`, `metrics.jsonl`, `run.json`, `alerts.jsonl`, `deck_records.tsv` (through gen 18) | same HF prefix, top level |
| All logs: 711 JVM game logs, train and test logs, loop, watchdog, diskguard, the config; final `games.jsonl` and `metrics.jsonl` including gen 19's 19 games | HF `…/final/run1_logs.tar.gz` (104 MB) |
| Replay shards (`data/FDN_exp2/ver1`, training and testing) | HF `…/final/run1_training_data.tar` (5.4 GB) |
| Analysis scripts, the bad-target audit rows (`run1_targets_classified.jsonl`, every classified target), figures' sources | `bench_local/exp2_run1/` in the main checkout (laptop only) |

The pod was removed on Sep 28, within half an hour of the stop, after the uploads were verified.

## Appendix: run 1's commons against 17lands

Gens 1–18, 1,964 network games (487 self-play, 1,477 league), 86 commons with ≥ 30 games in hand,
sorted by the 95% Wilson lower bound on the bot's GIH win rate. The Δ columns are relative to each
source's mean over these commons (bot 53.0%, 17lands 54.2%). 17lands rank is among all 91 FDN
commons.

<details>
<summary>Full table (86 commons)</summary>

| # | Card | Color | Bot GIH | 95% CI | In hand | Bot Δ | 17lands GIH | 17lands Δ | 17lands rank |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Scoured Barrens | C | 66.7% | 57%–75% | 105 | +13.7 | 54.9% | +0.6 | 36 |
| 2 | Dazzling Angel | W | 61.9% | 56%–67% | 315 | +8.9 | 57.8% | +3.6 | 4 |
| 3 | Luminous Rebuke | W | 62.4% | 56%–68% | 263 | +9.3 | 57.7% | +3.5 | 5 |
| 4 | Helpful Hunter | W | 61.1% | 56%–66% | 314 | +8.1 | 57.4% | +3.2 | 8 |
| 5 | Infestation Sage | B | 59.7% | 54%–65% | 318 | +6.7 | 56.7% | +2.5 | 13 |
| 6 | Blossoming Sands | C | 65.6% | 53%–76% | 61 | +12.6 | 53.7% | -0.5 | 55 |
| 7 | Bake into a Pie | B | 57.5% | 52%–62% | 372 | +4.5 | 58.0% | +3.8 | 1 |
| 8 | Marauding Blight-Priest | B | 63.1% | 52%–73% | 84 | +10.1 | 52.7% | -1.5 | 66 |
| 9 | Vanguard Seraph | W | 60.4% | 52%–68% | 144 | +7.4 | 54.5% | +0.3 | 41 |
| 10 | Felidar Savior | W | 57.5% | 52%–63% | 285 | +4.5 | 57.4% | +3.1 | 10 |
| 11 | Sanguine Syphoner | B | 60.9% | 52%–70% | 110 | +7.9 | 53.3% | -1.0 | 59 |
| 12 | Banishing Light | W | 57.3% | 51%–63% | 260 | +4.3 | 57.4% | +3.2 | 9 |
| 13 | Inspiring Paladin | W | 59.0% | 51%–66% | 156 | +6.0 | 53.9% | -0.3 | 50 |
| 14 | Burglar Rat | B | 56.1% | 51%–61% | 353 | +3.1 | 56.3% | +2.1 | 18 |
| 15 | Cackling Prowler | G | 60.9% | 51%–70% | 92 | +7.9 | 54.1% | -0.1 | 46 |
| 16 | Squad Rallier | W | 58.3% | 51%–66% | 163 | +5.3 | 55.6% | +1.4 | 26 |
| 17 | Llanowar Elves | G | 57.8% | 50%–65% | 180 | +4.8 | 57.1% | +2.8 | 12 |
| 18 | Prideful Parent | W | 56.9% | 50%–63% | 232 | +3.9 | 54.8% | +0.6 | 38 |
| 19 | Gnarlid Colony | G | 59.3% | 50%–68% | 118 | +6.3 | 53.0% | -1.2 | 61 |
| 20 | Healer's Hawk | W | 56.4% | 50%–62% | 250 | +3.4 | 57.5% | +3.3 | 7 |
| 21 | Stab | B | 54.6% | 50%–59% | 401 | +1.6 | 57.9% | +3.6 | 3 |
| 22 | Evolving Wilds | C | 54.4% | 49%–59% | 371 | +1.4 | 54.6% | +0.4 | 40 |
| 23 | Hungry Ghoul | B | 54.7% | 49%–60% | 311 | +1.7 | 55.6% | +1.4 | 25 |
| 24 | Wind-Scarred Crag | C | 59.8% | 49%–70% | 82 | +6.7 | 53.7% | -0.6 | 56 |
| 25 | Pilfer | B | 56.4% | 49%–64% | 163 | +3.4 | 55.7% | +1.4 | 23 |
| 26 | Courageous Goblin | R | 57.3% | 49%–65% | 131 | +4.2 | 53.8% | -0.4 | 51 |
| 27 | Bushwhack | G | 56.6% | 49%–64% | 152 | +3.6 | 53.8% | -0.5 | 54 |
| 28 | Sower of Chaos | R | 61.8% | 49%–73% | 55 | +8.8 | 54.2% | +0.0 | 43 |
| 29 | Bite Down | G | 55.5% | 48%–63% | 173 | +2.5 | 55.6% | +1.3 | 29 |
| 30 | Eaten Alive | B | 53.2% | 48%–58% | 333 | +0.1 | 56.1% | +1.9 | 20 |
| 31 | Think Twice | U | 53.1% | 48%–59% | 305 | +0.1 | 56.6% | +2.4 | 15 |
| 32 | Cathar Commando | W | 53.8% | 47%–60% | 225 | +0.8 | 55.6% | +1.4 | 27 |
| 33 | Burst Lightning | R | 52.7% | 47%–58% | 294 | -0.3 | 57.9% | +3.7 | 2 |
| 34 | Bigfin Bouncer | U | 51.8% | 47%–57% | 355 | -1.2 | 57.1% | +2.9 | 11 |
| 35 | Make Your Move | W | 54.3% | 47%–62% | 162 | +1.3 | 55.3% | +1.0 | 32 |
| 36 | Goldvein Pick | C | 52.7% | 46%–59% | 222 | -0.3 | 53.6% | -0.7 | 58 |
| 37 | Tolarian Terror | U | 53.6% | 46%–61% | 168 | +0.6 | 52.2% | -2.0 | 72 |
| 38 | Dismal Backwater | C | 53.5% | 46%–61% | 155 | +0.5 | 56.3% | +2.0 | 19 |
| 39 | Fleeting Distraction | U | 51.7% | 45%–58% | 234 | -1.3 | 54.8% | +0.6 | 37 |
| 40 | Macabre Waltz | B | 53.3% | 45%–61% | 137 | +0.3 | 51.9% | -2.4 | 74 |
| 41 | Axgard Cavalry | R | 54.1% | 45%–63% | 111 | +1.0 | 52.1% | -2.1 | 73 |
| 42 | Campus Guide | C | 54.5% | 45%–64% | 101 | +1.4 | 51.4% | -2.8 | 78 |
| 43 | Witness Protection | U | 53.6% | 44%–63% | 110 | +0.6 | 52.8% | -1.4 | 64 |
| 44 | Icewind Elemental | U | 51.9% | 44%–59% | 162 | -1.2 | 53.8% | -0.4 | 52 |
| 45 | Hare Apparent | W | 57.1% | 44%–69% | 56 | +4.1 | 50.7% | -3.5 | 81 |
| 46 | Fake Your Own Death | B | 50.7% | 44%–57% | 207 | -2.3 | 55.8% | +1.6 | 21 |
| 47 | Gutless Plunderer | B | 51.2% | 44%–59% | 168 | -1.8 | 53.9% | -0.3 | 49 |
| 48 | Uncharted Voyage | U | 49.3% | 44%–55% | 294 | -3.7 | 56.3% | +2.1 | 17 |
| 49 | Elementalist Adept | U | 51.8% | 44%–60% | 139 | -1.2 | 52.6% | -1.6 | 68 |
| 50 | Vampire Soulcaller | B | 50.9% | 44%–58% | 175 | -2.2 | 54.0% | -0.2 | 47 |
| 51 | Apothecary Stomper | G | 55.7% | 43%–67% | 61 | +2.7 | 51.6% | -2.6 | 75 |
| 52 | Grow from the Ashes | G | 56.1% | 43%–68% | 57 | +3.1 | 48.1% | -6.1 | 90 |
| 53 | Ambush Wolf | G | 52.2% | 43%–61% | 115 | -0.8 | 55.3% | +1.1 | 31 |
| 54 | Refute | U | 48.3% | 43%–54% | 323 | -4.7 | 57.6% | +3.3 | 6 |
| 55 | Firebrand Archer | R | 53.8% | 43%–64% | 78 | +0.8 | 49.7% | -4.5 | 86 |
| 56 | Wary Thespian | G | 50.7% | 43%–58% | 152 | -2.4 | 55.4% | +1.2 | 30 |
| 57 | Dwynen's Elite | G | 52.5% | 43%–62% | 99 | -0.5 | 53.1% | -1.1 | 60 |
| 58 | Gleaming Barrier | C | 60.0% | 42%–75% | 30 | +7.0 | 47.8% | -6.4 | 91 |
| 59 | Thornwood Falls | C | 53.5% | 42%–65% | 71 | +0.5 | 53.8% | -0.4 | 53 |
| 60 | Giant Growth | G | 52.2% | 42%–62% | 90 | -0.8 | 54.2% | -0.0 | 45 |
| 61 | Jungle Hollow | C | 53.6% | 42%–65% | 69 | +0.6 | 52.8% | -1.4 | 65 |
| 62 | Strix Lookout | U | 48.1% | 42%–54% | 233 | -4.9 | 55.0% | +0.7 | 35 |
| 63 | Soul-Shackled Zombie | B | 49.1% | 42%–57% | 169 | -3.9 | 55.1% | +0.8 | 34 |
| 64 | Beast-Kin Ranger | G | 49.3% | 41%–57% | 144 | -3.7 | 55.2% | +1.0 | 33 |
| 65 | Lightshell Duo | U | 48.0% | 41%–55% | 198 | -5.0 | 55.7% | +1.5 | 22 |
| 66 | Bloodfell Caves | C | 50.5% | 41%–60% | 93 | -2.5 | 54.4% | +0.2 | 42 |
| 67 | Fleeting Flight | W | 46.9% | 40%–54% | 211 | -6.1 | 56.4% | +2.2 | 16 |
| 68 | Tranquil Cove | C | 48.5% | 40%–57% | 132 | -4.5 | 55.6% | +1.4 | 24 |
| 69 | Thrill of Possibility | R | 50.0% | 40%–60% | 86 | -3.0 | 51.5% | -2.7 | 77 |
| 70 | Swiftwater Cliffs | C | 50.0% | 39%–61% | 80 | -3.0 | 54.2% | -0.0 | 44 |
| 71 | Elfsworn Giant | G | 49.3% | 38%–61% | 73 | -3.7 | 52.6% | -1.6 | 67 |
| 72 | Goblin Surprise | R | 45.8% | 38%–54% | 144 | -7.2 | 52.6% | -1.7 | 69 |
| 73 | Gorehorn Raider | R | 44.6% | 37%–52% | 175 | -8.4 | 56.7% | +2.4 | 14 |
| 74 | Goblin Boarders | R | 46.3% | 37%–56% | 108 | -6.7 | 53.6% | -0.6 | 57 |
| 75 | Fanatical Firebrand | R | 45.9% | 36%–56% | 98 | -7.1 | 53.0% | -1.2 | 62 |
| 76 | Treetop Snarespinner | G | 44.4% | 36%–53% | 133 | -8.6 | 54.6% | +0.4 | 39 |
| 77 | Incinerating Blast | R | 44.7% | 36%–54% | 114 | -8.3 | 52.4% | -1.8 | 71 |
| 78 | Involuntary Employment | R | 45.6% | 36%–56% | 90 | -7.5 | 55.6% | +1.3 | 28 |
| 79 | Run Away Together | U | 45.8% | 35%–56% | 83 | -7.2 | 51.5% | -2.7 | 76 |
| 80 | Rugged Highlands | C | 47.3% | 35%–60% | 55 | -5.7 | 53.9% | -0.3 | 48 |
| 81 | Erudite Wizard | U | 42.4% | 33%–53% | 92 | -10.6 | 51.1% | -3.2 | 79 |
| 82 | Aegis Turtle | U | 48.5% | 33%–65% | 33 | -4.5 | 49.9% | -4.3 | 85 |
| 83 | Sure Strike | R | 42.4% | 31%–54% | 66 | -10.6 | 52.4% | -1.8 | 70 |
| 84 | Quick-Draw Katana | C | 41.0% | 27%–57% | 39 | -12.0 | 50.9% | -3.3 | 80 |
| 85 | Armasaur Guide | W | 40.0% | 26%–55% | 40 | -13.0 | 50.5% | -3.7 | 82 |
| 86 | Mocking Sprite | U | 36.9% | 26%–49% | 65 | -16.1 | 50.0% | -4.2 | 84 |

</details>
