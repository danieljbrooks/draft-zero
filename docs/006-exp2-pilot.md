# Experiment #2 pilot: settings, single-JVM tuning, parallel JVMs

Run 2026-09-26 (UTC) on two RunPod pods. Both were RTX 3090, Secure Cloud, $0.50/hr, with a real quota of 31.1 cores and 116 GB. The pods report 256 cores and 1,007 GB from the host; see [005-runpod-tips.md](005-runpod-tips.md).

- **Pod A** ran the settings pilot (experiment 1).
- **Pod B** ran the throughput benchmark: experiment 3 (JVM layout), then experiment 2 (single-JVM tuning).
- **Engine:** experiment #1's v0.1 stack. That is MageZero `bcc76de` plus the generalist XMage build `5a32441c`. The v0.2 Java source isn't public (ROADMAP B1).
- **Settings:** search budget 300, search timeout 60 s, λ (`td_discount`) = 0.95, replay buffer 150,000 states, max game length 50 min. The configs are `configs/exp2_pilot_pod.yml` and `configs/curriculum_exp2.yml`.
- **Status:** all three experiments are complete.

## Conclusions

1. **Searches complete.** Of 3,588 searches, 0 hit the 60 s timeout. The longest took 19.3 s and the p95 was 6.7 s with the network on. Budget 300 is affordable, and the 60 s timeout never binds.
2. **The 50-minute game cap never binds either.** Mean game time was 5.9 min offline and 9.7 min with the network. Games lasted 11–32 turns (median 18.5, n = 32).
3. **λ = 0.95 gives well-spread value labels.** The median |label| is 0.41–0.49 and only 7–17% of labels are near 0. After one generation, the value head predicts the next generation's labels with a correlation of 0.81.
4. **Parallel JVMs win decisively offline.** The same 28 game threads as 7 JVMs × 4 threads gave **3.3×** the search throughput of 1 JVM × 28 (554 vs 167 sims/s). They also finished 25 games in 10 minutes against 0. This confirms Will's #1 issue: one big JVM wastes most of the machine.
5. **With the network on, inference becomes the bottleneck, and one shared server beats a server per JVM.** At 7 × 4:
   - a server per JVM reached 177 sims/s, only 1.6× over 1 × 28's 112;
   - one shared server reached 215 sims/s (1.9×).

   Even so, the network path runs at 2.6× below offline. Fixing the JVM layout alone isn't enough; the serving path is the next thing to fix. This also answers ROADMAP B3: prefer one shared server.
6. **Tuning a single JVM gets little.** On the same pod, 1 JVM × 14 threads (115 sims/s) matched 1 × 28 (112): the second 14 threads added nothing. Generational ZGC added 10% (123 vs 112). Neither comes close to what splitting the JVM gives.
7. **Bad targets are measurable now, and they happen.** 11 of 39 classifiable targets were bad (28%, n = 39). The side that made the bad target lost all 11 of those games. But 79% of targets can't be classified yet, so the classifier needs work before this number means much.
8. **The 17lands-style stats and color win rates work end to end, but a pilot has too few games.** GIH ρ against 17lands was 0.11 (gen 0) and 0.02 (gen 1) over 32 games, which is noise. At this scale that metric needs hundreds of games per generation.

## Experiment 1: settings pilot (Pod A)

The run had two generations of 16 games, plus a 2-game eval against the offline baseline. The model started from scratch (0 checkpoint rows at gen 0).

| stage | mode | games | wall | notes |
|---|---|---|---|---|
| gen 0 bootstrap | heuristic search, offline | 16 | 7.9 min | A won 9 |
| gen 0 eval vs offline | | 2 | 4.6 min | gen 0 won 2 of 2 (n = 2) |
| gen 0 train | 2 epochs, 3,219 states | | 22 s | |
| gen 1 self-play | network, 1 JVM × 16 threads | 16 | 13.9 min | A won 7 |
| gen 1 train | 1 epoch, 6,270 states | | 28 s | |

A generation of 16 games took 13–15 min. Container memory peaked at 70 GB of 116 with a 48 GB heap.

### Search completion (the 60 s timeout)

| log | searches | median s | p95 s | max s | timeouts |
|---|---|---|---|---|---|
| gen 0, offline, 1 × 16 | 1,712 | 0.75 | 4.2 | 8.2 | 0 |
| gen 0 eval | 227 | 0.45 | 2.7 | 4.4 | 0 |
| gen 1, network, 1 × 16 | 1,649 | 2.33 | 6.7 | 19.3 | 0 |

With the network on, a search takes about 3× longer (median 2.33 s vs 0.75 s), which matches experiment 3.

### Value labels at λ = 0.95

These are per training file; "near 0" means |label| < 0.1.

| file | states | mean | mean \|v\| | median \|v\| | near 0 | > 0 |
|---|---|---|---|---|---|---|
| gen 0, A | 1,614 | 0.13 | 0.52 | 0.49 | 8% | 64% |
| gen 0, B | 1,605 | 0.02 | 0.52 | 0.48 | 7% | 47% |
| gen 1, A | 1,544 | 0.11 | 0.45 | 0.41 | 17% | 57% |
| gen 1, B | 1,507 | 0.16 | 0.47 | 0.45 | 9% | 68% |

Labels cover the whole range of [-1, 1] (histogram in 0.2-wide bins, gen 1 A: 97 / 88 / 87 / 150 / 242 / 222 / 139 / 142 / 238 / 139). The gen 0 model evaluated on gen 1's data scored value correlation 0.81 and sign accuracy 0.81. Policy targets are one-hot in 46–47% of decisions, with 1.8 legal actions on average.

### Bad targets

`tools/audit_targets.py` resolves each game's decks from that game's own GAME_SUMMARY, and reads oracle text from Scryfall. Experiment #1 couldn't classify any target at all, so this is the first time the metric has worked.

| | targets | classified | bad | unclassifiable | owner unknown |
|---|---|---|---|---|---|
| gen 0 (heuristic) | 119 | 16 | 4 | 101 | 2 |
| gen 0 eval | 8 | 0 | 0 | 8 | 0 |
| gen 1 (network) | 99 | 23 | 7 | 69 | 7 |
| **total** | **226** | **39** | **11 (28%)** | **178** | **9** |

Every bad target was made by the side that went on to lose:

| gen | deck colors | effect | source | target (owner) |
|---|---|---|---|---|
| 0 | WB | harmful | Stroke of Midnight | own Hungry Ghoul |
| 0 | UB | harmful | Fleeting Distraction | own Dreadwing Scavenger |
| 0 | WG | beneficial | Giant Growth | opponent's Diregraf Ghoul |
| 0 | UB | harmful | Fleeting Distraction | own Infernal Vessel |
| 1 | UG | beneficial | Giant Growth | opponent's Elementalist Adept |
| 1 | BG | harmful | Hero's Downfall | own Hungry Ghoul |
| 1 | WG | harmful | planeswalker −3 destroy | own Healer's Hawk |
| 1 | BR | beneficial | "{T}: target creature gains haste" ×3 | opponent's Felidar Savior, Healer's Hawk, Elfsworn Giant |
| 1 | UGpB | beneficial | Giant Growth | opponent's Nine-Lives Familiar |

Some of these are defensible. Fleeting Distraction is a cantrip, so putting its −1/−0 on your own creature can be fine when there's no other target. Killing your own creature can be right with death triggers. Giant Growth on an opposing creature and a planeswalker killing your own Hawk are real mistakes.

Why 79% are unclassifiable:

- **Draw/discard and loot prompts aren't filtered out as non-target prompts.** Examples are "draw a card, then discard", "each opponent discards" and "{1}{U},{T}: draw, then discard". `NON_TARGET_PROMPT_RE` should drop them.
- **`classify_effect` misses common removal and tricks.** Examples are Fiery Annihilation, Bite Down, Felling Blow, Eaten Alive, Luminous Rebuke, Uncharted Voyage, bounce ETBs and tap effects.
- **Neutral effects.** Graveyard exile, equip and +1/+1 counters "on each other" are neither harmful nor beneficial in a way the rules can decide.

### 17lands-style stats and color win rates

`format.html` renders:

- GIH win rate per card next to the 17lands reference;
- color-pair win rates with confidence intervals;
- GIH ρ against 17lands for each generation.

The pilot saw 205 distinct cards drawn, but only 44 appeared in ≥ 5 games and 9 in ≥ 10. Single-color results over 32 self-play games (64 deck-sides) were:

| color | games | wins |
|---|---|---|
| W | 26 | 15 |
| U | 25 | 13 |
| B | 29 | 14 |
| R | 19 | 9 |
| G | 25 | 14 |

Pairs ranged from WU 5 of 5 to UB 0 of 7. With n of 3–8 per pair, none of these differ meaningfully. The player on the play won 20 of 32.

### Problems the pilot found

These need fixing before the full run.

1. **The watchdog treats a normal loop exit as a crash.** When the 2-generation loop finished, the watchdog logged "TRAINER GONE" and restarted it with `--resume`. The restart opened a new run directory, replayed gen 0 and overwrote `testing/*.hdf5`. I killed it by hand.
2. **The persist sync recursed into itself.** The rsync of `data/` into `data/persist/` copies the destination into itself (`data/persist/data/persist/…`). It fired a `sync_failed` alert from the first minute, so checkpoints were not being synced to the volume at any point.
3. **CPU% can't be trusted in a container.** `psutil.cpu_percent` reads the host's `/proc/stat`. That counts all 256 host cores, other tenants included, and MageZero's `ResourceMonitor` scales it to the quota. Neither pod produced a valid CPU-utilization figure. `tools/throughput_bench.py` now reads the cgroup counter instead (`cpuacct.usage` or `cpu.stat`); `ResourceMonitor` has the same flaw.

## Experiment 3: JVM layout (Pod B)

Every run used the same pod and the same 28 game threads, for 10 minutes each, at budget 300. The network runs used the gen33 checkpoint with fp16 inference, batch size 32 and 1 torch thread per server.

| run | layout | mode | sims/s | searches | timeouts | games in 10 min |
|---|---|---|---|---|---|---|
| off_1x28 | 1 × 28, 48 GB heap | offline | 167 | 1,143 | 0 | 0 |
| off_7x4 | 7 × 4, 11 GB heaps | offline | **554** | 3,651 | 0 | 25 |
| net_1x28 | 1 × 28 | network, 1 server | 112 | 688 | 2 | 0 |
| net_7x4_own | 7 × 4 | network, server per JVM | 177 | 1,168 | 0 | 1 |
| net_7x4_shared | 7 × 4 | network, 1 shared server | **215** | 1,412 | 11 | 7 |

- **Offline:** 7 × 4 beats 1 × 28 by 3.3×. The JVM layout alone decides that. No JVM died, since each JVM now has its own XMage directory and temp directory, and they start 5 s apart.
- **Network:** splitting helps 1.6× with a server per JVM and 1.9× with one shared server. The network path still costs 2.6× relative to offline at 7 × 4, and even at 1 × 28 it loses 33%.
  - Seven separate servers each batch requests from only 4 clients, so batches are tiny, and seven CUDA contexts time-share one GPU.
  - One shared server sees all 28 clients and batches better (+21% over per-JVM servers).
  - The shared server did produce 11 timeouts in 4,460 searches (0.25%), the only real count anywhere in the pilot. At that load it queues requests: a sign it is near saturation.
  - This answers ROADMAP B3: prefer one shared server, and optimize its batching next.
- **Cross-check from Pod A:** 1 × 16 offline ran about 228 searches/min, against about 110/min for 1 × 28 on Pod B. Adding threads to one JVM reduced total throughput. Experiment 2's 1 × 14 run confirms the plateau on a single pod.

Caveats:

- The sims/s figures undercount, because the JVM doesn't log a per-player line for allMana searches. The bias is the same across runs, so compare ratios, not absolute values.
- "Games in 10 min" is meaningless for 1-JVM layouts. With 28 threads in one JVM a game takes more than 10 minutes, so the window closes before any game ends.

## Experiment 2: single-JVM tuning (Pod B)

Same pod and settings as experiment 3, with the network on and one server.

| run | layout | GC | sims/s | searches | vs baseline |
|---|---|---|---|---|---|
| net_1x28 (baseline) | 1 × 28 | ZGC | 112 | 688 | |
| net_1x28_zgcgen | 1 × 28 | generational ZGC | 123 | 790 | +10% |
| net_1x14 | 1 × 14 | ZGC | 115 | 742 | +3% with half the threads |

- **One JVM plateaus at about 14 threads.** Doubling the threads to 28 added no throughput. So Will's issue is contention inside the JVM, not a lack of cores.
- **Generational ZGC is a small, free win.** Worth turning on (`--gc zgc-gen`, JDK 21+), but at +10% it is no substitute for splitting the JVM (+90% at 7 × 4 with a shared server).
- **No 1-JVM run finished a game in 10 minutes.** At 14–28 threads in one JVM, each game takes longer than the window.

## Recommendations for the full experiment #2

- **Run K × 4 JVMs, not one big JVM, with one shared inference server.** On a 31-core pod that is 7 × 4. Turn on generational ZGC too.
- **Next, optimize the shared server.** Candidates are larger batches, a batching wait window, and more server threads. With the network on, most of the 3.3× offline gain from parallel JVMs is still lost, and the shared server is already queuing (0.25% timeouts).
- **Keep budget 300 and the 60 s timeout.** Neither binds. A larger budget is affordable if it's wanted later.
- **Fix the watchdog's exit handling and the persist-sync recursion before any run long enough to need them.**
- **Before quoting a bad-target rate,** extend `classify_effect` and `NON_TARGET_PROMPT_RE` to cover the unclassified cases listed above.
- **Budget hundreds of games per generation** if GIH ρ and color win rates are meant to show a trend.

## Cost

$1.24 for both pods: the balance went from $60.56 to $59.32. Pod A was removed at 00:59 UTC and Pod B at 01:35 UTC, both after their results were saved locally.

## Reproducing

- **Pod A:** `deploy/pilot_setup.sh`, then launch with `configs/exp2_pilot_pod.yml` and `DZ_STALL_MINUTES=90`.
- **Pod B:** `deploy/pilot_setup.sh`, then `deploy/pilot_bench.sh 10`.
- **Analysis:** `tools/audit_targets.py <run>/jvm_*.log --deck-root data/deckgen/FDN_PremierDraft_wr60/top_player_FDN_decks`. It needs `tools/fetch_oracle.py` to have been run first.
