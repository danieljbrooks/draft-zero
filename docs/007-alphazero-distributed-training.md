# AlphaZero-style distributed training

The goal of this document is to explore how other RL self-play implementations scale compute,
and to turn what they do into a plan for DraftZero.
- **Part 1** surveys the systems: KataGo first, then ELF OpenGo, AlphaGo Zero and AlphaZero, the
  volunteer-run projects, Minigo, a few general distributed-RL systems, and a single-machine
  reference implementation.
- **Part 2** uses them to lay out how this repo could scale from one pod to many. It ends in a
  recommended architecture to come back to if we decide to build it.

**Nothing here is implemented or decided.** It's a reference.

Claude wrote this on 2026-09-25 at Dan's request:
- "We" and "our" mean DraftZero, and "I" is Claude.
- Figures about other systems are their published numbers.
- Figures about DraftZero come from experiment #1's published run files and from the code.
- The last section lists what I checked directly.

## Summary

- **Every system splits the work the same way.**
  - Many self-play workers, which need only the latest weights.
  - One trainer, which needs only their games.
  - A shared store between them. Workers never talk to each other.
- **The systems that scaled all ended up asynchronous.**
  - AlphaZero dropped AlphaGo Zero's iterations and its gating test.
  - ELF OpenGo started synchronous and switched, which gave it over 5× the self-play throughput.
  - KataGo runs a synchronous loop on one machine and asynchronous processes across machines.
- **Two ratios decide whether scaling works, not the machine count:**
  - how many times each sample is trained on (*sample reuse*);
  - how many versions old the weights that played it are (*staleness*).
  - ELF, KataGo, OpenSpiel and OpenAI Five all control or measure them explicitly.
- **For DraftZero, a GPU pod is ~30 CPU cores of XMage search plus a small GPU for its own
  inference.** The GPU was 11–14% busy during network self-play.
  - Will's parallel JVMs are the unit of work inside a pod.
  - The pod is the unit of distribution, because inference has to stay on the same machine as its
    JVMs.
- **Recommended end state:** stateless actor pods, one learner GPU, and an S3-style bucket between
  them. Actors only make outbound calls.
  - It should reach roughly 25 pods without any distributed-systems framework.
  - Past that, the learner and operations need work.
- **Money sets the bill, not the pod count.** If nothing idles, cost per game is pod price ÷ games
  per pod-hour at any scale, so more pods buy wall-clock time.
- **One finding from exp #1's data: it trained each state about 11.5 times.** That comes from a
  150k-state window trained for one epoch per ~13k new states.
  - KataGo calls 4 conservative, and OpenAI Five targeted about 1.
  - I found no clear sign of overfitting, but that check is weak (§2.5).

---

## Part 1: How other self-play systems scale

### 1.1 KataGo

KataGo (David Wu) is the most relevant example. It cut AlphaZero-style compute costs sharply,
and it documents how its pipeline runs, in
[SelfplayTraining.md](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md).

- **Scale.**
  - The paper ([Wu 2019](https://arxiv.org/abs/1902.10565)) reports passing ELF OpenGo's final
    model after 19 days on fewer than 30 GPUs, about 50× less compute than comparable runs.
  - It now trains in public: volunteers contribute games to the
    [kata1](https://katagotraining.org/) run.
- **Five processes that share nothing but directories:**
  - self-play (C++) plays with the latest accepted net and writes data;
  - a shuffler mixes recent data into training files;
  - the trainer trains continuously and saves models;
  - an exporter converts saved models for the C++ engine;
  - an optional gatekeeper tests each new net against the current one.
- **Across machines, everything runs asynchronously.**
  - Each process runs on its own machine and never waits for another.
  - They share one base directory, which the doc expects on a fast networked filesystem.
  - Several self-play machines write into the same directories, one instance per machine.
- **On one machine, the same steps run in a loop.** `synchronous_loop.sh` runs the five steps in
  turn, which is what our `loop.py` does. The doc notes its defaults are lightly tested and weren't
  used for KataGo's main runs.
- **The trainer is throttled to the data.** `-max-train-bucket-per-new-data 4` allows at most 4
  training samples per new self-play sample. The doc calls 4 conservative, and warns that going
  much higher risks overfitting.
- **Hardware balance.** It suggests spending 4–40× more GPU on self-play than on training.
- **Many more game threads than cores.** Each thread waits on neural-net queries, so batching
  needs many games in flight.
- **The replay window grows with the run.** `shuffle.py` sizes it as a power law of the rows
  produced so far, with a 250k-row minimum by default. We use a fixed 150k states.
- **Gating is optional.** The doc says the loop works fine without it and is faster.
- **Everything can be killed and restarted** against the same directory, and the run continues.

**For us:** this is the design to copy. The one change is the shared filesystem: community pods
don't have one, so a bucket plays that role (§2.5).

### 1.2 ELF OpenGo

Facebook's reimplementation ([Tian et al., ICML 2019](https://arxiv.org/abs/1902.04522)) is worth
reading for its Appendix A, because it hit our timing problem.

- **Scale.** 2,000 V100 GPUs for self-play and 8 for training. A superhuman 20-block model came
  after 9 days. Its server supported up to 2,000 clients.
- **Slow games, fast models.**
  - They ran several self-play workers on each GPU to raise throughput, which pushed game latency
    to about an hour.
  - The trainer published a new model every 10–15 minutes, so models arrived faster than games
    finished.
  - For comparison, a DraftZero game at budget 300 is estimated at up to ~40 minutes.
- **They tried both designs:**
  - *Synchronous (AlphaGo Zero mode).* On a new model, self-play clients drop their current games
    and restart, and evaluation clients gate the model at 55%.
  - *Asynchronous (AlphaZero mode).* No locks and no evaluation clients. Clients always use the
    newest model and never abandon a game, so one game can be played partly by model A and partly
    by model B.
  - They switched from synchronous to asynchronous. That gave over 5× the self-play throughput
    and ~1.5× more games per training minibatch, which helped against overfitting. The cost is
    that a game no longer comes from a single model.
- **The ratio they watched: ~13 self-play games per training minibatch of 2,048.**
  - Going well below 10:1 hurt training, likely through overfitting.
  - So they stayed with one training worker instead of adding more.
  - They cite AlphaZero at ~30:1 and AlphaGo Zero at ~7:1.

**For us:** asynchronous self-play with mixed-model games is proven at scale. What limits
training is the ratio, not hardware.

### 1.3 AlphaGo Zero and AlphaZero

- **AlphaGo Zero** ([Silver et al., *Nature* 2017](https://www.nature.com/articles/nature24270);
  [free copy](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf)):
  - It ran three components asynchronously in parallel: optimization, evaluation and self-play.
  - Training ran on 64 GPU workers and 19 CPU parameter servers. Minibatches of 2,048 were
    sampled from the most recent 500,000 games.
  - The best player so far played 25,000 games per iteration, at 1,600 simulations per move.
  - A new network replaced it only after winning by a 55% margin (gating).
- **AlphaZero** ([Silver et al., 2017 preprint](https://arxiv.org/abs/1712.01815); *Science*
  2018):
  - 5,000 first-generation TPUs generated self-play, and 64 second-generation TPUs trained, for
    700,000 steps of 4,096 positions.
  - It kept one network that's updated continually. Self-play always uses the latest parameters,
    with no evaluation step and no best-player selection.

**For us:** AlphaZero's change is the one we'd make. Drop the per-generation barrier, and let
self-play use whatever the newest weights are. We never gated anyway.

### 1.4 Volunteer runs: Leela Zero, Leela Chess Zero, KataGo's kata1

- **[Leela Zero](https://github.com/leela-zero/leela-zero)** (Go) set out to repeat AlphaGo Zero's
  work as a public, distributed effort. Its `autogtp` client connects to the project server, plays in the
  background, and uploads results after each game.
- **[Leela Chess Zero](https://lczero.org/dev/wiki/neural-net-training/)** does the same for
  chess. Clients upload self-play games, a central server trains on them, and new networks are
  published for download.
- **KataGo's [kata1](https://katagotraining.org/)** run works the same way today.

**For us:** these are the closest match to many cheap rented pods.
- Clients are unreliable, come and go, and only make outbound HTTP calls.
- None of these systems needs a private network, which community pods don't have.

### 1.5 Minigo

Google's open reimplementation ([code](https://github.com/tensorflow/minigo);
[Lee et al., ICLR 2019 workshop](https://openreview.net/pdf?id=H1eerhIpLV)):
- It ran on Google Cloud with Kubernetes, with Cloud Storage as the shared file store.
- After 10 days on 800 Cloud TPUs, it played evenly against Leela Zero and ELF OpenGo.
- The paper is mostly about what made scaling hard: reproducibility, and the monitoring you need
  to see how hyperparameters interact.

**For us:** a bucket as the meeting point works at scale. Kubernetes suits an owned cluster, not
rented pods spread across hosts.

### 1.6 General distributed RL: IMPALA, SEED RL, OpenAI Five

These aren't MCTS systems. They're where the vocabulary and the measurements come from.

- **[IMPALA](https://arxiv.org/abs/1802.01561)** (Espeholt et al., 2018) is the standard decoupled
  actor–learner design.
  - Actors generate experience with slightly old weights, and a learner trains on it.
  - A correction called V-trace accounts for the lag.
  - It scales to thousands of machines.
- **[SEED RL](https://arxiv.org/abs/1910.06591)** (Espeholt et al., 2020) moves all inference to
  the learner's accelerators.
  - Actors only run environments, talking to the learner over an optimized communication layer.
  - It reports 40–80% lower experiment cost in its setups.
  - That fits one datacenter. With our pods on different hosts, it would put a network round trip
    inside every MCTS simulation, so it's the design we can't use.
- **[OpenAI Five](https://arxiv.org/abs/1912.06680)** (Berner et al., 2019), Appendix M, has the
  clearest measurements of the two ratios.
  - *Staleness*: how many versions older the generating weights are than the weights being
    trained. About 8 versions of staleness slowed learning significantly.
  - *Sample reuse*: the rate optimizers consume data relative to the rate rollouts produce it.
    Reusing data 2–3 times could halve the speed of learning, and 8 times could stop a competent
    policy from being learned at all. They targeted about 1.
  - The caveat: this is PPO, which learns from its own actions and is much more sensitive to both
    ratios than AlphaZero-style training on search results. KataGo's 4 and ELF's ratios are the
    better guides for us.

### 1.7 A single-machine reference: OpenSpiel's AlphaZero

[`alpha_zero.py`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/alpha_zero/alpha_zero.py)
runs the whole design on one machine, in about one file:
- N actor processes play games.
- M evaluator processes play the latest checkpoint against plain MCTS.
- One learner process collects games. For every `replay_buffer_size / replay_buffer_reuse` new
  states, it:
  - trains about one pass over the buffer;
  - saves a checkpoint;
  - tells the actors to load it.

So each state is trained about `replay_buffer_reuse` times.

**Exp #1's loop has exactly this shape.** It used a 150,000-state buffer, trained one epoch per
generation, and each generation added ~13,000 new states. That makes its implicit reuse about 11.5.

### 1.8 Side by side

| System | Self-play | Training | Loop | Gating | Where data meets | Training per new data |
|---|---|---|---|---|---|---|
| AlphaGo Zero | — | 64 GPUs + 19 CPU parameter servers | async components, iterations | yes, 55% | internal | ~7 games per 2,048-sample minibatch |
| AlphaZero | 5,000 TPUs | 64 TPUs | continual | no | internal | ~30 games per 4,096-sample minibatch |
| ELF OpenGo | 2,000 GPUs | 8 GPUs | sync, then async | sync mode only | ELF's server | ~13 games per 2,048-sample minibatch; below 10 hurt |
| KataGo | <30 GPUs in total (paper run) | — | async across machines, sync on one | optional | shared networked filesystem | ≤4 training samples per new sample |
| Leela Zero, Lc0, kata1 | volunteers | central server | async | — | HTTP server | — |
| Minigo | 800 Cloud TPUs, 10 days | — | — | — | Cloud Storage | — |
| OpenSpiel | N processes, one machine | 1 process | learner triggers on new data | no | in-memory queues | `replay_buffer_reuse` |
| **DraftZero exp #1** | **1 JVM × 18 threads, 1 L40S** | **same GPU** | **synchronous generations** | **no** | **local disk** | **~11.5 passes per state** |

### 1.9 What they agree on

1. **Don't make self-play wait for training.** Every system that scaled went asynchronous
   (AlphaZero, ELF, KataGo). A barrier idles the whole self-play fleet whenever the trainer runs.
2. **Workers exchange only weights and games,** through a store: a filesystem, a bucket, or an
   HTTP server. Workers never coordinate with each other.
3. **Inference runs next to the games.** It's batched across many concurrent games on the
   worker's own GPU, which is why KataGo runs so many threads and ELF put several workers on each
   GPU. SEED RL's central inference is the exception, and it needs a fast datacenter network.
4. **Training speed is tied to data arrival on purpose** (KataGo's flag, OpenSpiel's reuse, ELF's
   ratio). For ELF, adding trainers without adding data would have made training worse.
5. **Stale weights are tolerated but measured** (ELF's mixed-model games, OpenAI Five's staleness
   metric).
6. **Gating is optional and costs compute.** AlphaZero dropped it, and KataGo makes it optional.
7. **Every process can be killed and restarted from the shared store** (KataGo). That's what makes
   cheap, unreliable machines usable.

---

## Part 2: Scaling DraftZero

### 2.1 Where we start

**The loop.** [`loop.py`](../src/draftzero/loop.py) runs synchronous generations: play, then
train, then eval.
- Play runs one JVM job at a time, through MageZero's `launch_jvm`.
- The current net is served by one inference server on port 50052, and a frozen opponent on
  50053. Both are bound to 127.0.0.1.
- Training runs `run_train` on the newest 150,000 states.

**What exp #1 measured.** These come from its published `run.json` and `metrics.jsonl` unless
noted:

| | |
|---|---|
| Where the 31.9 h went | play 85%, training 11%, eval 4% |
| Median per generation (gens 2+) | 48 min of play, 6.2 min of training |
| New training states per generation | ~13,000 from 72 games (gens 10+): ~106 per player per game |
| Training speed | ~540 states/s on the L40S (150k states, 4,704 steps of batch 32, ~280 s) |
| Checkpoint | 224 MB gzipped: 83 MiB of weights, 167 MiB of optimizer state |
| Replay data | ~3 MB per game (6.9 GB over 269 shards) |
| GPU during network self-play | 11–14% busy (README) |
| Inference's share of a simulation | about half: 41 vs 84 simulations/s with and without the net (README) |
| Inference server ceiling | ~200 requests/s from one worker loop, seen only in single-JVM runs ([report](003-fdn-generalist-report.md) §4.1) |
| Cost | ~$0.011 per game at budget 96, on a $0.79/hr L40S |

**How our workload differs from Go:**
- **Self-play is CPU-bound.** XMage's rules engine and MCTS use the cores. The GPU only serves small
  batches.
- **Games are long.** At budget 300 a game is estimated at up to ~40 minutes (exp #2 pilot
  config), and it holds a thread the whole time.
- **Every game draws two random decks from 28,366.** That's why this is one model for a whole
  format, and why game length varies so much.

So **"more GPUs" really means more CPU cores.** Choose pods by usable (cgroup) cores per dollar and
by RAM per core, not by GPU.

### 2.2 The unit of work and the unit of distribution

**The unit of work: a JVM job.** Will's advice (ROADMAP, "Will's review of experiment #1") is
several JVMs per pod, each with 4 threads, ZGC and a large heap: up to 48 GB if RAM allows. A job
is self-contained:
- **in:** a game config (decks, search budget, λ, priors) and the checkpoint(s) it plays with;
- **out:** an HDF5 shard, a JVM log, and one summary row per game.

Once `play_generation` can run K jobs at once instead of one, the same jobs can run anywhere.

**The unit of distribution: a pod.** Inference can't leave the pod:
- The server listens on 127.0.0.1, and every MCTS simulation waits for its reply.
- Inference is already about half of each simulation's time, so any network latency would come
  straight out of search speed.
- So each pod runs its own inference: one or more servers per checkpoint in use, on its own GPU.

Whether that's one server per JVM or one shared per pod is ROADMAP question **B3**, which the
pilot measures. On a multi-GPU pod, pin servers to GPUs with `CUDA_VISIBLE_DEVICES`. Otherwise
every server lands on GPU 0.

**What crosses between pods is small and infrequent:**
- **down:** the weights, once per version. That's 83 MiB, or about half as much exported in fp16,
  which the server converts to anyway;
- **up:** about 3 MB of replay data per game.

**A v0.1 detail:** JVMs that share one XMage directory collide on its H2 card database.
[`tools/throughput_bench.py`](../tools/throughput_bench.py) gives each JVM its own copy, and v0.2
fixes the collision.

### 2.3 Four ways to scale, in order of effort

| | What | New code | Good up to | Weakness |
|---|---|---|---|---|
| **A. One bigger pod** | parallel JVMs on the biggest affordable box | Phase 2's parallel-JVM change | one machine (exp #1's 4×5090 test box had 163 usable cores) | limited by the biggest box in stock, and by its RAM |
| **B. Independent runs** | one run per pod: pilot arms, λ values, seeds | provisioning fixes only | any number | no single run gets faster |
| **C. Synchronous fan-out** | one run, with each generation's jobs spread over pods, then everyone waits | a remote job runner | ~5 pods | the generation barrier (§2.4) |
| **D. Asynchronous actor–learner** | one run: pods play continuously, and the learner trains as data arrives | actor supervisor, bucket, version tags, learner trigger | ~25 pods with one learner; more with learner work | more moving parts, and stale data has to be tracked |

**A and B come almost free after Phase 2, and they're the right first steps.**
- **A:** multi-GPU community boxes are often the cheapest cores. The report (§4.2) saw a 2×4090
  listed with 170 vCPU at $0.68/hr, though it sold out within minutes. A box with ~150 usable cores
  has ~6× exp #1's cores. Whether throughput follows is what the parallel-JVM change and the pilot
  will show.
- **B:** fits the trial-and-error phase Will predicted.

**D is the destination for a single large run.** C is only worth building as a stopgap.

### 2.4 Why the generation barrier stops working

On one pod, exp #1 lost 15% of its time to stages other than play. Spread the same loop over N
pods, and three things go wrong:
- **All N pods sit idle** while the learner trains and evaluates: 6.2 minutes per generation at
  exp #1's size.
- **A generation can't finish before its slowest game.**
  - At up to ~40 minutes per game, more pods give bigger generations, not more generations per hour.
  - 25 pods × ~30 threads is ~750 games in flight, so each generation becomes ~750 games instead of
    72.
  - That changes the learning recipe, not just the speed: Will's fixed-λ advice assumed small
    generations.
- **Slow games and dead pods hold everyone up**, unless a deadline cuts them off.

An asynchronous design removes all three. ELF measured the difference as over 5× the self-play
throughput.

### 2.5 Recommended architecture

```
learner pod (1 GPU)
  controller   run state, settings per version, opponent mix, eval schedule, budget
  learner      trains on the newest W states at reuse R, publishes version v+1
       ▲  shards, game rows, logs                │  weights for version v (~83 MiB), control.json
       │  tagged with versions, ~3 MB per game   ▼
object storage (S3-compatible bucket)
       ▲  upload                                 │  poll
       │  actors only make outbound calls        ▼
actor pod × N (any provider)
  supervisor       sizes itself from the cgroup, picks jobs, caches weights, heartbeats
  inference tier   S servers per version in use, on 127.0.0.1, one GPU
  K JVMs × 4 threads   XMage + MCTS; each job plays 2–4 waves of games
```

**Processes**

| Process | Where | What it does | Grows out of |
|---|---|---|---|
| Controller | learner pod | Run state; settings per version (curriculum: λ, budget, priors); opponent mix; eval schedule; budget and stall limits; 17lands stats and dashboards | `loop.py`, `watchdog.py`, `stats.py`, `dashboard.py` |
| Learner | learner pod, 1 GPU | Ingests finished jobs; trains at a fixed sample reuse on a replay window; publishes version v+1; logs its loss on fresh games | `train_generation`, MageZero `run_train` and `run_test` |
| Actor supervisor | each actor pod | Sizes itself from the cgroup quota; runs K JVMs; picks jobs; caches weights; uploads results; heartbeats | `play_generation`, `play_job`, `tools/throughput_bench.py`, `deploy/pilot_bench.sh` |
| Inference servers | each actor pod | Serve one checkpoint each to local JVMs, on 127.0.0.1 | MageZero `server.py` |
| JVMs | each actor pod | Play games, write shards | XMage + MageZero, unchanged |
| Store | S3-compatible bucket | The only shared state: control file, weights, shards, game rows, logs, heartbeats | new |
| Provisioner | laptop or controller | Rents pods by usable cores per dollar; checks CUDA; keeps a denylist; tears pods down | `workers/runpod.py`, `dz workers rank`, the exp #1 grab loop (report §4.2) |
| Monitor | controller | Replaces dead pods; enforces the budget; alerts; backs checkpoints up to Hugging Face | `watchdog.py`, `alerts.py`, `hfsync.py` |

Not needed at first:
- **A shuffler** (KataGo) only earns its place if loading data slows the learner down.
- **A gatekeeper** isn't needed: AlphaZero dropped it, and our strength evals already cover
  monitoring.

**Job types.** Every job is one JVM run with fixed versions.
- **self-play:** the newest version on both sides. Both sides are training data.
- **league:** the newest version against a past one. One side is training data.
- **gen 0:** the newest version against gen 0's net.
- **bootstrap:** heuristic search on both sides, before any net exists. It needs no GPU, so it can
  run on CPU-only hosts.
- **eval:** version v against a fixed baseline on the paired eval decks. It produces no training
  data.

**Store layout (sketch).**
```
runs/<run_id>/
  control.json                  newest version, settings, mix, eval assignments, stop flag
  models/v0042/weights.pt.gz    inference export: what actors load
  models/v0042/full.pt.gz       plus optimizer state, to resume the learner (keep every Nth)
  models/v0042/meta.json        parent version, states trained, settings
  jobs/<actor>/<job_id>/        shard(s), log.gz, games.jsonl
  heartbeats/<actor>.json       last seen, pod type, cgroup cores, jobs running
```
- **A job counts only once its `games.jsonl` exists**, and that file is uploaded last. So partial
  uploads are ignored, and retrying a job is harmless.
- **Every game row carries:**
  - which job produced it: `job_id`, `actor`, pod type;
  - which versions played: `agent_version`, `opponent_version`;
  - the game itself: decks, seed, winner, turns, start and end times;
  - staleness: the newest version at upload minus `agent_version`.

**The actor loop:**
```
until control.json says stop:
  job = this actor's assigned eval, if any; otherwise sample the mix (self / past / gen 0)
  make sure the job's weights are cached locally and a server is running for each version
  run a JVM for 2–4 waves of games (games = 2–4 × threads), so idle tails stay small
  upload shard, log and game rows (games.jsonl last); write a heartbeat
```
Each pod runs K of these at once, one per JVM slot, staggered so the pod never waits on a single
JVM.

**The learner loop:**
```
until stopped:
  ingest finished jobs and count the new training states
  once there are S new states: train R × S samples drawn from the newest W states,
    publish version v+1, update control.json, back up to Hugging Face every Nth version
```

**Policies to set explicitly.** These replace the generation as the knobs.

- **Sample reuse R: how many times each state is trained on.**
  - Exp #1's implicit R was ~11.5 (a 150k window, one epoch per ~13k new states).
  - KataGo's conservative cap is 4, and OpenAI Five's PPO target was ~1.
  - For AlphaZero and ELF, the game-to-minibatch ratios suggest roughly 1–2, if a Go game gives
    ~200 positions. That per-game figure is my assumption, not theirs.
  - Whether ~11.5 held exp #1 back is open. My check was weak. The previous model's loss on each
    generation's fresh games stayed close to its training loss (value 0.018 vs 0.015 at gen 33),
    which argues against heavy overfitting. But the training loss includes dropout, so the two
    aren't strictly comparable.
  - Make R a config value, log it, and test it rather than inheriting it.
- **Replay window W: how far back training reaches.** It's fixed at 150k states now. KataGo grows
  its window as a power law of total data. At many more games per hour, a fixed window turns over
  faster.
- **Version cadence S, and staleness.** Publishing every S new states sets how stale games are.
  - At 25 pods and 50 games per pod-hour, exp #1's S (~13k states) would mean a new version about
    every 5 minutes. A 40-minute game would then span ~8 versions, the level where OpenAI Five saw
    PPO slow down.
  - Start with versions every 15–30 minutes, log staleness per game, and tighten the cadence if
    staleness stays low.
- **Where versions change: per job or mid-game.**
  - Default: a JVM job keeps its weights for its whole run. That's simple, and every game has one
    clear version.
  - The alternative is ELF's asynchronous mode: servers swap in new weights between batches, so
    jobs never restart and data is fresher, but games mix versions.
  - Swapping needs a reload mechanism in MageZero's server, which would be an upstream
    conversation with Will.
- **The opponent mix is sampled per job, on the actor.** The 20 / 70 / 10 mix needs no central
  queue. "Past" opponents come from a league of every k-th version, so each pod caches a bounded
  number of checkpoints.
- **Evals are jobs pinned to a version.** The controller assigns them to specific actors in
  `control.json`, so no claiming protocol is needed. With ~750 thread slots, a 400-game eval
  finishes in one wave.
- **Settings come from the version, not the clock.** The curriculum (λ, budget, priors) resolves
  per version, and every job for version v uses v's settings. λ is applied inside the JVM, so an
  asynchronous loop doesn't change how value targets are computed.

**Failures, cost and security**

- **Actors are disposable.** A dead pod loses only its in-flight jobs, and heartbeats let the
  monitor replace it. Spot or interruptible machines are fine for actors.
- **The learner isn't.** Put it on the most reliable machine you have. Checkpoint it to the bucket
  every version, and to Hugging Face every Nth.
- **Hard limits at two levels:** each pod gets its own `DZ_MAX_HOURS` kill, and the controller
  enforces a total budget. That's the working agreements applied: every long job gets an external
  hard kill, and on RunPod we spend only the existing balance.
- **Credentials:**
  - Actors get bucket credentials limited to their own prefix.
  - Tokens are parsed, never sourced (report §4.3).
  - Nothing sensitive goes in a URL.
- **Hugging Face gets backups only.** A watchdog cascade once overloaded its API (report §4.3), and
  dozens of pods pushing shards there would run into its limits.
- **No Ray or Kubernetes.** Both assume nodes can reach each other on many ports. Community pods
  can't, and outbound-only actors don't need to.

**The upstream boundary.** The orchestration lives in draft-zero, per the repo boundaries in the
[ROADMAP](../ROADMAP.md). Only two pieces touch Will's engine, and each would be generic and opt-in
if he wants it:
- how servers pair with JVMs (B3);
- optionally, reloading weights in the server.

### 2.6 How far it scales, and what breaks first

**Throughput and wall clock.** The JVM-layout pilot measures games per pod-hour (G) at budget 300.
Until then, three placeholders:
- **G = 25:** exp #1's single-JVM rate, cut ~3× for budget 300.
- **G = 50.**
- **G = 100:** closer to what Will's numbers suggest. He sees ~1,600 games/day at budget 1,000 on
  32 cores, with single-deck games on different hardware.

| Pods | G = 25 | G = 50 | G = 100 |
|---|---|---|---|
| 1 | 25 games/hr (800 h for 20k games) | 50 (400 h) | 100 (200 h) |
| 5 | 125 (160 h) | 250 (80 h) | 500 (40 h) |
| 25 | 625 (32 h) | 1,250 (16 h) | 2,500 (8 h) |
| 100 | 2,500 (8 h) | 5,000 (4 h) | 10,000 (2 h) |

**Cost doesn't depend on the pod count.** Cost per 1,000 games = 1,000 × pod price ÷ G.
- At $0.50 per pod-hour, that's $20, $10 or $5 per 1,000 games for G = 25, 50 or 100.
- Exp #1 was ~$11 per 1,000 games at budget 96.
- Pod prices vary about 10× per usable core, so which pods you pick matters more than how many.
- RunPod's price quotes need care. On 2026-09-25 we found that a quote without a cloud filter
  blends community prices with secure-cloud specs. Pods also get less CPU than advertised, through
  the cgroup. Record each pod's quote, allocation and cgroup quota.

**What breaks first as the pod count grows:**
- **Money.** 25 pods at $0.50/hr is $12.50/hr. The remaining ~$61 of RunPod balance would last
  about 5 hours, so a big run depends on the compute-funding side track.
- **The learner.** It trained at ~540 states/s. Exp #2's mix trains on only one side of most games,
  so ~130 states per game.
  - At exp #1's reuse of 11.5, one learner keeps up with ~1,300 games/hr: 25 pods at G = 50.
  - At R = 4, it keeps up with ~3,700 games/hr.
  - The training code has headroom: batch size 32, and data loading in the main process
    (`num_workers=0`). A bigger batch changes the training recipe, though.
  - ELF's lesson is to keep R in range, not to add trainers.
- **Provisioning.** Community stock turns over in minutes (report §4.2). Grabbing 25 pods takes an
  automated loop. Mixed pod types are fine, because each actor sizes itself.
- **RAM per JVM.** Will suggests up to 48 GB of heap per 4-thread JVM, and a 117 GB pod fits only
  two of those. The pilot measures what's actually needed.
- **Operations.** At 25 pods, a dead pod is routine rather than an incident, so heartbeats and
  automatic replacement become necessary.
- **The store isn't a limit.** At 25 pods and G = 50, it's ~1 MB/s up, and ~3.5 MB/s down with a
  new version every 10 minutes. 20k games is ~60 GB of shards.

**Scale bands**

| Pods | What it takes |
|---|---|
| 1 big box | Phase 2's parallel JVMs (option A) |
| 2–5 | Independent runs (B), or D at small scale as the first real test of the design |
| 5–25 | D with one learner GPU and a bucket. This is the "easy" range |
| 25–100 | Learner work (bigger batches, parallel data loading, maybe a shuffler), automated provisioning and replacement, a tuned version cadence |
| 100+ | Everything above, plus multi-GPU training if R can't be held any other way. At this point it looks like KataGo's public run |

### 2.7 Where to get compute

| Role | Needs | Options | Notes |
|---|---|---|---|
| Actor pods | 16–200 usable cores, a small GPU with working CUDA, RAM for the JVM heaps | RunPod community or secure; other GPU marketplaces such as Vast.ai | Cheapest cores, but unreliable hosts. Check the cgroup quota and `torch.cuda.is_available()` on arrival (report §4.2) |
| CPU-only work: bootstrap games | Cores and RAM, no GPU | SaladCloud (already a ROADMAP side track), dedicated or auction CPU servers, spot CPU instances | Only jobs where neither side needs a net. CPU inference for network games is untested, and would compete with XMage for cores |
| Learner and controller | One reliable GPU and disk | RunPod secure, any GPU cloud | Losing it stalls the run |
| Store | An S3-compatible bucket | Cloudflare R2 (no egress fees), Backblaze B2, AWS S3, Google Cloud Storage | Hugging Face for per-version backups only |

**A RunPod-only variant.**
[RunPod's docs](https://docs.runpod.io/storage/network-volumes) say network volumes are Secure
Cloud only, must be attached when a pod is created, and can be corrupted by simultaneous writes
unless the application coordinates them.
- If one volume can be shared by several pods in a datacenter, it gives KataGo's shared-directory
  design almost as written. Check that it can before relying on it.
- KataGo's pattern handles the concurrent writes: each writer gets its own subdirectory, and files
  are renamed into place once complete.
- The costs are secure-cloud prices, and every pod in one datacenter.

### 2.8 Implementation path

1. **Parallel JVMs on one pod** (ROADMAP Phase 2).
   - Give `play_generation` a job-runner interface: run a job, return its shard, log and game rows.
     Start with a local backend that runs K jobs at once.
   - Tag every game row and shard with its agent and opponent version. For now, version =
     generation.
   - *Done when:* the pilot's best layout runs inside `loop.py` and beats 1 JVM × T threads in games
     per hour at the same budget.
2. **Independent runs on several pods.**
   - Let `RunPodWorker.provision` rent community pods. It hard-codes `--cloud-type SECURE` today.
   - Bring the grab loop, the CUDA check and the denylist into the repo, and give each pod its own
     hard time cap.
   - *Done when:* two pilot arms run at once on separate pods, each stopped by its own cap.
3. **Asynchronous actor–learner on 2–5 pods.**
   - Build the bucket layout and control file, the actor supervisor, and a learner that triggers on
     new states with an explicit R and W.
   - Key the stats and dashboards to versions, and show staleness and R on the dashboard.
   - *Done when:* at the same settings and total games, a 2-pod asynchronous run's strength and
     loss curves match the synchronous loop's within noise.
4. **10–25 pods.**
   - Heartbeats and automatic replacement, plus a controller-level budget.
   - Faster training if the learner can't hold R, and a league policy for past versions.
   - *Done when:* pods can die mid-run and be replaced, losing only their in-flight jobs, while the
     learner holds R on target.

Skip option C unless we need a 2–5 pod run before step 3 exists.

### 2.9 Open questions

| Question | Answered by |
|---|---|
| One server per JVM, or shared per pod (B3)? | Will, and the pilot |
| Games per pod-hour at budget 300 with parallel JVMs (G) | The JVM-layout pilot |
| Heap per JVM, and so RAM per core | The pilot |
| Sample reuse and window: keep ~11.5, or move toward KataGo's ≤4? | An experiment. Part of the exp #2 design, which Will reviews |
| Version cadence and staleness, and whether weights change per job or mid-game | Measurements from step 3 |
| Can CPU-only hosts run network self-play economically, with CPU inference? | A benchmark |
| More game threads than cores, as KataGo runs? Our threads wait on inference for ~half of each simulation | A pilot arm. Watch the search timeout rate |

---

## What I checked

- **DraftZero code:** `src/draftzero/loop.py`, `src/draftzero/workers/`, `tools/throughput_bench.py`,
  `deploy/pilot_bench.sh` and the configs.
- **MageZero:**
  - the v0.1 fork as installed: runner, server, train and model;
  - upstream `WillWroble/MageZero` main at `11a5974`: runner and server.
- **Exp #1's published run files:** `run/run.json` and `run/metrics.jsonl` in
  [danbrooks/draftzero-fdn-exp1](https://huggingface.co/danbrooks/draftzero-fdn-exp1). From them I
  computed:
  - the time split and the per-generation medians;
  - new states per generation;
  - training speed;
  - the reuse estimate;
  - the loss comparison on fresh games.
- **The checkpoint:** I loaded `gen33.pt.gz` to separate the weights from the optimizer state.
- **Sources I read in full or searched as text:**
  - the AlphaZero preprint, the AlphaGo Zero paper (UCL copy), ELF OpenGo, and OpenAI Five;
  - KataGo's `SelfplayTraining.md`, `python/train.py` and `python/shuffle.py`;
  - OpenSpiel's `alpha_zero.py`, and Leela Zero's README.
- **Abstracts only:** KataGo, IMPALA and SEED RL.
- **Project pages or search summaries:** Minigo (its README, and the paper's listing), Leela Chess
  Zero's training page, and RunPod's network-volume docs.
- **Not checked:**
  - the KataGo paper's exact split between self-play and training GPUs;
  - whether Leela Zero or Lc0 gate new networks;
  - whether one RunPod network volume can be attached to several pods at once.
