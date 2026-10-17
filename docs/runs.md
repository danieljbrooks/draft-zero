# Runs

*Applies to v0.2.0. Kept current with the code; [architecture.md](architecture.md) explains the roles, and [docs/033](033-v0.2.0-refactor-plan.md) the plan behind them.*

A run is one training job: one config, one store, one manifest. This page says how runs are organised, where their data lives, how to use a published model, how to reproduce and publish a run, and which experiments are registered. Commands marked **(planned)** are not implemented yet.

## How a run is organised

- **Run id:** `<date>-<experiment>-<variant>`, for example `2026-10-20-e1-mlp`.
- **Config:** one YAML file in `configs/runs/`, usually extending `configs/base/player.yml`. It names the run (`run:`), the release it starts from (`code: v0.2.0`, or `dev`), the seed, the mode (`async` or `sync`), the budget and the store. Unknown keys are an error. `configs/smoke.yml` is the smallest run that exercises every role.
- **Store:** where the run's files go while it runs: a local folder (`runs/<id>`), a folder on a RunPod network volume that every pod of the run mounts (`/workspace/runs/<id>/store`, docs/035), or a Hugging Face Storage Bucket (`hf://buckets/danbrooks/dz-<id>`). Machines without a token run their actors on a local folder of their own, and `dz relay` carries files between it and the store over ssh.
- **Manifest:** `manifest.json`, written at the start: the DraftZero version, commit, branch and dirty flag; the config and its hash; the engine (gorge fork commit, binary SHA-256, build record, card lock, census setting); with `:closedlist`, the belief corpus (decks, pool, split and its hash); Python, torch and numpy versions; the machine; the seed. `dz manifest CONFIG` prints the same record for this machine.

Every role writes into the same layout:

```
config.yml, manifest.json                  the run as launched
control.json                               which versions play, evaluation jobs and their shard assignments,
                                           stop and pause flags
status.md, status.json                     the controller's status page: versions, games, evaluations, machines
events.jsonl                               every plan change: time, path, old and new value, who, at which version
plan/requests/<uuid>.json                  requested changes, waiting for the controller
plan/rejected/<uuid>.json                  requests the controller refused, with the reason
versions/player/v0001/                     model.pt, model.dzgw, meta.json (parent, SHA-256s, the shards it trained on)
versions/player/latest.json                the newest version
shards/<actor>/000001/                     corpus.jsonl.gz (recorded decisions), games.jsonl, meta.json
evals/<job>/                               <shard>.games.jsonl (10 pairs each) and summary.json
heartbeats/<machine>.json                  liveness
metrics/<component>/<machine>/000001.jsonl metrics, schema draftzero-metrics/1
learner/                                   lease, ingested shards, status
relay/                                     the relay's last sync and upload ledger (relay runs only)
stop.request                               present = stop the run
```

A run with `builder.learn` (below) adds `versions/builder/v0001/` (`ens.pt`, `meta.json`) and `versions/builder/latest.json`, `builder/shards/<actor>/000001/` (`decks.jsonl`, `games.jsonl`, `meta.json`), `builder/learner/` (lease, ingested shards, status) and evaluation jobs `evals/builder-v0001-vs-best23/`.

Each metrics line has the time, run, component, machine, metric name, value and, when they apply, the version, step, sample size and 95% interval. Standard names include `throughput.games_per_hour`, `learn.policy_ce`, `learn.value_logloss`, `eval.score` and `eval.elo`.

### Running one

```bash
dz engine build --dir /work/gorge-fork                # build dzgorge at engine/gorge/GORGE_REF (configs' engine.root)
dz run configs/smoke.yml                              # every role on this machine; store runs/smoke
dz status runs/smoke                                  # versions, games, evaluations, machines
dz plan runs/smoke set learner.lr=3e-4                # a live change, applied at the next version
dz plan runs/smoke set paused=true                    # paused and budget.* apply at once
dz plan runs/smoke list                               # pending, rejected and applied changes
dz stop runs/smoke
dz report runs/smoke --out runs/smoke/report          # the standard plots
```

`dz run` checks that the engine and the deck pool are where the config says before it starts anything, and takes
`--store`, `--actors N`, `--max-hours H`, `--work-dir DIR` (caches and role logs, default `runs/<id>/work`) and
`--set PATH=VALUE` (any field, at launch; the store's `config.yml` records the result). It returns 0 when the run
stops at its budget or `dz stop`, 1 when a role failed, 2 at `--max-hours`, 130 on Ctrl-C and 143 or 129 on
SIGTERM or SIGHUP (the roles are stopped in every case). Starting it again on the same store resumes the run, on the
store's `config.yml` plus `events.jsonl`: the config given then may differ only in machine paths (`engine.root`,
`data.*` paths, `search.belief` paths, `player.init`, `store`, `tracking`); the controller refuses one that changes
any other field (use `dz plan set` for live ones) unless `dz run` is given `--allow-config-change`, which it passes
on to the controller (`dz controller` takes it too), and the changes are then recorded in `events.jsonl`. When a
role fails, `dz run` asks the controller to stop the run (`stop.request`), but when the controller itself fails it
stops the other roles directly and leaves no `stop.request` behind (one no controller acted on is withdrawn), so the
next `dz run` does not stop at its first tick. `dz status`, `dz plan` and `dz stop` refuse a path that is not a run's
store (no `config.yml` or `control.json`), so a typo creates nothing and reports no success; `dz plan STORE set ...
--before-start` queues a change in a store whose run has not started yet.

The controller is the only writer of `control.json`, `events.jsonl` and the status page. Each version it opens one
evaluation job per opponent, split into shards of 10 pairs that it assigns to the live evaluators; a shard whose
evaluator goes quiet moves to another, and whichever evaluator uploads a job's last shard writes its `summary.json`
and the `eval.score` metric. `paused` and `budget.*` changes apply at once (otherwise a pause could never be lifted,
since no version arrives while actors are paused); every other live change waits for the next version.
`budget.max_hours` and `dz stop` stop the run at once; `max_versions` and `max_games` first pause the actors, then
stop once every evaluation job is closed.

### Actor options from KataGo (opt-in)

Each is off by default, and off leaves the games, corpus and packs byte-identical to the fork without them (`5fd039e`).
Both are live, like `actor.pcr`: `dz plan STORE set actor.decided.high=0.9` reaches the actors at the next version.
They are self-play only: the evaluator's specs never carry them, so evaluation games are unchanged.

| Field | Default | What it does |
| --- | --- | --- |
| `actor.decided` | `{high: 0.0, n: 3, sims: 40, weight: 0.1}` | Decided-game cheap search (fork keys `:decided=H:decidedn=N:decidedsims=S:decidedw=W`). Once `n` searched root values in a row of a seat are all above `high` (or all below 1 − `high`), its full searches use `sims` simulations, no root noise and the most-visited move, until a value leaves that band. Those searches are recorded at training weight `weight`: 0 does not record them, 1 records them as full searches. `high` is in (0.5, 1); 0 is off. `sims` is at most `actor.sims`. KataGo uses 0.9, 3 and 0.1. |
| `actor.forced` | `{k: 0.0}` | Forced playouts with policy target pruning (fork key `:forced=K`). At the root of a full search, each candidate gets at least √(k · prior · root visits) visits, and the recorded visits are the pruned target. Under `:explore` the sampled move comes from the pruned visits too. 0 is off; KataGo uses 2. |

A decided search's record carries its weight into the learner's packs as `weight.npy` (`src/draftzero/nets/SPEC.md`
§2). `nets.train` then weights each position's policy and value loss by it, so a batch's loss is the weighted mean
(`--no-record-weights` ignores the weights). Shards without weighted records train exactly as before. The shard
`meta.json` records `decided` and `forced`. The actor logs `data.records_weighted` and `search.pruned_visit_share`
when they are on.

### Learner options from KataGo (opt-in)

Each is off by default, and off leaves training bit-identical (the tests in `tests/unit/nets/test_train.py` check
it). None is live: they are fixed for a run, but `learner.window_positions` stays live and caps the window.

| Field | Default | What it does |
| --- | --- | --- |
| `learner.surprise_weight` | `0.0` | Policy surprise weighting (`nets.train --surprise-weight`). Each state's policy loss gets weight (1 − x) + x · KL(target ‖ prior) / its batch mean, where the prior is the one the search used (`cand_prior`). The weights average 1 over a batch. KataGo uses 0.5. |
| `learner.value_surprise` | `0.0` | The same for the value loss (`--value-surprise`), with \|value target − the network's prediction\| as the surprise. KataGo uses 0.1. KataGo's own form needs the actor network's raw root value, which the shards do not record. |
| `learner.ema_decay` | `0.0` | A moving average of the weights (`--ema-decay D --export-ema`), bias-corrected so the starting weights drop out. Each evaluation also scores it and saves it as `ema.pt`. The published `model.pt` is the moving average when its holdout loss is lower than the raw weights'. `meta.json` then says `"weights": "ema"`, and its metrics are the average's. Try 0.999. |
| `learner.window` | `{kind: fixed, c: 150000, alpha: 0.75, beta: 0.4}` | `fixed` trains on the newest `window_positions` positions. `katago` uses KataGo's growing window over the N positions ingested so far: all N up to c, then c(1 + β((N/c)^α − 1)/α), at most `window_positions`. With these values that is ~400k positions at N = 1M and ~1.9M at 10M. `meta.json` records `window_cap`. |

The training log's `policy_ce` and `value_logloss` are always unweighted. Model selection uses the unweighted
holdout total.

### The deck builder on the loop (opt-in)

`builder.learn: true` trains docs/032's sealed deck builder (`draftzero.build`) on the same store, beside the player
or alone (`player.learn: false`). It is tabula rasa: the train pools are gorge draftlab sealed pools, made by the
fork's `dzgorge sealed-pools` (six FDN Play Boosters a pool, a pure function of the seed and the pool's index) and
cached once per machine; no 17lands pool or human decklist enters. `configs/smoke-sealed.yml` is the smallest such
run; it needs a `dzgorge` with `sealed-pools` (the fork's `draftzero-sealed` branch) until the pin includes it:

```bash
dz run configs/smoke-sealed.yml --set engine.root=DIR     # DIR: bin/dzgorge with sealed-pools, and .cards
dz builder-actor --config C --store S --machine pod1      # more builder actors, on any machine
```

| Role | How many | What it does |
| --- | --- | --- |
| Builder actor | many (`dz run --builder-actors M`, default 1) | Draws `pools_per_chunk` train pools, builds a deck of each kind in `kinds` from each with the current builder version (b best, t Thompson, p perturbed, l land-shifted, r random; version 0 is untrained: four random deck-shaped decks a pool), pairs decks of different pools, each pairing dealt twice with the seats swapped, and plays them with `dzgorge play -deckfile -pairs-file -single`. Idles once the version has `games_per_version` decided games |
| Builder learner | one (a lease) | Fits a `members`-network ensemble (Bradley–Terry on the results, bootstrap resamples) on the window's games, warm-started from the previous version, bounds the land count by `land_support`, publishes `versions/builder/vNNNN/ens.pt` |
| Builder evaluator | one or more | Plays each version's best decks against gorge's `best23@1` decks on `eval.pools` held-out pools (another seed): pools 2i and 2i+1 crossed, each crossing dealt `eval.pairs` times, both seat orders; `evals/builder-vNNNN-vs-best23/summary.json` and `eval.score` (opponent `best23`, tag `model: builder`) |

The pilot of every builder game is `builder.pilot` (a fixed `.dzgw`) when set, else the player version actors play
now (its `.dzgw`; an evaluation job keeps the version current when it opened), else the search without a network,
at `builder.sims` simulations (null: `actor.sims`; 0: gorge's heuristic bot) with `builder.closedlist` (default off,
as in docs/032). `budget.max_versions` counts every learning component's versions: the run drains once each has
reached it. In a builder-only run, plan changes wait for builder versions. No `builder.*` field is live yet.

| Field | Default | What it does |
| --- | --- | --- |
| `player.learn` | `true` | `false`: no player actors, learner or evaluator (needs `builder.learn`) |
| `builder.learn` | `false` | Start the builder's roles; off, nothing of the builder runs and control.json has no builder fields |
| `builder.kinds`, `random_share` | `btplr`, `0.15` | Decks per pool (also `c` cautious, `k` a random colour pair); `r` is a wide random build with probability `random_share`, else Thompson |
| `builder.members`, `epochs`, `lr`, `warm_lr`, `warm_start` | 5, 40, 3e-3, 1e-3, true | The ensemble and its training (`build.train.fit_games`; early stopping on 5% of pairings) |
| `builder.pools_per_chunk`, `games_per_chunk`, `games_per_version` | 50, 400, 10000 | A chunk's pools and games; decided games of a version before the next one (docs/032's generation: 10k–40k games, 600–800 pools) |
| `builder.window` | `{kind: versions, size: 4}` | The newest 4 versions' games, or `{kind: games, size: N}`: the newest shards holding N decided games |
| `builder.draws`, `castable`, `land_support` | false, false, 0.02 | docs/032's draw term, castability weighting and land range |
| `builder.sims`, `pilot`, `closedlist`, `workers`, `chunk_secs` | null, null, false, 2, 3600 | The pilot, and each actor's dzgorge workers and chunk time limit |
| `builder.init` | null | An `ens.pt` for version 0 (for example docs/032's released builder) instead of the untrained one |
| `builder.pools` | `{kind: draftlab, set: FDN, seed: 1, packs: 6, collation: arena, count: 5000}` | The train pools (pool indices 0..count-1 of the seed) |
| `builder.eval` | `{every_versions: 1, pools: 200, pairs: 1, seed: 2, sims: 100, opponent: best23, workers: 2}` | Held-out pools of another seed; 2 × pools × pairs games a version |

`best23@1` ranks cards by 17lands games-in-hand win rates, so it is a yardstick, never training data. docs/033's step
6 success criterion (docs/032's generation-1 result against the human decks) needs the 17lands val pools, which this
evaluation does not use.

For several machines, start each role on its own, against a shared store:

```bash
dz controller --config configs/runs/X.yml --store hf://buckets/danbrooks/dz-X
dz learner    --config configs/runs/X.yml --store hf://buckets/danbrooks/dz-X
dz evaluator  --config configs/runs/X.yml --store hf://buckets/danbrooks/dz-X
dz actor      --config configs/runs/X.yml --store hf://buckets/danbrooks/dz-X --machine pod1
```

A machine without a Hugging Face token runs its actor against a local folder, and a relay beside the store moves files over rsync: it pushes `config.yml`, `control.json` and the newest versions (`versions/player/` and, with `builder.learn`, `versions/builder/`) to each host and pulls complete shards (`shards/` and `builder/shards/`), heartbeats and metrics back. A builder actor on such a host (`dz builder-actor --config <root>/store/config.yml --store <root>/store --machine NAME`, a name of its own) generates its sealed pools there with the bundled `dzgorge sealed-pools` and `.cards`. With a bucket store, the relay keeps a local mirror folder and is the only process holding the token. `configs/smoke-distributed.yml` describes this path.

```bash
dz relay configs/runs/X.yml --host pod1 /root/dz/run "root@1.2.3.4 -p 40066" --once   # first sync
dz actor --config /root/dz/run/store/config.yml --store /root/dz/run/store --machine pod1   # on pod1
dz relay configs/runs/X.yml --host pod1 /root/dz/run "root@1.2.3.4 -p 40066"          # until the run stops
```

RunPod pods for such actors (Community Cloud GPU pods bought for their vCPUs; CPU pods on Secure Cloud with
`--secure`): `dz pod offers`, `dz pod up --name dz-a1 --gpu ID --max-price 0.25 --max-hours 2` (both caps are
required; it prints the cost guard, asks, arms the pod to terminate itself `--max-hours` after its create, the wait
for SSH included, then copies the engine, cards, decks and repo and bootstraps a venv; any failure removes the pod,
and `--keep-on-failure` keeps only a pod that is already armed), `dz pod list`, `dz pod balance` and `dz pod down
ID...` (it tries every pod and exits 1 if any terminate failed). Billing stops at `dz pod down` or the self-destruct.
The flow is in the `draftzero.workers.runpod` docstring; E1 and E2b ran on it (docs/034).

**On a RunPod network volume** ([docs/035](035-runpod-network-volumes.md)), every role opens the same folder on the
volume and no relay is needed: Secure Cloud pods in the volume's data center mount it at `/workspace`, and the run's
store is `--store /workspace/runs/<id>/store` on each of them. Global volumes do not work for this (a running pod
does not see later writes).

```bash
dz pod volume create --name dz-vol --size 10 --datacenter EUR-IS-1          # about $0.70 a month until `dz pod volume rm`
dz pod up --name dz-vh --cpu cpu3c --vcpu 4 --max-price 0.13 --max-hours 4 --secure --volume VOL     # the head
dz pod up --name dz-va --cpu cpu3c --vcpu 4 --max-price 0.13 --max-hours 4 --secure --volume VOL     # an actor pod
dz run configs/smoke-volume.yml --actors 0                                   # on the head, from /root/dz/draftzero
dz actor --config /workspace/runs/smoke-volume/store/config.yml --store /workspace/runs/smoke-volume/store --machine dz-va
```

The first pod of a commit uploads the bundle and leaves it on the volume, and later pods unpack it from there in
seconds. Start remote actors once `dz status STORE` shows the run up: an actor exits at once on a `control.json` that
says stop.

## Where data lives

| What | Where | Visibility |
| --- | --- | --- |
| A small run | a local folder, `runs/<id>` (gitignored) | local |
| A run in progress | a Hugging Face Storage Bucket, `danbrooks/dz-<id>` | private |
| A published run: config, manifest, events, weights, evaluations, metrics, model card | Hugging Face model repo `danbrooks/draftzero-<id>` | public when published |
| A published run's games and recorded decisions | Hugging Face dataset repo `danbrooks/draftzero-<id>-games` | public, optional per run |
| The code that produced it | the GitHub release its manifest names | public |
| v0.1.0-era public models | [`draftzero-fdn-exp1`](https://huggingface.co/danbrooks/draftzero-fdn-exp1) (experiment #1), `draftzero-fdn-gnn` (the GNN), dataset `draftzero-selfplay-games`, [`draftzero-fdn-sealed-builder`](https://huggingface.co/danbrooks/draftzero-fdn-sealed-builder) (docs/032) | public |
| v0.1.0-era checkpoints | `danbrooks/draftzero-checkpoints`: XMage-era runs, `exp4/`, `gnn/`, `sp028/`, docs/027's `gorge027/`. About 49 GB of regenerable or duplicated files were deleted on 14 October; the space returns once its history is squashed | private |
| Forge card scripts and gorge's compiled cards | never committed or published (GPL-3.0); `dz engine build` downloads them at the pinned ref | — |
| Pools and splits derived from 17lands | `assets/`, with attribution | CC BY 4.0 |

**Limits.** The account is free: 1,000 Hub API calls per 5 minutes shared by every machine using the token, 100 GB of private storage, and best-effort public storage. An actor uploads one batch per chunk, so 30 actors at one chunk per 5 minutes use about 30 calls a window. Private storage is the binding limit: prune shards older than the learner's window **(planned for buckets)** and keep only what gets published.

**Secrets.** Only machines that need the Hugging Face token hold it: the controller's machine and the relay. Actors need no tracker or Hub credentials. Pods can hold a fine-grained token limited to the run's bucket. Never commit a token.

## Using a published model

```bash
dz models pull hf://danbrooks/draftzero-2026-10-20-e1-mlp@v0042/model.dzgw    # into the local cache
dz play --a "az:sims=100:autopay:oppnodes:oppfull:morekinds:closedlist:topk=6:gonet=<path>/model.dzgw" \
        --b sbtactical --pairs 100
dz eval <path>/model.dzgw --opponents sbtactical,nonet
dz eval runs/2026-10-20-e1-mlp@v42 --opponents prev,sbtactical --pairs 50       # a version in a run's store
dz eval model:2026-10-20-e1-mlp/v0042 --gauntlet standard-v1                    # (planned)
```

`dz eval` plays the evaluator's protocol (paired games on the eval split at `eval.seed`, no exploration, the
config's search keys and `:closedlist`) against each opponent and prints a table and the JSON summary (`--out DIR`
keeps the games and `summary.json`). Opponents: `prev` and `v<N>` (with a store), `nonet`, `sbtactical`, `bot`,
`random`, `file:<path>.dzgw`, or any policy spec. A run's own evaluator plays only the opponents a config may name
(`prev`, `v<N>`, `sbtactical`, `bot`, `nonet`, `file:<path>.dzgw`).

From Python:

```python
from pathlib import Path
from draftzero.hub import resolve
from draftzero.engine.gorge import Gorge
from draftzero.engine.specs import PolicySpec

net = resolve("hf://danbrooks/draftzero-2026-10-20-e1-mlp@v0042/model.dzgw")   # a local Path
gorge = Gorge(Path("/path/to/gorge-fork"), Path("data/gorge/decks"), Path("data/gorge/pool.tsv"))
result = gorge.play(PolicySpec(sims=100, closedlist=True, topk=6, gonet=str(net)), "sbtactical",
                    pairs=100, seed=3131, out_dir=Path("out/play"), split="eval")
```

`draftzero.models.load("hf://…@v0042")`, which returns the network and refuses one whose encoding version differs from the engine's, is **(planned)**.

A `.dzgw` runs inside gorge's search. A `.pt` holds the architecture, config and weights for training and for networks that cannot run in-engine (the transformer and the GNN).

## Reproducing a run

`dz run reproduce hf://danbrooks/draftzero-<id>` **(planned)** will do the whole chain. By hand, from the run's `manifest.json`:

1. Check out the release it names (`code`, or `draftzero.commit` for a dev run) and `pip install -e .`.
2. `dz engine build`, then compare `dz engine version` with the manifest's `engine` record: fork commit, binary SHA-256 and card lock must match. With `:closedlist`, `dz manifest config.yml` must also show the manifest's `belief.corpus_hash`.
3. `dz run config.yml --store runs/<id>-repro`, with the manifest's config.

A sync-mode run is built to replay exactly: the same seeds and versions should give byte-identical games (no test checks this end to end yet). An async run reproduces statistically, because which version an actor plays depends on timing.

## Publishing a run

`dz publish <id>` **(planned)** uploads the chosen versions, manifest, events, metrics and evaluations to `danbrooks/draftzero-<id>`, writes the model card, uploads the games to `danbrooks/draftzero-<id>-games` if asked, and drafts the GitHub release. Hugging Face cannot copy from a bucket into a repo, so publishing re-uploads.

Rules, whether by command or by hand:

- **Dan confirms before anything goes public.** The command stops and asks; an agent never publishes on its own.
- Only a run launched from a release tag (`code: v0.2.x`) and a clean tree is publishable. `code: dev` runs are for development.
- After publishing, the release notes get a Runs section linking the Hugging Face repo and the report doc, and the run gets a row below.

## Experiment register

Every experiment gets a row **before** launch: its question, its success criterion and its config. After the run, the row gets its release, its status and its result. Writing the criterion first keeps reports and papers honest. The criteria below are proposals until Dan signs them off at launch. Experiments 1–5 from v0.1.0 are summarised in the [README](../README.md#status); their numbers are not comparable with 0.2.0's (gorge main turned the paid-mana census on, and the search now hides the opponent's decklist).

| ID | Question | Success criterion | Config | Release | Status | Result |
| --- | --- | --- | --- | --- | --- | --- |
| E0 | **The reference on the new engine.** How strong is B17 (docs/031's loop B, generation 17, trained on the old engine with the census off and open decklists) on the 0.2.0 engine, with the census on and `:closedlist`? | None: a measurement, the reference E1 and E2b are judged against | a `dz play` match: B17 at 100 simulations, `cpuct` 0.25, top 6, against `sbtactical`, 50 pairs, eval split, seed 3131 ([docs/034](034-v0.2.0-first-experiments.md) §9) | dev (fork `5fd039e`) | done 14 Oct | 61.0% (95% 54.0-68.0) against `sbtactical`, 50 pairs ([docs/034](034-v0.2.0-first-experiments.md) §3) |
| E1 | **Baseline self-play.** Does the 0.2.0 actor-learner, with decklists hidden (`:closedlist`), train a width-128 MLP player from scratch that keeps improving, as docs/031's loop F did? | Within 10 versions of loop F's size (1,200 games at 200 simulations each), the newest version beats `nonet` at 100 simulations by ≥ 100 Elo (about 64%, 1,000 paired games), and strength is still rising | `configs/runs/e1-baseline-mlp.yml`; launched 14 Oct as `configs/runs/e1-d1-pod.yml` (actors on a RunPod pod, learner and evaluator on d1) | dev (`6eac619`); v0.2.0 planned | done 14 Oct 10:10-19:02 PT (version 36, 9,936 games); **criterion narrowly not met** | Version 36 scores 63.4% (61.2-65.6) against `nonet` over 500 pairs, about 95 Elo against the ≥ 100 asked, and was still rising (52.0% → 57.4% → 61.7% → 64.0% on the same 25 deals). It scores 67.5% (62.4-72.6) against `sbtactical` over 100 pairs: +8.0 points (+1.5 to +14.5) over B17 and +3.0 (−2.9 to +8.9) over version 11, paired by deal on 50 pairs. Versions 10-22 scored 57.4% (51.1-63.7) against `nonet` over 175 pairs ([docs/034](034-v0.2.0-first-experiments.md) §4) |
| E2 | **Actor and learner budgets.** Which mix of simulations, recorded share, replay ratio and window gives the most strength per CPU-hour? Arms: the loop-F control (200 simulations, every decision recorded); playout-cap randomisation P1 (full 200 / cheap 40, full share 0.5) and P3 (500 / 100, 0.25); actor simulations 25, 50, 100; replay ratio and window | An arm beats the control head to head by ≥ 20 Elo at equal CPU-hours (1,000 paired games at 100 and at 25 simulations, interval above 0); otherwise loop F's budget stays the default. The playout-cap arms use the fork's `pcr` keys (`5fd039e`) | `configs/runs/e2-p0-control.yml`, `e2-p1-pcr-200-40.yml`, `e2-p3-pcr-500-100.yml`; the simulation, replay-ratio and window arms (planned) | v0.2.x | planned | — |
| E2b | **Playout-cap randomisation, small scale.** E2's three playout-cap arms (P0, P1, P3), each warm-started from B17 (E0), with their actors on one CPU pod at once: which gains most at equal CPU-hours? | As E2: an arm beats the control head to head by ≥ 20 Elo at 100 and at 25 simulations, interval above 0. Tonight's match is smaller than E2's 1,000 paired games, so it can only show a large difference. Predicted from docs/034's offline study D1: P1 ≈ control, P3 worse | `configs/runs/e2b-p0-control.yml`, `e2b-p1-pcr-200-40.yml`, `e2b-p3-pcr-500-100.yml` (`--set actor.workers=5`) | dev (`a3b0893`) | running 14 Oct 13:43-18:00 PT; head-to-head against B17 to follow | done 14 Oct 13:43-17:17 PT (3.5 h), head-to-head against B17 17:17-17:41 | Control 60.0% [55.4, 64.6] vs B17 at 100 sims, P1 55.5%, P3 54.0%; at 25 sims P1 55.5%, control 52.5%, P3 51.5% (100 pairs each). No arm beats the control; differences not significant ([docs/034](034-v0.2.0-first-experiments.md) §7) 
| E2c | **E2b's control and PCR 200/40 arms, continued** for 2.5 hours from their stores on a RunPod network volume (also the volume's live test, docs/035) | As E2b | E2b's configs with `--set actor.workers=2 --set player.init=/workspace/b17/best.pt --store /workspace/runs/e2b-<arm>/store --allow-config-change` | dev (`68b62fd`, fork `8aa1f37dc`) | done 14 Oct 18:41-21:11 PT, final matches 21:14-21:36 | P1 v10 against P0 v12: 50.2% (47.7-52.8) over 200 pairs at 100 simulations, 50.0% at 25. Against B17: P0 54.0%, P1 56.5% at 100 simulations. PCR neither helps nor hurts; neither arm clearly gained from 3× the data ([docs/034](034-v0.2.0-first-experiments.md) §7) |
| E6 | **Value grounding.** Does training the value partly on game results (`learner.value_blend` 0.5) instead of only on the search's root values (1.0, E1's recipe) restart the player's improvement, which stalled in E1 and E2c? From E1's final version, at equal CPU-hours | `e6-p1-value-half` beats `e6-p0-control` head to head by ≥ 20 Elo (about 3 points) over 500 pairs at 100 simulations, interval above 50%, and its score against `sbtactical` rises over the run | `configs/runs/e6-p0-control.yml`, `configs/runs/e6-p1-value-half.yml` (both: decided-game search on, PCR off, E1's replay of 2 epochs of the window a version) | dev (`f4d7d77`, fork `27ff42e4c`) | done 15 Oct 08:48-13:18 PT (4.5 h, ~1,900 games an arm, $3.04) | **Not met.** Value half v6 against control v6: 50.4% (48.4-52.4) over 500 pairs, +3 Elo. Neither arm improved on its start (49.0% and 50.0% against E1's version 36; 67.5% and 66.0% against `sbtactical`) ([docs/034](034-v0.2.0-first-experiments.md) §11) |
| E3 | **Learn deckbuilding.** Does docs/032's builder, trained on the loop with E1's player (frozen, then alternating), still beat human decks with hidden decklists, and by more with a stronger player? | On the sealed test pools (used once), its decks beat the human decks from the same pools in ≥ 60% of 2,000 games (docs/032: 63.8% with open lists), and the alternating arm beats the frozen arm on val | `configs/runs/e3-builder.yml` (planned); its first stage runs as `configs/runs/e3-builder-d1.yml` (frozen E1 pilot, tabula rasa, draftlab pools, on d1) | dev (`dddec19`, fork `8aa1f37dc`) | running on d1 since 14 Oct 16:17 PT | Builder v1 (5,000 games) 28.5% (21.8-35.2) against `best23` on 100 held-out pools; v2 (10,000) 47.0% (40.5-53.5); v3 (15,000) 50.5% (42.8-58.2); v4 (20,000) 55.8% (48.0-63.5); v5 (25,000) 53.0% (44.9-61.1); v6 (30,000) 50.5% (42.8-58.2); v7 (35,000) 56.0% (49.1-62.9); v8 (40,000) 56.5% (49.7-63.3); v9 (45,000) **60.5% (53.5-67.5)**, v10 (50,000) **59.5% (53.4-65.6)** v11 (55,000) **58.5% (51.0-66.0)**, v12 (60,000) **61.5% (54.9-68.1)** v13 (65,000) **59.5% (53.1-65.9)** v14 (70,000) **58.0% (50.7-65.3)**, v15 (75,000) **58.5% (51.6-65.4)** v16 (80,000) **59.0% (51.4-66.6)** v17 (85,000) **61.5% (54.8-68.2)** and v18 (90,000) **60.0% (52.7-67.3)**, all above 50%; v19-v22 (95,000-110,000) dipped to 53.5%, 55.0%, 54.5% and 51.5%, 6.0 points (−9.9 to −2.1) below v9-v18 paired by deal, and v23-v26 (115,000-130,000) recovered to 59.0%, 58.5%, 57.5% and 63.5% (57.1-69.9), level with v9-v18 paired (−0.0, −3.9 to +3.9); offline refits show a fresh fit on the 4-version window forgets (10-42%), and fresh fits on 65,000-105,000 games score 61-63%, after v1's 3-colour decks gave way to 2-colour ones ([docs/034](034-v0.2.0-first-experiments.md) §10) |
| E3b | **Builder without forgetting.** Does fitting every version fresh on the newest 100,000 games remove the risk of forgetting and lift the builder above E3's plateau? | Over the 10 versions after the switch, no paired decline against E3's versions 9-18 on the 100 held-out pools, and the mean above them | `configs/runs/e3b-builder-wide.yml` | dev | proposed 17 Oct (continues E3's store, or starts afresh) | — |
| E4 | **Learn drafting.** Can a picker learn from the builder's valuation of the pools it drafts, in gorge's `draftlab` pods? | In 8-seat pods with the same builder and player, decks from the learned picker beat decks from `rating-colour@1` drafters in ≥ 55% of 1,000 paired games | `configs/runs/e4-drafter.yml` (planned) | v0.2.x | planned. Since 14 October gorge main runs its own draft-bot RL program (`docs/superpowers/specs/2026-10-14-draft-rl-design.md` in gorge: a value-table drafter, behaviour cloning, then PPO with the az-c4 pilot, gated by a 16-deck gauntlet; `cmd/draftrl`, `limited/draftnet` `.dpkw` weights). E4 should build on it, not duplicate it. Its "B17" is a 17Lands-rated drafter, not docs/031's B17 network | — |
| E5 | **Joint training.** Do player, builder and drafter learn better together from the same games than in sequence (E1, then E3, then E4)? | At equal total compute, the joint run's agents beat the sequential run's in ≥ 55% of 1,000 paired games over whole events (draft, build, play) | `configs/runs/e5-joint.yml` (planned) | v0.2.x | planned | — |

## Published runs

None from v0.2.0 yet. Each published run gets a row here: run id, experiment, release, Hugging Face repo, report doc.
