# Runs

*Applies to v0.2.0. Kept current with the code; [architecture.md](architecture.md) explains the roles, and [docs/033](033-v0.2.0-refactor-plan.md) the plan behind them.*

A run is one training job: one config, one store, one manifest. This page says how runs are organised, where their data lives, how to use a published model, how to reproduce and publish a run, and which experiments are registered. Commands marked **(planned)** are not implemented yet.

## How a run is organised

- **Run id:** `<date>-<experiment>-<variant>`, for example `2026-10-20-e1-mlp`.
- **Config:** one YAML file in `configs/runs/`, usually extending `configs/base/player.yml`. It names the run (`run:`), the release it starts from (`code: v0.2.0`, or `dev`), the seed, the mode (`async` or `sync`), the budget and the store. Unknown keys are an error. `configs/smoke.yml` is the smallest run that exercises every role.
- **Store:** where the run's files go while it runs: a local folder (`runs/<id>`) or a Hugging Face Storage Bucket (`hf://buckets/danbrooks/dz-<id>`). Machines without a token run their actors on a local folder of their own, and `dz relay` carries files between it and the store over ssh.
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
stops at its budget or `dz stop`, 1 when a role failed, 2 at `--max-hours`. Starting it again on the same store
resumes the run.

The controller is the only writer of `control.json`, `events.jsonl` and the status page. Each version it opens one
evaluation job per opponent, split into shards of 10 pairs that it assigns to the live evaluators; a shard whose
evaluator goes quiet moves to another, and whichever evaluator uploads a job's last shard writes its `summary.json`
and the `eval.score` metric. `paused` and `budget.*` changes apply at once (otherwise a pause could never be lifted,
since no version arrives while actors are paused); every other live change waits for the next version.
`budget.max_hours` and `dz stop` stop the run at once; `max_versions` and `max_games` first pause the actors, then
stop once every evaluation job is closed.

For several machines, start each role on its own, against a shared store:

```bash
dz controller --config configs/runs/X.yml --store hf://buckets/danbrooks/dz-X
dz learner    --config configs/runs/X.yml --store hf://buckets/danbrooks/dz-X
dz evaluator  --config configs/runs/X.yml --store hf://buckets/danbrooks/dz-X
dz actor      --config configs/runs/X.yml --store hf://buckets/danbrooks/dz-X --machine pod1
```

A machine without a Hugging Face token runs its actor against a local folder, and a relay beside the store moves files over rsync: it pushes `config.yml`, `control.json` and the newest versions to each host and pulls complete shards, heartbeats and metrics back. With a bucket store, the relay keeps a local mirror folder and is the only process holding the token. `configs/smoke-distributed.yml` describes this path.

```bash
dz relay configs/runs/X.yml --host pod1 /root/dz/run "root@1.2.3.4 -p 40066" --once   # first sync
dz actor --config /root/dz/run/store/config.yml --store /root/dz/run/store --machine pod1   # on pod1
dz relay configs/runs/X.yml --host pod1 /root/dz/run "root@1.2.3.4 -p 40066"          # until the run stops
```

RunPod pods for such actors (Community Cloud GPU pods bought for their vCPUs; CPU pods on Secure Cloud with
`--secure`): `dz pod offers`, `dz pod up --name dz-a1 --gpu ID --max-price 0.25 --max-hours 2` (both caps are
required; it prints the cost guard, asks, arms the pod to terminate itself after `--max-hours`, then copies the
engine, cards, decks and repo and bootstraps a venv), `dz pod list`, `dz pod balance` and `dz pod down ID`. Billing
stops only at `dz pod down`. The flow is in the `draftzero.workers.runpod` docstring; no real pod has run it yet.

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

A sync-mode run replays exactly: the same seeds and versions give byte-identical games. An async run reproduces statistically, because which version an actor plays depends on timing.

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
| E1 | **Baseline self-play.** Does the 0.2.0 actor-learner, with decklists hidden (`:closedlist`), train a width-128 MLP player from scratch that keeps improving, as docs/031's loop F did? | Within 10 versions of loop F's size (1,200 games at 200 simulations each), the newest version beats `nonet` at 100 simulations by ≥ 100 Elo (about 64%, 1,000 paired games), and strength is still rising | `configs/runs/e1-baseline-mlp.yml` | v0.2.0 | planned | — |
| E2 | **Actor and learner budgets.** Which mix of simulations, recorded share, replay ratio and window gives the most strength per CPU-hour? Arms: the loop-F control (200 simulations, every decision recorded); playout-cap randomisation P1 (full 200 / cheap 40, full share 0.5) and P3 (500 / 100, 0.25); actor simulations 25, 50, 100; replay ratio and window | An arm beats the control head to head by ≥ 20 Elo at equal CPU-hours (1,000 paired games at 100 and at 25 simulations, interval above 0); otherwise loop F's budget stays the default. The playout-cap arms use the fork's `pcr` keys (`5fd039e`) | `configs/runs/e2-p0-control.yml`, `e2-p1-pcr-200-40.yml`, `e2-p3-pcr-500-100.yml`; the simulation, replay-ratio and window arms (planned) | v0.2.x | planned | — |
| E3 | **Learn deckbuilding.** Does docs/032's builder, trained on the loop with E1's player (frozen, then alternating), still beat human decks with hidden decklists, and by more with a stronger player? | On the sealed test pools (used once), its decks beat the human decks from the same pools in ≥ 60% of 2,000 games (docs/032: 63.8% with open lists), and the alternating arm beats the frozen arm on val | `configs/runs/e3-builder.yml` (planned) | v0.2.x | planned | — |
| E4 | **Learn drafting.** Can a picker learn from the builder's valuation of the pools it drafts, in gorge's `draftlab` pods? | In 8-seat pods with the same builder and player, decks from the learned picker beat decks from `rating-colour@1` drafters in ≥ 55% of 1,000 paired games | `configs/runs/e4-drafter.yml` (planned) | v0.2.x | planned | — |
| E5 | **Joint training.** Do player, builder and drafter learn better together from the same games than in sequence (E1, then E3, then E4)? | At equal total compute, the joint run's agents beat the sequential run's in ≥ 55% of 1,000 paired games over whole events (draft, build, play) | `configs/runs/e5-joint.yml` (planned) | v0.2.x | planned | — |

## Published runs

None from v0.2.0 yet. Each published run gets a row here: run id, experiment, release, Hugging Face repo, report doc.
