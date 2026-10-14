# DraftZero

[![Release](https://img.shields.io/github/v/release/danieljbrooks/draft-zero?include_prereleases&label=release)](https://github.com/danieljbrooks/draft-zero/releases)
[![Licence: MIT](https://img.shields.io/github/license/danieljbrooks/draft-zero?label=licence)](LICENSE)

Agents that play Magic: The Gathering limited tournaments end to end (drafting, building and playing) learned from self-play.

Limited players build a deck on the spot from the cards they open, so almost every game is played with a deck nobody has played before. Playing limited well means understanding a whole set of cards, not memorising one deck. DraftZero trains AlphaZero-style agents for Magic's Foundations set (FDN): a network guides a tree search, the search plays games against itself, and those games teach the next network. The games run on [gorge](https://github.com/adams-shaun/gorge), a fast Go rules engine.

**Version 0.2.0 is being built.** It turns the repository into one reproducible, config-driven training pipeline on gorge ([docs/033](docs/033-v0.2.0-refactor-plan.md)). Everything earlier, including the XMage pipeline and imitation learning, is at the [v0.1.0 release](https://github.com/danieljbrooks/draft-zero/releases/tag/v0.1.0).

## Status

The best result so far for each component. They come from v0.1.0, where gorge's search could see the opponent's decklist and the paid-mana census was off, so 0.2.0 runs re-baseline from experiment E1 ([docs/runs.md](docs/runs.md#experiment-register)).

| Component | Best so far | Doc | Model |
| --- | --- | --- | --- |
| Player | A width-128 MLP whose value learned from its own 200-simulation searches, inside gorge's search at 100 simulations and `cpuct` 0.25: 82.0% of 1,000 games against gorge's heuristic bot, about 270 Elo above it | [docs/031 §7](docs/031-gorge-recipe-search.md) | not published yet |
| Sealed deck builder | Trained only on its own games, its decks beat the human decks from the same pools in 63.8% of 2,000 games on 1,000 test pools (17lands heuristic 55.2%, imitation 53.0%) | [docs/032](docs/032-sealed-deckbuilding.md) | [draftzero-fdn-sealed-builder](https://huggingface.co/danbrooks/draftzero-fdn-sealed-builder) |
| Opponent-deck model | Version 0: gorge's closed-list worlds, whole decklists drawn from the run's own decks | [docs/033](docs/033-v0.2.0-refactor-plan.md) | — |
| Drafter | Not started; waits for experiment E4 on gorge's `draftlab` | [docs/033](docs/033-v0.2.0-refactor-plan.md) | — |

**Experiments so far (v0.1.0):**

| # | Experiment | Headline | Details |
| --- | --- | --- | --- |
| 1 | One agent for every FDN deck, self-play from scratch on XMage | Generation 33 beat plain search in 55.8% of 197 games (95% CI 49–63%), then plateaued: it tied generation 10 (49.0%) | [report](docs/003-fdn-generalist-report.md) · tag [`exp1-fdn-generalist`](https://github.com/danieljbrooks/draft-zero/tree/exp1-fdn-generalist) · [Hugging Face](https://huggingface.co/danbrooks/draftzero-fdn-exp1) |
| 2a | The same on MageZero v0.2, with more search per move | No gain over plain search (generation 16: 49.7% of 191 games); learning stopped around generation 8 | [report](docs/014-exp2-run1-report.md) |
| 2b | As 2a, starting from a network pretrained on 17lands players' decisions | The human start didn't help (45% against plain search at generation 8); heavy pretraining cost the network its ability to learn | [report](docs/013-exp2-imitation-report.md) · [setup](docs/011-exp2-run2-imitation.md) |
| 3 | Which tree search best matches top players' decisions, with hidden information respected | Searches that hide the opponent's cards agree with top players as well as one that peeks: 47–55% of decisions, against 47% for a rule heuristic and 44% for chance | [results](docs/016-search-benchmark-results.md) · [plan](docs/012-search-benchmark.md) |
| 4 | Imitation learning at scale: 12.1M decisions from 161k games by top 17lands players | A transformer and an MLP won 56% of 314 paired games against the heuristic bot at equal search, and 61–66% against it at 100 simulations with 300 or more of their own | [report](docs/019-imitation-learning-report.md) · [plan](docs/017-experiment4-scaling-up-imitation-learning.md) · [run log](docs/018-experiment4-run-log.md) |
| 5 | A sealed deck builder learned only from self-play on gorge | 63.8% of 2,000 games against the human decks from the same test pools; more search in training made its decks more human-like | [report](docs/032-sealed-deckbuilding.md) · tag [`sealed-builder-v1`](https://github.com/danieljbrooks/draft-zero/tree/sealed-builder-v1) · [Hugging Face](https://huggingface.co/danbrooks/draftzero-fdn-sealed-builder) |
| — | Self-play on gorge | 611,000 bot games an hour on 4 vCPUs (4,400 with search on both seats); a width-64 MLP in the search won 58.2% of 600 paired games against the search without it | [docs/026](docs/026-gorge-fdn-self-play.md) · [docs/027](docs/027-gorge-scaling-up.md) |
| — | A self-play recipe that scales | Cheap 25-simulation games stall; 100–200-simulation searches with every decision recorded make the policy learn (policy alone against the bot: 50.5% → 64.9%) | [docs/031](docs/031-gorge-recipe-search.md) |

## How it works

![The actor-learner loop](docs/img/033-actor-learner-light.png)

Actors play games on gorge with the latest networks and write shards of games and recorded decisions to a shared store. Learners train on a window of recent shards and publish new versions. An evaluator plays each version against a fixed set of opponents in paired games, and a controller decides which versions play and applies plan changes. The store is a local folder or a private Hugging Face Storage Bucket, and a relay carries shards and versions over ssh for machines without a token, so actors can run on any machine. Ten interfaces between the parts are rigid and need Dan's review to change. [docs/architecture.md](docs/architecture.md) has the details.

## Quick start

Needs Python 3.10 or newer, Go 1.25 or newer, make and git.

```bash
git clone https://github.com/danieljbrooks/draft-zero && cd draft-zero
pip install -e ".[dev,plots]"
pytest                                                # unit and contract tests, no engine needed

export GORGE=$HOME/gorge-dz
dz engine build --dir $GORGE                          # the gorge fork at engine/gorge/GORGE_REF, its cards, bin/dzgorge
dz run configs/smoke.yml --set engine.root=$GORGE     # a tiny run on the 80 sample decks: every role, a few minutes
dz status runs/smoke
dz report runs/smoke                                  # figures in runs/smoke/report

python tools/extract_decks.py --set FDN --format PremierDraft --min-winrate 0.60   # the full pool: ~57 MB from 17lands
python -m draftzero.data.decks                        # -> data/gorge/decks/*.json, data/gorge/pool.tsv
```

`configs/base/player.yml`, which every run config extends, expects the engine at `/work/gorge-fork`: build it there,
or set `engine.root` in your config or with `--set` as above. `dz engine version` prints the build record (fork
commit, binary SHA-256, card lock) that every manifest records. `dz --help` lists every command.

Optional extras: `tracking` (Trackio), `plots` (matplotlib, for `dz report`), `sealed` (pandas, scipy). CI
(`.github/workflows/ci.yml`) runs the unit and contract tests on Python 3.11 and 3.12; the engine and smoke tiers
(`pytest -m engine`, `pytest -m smoke`) need a built engine and run locally.

## Running experiments

- **Configs.** A run is one YAML file in `configs/runs/`, extending `configs/base/player.yml`. It is checked against the schema: unknown keys fail with their dotted path.
- **Launch.** `dz run CONFIG` starts every role on one machine (`--actors N`, `--max-hours H`, `--store S`, `--set PATH=VALUE`). For several machines, run `dz controller`, `dz learner`, `dz evaluator` and any number of `dz actor --machine NAME`, each with `--config CONFIG --store S`, against a shared store such as `hf://buckets/danbrooks/dz-<run>`. Machines without a token (RunPod pods from `dz pod up`, borrowed hosts) run actors on a local folder that `dz relay` keeps in step with the store over rsync.
- **Change the plan mid-run.** `dz plan STORE set learner.lr=3e-4` changes a live field (budgets, learning rate, window, evaluation schedule, pause). The controller applies `paused` and `budget.*` at once and anything else at the next version, logs each change to `events.jsonl`, and moves a rejected request to `plan/rejected/` with the reason. Fixed fields (engine pin, architecture, seed, splits) need a new run; `dz run fork RUN --at v12` **(planned)**.
- **Resume.** Start the roles again with the same config and store; they continue from what the store holds.
- **Watch it.** `dz status STORE` prints the controller's `status.md` (`--json` for `status.json`); Trackio mirrors the metrics when `tracking.trackio` is on. `dz report STORE` draws the standard plots. `dz stop STORE` ends the run.
- **Register first.** Every experiment gets a row in the [experiment register](docs/runs.md#experiment-register), with its question and success criterion, before it launches.

## Using trained models

```bash
dz models pull hf://danbrooks/draftzero-<run>@v0042/model.dzgw    # prints the local path
dz play --a "az:sims=100:closedlist:topk=6:gonet=<path>/model.dzgw" --b sbtactical --pairs 100
dz eval <path>/model.dzgw --opponents sbtactical,nonet            # a table and JSON; or a run's STORE[@vN]
```

`dz play` and `dz eval` take the engine, deck pool and search settings from `--config` (default
`configs/base/player.yml`; `dz eval STORE` uses the run's own `config.yml`).

In Python, `draftzero.hub.resolve("hf://…")` returns a local path, and `draftzero.engine.Gorge` plays it. A one-call `draftzero.models.load("hf://…")` is **(planned)**. [docs/runs.md](docs/runs.md) lists every published run.

## Reproducing a published run

`dz run reproduce hf://danbrooks/draftzero-<run>` **(planned)** checks out the release the run's manifest names, builds gorge at its pin, downloads the cards at its lock, verifies both digests and runs the config again. Until then, [docs/runs.md](docs/runs.md#reproducing-a-run) gives the same steps by hand. Sync-mode runs replay exactly; async runs reproduce statistically.

## Repository layout

| Path | What |
| --- | --- |
| `engine/gorge/` | The engine pin (`GORGE_REPO`, `GORGE_REF`) and `build.sh` |
| `src/draftzero/cli.py` | The `dz` command |
| `src/draftzero/config.py`, `manifest.py`, `metrics.py` | Run config schema, run provenance, metrics schema |
| `src/draftzero/engine/` | The gorge wrapper and policy strings |
| `src/draftzero/loop/` | Actor, learner, evaluator, controller, store and relay |
| `src/draftzero/nets/` | Player networks: data format, models, training, serving, export to `.dzgw` |
| `src/draftzero/build/` | The sealed deck builder |
| `src/draftzero/oppdeck/` | The opponent-deck belief corpus |
| `src/draftzero/data/` | 17lands decks, pools and splits |
| `src/draftzero/eval/` | Paired games, ratings, card statistics against 17lands |
| `src/draftzero/plots/` | The standard figures |
| `src/draftzero/workers/` | Local, ssh and RunPod machines |
| `configs/` | `base/`, `runs/`, the smoke configs |
| `assets/` | Small versioned inputs: deck splits, sealed pools, card tables, the 17lands reference, sample decks |
| `docs/` | Numbered, dated docs and their [index](docs/README.md), [architecture.md](docs/architecture.md), [runs.md](docs/runs.md) |
| `tests/` | `unit/`, `contract/`, `engine/`, `smoke/`; CI in `.github/workflows/ci.yml` runs the first two |
| `data/`, `runs/` | Generated and gitignored: decks, pools, downloads, local runs |

## Data and licences

- **This code:** [MIT](LICENSE), for every version of this repository.
- **17lands:** decks, pools and card statistics come from the [17lands public datasets](https://www.17lands.com/public_datasets), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Files derived from them (`assets/decks.tsv`, `assets/sealed/`, `assets/reference/`, `assets/sample/` and published releases) carry the same licence.
- **Forge card scripts:** GPL-3.0. gorge downloads them at its pinned `FORGE_REF` when it builds; they are never committed or published here.
- **gorge:** Apache-2.0.

## Related projects

| Project | How it relates |
| --- | --- |
| [gorge](https://github.com/adams-shaun/gorge) (dama) | The rules engine and search every 0.2.0 game runs on. Our fork [danieljbrooks/gorge](https://github.com/danieljbrooks/gorge) adds five commits on its main branch (branch `draftzero`) |
| [Forge](https://github.com/Card-Forge/forge) and dama's fork [adams-shaun/forge](https://github.com/adams-shaun/forge) | Forge's card scripts define what every card does; gorge compiles them from dama's fork |
| [mtg-kernel](https://github.com/jackmaiorino/mtg-kernel) | A Rust rules engine; docs/025 tried FDN self-play on it, and it may join later behind the same engine interface |
| [MageZero](https://github.com/WillWroble/MageZero) (Will Wroble) | AlphaZero-style training on XMage; DraftZero's v0.1.0 experiments 1–4 built on it |
| [XMage](https://github.com/magefree/mage) | The full-rules Java engine behind v0.1.0's experiments 1–4 |
| [17lands](https://www.17lands.com/) | Human decks, pools and card statistics: the data splits and the reference every card statistic is checked against |

## Docs

The [docs index](docs/README.md) lists every doc with its date, the version it applies to and its kind. [ROADMAP.md](ROADMAP.md) says what comes next, and [CHANGELOG.md](CHANGELOG.md) what changed in each version.

## Citing

```bibtex
@misc{brooks2026draftzero,
  author       = {Brooks, Daniel},
  title        = {{DraftZero}: self-play agents for Magic: The Gathering limited},
  year         = {2026},
  howpublished = {\url{https://github.com/danieljbrooks/draft-zero}},
  note         = {Version 0.2.0}
}
```

Cite the release a result came from: each run's manifest names it.

## Credits and licence

- **[17lands](https://www.17lands.com/)** and the players who share their data: every deck and human reference statistic.
- **dama ([adams-shaun](https://github.com/adams-shaun))** for gorge, its search and `draftlab`, for taking our patches upstream, and for lending r1, the GPU machine behind docs/024 and docs/031.
- **[Will Wroble](https://github.com/WillWroble)** for MageZero, its graph network and advice on training it.
- **[chrismaghuhn](https://github.com/chrismaghuhn)** for advice on compute and performance.
- **[Forge](https://github.com/Card-Forge/forge)** and **[XMage](https://github.com/magefree/mage)** for their rules implementations.

The code is [MIT](LICENSE). Data derived from 17lands carries CC BY 4.0.
