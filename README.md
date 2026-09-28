# DraftZero

One network that plays a Magic limited format with **any** deck. Every game draws both
decks at random from ~31.5k Foundations (FDN) Premier Draft decks built by top 17lands
players, and the agent is scored with 17lands-style statistics: card GIH win rate, rank
correlation against the public data, and per-colour deck records.

This is deliberately the opposite of [MageZero](https://github.com/WillWroble/MageZero),
which decomposes Magic into deck-local subgames. DraftZero uses MageZero as a library —
its RL framework, XMage bridge, trainer and evaluator — and adds the format-level parts.

## Experiments

| # | What | Result | Code | Artifacts |
|---|---|---|---|---|
| 1 | One agent for all of FDN: 2,507 games, 34 generations, one L40S, ~$28 | Beats raw search 110/197 (55.8%); plateaued by gen 10 | tag [`exp1-fdn-generalist`](https://github.com/danieljbrooks/draft-zero/tree/exp1-fdn-generalist) | [report](docs/003-fdn-generalist-report.md) · [Hugging Face](https://huggingface.co/danbrooks/draftzero-fdn-exp1) |

Exp #2 is under way ([ROADMAP](ROADMAP.md), Phase 4). Its imitation arm has a report:
**[does an imitation start help?](docs/013-exp2-imitation-report.md)** Starting self-play from a network
pretrained on 17lands decisions (73% agreement with humans) did not help: the network stopped learning
within three generations, while the heuristic start kept improving. The heuristic-start arm has its own
report: **[run 1, final](docs/014-exp2-run1-report.md)**. It learned for about eight generations (league
59%), then plateaued at 50% against raw search as its value head overfit; 17lands agreement on
commons ended at ρ +0.20.

Studies: **[human gameplay data](docs/008-gameplay-data.md)**. 17lands and Arena logs mapped onto XMage
states (a recorded turn replays to the same next state ~89% of the time), human-trained heads versus
gen 33, and a first test of MCTS coaching. **[Hidden information](docs/009-hidden-information.md)**.
MageZero's network and search both see the opponent's hand and the library order. The search plays
around a counterspell only when it's really there. The doc:
- compares how other hidden-information games handle this;
- evaluates IS-MCTS;
- checks XMage's own bots (the MAD AI never simulates an opponent's response, and upstream's Monte
  Carlo player determinizes);
- plans fair baselines and experiment #2.

What comes next, and why: **[ROADMAP.md](ROADMAP.md)**.

## How the pieces relate

```
mage (XMage fork, Java)      pinned release artifact — rules engine, deck pools,
  ▲                          GAME_SUMMARY lines, action vocab
  │                          (danieljbrooks/mage v0.2-generalist, built into Will's v0.2 bundle)
MageZero (Python)            pinned dependency — model, trainer, inference server
  ▲                          (Will's v0.2 plus three opt-in commits: danieljbrooks/MageZero draftzero)
DraftZero                    this repo — pools, generalist loop, parallel JVMs, metrics and
                             dashboards, format stats, workers (draftzero/engine.py runs
                             MageZero's scripts unmodified)
```

Dependencies point one way only. Nothing in MageZero knows DraftZero exists. Experiment #1
ran on an older MageZero fork (`mz-engine`, `bcc76de`) and the v0.1 XMage build
(`danieljbrooks/mage` `exp1-fdn-generalist`); its checkpoints don't load under v0.2.

## Layout

| Path | What |
|---|---|
| `src/draftzero/loop.py` | the generation loop: play → train → eval |
| `src/draftzero/stats.py` | GIH win rate, Spearman ρ vs 17lands, deck/colour records |
| `src/draftzero/pools.py` | build deck pools from 17lands exports |
| `src/draftzero/reference.py` | build the 17lands GIH reference |
| `src/draftzero/dashboard.py` | the format-knowledge page (`format.html`) |
| `src/draftzero/paths.py` | resolve deck stems against a machine-local `deck_root` |
| `src/draftzero/provenance.py` | record which code produced a run |
| `src/draftzero/resources.py` | CPU/RAM/GPU sampling that respects cgroup limits |
| `src/draftzero/watchdog.py` | persist weights, detect stalls, end the run |
| `src/draftzero/workers/` | where self-play runs: local, ssh, runpod |
| `tools/extract_decks.py` | build the deck pool from 17lands public game data |
| `src/draftzero/gameplay/` | human gameplay data ([docs/008](docs/008-gameplay-data.md)): 17lands replays and Arena logs → XMage states (`StateSpec`), turn replay, imitation data, coaching; `dz gameplay <tool>` |
| `java/mzbridge/` | a long-lived XMage worker, no fork change: builds any `StateSpec` and answers build / encode / coach / replay_turn requests |
| `tools/gameplay/` | experiment scripts for the gameplay-data study |
| `assets/` | small versioned inputs: deck metadata, action vocab, GIH reference |
| `assets/sample/` | 80 decks and pools, so a fresh clone runs without the full pool |
| `configs/` | run configs (`fdn_l40s.yml` produced experiment #1) and the curriculum |
| `docs/` | design notes, the [RUNBOOK](docs/002-RUNBOOK.md), experiment reports and reviews, numbered in the order they were written (`001-`, `002-`, …); a new doc takes the next number |
| `data/` | **generated, gitignored**: decks, pools, 17lands downloads, runs, checkpoints |

`assets/` versus `data/` is the important line. Assets are small and reviewable and you
want to diff them. Data is large and regenerable and must never be committed.

## Install

```bash
pip install -e .
```

MageZero is pinned by git URL in `pyproject.toml`. **Replace the branch pin with a commit
SHA before any run you intend to reproduce** — a branch pin silently changes what your
results were produced against.

For local development against a MageZero checkout:

```bash
pip install -e ../MageZero && pip install -e . --no-deps
```

**Two engines.** Everything new runs on MageZero v0.2 (the pin above) with the v0.2 XMage bundle,
imitation learning included (`draftzero.gameplay.imitation`: build its tables with the v0.2 bundle
and pass a v0.2 checkpoint to `load_checkpoint`). Only the stages built on experiment #1's gen-33
checkpoint (`gen33*`, `heads`, `finetune`, `extras`, and remote coaching with gen 33, docs/008) need
exp #1's engine, because v0.2 neither produces gen 33's state encodings nor loads it the same way.
Keep a separate venv for those:

```bash
python -m venv .venv-exp1 && .venv-exp1/bin/pip install \
  "magezero @ git+https://github.com/danieljbrooks/MageZero@bcc76de" && .venv-exp1/bin/pip install -e . --no-deps
```

Pair it with the v0.1 XMage bundle (`danieljbrooks/mage` `exp1-fdn-generalist`). The gameplay
bridge (`java/mzbridge`) builds and passes its tests against either bundle.

## Data

**Decks come from [17lands](https://www.17lands.com/)**, whose
[public datasets](https://www.17lands.com/public_datasets) are licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The pool is every deck in the
FDN Premier Draft game data whose player sits in the ≥60% win-rate bucket: 31,516 decks,
split by draft into 28,366 train and 3,150 eval.

**To try it without the full pool**, use the 80 decks committed in
[`assets/sample/`](assets/sample/README.md) with `configs/sample.yml`.

**To build the full pool** (downloads ~57 MB from 17lands, writes ~150 MB of `.dck` files):

```bash
python tools/extract_decks.py --set FDN --format PremierDraft --min-winrate 0.60 --dck
dz pools build          # assets/decks.tsv -> data/pools/{train,eval}.txt
export MZ_DECK_DIR=data/deckgen/FDN_PremierDraft_wr60/top_player_FDN_decks
dz pools check --deck-root $MZ_DECK_DIR
```

`extract_decks.py` reads XMage collector numbers from the XMage build's set jar, via the
`xmage` symlink or `--xmage-jar`. After one run it caches them under `data/17lands/`. This
reproduces experiment #1's pool **byte for byte**: all 31,516 `.dck` files. That was checked
on 2026-09-25. The committed split is regenerated the same way with
`python -m draftzero.pools --decks data/deckgen/FDN_PremierDraft_wr60 --out <dir>`. Its
`decks.tsv` also matched `assets/decks.tsv` exactly.

`assets/decks.tsv` is the source of truth for the train/eval split, and nothing large is
committed. Point a run at the `.dck` files with `deck_root` in the config or `MZ_DECK_DIR`,
and at the XMage build with the `xmage` symlink. The full pool is also published with the
experiment #1 release on [Hugging Face](https://huggingface.co/danbrooks/draftzero-fdn-exp1).

## Running

```bash
bash deploy/bootstrap.sh                          # install deps, report the REAL cpu/ram quota
bash deploy/launch.sh configs/sample.yml          # ~15 min, on the committed sample decks
bash deploy/launch.sh configs/runpod_smoke.yml    # ~20 min end-to-end check on the full pool
bash deploy/launch.sh configs/fdn_l40s.yml        # experiment #1's config
```

`configs/sample.yml` is verified to load and to resolve every deck. A full game run from a
fresh clone still needs MageZero installed and the XMage build linked.

**[docs/002-RUNBOOK.md](docs/002-RUNBOOK.md)** is the operating guide for the long run: provisioning,
what to verify in the first 24 hours, how to tell whether it is actually learning, and how to
resume after a crash.

`launch.sh` starts the loop and a watchdog. The watchdog mirrors checkpoints to
`$DZ_PERSIST` and ends the run on whichever comes first: `target_games`, `DZ_MAX_HOURS`
(a wall-clock budget cap — hours × rate, needing no credentials on the worker), or
`DZ_STALL_MINUTES` with no new game. Then it runs `$DZ_ON_COMPLETE` — **but only after
verifying the final sync**. A failed sync never triggers it. Losing a worker is cheap;
losing the weights is not.

## Workers

```python
from draftzero.workers import make_worker

w = make_worker("local")
w = make_worker("ssh", host="gpubox.lan", user="dan", workdir="~/draftzero")
w = make_worker("runpod", gpu_id="NVIDIA RTX 4000 Ada Generation", datacenter="EU-RO-1")
```

Adding a worker means implementing `run`, `push` and `fetch`, plus `provision`/`teardown`
if the machine is rented. Everything above that interface is identical.

### Sizing a worker

**Before reserving a RunPod pod, read [docs/005-runpod-tips.md](docs/005-runpod-tips.md)**:
stock, networks that block SSH, image and CUDA pairing, self-destruct, and billing.

Self-play is **CPU-bound** XMage/MCTS up to about 24 cores per GPU. The GPU only serves
small batched inference: 2% utilization during heuristic play, 11–14% during network
self-play. Run several JVMs of 4 game threads (`jvm.jvms`), about one per 4 cores: in the
exp #2 pilot 7 × 4 beat 1 × 28 by 3.3× offline ([docs/006](docs/006-exp2-pilot.md)). With the
network on, the shared inference server becomes the cap. Pick a machine by vCPU, not VRAM:

```bash
dz workers rank --min-vcpu 8    # in-stock RunPod GPUs by vCPU per dollar
```

Two things that cost real time to learn: containers report the **host's** CPU and RAM
(a pod sold as 9 vCPU / 50 GB reports `nproc=48` / 251 GB and is cgroup-capped to ~7.65
cores / 46 GB), so never size threads from `nproc`; and the JVM thread pool is
`min(jvm.threads, chunk_games)`, so `chunk_games` must be ≥ `threads` or the extra threads
do nothing.

## Dashboards

Each run writes three pages into `runs/<id>/`:

| Page | Rendered by | Shows |
|---|---|---|
| `index.html` | DraftZero | landing page linking the other two |
| `dashboard.html` | DraftZero (`report.py`, from exp #1's MageZero fork) | win rates, losses, throughput, CPU/GPU/RAM |
| `format.html` | DraftZero | card GIH WR vs 17lands, Spearman ρ, deck colour records |

`dashboard.html` is generic training health; `format.html` is the draft-format view. Both are
standalone: no build step and no CDN.

Both refresh automatically after every chunk of games. To rebuild by hand:

```bash
dz dashboard runs/<run_id>
```

## Measured throughput

**Experiment #1** — L40S, ~24 usable cores, 18 game threads, `search_budget` 96, $0.79/hr:

| Setting | Measured |
|---|---|
| network self-play | ~64–90 games/hr |
| simulations/s, with the network | ~41 |
| simulations/s, without it (gen 0) | ~84 |
| cost | 2,507 games for ~$28, about $0.011/game |

The network roughly halves simulations per second, so inference is about half the cost of
a simulation. Throughput scales with `search_budget`, so expect up to ~3× the cost per
game at 300 simulations. [ROADMAP.md](ROADMAP.md) Phase 3 measures that directly.

*Early smoke-test pod* — RTX 4000 Ada, 7.65-core quota:

| Setting | games/hr |
|---|---|
| gen 0 heuristic, `search_budget` 40, 6 threads | 112 |
| gen 0 heuristic, `search_budget` 200, 12 threads | 45 |
| network self-play, `search_budget` 40 | 29 |
| network self-play, `search_budget` 200 (projected) | ~12 |

Network play is ~4× slower than gen-0 heuristic, and games roughly double in length once
a net is driving them (20 → 40 turns). Budget from network self-play, not gen 0.

## Credits and license

- **[17lands](https://www.17lands.com/)** — every deck and the human reference statistics
  come from its public datasets, licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Thank you to 17lands and the
  players who share their data.
- **[MageZero](https://github.com/WillWroble/MageZero)** by Will Wroble — the RL
  framework, model, trainer and evaluator this repo builds on, plus advice on training it.
- **[chrismaghuhn](https://github.com/chrismaghuhn)** — advice on compute and on
  performance ([WillWroble/MageZero#3](https://github.com/WillWroble/MageZero/issues/3)).
- **[XMage](https://github.com/magefree/mage)** — the rules engine every game runs in.

**License.** The code is [MIT](LICENSE). That covers every version of this repository,
including commits from before `LICENSE` was added, such as the `exp1-fdn-generalist` tag.
Deck files and statistics derived from 17lands
data (`assets/decks.tsv`, `assets/sample/`, `assets/reference/`, and the published
releases) carry CC BY 4.0.
