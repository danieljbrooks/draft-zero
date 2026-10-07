# gorge: FDN self-play on a fast rules engine

Code for playing DraftZero's FDN Limited decks on [gorge](https://github.com/adams-shaun/gorge), a pure-Go
Magic rules engine that compiles Forge's card scripts, and for training and evaluating networks there. Results:
[docs/026](../docs/026-gorge-fdn-self-play.md).

| File | What |
|---|---|
| `GORGE_REF` | the gorge commit everything here was built and measured on. gorge changes hundreds of times a day: pin it |
| `build.sh` | clones gorge at `GORGE_REF`, fetches and compiles the card corpus, copies `cmd/dzgorge` into the checkout and builds `dzgorge`, `policytrain` and `botbench` |
| `cmd/dzgorge/` | the driver (Go). It is built inside gorge's module because it uses gorge's internal packages: the search (`internal/azmcts`), the network (`internal/policynet`) and the game runner (`internal/bench`) |
| `decks.py` | writes the 31,516-deck pool (`tools/extract_decks.py`'s output) as gorge deck files, under the XMage pool's names, so `assets/decks.tsv`'s train/eval split applies |
| `bench.sh` | the throughput and strength benchmark (docs/026 §2–3) |
| `azloop.sh` | one AlphaZero-style generation: search self-play on the train decks, then `policytrain` on the searched decisions (§4) |
| `evalnet.sh` | a trained network's games on the eval decks: alone against `bot` and `random`, and inside the search against the search without it (§4) |
| `experiments.sh` | every run behind docs/026 §4.2–4.5 and §5 after gen 0, in order: the held-out positions, the networks, their games, the confirmations |
| `gih_runs.sh` | self-play for 17lands card statistics (§5) |
| `analyze.py` | win rates with paired confidence intervals, games-in-hand win rates against 17lands, throughput |
| `sortcorpus.py` | puts a visit corpus in game order; `dzgorge` now writes in game order itself |
| `figures.py` | docs/026's figures |

## Quick start

```bash
python tools/extract_decks.py --set FDN --format PremierDraft --min-winrate 0.60   # 17lands, CC BY 4.0
python gorge/decks.py                     # -> data/gorge/decks/*.json, data/gorge/pool.tsv
bash gorge/build.sh                       # needs Go 1.25+; GORGE_DIR defaults to ../ext/gorge
D=$PWD/data/gorge; G=${GORGE_DIR:-../ext/gorge}
(cd $G && ./bin/dzgorge coverage -decks $D/decks -pool $D/pool.tsv)
(cd $G && ./bin/dzgorge play -decks $D/decks -pool $D/pool.tsv -split eval -a az:sims=25 -b bot -pairs 100 -out $D/try.jsonl)
python gorge/analyze.py wins $D/try.jsonl
```

## Policies (`-a`, `-b`)

| Spec | Plays |
|---|---|
| `random` | uniformly among the legal options, mana paid automatically (gorge's `sb-uniform`) |
| `bot` | gorge's production heuristic, no search |
| `az:sims=N` | AlphaZero-style tree search over the seat's own decisions (`internal/azmcts`), N simulations each. Honest worlds: each simulation re-deals the cards the seat can't see. It knows the opponent's 40-card list, as DraftZero's §4.2 bots did, but never their hand or library order. Without `net=`, generation 0: a uniform prior and gorge's heuristic leaf |
| `az:sims=N:net=F.gpol` | the same with a trained network's prior and value. `leaf=heuristic` keeps the heuristic leaf, `prior=uniform` the uniform prior |
| `prior:net=F.gpol` | the network's policy alone: its most likely candidate, no simulation |

Each deck pair is played twice on one seed with the policies swapped (DraftZero's paired games), unless both sides
are the same policy. Decisions the search doesn't cover (mulligans, modes, ordering triggers, X values and the like)
are gorge's `bot`'s. The London mulligan round is off by default: the bot mulligans a random third of its hands.

## Licences

gorge is Apache-2.0. The Forge card scripts it compiles are GPL-3.0; `build.sh` fetches them into the gorge
checkout's `.cards/`, and nothing here contains them. Decks come from 17lands' public data (CC BY 4.0).

## DraftZero's networks in gorge's search (docs/027)

| File | What |
|---|---|
| `dzg/` | Python: MLP, transformer and GNN networks on gorge's entity encoding; training (`python -m dzg.train`), the batching inference server (`python -m dzg.serve`), the export for the engine (`python -m dzg.export`); `dzg/SPEC.md` is the data and socket contract |
| `patches/0001`–`0007` | opt-in changes to gorge's search, applied by `build.sh`: an external scorer, casts searched before mana is tapped (`autopay`), network-chosen candidates (`topk=K`), the opponent in the tree (`oppnodes`, `oppfull`), a land-count mulligan (`-mull-heuristic`), recording of the new decisions, and modes, choices and multiple targets searched (`morekinds`) |
| `cmd/dzgorge` | also: `remote=` (a served network), `gonet=` (the MLP inside the engine), `pack` (visit corpora to training shards), `-record-features entity`, `-record-every K` |
| `dzg_gen.sh`, `dzg_eval.sh`, `dzg_loop.sh`, `dzg_tune.sh`, `dzg_bench.sh` | a generation's self-play, paired evaluation arms, the generation loop, tuning arms (any spec against any spec), games an hour |
| `dzg_mirror.sh`, `dzg_collect.py`, `dzg_figures.py` | gather results from every machine, tabulate them, draw docs/027's figures |
