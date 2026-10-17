# Sample decks

80 of the 31,516 FDN decks in gorge's deck format, so a fresh clone can play and run the smoke config without
building the full pool (`tools/extract_decks.py` + `python -m draftzero.data.decks`):

| Path | What |
|---|---|
| `decks/<stem>.json` | 80 gorge deck files (bare `{name, count}` JSON), byte-identical to the full pool's `data/gorge/decks/<stem>.json` |
| `pool.tsv` | their pool rows (`deck  split  colors  main_colors  cards`, the full pool's order): 40 `train`, 40 `eval` |
| `train.txt` | the 40 train-split stems: a seeded sample, `python -m draftzero.data.pools sample --split train --n 40` (seed `draftzero-sample-v1`) |
| `eval.txt` | the 40 eval-split stems: exactly the decks in experiment #1's fixed milestone eval |

Point a config at it with `data: {decks: assets/sample/decks, pool: assets/sample/pool.tsv}`; a `:closedlist`
seat's belief then defaults to the 40 train decks. dzgorge plays all 80 with supported cards only (`dzgorge coverage`
at the engine pin of 2026-10-14: 244 distinct cards, 0 unsupported).

Rebuild from a full pool:

```bash
python -m draftzero.data.pools subset --pool data/gorge/pool.tsv --decks data/gorge/decks \
  --names assets/sample/train.txt assets/sample/eval.txt --out assets/sample
```

v0.1.0 shipped these 80 decks as XMage `.dck` files (and experiment #1's eval pairs); those stay in v0.1.0.

## Provenance and license

Every deck comes from **[17lands](https://www.17lands.com/)** public FDN Premier Draft game data
(`game_data_public.FDN.PremierDraft.csv.gz`, updated 2024-12-18 on the
[public datasets page](https://www.17lands.com/public_datasets)), keeping decks whose player sits in the ≥60%
win-rate bucket; each deck file's `notes` names its 17lands draft id. 17lands publishes that data under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). These files are a derivative of it, redistributed under
the same license. Credit 17lands if you use them. 17lands does not endorse this project.
