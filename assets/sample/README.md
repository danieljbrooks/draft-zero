# Sample decks

80 of the 31,516 FDN decks, listed by stem (`FDN_top_00001_WB`):

| File | What |
|---|---|
| `eval.txt` | 40 eval-split decks: exactly the decks in experiment #1's fixed milestone eval |
| `train.txt` | 40 train-split decks, a seeded random sample (`random.Random("draftzero-sample-v1")`) |

v0.1.0 shipped these as XMage `.dck` files (and experiment #1's eval pairs); those stay in v0.1.0. The same 80
decks in gorge's deck format (bare `{name, count}` JSON, as `python -m draftzero.data.decks` writes them, plus a
`pool.tsv`) come next, so a fresh clone can play without building the full pool. Until then, the stems resolve
against a full pool in `data/gorge/decks/<stem>.json`.

## Provenance and license

Every deck comes from **[17lands](https://www.17lands.com/)** public FDN Premier Draft game data
(`game_data_public.FDN.PremierDraft.csv.gz`, updated 2024-12-18 on the
[public datasets page](https://www.17lands.com/public_datasets)), keeping decks whose player sits in the ≥60%
win-rate bucket. 17lands publishes that data under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
These files are a derivative of it, redistributed under the same license. Credit 17lands if you use them.
