# Sample decks

80 of the 31,516 FDN decks, committed so a fresh clone can run
[`configs/sample.yml`](../../configs/sample.yml) without building the full pool.

| File | What |
|---|---|
| `decks/*.dck` | 80 XMage deck files |
| `eval.txt` | 40 eval-split decks: exactly the decks in experiment #1's fixed milestone eval |
| `train.txt` | 40 train-split decks, a seeded random sample (`random.Random("draftzero-sample-v1")`) |
| `exp1_eval_pairs.tsv` | the 40 games (20 deck pairs × both seatings) that every milestone eval in experiment #1 played, as `deck_a` vs `deck_b` |

**Why ship the pair list separately.** The loop draws eval pairs with a seeded RNG from
the *eval pool*. A 40-deck pool yields different pairs than the full 3,150-deck pool did,
so `configs/sample.yml` does not replay experiment #1's eval. `exp1_eval_pairs.tsv` is
the reference if you want to.

## Provenance and license

Every deck comes from **[17lands](https://www.17lands.com/)** public FDN Premier Draft
game data (`game_data_public.FDN.PremierDraft.csv.gz`, updated 2024-12-18 on the
[public datasets page](https://www.17lands.com/public_datasets)), keeping decks whose
player sits in the ≥60% win-rate bucket. 17lands publishes that data under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). These files are a
derivative of it, redistributed under the same license. Credit 17lands if you use them.

To rebuild the full pool yourself (it reproduces these files byte for byte), see
"Data" in the [README](../../README.md#data).
