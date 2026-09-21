# FDN generalist

One network that plays FDN Premier Draft limited with any deck. Every game draws both decks at random from about 31.5k decks built by top 17lands players.

## Pieces

| | |
|---|---|
| `vocab/FDN_SPG.tsv` | Set-wide action vocabulary: Foundations plus its 10 Special Guests, 1024 slots, with a hashed tail for tokens. Set `MZ_ACTION_VOCAB` to it for both the JVM and Python. Rebuild it with `../action_vocab/java/tools/VocabDump.java` (see `vocab/FDN_SPG_extra_cards.txt`). |
| `pools/` | `make_pools.py` output. `train.txt` has 28,366 decks and `eval.txt` has 3,150. The split is by draft, so no build of an eval draft is trained on. `decks.tsv` holds colors and player bucket. |
| `reference/FDN_gih.json` | `reference_gih.py` output: 17lands GIH WR per card and WR per main colors, over all players in the public game data (791k games). Spot-checked against the 17lands card table to within ~1pp. |
| `src/magezero/generalist.py` | The training loop. |
| `src/magezero/generalist_stats.py` | Card GIH WR, rank correlation with 17lands, and deck/color records, all from the `GAME_SUMMARY` lines the XMage fork logs per game. |

This needs the XMage fork branch `fdn-generalist`, which adds deck pools, `GAME_SUMMARY` lines, a configurable game time cap, and the action vocab. `xmage/` must point at that build.

## A generation

1. **Play.**
   - Gen 0 is heuristic search (no network) on both sides.
   - After that, about 70% of games are self-play: both sides use the current net and both sides become training data.
   - About 20% are against a random older checkpoint and 10% against gen 0's net. In those games only the current net's side is kept.
   - Games run in chunks of 12, and the dashboard refreshes after each chunk.
2. **Train** on the newest `replay_states` states. Older shards move to `data/<model>/ver1/archive/`. The feature vocab is always dense.
3. **Eval.**
   - The new checkpoint plays paired games on held-out decks: deck X vs Y, then Y vs X, with the agent always player A.
   - Opponents are heuristic search at the same budget (`offline`) and XMage's minimax AI.
   - Each dashboard point pools the last `window_gens` generations.

Generations are open-ended. A crash resumes at the interrupted stage.

## Running

```bash
experiments/generalist/run.sh experiments/generalist/configs/smoke.yml --fresh
```

```bash
nohup caffeinate -ims experiments/generalist/supervise.sh experiments/generalist/configs/laptop.yml --fresh > runs/generalist_supervise.log 2>&1 &
```

The dashboard is `runs/<id>/dashboard.html`, and `runs/<id>/deck_records.tsv` holds every deck's W/L.

## Reading the format stats

GIH WR comes from self-play, so it averages 50% by construction and only its ranking is comparable to 17lands. Spearman ρ uses cards seen in at least `min_card_games` games. Gen 0's heuristic self-play gives a baseline ρ for search with no learned card sense. Each card shows up in only a few games per generation (~5 at 48 games/gen), so ρ is noisy for many generations.
