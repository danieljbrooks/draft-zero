# Gameplay-data test fixtures

## arena_*.log (Arena Player.log parser, `tests/test_gameplay_arena.py`)

`arena_synthetic.log` is a **synthetic** MTG Arena detailed log, written by hand (with a small
generator script) to mimic the real file's structure: `[UnityCrossThreadLogger]` headers, one-line
GRE->client JSON, pretty-printed client->GRE JSON, Unity trace lines, a hover UI message, a
`[Message summarized ...]` event (Arena's 50-object logging limit), a QueuedGameStateMessage, and
two games of one match (the second after a sideboarded `SubmitDeckResp`).

Nothing in it is copied from a real log. Every identifier is fake: the account id `FAKEACCT0000042`,
the screen names `Testy McFakeface#12345` / `Opponent Fakename#67890`, the user ids, the match id
`00000000-1111-2222-3333-444444444444`, the transaction/session UUIDs and the path user `fakeuser`.
The privacy tests check that none of them reaches the parser's output.

Card ids are Arena grpIds of FDN cards (Plains, Savannah Lions, Pacifism, Felidar Savior, Vanguard
Seraph, Prideful Parent, the 1/1 Cat token); the names and the two ability texts used in the test
come from 17lands' public `cards.csv` / `abilities.csv`, CC BY 4.0 (https://www.17lands.com/public_datasets).

## fdn_premier_rows.csv.gz (17lands replay parser, `tests/test_gameplay_17lands.py`)

Nine games from the 17lands public replay dataset for Foundations Premier Draft
(`replay_data_public.FDN.PremierDraft.csv.gz`), **data by 17lands, CC BY 4.0**
(https://www.17lands.com/public_datasets). 17lands does not endorse this project.

| row | why it is here |
|---|---|
| 0 | on the play; hand-inspected in the research (Kaito tokens and loyalty, Pridemate, opponent flashback) |
| 1 | the user's own Think Twice flashback, Clinquant Skymage counters, a unique block, an opponent Equip |
| 4 | on the draw; hand-inspected in the research (Imprisoned in the Moon, Goldvein Pick, Banishing Light) |
| 8 | a mulligan on the play with one bottomed card |
| 26 | the opponent plays a tapped dual land (Thornwood Falls); its Imprisoned in the Moon turns the user's Lathril into a land |
| 42 | an ambiguous blocker -> attacker pairing |
| 76 | Mossborn Hydra: enters with a +1/+1 counter, landfall doubles it |
| 130 | the opponent's Rune-Sealed Wall and Strix Lookout tap for their abilities on its own turn |
| 198 | the user's Phyrexian Arena: an upkeep draw before the draw step's card |

It is an excerpt, not a copy of the rows: `replay.write_excerpt` keeps the metadata, hand and
per-turn columns (turn slots past the rows' largest `num_turns` dropped, empty deck/sideboard
columns dropped, the derivable `*_total_*` columns dropped), blanks `draft_id`, and adds a
`source_row` column with the row's index in the full file. Regenerate with:

    python -c "from draftzero.gameplay import replay; replay.write_excerpt([0, 1, 4, 8, 26, 42, 76, 130, 198], 'tests/fixtures/gameplay/fdn_premier_rows.csv.gz')"

`fdn_premier_rows_extra.csv.gz` holds two more rows of the same file, excerpted the same way (same
licence and attribution). They live in their own file so that tests counting over the nine rows
above (fingerprints, belief) keep their numbers:

| row | why it is here |
|---|---|
| 64435 | at the end of the opponent's turn 5 it controls the user's Helpful Hunter (a control change: `Perm.owner`) |
| 89383 | the opponent's Imprisoned in the Moon arrives and no creature turns into a land, so it enchants a land; a three-attacker block |

    python -c "from draftzero.gameplay import replay; replay.write_excerpt([64435, 89383], 'tests/fixtures/gameplay/fdn_premier_rows_extra.csv.gz')"

No mirrored pair is committed: no two fixture rows are the two sides of one game. The pair tests
build a synthetic partner row from a fixture row (sides swapped, made-up hands of basic lands), and
the real-pair test reads `data/gameplay/pairs_FDN_PremierDraft.jsonl` and the replay file when they
are there (skipped otherwise).
