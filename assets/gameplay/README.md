# Gameplay id tables (derived)

Small derived tables that `draftzero.gameplay.ids` loads to map 17lands / MTG Arena ids for
Foundations (FDN + Special Guests) onto XMage objects and MageZero action keys
(`assets/vocab/FDN_SPG.tsv`). Tab-separated, one `#` comment line, then a header.

| file | rows | what |
|---|---:|---|
| `FDN_tokens.tsv` | 37 | token grpId -> XMage token class (`mage.game.permanent.token.*`), with that class's P/T and colour. Tokens are mapped **by id**: Cat 94156 is `CatToken3` (1/1), Cat 94157 is `CatToken` (2/2). 94179 `Copy` has no class: token copies carry the copied card's id. |
| `FDN_abilities.tsv` | 228 | every ability id seen in the FDN replays -> category (triggered / activated / mana), source card(s), how the source was found (`arena_db` link or replay `cooccurrence`), and for activated non-mana abilities the XMage `toString()` key per source (JSON) with its vocab index. Also derived counters facts (`self_p1p1`, `self_double` for Mossborn Hydra's landfall, `counter_other`, planeswalker `loyalty` cost) and three flags read off 17lands' text (`text_flags`: `counter_on`, `mill_surveil`, `attack_trigger`). No ability text is stored; the 26 ids missing from 17lands' `abilities.csv` get a source card and XMage key only. |
| `FDN_cards.tsv` | 623 | per FDN/SPG card name: XMage set/number, mana cost, printed P/T, starting loyalty, combat keywords, mechanics tags (regexes over the XMage card source; besides the research tags, `etb_tapped`, `doesnt_untap`, `turn_start_trigger`, `upkeep_draw` and `opp_draw_trigger` for the tapped state and the turn-start draws; `cant_block`, `blocks_only_flyers` and `host_cant_attack_block` (Pacifism) for who can block), flashback key, land colours, attachment kind, counter tracking mode, and `etb_counters` (N or X +1/+1 counters it always enters with). |
| `FDN_card_ids.tsv` | 1,166 | every card grpId seen in the FDN replay files (FDN, SPG, tokens and basic lands from 56 cosmetic sets) with name, set, rarity, type line and mana value: a subset of 17lands' `cards.csv`, so the package and its tests work without the 17lands download. |

## Sources and licences

- **17lands** public datasets, **CC BY 4.0** (https://www.17lands.com/public_datasets): card and
  ability ids, names, type lines (`cards.csv`), the text flags (`abilities.csv`), and the
  co-occurrence statistics over the FDN Premier Draft replay file. 17lands does not endorse this
  project.
- **XMage** (MIT), mage 1.4.58 as built in `xmage/` and its source at
  `~/Desktop/Code/DEPRECATED_PREDRAFTZERO_MageZero-Experiments/pr-dense-vocab/mage`: token classes,
  ability keys, P/T, keywords, mechanics tags.
- The research run also read the local MTG Arena client card database (read-only) for token P/T,
  ability -> card links and ability categories. Only the derived links are kept here (which card
  an ability id belongs to, which category it is); **no Arena card-database text is stored**.

## Regenerating

The tables come from the research phase's id mapping (docs/008; research scripts
`id_mapping.py` + `IdMapDump.java`), which scans the replay file, dumps every FDN/SPG card and token
class from the XMage jars, infers ability sources by co-occurrence and joins the optional Arena card
DB. Its output directory (`id_mapping/`: `token_ids.json`, `ability_ids.json`, `card_ids.tsv`,
`xmage_cards.jsonl`, `xmage_tokens.jsonl`) is turned into these tables by:

    python -m draftzero.gameplay.ids build-tables --research <research>/id_mapping \
        --mage-src ~/Desktop/Code/DEPRECATED_PREDRAFTZERO_MageZero-Experiments/pr-dense-vocab/mage

and checked with

    python -m draftzero.gameplay.ids check        # every FDN booster card maps, keys are in the vocab
    python -m draftzero.gameplay.seventeenlands stats --every 158 --limit 5000   # id coverage on real rows

Re-run both after an XMage or vocab update: keys are XMage rule text and change with Oracle wording.
