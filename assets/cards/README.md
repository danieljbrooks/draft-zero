# FDN card tables (frozen)

Two tab-separated tables (one `#` comment line, then a header) that the sealed deck builder
(`draftzero.build.cards`) and the evaluation code (`draftzero.eval.stats`, `draftzero.eval.analyze`) read.
They came from v0.1.0's XMage-era id mapping (docs/008) and moved here from `assets/gameplay/` in v0.2.0.

| file | rows | what |
|---|---:|---|
| `FDN_cards.tsv` | 623 | per FDN/SPG card name: XMage set/number, mana cost, printed P/T, starting loyalty, combat keywords, mechanics tags (regexes over the XMage card source; besides the research tags, `etb_tapped`, `doesnt_untap`, `turn_start_trigger`, `upkeep_draw` and `opp_draw_trigger` for the tapped state and the turn-start draws; `cant_block`, `blocks_only_flyers` and `host_cant_attack_block` (Pacifism) for who can block), flashback key, land colours, attachment kind, counter tracking mode, and `etb_counters` (N or X +1/+1 counters it always enters with). |
| `FDN_card_ids.tsv` | 1,166 | every card grpId seen in the FDN replay files (FDN, SPG, tokens and basic lands from 56 cosmetic sets) with name, set, rarity, type line and mana value: a subset of 17lands' `cards.csv`, so the package and its tests work without the 17lands download. |

**Frozen.** The generator (`draftzero.gameplay.ids build-tables`, which read XMage's card source and class dump)
went with XMage; it is in v0.1.0. These files are versioned data now, not build output: the deck builder's
checkpoints depend on `FDN_cards.tsv`'s facts (and on `assets/sealed/FDN_cardnames.txt`'s order). A new set needs
these facts from gorge's card IR or the Forge scripts instead.

## Sources and licences

- **17lands** public datasets, **CC BY 4.0** (https://www.17lands.com/public_datasets): card ids, names, rarities,
  type lines and mana values (`cards.csv`). 17lands does not endorse this project.
- **XMage** (MIT), mage 1.4.58 source and class dump: set numbers, P/T, keywords, mechanics tags.
