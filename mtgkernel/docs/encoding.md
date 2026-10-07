# Decision encoding `dzk-enc-v2`

Changes from `dzk-enc-v1`: the object cap is per token group with duplicates dropped first (v1 truncated the token
list in order, so a runaway board of the actor's could leave the opponent's battlefield without a single token), and
the trigger-order counts (action features 11-13) are `ln(1 + x) / ln 8` instead of `x / 7` (v1 saturated at the
±8 clamp from 56 on, so triggers 56 and 57 of a 65-trigger group encoded identically). Data, checkpoints and
networks of v1 are refused by the v2 loaders.

`src/encode.rs` turns one **real** decision (two or more actions after mana masking; forced decisions are never
encoded) into fixed-schema numbers, from the **acting player's** point of view. This file is the reference; any
change to a feature, its order or its scale must bump the version string (`encode::ENC_VERSION`), which every data
shard (`dzk-data-v1`), every checkpoint (`.pt`) and every exported network (`dzk-net-v1`) carries and every loader
checks.

## What the encoder may read

- The acting player's redacted observation, `ObservationV5`, exactly as the engine hands it to that player: own
  hand, both battlefields, graveyards, exile, the stack, counts, combat state, the scan context of the current
  attacker/blocker selection, pending casts/effects, London-mulligan state, and the actor's **known** cards
  (cards of the opponent's hand or of either library that an effect revealed to the actor).
- The decision's masked legal actions (`LegalActionV5`, in the masked order the bots index).
- Static card-registry facts (`CARD_DEFS`): mana value, types, colours, printed power/toughness and keywords,
  Ward, token flag, whether the card makes mana or has activated abilities.

`encode(obs, actions, out)` takes nothing else (it never sees the `GameState`), so it cannot read the opponent's
hand, the library order or anything else hidden. `encode_game(game, out)` is the convenience wrapper: it passes
`game.observation()` and `game.choice(k)` only.

`arena_id` is **never a feature**: mtg-kernel numbers objects in library order at the deal, so an arena id leaks
how deep in the library a card started. Arena ids (with the zone) are used only to link an action's object
references to the object tokens of the same decision.

Everything is actor-relative: "me" is the acting player, "opp" the other one; seat P0 = on the play (A). Every
feature is clamped to [-8, 8] after scaling (outliers: random play has reached 126 creatures on one side).

## Card vocabulary

`card` ids are vocabulary ids: 0 = no card; registry id `d` maps to `d + 1`, capped at 511 (VOCAB = 512; the
registry has fewer than 511 entries, and FDN ids are below 250). The network embeds them (E = 32, id 0 embeds to
zeros).

## Global features `global` f32[96]

| Index | Feature | Scale |
| --- | --- | --- |
| 0, 1 | life me, opp | /20 |
| 2, 3 | hand size me, opp | /10 |
| 4, 5 | library size me, opp | /40 |
| 6, 7 | graveyard size me, opp | /20 |
| 8, 9 | cards in exile owned by me, opp | /10 |
| 10 | turn (engine rounds) | /30 |
| 11 | I am the active player | 0/1 |
| 12 | I am the priority player | 0/1 |
| 13-24 | step one-hot: untap, upkeep, draw, main1, begin combat, declare attackers, declare blockers, combat damage, end combat, main2, end, cleanup | |
| 25-30 | my mana pool W U B R G C | /5 |
| 31-36 | opponent's mana pool | /5 |
| 37, 38 | lands controlled me, opp | /10 |
| 39, 40 | untapped lands me, opp | /10 |
| 41, 42 | creatures me, opp | /10 |
| 43, 44 | untapped creatures me, opp | /10 |
| 45, 46 | total effective power of creatures me, opp | /20 |
| 47 | lands I played this turn | raw |
| 48, 49 | spells cast this turn me, opp | /5 |
| 50 | cards I drew this turn | /5 |
| 51 | London mulligan phase active (someone has not kept) | 0/1 |
| 52, 53 | mulligans taken me, opp | /7 |
| 54, 55 | kept me, opp | 0/1 |
| 56-65 | decision family one-hot (from the first masked action): priority, target, attack scan, block scan, damage assignment, mulligan, bottom, trigger order, legend rule, other | |
| 66 | stack size | /5 |
| 67, 68 | attackers declared, blockers declared | 0/1 |
| 69 | attackers | /5 |
| 70, 71 | policy stage: attacker inclusion scan, blocker inclusion scan | 0/1 |
| 72 | scan progress candidate_index / candidate_count | |
| 73 | scan candidate count | /10 |
| 74 | scan selected so far | /5 |
| 75 | masked actions, ln(1 + n) | /5 |
| 76 | I am on the play (seat P0) | 0/1 |
| 77 | a cast or activation is pending | 0/1 |
| 78 | an effect choice is pending | 0/1 |
| 79, 80, 81 | pending target choice: min, max, selected so far | /5 |
| 82 | Foundations combat-damage assignment in progress | 0/1 |
| 83 | known cards in the opponent's hand | /5 |
| 84, 85 | top of the stack is mine, the opponent's | 0/1 |
| 86, 87 | priority passed me, opp | 0/1 |
| 88, 89 | me, opp chosen as a target of the pending cast/activation/effect | 0/1 |
| 90, 91 | has lost me, opp | 0/1 |
| 92-95 | reserved (0) | |

## Object tokens

One token per visible object, in this order: my hand; stack items (top first); battlefield (my list, then the
opponent's; the zone category follows the controller); the actor's known cards in the opponent's hand, of my
library, then of the opponent's library; graveyards (mine, then the opponent's; by owner); exile (by owner). The
network is invariant to the order.

The cap: at most 255 tokens (`MAX_OBJECTS`). Past it, five groups get quotas (`cap_quotas`): my hand, the stack, my
battlefield, the opponent's battlefield, everything else; groups are served smallest first, each up to its size or
an equal share of what is left, so a small group is kept whole and every non-empty group keeps tokens. Within an
over-quota group the first token of each distinct (card, zone, features) is kept before any duplicate. A dropped
token that has an identical kept token links its action references to that one (the network computes the same
vector for both); one without links nowhere (`unlinked_refs`). Dropped tokens are counted (`dropped_objects`).
Normal play stays far below the cap (the census peaks at 82 tokens; 4,320 `mcts:100`/`pmcts:100` self-play games
never had more than 39 permanents); only runaway token boards (Homunculus Horde with a looter doubles every turn)
reach it, and `dzk play` now ends a game at 100 permanents (`--max-permanents`).

`card` (u16 vocabulary id), `zone` (u8 category), `feat` f32[72].

Zone categories: 0 my hand, 1 my battlefield, 2 opponent's battlefield, 3 my graveyard, 4 opponent's graveyard,
5 my exile, 6 opponent's exile, 7 stack item I control, 8 stack item the opponent controls, 9 known card in the
opponent's hand, 10 known card of my library, 11 known card of the opponent's library (12-15 unused; one-hot
width 16). Pooling groups: my hand (0), my battlefield (1), opponent's battlefield (2), everything else (3).

| Index | Feature | Scale |
| --- | --- | --- |
| 0 | tapped | 0/1 |
| 1 | summoning sick | 0/1 |
| 2 | damage marked | /10 |
| 3, 4 | power, toughness: effective (public permanents) or printed (other cards) | /10 |
| 5, 6 | base power, toughness | /10 |
| 7 | +1/+1 counters | /10 |
| 8 | -1/-1 counters | /10 |
| 9 | other counters (-0/-1, stun, lore) | /10 |
| 10 | attacking | 0/1 |
| 11 | blocking | 0/1 |
| 12 | blocked (an attacker with blockers) | 0/1 |
| 13 | attached to something | 0/1 |
| 14 | attached to a permanent I control | 0/1 |
| 15 | attachments on it | /3 |
| 16 | token | 0/1 |
| 17 | mana value | /10 |
| 18-24 | creature, land, instant, sorcery, enchantment, artifact, planeswalker | 0/1 |
| 25-30 | colours W U B R G, colourless (effective colours for public permanents) | 0/1 |
| 31-44 | flying, reach, haste, vigilance, trample, first strike, double strike, deathtouch, menace, defender, lifelink, hexproof, indestructible, protection from monocolored (effective for public permanents, printed otherwise) | 0/1 |
| 45 | Ward (generic amount /3 public; 1/3 if printed Ward) | |
| 46 | needs more than one blocker | 0/1 |
| 47 | landwalk | 0/1 |
| 48 | flash (printed) | 0/1 |
| 49 | can't be blocked (printed) | 0/1 |
| 50 | entered the battlefield this turn | 0/1 |
| 51 | selected in the current attacker/blocker scan (or an accumulated blocker) | 0/1 |
| 52 | the scan's current candidate | 0/1 |
| 53 | a later candidate of the scan | 0/1 |
| 54 | the attacker whose blockers are being scanned | 0/1 |
| 55 | already chosen as a target of the pending cast/activation/effect | 0/1 |
| 56, 57 | targeted by a stack item I control, the opponent controls | 0/1 |
| 58, 59, 60 | stack item: spell, activated ability, triggered ability (or madness offer) | 0/1 |
| 61 | stack depth from the top | /5 |
| 62 | stack item's targets | /3 |
| 63 | stack item kicked | 0/1 |
| 64 | stack item is a copy | 0/1 |
| 65, 66 | stack item targets me, the opponent (player targets) | 0/1 |
| 67 | known library card's position | /40 |
| 68 | an ability of it was used this turn | 0/1 |
| 69 | skips its next untap | 0/1 |
| 70 | makes mana (printed) | 0/1 |
| 71 | has activated abilities (printed) | 0/1 |

## Action tokens

One token per masked legal action, in the masked order (logit i is the masked action i): `kind` (u8),
`src` / `tgt` (i16 object token index, -1 if none or not a token), `src_card` / `tgt_card` (u16 vocabulary id of
the referenced card, 0 if none; set even when the object has no token), `tgt_player` (u8: 0 none, 1 me, 2 the
opponent), `feat` f32[32].

Kinds (one-hot width 40): 0 pass, 1 play land, 2 cast spell, 3 activate mana ability, 4 activate ability, 5 plot,
6 choose target, 7 choose cost target, 8 choose cast mode, 9 choose kicker, 10 choose spell mode, 11 choose effect
option, 12 choose effect target, 13 finish effect selection, 14 choose effect colour, 15 choose effect number,
16 choose effect boolean, 17 finish target selection, 18 optional cost use, 19 optional cost which, 20 spell copy
payment, 21 spell copy retarget, 22 madness cast, 23 discard, 24 declare attackers, 25 declare blockers for an
attacker, 26 attacker inclusion, 27 blocker inclusion, 28 order triggers, 30 combat damage range, 31 legend rule
keep, 32 next trigger in order, 33 London mulligan, 34 London bottom, 39 other/ambiguous (29, 35-38 unused).

`src` / `tgt` per kind: the source card of casts, abilities, targets and choices; target choices put the target
object in `tgt` (or a player in `tgt_player`); cost targets put the candidate in `tgt`; blocker inclusion: `src` =
blocker, `tgt` = attacker; attacker inclusion: `src` = attacker; declare blockers: `src` = attacker, `tgt` = first
blocker; discard / declare attackers: first and last card; order triggers: the first and the last trigger of the
order (the full permutation is not encoded); damage range: `src` = source, `tgt` = recipient (or `tgt_player`);
legend rule: the kept permanent; bottom: the card.

| Index | Feature | Scale |
| --- | --- | --- |
| 0 | mulligan (vs keep) | 0/1 |
| 1 | mulligans so far | /7 |
| 2 | include (attacker/blocker inclusion) | 0/1 |
| 3, 4, 5 | damage range minimum, maximum, split point | /10 |
| 6 | upper half of the split | 0/1 |
| 7 | yes/no answer (kicker pay, copy payment, effect boolean, optional cost use, retarget, madness cast) | 0/1 |
| 8, 9 | spell mode index, mode count | /3 |
| 10 | ability index | /3 |
| 11 | trigger-order prefix length (or order length) | ln(1 + x) / ln 8 |
| 12 | pending triggers | ln(1 + x) / ln 8 |
| 13 | trigger index | ln(1 + x) / ln 8 |
| 14 | targets remaining | /5 |
| 15, 16, 17 | effect selection: selected so far, min, max | /5 |
| 18, 19, 20 | effect number, its min, max | /10 |
| 21, 22 | effect option index, option count | /5 |
| 23 | cards / attackers / blockers / legend candidates in the action | /5 |
| 24-29 | colour W U B R G C (effect colour, mana choice) | 0/1 |
| 30 | alternative cast mode | 0/1 |
| 31 | cost kind (1-12)/12, or optional-cost choice/4 | |

## Census (gen-0 `mcts:100` self-play, 40 games; random self-play, 30 games)

Objects per decision: mean 24-31, p99 49-62, max 82; unique object tokens (identical tokens, e.g. untapped basic
lands, are computed once by the Rust forward) mean 19. Actions per decision: mean 2.5-2.8, p99 6-7, max 720 (a
simultaneous trigger group of 6 offered as all 720 orders). No action reference was left unlinked and no object
was dropped. Encode cost about 1-2 µs per decision on the M1 (`dzk nnbench`).

## Known limitations

- Order-trigger decisions see only the first and last trigger of each order.
- Graveyards and exile are tokens like the battlefield (no recency feature); the "other" pool mixes graveyards,
  exile, stack and known cards.
- The Foundations combat-damage assignments made so far are not features (only the current option's range is).
- Continuous effects enter only through the effective characteristics (power, toughness, keywords, colours) the
  observation already applies; their durations are not encoded.
