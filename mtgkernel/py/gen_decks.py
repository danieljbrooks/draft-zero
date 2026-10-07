"""Generate 40-card FDN decks from the cards mtg-kernel fully supports (no new cards).

The supported pool is a white/blue/green mini-format: 38 non-basic FDN names on mtg-kernel main
(a4e1474), only one of them red and none black. Each deck is two-colour (WG, UG or WU): 23 spells
drawn by **uniform random inclusion** under per-card copy caps and curve constraints, then 17 lands
(the pair's gainland, if supported, plus basics split by coloured pips).

Uniform random inclusion is deliberate. 17lands GIH win rates mix a card's strength with the
strength of the decks that play it; here every on-colour card is equally likely to be in any deck,
so a card's win rate in self-play measures the card, not the drafter's choices.

    python3 mtgkernel/py/gen_decks.py --out mtgkernel/decks/gen_v1 --per-pair 40 --eval-per-pair 10 --seed 1

Writes one .dck per deck (draft-zero's format: `N [SET:NUM] Name`, each card's most common printing),
manifest.json, and pair lists pairs_train.tsv / pairs_eval.tsv (deck1<TAB>deck2 stems).
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CARDS = HERE.parent / "data" / "fdn_supported_cards.json"

PAIRS = {"WG": "WG", "UG": "UG", "WU": "WU"}
BASIC = {"W": "Plains", "U": "Island", "G": "Forest"}
# Red has one supported card and black none, so neither can make a two-colour deck.
EXCLUDE = {"Goblin Bushwhacker"}
CAP = {"common": 3, "uncommon": 2, "rare": 1, "mythic": 1}


def load_pool(path=CARDS):
    data = json.loads(Path(path).read_text())
    cards = [c for c in data["cards"] if c["status_main"] == "full" and c["name"] not in EXCLUDE]
    return cards


def pips(cost):
    """Coloured pips in a mana cost like {2}{G}{G}."""
    out = Counter()
    for sym in cost.replace("}", "").split("{"):
        if sym in BASIC:
            out[sym] += 1
    return out


def cap(card):
    return 1 if card["type_line"].startswith("Legendary") else CAP[card["rarity"]]


def on_colour(card, pair):
    return card["colors"] and set(card["colors"]) <= set(pair)


def ok(spells, pair):
    creatures = sum(c["type_line"].split(" — ")[0].endswith("Creature") for c in spells)
    cheap = sum(c["mana_value"] <= 2 for c in spells)
    top = sum(c["mana_value"] >= 5 for c in spells)
    by_colour = Counter(col for c in spells for col in c["colors"])
    return (14 <= creatures <= 17 and cheap >= 5 and top <= 4
            and all(by_colour[col] >= 5 for col in pair))


def sample_deck(pool, pair, rng, tries=10000):
    slots = [c for c in pool if not c["is_land"] and on_colour(c, pair) for _ in range(cap(c))]
    for _ in range(tries):
        spells = rng.sample(slots, 23)
        if ok(spells, pair):
            break
    else:
        raise RuntimeError(f"no {pair} deck satisfies the constraints")
    lands = [c for c in pool if c["is_land"] and set(c["colors"]) == set(pair)]
    deck = Counter(c["name"] for c in spells)
    n_basic = 17 - len(lands)
    for land in lands:
        deck[land["name"]] += 1
    p = Counter()
    for c in spells:
        p.update(pips(c["mana_cost"]))
    a, b = pair
    share = p[a] / max(1, p[a] + p[b])
    n_a = min(n_basic - 6, max(6, round(n_basic * share)))
    deck[BASIC[a]] += n_a
    deck[BASIC[b]] += n_basic - n_a
    assert sum(deck.values()) == 40
    return deck


BASIC_PRINTING = {"Plains": "FDN:272", "Island": "FDN:274", "Forest": "FDN:280"}


def write_dck(path, name, deck, printing):
    """draft-zero's .dck lines, `N [SET:NUM] Name` (card_stats.py reads only lines with a printing)."""
    lines = [f"NAME:{name}"] + [f"{n} [{printing[card]}] {card}" for card, n in sorted(deck.items())]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-pair", type=int, default=40, help="decks per colour pair")
    ap.add_argument("--eval-per-pair", type=int, default=10, help="of those, held out for evaluation")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--pairings-per-deck", type=int, default=4,
                    help="train pairings: each deck meets this many random opponents")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = load_pool()
    printing = {c["name"]: c["most_common_printing"] for c in pool} | BASIC_PRINTING
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "source": str(CARDS.relative_to(HERE.parent.parent)),
                "excluded": sorted(EXCLUDE), "decks": []}
    split = {"train": [], "eval": []}
    for pair in PAIRS:
        for i in range(args.per_pair):
            stem = f"GEN_{args.tag}_{pair}_{i:03d}"
            deck = sample_deck(pool, pair, rng)
            write_dck(out / f"{stem}.dck", f"Generated {pair} deck {i} ({args.tag}, seed {args.seed})", deck, printing)
            part = "eval" if i < args.eval_per_pair else "train"
            split[part].append(stem)
            manifest["decks"].append({"stem": stem, "pair": pair, "split": part, "cards": dict(deck)})
    for part, stems in split.items():
        k = args.pairings_per_deck if part == "train" else 2
        lines = []
        for r in range(k):
            order = stems[:]
            rng.shuffle(order)
            # Pair neighbours in a shuffled ring: every deck appears twice per round, once per seat.
            lines += [f"{order[j]}\t{order[(j + 1) % len(order)]}" for j in range(len(order))]
        (out / f"pairs_{part}.tsv").write_text("\n".join(lines) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {len(manifest['decks'])} decks to {out}: "
          f"{len(split['train'])} train, {len(split['eval'])} eval")


if __name__ == "__main__":
    main()
