"""
FDN card facts as numbers, for the deck builders.

Every card the 17lands FDN data names (286, five of them basic lands) gets an index and a feature vector built only
from printed facts: mana value, coloured pips, card types, power and toughness, rarity, keywords, mechanics tags,
and the colours a land makes. Nothing here comes from game results or human decks.

Sources: assets/gameplay/FDN_cards.tsv (XMage card facts) and FDN_card_ids.tsv (17lands cards.csv, CC BY 4.0).
"""
import csv
import os
import re
from functools import lru_cache

import numpy as np

# DZ_REPO: the checkout whose assets/ and data/ to use, when this package runs from a frozen copy elsewhere
REPO = os.environ.get("DZ_REPO") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COLORS = "WUBRG"
BASICS = {"Plains": "W", "Island": "U", "Swamp": "B", "Mountain": "R", "Forest": "G"}
KEYWORDS = ["Flying", "Trample", "Flash", "Haste", "Vigilance", "FirstStrike", "Lifelink", "Deathtouch",
            "Indestructible", "DoubleStrike", "Menace", "Reach", "Ward", "Defender"]
TAGS = ["spell_targets", "etb_trigger", "token_maker", "counters", "mana_ability_nonland", "pump_until_eot", "draw",
        "p1p1_counters", "lifegain", "noncombat_damage", "etb_tapped", "sacrifice", "search_library", "dies_trigger",
        "scry_surveil", "discard", "modal", "exile", "equipment", "gy_to_battlefield", "aura", "copy",
        "bounce_to_hand", "counterspell", "cast_from_other_zone", "kicker_or_alt_cost", "gain_control", "fight",
        "x_cost"]
TYPES = ["Creature", "Instant", "Sorcery", "Enchantment", "Artifact", "Planeswalker", "Land"]
RARITIES = ["common", "uncommon", "rare", "mythic", "basic"]


def _rows(name):
    with open(os.path.join(REPO, "assets/gameplay", name)) as f:
        return list(csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t"))


class Cards:
    """The card table: names, indices, printed facts and the feature matrix X (n_cards x n_features)."""

    def __init__(self, names):
        facts = {r["name"]: r for r in _rows("FDN_cards.tsv")}
        ids = {}
        for r in _rows("FDN_card_ids.tsv"):
            if r["expansion"] in ("FDN", "SPG") or r["name"] in BASICS:
                ids.setdefault(r["name"], r)
        self.names = list(names)
        self.index = {n: i for i, n in enumerate(self.names)}
        n = len(self.names)
        self.mv = np.zeros(n)
        self.pips = np.zeros((n, 5))           # coloured pips per colour (hybrid counts half to each)
        self.colors = np.zeros((n, 5), bool)   # colour identity of the cost
        self.makes = np.zeros((n, 5))          # colours a land (or mana rock/dork) can make
        self.types = np.zeros((n, len(TYPES)), bool)
        self.is_land = np.zeros(n, bool)
        self.is_basic = np.zeros(n, bool)
        self.is_creature = np.zeros(n, bool)
        self.rarity = []
        feats = []
        for i, name in enumerate(self.names):
            fa, idr = facts[name], ids[name]
            cost = fa["mana_cost"]
            for sym in re.findall(r"\{([^}]*)\}", cost):
                parts = [p for p in sym.split("/") if p in COLORS]
                for p in parts:
                    self.pips[i, COLORS.index(p)] += 1 / len(parts)
                    self.colors[i, COLORS.index(p)] = True
            self.mv[i] = float(idr["mana_value"] or 0)
            tl = idr["types"]
            for t, ty in enumerate(TYPES):
                self.types[i, t] = ty in tl
            self.is_land[i] = "Land" in tl
            self.is_basic[i] = name in BASICS
            self.is_creature[i] = "Creature" in tl
            lm = BASICS.get(name, fa["land_mana"])
            if name == "Evolving Wilds":        # fetches a basic: any colour the deck's basics make
                lm = "WUBRG"
            for ch in lm:
                if ch in COLORS:
                    self.makes[i, COLORS.index(ch)] = 1
            self.rarity.append(idr["rarity"])
            pw = fa["power"]; th = fa["toughness"]
            kws = set(fa["keywords"].split(",")); tags = set(fa["features"].split(","))
            f = [self.mv[i] / 8, *(self.pips[i] / 3), *self.colors[i], *self.makes[i], *self.types[i],
                 float(pw) / 6 if pw.lstrip("-").isdigit() else 0, float(th) / 6 if th.lstrip("-").isdigit() else 0,
                 *(idr["rarity"] == r for r in RARITIES), *(k in kws for k in KEYWORDS), *(t in tags for t in TAGS),
                 "etb_tapped" in tags, self.colors[i].sum() == 0 and not self.is_land[i], self.colors[i].sum() > 1]
            feats.append(f)
        self.X = np.asarray(feats, dtype=np.float32)
        self.spell = ~self.is_land

    def __len__(self):
        return len(self.names)

    def vec(self, counts: dict) -> np.ndarray:
        """A {name: count} deck or pool as a count vector."""
        v = np.zeros(len(self.names), np.float32)
        for c, k in counts.items():
            v[self.index[c]] += k
        return v

    def dict(self, v) -> dict:
        return {self.names[i]: int(round(v[i])) for i in np.nonzero(np.round(v))[0]}


@lru_cache(None)
def load() -> Cards:
    """The FDN table in 17lands' column order (alphabetical), basics included."""
    names = sorted(r["name"] for r in _rows("FDN_cards.tsv") if r["xmage_set"] == "FDN" or r["name"] in BASICS)
    hdr = os.path.join(REPO, "assets/sealed/FDN_cardnames.txt")   # 17lands column order
    if os.path.exists(hdr):
        names = [l.rstrip("\n") for l in open(hdr) if l.strip()]
    return Cards(names)
