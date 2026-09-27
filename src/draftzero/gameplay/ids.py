"""
Card and ability identity for 17lands / Arena ids: grpId -> XMage card or token, ability id ->
source card(s) and MageZero action key.

17lands replay data records cards as Arena card ids ("grpIds") and abilities as Arena ability ids.
XMage identifies cards by name (+ set and collector number), tokens by class
(mage.game.permanent.token.*), and MageZero's policy head indexes priority actions by
`ability.toString()` through assets/vocab/<SET>_SPG.tsv ("Cast X", "Play X", activated-ability
rule text). This module joins them:

  CardRef     grpId -> name, kind (card | basic | token | spg | unknown), XMage set/number,
              token class, "Cast X" / "Play X" key and vocab index
  AbilityRef  ability id -> category (triggered | activated | mana | unknown), source card(s),
              XMage key and vocab index for activated non-mana abilities (the only abilities that
              are MCTS priority actions under autoTap), confidence and method
  CardInfo    XMage-side facts per card name (P/T, keywords, mechanics, flashback key, mana cost,
              land colours, attachment kind) that reconstruction and labelling need

Inputs, all loaded lazily by `Ids.load()`:
  data/17lands/cards.csv, abilities.csv            17lands (CC BY 4.0); `fetch()` downloads them
  data/17lands/xmage_Foundations_SpecialGuests.json  name -> XMage (set, number), tools/extract_decks.py
  assets/vocab/FDN_SPG.tsv                          MageZero action vocabulary
  assets/gameplay/FDN_{tokens,abilities,cards}.tsv  committed derived tables (see README there)

Rules the tables encode (docs/008 research, id_mapping.md):
  - Tokens are mapped BY ID, never by name: Cat 94156 is 1/1 (CatToken3), 94157 is 2/2 (CatToken).
  - Basic lands come from ~56 cosmetic sets: they are classified by the "Basic Land" type line.
  - 17lands records a flashback cast as an ordinary cast; `cast_key()` returns "Flashback {cost}"
    when no copy is left in hand but one is in the graveyard (critique §3.9).
  - One Arena ability can map to different vocab slots per source (Equip {1} with and without
    reminder text): `Ids.ability_key(aid, board)` picks the source present on the board.
  - Reconstruction also needs what the replay leaves implicit: lands that enter tapped, "doesn't
    untap", upkeep draws, "enters with N +1/+1 counters", counter-doubling triggers, and which
    activated abilities tap their source (`cost_taps_source`).

Usage:
  python -m draftzero.gameplay.ids fetch --set FDN --format PremierDraft --what cards replay
  python -m draftzero.gameplay.ids lookup 93965 175817 94177
  python -m draftzero.gameplay.ids check
  python -m draftzero.gameplay.ids build-tables --research <research>/id_mapping \\
      --mage-src ~/Desktop/Code/DEPRECATED_PREDRAFTZERO_MageZero-Experiments/pr-dense-vocab/mage

17lands public data is CC BY 4.0 (https://www.17lands.com/public_datasets): credit 17lands in
anything built from it. Nothing here reads or stores the local Arena client database.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import struct
import sys
import urllib.request
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("SEVENTEENLANDS_DIR", REPO / "data" / "17lands"))
ASSET_DIR = REPO / "assets" / "gameplay"
VOCAB_DIR = REPO / "assets" / "vocab"
S3 = "https://17lands-public.s3.amazonaws.com/analysis_data"
URLS = {
    "cards": f"{S3}/cards/cards.csv",
    "abilities": f"{S3}/cards/abilities.csv",
    "replay": f"{S3}/replay_data/replay_data_public.{{set}}.{{format}}.csv.gz",
    "game": f"{S3}/game_data/game_data_public.{{set}}.{{format}}.csv.gz",
}
BASIC_MANA = {"Plains": "W", "Island": "U", "Swamp": "B", "Mountain": "R", "Forest": "G", "Wastes": "C"}
COLORS = "WUBRG"
KINDS = ("card", "basic", "token", "spg", "unknown")
CATEGORIES = ("triggered", "activated", "mana", "unknown")


# --- download -------------------------------------------------------------------------------------

def data_path(what: str, set_code: str = "FDN", fmt: str = "PremierDraft", data_dir=None) -> Path:
    d = Path(data_dir or DATA_DIR)
    if what in ("cards", "abilities"):
        return d / f"{what}.csv"
    return d / f"{what}_data_public.{set_code}.{fmt}.csv.gz"


def fetch(what=("cards", "abilities", "replay"), set_code: str = "FDN", fmt: str = "PremierDraft",
          data_dir=None, force: bool = False) -> list[Path]:
    """Download 17lands files into data/17lands (skipping ones already present). The replay file
    for FDN Premier is 438 MB. cards.csv / abilities.csv are re-issued by 17lands as new sets land,
    which is harmless: ids are never reused."""
    out = []
    for w in what:
        path = data_path(w, set_code, fmt, data_dir)
        if force or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = URLS[w].format(set=set_code, format=fmt)
            print(f"downloading {url}", file=sys.stderr)
            urllib.request.urlretrieve(url, str(path) + ".part")
            os.replace(str(path) + ".part", path)
        out.append(path)
    return out


# --- vocab ------------------------------------------------------------------------------------------

def java_string_hash(s: str) -> int:
    """Java String.hashCode over UTF-16 code units (ActionEncoder hashes out-of-vocab keys)."""
    h = 0
    b = s.encode("utf-16-le")
    for unit in struct.unpack(f"<{len(b) // 2}H", b):
        h = (31 * h + unit) & 0xFFFFFFFF
    return h - (1 << 32) if h >= 1 << 31 else h


@dataclass
class Vocab:
    actions: dict[str, int]
    targets: dict[str, int]
    dim: int

    @classmethod
    def load(cls, path) -> "Vocab":
        actions, targets, dim = {}, {}, 1024
        for line in open(path, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 2)
            if parts[0] == "dim":
                dim = int(parts[1])
            elif parts[0] == "A":
                actions[parts[2].replace("\\n", "\n")] = int(parts[1])
            elif parts[0] == "T":
                targets[parts[2].replace("\\n", "\n")] = int(parts[1])
        return cls(actions, targets, dim)

    def _index(self, key: str, table: dict[str, int]) -> tuple[int, bool]:
        # ActionEncoder.getActionIndex: exact slot, else Java hash into [vocab size, dim)
        if key in table:
            return table[key], True
        start = max(table.values()) + 1
        return start + (java_string_hash(key) % (self.dim - start)), False

    def action(self, key: str) -> tuple[int, bool]:
        return self._index(key, self.actions)

    def target(self, key: str) -> tuple[int, bool]:
        return self._index(key, self.targets)


# --- text normalisation (Arena markup <-> XMage rule text) -----------------------------------------

_ARENA_MANA = re.compile(r"\{((?:o(?:\([^)]*\)|\d+|[A-Z]))+)\}")
_SELF_REFS = (r"(\{this\}|this creature|this artifact|this enchantment|this land|this token|this card|"
              r"this equipment|this vehicle|this aura)")


def arena_to_xmage(text: str | None) -> str | None:
    """{o2oG} -> {2}{G}, {oT} -> {T}, CARDNAME -> {this}; drop Arena layout tags."""
    if text is None:
        return None
    t = _ARENA_MANA.sub(lambda m: "".join("{" + s.strip("()") + "}"
                                          for s in re.findall(r"o(\([^)]*\)|\d+|[A-Z])", m.group(1))), text)
    t = re.sub(r"</?(nobr|indent[^>]*|b|sprite[^>]*|size[^>]*|color[^>]*)>", "", t)
    t = t.replace("CARDNAME", "{this}")
    return re.sub(r"[ \t]+", " ", t).strip()


def norm(text: str | None, name: str | None = None) -> str:
    """Aggressive normalisation for comparing Arena text, XMage rules and vocab keys."""
    if text is None:
        return ""
    t = arena_to_xmage(text)
    t = t.replace("&mdash;", "—").replace("&bull;", "•").replace("&bull", "•").replace("&nbsp;", " ")
    t = re.sub(r"<br\s*/?>", " ", t)
    t = re.sub(r"<i>\(.*?\)</i>", " ", t)          # XMage reminder text
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"\((?:[^()]*)\)", " ", t)
    if name:
        t = t.replace(name, "{this}")
    t = t.lower()
    t = re.sub(_SELF_REFS, "SELF", t)
    t = t.replace("’", "'").replace("—", "-").replace("−", "-").replace("•", " ")
    return re.sub(r"\s+", " ", t).strip(" .")


# --- records ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CardRef:
    grpId: int
    name: str
    kind: str                      # one of KINDS
    types: str = ""                # 17lands type line, e.g. "Creature - Cat Soldier"
    mana_value: int = 0
    xmage_set: str | None = None
    xmage_num: str | None = None
    token_class: str | None = None  # simple class name under mage.game.permanent.token
    cast_key: str | None = None    # "Cast <name>" (non-land cards)
    play_key: str | None = None    # "Play <name>" (lands)
    vocab_idx: int | None = None   # action index of cast_key / play_key
    vocab_exact: bool = False

    @property
    def is_land(self) -> bool:
        return "Land" in self.types

    @property
    def is_creature(self) -> bool:
        return "Creature" in self.types

    @property
    def is_token(self) -> bool:
        return self.kind == "token"

    @property
    def key(self) -> str | None:
        return self.play_key or self.cast_key


@dataclass(frozen=True)
class AbilityRef:
    id: int
    text: str | None               # abilities.csv text (None for the 26 ids 17lands lacks)
    category: str                  # one of CATEGORIES
    source_cards: tuple[str, ...]
    xmage_key: str | None          # activated non-mana: key of the most common source
    vocab_idx: int | None
    vocab_exact: bool
    method: str                    # how the source was found: arena_db | cooccurrence
    confidence: str                # high | weak | multi | basic_land
    keys_by_source: tuple[tuple[str, str, int, bool], ...] = ()   # (source, key, idx, exact)
    self_p1p1: int = 0             # +1/+1 counters it puts on its own source per resolution
    counter_other: bool = False    # puts counters on something other than its source
    loyalty: int | None = None     # planeswalker loyalty cost (+1, -2, ...)
    note: str = ""
    text_flags: frozenset = frozenset()   # from the 17lands text: "counter_on", "mill_surveil", "attack_trigger"
    self_double: bool = False      # doubles the +1/+1 counters on its source (Mossborn Hydra's landfall)

    @property
    def is_action(self) -> bool:
        """A MageZero priority action (activated, not a mana ability)."""
        return self.category == "activated"


@dataclass(frozen=True)
class CardInfo:
    """XMage facts about one card name (assets/gameplay/<SET>_cards.tsv)."""
    name: str
    xmage_set: str = ""
    xmage_num: str = ""
    mana_cost: str = ""
    power: int | None = None
    toughness: int | None = None
    loyalty: int | None = None
    keywords: frozenset = frozenset()     # combat keywords: Flying, Vigilance, Deathtouch, ...
    features: frozenset = frozenset()     # mechanics regex tags (card_features.py of the research)
    flashback_key: str | None = None      # e.g. "Flashback {2}{U}"
    land_mana: str = ""                   # colours a land can produce, e.g. "WU"; "C" colourless
    attach: str = ""                      # aura:<hostile|freeze|friendly|control>:<creature|land|permanent|player>, equipment[:etb]
    counter_mode: str = "none"            # none | self_tracked | untracked | loyalty
    etb_counters: str = ""                # "{this} enters with N +1/+1 counters": "N", "X", or "" (none / conditional)

    @property
    def pips(self) -> Counter:
        """Coloured mana symbols in the mana cost (hybrid and phyrexian counted as their first colour)."""
        c = Counter()
        for sym in re.findall(r"\{([^}]+)\}", self.mana_cost or ""):
            for ch in sym:
                if ch in COLORS:
                    c[ch] += 1
                    break
        return c


def _kind(row: dict) -> str:
    if row is None:
        return "unknown"
    if row["rarity"] == "token":
        return "token"
    if "Basic Land" in row["types"]:
        return "basic"
    if row["expansion"] == "SPG":
        return "spg"
    return "card"


# --- the joined map --------------------------------------------------------------------------------

class Ids:
    """All id lookups for one set. Build with `Ids.load()` (cached)."""

    def __init__(self, cards: dict[int, dict], ability_text: dict[int, str], name_map: dict,
                 vocab: Vocab, tokens: dict[int, dict], abilities: dict[int, dict],
                 infos: dict[str, CardInfo], set_code: str = "FDN"):
        self.set_code = set_code
        self.cards17 = cards
        self.ability_text = ability_text
        self.name_map = name_map
        self.vocab = vocab
        self.token_rows = tokens
        self.ability_rows = abilities
        self.infos = infos
        self._card_cache: dict[int, CardRef] = {}
        self._ability_cache: dict[int, AbilityRef] = {}
        self._name_cache: dict[int, str] = {}

    @classmethod
    def load(cls, set_code: str = "FDN", data_dir=None, asset_dir=None, vocab_path=None) -> "Ids":
        return _load_cached(set_code, str(data_dir or DATA_DIR), str(asset_dir or ASSET_DIR),
                            str(vocab_path or VOCAB_DIR / f"{set_code}_SPG.tsv"))

    # cards -------------------------------------------------------------------------------------

    def card(self, grp_id: int) -> CardRef:
        ref = self._card_cache.get(grp_id)
        if ref is None:
            ref = self._card_cache[grp_id] = self._make_card(grp_id)
        return ref

    def _make_card(self, gid: int) -> CardRef:
        r = self.cards17.get(gid)
        if r is None:
            return CardRef(gid, f"?{gid}", "unknown")
        kind = _kind(r)
        name, types = r["name"], r["types"]
        try:
            mv = int(float(r["mana_value"] or 0))
        except ValueError:
            mv = 0
        if kind == "token":
            t = self.token_rows.get(gid, {})
            return CardRef(gid, name, kind, types, mv, token_class=t.get("token_class") or None)
        xs = self.name_map.get(name)
        is_land = "Land" in types
        key = ("Play " if is_land else "Cast ") + name
        idx, exact = self.vocab.action(key)
        return CardRef(gid, name, kind, types, mv,
                       xmage_set=xs[0] if xs else None, xmage_num=str(xs[1]) if xs else None,
                       cast_key=None if is_land else key, play_key=key if is_land else None,
                       vocab_idx=idx, vocab_exact=exact)

    def name(self, grp_id: int) -> str:
        n = self._name_cache.get(grp_id)
        if n is None:
            r = self.cards17.get(grp_id)
            n = self._name_cache[grp_id] = r["name"] if r else f"?{grp_id}"
        return n

    def types(self, grp_id: int) -> str:
        r = self.cards17.get(grp_id)
        return r["types"] if r else ""

    def is_token(self, grp_id: int) -> bool:
        r = self.cards17.get(grp_id)
        # NB: expansion codes starting with 'T' are not a token test (THB, TDM, TLA are sets)
        return bool(r) and r["rarity"] == "token"

    def is_land(self, grp_id: int) -> bool:
        return "Land" in self.types(grp_id)

    def is_creature(self, grp_id: int) -> bool:
        return "Creature" in self.types(grp_id)

    def mv(self, grp_id: int) -> int:
        r = self.cards17.get(grp_id)
        try:
            return int(r["mana_value"]) if r else 0
        except ValueError:
            return 0

    def info(self, name: str) -> CardInfo:
        return self.infos.get(name) or _EMPTY_INFO.get(name) or _empty_info(name)

    def info_id(self, grp_id: int) -> CardInfo:
        return self.info(self.name(grp_id))

    def token_names(self) -> set[str]:
        return {t["name"] for t in self.token_rows.values()}

    def is_mapped(self, grp_id: int) -> bool:
        """Maps to an XMage object (card with a set/number, or token with a class)."""
        c = self.card(grp_id)
        return bool(c.token_class) if c.kind == "token" else c.xmage_set is not None

    # abilities ---------------------------------------------------------------------------------

    def ability(self, aid: int) -> AbilityRef:
        ref = self._ability_cache.get(aid)
        if ref is None:
            ref = self._ability_cache[aid] = self._make_ability(aid)
        return ref

    def _make_ability(self, aid: int) -> AbilityRef:
        r = self.ability_rows.get(aid)
        text = self.ability_text.get(aid)
        if r is None:
            return AbilityRef(aid, text, _category_from_text(text), (), None, None, False, "", "", (),
                              text_flags=frozenset(_text_flags(text)))
        by_src = tuple((s, k, *self.vocab.action(k)) for s, k in r["keys"])
        key = by_src[0][1] if by_src else None
        idx, exact = (by_src[0][2], by_src[0][3]) if by_src else (None, False)
        return AbilityRef(aid, text, r["category"], tuple(r["sources"]), key, idx, exact, r["method"],
                          r["confidence"], by_src, r["self_p1p1"], r["counter_other"], r["loyalty"], r["note"],
                          frozenset(r["text_flags"]), r["self_double"])

    def ability_key(self, aid: int, board: "Counter | set | list | None" = None) -> tuple[str | None, int | None, bool, bool]:
        """(key, vocab idx, vocab exact, source unique) for an ability event. With several sources
        (Equip {1} on four equipment, two of which spell the key with reminder text), the source
        is chosen among the names on `board` (the controller's permanents); `source unique` is
        False when the board does not settle it."""
        a = self.ability(aid)
        if not a.keys_by_source:
            return None, None, False, False
        keys = {k for _, k, _, _ in a.keys_by_source}
        if len(keys) == 1:
            s, k, i, e = a.keys_by_source[0]
            return k, i, e, True
        present = [x for x in a.keys_by_source if board and x[0] in board]
        pk = {k for _, k, _, _ in present}
        if len(pk) == 1:
            s, k, i, e = present[0]
            return k, i, e, True
        s, k, i, e = (present or a.keys_by_source)[0]
        return k, i, e, False

    # casts -------------------------------------------------------------------------------------

    def cast_key(self, name: str, hand: Counter | None = None, graveyard: Counter | None = None) -> tuple[str, int, bool, str]:
        """(key, vocab idx, vocab exact, zone) for a recorded cast of `name`.

        17lands records flashback casts as ordinary casts. A cast of a card with no copy left in
        `hand` but one in `graveyard` is a flashback cast when the card has flashback; zone is then
        "graveyard" and the key "Flashback {cost}". Otherwise "Cast <name>" from zone "hand" (or
        "unknown" when the hand has no copy either: exile-cast, stolen cards)."""
        info = self.info(name)
        in_hand = hand is None or hand.get(name, 0) > 0
        if not in_hand and graveyard is not None and graveyard.get(name, 0) > 0 and info.flashback_key:
            idx, exact = self.vocab.action(info.flashback_key)
            return info.flashback_key, idx, exact, "graveyard"
        key = "Cast " + name
        idx, exact = self.vocab.action(key)
        return key, idx, exact, "hand" if in_hand else "unknown"

    def play_key(self, name: str) -> tuple[str, int, bool]:
        key = "Play " + name
        return (key, *self.vocab.action(key))


def is_flashback_cast(name: str, hand: Counter, graveyard: Counter, ids: Ids) -> bool:
    """The critique §3.9 rule: no copy left in hand, one in the graveyard, and the card has flashback."""
    return hand.get(name, 0) <= 0 and graveyard.get(name, 0) > 0 and bool(ids.info(name).flashback_key)


def cost_taps_source(a: AbilityRef) -> bool:
    """An activated ability whose cost includes {T} (the source stays tapped until its controller's
    next untap step). Read off the XMage key, else the 17lands text ("{oT}" markup)."""
    text = a.xmage_key or a.text or ""
    if ":" not in text:
        return False
    cost = text.split(":", 1)[0]
    return "{T}" in cost or "{oT}" in cost


def _text_flags(text: str | None) -> list[str]:
    """What the research read off an ability's 17lands text (replay_empirics.md §8 flags)."""
    t = (text or "").lower()
    out = []
    if "counter on" in t or "counters on" in t:
        out.append("counter_on")
    if "surveil" in t or "mill" in t:
        out.append("mill_surveil")
    if re.search(r"\bwhen(ever)?\b[^.]*\battacks\b", t):
        out.append("attack_trigger")          # fires as attackers are declared (a block spec skips it)
    return out


def _category_from_text(text: str | None) -> str:
    if not text:
        return "unknown"
    t = text.strip()
    if re.match(r"^(When|Whenever|At the beginning|At end|At the end)", t):
        return "triggered"
    if re.match(r"^\{o?T\}: Add", t):
        return "mana"
    if re.search(r"^[^.]*\}[^.]*:|^[^.:]*Sacrifice[^.:]*:|^Equip|^[+\-−]?\d+:", t):
        return "activated"
    return "unknown"


_EMPTY_INFO: dict[str, CardInfo] = {}


def _empty_info(name: str) -> CardInfo:
    info = CardInfo(name, land_mana=BASIC_MANA.get(name, ""))
    _EMPTY_INFO[name] = info
    return info


# --- table loading ---------------------------------------------------------------------------------

def _read_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        lines = [l for l in f if not l.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t", quoting=csv.QUOTE_NONE))


def load_tokens(path: Path) -> dict[int, dict]:
    out = {}
    for r in _read_tsv(path):
        out[int(r["grpId"])] = {k: r[k] for k in ("name", "power", "toughness", "colors", "token_class",
                                                  "xmage_name", "games", "note")}
    return out


def load_abilities(path: Path) -> dict[int, dict]:
    out = {}
    for r in _read_tsv(path):
        keys = json.loads(r["keys"]) if r["keys"] else []
        out[int(r["id"])] = {
            "category": r["category"], "sources": [s for s in r["sources"].split("; ") if s],
            "method": r["method"], "confidence": r["confidence"], "keys": [tuple(k) for k in keys],
            "self_p1p1": int(r["self_p1p1"] or 0), "counter_other": r["counter_other"] == "1",
            "loyalty": int(r["loyalty"]) if r["loyalty"] else None, "note": r["note"],
            "text_flags": [x for x in r.get("text_flags", "").split(",") if x],
            "self_double": r.get("self_double") == "1"}
    return out


def load_infos(path: Path) -> dict[str, CardInfo]:
    out = {}
    for r in _read_tsv(path):
        ints = {k: (int(r[k]) if r[k] not in ("", None) else None) for k in ("power", "toughness", "loyalty")}
        out[r["name"]] = CardInfo(
            r["name"], r["xmage_set"], r["xmage_num"], r["mana_cost"], ints["power"], ints["toughness"],
            ints["loyalty"], frozenset(x for x in r["keywords"].split(",") if x),
            frozenset(x for x in r["features"].split(",") if x), r["flashback_key"] or None,
            r["land_mana"], r["attach"], r["counter_mode"] or "none", r.get("etb_counters") or "")
    return out


@lru_cache(maxsize=4)
def _load_cached(set_code: str, data_dir: str, asset_dir: str, vocab_path: str) -> Ids:
    d, a = Path(data_dir), Path(asset_dir)
    cards_path, ab_path = d / "cards.csv", d / "abilities.csv"
    if cards_path.exists():
        with open(cards_path, encoding="utf-8", newline="") as f:
            cards = {int(r["id"]): r for r in csv.DictReader(f)}
    else:
        # no 17lands download: the committed table of every card id seen in this set's replays
        cards = {int(r["id"]): r for r in _read_tsv(a / f"{set_code}_card_ids.tsv")}
    text = {}
    if ab_path.exists():
        with open(ab_path, encoding="utf-8", newline="") as f:
            text = {int(r["id"]): r["text"] for r in csv.DictReader(f)}
    nm_path = d / ("xmage_Foundations_SpecialGuests.json" if set_code == "FDN" else f"xmage_{set_code}.json")
    name_map = {k: tuple(v) for k, v in json.load(open(nm_path)).items()} if nm_path.exists() else {}
    infos = load_infos(a / f"{set_code}_cards.tsv")
    if not name_map:   # the card table carries the same (set, number) pairs
        name_map = {n: (i.xmage_set, i.xmage_num) for n, i in infos.items() if i.xmage_set}
    return Ids(cards, text, name_map, Vocab.load(vocab_path), load_tokens(a / f"{set_code}_tokens.tsv"),
               load_abilities(a / f"{set_code}_abilities.tsv"), infos, set_code)


# --- building the committed tables -----------------------------------------------------------------
# The tables were produced once from the research id-mapping run (research/scripts/id_mapping.py:
# a replay scan, an XMage dump of every FDN/SPG card and token class via IdMapDump.java, replay
# co-occurrence, and the local Arena client DB for token P/T and ability->card links). Only derived
# id -> key facts are kept here: no Arena card-database text is written.

# Mechanics tags by regex over the XMage card source (port of research card_features.py; the
# replay_empirics.md §8 flags depend on exactly these patterns).
FEATURE_PATTERNS = {
    'counters': r'CounterType\.|AddCounters|addCounters|DistributeCounter|ProliferateEffect',
    'p1p1_counters': r'CounterType\.P1P1',
    'token_maker': r'CreateTokenEffect|CreateTokenTargetEffect|Token\(\)|TokenImpl|CreateTokenCopy',
    'aura': r'EnchantAbility|SubType\.AURA',
    'equipment': r'EquipAbility|SubType\.EQUIPMENT',
    'flash': r'FlashAbility',
    'bounce_to_hand': r'ReturnToHandTargetEffect|ReturnToHandSourceEffect|ReturnToHandChosen|ReturnToHandFromBattlefield',
    'gy_to_hand': r'ReturnFromGraveyardToHandTargetEffect|ReturnFromYourGraveyardToHand|ReturnSourceFromGraveyardToHand',
    'gy_to_battlefield': r'ReturnFromGraveyardToBattlefield|ReturnSourceFromGraveyardToBattlefield',
    'exile': r'ExileTargetEffect|ExileUntilSourceLeaves|ExileAllEffect|ExileSourceEffect|ExileReturnBattlefield|ExileTargetForSourceEffect|ExileThenReturn',
    'flicker': r'ExileThenReturn|ExileReturnBattlefield|ReturnToBattlefieldUnderOwnerControl|ExileTargetForSourceEffect.*Return',
    'library_put': r'PutOnLibraryTargetEffect|PutOnTopOrBottomLibrary|PutIntoLibraryNFromTop|ShuffleIntoLibrary',
    'mill': r'MillCards|MillCardsTargetEffect|MillCardsEachPlayer',
    'search_library': r'SearchLibrary',
    'scry_surveil': r'ScryEffect|SurveilEffect|LookLibrary',
    'draw': r'DrawCard|DrawDiscard',
    'discard': r'DiscardTargetEffect|DiscardControllerEffect|DiscardCardYouChoose|DiscardEachPlayer|DrawDiscard|DiscardHand',
    'sacrifice': r'SacrificeTargetCost|SacrificeSourceCost|SacrificeEffect|SacrificeAll|SacrificeControllerEffect|SacrificeOpponents',
    'fight': r'FightTargetsEffect|FightTargetSourceEffect',
    'gain_control': r'GainControl',
    'tap_effect': r'TapTargetEffect|DoesntUntap|TapAllEffect|TapEnchantedEffect|DontUntapInControllers',
    'type_change': r'BecomesCreature|AddCardTypeTargetEffect|AddCardTypeSourceEffect|CrewAbility|BecomesCreatureAttached',
    'vehicle_crew': r'CrewAbility',
    'counterspell': r'CounterTargetEffect|CounterUnlessPaysEffect',
    'copy': r'CopyTargetStackObject|CopyPermanentEffect|CreateTokenCopy|CopyEffect',
    'cast_from_other_zone': r'FlashbackAbility|MayCastFromGraveyard|PlayFromNotOwnHand|MayPlayFromExile|CastFromExile|Adventure|OmenCard|Foretell|Escape',
    'pump_until_eot': r'BoostTargetEffect|BoostControlledEffect|BoostSourceEffect|BoostAllEffect',
    'etb_trigger': r'EntersBattlefieldTriggeredAbility|EntersBattlefieldAllTriggeredAbility|EntersBattlefieldControlledTriggered',
    'dies_trigger': r'DiesSourceTriggeredAbility|DiesCreatureTriggeredAbility|DiesThisOrAnother',
    'lifegain': r'GainLifeEffect|LifelinkAbility',
    'noncombat_damage': r'DamageTargetEffect|DamagePlayersEffect|DamageEverythingEffect|DamageAllEffect|LoseLifeOpponentsEffect|LoseLifeTargetEffect',
    'x_cost': r'ManacostVariableValue|GetXValue|VariableManaCost',
    'modal': r'getModes\(\)\.setMinModes|addMode\(',
    'spell_targets': r'getSpellAbility\(\)\.addTarget\(|this\.getSpellAbility\(\)\.getTargets|TargetCreaturePermanent|TargetPermanent|TargetPlayer|TargetAnyTarget',
    'exile_until_leaves': r'ExileUntilSourceLeaves|OnLeaveReturnExiled|ExileUntilSourceLeavesEffect',
    'kicker_or_alt_cost': r'KickerAbility|AlternativeCost|OffspringAbility|ConvokeAbility',
    'mana_ability_nonland': r'ManaAbility|AddManaOfAnyColor|BasicManaAbility|GreenManaAbility|TreasureToken',
    # tapped-state facts for reconstruction (not research tags): a gain land / Guildgate / Temple
    # played on the opponent's turn stays tapped through ours; Slumbering Cerberus skips its untap
    'etb_tapped': r'new EntersBattlefieldTappedAbility\(',
    'doesnt_untap': r'DontUntapInControllersUntapStepSourceEffect',
    # triggers between the untap step and the first main phase, which a main1 entry does not play;
    # an upkeep draw (Scrawling Crawler, Phyrexian Arena) comes before the draw step's card
    'turn_start_trigger': r'BeginningOfUpkeepTriggeredAbility|BeginningOfDrawTriggeredAbility',
    'upkeep_draw': r'BeginningOfUpkeepTriggeredAbility\(\s*new DrawCard',
    'opp_draw_trigger': r'DrawCardOpponentTriggeredAbility',
    # who can block, for the block labels (the engine asks only these): Vampire Soulcaller can't
    # block, Brazen Borrower blocks only flyers, a Pacifism host can neither attack nor block
    # (Witness Protection and Eaten by Piranhas hosts can: they are plain 1/1s)
    'cant_block': r'new CantBlockAbility\(',
    'blocks_only_flyers': r'CanBlockOnlyFlyingAbility',
    'host_cant_attack_block': r'CantAttackBlockAttachedEffect',
}
COMBAT_KW = ['Flying', 'Reach', 'Deathtouch', 'FirstStrike', 'DoubleStrike', 'Trample', 'Vigilance',
             'Lifelink', 'Menace', 'Indestructible', 'Defender', 'Haste', 'Flash', 'Ward']

_SELF_P1P1 = re.compile(r"put (a|an|one|two|three|four|\d+) \+1/\+1 counters? on (\{this\}|it\b|cardname\b)", re.I)
# rules text that places or removes counters (not "counter target spell", "counter it unless")
_COUNTER_RULE = re.compile(r"\b(put|puts|enters with|remove|removes|distribute|double)\b[^.]*\bcounters?\b", re.I)
_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def _self_p1p1(text: str) -> int:
    """+1/+1 counters an ability puts on its own source each time it resolves; 0 when none or when
    the placement is conditional at resolution ("If it was a creature card, put ...")."""
    m = _SELF_P1P1.search(text or "")
    if not m:
        return 0
    sentence = re.split(r"[.:]\s", (text or "")[:m.start()])[-1]
    if re.search(r"\bif (it|that|you do|you)\b", sentence, re.I) and not re.match(r"^(At|When|Whenever)", sentence):
        return 0
    w = m.group(1).lower()
    return _WORDS.get(w) or int(w)


_ETB_COUNTERS = re.compile(r"\{this\} enters with (a|an|one|two|three|four|five|\d+|X) \+1/\+1 counters? on it"
                           r"([^.]*)", re.I)


def _etb_counters(rules: list[str]) -> str:
    """"N" / "X" when the card always enters with that many +1/+1 counters (Mossborn Hydra 1,
    Heroes' Bane 4, Wildwood Scourge X); "" when none or conditional (kicker, raid, "where X is")."""
    for rule in rules:
        m = _ETB_COUNTERS.search(rule or "")
        if m and not re.search(r"\b(if|for each|where)\b", m.group(2), re.I) \
                and not re.search(r"\bif\b", (rule or "")[:m.start()], re.I):
            w = m.group(1)
            return "X" if w == "X" else str(_WORDS.get(w.lower()) or int(w))
    return ""


_SELF_DOUBLE = re.compile(r"double the number of \+1/\+1 counters on \{this\}", re.I)


def _tracked_counter_rule(rule: str, has_self_id: bool, has_double_id: bool) -> bool:
    """A counter rule whose effect on the card's own +1/+1 counters the replay lets us count: a
    fixed "enters with N", or a self-counter / doubling trigger whose ability id the replays log."""
    if _etb_counters([rule]) not in ("", "X"):
        return True
    if _SELF_DOUBLE.search(rule or ""):
        return has_double_id
    return bool(_self_p1p1(rule)) and has_self_id


def _counter_other(text: str) -> bool:
    t = (text or "").lower()
    if "counter" not in t or "counter target" in t:
        return False
    return bool(re.search(r"counters? on (each|target|up to|another|that|those)|double the number of each kind of counter|"
                          r"-1/-1 counter|distribute", t))


def _attach_kind(card: dict) -> str:
    subs = card.get("subtypes", "")
    rules = " ".join(a.get("rule") or "" for a in card.get("abilities", []) if a.get("kind") != "spell")
    low = rules.lower()
    if "Equipment" in subs:
        return "equipment:etb" if re.search(r"when \{this\} enters, attach it", low) else "equipment"
    if "Aura" not in subs:
        return ""
    m = re.search(r"enchant (creature, land, or planeswalker|creature|land|player|permanent)", low)
    what = (m.group(1) if m else "creature").split(",")[0]
    if "you control enchanted" in low:
        mode = "control"
    elif "doesn't untap" in low:
        mode = "freeze"                  # hostile, and the host stays tapped (Starlight Snare)
    elif re.search(r"can't attack|loses all abilities|is a colorless land|gets -", low):
        mode = "hostile"
    else:
        mode = "friendly"
    return f"aura:{mode}:{what}"


def build_tables(research_dir, mage_src, out_dir=ASSET_DIR, set_code: str = "FDN",
                 vocab_path=None, data_dir=None) -> dict[str, int]:
    """Regenerate assets/gameplay/<SET>_{tokens,abilities,cards}.tsv from the research id-mapping
    outputs (token_ids.json, ability_ids.json, xmage_cards.jsonl, xmage_tokens.jsonl) and the XMage
    card source. Returns row counts."""
    research_dir, mage_src, out_dir = Path(research_dir), Path(mage_src).expanduser(), Path(out_dir)
    vocab = Vocab.load(vocab_path or VOCAB_DIR / f"{set_code}_SPG.tsv")
    ddir = Path(data_dir or DATA_DIR)
    with open(ddir / "cards.csv", encoding="utf-8", newline="") as f:
        cards17 = {int(r["id"]): r for r in csv.DictReader(f)}
    with open(ddir / "abilities.csv", encoding="utf-8", newline="") as f:
        text17 = {int(r["id"]): r["text"] for r in csv.DictReader(f)}
    dump: dict[str, dict] = {}
    for line in open(research_dir / "xmage_cards.jsonl", encoding="utf-8"):
        c = json.loads(line)
        dump.setdefault(c["name"], c)
    xtokens = {json.loads(l)["class"]: json.loads(l) for l in open(research_dir / "xmage_tokens.jsonl", encoding="utf-8")}
    out_dir.mkdir(parents=True, exist_ok=True)

    # tokens ------------------------------------------------------------------------------------
    tok = json.load(open(research_dir / "token_ids.json"))
    trows = []
    for g, r in sorted(tok.items(), key=lambda kv: int(kv[0])):
        cls = r.get("xmage_token_class") or ""
        x = xtokens.get(cls, {})
        trows.append({"grpId": g, "name": r["name"], "power": x.get("power", "") if x.get("types", "").find("Creature") >= 0 else "",
                      "toughness": x.get("toughness", "") if x.get("types", "").find("Creature") >= 0 else "",
                      "colors": x.get("color", ""), "token_class": cls.rsplit(".", 1)[-1] if cls else "",
                      "xmage_name": x.get("name", ""), "games": r.get("games", 0),
                      "note": "Arena copy placeholder: token copies carry the copied card's id" if not cls else ""})
    _write_tsv(out_dir / f"{set_code}_tokens.tsv", trows,
               ["grpId", "name", "power", "toughness", "colors", "token_class", "xmage_name", "games", "note"],
               f"# {set_code} token grpId -> XMage token class (mage.game.permanent.token.*); P/T and colour of that class. "
               f"Map tokens by id, never by name.")

    # abilities ---------------------------------------------------------------------------------
    ab = json.load(open(research_dir / "ability_ids.json"))
    tok_names = {r["name"] for r in tok.values()}
    arows = []
    for i, r in sorted(ab.items(), key=lambda kv: -kv[1]["occurrences"]):
        m = r.get("xmage_match") or {}
        # the Arena DB links some shared abilities to cards of other sets; keep FDN/SPG sources
        srcs = [s for s in r["sources"] if s in dump or s in BASIC_MANA or s in tok_names]
        r = dict(r, sources=srcs or r["sources"])
        kind = r["arena_kind"].replace("(inferred)", "")
        cat = {"triggered": "triggered", "activated": "activated", "mana": "mana"}.get(kind, "unknown")
        if m.get("kind") == "embedded" and cat == "activated":
            note = "granted ability: not in the vocab (hashed tail)"
        elif m.get("kind") == "embedded":
            note = "reflexive/delayed/emblem trigger inside another XMage ability"
        else:
            note = ""
        keys = []
        if cat == "activated" and m.get("key") and m.get("kind") != "embedded":
            # per source: the source's own XMage ability with the same normalised text (Equip keys
            # differ in reminder text between equipment)
            want = norm(m["key"], m.get("source"))
            for s in r["sources"]:
                cand = [a["key"] for a in dump.get(s, {}).get("abilities", [])
                        if a["kind"] == "activated" and norm(a["key"], s) == want]
                keys.append([s, cand[0] if cand else m["key"]])
            if not keys:
                keys = [[m.get("source") or "", m["key"]]]
            # most common key first
            order = Counter(k for _, k in keys)
            keys.sort(key=lambda sk: (-order[sk[1]], sk[0]))
        co = r.get("source_cooccurrence") or {}
        conf = "multi" if len(r["sources"]) > 1 else (co.get("confidence") or "high")
        xkey = m.get("key") or ""
        loyalty = re.match(r"^([+\-−]\d+):", xkey)
        in17 = int(i) in text17
        arows.append({
            "id": i, "category": cat, "sources": "; ".join(r["sources"]), "method": r["source_method"] or "",
            "confidence": conf, "in_abilities_csv": int(in17),
            "keys": json.dumps(keys, ensure_ascii=False) if keys else "",
            "vocab_idx": ",".join(str(vocab.action(k)[0]) for _, k in keys),
            "self_p1p1": _self_p1p1(xkey) if m.get("kind") != "embedded" else 0,
            "self_double": int(bool(_SELF_DOUBLE.search(xkey))) if m.get("kind") != "embedded" else 0,
            "counter_other": int(_counter_other(xkey) or _counter_other(text17.get(int(i), ""))),
            "loyalty": loyalty.group(1).replace("−", "-") if loyalty else "",
            "text_flags": ",".join(_text_flags(text17.get(int(i)))),
            "occurrences": r["occurrences"], "note": note})
    _write_tsv(out_dir / f"{set_code}_abilities.tsv", arows,
               ["id", "category", "sources", "method", "confidence", "in_abilities_csv", "keys", "vocab_idx",
                "self_p1p1", "self_double", "counter_other", "loyalty", "text_flags", "occurrences", "note"],
               f"# {set_code} ability id -> source card(s), category and (activated non-mana only) XMage key per source. "
               f"Text: 17lands abilities.csv at runtime; not stored here.")

    # card ids seen in the replays (a subset of 17lands cards.csv, so the package works without it)
    seen = [int(r["id"]) for r in _read_tsv(research_dir / "card_ids.tsv") if r["id"].lstrip("-").isdigit()]
    idrows = [{k: cards17[g][k] for k in ("id", "expansion", "name", "rarity", "types", "mana_value", "is_booster")}
              for g in sorted(set(seen)) if g in cards17]
    _write_tsv(out_dir / f"{set_code}_card_ids.tsv", idrows,
               ["id", "expansion", "name", "rarity", "types", "mana_value", "is_booster"],
               f"# every card grpId seen in the {set_code} replay files, from 17lands cards.csv (CC BY 4.0)")

    # cards -------------------------------------------------------------------------------------
    sets_dir = mage_src / "Mage.Sets" / "src" / "mage" / "sets"
    feats: dict[str, dict] = {}
    for sfile in ("Foundations.java", "SpecialGuests.java"):
        src = (sets_dir / sfile).read_text()
        for mm in re.finditer(r'SetCardInfo\("([^"]+)",\s*"?([^,"]+)"?,\s*Rarity\.(\w+),\s*([\w.]+)\.class', src):
            name, num, rar, cls = mm.groups()
            if name in feats:
                continue
            fp = mage_src / "Mage.Sets" / "src" / Path(*cls.split("."))
            fp = fp.with_suffix(".java")
            code = fp.read_text() if fp.exists() else ""
            fs = [k for k, pat in FEATURE_PATTERNS.items() if code and re.search(pat, code)]
            kw = [k for k in COMBAT_KW if code and re.search(r'\b' + k + r'Ability\.getInstance\(\)|new ' + k + r'Ability\(', code)]
            loy = re.search(r"setStartingLoyalty\((\d+)\)", code)
            feats[name] = {"features": fs, "kw": kw, "loyalty": loy.group(1) if loy else ""}
    name_map = json.load(open(ddir / "xmage_Foundations_SpecialGuests.json")) if set_code == "FDN" else {}
    # which ability ids put counters on their own source, per source card
    self_ids, double_ids = {}, set()
    for r in arows:
        if r["self_p1p1"]:
            for s in r["sources"].split("; "):
                self_ids.setdefault(s, set()).add(r["id"])
        if r["self_double"]:
            double_ids.update(r["sources"].split("; "))
    crows = []
    for name in sorted(set(feats) | set(dump)):
        c = dump.get(name, {})
        f = feats.get(name, {"features": [], "kw": [], "loyalty": ""})
        pt = (c.get("power", ""), c.get("toughness", "")) if "Creature" in c.get("types", "") else ("", "")
        fb = [a["key"] for a in c.get("abilities", []) if a["key"].startswith("Flashback")]
        mana = ""
        for a in c.get("abilities", []):
            if a["kind"] == "mana":
                k = a["key"]
                if "any color" in k or "any one color" in k:
                    mana += COLORS
                mana += "".join(re.findall(r"\{([WUBRGC])\}", k.split(":", 1)[-1]))
        if name in BASIC_MANA:
            mana = BASIC_MANA[name]
        mana = "".join(ch for ch in COLORS + "C" if ch in mana)
        # counters on this card are exact when every counter ability of the card is a tracked
        # self-counter ability id; otherwise only "possible"
        counter_rules = [a for a in c.get("abilities", []) if a["kind"] != "spell"
                         and _COUNTER_RULE.search(a.get("rule") or "")]
        if f["loyalty"]:
            cmode = "loyalty"
        elif not counter_rules:
            cmode = "none"
        elif all(_tracked_counter_rule(a["rule"], name in self_ids, name in double_ids) for a in counter_rules):
            cmode = "self_tracked"
        else:
            cmode = "untracked"
        xs = name_map.get(name) or [c.get("set", ""), c.get("number", "")]
        crows.append({"name": name, "xmage_set": xs[0], "xmage_num": xs[1], "mana_cost": c.get("mana_cost", ""),
                      "power": pt[0], "toughness": pt[1], "loyalty": f["loyalty"], "keywords": ",".join(f["kw"]),
                      "features": ",".join(f["features"]), "flashback_key": fb[0] if fb else "",
                      "land_mana": mana if "Land" in c.get("types", "") or name in BASIC_MANA else "",
                      "attach": _attach_kind(c), "counter_mode": cmode,
                      "etb_counters": _etb_counters([a.get("rule") or "" for a in c.get("abilities", [])
                                                     if a["kind"] != "spell"])})
    _write_tsv(out_dir / f"{set_code}_cards.tsv", crows,
               ["name", "xmage_set", "xmage_num", "mana_cost", "power", "toughness", "loyalty", "keywords",
                "features", "flashback_key", "land_mana", "attach", "counter_mode", "etb_counters"],
               f"# {set_code}+SPG card facts from XMage (mage-1.4.58 source and class dump): printed P/T, combat keywords, "
               f"mechanics tags, flashback key, land colours, attachment kind, counter tracking, +1/+1 counters it enters with.")
    return {"tokens": len(trows), "abilities": len(arows), "cards": len(crows), "card_ids": len(idrows)}


def _write_tsv(path: Path, rows: list[dict], cols: list[str], header: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")).replace("\t", " ").replace("\n", "\\n") for c in cols) + "\n")


# --- checks -----------------------------------------------------------------------------------------

def check(ids: Ids) -> list[str]:
    """Consistency problems in the committed tables (empty = fine)."""
    errs = []
    fdn = [g for g, r in ids.cards17.items() if r["expansion"] == ids.set_code and r["is_booster"] == "True"
           and r["rarity"] != "token"]
    for g in fdn:
        c = ids.card(g)
        if c.kind in ("card", "basic") and c.xmage_set is None:
            errs.append(f"{g} {c.name}: no XMage printing")
        if c.key and not c.vocab_exact and c.kind != "basic":
            errs.append(f"{g} {c.name}: {c.key!r} not in the vocab")
    for g, t in ids.token_rows.items():
        if not t["token_class"] and t["name"] != "Copy":
            errs.append(f"token {g} {t['name']}: no class")
    for aid, r in ids.ability_rows.items():
        if r["category"] == "activated" and not r["keys"] and "granted" not in r["note"]:
            errs.append(f"ability {aid}: activated without a key")
        for s in r["sources"]:
            if s not in ids.infos and s not in ids.token_names():
                errs.append(f"ability {aid}: source {s!r} unknown")
    return errs


# --- CLI --------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m draftzero.gameplay.ids", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download 17lands files into data/17lands")
    f.add_argument("--set", default="FDN")
    f.add_argument("--format", default="PremierDraft")
    f.add_argument("--what", nargs="+", default=["cards", "abilities", "replay"], choices=sorted(URLS))
    f.add_argument("--force", action="store_true")
    lk = sub.add_parser("lookup", help="print the mapping of card / ability ids")
    lk.add_argument("ids", nargs="+", type=int)
    sub.add_parser("check", help="validate the committed tables against cards.csv and the vocab")
    b = sub.add_parser("build-tables", help="regenerate assets/gameplay tables from research outputs")
    b.add_argument("--research", required=True, help="research id_mapping directory")
    b.add_argument("--mage-src", required=True, help="XMage source tree (for mechanics regexes)")
    b.add_argument("--out", default=str(ASSET_DIR))
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        for p in fetch(args.what, args.set, args.format, force=args.force):
            print(p)
        return 0
    if args.cmd == "build-tables":
        print(build_tables(args.research, args.mage_src, args.out))
        return 0
    ids = Ids.load()
    if args.cmd == "lookup":
        for i in args.ids:
            if i in ids.cards17:
                print(ids.card(i))
            if i in ids.ability_rows or i in ids.ability_text:
                print(ids.ability(i))
        return 0
    errs = check(ids)
    for e in errs:
        print(e)
    print(f"{len(errs)} problems; {len(ids.token_rows)} tokens, {len(ids.ability_rows)} abilities, "
          f"{len(ids.infos)} card infos")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
