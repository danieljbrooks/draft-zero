"""
Arena Player.log -> StateSpec v1: one spec per decision the local player made in MTG Arena.

MTG Arena writes its Game Rules Engine (GRE) protocol to Player.log when "Detailed Logs (Plugin
Support)" is on (Options > Account; look for "DETAILED LOGS: ENABLED" near the top of the log).
Arena empties Player.log at every launch and keeps only the previous session as Player-prev.log:

    macOS    ~/Library/Logs/Wizards Of The Coast/MTGA/Player.log
    Windows  %USERPROFILE%/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log

What the log holds (research: arena_log.md and the completeness critique, docs/008):
- GRE->client messages, one JSON line each: GameStateMessages (a Full state per game, then Diffs;
  the local seat's view) and decision requests to the local seat (ActionsAvailableReq,
  DeclareAttackersReq, SelectTargetsReq, ...), each with a msgId.
- client->GRE messages, JSON pretty-printed over many lines: the human's answers. Each echoes the
  request's msgId as respId. Attacks, blocks and targets take several round trips (a toggle, a
  refreshed request, then Submit*Req); they are merged here into one logical decision.
- The legal set of a priority decision is ActionsAvailableReq.actions (plus inactiveActions:
  castable in principle, not now). Its spells are timing-legal but not always affordable: those
  Arena can pay for carry an auto-tap solution (`autotap` in the options). The per-state `actions`
  list in GameStateMessages is NOT a legal set: it covers both seats and ignores timing
  (critique N5), so it is never used.
- Arena drops a whole event when one of its GameStateMessages has more than 50 objects or 50
  annotations, logging a one-line summary instead. That shows up as a break in the
  prevGameStateId chain. Snapshots taken while objects may still be stale are flagged.

Each logical decision becomes one StateSpec (statespec.py). Seat A is the local player, B the
opponent. A's decklist is exact (from ConnectResp). B's is a placeholder (every card B has shown
plus basics of B's colours), and B's hand is a count (handUnknown) unless cards were revealed.
`labels` holds the decision: its type, the Arena-offered options with MageZero action keys
(mz_key, and mz_idx when the key is in assets/vocab/FDN_SPG.tsv) and the human's choice, plus
`labels.bridge`, the bridge request options that reach the same decision (decisionPlayer A, and
decideFrom for attack and block specs, which are entered a step early).

Entry points: a priority decision (ActionsAvailableReq) is entered at its own step, PRIORITY_FRESH
when the active player holds priority over an empty stack, else PRIORITY_HELD. Inside
DECLARE_ATTACKERS and DECLARE_BLOCKERS it is always PRIORITY_HELD: XMage resumes a fresh entry to
those steps by replaying the declaration, so attack and block triggers would fire a second time.
An attack decision is entered one step early (BEGIN_COMBAT, PRIORITY_FRESH) so XMage's
declare-attackers step asks for it; a block decision at DECLARE_ATTACKERS, PRIORITY_HELD, with the
attacks in place (the MCTS anchor fix in xmage_state.md). Other requests (targets, searches, X,
...) arrive mid-action; their specs are the state at that moment, flagged "mid_action", and
`labels.part_of` points to the priority decision whose Cast/Activate led to them. StateSpec has no
controller field, so a permanent is listed under its owner; one controlled by the other player is
flagged "control_changed" (T2) and left out of combat.

Privacy: the log holds the account id (in every header line), screen names, user ids, match ids
and transaction ids. None of them is kept: games are numbered, players are A/B, times are seconds
since the game started. Before anything is written, every identifier found in the log is looked
for in the output text, and the run fails, writing nothing, if one would leak.

Card names come from data/17lands/cards.csv (17lands card ids are Arena grpIds) and, for cards
not in it, from Arena's local card database (Raw_CardDatabase_*.mtga, found automatically; the
one whose GRP build matches the log is preferred). Ability text for mz_key comes from the Arena
database when present, else data/17lands/abilities.csv.

Usage:
    python -m draftzero.gameplay.arena ~/Library/Logs/"Wizards Of The Coast"/MTGA/Player-prev.log
    python -m draftzero.gameplay.arena Player-prev.log Player.log --out data/gameplay/arena/
    python -m draftzero.gameplay.arena Player-prev.log --list-games
    python -m draftzero.gameplay.arena Player.log --card-db /path/to/Raw_CardDatabase_x.mtga

Writes <out>/decisions.jsonl (one StateSpec per decision) and <out>/summary.json (counts only).
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import re
import sqlite3
import statistics
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from draftzero.gameplay.statespec import (Attack, Block, Perm, PlayerState, Provenance, StackItem,
                                          StateSpec)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO / "data" / "gameplay" / "arena"
CARDS_CSV = REPO / "data" / "17lands" / "cards.csv"
ABILITIES_CSV = REPO / "data" / "17lands" / "abilities.csv"
VOCAB_TSV = REPO / "assets" / "vocab" / "FDN_SPG.tsv"
# Token grpId -> XMage token class: the ids module's committed tables (assets/gameplay/<SET>_tokens.tsv).
# Tokens missing from them (other sets) are written by name (Perm.token + set) and counted as unmapped.
TOKEN_TABLES = tuple(sorted((REPO / "assets" / "gameplay").glob("*_tokens.tsv")))

LOG_DIRS = (Path.home() / "Library/Logs/Wizards Of The Coast/MTGA",
            Path.home() / "AppData/LocalLow/Wizards Of The Coast/MTGA")
CARD_DB_GLOBS = (
    "~/Library/Application Support/com.wizards.mtga/Downloads/Raw/Raw_CardDatabase_*.mtga",
    "C:/Program Files/Wizards of the Coast/MTGA/MTGA_Data/Downloads/Raw/Raw_CardDatabase_*.mtga",
    "C:/Program Files (x86)/Wizards of the Coast/MTGA/MTGA_Data/Downloads/Raw/Raw_CardDatabase_*.mtga",
    "C:/Program Files (x86)/Steam/steamapps/common/MTGA/MTGA_Data/Downloads/Raw/Raw_CardDatabase_*.mtga",
    "C:/Program Files/Epic Games/MagicTheGathering/MTGA_Data/Downloads/Raw/Raw_CardDatabase_*.mtga",
)


# ------------------------------------------------------------------------------------------------
# 1. Reading the log
# ------------------------------------------------------------------------------------------------
# Header lines look like "[UnityCrossThreadLogger]<locale timestamp>: Match to <account>: <Kind>"
# (inbound) or "...: <account> to Match: <Kind>" (outbound), followed by the JSON body. The
# timestamp is locale-formatted, so it is optional here and never parsed: the JSON carries its own.
HEADER = re.compile(r"^\[UnityCrossThreadLogger\](?:[^:]*?\d+:\d+:\d+[^:]*?: )?"
                    r"(?P<dir>Match to \S+|\S+ to Match): (?P<kind>\w+)\s*$")
SUMMARIZED = "[Message summarized"
DETAILED = re.compile(r"DETAILED LOGS: (ENABLED|DISABLED)")
CLIENT_VERSION = re.compile(r"^Version: (\d{4}\.\d+\.\d+)")
GRP_VERSION = re.compile(r'"grpVersion":\s*\{\s*"majorVersion":\s*(\d+),\s*"minorVersion":\s*(\d+),'
                         r'\s*"buildVersion":\s*(\d+)')
MAX_BODY_LINES = 100_000


@dataclass
class LogMessage:
    line: int
    direction: str                  # "in" (GRE -> client) or "out" (client -> GRE)
    kind: str                       # GreToClientEvent, ClientToGremessage, ...
    obj: dict | None = None         # the parsed JSON body
    summary: list[str] | None = None  # message types Arena dropped (the 50-object logging limit)


def iter_messages(path) -> Iterator[LogMessage]:
    """Every header+body in the log. GRE->client bodies are one line; client->GRE bodies are
    pretty-printed, so lines are added until the JSON parses."""
    dec = json.JSONDecoder()
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = enumerate(f, 1)
        held = None
        while True:
            item = held or next(lines, None)
            held = None
            if item is None:
                return
            n, line = item
            m = HEADER.match(line)
            if not m:
                continue
            direction = "in" if m.group("dir").startswith("Match to") else "out"
            kind = m.group("kind")
            body = next(lines, None)
            if body is None:
                return
            if body[1].startswith(SUMMARIZED):
                summary = []
                while True:
                    nxt = next(lines, None)
                    if nxt is None or not nxt[1].startswith(":"):
                        held = nxt
                        break
                    summary.append(nxt[1].strip().strip(":").strip())
                yield LogMessage(n, direction, kind, summary=summary)
                if held is None:
                    return
                continue
            if not body[1].lstrip().startswith("{"):
                held = body
                continue
            text, obj = body[1].strip(), None
            for _ in range(MAX_BODY_LINES):
                try:
                    obj = dec.raw_decode(text)[0]
                    break
                except json.JSONDecodeError:
                    pass
                # Pretty-printed JSON indents every line but its closing brace, so only a line at
                # column 0 can complete it: indented lines are appended without re-decoding (which
                # made a truncated body quadratic), and a Unity line means the body was cut short.
                nxt = next(lines, None)
                while nxt is not None and nxt[1][:1] in (" ", "\t"):
                    text += "\n" + nxt[1].rstrip("\n")
                    nxt = next(lines, None)
                if nxt is None:
                    break
                if HEADER.match(nxt[1]) or nxt[1].startswith(("[Unity", "UnityEngine")):
                    held = nxt                 # truncated body: re-read that line as a header
                    break
                text += "\n" + nxt[1].rstrip("\n")
            yield LogMessage(n, direction, kind, obj=obj)


def log_info(path) -> dict:
    """Cheap first pass over raw lines: detailed-logs flag, client version, GRP build."""
    info: dict[str, Any] = {"detailed_logs": None, "client_version": None, "grp_build": None}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if info["detailed_logs"] is None and (m := DETAILED.search(line)):
                info["detailed_logs"] = m.group(1) == "ENABLED"
            elif info["client_version"] is None and (m := CLIENT_VERSION.match(line)):
                info["client_version"] = m.group(1)
            if info["grp_build"] is None and '"grpVersion"' in line and (m := GRP_VERSION.search(line)):
                info["grp_build"] = int(m.group(3))
    return info


def gre_ms(ts) -> int | None:
    """GRE timestamps are Unix epoch ms; client ones are .NET DateTime ticks (100 ns since year 1)."""
    try:
        v = int(ts)
    except (TypeError, ValueError):
        return None
    return (v - 621355968000000000) // 10000 if v > 10 ** 16 else v


# ------------------------------------------------------------------------------------------------
# 2. Privacy: identifiers are collected from the log in memory and must never reach the output
# ------------------------------------------------------------------------------------------------
class PrivacyError(RuntimeError):
    pass


_UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_HEADER_ID = re.compile(r"^\[UnityCrossThreadLogger\].*?(?:Match to (\S+?)|(\S+) to Match): \w+\s*$")
_ID_KEYS = re.compile(r'"(screenName|displayName|playerName|opponentScreenName|opponentPlayerName|'
                      r'userId|accountId|personaId|clientId|sessionId|matchId|matchID|transactionId|'
                      r'playerId|eventInstanceId|courseId|email)"\s*:\s*"([^"]+)"')
_MATCH_LINE = re.compile(r"\b(?:matchId|MatchId|matchID)\b[ =:]+\"?([A-Za-z0-9-]{8,})")
_OS_USER = re.compile(r"(?:/Users/|/home/|\\Users\\|\\\\Users\\\\)([^/\\\s\"']+)")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_NOT_USERS = {"Shared", "Public", "Default", "All Users"}


@dataclass
class Identifiers:
    """Strings from the log that identify a person or a match. Held in memory only."""
    exact: dict[str, set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    words: dict[str, set[str]] = field(default_factory=lambda: collections.defaultdict(set))

    def add(self, category: str, value) -> None:
        v = str(value).strip()
        name = category.lower().endswith("name")
        if len(v) < (3 if name else 4) or (v.isdigit() and len(v) < 8):
            return          # too short to search for without matching card text or numbers
        if not name:
            self.exact[category].add(v)
        elif "#" in v:      # "Name#12345": the full tag is searched as is, the bare name as words
            self.exact[category].add(v)
            bare = v.split("#", 1)[0].strip()
            if len(bare) >= 3:
                self.words[category + "_bare"].add(bare)
        else:
            # Arena dropped the #tag in 2024, so names come bare and are often words that also occur
            # in card names ("Goblin"): search them as whole words, like the bare part of a tag
            self.words[category].add(v)

    def add_line(self, line: str) -> None:
        if (m := _HEADER_ID.match(line)):
            self.add("account_id", m.group(1) or m.group(2))
        for u in _UUID.findall(line):
            self.add("uuid", u)
        for k, v in _ID_KEYS.findall(line):
            self.add(k, v)
            if "\\" in v:               # a JSON-escaped value ("Zoë"): also search it decoded
                try:
                    self.add(k, json.loads(f'"{v}"'))
                except json.JSONDecodeError:
                    pass
        for v in _MATCH_LINE.findall(line):
            self.add("matchId", v)
        for v in _OS_USER.findall(line):
            if v not in _NOT_USERS:
                self.add("os_user", v)
        for v in _EMAIL.findall(line):
            self.add("email", v)

    @classmethod
    def from_logs(cls, paths: Iterable) -> "Identifiers":
        ids = cls()
        for p in paths:
            with open(p, encoding="utf-8", errors="replace") as f:
                for line in f:
                    ids.add_line(line)
        return ids

    def counts(self) -> dict[str, int]:
        out = {k: len(v) for k, v in self.exact.items()}
        out.update({k: len(v) for k, v in self.words.items()})
        return out

    def scan(self, text: str, allowed_words: set[str] = frozenset()) -> dict[str, int]:
        """Category -> number of hits in text. Bare screen names (without the #tag) are matched as
        whole words and ignored when every word of the name is part of a card name or rules text
        we emit."""
        hits: dict[str, int] = {}
        uuids = collections.Counter(_UUID.findall(text))    # thousands of transaction ids: look up
        for cat, vals in self.exact.items():
            n = 0
            for v in vals:
                n += uuids.get(v, 0) if _UUID.fullmatch(v) else text.count(v)
            if n:
                hits[cat] = n
        for cat, vals in self.words.items():
            n = 0
            for v in vals:
                if all(w in allowed_words for w in re.findall(r"\w+", v)):
                    continue
                n += len(re.findall(r"(?<!\w)" + re.escape(v) + r"(?!\w)", text))
            if n:
                hits[cat] = n
        return hits


def check_privacy(texts: Iterable[str], ids: Identifiers, allowed_words: set[str] = frozenset()) -> None:
    """Raise PrivacyError (naming categories and counts, never values) if any text holds an id.
    The texts must be exactly what is written, with non-ASCII characters unescaped (json.dumps
    with ensure_ascii=False): an accented name escaped as \\u00eb would not be found."""
    total: collections.Counter = collections.Counter()
    for t in texts:
        total.update(ids.scan(t, allowed_words))
    if total:
        raise PrivacyError("refusing to write: identifiers from the log found in the output: "
                           + ", ".join(f"{k} x{v}" for k, v in sorted(total.items())))


# ------------------------------------------------------------------------------------------------
# 3. Card names, ability text, token classes and the MageZero vocabulary
# ------------------------------------------------------------------------------------------------
@dataclass
class CardInfo:
    grp: int
    name: str
    set: str = ""
    token: bool = False
    front: bool = True               # False for the back face / secondary half of a multi-part card
    source: str = "cards.csv"


# Arena LinkedFaceType values of a secondary face (checked on DFC, MDFC and adventure cards):
# 1 transform back, 7 adventure half, 9 MDFC back.
_SECONDARY_FACE = {1, 7, 9}


class ArenaCardDB:
    """Read-only view of Arena's Raw_CardDatabase_*.mtga (SQLite). Never copied into the repo."""

    def __init__(self, path):
        self.path = Path(path)
        self.con = sqlite3.connect(f"file:{self.path}?mode=ro&immutable=1", uri=True)
        self.version = dict(self.con.execute("select Type, Version from Versions").fetchall())
        self._loc: dict[int, str | None] = {}
        self._card: dict[int, tuple | None] = {}
        self._abil: dict[int, tuple | None] = {}

    def loc(self, loc_id) -> str | None:
        if loc_id not in self._loc:
            rows = dict(self.con.execute("select Formatted, Loc from Localizations_enUS where LocId=?",
                                         (loc_id,)).fetchall())
            self._loc[loc_id] = rows.get(1) or rows.get(0) or rows.get(2)
        return self._loc[loc_id]

    def _row(self, grp):
        if grp not in self._card:
            self._card[grp] = self.con.execute(
                "select TitleId, ExpansionCode, IsToken, LinkedFaceType, AbilityIds from Cards where GrpId=?",
                (grp,)).fetchone()
        return self._card[grp]

    def card(self, grp) -> CardInfo | None:
        r = self._row(grp)
        if r is None or not (name := self.loc(r[0])):
            return None
        return CardInfo(grp, xmage_name(strip_markup(name)), r[1] or "", bool(r[2]),
                        r[3] not in _SECONDARY_FACE, "arena_db")

    def ability_text(self, ability_id, card_grp=None) -> str | None:
        """Raw Arena rules text. A card-specific text id (Cards.AbilityIds "id:textId") is preferred
        because it carries the card's own wording; loyalty costs live in their own column."""
        if ability_id not in self._abil:
            self._abil[ability_id] = self.con.execute(
                "select TextId, LoyaltyCost from Abilities where Id=?", (ability_id,)).fetchone()
        a = self._abil[ability_id]
        text_id = None
        if card_grp and (r := self._row(card_grp)):
            for pair in (r[4] or "").split(","):
                aid, _, tid = pair.partition(":")
                if aid.strip() == str(ability_id) and tid.strip().isdigit():
                    text_id = int(tid)
        if text_id is None and a:
            text_id = a[0]
        txt = self.loc(text_id) if text_id else None
        if txt and a and a[1]:
            txt = f"{a[1]}: {txt}"
        return txt


def xmage_name(name: str) -> str:
    """XMage names cards in ASCII: "Troll of Khazad-dum", "Aether Vial" (Arena and 17lands keep the
    diacritics). FDN names are ASCII already."""
    if name.isascii():
        return name
    name = name.replace("Æ", "Ae").replace("æ", "ae")
    return "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))


def find_card_dbs() -> list[Path]:
    out = []
    for g in CARD_DB_GLOBS:
        out += [Path(p) for p in glob.glob(os.path.expanduser(g))]
    return out


def pick_card_db(build: int | None, candidates: list[Path] | None = None) -> Path | None:
    """The Raw_CardDatabase whose GRP build matches the log (Arena keeps the previous DB after an
    update), else the newest one."""
    dbs = find_card_dbs() if candidates is None else candidates
    if not dbs:
        return None
    if build is not None:
        for p in dbs:
            try:
                con = sqlite3.connect(f"file:{p}?mode=ro&immutable=1", uri=True)
                grp = dict(con.execute("select Type, Version from Versions").fetchall()).get("GRP", "")
                con.close()
            except sqlite3.Error:
                continue
            if str(grp).split(".")[0] == str(build):
                return p
    return max(dbs, key=os.path.getmtime)


class CardNames:
    """grpId -> card, ability id -> text, token grpId -> XMage class, key -> vocab index.
    Counts where every answer came from, for the summary."""

    def __init__(self, cards: dict[int, CardInfo] | None = None, abilities: dict[int, str] | None = None,
                 arena_db: ArenaCardDB | None = None, token_classes: dict[int, str] | None = None,
                 vocab: dict[str, int] | None = None):
        self.cards = cards or {}
        self.abilities = abilities or {}
        self.arena_db = arena_db
        self.token_classes = token_classes or {}
        self.vocab = vocab or {}
        self.via_arena_db: dict[int, str] = {}      # grpId -> name, for cards not in cards.csv
        self.unresolved: set[int] = set()
        self.emitted: set[str] = set()              # every name/text handed out (privacy allowlist)
        self.ability_sources: collections.Counter = collections.Counter()   # distinct ability ids
        self._cache: dict[int, CardInfo | None] = {}
        self._seen_abilities: set = set()

    @classmethod
    def load(cls, cards_csv=CARDS_CSV, abilities_csv=ABILITIES_CSV, card_db=None, vocab_tsv=VOCAB_TSV,
             token_tables=TOKEN_TABLES) -> "CardNames":
        tokens: dict[int, str] = {}
        for t in token_tables or ():
            tokens.update(read_token_table(t))
        return cls(read_cards_csv(cards_csv), read_abilities_csv(abilities_csv),
                   ArenaCardDB(card_db) if card_db else None, tokens, read_vocab(vocab_tsv))

    def card(self, grp) -> CardInfo | None:
        if not grp:
            return None
        if grp in self._cache:
            return self._cache[grp]
        info = self.cards.get(grp)
        if info is None and self.arena_db is not None:
            info = self.arena_db.card(grp)
            if info:
                self.via_arena_db[grp] = info.name
        if info is None:
            self.unresolved.add(grp)
        else:
            self.emitted.add(info.name)
        self._cache[grp] = info
        return info

    def name(self, grp) -> str | None:
        c = self.card(grp)
        return c.name if c else None

    def is_front(self, grp) -> bool:
        """False for the back face / secondary half of a multi-part card. Arena's LinkedFaceType
        decides when the DB is there; cards.csv's is_booster (False for back faces, but also for
        non-booster printings) only when it is not."""
        if self.arena_db is not None and (info := self.arena_db.card(grp)) is not None:
            return info.front
        c = self.card(grp)
        return c.front if c else True

    def ability_text(self, ability_id, card_grp=None) -> str | None:
        txt, source = None, "missing"
        if self.arena_db is not None and ability_id:
            txt = self.arena_db.ability_text(ability_id, card_grp)
            source = "arena_db" if txt else source
        if txt is None and (txt := self.abilities.get(ability_id)):
            source = "abilities.csv"
        if ability_id not in self._seen_abilities:
            self._seen_abilities.add(ability_id)
            self.ability_sources[source] += 1
        if txt:
            self.emitted.add(txt)
        return txt

    def token_class(self, grp) -> str | None:
        return self.token_classes.get(grp)

    def vocab_index(self, key: str | None) -> int | None:
        """Index of an action key in the MageZero vocab. XMage appends reminder text to some keyword
        abilities ("Equip {1} <i>(...)</i>"), so a key also matches its reminder-text variant."""
        if not key or not self.vocab:
            return None
        if key in self.vocab:
            return self.vocab[key]
        pref = key + " <i>("
        for k, i in self.vocab.items():
            if k.startswith(pref):
                return i
        return None


def read_cards_csv(path) -> dict[int, CardInfo]:
    out: dict[int, CardInfo] = {}
    if not path or not Path(path).exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                grp = int(row["id"])
            except (KeyError, ValueError):
                continue
            token = row.get("rarity") == "token"
            exp = row.get("expansion", "")
            if token and len(exp) == 4 and exp.startswith("T"):
                exp = exp[1:]            # 17lands files tokens under "T<SET>"; XMage under "<SET>"
            # is_booster is False for secondary faces, but also for non-booster printings, so it is
            # only used to break ties between the two faces of one object (see Parser.obj_name)
            out[grp] = CardInfo(grp, xmage_name(row["name"]), exp, token, row.get("is_booster") == "True")
    return out


def read_abilities_csv(path) -> dict[int, str]:
    out: dict[int, str] = {}
    if not path or not Path(path).exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out[int(row["id"])] = row["text"]
            except (KeyError, ValueError):
                continue
    return out


def read_token_table(path) -> dict[int, str]:
    """grpId -> XMage token class (short name), from a TSV with an id/grpId column and a
    token_class/xmage_token_class column; '#' lines are comments."""
    out: dict[int, str] = {}
    if not path or not Path(path).exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t"):
            cls = row.get("xmage_token_class") or row.get("token_class") or ""
            try:
                grp = int(row.get("id") or row.get("grpId") or "")
            except ValueError:
                continue
            if cls:
                out[grp] = cls.rsplit(".", 1)[-1]
    return out


def read_vocab(path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not path or not Path(path).exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) == 3 and parts[0] == "A":
                out[parts[2].replace("\\n", "\n")] = int(parts[1])
    return out


# --- Arena rules text -> XMage Ability.toString() (MageZero action keys) -------------------------
_MARKUP = re.compile(r"</?(?:nobr|i|b|sprite[^>]*)>")
_SYM = re.compile(r"\{(o[^{}]*)\}")
_THIS = re.compile(r"\b[Tt]his (?:creature|artifact|enchantment|land|Equipment|Vehicle|planeswalker|"
                   r"permanent|Aura|Saga)\b")


def strip_markup(s: str | None) -> str | None:
    return _MARKUP.sub("", s) if s else s


def _expand_symbols(m) -> str:
    """Arena packs mana symbols as {o3oGoG} / {o(B/G)} / {oT}; XMage writes {3}{G}{G} / {B/G} / {T}."""
    parts = re.findall(r"o(\([^)]*\)|[^o(]+)", m.group(1))
    return "".join("{%s}" % p.strip("()") for p in parts) if parts else m.group(0)


def xmage_text(arena_text: str | None, card_name: str | None = None) -> str | None:
    """Normalise Arena rules text toward XMage's: mana symbols, CARDNAME / the card's own name /
    a legendary short name / "this creature" -> {this}, markup stripped."""
    if not arena_text:
        return arena_text
    s = _SYM.sub(_expand_symbols, strip_markup(arena_text))
    s = s.replace("CARDNAME", "{this}")
    if card_name:
        s = s.replace(card_name, "{this}")
        if ", " in card_name:
            s = re.sub(r"\b%s\b" % re.escape(card_name.split(", ")[0]), "{this}", s)
    return _THIS.sub("{this}", s).strip()


_MANA = {"ManaColor_White": "W", "ManaColor_Blue": "U", "ManaColor_Black": "B", "ManaColor_Red": "R",
         "ManaColor_Green": "G", "ManaColor_Colorless": "C", "ManaColor_X": "X"}
_CARD_COLOR = {"CardColor_White": "W", "CardColor_Blue": "U", "CardColor_Black": "B",
               "CardColor_Red": "R", "CardColor_Green": "G"}
_BASIC_TYPE = {"SubType_Plains": "W", "SubType_Island": "U", "SubType_Swamp": "B",
               "SubType_Mountain": "R", "SubType_Forest": "G"}
BASICS = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}


def mana_str(cost) -> str | None:
    if not cost:
        return None
    out = []
    for c in cost:
        cols, n = c.get("color", []), c.get("count", 1)
        if not cols:
            continue
        if cols == ["ManaColor_Generic"]:
            out.append("{%d}" % n)
        else:
            out.append("{%s}" % "/".join(_MANA.get(x, "?") for x in cols) * n)
    return "".join(out) or None


# Arena prefixes an ability word ("Channel — {1}{G}, ...", "Vivid — {T}: ..."); XMage's
# Ability.toString() does not (id_mapping.md §5.5), so activated-ability keys drop it.
_ABILITY_WORD = re.compile(r"^[A-Z][A-Za-z' ]{2,30} — (?=\{|[+\u2212-]?\d+:)")
_REMINDER = re.compile(r"\s*<i>\(.*\)</i>\s*$", re.S)


def same_action(arena_key: str | None, xmage_label: str | None) -> bool:
    """Does an Arena option's mz_key name the same action as an XMage legal-action label? XMage
    appends reminder text to keyword abilities ("Swampcycling {1} <i>(...)</i>")."""
    if not arena_key or not xmage_label:
        return False
    return arena_key == xmage_label or arena_key == _REMINDER.sub("", xmage_label)


CAST_TYPES = {"Cast", "CastAdventure", "CastMdfc", "CastLeft", "CastRight", "CastPrototype", "CastOmen",
              "CastLeftRoom", "CastRightRoom"}
PLAY_TYPES = {"Play", "PlayMdfc"}
ACTIVATE_TYPES = {"Activate", "Activate_Mana", "ActivateMana"}


def mz_key(action_type: str, name: str | None, ability_text: str | None = None,
           alt_text: str | None = None) -> str | None:
    """MageZero's action key (XMage Ability.toString(); ActionEncoder looks it up in the vocab)."""
    if action_type == "Pass":
        return "Pass"
    if action_type in CAST_TYPES:
        if alt_text and alt_text.startswith("Flashback"):
            return alt_text                  # XMage models flashback as its own ability
        return f"Cast {name}" if name else None
    if action_type in PLAY_TYPES:
        return f"Play {name}" if name else None
    if action_type in ACTIVATE_TYPES:
        return _ABILITY_WORD.sub("", ability_text) if ability_text else ability_text
    return None


# Arena CounterType enum value -> XMage mage.counters.CounterType name. Derived once by joining
# Arena's Enums table (CounterType) with XMage's CounterType names; P/T counters by their numbers.
# Arena types XMage lacks (e.g. 20 Control, 111 +0/+2) are absent and reported as unmapped.
COUNTER_TYPES = {
    1: "P1P1", 2: "M1M1", 3: "POISON", 4: "WIND", 5: "TIME", 6: "FADE", 7: "LOYALTY", 8: "WISH",
    9: "AGE", 10: "AIM", 11: "ARROW", 12: "ARROWHEAD", 13: "AWAKENING", 14: "BLAZE", 15: "BLOOD",
    16: "BOUNTY", 17: "BRIBERY", 18: "CARRION", 19: "CHARGE", 21: "CORPSE", 22: "CREDIT", 23: "CUBE",
    24: "CURRENCY", 25: "DEATH", 26: "DELAY", 27: "DEPLETION", 28: "DESPAIR", 29: "DEVOTION",
    30: "DIVINITY", 31: "DOOM", 32: "DREAM", 33: "ECHO", 34: "ELIXIR", 35: "ENERGY", 36: "EON",
    37: "EYEBALL", 38: "FATE", 39: "FEATHER", 40: "FILIBUSTER", 41: "FLAME", 42: "FLOOD",
    43: "FUNGUS", 44: "FUSE", 45: "GLYPH", 46: "GOLD", 47: "GROWTH", 48: "HATCHLING", 49: "HEALING",
    50: "HOOFPRINT", 51: "HOURGLASS", 52: "HUNGER", 53: "ICE", 54: "INFECTION", 55: "INTERVENTION",
    56: "JAVELIN", 57: "KI", 58: "LEVEL", 59: "LUCK", 61: "MANNEQUIN", 62: "MATRIX", 64: "MINE",
    65: "MINING", 66: "MIRE", 67: "MUSTER", 68: "NET", 69: "OMEN", 70: "ORE", 71: "PAGE", 72: "PAIN",
    73: "PARALYZATION", 74: "PETAL", 75: "PETRIFICATION", 76: "PHYLACTERY", 77: "PIN", 78: "PLAGUE",
    79: "POLYP", 80: "PRESSURE", 81: "PUPA", 82: "QUEST", 83: "SCREAM", 85: "SHELL", 86: "SHIELD",
    87: "SHRED", 88: "SLEEP", 90: "SLIME", 91: "SOOT", 93: "SPORE", 94: "STORAGE", 95: "STRIFE",
    96: "STUDY", 97: "THEFT", 98: "TIDE", 100: "TOWER", 101: "TRAINING", 102: "TRAP",
    103: "TREASURE", 104: "VERSE", 105: "VITALITY", 106: "WAGE", 107: "WINCH", 108: "LORE",
    109: "P1P2", 110: "P0P1", 112: "P1P0", 113: "P2P2", 114: "M0M1", 115: "M0M2", 116: "M1M0",
    117: "M2M1", 118: "M2M2", 119: "MANIFESTATION", 120: "GEM", 121: "CRYSTAL", 122: "ISOLATION",
    123: "HOUR", 124: "UNITY", 125: "VELOCITY", 126: "BRICK", 127: "LANDMARK", 128: "PREY",
    129: "SILVER", 130: "EGG", 131: "HIT", 132: "KNOWLEDGE", 133: "TASK", 134: "COIN",
    135: "DEATHTOUCH", 136: "FIRST_STRIKE", 137: "FLYING", 138: "HEXPROOF", 139: "LIFELINK",
    140: "MENACE", 141: "REACH", 142: "TRAMPLE", 143: "VIGILANCE", 144: "FORESHADOW",
    145: "INCARNATION", 146: "SOUL", 147: "VOYAGE", 148: "GHOSTFORM", 149: "NIGHT",
    150: "INDESTRUCTIBLE", 151: "HONE", 152: "BOOK", 153: "POINT", 154: "ENLIGHTENED",
    155: "HARMONY", 156: "VOID", 157: "EMBER", 158: "RITUAL", 159: "VALOR", 160: "JUDGMENT",
    161: "INVITATION", 162: "CROAK", 163: "BLOODLINE", 164: "SUSPECT", 165: "ACORN", 167: "STASH",
    168: "COLLECTION", 169: "ROPE", 170: "INGENUITY", 171: "PHYRESIS", 172: "STUN", 173: "OIL",
    174: "DEFENSE", 175: "EXPERIENCE", 176: "SLUMBER", 177: "REJECTION", 178: "BURDEN", 179: "HOPE",
    180: "INFLUENCE", 181: "SKEWER", 182: "BORE", 183: "CHORUS", 184: "DREAD", 185: "FINALITY",
    186: "BLOODSTAIN", 187: "IMPOSTOR", 188: "UNLOCK", 189: "FETCH", 190: "LOOT", 191: "EXALTED",
    192: "STORY", 193: "EVERYTHING", 194: "BLIGHT", 195: "SUPPLY", 196: "FEEDING", 197: "NEST",
    198: "POSSESSION", 199: "REV", 200: "INCUBATION", 201: "REVIVAL", 202: "FELLOWSHIP", 203: "BAIT",
    205: "DECAYED", 207: "FIRE", 208: "FILM", 209: "DOUBLE_STRIKE", 215: "HASTE", 216: "SHADOW",
}

# Arena turn structure -> (mage.constants.TurnPhase, PhaseStep). Main phases carry no step.
PHASES = {
    ("Phase_Beginning", "Step_Untap"): ("BEGINNING", "UNTAP"),
    ("Phase_Beginning", "Step_Upkeep"): ("BEGINNING", "UPKEEP"),
    ("Phase_Beginning", "Step_Draw"): ("BEGINNING", "DRAW"),
    ("Phase_Main1", None): ("PRECOMBAT_MAIN", "PRECOMBAT_MAIN"),
    ("Phase_Combat", "Step_BeginCombat"): ("COMBAT", "BEGIN_COMBAT"),
    ("Phase_Combat", "Step_DeclareAttack"): ("COMBAT", "DECLARE_ATTACKERS"),
    ("Phase_Combat", "Step_DeclareBlock"): ("COMBAT", "DECLARE_BLOCKERS"),
    ("Phase_Combat", "Step_FirstStrikeDamage"): ("COMBAT", "FIRST_COMBAT_DAMAGE"),
    ("Phase_Combat", "Step_CombatDamage"): ("COMBAT", "COMBAT_DAMAGE"),
    ("Phase_Combat", "Step_EndCombat"): ("COMBAT", "END_COMBAT"),
    ("Phase_Main2", None): ("POSTCOMBAT_MAIN", "POSTCOMBAT_MAIN"),
    ("Phase_Ending", "Step_End"): ("END", "END_TURN"),
    ("Phase_Ending", "Step_Cleanup"): ("END", "CLEANUP"),
}
_PHASE_ONLY = {"Phase_Beginning": ("BEGINNING", "UPKEEP"), "Phase_Main1": PHASES[("Phase_Main1", None)],
               "Phase_Combat": ("COMBAT", "BEGIN_COMBAT"), "Phase_Main2": PHASES[("Phase_Main2", None)],
               "Phase_Ending": ("END", "END_TURN")}


def xmage_phase(phase: str | None, step: str | None) -> tuple[str, str] | None:
    if (phase, step) in PHASES:
        return PHASES[(phase, step)]
    if phase in ("Phase_Main1", "Phase_Main2"):
        return PHASES[(phase, None)]
    return _PHASE_ONLY.get(phase)


def short(enum: Any) -> Any:
    """'Phase_Main1' -> 'Main1', 'ZoneType_Hand' -> 'Hand'."""
    return enum.split("_", 1)[1] if isinstance(enum, str) and "_" in enum else enum


def details(annotation: dict) -> dict:
    out = {}
    for d in annotation.get("details", []):
        out[d.get("key")] = d.get("valueInt32", d.get("valueString", []))
    return out


def first(xs, default=None):
    return xs[0] if xs else default


# ------------------------------------------------------------------------------------------------
# 4. Game state: the GRE's client-side view, rebuilt from Full and Diff GameStateMessages
# ------------------------------------------------------------------------------------------------
_GAME_ZONES = ("Battlefield", "Hand", "Graveyard", "Exile", "Stack", "Library")
_KEEP_INFO = ("gameNumber", "stage", "type", "variant", "matchState", "matchWinCondition",
              "superFormat", "mulliganType", "results")      # matchID is deliberately not kept


class GameState:
    """Diffs replace zones and objects whole by id (a re-sent land without isTapped is untapped),
    replace turnInfo whole (main phases have no step, so merging would keep a stale one), replace
    players by seat, delete by diffDeletedInstanceIds, and add/delete persistent annotations."""

    def __init__(self):
        self.objects: dict[int, dict] = {}
        self.zones: dict[int, dict] = {}
        self.players: dict[int, dict] = {}
        self.turn: dict = {}
        self.pann: dict[int, dict] = {}
        self.info: dict = {}
        self.gsid: int | None = None
        self.gaps: list[dict] = []
        self.parent: dict[int, int] = {}         # new instanceId -> previous one
        self.stale_objects: set[int] = set()     # not re-sent since the last chain gap
        self.stale_zones: set[int] = set()
        # persistent annotations (counters, attachments, targets) the lost message added or deleted
        # are only re-sent when they change again, so after a gap they stay unverified until a Full
        self.gap_open = False
        self.n_full = self.n_diff = self.n_skipped = 0
        self.annotations: list[dict] = []        # transient annotations of the last message

    def apply(self, gsm: dict) -> bool:
        gsid = gsm.get("gameStateId")
        if gsm.get("type") == "GameStateType_Full":
            self.n_full += 1
            self.objects.clear()
            self.zones.clear()
            self.pann.clear()
            self.players.clear()
            self.stale_objects.clear()
            self.stale_zones.clear()
            self.gap_open = False
        else:
            if self.gsid is not None and gsid is not None and gsid <= self.gsid:
                self.n_skipped += 1          # a re-send of a state already applied
                return False
            prev = gsm.get("prevGameStateId")
            if self.gsid is not None and prev is not None and prev != self.gsid:
                # a message was lost (Arena's 50-object logging limit): whatever it changed and is
                # not re-sent later stays stale
                self.gaps.append({"expected_prev": self.gsid, "got_prev": prev, "gameStateId": gsid})
                self.stale_objects = set(self.objects)
                self.stale_zones = set(self.zones)
                self.gap_open = True
            self.n_diff += 1
        if "gameInfo" in gsm:
            self.info.update({k: v for k, v in gsm["gameInfo"].items() if k in _KEEP_INFO})
        for z in gsm.get("zones", []):
            self.zones[z["zoneId"]] = z
            self.stale_zones.discard(z["zoneId"])
        for o in gsm.get("gameObjects", []):
            self.objects[o["instanceId"]] = o
            self.stale_objects.discard(o["instanceId"])
        for i in gsm.get("diffDeletedInstanceIds", []):
            self.objects.pop(i, None)
            self.stale_objects.discard(i)
        for p in gsm.get("players", []):
            self.players[p.get("systemSeatNumber", p.get("controllerSeatId"))] = p
        if "turnInfo" in gsm:
            self.turn = dict(gsm["turnInfo"])
        for a in gsm.get("persistentAnnotations", []):
            self.pann[a["id"]] = a
        for i in gsm.get("diffDeletedPersistentAnnotationIds", []):
            self.pann.pop(i, None)
        self.annotations = gsm.get("annotations", [])
        for a in self.annotations:
            types = a.get("type", [])
            if "AnnotationType_ObjectIdChanged" in types:
                d = details(a)
                if d.get("orig_id") and d.get("new_id"):
                    self.parent[d["new_id"][0]] = d["orig_id"][0]
            elif "AnnotationType_Shuffle" in types:
                d = details(a)
                for old, new in zip(d.get("OldIds", []), d.get("NewIds", [])):
                    self.parent[new] = old
        if gsid is not None:
            self.gsid = gsid
        return True

    # --- derived views -------------------------------------------------------------------------
    def zone_type(self, zone_id) -> str | None:
        z = self.zones.get(zone_id)
        return short(z.get("type")) if z else None

    def membership(self) -> dict[int, dict]:
        idx = {}
        for z in self.zones.values():
            for i in z.get("objectInstanceIds", []):
                idx[i] = z
        return idx

    def zones_of(self, ztype: str) -> list[dict]:
        return [z for z in self.zones.values() if short(z.get("type")) == ztype]

    def counters(self) -> dict[int, dict[int, int]]:
        """instanceId (or player seat) -> {Arena CounterType value: count}, persistent Counter."""
        out: dict[int, dict[int, int]] = collections.defaultdict(dict)
        for a in self.pann.values():
            if "AnnotationType_Counter" in a.get("type", []):
                d = details(a)
                ct, n = first(d.get("counter_type")), first(d.get("count"), 0)
                for i in a.get("affectedIds", []):
                    out[i][ct] = n
        return out

    def attachments(self) -> dict[int, int]:
        """Attached object -> host. Persistent Attachment: affectorId is the Aura or Equipment and
        affectedIds the host (checked on the research log: Aura affector, creature affected; the
        research prototype had the direction reversed)."""
        out = {}
        for a in self.pann.values():
            if "AnnotationType_Attachment" in a.get("type", []) and a.get("affectorId"):
                for host in a.get("affectedIds", [])[:1]:
                    out[a["affectorId"]] = host
        return out

    def target_slots(self) -> dict[int, list[tuple[int, list[int]]]]:
        """Stack object -> [(slot index, its targets)] in slot order (persistent TargetSpec; the
        index is 1-based, like SelectTargetsReq.targetIdx). A slot can hold several targets ("any
        number of target creatures"); a slot left empty has no annotation."""
        out: dict[int, dict[int, list[int]]] = collections.defaultdict(lambda: collections.defaultdict(list))
        for a in self.pann.values():
            if "AnnotationType_TargetSpec" in a.get("type", []):
                idx = first(details(a).get("index"), 0)
                out[a.get("affectorId")][idx] += a.get("affectedIds", [])
        return {k: [(i, v[i]) for i in sorted(v)] for k, v in out.items()}

    def targets(self) -> dict[int, list[int]]:
        """Stack object -> all its targets in slot order."""
        return {k: [t for _, slot in v for t in slot] for k, v in self.target_slots().items()}

    def entered_battlefield_this_turn(self) -> set[int]:
        bf = {z["zoneId"] for z in self.zones_of("Battlefield")}
        return {i for a in self.pann.values() if "AnnotationType_EnteredZoneThisTurn" in a.get("type", [])
                and a.get("affectorId") in bf for i in a.get("affectedIds", [])}

    def layered_effects(self) -> int:
        return sum(1 for a in self.pann.values() if "AnnotationType_LayeredEffect" in a.get("type", []))

    def root(self, iid):
        for _ in range(1000):
            if iid not in self.parent:
                break
            iid = self.parent[iid]
        return iid


# ------------------------------------------------------------------------------------------------
# 5. Decisions
# ------------------------------------------------------------------------------------------------
NON_DECISION = {"GameStateMessage", "QueuedGameStateMessage", "TimerStateMessage", "UIMessage",
                "ConnectResp", "SetSettingsResp", "GetSettingsResp", "DieRollResultsResp",
                "SubmitAttackersResp", "SubmitBlockersResp", "SubmitTargetsResp", "EdictalMessage",
                "TimeoutMessage", "IllegalRequest", "AssignDamageConfirmation",
                "OrderDamageConfirmation", "SubmitDeckConfirmation", "PredictionResp",
                "SubmitDeckReq", "BinaryGameState", "GameStateBinary"}
INFORMATIONAL = {"PromptReq", "IntermissionReq"}         # no answer expected
CHAINED = {"DeclareAttackersReq", "DeclareBlockersReq", "SelectTargetsReq"}
TOGGLES = {"DeclareAttackersResp", "DeclareBlockersResp", "SelectTargetsResp"}
FINALIZERS = {"SubmitAttackersReq", "SubmitBlockersReq", "SubmitTargetsReq"}
IGNORED_CLIENT = {"SetSettingsReq", "UIMessage", "ConnectReq", "GetSettingsReq", "ChatMessage"}
# Requests that are steps inside casting or activating something the human chose just before.
PART_OF_ACTION = {"SelectTargetsReq", "CastingTimeOptionsReq", "PayCostsReq", "DistributionReq",
                  "NumericInputReq", "SelectNReq"}

# Arena request -> MageZero decision kind (arena_log.md §4.3); None = not learned by MageZero.
MZ_DECISION = {
    "ActionsAvailableReq": "PRIORITY", "DeclareAttackersReq": "CHOOSE_USE",
    "DeclareBlockersReq": "CHOOSE_TARGET", "SelectTargetsReq": "CHOOSE_TARGET",
    "SearchReq": "CHOOSE_TARGET", "SelectNReq": "CHOOSE_TARGET", "GroupReq": "MAKE_CHOICE",
    "OrderReq": "MAKE_CHOICE", "SelectReplacementReq": "MAKE_CHOICE", "SelectNGroupReq": "MAKE_CHOICE",
    "CastingTimeOptionsReq": "CHOOSE_NUM", "NumericInputReq": "CHOOSE_NUM",
    "OptionalActionMessage": "CHOOSE_USE", "MulliganReq": "CHOOSE_USE",
    "PayCostsReq": None, "AssignDamageReq": None, "DistributionReq": None, "ChooseStartingPlayerReq": None,
}
TIER_NAMES = ("T0", "T1", "T2", "T3")


@dataclass
class Request:
    msg_id: int | None
    kind: str
    gsid: int | None
    t: float | None
    body: dict                       # the raw GRE message (memory only)
    decision: "Decision | None" = None


@dataclass
class Decision:
    game: int
    index: int
    kind: str
    spec: StateSpec
    requests: list[Request] = field(default_factory=list)
    responses: list[dict] = field(default_factory=list)
    committed: bool = False
    t_request: float | None = None
    t_response: float | None = None
    source_root: int | None = None   # the spell/ability being set up (target chains)
    commit_gsid: int | None = None
    check_pending: bool = False

    @property
    def labels(self) -> dict:
        return self.spec.labels


@dataclass
class Game:
    index: int
    key: str | None                  # hash of (matchID, gameNumber); memory only, never written
    local_seat: int | None
    deck: list[int]
    state: GameState = field(default_factory=GameState)
    decisions: list[Decision] = field(default_factory=list)
    by_msg: dict[int, Request] = field(default_factory=dict)
    chain: Decision | None = None
    last_request: Request | None = None
    last_priority: Decision | None = None
    t0: int | None = None
    lands_turn: int | None = None
    lands_played: collections.Counter = field(default_factory=collections.Counter)
    seen_opp: dict[int, str] = field(default_factory=dict)        # root instanceId -> card name
    seen_opp_proxy: dict[int, str] = field(default_factory=dict)  # RevealedCard id -> card name
    opp_colors: collections.Counter = field(default_factory=collections.Counter)
    starting_seat: int | None = None
    starting_from_parity: bool = False   # wrong only if an extra turn was taken before the log began
    counts: collections.Counter = field(default_factory=collections.Counter)
    prio_gsids: set = field(default_factory=set)
    req_gsids: set = field(default_factory=set)
    requests_by_type: collections.Counter = field(default_factory=collections.Counter)
    answered_by_type: collections.Counter = field(default_factory=collections.Counter)
    resyncs: int = 0

    def opp_seat(self) -> int | None:
        seats = [s for s in self.state.players if s != self.local_seat]
        if seats:
            return seats[0]
        return (3 - self.local_seat) if self.local_seat in (1, 2) else None

    def seat_letter(self, s) -> str | None:
        return "A" if s == self.local_seat else ("B" if s is not None else None)

    def result(self) -> str | None:
        local_team = (self.state.players.get(self.local_seat) or {}).get("teamId", self.local_seat)
        for r in self.state.info.get("results", []) or []:
            if r.get("scope") == "MatchScope_Game":
                if r.get("result") == "ResultType_Draw":
                    return "draw"
                if r.get("winningTeamId") is not None:
                    return "win" if r["winningTeamId"] == local_team else "loss"
        return None


class Parser:
    """Feed it LogMessages (or whole files); it keeps one Game per Arena game and one Decision per
    logical decision, each with a StateSpec taken at the decision's first request."""

    def __init__(self, names: CardNames):
        self.names = names
        self.games: list[Game] = []
        self.cur: Game | None = None
        self.local_seat: int | None = None
        self.deck: list[int] = []
        self.stats: collections.Counter = collections.Counter()
        self.summarized: list[dict] = []

    # --- input -----------------------------------------------------------------------------------
    def parse_file(self, path) -> None:
        for msg in iter_messages(path):
            self.feed(msg)

    def feed(self, msg: LogMessage) -> None:
        if msg.summary is not None:
            self.stats["summarized_events"] += 1
            dropped = [s for s in msg.summary if s and not s[0].isdigit() and "Count" not in s]
            self.summarized.append({"game": self.cur.index if self.cur else None, "dropped": dropped})
            if self.cur is not None:
                self.cur.counts["summarized_events"] += 1
                for s in dropped:
                    if s not in ("GameStateMessage", "QueuedGameStateMessage"):
                        self.cur.counts["requests_lost_to_summary"] += 1
            return
        if msg.obj is None:
            self.stats["unparsed_bodies"] += 1
            return
        if msg.kind == "GreToClientEvent":
            ev = msg.obj.get("greToClientEvent") or {}
            for m in ev.get("greToClientMessages", []):
                self.on_gre(m, msg.obj.get("timestamp"))
        elif msg.kind == "ClientToGremessage":
            payload = msg.obj.get("payload")
            if isinstance(payload, str):          # some client payloads are JSON inside a string
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = None
            if isinstance(payload, dict):
                self.on_client(payload, msg.obj.get("timestamp"))
        # ClientToGreuimessage (hovers), AuthenticateResponse and MatchGameRoomStateChangedEvent
        # (screen names, user ids) are never read.

    def rel_t(self, ts) -> float | None:
        ms = gre_ms(ts)
        if ms is None or self.cur is None:
            return None
        if self.cur.t0 is None:
            self.cur.t0 = ms
        return round((ms - self.cur.t0) / 1000.0, 1)

    # --- GRE -> client ---------------------------------------------------------------------------
    def on_gre(self, m: dict, ts) -> None:
        kind = short(m.get("type"))
        self.stats["gre:" + str(kind)] += 1
        if kind == "ConnectResp":
            seats = m.get("systemSeatIds") or []
            self.local_seat = seats[0] if seats else self.local_seat
            deck = ((m.get("connectResp") or {}).get("deckMessage") or {}).get("deckCards")
            if deck:
                self.deck = list(deck)
            if self.cur is not None:            # a reconnect: msgIds restart
                self.cur.by_msg.clear()
                self.cur.local_seat = self.cur.local_seat or self.local_seat
            return
        if kind == "SubmitDeckReq":
            deck = (((m.get("submitDeckReq") or {}).get("deck")) or {}).get("deckCards")
            if deck:
                self.deck = list(deck)
            return
        if kind in ("GameStateMessage", "QueuedGameStateMessage"):
            gsm = m.get("gameStateMessage") or {}
            if self.local_seat is None and len(m.get("systemSeatIds") or []) == 1:
                self.local_seat = m["systemSeatIds"][0]
            self.route_game(gsm)
            if self.cur.t0 is None:
                self.rel_t(ts)
            if self.cur.state.apply(gsm):
                self.after_state(gsm)
            return
        if kind in NON_DECISION or self.cur is None:
            return
        if kind == "IntermissionReq":
            res = (m.get("intermissionReq") or {}).get("result")
            if res and "results" not in self.cur.state.info:
                self.cur.state.info["results"] = [dict(res, scope="MatchScope_Game")]
            return
        if kind in INFORMATIONAL:
            return
        seats = m.get("systemSeatIds") or []
        if self.cur.local_seat is None and len(seats) == 1:
            self.cur.local_seat = seats[0]
            self.local_seat = self.local_seat or seats[0]
        if seats and self.cur.local_seat not in seats:
            self.stats["requests_to_other_seat"] += 1
            return
        self.on_request(m, kind, ts)

    def route_game(self, gsm: dict) -> None:
        """A Full state of a different (match, game number) starts a new game; one of the same game
        is a resync (reconnect) and just replaces the state."""
        info = gsm.get("gameInfo") or {}
        key = None
        if info.get("matchID") is not None and info.get("gameNumber") is not None:
            key = hashlib.sha1(f"{info['matchID']}:{info['gameNumber']}".encode()).hexdigest()[:12]
        cur = self.cur
        if gsm.get("type") == "GameStateType_Full":
            if cur is not None and key is not None and key == cur.key:
                cur.resyncs += 1
                return
            new = (cur is None or (key is not None and key != cur.key)
                   or (key is None and (cur.state.gsid is None or (gsm.get("gameStateId") or 0) <= cur.state.gsid)))
            if not new:
                return
        elif cur is not None:
            return
        g = Game(index=len(self.games) + 1, key=key, local_seat=self.local_seat, deck=list(self.deck))
        if cur is not None and cur.chain is not None:
            cur.chain = None
        self.games.append(g)
        self.cur = g

    def after_state(self, gsm: dict) -> None:
        g, st = self.cur, self.cur.state
        loc = g.local_seat
        ti = st.turn
        turn = ti.get("turnNumber")
        if g.starting_seat is None and turn and (act := ti.get("activePlayer")) in st.players:
            # Arena numbers turns globally and the player on the play takes the odd ones; a log that
            # starts mid-game (after a restart) never shows turn 1, so fall back on the parity
            other = next((x for x in st.players if x != act), None)
            g.starting_seat = act if turn % 2 == 1 else other
            g.starting_from_parity = turn != 1
        if turn != g.lands_turn:
            g.lands_turn, g.lands_played = turn, collections.Counter()
        for a in st.annotations:
            types = a.get("type", [])
            if "AnnotationType_ZoneTransfer" in types and first(details(a).get("category")) == "PlayLand":
                o = st.objects.get(first(a.get("affectedIds")), {})
                g.lands_played[o.get("controllerSeatId")] += 1
            elif "AnnotationType_UserActionTaken" in types and a.get("affectorId") == loc:
                g.counts["local_actions_taken"] += 1
        # everything the opponent has shown, for B's placeholder decklist
        opp = g.opp_seat()
        for o in gsm.get("gameObjects", []):
            if o.get("ownerSeatId") != opp or not o.get("grpId"):
                continue
            otype = o.get("type")
            if otype == "GameObjectType_Card" and not o.get("isCopy"):
                g.seen_opp[st.root(o["instanceId"])] = self.obj_name(o)     # front face of a DFC
            elif otype == "GameObjectType_RevealedCard":
                g.seen_opp_proxy[o["instanceId"]] = self.obj_name(o)
            else:
                continue
            for c in o.get("color", []):
                if c in _CARD_COLOR:
                    g.opp_colors[_CARD_COLOR[c]] += 1
            for s in o.get("subtypes", []):
                if s in _BASIC_TYPE:
                    g.opp_colors[_BASIC_TYPE[s]] += 1
        if ti.get("priorityPlayer") == loc and gsm.get("type") == "GameStateType_Diff":
            g.prio_gsids.add(st.gsid)
        # cross-check committed attacks/blocks against the state that follows them
        for d in g.decisions[-6:]:
            if d.check_pending and d.commit_gsid is not None and st.gsid > d.commit_gsid:
                self.cross_check(d)

    # --- requests --------------------------------------------------------------------------------
    def on_request(self, m: dict, kind: str, ts) -> None:
        g = self.cur
        t = self.rel_t(ts)
        req = Request(m.get("msgId"), kind, m.get("gameStateId"), t, m)
        g.requests_by_type[kind] += 1
        if kind == "ActionsAvailableReq":
            g.req_gsids.add(m.get("gameStateId"))
        if m.get("gameStateId") is not None and m.get("gameStateId") != g.state.gsid:
            g.counts["request_state_mismatch"] += 1
        chain = g.chain
        if (kind in CHAINED and chain is not None and chain.kind == kind and not chain.committed
                and self.source_root(m) == chain.source_root):
            req.decision = chain                     # a refreshed request inside the same choice
            chain.requests.append(req)
        else:
            if chain is not None and not chain.committed:
                chain.labels["uncommitted"] = True
            g.chain = None
            d = self.new_decision(g, kind, req)
            req.decision = d
            if kind in CHAINED:
                g.chain = d
        if req.msg_id is not None:
            g.by_msg[req.msg_id] = req
        g.last_request = req

    def source_root(self, m: dict) -> int | None:
        body = m.get("selectTargetsReq") or {}
        src = body.get("sourceId")
        return self.cur.state.root(src) if src else None

    def new_decision(self, g: Game, kind: str, req: Request) -> Decision:
        index = len(g.decisions) + 1
        spec = self.build_spec(g, kind, index)
        d = Decision(g.index, index, kind, spec, [req], t_request=req.t, source_root=self.source_root(req.body))
        spec.labels.update({k: v for k, v in self.decode_request(g, kind, req.body).items()
                            if not k.startswith("_")})
        spec.labels["t"] = req.t
        if kind in PART_OF_ACTION and (parent := self.parent_of(g, req.body)) is not None:
            spec.labels["part_of"] = parent.index
        if kind == "ActionsAvailableReq":
            g.last_priority = d
        g.decisions.append(d)
        return d

    def parent_of(self, g: Game, m: dict) -> Decision | None:
        """The priority decision whose chosen Cast/Activate this request is a step of."""
        p = g.last_priority
        if p is None or not p.committed:
            return None
        chosen = p.labels.get("chosen") or []
        if not chosen or chosen[0].get("arena_action_type") not in CAST_TYPES | {"Activate"}:
            return None
        src = _request_source(m)
        if src is None:
            return None
        st = g.state
        want = st.root(chosen[0].get("instanceId"))
        o = st.objects.get(src, {})
        if st.root(src) == want or st.root(o.get("parentId")) == want:
            return p
        return None

    # --- client -> GRE ---------------------------------------------------------------------------
    def on_client(self, p: dict, ts) -> None:
        kind = short(p.get("type"))
        self.stats["client:" + str(kind)] += 1
        if kind == "SubmitDeckResp":
            deck = (((p.get("submitDeckResp") or {}).get("deck")) or {}).get("deckCards")
            if deck:
                self.deck = list(deck)
                if self.cur is not None and self.cur.state.gsid is None:
                    self.cur.deck = list(deck)
            return
        if self.cur is None or kind in IGNORED_CLIENT:
            return
        g = self.cur
        rid = p.get("respId")
        req = g.by_msg.get(rid) if rid is not None else None
        if req is None and kind == "CancelActionReq":
            req = g.last_request
        if req is None or req.decision is None:
            g.counts["unpaired_client_messages"] += 1
            return
        d = req.decision
        t = self.rel_t(ts)
        resp = self.decode_response(g, p, req)
        resp["t"] = t
        if not any(r.get("_msg") == req.msg_id for r in d.responses):
            g.answered_by_type[req.kind] += 1
        resp["_msg"] = req.msg_id
        d.responses.append(resp)
        d.t_response = t
        if kind in TOGGLES and d.kind in CHAINED:
            d.labels["round_trips"] = d.labels.get("round_trips", 0) + 1
            if resp.get("auto_declare"):
                d.labels["auto_declare"] = True
            return
        if kind == "CancelActionReq":
            d.labels["cancelled"] = True
        if d.committed:
            d.labels["extra_responses"] = d.labels.get("extra_responses", 0) + 1
            return
        d.committed = True
        d.commit_gsid = p.get("gameStateId", g.state.gsid)
        if g.chain is d:
            g.chain = None
        self.finalize(g, d, req, resp)

    def finalize(self, g: Game, d: Decision, req: Request, resp: dict) -> None:
        """Write the human's choice into the decision's labels."""
        L = d.labels
        L["answered_with"] = resp.get("type")
        if d.t_request is not None and resp.get("t") is not None:
            L["latency_s"] = round(resp["t"] - d.t_request, 1)
        if L.get("cancelled"):
            L["chosen"] = None
            return
        kind = d.kind
        if kind == "ActionsAvailableReq":
            chosen = resp.get("chosen", [])
            L["chosen"] = chosen
            key = lambda o: (o.get("arena_action_type"), o.get("instanceId"), o.get("ability_grpId"),  # noqa: E731
                             o.get("alt_grpId"))
            keys = [key(o) for o in L.get("options", [])]
            if chosen:
                # the alternative cost (flashback, ...) tells two casts of one card apart; fall back
                # to the card and ability alone in case a response does not echo it
                k = key(chosen[0])
                L["chosen_index"] = next((i for i, x in enumerate(keys) if x == k),
                                         next((i for i, x in enumerate(keys) if x[:3] == k[:3]), None))
            if resp.get("auto_pass"):
                L["auto_pass"] = resp["auto_pass"]
            return
        if kind in CHAINED:
            final = self.decode_request(g, kind, req.body)       # the last request shows the selection
            if kind == "DeclareAttackersReq":
                L["chosen"] = final.get("_selected", [])
                picked = {a["attacker"] for a in L["chosen"]}
                L["attack"] = {o["attacker"]: o["attacker"] in picked for o in L.get("options", [])}
                d.check_pending = True
            elif kind == "DeclareBlockersReq":
                L["chosen"] = final.get("_selected", [])
                L["block"] = {o["blocker"]: next((c["attackers"] for c in L["chosen"]
                                                  if c["blocker"] == o["blocker"]), [])
                              for o in L.get("options", [])}
                d.check_pending = True
            else:
                L["chosen"] = final.get("_selected", [])
                L["chosen_by_slot"] = final.get("_selected_by_slot", {})
                self.attach_to_parent(g, d, {"targets": L["chosen"]})
            for k in ("_selected", "_selected_by_slot"):
                L.pop(k, None)
            return
        for k in ("chosen", "value", "decision"):
            if k in resp:
                L["chosen"] = resp[k] if isinstance(resp[k], list) else [resp[k]]
                break
        else:
            L["chosen"] = {k: v for k, v in resp.items() if k not in ("t", "_msg", "type")} or [resp.get("type")]
        if kind == "CastingTimeOptionsReq" and "value" in resp:
            self.attach_to_parent(g, d, {"x": resp["value"]})
        elif kind == "DistributionReq":
            self.attach_to_parent(g, d, {"distribution": resp.get("chosen")})

    def attach_to_parent(self, g: Game, d: Decision, extra: dict) -> None:
        pi = d.labels.get("part_of")
        if pi is None:
            return
        parent = g.decisions[pi - 1]
        for c in parent.labels.get("chosen") or []:
            c.update(extra)
            break

    def cross_check(self, d: Decision) -> None:
        """Committed attackers/blockers should equal the objects that are attacking/blocking in
        the next state (arena_log.md §1.4: 8/8 in the research log)."""
        d.check_pending = False
        st = self.cur.state
        if d.labels.get("chosen") is None:
            return
        bf = {i for z in st.zones_of("Battlefield") for i in z.get("objectInstanceIds", [])}
        def iid(ref: str):
            tail = ref.rsplit(":", 1)[-1]
            return st.root(int(tail)) if tail.isdigit() else ref
        if d.kind == "DeclareAttackersReq":
            seen = {i for i in bf if st.objects.get(i, {}).get("attackState") in
                    ("AttackState_Declared", "AttackState_Attacking")}
            want = {iid(a["attacker"]) for a in d.labels["chosen"]}
        else:
            seen = {i for i in bf if st.objects.get(i, {}).get("blockState") in
                    ("BlockState_Declared", "BlockState_Blocking")}
            want = {iid(b["blocker"]) for b in d.labels["chosen"] if b["attackers"]}
        d.labels["state_check"] = {st.root(i) for i in seen} == want

    # --- decoding --------------------------------------------------------------------------------
    def obj_name(self, o: dict, flags: list | None = None) -> str | None:
        """Card name of a game object. A double-faced card is built from its front face in XMage."""
        grp = o.get("grpId")
        info = self.names.card(grp)
        other = o.get("othersideGrpId")
        if other and info is not None:
            oi = self.names.card(other)
            if oi is not None and not self.names.is_front(grp) and self.names.is_front(other):
                if flags is not None:
                    flags.append("back_face")
                return oi.name
        return info.name if info else (f"grpId:{grp}" if grp else None)

    def ref(self, g: Game, iid, placement: dict | None = None) -> str:
        """A spec-level reference: "player:A", "<seat>:<instanceId>" for a permanent (the spec's
        alias), otherwise "<seat>.<zone>:<name>"."""
        st = g.state
        if iid in st.players:
            return f"player:{g.seat_letter(iid)}"
        o = st.objects.get(iid)
        if o is None:
            return "hidden"
        zt = st.zone_type(o.get("zoneId")) or "?"
        seat = g.seat_letter(o.get("ownerSeatId"))
        if zt == "Battlefield":
            return f"{(placement or {}).get(iid, seat)}:{iid}"
        if short(o.get("type")) == "Ability":
            return f"stack:ability of {self.names.name(o.get('objectSourceGrpId'))}"
        return f"{seat}.{zt.lower()}:{self.obj_name(o)}"

    def option(self, g: Game, a: dict) -> dict:
        """One Arena action -> {arena_action_type, grpId, name, mz_key, ...}."""
        st = g.state
        kind = str(a.get("actionType", "")).replace("ActionType_", "")
        iid = a.get("instanceId")
        o = st.objects.get(iid, {}) if iid else {}
        grp = a.get("grpId") or o.get("grpId")
        name = self.obj_name(dict(o, grpId=grp)) if grp else None
        out: dict[str, Any] = {"arena_action_type": kind, "grpId": grp, "name": name}
        text = None
        if a.get("abilityGrpId"):
            text = xmage_text(self.names.ability_text(a["abilityGrpId"], grp), name)
            out["ability_grpId"] = a["abilityGrpId"]
        alt = None
        if a.get("alternativeGrpId"):
            alt = xmage_text(self.names.ability_text(a["alternativeGrpId"], grp), name)
            out["alt_grpId"] = a["alternativeGrpId"]     # e.g. flashback: another way to cast the same card
        key = mz_key(kind, name, text, alt)
        out["mz_key"] = key
        if (idx := self.names.vocab_index(key)) is not None:
            out["mz_idx"] = idx
        if iid:
            out["instanceId"] = iid
            out["zone"] = (st.zone_type(o.get("zoneId")) or "").lower() or None
        if (cost := mana_str(a.get("manaCost"))):
            out["cost"] = cost
            # Arena offers spells whose timing allows them even when the mana is short; only the
            # ones it can pay for carry an auto-tap solution (checked on the research log: 10 of the
            # 11 casts chosen had one, the other an X spell paid by hand)
            out["autotap"] = "autoTapSolution" in a
        return {k: v for k, v in out.items() if v is not None}

    def recipient(self, g: Game, r: dict | None) -> str | None:
        if not r:
            return None
        if r.get("type") == "DamageRecType_Player":
            return f"player:{g.seat_letter(r.get('playerSystemSeatId'))}"
        pw = r.get("planeswalkerInstanceId") or r.get("battleInstanceId")
        return self.ref(g, pw) if pw else short(r.get("type"))

    def decode_request(self, g: Game, kind: str, m: dict) -> dict:
        st = g.state
        R = lambda i: self.ref(g, i)                                   # noqa: E731
        if kind == "ActionsAvailableReq":
            a = m.get("actionsAvailableReq") or {}
            return {"options": [self.option(g, x) for x in a.get("actions", [])],
                    "inactive": [self.option(g, x) for x in a.get("inactiveActions", [])]}
        if kind == "DeclareAttackersReq":
            a = m.get("declareAttackersReq") or {}
            quals = a.get("qualifiedAttackers") or a.get("attackers") or []
            return {"options": [{"attacker": R(x["attackerInstanceId"]),
                                 "name": self.obj_name(st.objects.get(x["attackerInstanceId"], {})),
                                 "recipients": [self.recipient(g, r) for r in x.get("legalDamageRecipients", [])]}
                                for x in quals],
                    "_selected": [{"attacker": R(x["attackerInstanceId"]),
                                   "name": self.obj_name(st.objects.get(x["attackerInstanceId"], {})),
                                   "defender": self.recipient(g, x["selectedDamageRecipient"])}
                                  for x in a.get("attackers", []) if x.get("selectedDamageRecipient")]}
        if kind == "DeclareBlockersReq":
            a = m.get("declareBlockersReq") or {}
            bl = a.get("blockers", [])
            return {"options": [{"blocker": R(b["blockerInstanceId"]),
                                 "name": self.obj_name(st.objects.get(b["blockerInstanceId"], {})),
                                 "can_block": [R(i) for i in b.get("attackerInstanceIds", [])],
                                 "max": b.get("maxAttackers")} for b in bl],
                    "_selected": [{"blocker": R(b["blockerInstanceId"]),
                                   "name": self.obj_name(st.objects.get(b["blockerInstanceId"], {})),
                                   "attackers": [R(i) for i in b.get("selectedAttackerInstanceIds", [])]}
                                  for b in bl if b.get("selectedAttackerInstanceIds")]}
        if kind == "SelectTargetsReq":
            a = m.get("selectTargetsReq") or {}
            src = a.get("sourceId")
            src_o = st.objects.get(src, {})
            src_grp = src_o.get("objectSourceGrpId") or src_o.get("grpId")
            src_name = self.names.name(src_grp)
            slots, sel, by_slot = [], [], {}
            for x in a.get("targets", []):
                cands = x.get("targets", [])
                slots.append({"slot": x.get("targetIdx"), "min": x.get("minTargets", 0),
                              "max": x.get("maxTargets"), "candidates": [R(c["targetInstanceId"]) for c in cands]})
                chosen = [R(c["targetInstanceId"]) for c in cands if c.get("legalAction") == "SelectAction_Unselect"]
                sel += chosen
                if chosen:
                    by_slot[str(x.get("targetIdx"))] = chosen
            out = {"source": src_name, "options": slots, "_selected": sel, "_selected_by_slot": by_slot}
            if a.get("abilityGrpId"):
                out["ability"] = xmage_text(self.names.ability_text(a["abilityGrpId"], src_grp), src_name)
            return out
        if kind == "SelectNReq":
            a = m.get("selectNReq") or {}
            inst = a.get("idType") == "IdType_InstanceId"
            return {"options": [R(i) if inst else i for i in a.get("ids", [])], "min": a.get("minSel"),
                    "max": a.get("maxSel"), "context": short(a.get("context")),
                    "source": self.names.name(st.objects.get(a.get("sourceId"), {}).get("grpId"))}
        if kind == "SearchReq":
            a = m.get("searchReq") or {}
            return {"options": [R(i) for i in a.get("itemsSought", [])], "max": a.get("maxFind"),
                    "searched": len(a.get("itemsToSearch", [])),
                    "source": R(a.get("sourceId")) if a.get("sourceId") else None}
        if kind == "MulliganReq":
            params = {p.get("parameterName"): p.get("numberValue")
                      for p in (m.get("prompt") or {}).get("parameters", []) if "numberValue" in p}
            hand = next(iter(params.values()), None)
            return {"options": ["AcceptHand", "Mulligan"], "hand_size": hand,
                    "mulligan_type": short((m.get("mulliganReq") or {}).get("mulliganType"))}
        if kind == "CastingTimeOptionsReq":
            return {"options": [{"type": short(c.get("castingTimeOptionType")), "card": R(c.get("affectedId")),
                                 "max": (c.get("numericInputReq") or {}).get("maxValue")}
                                for c in (m.get("castingTimeOptionsReq") or {}).get("castingTimeOptionReq", [])]}
        if kind == "PayCostsReq":
            a = m.get("payCostsReq") or {}
            return {"cost": mana_str(a.get("manaCost")),
                    "options": [self.option(g, x) for x in (a.get("paymentActions") or {}).get("actions", [])],
                    "autotap_solutions": len((a.get("autoTapActionsReq") or {}).get("autoTapSolutions", []))}
        if kind == "AssignDamageReq":
            return {"options": [{"attacker": R(d["instanceId"]), "total": d.get("totalDamage"),
                                 "to": [{"target": R(x["instanceId"]), "min": x.get("minDamage")}
                                        for x in d.get("assignments", [])]}
                                for d in (m.get("assignDamageReq") or {}).get("damageAssigners", [])]}
        if kind == "DistributionReq":
            a = m.get("distributionReq") or {}
            return {"source": R(a.get("sourceId")), "amount": a.get("maxAmount"),
                    "options": [R(i) for i in a.get("targetIds", [])]}
        if kind == "OptionalActionMessage":
            src = st.objects.get((m.get("optionalActionMessage") or {}).get("sourceId"), {})
            return {"options": ["Yes", "No"],
                    "source": self.names.name(src.get("objectSourceGrpId") or src.get("grpId"))}
        if kind == "ChooseStartingPlayerReq":
            return {"options": ["A", "B"]}
        if kind == "GroupReq":
            a = m.get("groupReq") or {}
            return {"options": [R(i) for i in a.get("instanceIds", [])],
                    "groups": [short(s.get("zoneType")) + ":" + str(short(s.get("subZoneType")))
                               for s in a.get("groupSpecs", [])]}
        if kind == "OrderReq":
            return {"options": [R(i) for i in (m.get("orderReq") or {}).get("ids", [])]}
        if kind == "NumericInputReq":
            a = m.get("numericInputReq") or {}
            return {"min": a.get("minValue"), "max": a.get("maxValue")}
        body = {k: v for k, v in m.items() if k not in ("type", "systemSeatIds", "msgId", "gameStateId", "prompt")}
        return {"request_fields": sorted(body)}

    def decode_response(self, g: Game, p: dict, req: Request | None = None) -> dict:
        kind = short(p.get("type"))
        R = lambda i: self.ref(g, i)                                   # noqa: E731
        out: dict[str, Any] = {"type": kind}
        if kind == "PerformActionResp":
            a = p.get("performActionResp") or {}
            out["chosen"] = [self.option(g, x) for x in a.get("actions", [])]
            if a.get("autoPassPriority"):
                out["auto_pass"] = short(a["autoPassPriority"])
        elif kind == "DeclareAttackersResp":
            a = p.get("declareAttackersResp") or {}
            if a.get("autoDeclare"):
                out["auto_declare"] = True
        elif kind == "SelectNResp":
            # the ids are instance ids only when the request says so (as in decode_request)
            sel = ((req.body if req else {}).get("selectNReq") or {}).get("idType")
            out["chosen"] = [R(i) if sel == "IdType_InstanceId" else i
                             for i in (p.get("selectNResp") or {}).get("ids", [])]
        elif kind == "SearchResp":
            out["chosen"] = [R(i) for i in (p.get("searchResp") or {}).get("itemsFound", [])]
        elif kind == "MulliganResp":
            out["decision"] = short((p.get("mulliganResp") or {}).get("decision"))
        elif kind == "CastingTimeOptionsResp":
            c = (p.get("castingTimeOptionsResp") or {}).get("castingTimeOptionResp") or {}
            if "numericInputResp" in c:
                out["value"] = c["numericInputResp"].get("numericInputValue")
            else:
                out["chosen"] = [short(c.get("castingTimeOptionType"))]
        elif kind == "AssignDamageResp":
            out["chosen"] = [{"attacker": R(d["instanceId"]),
                              "to": {R(x["instanceId"]): x.get("assignedDamage") for x in d.get("assignments", [])}}
                             for d in (p.get("assignDamageResp") or {}).get("assigners", [])]
        elif kind == "DistributionResp":
            out["chosen"] = {R(d["instanceId"]): d.get("amount")
                             for d in (p.get("distributionResp") or {}).get("distributions", [])}
        elif kind == "OptionalActionResp":
            ans = short((p.get("optionalResp") or {}).get("response"))
            out["decision"] = "Yes" if str(ans).endswith("Yes") else ("No" if str(ans).endswith("No") else ans)
        elif kind == "ChooseStartingPlayerResp":
            seat = (p.get("chooseStartingPlayerResp") or {}).get("systemSeatId")
            out["decision"] = g.seat_letter(seat)
            g.starting_seat = g.starting_seat or seat
        elif kind == "PerformAutoTapActionsResp":
            out["decision"] = "AutoTap"
        elif kind == "GroupResp":
            out["chosen"] = [{"zone": short(x.get("zoneType")), "sub": short(x.get("subZoneType")),
                              "cards": [R(i) for i in x.get("ids", [])]}
                             for x in (p.get("groupResp") or {}).get("groups", [])]
        elif kind == "OrderResp":
            out["chosen"] = [R(i) for i in (p.get("orderResp") or {}).get("ids", [])]
        elif kind == "NumericInputResp":
            out["value"] = (p.get("numericInputResp") or {}).get("value")
        return out

    # --- StateSpec -------------------------------------------------------------------------------
    def build_spec(self, g: Game, kind: str, index: int) -> StateSpec:
        st, names = g.state, self.names
        loc, opp = g.local_seat, g.opp_seat()
        S = g.seat_letter
        flags: list[str] = []
        tier = [0]

        def lower(t: int, why: str) -> None:
            tier[0] = max(tier[0], t)
            if why not in flags:
                flags.append(why)

        ti = st.turn
        turn = ti.get("turnNumber") or 0
        ph = xmage_phase(ti.get("phase"), ti.get("step"))
        pregame = not turn or ph is None
        phase, step = ph if ph and not pregame else ("BEGINNING", "UPKEEP")

        ps = {"A": PlayerState(name="PlayerA"), "B": PlayerState(name="PlayerB")}
        for seat, p in st.players.items():
            s = S(seat)
            if s in ps:
                ps[s].life = p.get("lifeTotal", 20)
                pool = "".join(_MANA.get(x.get("color"), "") * x.get("count", 1)
                               for x in p.get("manaPool", []) or [] if x.get("color") != "ManaColor_X")
                ps[s].manaPool = pool or None
                ps[s].landsPlayed = g.lands_played.get(seat, 0)

        counters = st.counters()
        entered = st.entered_battlefield_this_turn()
        name_flags: list[str] = []
        visible = lambda o: bool(o and o.get("grpId") and short(o.get("type")) in ("Card", "Token"))  # noqa: E731

        # hidden-zone cards revealed through RevealedCard proxies (zoneId = the hidden zone)
        revealed = collections.defaultdict(list)
        for z in st.zones_of("Revealed"):
            for i in z.get("objectInstanceIds", []):
                o = st.objects.get(i)
                if o and o.get("type") == "GameObjectType_RevealedCard" and o.get("grpId"):
                    hz = st.zones.get(o.get("zoneId"), {})
                    revealed[(S(hz.get("ownerSeatId")), short(hz.get("type")))].append(self.obj_name(o))

        for z in st.zones_of("Hand"):
            s = S(z.get("ownerSeatId"))
            if s not in ps:
                continue
            ids = z.get("objectInstanceIds", [])
            known = [self.obj_name(st.objects[i], name_flags) for i in ids if visible(st.objects.get(i))]
            known += revealed.get((s, "Hand"), [])[:max(0, len(ids) - len(known))]
            ps[s].hand = known
            ps[s].handUnknown = len(ids) - len(known)
            if s == "A" and ps[s].handUnknown:
                lower(1, "A.hand_objects_missing")
        for z in st.zones_of("Library"):
            s = S(z.get("ownerSeatId"))
            if s not in ps:
                continue
            ids = z.get("objectInstanceIds", [])
            ps[s].librarySize = len(ids)
            vis = [i for i in ids if visible(st.objects.get(i))]
            if s == "A" and vis and len(vis) < len(ids):
                top = []
                for i in ids:                     # Arena lists library ids top first
                    if not visible(st.objects.get(i)):
                        break
                    top.append(self.obj_name(st.objects[i], name_flags))
                if top:
                    ps[s].libraryTop = top
                    flags.append("library_top_from_zone_order")
        for z in st.zones_of("Graveyard"):
            s = S(z.get("ownerSeatId"))
            if s not in ps:
                continue
            for i in reversed(z.get("objectInstanceIds", [])):   # Arena lists the newest first
                o = st.objects.get(i)
                if not visible(o):
                    lower(3, "missing_public_object")
                elif o.get("type") == "GameObjectType_Card":   # a dead token lingers for a moment
                    ps[s].graveyard.append(self.obj_name(o, name_flags))
        hidden_exile = 0
        for z in st.zones_of("Exile"):
            for i in reversed(z.get("objectInstanceIds", [])):
                o = st.objects.get(i)
                s = S(o.get("ownerSeatId")) if o else None
                if o is not None and o.get("type") == "GameObjectType_Token":
                    continue
                if visible(o) and s in ps and not o.get("isFacedown"):
                    ps[s].exile.append(self.obj_name(o, name_flags))
                else:
                    hidden_exile += 1
        if hidden_exile:
            flags.append(f"hidden_exile:{hidden_exile}")

        # battlefield: a permanent is listed under its owner (XMage injects with owner = controller)
        placement: dict[int, str] = {}
        perms: dict[int, Perm] = {}
        tokens_unmapped = 0
        for z in st.zones_of("Battlefield"):
            for i in reversed(z.get("objectInstanceIds", [])):
                o = st.objects.get(i)
                if not visible(o):
                    lower(3, "missing_public_object")
                    continue
                owner, ctrl = S(o.get("ownerSeatId")), S(o.get("controllerSeatId"))
                if owner not in ps:
                    continue
                if ctrl != owner:
                    lower(2, "control_changed")
                perm = Perm(id=str(i), tapped=bool(o.get("isTapped")),
                            sick=bool(o.get("hasSummoningSickness")) or i in entered,
                            damage=int(o.get("damage") or 0), faceDown=bool(o.get("isFacedown")))
                if o.get("type") == "GameObjectType_Token":
                    info = names.card(o.get("grpId"))
                    cls = names.token_class(o.get("grpId"))
                    if o.get("isCopy"):
                        lower(2, "token_copy")
                    if cls:
                        perm.tokenClass = cls
                    else:
                        tokens_unmapped += 1
                        perm.token = info.name if info else f"grpId:{o.get('grpId')}"
                        perm.set = (info.set or None) if info else None
                else:
                    perm.name = self.obj_name(o, name_flags)
                    if o.get("isCopy"):
                        lower(2, "copy_permanent")
                if perm.faceDown:
                    lower(2, "face_down")
                cs = {}
                for ct, n in counters.get(i, {}).items():
                    if ct in COUNTER_TYPES:
                        cs[COUNTER_TYPES[ct]] = n
                    else:
                        lower(1, f"counter_type_unmapped:{ct}")
                if "loyalty" in o and "LOYALTY" not in cs:
                    cs["LOYALTY"] = (o.get("loyalty") or {}).get("value", 0)
                perm.counters = cs
                placement[i] = owner
                perms[i] = perm
                ps[owner].battlefield.append(perm)
        if tokens_unmapped:
            flags.append(f"tokens_by_name:{tokens_unmapped}")
        for i, host in st.attachments().items():
            if i in perms and host in placement:
                perms[i].attachTo = f"{placement[host]}:{host}"
            elif i in perms:
                lower(2, "attachment_unresolved")
        if any(c for s, c in counters.items() if s in st.players):
            lower(1, "player_counters_dropped")

        stack: list[StackItem] = []
        stack_abilities = []
        targets = st.target_slots()
        # a target the spec can name: a player or a permanent alias (None for a card in a graveyard, ...)
        target_ref = lambda t: (f"player:{S(t)}" if t in st.players                   # noqa: E731
                                else f"{placement[t]}:{t}" if t in placement else None)
        for z in st.zones_of("Stack"):
            for i in reversed(z.get("objectInstanceIds", [])):   # Arena lists the top first
                o = st.objects.get(i)
                if o is None:
                    lower(3, "missing_public_object")
                    continue
                ctrl = S(o.get("controllerSeatId"))
                slots = [(k, [target_ref(t) for t in slot]) for k, slot in targets.get(i, [])]
                tg = [t for _, slot in slots for t in slot]
                if short(o.get("type")) == "Ability":
                    src = o.get("objectSourceGrpId")
                    stack_abilities.append({"controller": ctrl, "source": names.name(src),
                                            "ability": xmage_text(names.ability_text(o.get("grpId"), src),
                                                                  names.name(src)),
                                            "targets": [x or "unrepresentable" for x in tg]})
                    lower(2, "stack_ability_omitted")
                    continue
                if o.get("isCopy"):
                    lower(2, "stack_copy_omitted")
                    continue
                # the bridge fills target slot k with targets[k]: keep one target per slot, and stop
                # at a slot it cannot fill (a card in a graveyard, an optional slot left empty) so
                # that no later target lands in the wrong slot
                item_targets = []
                for k, slot in slots:
                    if len(slot) > 1:
                        lower(2, "stack_multi_target_slot")
                    if k != len(item_targets) + 1 or not slot or slot[0] is None:
                        lower(2, "stack_target_unrepresentable")
                        break
                    item_targets.append(slot[0])
                stack.append(StackItem(controller=ctrl, card=self.obj_name(o, name_flags), targets=item_targets))

        attackers, blockers = [], []
        for i in placement:
            o = st.objects[i]
            alias = f"{placement[i]}:{i}"
            in_combat = (o.get("attackState") in ("AttackState_Declared", "AttackState_Attacking")
                         or o.get("blockState") in ("BlockState_Declared", "BlockState_Blocking"))
            if in_combat and placement[i] != S(o.get("controllerSeatId")):
                lower(2, "combat_by_control_changed_omitted")
                continue
            if o.get("attackState") in ("AttackState_Declared", "AttackState_Attacking"):
                tgt = (o.get("attackInfo") or {}).get("targetId")
                if tgt in st.players:
                    defender = f"player:{S(tgt)}"
                elif tgt in placement:
                    defender = f"{placement[tgt]}:{tgt}"
                else:
                    defender = f"player:{'B' if placement[i] == 'A' else 'A'}"
                    lower(2, "attack_target_unknown")
                attackers.append(Attack(alias, defender))
            if o.get("blockState") in ("BlockState_Declared", "BlockState_Blocking"):
                for a in (o.get("blockInfo") or {}).get("attackerIds", []):
                    if a in placement:
                        blockers.append(Block(alias, f"{placement[a]}:{a}"))

        for f in name_flags:
            lower(2, f)
        unknown = [c for p in ps.values() for c in p.hand + p.graveyard + p.exile
                   + [b.name or b.token for b in p.battlefield] if c and c.startswith("grpId:")]
        if unknown or any(s.card.startswith("grpId:") for s in stack):
            lower(3, "unknown_card")
        if st.stale_objects or st.stale_zones:
            live = {i for z in st.zones.values() if short(z.get("type")) in _GAME_ZONES
                    for i in z.get("objectInstanceIds", [])}
            if st.stale_objects & live or any(st.zone_type(z) in _GAME_ZONES for z in st.stale_zones):
                lower(1, "stale_after_gap")
        if st.gap_open and (perms or stack or stack_abilities):
            lower(1, "annotations_unverified_after_gap")
        if g.starting_from_parity:
            flags.append("starting_player_from_turn_parity")
        if st.layered_effects():
            flags.append("layered_effects")          # results (P/T, types) visible, sources not rebuilt

        # decklists: A exact from ConnectResp, B a placeholder from what B has shown
        ps["A"].decklist = [names.name(x) or f"grpId:{x}" for x in g.deck]
        ps["A"].decklistSource = "exact"
        if not g.deck:
            lower(1, "A.deck_unknown")
            ps["A"].decklist = self.placeholder_deck(g, ps["A"], stack, "A", loc)
            ps["A"].decklistSource = "placeholder"
        ps["B"].decklist = self.placeholder_deck(g, ps["B"], stack, "B", opp)
        ps["B"].decklistSource = "placeholder"
        pool = collections.Counter(ps["A"].decklist)
        used = collections.Counter(ps["A"].hand + ps["A"].graveyard + ps["A"].exile + ps["A"].libraryTop
                                   + [b.name for b in ps["A"].battlefield
                                      if b.name and not b.tokenClass and not b.token]
                                   + [x.card for x in stack if x.controller == "A"])   # as the bridge counts
        if any(n > pool[c] for c, n in used.items()):
            flags.append("A.card_not_in_decklist")
        elif g.deck and ps["A"].librarySize is not None:
            # the rest of the exact deck should be the library: a difference is a card of A's the
            # spec does not show (face-down exile, lost with a summarized message, ...), and the
            # bridge trims the library to size, dropping a random card rather than that one
            extra = (len(ps["A"].decklist) - sum(used.values()) - ps["A"].handUnknown
                     - (ps["A"].librarySize - len(ps["A"].libraryTop)))
            if extra:
                flags.append(f"A.cards_unaccounted:{extra}")

        # entry point
        active = S(ti.get("activePlayer"))
        prio = S(ti.get("priorityPlayer")) or S(ti.get("decisionPlayer")) or "A"
        passed: list[str] = []
        enter = "PRIORITY_HELD"
        if pregame:
            # before turn 1 (play/draw, mulligans): the closest buildable state is the start of turn 1
            turn, enter, prio = 1, "BEGIN_STEP", None
            active = S(g.starting_seat) or "A"
            flags.append("pregame")
        elif kind == "DeclareAttackersReq":
            phase, step, enter, prio = "COMBAT", "BEGIN_COMBAT", "PRIORITY_FRESH", active
            flags.append("entry_rewound_to_begin_combat")
            if attackers or blockers:        # a pre-selection is part of the decision, not the state
                attackers, blockers = [], []
                flags.append("preselected_attackers_dropped")
        elif kind == "DeclareBlockersReq":
            phase, step, enter, prio = "COMBAT", "DECLARE_ATTACKERS", "PRIORITY_HELD", active
            flags.append("entry_rewound_to_declare_attackers")
            if blockers:
                blockers = []
                flags.append("preselected_blockers_dropped")
        else:
            if kind != "ActionsAvailableReq":
                flags.append("mid_action")
            if kind == "ActionsAvailableReq":
                prio = "A"
            other = "B" if prio == "A" else "A"
            if not stack and not stack_abilities:
                if prio == active:
                    enter = "PRIORITY_FRESH"
                else:
                    passed = [other]         # the active player passed to get priority here
            else:
                top_ctrl = self.stack_top_controller(g)
                if top_ctrl is not None and top_ctrl != prio:
                    passed = [other]         # the other player added the top object and passed
            if enter == "PRIORITY_FRESH" and step in ("DECLARE_ATTACKERS", "DECLARE_BLOCKERS"):
                # XMage resumes a paused PRE step through resumeBeginStep, which these two steps
                # override to replay the declaration (Combat.resumeSelectAttackers/-Blockers): attack
                # and block events fire again and attackers are re-tapped, in the live game and in
                # every MCTS copy. The same window entered mid-round (nobody passed) skips that.
                enter = "PRIORITY_HELD"
        if step == "UNTAP":
            # the bridge cannot enter UNTAP (no priority, and the untap has happened in this state):
            # run the upkeep from its start instead
            phase, step, enter, prio, passed = "BEGINNING", "UPKEEP", "BEGIN_STEP", None, []
            flags.append("entry_untap_as_upkeep")
        if step == "CLEANUP" and enter != "BEGIN_STEP":
            enter, prio, passed = "BEGIN_STEP", None, []
            flags.append("entry_begin_step_no_priority_window")
        if not active:
            active = "A"

        spec = StateSpec(
            comment=f"Arena game {g.index}, decision {index}: {kind} at turn {turn} {phase}/{step}",
            turn=max(1, turn), activePlayer=active, phase=phase, step=step, enterMode=enter,
            priorityPlayer=prio, passedPlayers=passed, startingPlayer=S(g.starting_seat),
            players=ps, stack=stack, attackers=attackers, blockers=blockers,
            provenance=Provenance(source="arena", ref=f"game{g.index}:decision{index}",
                                  tier=TIER_NAMES[tier[0]], flags=flags))
        spec.labels = {"decision_type": kind, "mz_decision": MZ_DECISION.get(kind),
                       "game": g.index, "decision": index,
                       "arena": {"turn": ti.get("turnNumber"), "phase": short(ti.get("phase")),
                                 "step": short(ti.get("step")), "active": S(ti.get("activePlayer")),
                                 "priority": S(ti.get("priorityPlayer")),
                                 "decision_player": S(ti.get("decisionPlayer"))}}
        # bridge options that reach this decision: every request here is to A (the bridge would
        # otherwise decide for the priority player, B in a block spec), and a spec entered a step
        # early would first stop at A's begin-combat or declare-attackers priority window
        bridge: dict[str, Any] = {"decisionPlayer": "A"}
        if not pregame and kind in ("DeclareAttackersReq", "DeclareBlockersReq"):
            bridge["decideFrom"] = {"turn": max(1, turn), "step": "DECLARE_ATTACKERS"
                                    if kind == "DeclareAttackersReq" else "DECLARE_BLOCKERS"}
        spec.labels["bridge"] = bridge
        if stack_abilities:
            spec.labels["stack_abilities"] = stack_abilities
        return spec

    def stack_top_controller(self, g: Game) -> str | None:
        for z in g.state.zones_of("Stack"):
            ids = z.get("objectInstanceIds", [])
            if ids:
                return g.seat_letter(g.state.objects.get(ids[0], {}).get("controllerSeatId"))
        return None

    def placeholder_deck(self, g: Game, p: PlayerState, stack: list[StackItem], s: str, seat) -> list[str]:
        """Every card this player has shown (one copy per physical card seen), at least the cards
        now in their zones, filled to the deck size with basics of the colours they have shown."""
        st = g.state
        seen: collections.Counter = collections.Counter()
        colors = collections.Counter()
        if s == "B":
            seen.update(g.seen_opp.values())
            proxies = collections.Counter(g.seen_opp_proxy.values())
            for c, n in proxies.items():
                seen[c] = max(seen[c], n)
            colors = g.opp_colors
        used = collections.Counter(p.hand + p.graveyard + p.exile + p.libraryTop
                                   + [b.name for b in p.battlefield if b.name and not b.token and not b.tokenClass]
                                   + [x.card for x in stack if x.controller == s])
        for c, n in used.items():
            seen[c] = max(seen[c], n)
        deck = list(seen.elements())
        # the deck size is exact: every card the player owns is listed in some zone
        total = sum(len(z.get("objectInstanceIds", [])) for z in st.zones.values()
                    if short(z.get("type")) in ("Library", "Hand", "Graveyard") and z.get("ownerSeatId") == seat)
        total += sum(1 for z in st.zones.values() if short(z.get("type")) in ("Battlefield", "Exile", "Stack")
                     for i in z.get("objectInstanceIds", [])
                     if (o := st.objects.get(i)) and o.get("ownerSeatId") == seat
                     and o.get("type") == "GameObjectType_Card")
        fill = max(40, total) - len(deck)
        if fill > 0:
            deck += basics_for(colors, fill)
        return deck

    # --- output ----------------------------------------------------------------------------------
    def finish(self) -> None:
        for g in self.games:
            if g.chain is not None and not g.chain.committed:
                g.chain.labels["uncommitted"] = True
            for d in g.decisions:
                if "chosen" not in d.labels:
                    d.labels["chosen"] = None
                    d.labels["unanswered"] = True
                d.check_pending = False

    def specs(self) -> list[StateSpec]:
        return [d.spec for g in self.games for d in g.decisions]


def _request_source(m: dict) -> int | None:
    for key, path in (("selectTargetsReq", ("sourceId",)), ("distributionReq", ("sourceId",)),
                      ("selectNReq", ("sourceId",)), ("numericInputReq", ("sourceId",))):
        if key in m:
            return (m[key] or {}).get(path[0])
    if "castingTimeOptionsReq" in m:
        opts = (m["castingTimeOptionsReq"] or {}).get("castingTimeOptionReq", [])
        return first(opts, {}).get("affectedId")
    if "payCostsReq" in m:
        return first((m["payCostsReq"] or {}).get("manaCost", []), {}).get("objectId")
    return None


def basics_for(colors: collections.Counter, n: int) -> list[str]:
    """n basic lands split over the colours by weight (largest remainder); all five if none seen."""
    weights = {c: colors[c] for c in "WUBRG" if colors.get(c)} or {c: 1 for c in "WUBRG"}
    total = sum(weights.values())
    raw = {c: n * w / total for c, w in weights.items()}
    counts = {c: int(v) for c, v in raw.items()}
    for c in sorted(raw, key=lambda c: raw[c] - counts[c], reverse=True)[:n - sum(counts.values())]:
        counts[c] += 1
    return [BASICS[c] for c in "WUBRG" for _ in range(counts.get(c, 0))]


# ------------------------------------------------------------------------------------------------
# 6. Running it
# ------------------------------------------------------------------------------------------------
def spec_json(spec: StateSpec) -> str:
    """One decisions.jsonl line. StateSpec.to_json() drops empty lists and None everywhere, which
    inside `labels` would erase answers: an attack with no attackers or a target choice left empty
    (chosen: []), an unanswered request (chosen: null). So labels are written as they are.
    Non-ASCII stays unescaped so that the privacy scan sees exactly what is written."""
    d = spec.to_dict()
    d["labels"] = spec.labels
    return json.dumps(d, ensure_ascii=False)


def parse_logs(paths: Iterable, names: CardNames) -> Parser:
    parser = Parser(names)
    for p in paths:
        parser.parse_file(p)
    parser.finish()
    return parser


def summarize(parser: Parser, specs: list[StateSpec], logs: list[dict], names: CardNames,
              ids: Identifiers | None = None) -> dict:
    """Aggregate counts only: no identifiers, no absolute times, no paths."""
    by_type = collections.Counter(s.labels["decision_type"] for s in specs)
    answered = collections.Counter(s.labels["decision_type"] for s in specs if "answered_with" in s.labels)
    valid = collections.Counter(s.labels["decision_type"] for s in specs if not s.validate())
    errors = collections.Counter()
    for s in specs:
        for e in s.validate():
            errors[re.sub(r"\{.*\}|\d+", "#", e)[:80]] += 1
    requests = sum((g.requests_by_type for g in parser.games), collections.Counter())
    answered_req = sum((g.answered_by_type for g in parser.games), collections.Counter())
    tiers = collections.Counter(s.provenance.tier for s in specs)
    flags = collections.Counter(f.split(":")[0] for s in specs for f in s.provenance.flags)
    opts = [o for s in specs if s.labels["decision_type"] == "ActionsAvailableReq"
            for o in s.labels.get("options", []) + s.labels.get("inactive", [])]
    keyed = [o for o in opts if o.get("mz_key")]
    tokens = [b for s in specs for p in s.players.values() for b in p.battlefield if b.token or b.tokenClass]
    unmapped_tokens = collections.Counter(b.token for b in tokens if not b.tokenClass)
    latencies = [s.labels["latency_s"] for s in specs if s.labels.get("latency_s") is not None]
    checks = [s.labels["state_check"] for s in specs if "state_check" in s.labels]
    counts = sum((g.counts for g in parser.games), collections.Counter())
    prio_states = sum(len(g.prio_gsids) for g in parser.games)
    prio_with_req = sum(len(g.prio_gsids & g.req_gsids) for g in parser.games)
    n_req = sum(requests.values())
    return {
        "logs": logs,
        "card_db_version": names.arena_db.version if names.arena_db else None,
        "games": len(parser.games),
        "per_game": [{"game": g.index, "game_number": g.state.info.get("gameNumber"),
                      "format": short(g.state.info.get("superFormat")),
                      "turns": g.state.turn.get("turnNumber"),
                      "on_play": (g.starting_seat == g.local_seat) if g.starting_seat else None,
                      "result": g.result(), "decisions": len(g.decisions),
                      "requests": sum(g.requests_by_type.values()),
                      "states_full": g.state.n_full, "states_diff": g.state.n_diff,
                      "states_skipped": g.state.n_skipped,      # a gameStateId not newer than the last one
                      "chain_gaps": len(g.state.gaps), "resyncs": g.resyncs,
                      "deck_size": len(g.deck)} for g in parser.games],
        "decisions": len(specs),
        "decisions_by_type": dict(by_type.most_common()),
        "decisions_answered_by_type": dict(answered.most_common()),
        "fraction_decisions_answered": round(sum(answered.values()) / max(1, len(specs)), 4),
        "requests_by_type": dict(requests.most_common()),
        "requests_answered": sum(answered_req.values()),
        "fraction_requests_paired": round(sum(answered_req.values()) / max(1, n_req), 4),
        "unpaired_client_messages": counts["unpaired_client_messages"],
        "summarized_events": parser.stats["summarized_events"],
        "requests_lost_to_summary": counts["requests_lost_to_summary"],
        "request_state_mismatch": counts["request_state_mismatch"],
        "specs_valid": sum(valid.values()),
        "fraction_specs_valid": round(sum(valid.values()) / max(1, len(specs)), 4),
        "specs_valid_by_type": dict(valid.most_common()),
        "spec_errors": dict(errors.most_common(20)),
        "specs_partial": sum(1 for s in specs if s.is_partial()),
        "tiers": dict(sorted(tiers.items())),
        "flags": dict(flags.most_common()),
        "attack_block_state_check": {"checked": len(checks), "consistent": sum(checks)},
        "action_options": len(opts), "action_options_with_mz_key": len(keyed),
        "action_options_in_fdn_vocab": sum(1 for o in keyed if "mz_idx" in o),
        "local_priority_states": prio_states, "local_priority_states_with_request": prio_with_req,
        "latency_s": ({"n": len(latencies), "median": statistics.median(latencies),
                       "max": max(latencies)} if latencies else None),
        "cards_resolved_via_arena_db": dict(sorted(collections.Counter(names.via_arena_db.values()).items())),
        "cards_unresolved_grpids": sorted(names.unresolved),
        "ability_text_sources": dict(names.ability_sources),
        "token_permanents": len(tokens),
        "token_permanents_with_class": sum(1 for b in tokens if b.tokenClass),
        "tokens_unmapped": dict(unmapped_tokens.most_common()),
        "privacy_identifier_categories": ids.counts() if ids else None,
    }


def list_games(parser: Parser) -> str:
    rows = []
    for g in parser.games:
        info = g.state.info
        play = {True: "on the play", False: "on the draw"}.get(
            (g.starting_seat == g.local_seat) if g.starting_seat else None, "play/draw unknown")
        rows.append(f"game {g.index}: {short(info.get('superFormat')) or '?'} game "
                    f"{info.get('gameNumber', '?')} of its match, {g.state.turn.get('turnNumber') or 0} turns, "
                    f"{play}, {g.result() or 'no result'}, "
                    f"{len(g.decisions)} decisions ({sum(g.requests_by_type.values())} requests), "
                    f"{len(g.state.gaps)} state-chain gap(s)")
    return "\n".join(rows) if rows else "no games found"


def default_logs() -> list[Path]:
    for d in LOG_DIRS:
        if (d / "Player-prev.log").exists() or (d / "Player.log").exists():
            return [p for p in (d / "Player-prev.log", d / "Player.log") if p.exists()]
    return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m draftzero.gameplay.arena", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", type=Path,
                    help="Player.log / Player-prev.log files, in session order (default: the local ones)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output directory (default {DEFAULT_OUT})")
    ap.add_argument("--card-db", type=Path, default=None,
                    help="Arena Raw_CardDatabase_*.mtga (default: found automatically)")
    ap.add_argument("--no-card-db", action="store_true", help="use data/17lands/cards.csv only")
    ap.add_argument("--token-table", type=Path, action="append", default=None,
                    help="TSV of token grpId -> XMage token class (grpId, token_class columns); "
                         "default: assets/gameplay/*_tokens.tsv")
    ap.add_argument("--list-games", action="store_true", help="list the games in the log and exit")
    args = ap.parse_args(argv)

    logs = [p.expanduser() for p in args.logs] or default_logs()
    if not logs:
        print("no log given and none found in the usual places", file=sys.stderr)
        return 2
    for p in logs:
        if not p.exists():
            print(f"no such log: {p.name}", file=sys.stderr)
            return 2
    infos = [dict(log_info(p), file=p.name) for p in logs]
    for i in infos:
        if i["detailed_logs"] is False:
            print(f"warning: {i['file']} says DETAILED LOGS: DISABLED; it holds no game data. "
                  "Turn on Options > Account > Detailed Logs (Plugin Support) and restart Arena.", file=sys.stderr)
    card_db = None
    if not args.no_card_db:
        card_db = args.card_db or pick_card_db(next((i["grp_build"] for i in infos if i["grp_build"]), None))
    names = CardNames.load(card_db=card_db, token_tables=args.token_table or TOKEN_TABLES)
    if not names.cards:
        print("warning: data/17lands/cards.csv not found; card names come from the Arena DB only",
              file=sys.stderr)
    ids = Identifiers.from_logs(logs)
    parser = parse_logs(logs, names)
    allowed = {w for s in names.emitted for w in re.findall(r"\w+", s)}

    if args.list_games:
        text = list_games(parser)
        check_privacy([text], ids, allowed)
        print(text)
        return 0

    specs = parser.specs()
    lines = [spec_json(s) for s in specs]
    summary = summarize(parser, specs, [{k: v for k, v in i.items() if k != "grp_build"} for i in infos],
                        names, ids)
    summary_text = json.dumps(summary, indent=1, ensure_ascii=False)
    check_privacy(["\n".join(lines), summary_text, list_games(parser)], ids, allowed)   # before any write
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "decisions.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    (args.out / "summary.json").write_text(summary_text + "\n", encoding="utf-8")
    rel = os.path.relpath(args.out.resolve(), Path.cwd())
    shown = rel if not rel.startswith("..") else args.out.name     # never print a home directory
    print(list_games(parser))
    print(f"{len(specs)} decisions ({summary['fraction_specs_valid']:.1%} valid specs, "
          f"{summary['fraction_requests_paired']:.1%} of requests paired) -> {shown}/decisions.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
