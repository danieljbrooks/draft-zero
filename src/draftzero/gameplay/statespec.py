"""
StateSpec v1: one JSON description of a two-player Magic game state, shared by everything that
maps human gameplay data onto XMage.

    17lands replay row ─┐
    Arena Player.log  ──┼──►  StateSpec (this module, JSON)  ──►  java/mzbridge  ──►  XMage Game
    synthetic / tests ──┘                                        (build, legal, coach, encode)

Python producers (`seventeenlands`, `arena`) write it; the Java bridge (`java/mzbridge`, Gson)
reads it and builds the game with the event-free injection recipe from the research proof of
concept. Keep the two sides in step: a field added here must be added to the Java mirror, and
`SCHEMA_VERSION` bumped when a change is not backward compatible.

Conventions
-----------
- Seats are "A" and "B". For 17lands data, A is the 17lands user (the player whose hand is known)
  and B the opponent. For Arena logs, A is the local seat.
- `turn` is the GLOBAL turn number (both players' turns counted), 1-based. The player on the
  play takes the odd turns. 17lands' per-player turn N maps to global turn 2N-1 for the player
  on the play and 2N for the player on the draw.
- Card names are XMage card names (identical to 17lands/Arena names for FDN). `set`/`number`
  optionally pin a printing. Tokens use `tokenClass` (a class under
  `mage.game.permanent.token`, e.g. "CatToken3"), or `token` + `set` (a tokens-database.txt name).
- Every player needs a full `decklist` (at least 40 cards; XMage's MCTS player refuses smaller
  decks). The library is the decklist minus every card named in the other zones; tokens are not in
  the decklist. `decklistSource` says whether the list is the real one ("exact"), a sample from a
  belief model ("belief") or filler ("placeholder").
- Hidden cards. `hand` lists the known cards; `handUnknown` counts cards whose identity is not
  known. A spec with `handUnknown > 0` is a *partial* spec: before XMage can build it, either a
  Python determinizer fills the hand (see `belief`), or the bridge draws the unknown cards at
  random from the remaining library with the request's seed. `libraryTop` lists known top cards
  (first = top), e.g. the card 17lands says was drawn next.
- `provenance` records where the spec came from and how much of it is certain; `labels` carries
  whatever the human actually did (a pass-through for training and coaching; the bridge ignores it).
  `labels["bridge"]`, when present, holds the bridge request options (decisionPlayer, decideFrom)
  that reach the decision the labels describe; pass them with every request.

Conventions the bridge fixes (java/mzbridge/README.md has the details):
- `stack` is listed bottom to top (cast order): the last item resolves first.
- `sick` follows the rules: the permanent has NOT been under its controller's control continuously
  since that player's most recent turn began (so a creature A cast on its own turn is still sick
  during B's next turn, for {T} abilities).
- `counters` SET the count of each listed type; unlisted types keep what the card entered with
  (e.g. a planeswalker's printed loyalty).
- An attacker's `tapped` flag is ignored: declaring the attack taps it (unless vigilance).
- A battlefield entry with count > 1 and id "x" defines the aliases "<seat>:x" (the first copy) and
  "<seat>:x#1" .. "<seat>:x#<count>"; such an entry cannot carry attachTo.
- `token` + `set` must name exactly one XMage token class (FDN "Beast" and "Cat" do not; use
  tokenClass).
- A permanent is listed under its CONTROLLER's seat. `owner` names the other seat when control has
  changed (a reanimated or stolen creature); its card then comes out of the owner's decklist.
- UNTAP cannot be entered (enter the previous turn's END_TURN instead); CLEANUP needs BEGIN_STEP;
  BEGIN_STEP needs an empty stack; turn 1 belongs to the starting player.

Entry points (`enterMode`, the same semantics as the proof of concept):
- PRIORITY_FRESH: resume inside the step's priority window, skipping its turn-based actions
  (no untap, draw or attacker declaration); priority starts with the active player.
- BEGIN_STEP: run the step from its start (turn-based actions such as the draw in DRAW, combat
  set-up in BEGIN_COMBAT, attacker declaration in DECLARE_ATTACKERS).
- PRIORITY_HELD: resume mid priority round: priority starts with `priorityPlayer`, and the players
  in `passedPlayers` count as having passed (e.g. responding to a spell on the stack).
The recommended entry for an end-of-turn snapshot is the END_TURN step of that turn with
PRIORITY_FRESH: the engine then plays cleanup and the next turn's untap, upkeep and draw itself.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any

SCHEMA_VERSION = 1

SEATS = ("A", "B")
ENTER_MODES = ("PRIORITY_FRESH", "BEGIN_STEP", "PRIORITY_HELD")
DECKLIST_SOURCES = ("exact", "belief", "placeholder")
SOURCES = ("17lands", "arena", "synthetic")

# mage.constants.TurnPhase -> the mage.constants.PhaseStep values inside it
PHASE_STEPS = {
    "BEGINNING": ("UNTAP", "UPKEEP", "DRAW"),
    "PRECOMBAT_MAIN": ("PRECOMBAT_MAIN",),
    "COMBAT": ("BEGIN_COMBAT", "DECLARE_ATTACKERS", "DECLARE_BLOCKERS",
               "FIRST_COMBAT_DAMAGE", "COMBAT_DAMAGE", "END_COMBAT"),
    "POSTCOMBAT_MAIN": ("POSTCOMBAT_MAIN",),
    "END": ("END_TURN", "CLEANUP"),
}

# Fidelity tiers for a reconstructed state, most to least certain. A tier is a promise about the
# spec, not about the data source: an Arena state is normally T0, a 17lands state T0..T3.
TIERS = {
    "T0": "every visible identity, tapped state, counter and attachment is known",
    "T1": "visible identities known; tapped state or counters inferred",
    "T2": "visible identities known; some attachment, exile link or copy is ambiguous",
    "T3": "some visible identity is ambiguous (e.g. an unexplained zone change)",
}


@dataclass
class Perm:
    """A permanent on the battlefield (a card or a token)."""
    name: str | None = None          # card name; None for tokens given by tokenClass/token
    set: str | None = None
    number: str | None = None
    token: str | None = None         # tokens-database.txt name (with `set`), e.g. "Beast"
    tokenClass: str | None = None    # mage.game.permanent.token class, e.g. "CatToken3"
    id: str | None = None            # local alias; referenced as "<seat>:<id>" by attachTo/targets
    count: int = 1
    tapped: bool = False
    sick: bool = False               # came under its controller's control this turn
    damage: int = 0
    counters: dict[str, int] = field(default_factory=dict)   # CounterType enum name -> amount, e.g. {"P1P1": 1}
    attachTo: str | None = None      # "<seat>:<id>" of the host (Aura / Equipment)
    faceDown: bool = False
    owner: str | None = None         # the other seat, when it owns this permanent (control changed)


@dataclass
class PlayerState:
    name: str = ""
    life: int = 20
    decklist: list[str] = field(default_factory=list)       # full maindeck, one entry per copy
    decklistSource: str = "exact"
    landsPlayed: int = 0             # lands played this turn
    hand: list[str] = field(default_factory=list)
    handUnknown: int = 0
    graveyard: list[str] = field(default_factory=list)      # bottom -> top
    exile: list[str] = field(default_factory=list)
    libraryTop: list[str] = field(default_factory=list)     # known top cards, first = top
    librarySize: int | None = None   # trim the library to this many cards (None = keep the remainder)
    manaPool: str | None = None      # e.g. "WWB"
    battlefield: list[Perm] = field(default_factory=list)


@dataclass
class StackItem:
    controller: str                  # "A" / "B"; stack items are listed bottom to top
    card: str
    targets: list[str] = field(default_factory=list)       # "<seat>:<id>" or "player:A"


@dataclass
class Attack:
    attacker: str                    # "<seat>:<id>"
    defender: str                    # "player:A" or "<seat>:<id>" (planeswalker / battle)


@dataclass
class Block:
    blocker: str
    attacker: str


@dataclass
class Provenance:
    source: str = "synthetic"        # one of SOURCES
    ref: str = ""                    # e.g. "FDN_PremierDraft:row=1234:user_turn=5"; never a user identifier
    tier: str | None = None          # one of TIERS
    flags: list[str] = field(default_factory=list)          # why the tier is not T0, and other caveats
    sampled: list[str] = field(default_factory=list)        # fields filled by a belief model, e.g. "B.decklist"


@dataclass
class StateSpec:
    version: int = SCHEMA_VERSION
    comment: str = ""
    turn: int = 1
    activePlayer: str = "A"
    phase: str = "PRECOMBAT_MAIN"
    step: str = "PRECOMBAT_MAIN"
    enterMode: str = "PRIORITY_FRESH"
    priorityPlayer: str | None = None
    passedPlayers: list[str] = field(default_factory=list)
    startingPlayer: str | None = None    # who was on the play; None = derive from turn parity
    players: dict[str, PlayerState] = field(default_factory=dict)
    stack: list[StackItem] = field(default_factory=list)
    attackers: list[Attack] = field(default_factory=list)
    blockers: list[Block] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)
    labels: dict[str, Any] = field(default_factory=dict)

    # --- serialisation -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        return _prune(asdict(self))

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_dict(cls, d: dict) -> "StateSpec":
        d = dict(d)
        d.pop("ai", None)            # the proof of concept kept AI settings in the spec; requests carry them now
        players = {k: _build(PlayerState, v, battlefield=lambda xs: [_build(Perm, x) for x in xs])
                   for k, v in d.pop("players", {}).items()}
        spec = _build(cls, d,
                      stack=lambda xs: [_build(StackItem, x) for x in xs],
                      attackers=lambda xs: [_build(Attack, x) for x in xs],
                      blockers=lambda xs: [_build(Block, x) for x in xs],
                      provenance=lambda x: _build(Provenance, x))
        spec.players = players
        return spec

    @classmethod
    def from_json(cls, s: str) -> "StateSpec":
        return cls.from_dict(json.loads(s))

    # --- checks --------------------------------------------------------------------------------

    def is_partial(self) -> bool:
        """True when some hidden card still has to be filled in before XMage can build the state."""
        return any(p.handUnknown > 0 for p in self.players.values())

    def aliases(self) -> set[str]:
        """Aliases defined by battlefield entries: "<seat>:<id>", plus "<seat>:<id>#k" for count > 1."""
        out = set()
        for seat, p in self.players.items():
            for perm in p.battlefield:
                if perm.id:
                    out.add(f"{seat}:{perm.id}")
                    if perm.count > 1:
                        out.update(f"{seat}:{perm.id}#{k}" for k in range(1, perm.count + 1))
        return out

    def validate(self) -> list[str]:
        """Problems that would make the bridge reject or misbuild this spec. Empty = fine.

        Mirrors Spec.validate() in java/mzbridge, so a spec that passes here also passes there.
        """
        errs: list[str] = []
        if self.version != SCHEMA_VERSION:
            errs.append(f"version {self.version} != {SCHEMA_VERSION}")
        if set(self.players) != set(SEATS):
            errs.append(f"players must be exactly {SEATS}, got {sorted(self.players)}")
            return errs
        if self.activePlayer not in SEATS:
            errs.append(f"activePlayer {self.activePlayer!r}")
        if self.startingPlayer is not None and self.startingPlayer not in SEATS:
            errs.append(f"startingPlayer {self.startingPlayer!r}")
        if self.phase not in PHASE_STEPS:
            errs.append(f"phase {self.phase!r}")
        elif self.step not in PHASE_STEPS[self.phase]:
            errs.append(f"step {self.step!r} is not in phase {self.phase}")
        if self.enterMode not in ENTER_MODES:
            errs.append(f"enterMode {self.enterMode!r}")
        if self.enterMode == "PRIORITY_HELD" and self.priorityPlayer not in SEATS:
            errs.append("PRIORITY_HELD needs priorityPlayer")
        if self.priorityPlayer is not None and self.priorityPlayer not in SEATS:
            errs.append(f"priorityPlayer {self.priorityPlayer!r}")
        errs += [f"passedPlayers entry {s!r}" for s in self.passedPlayers if s not in SEATS]
        if self.turn < 1:
            errs.append(f"turn {self.turn}")
        if self.turn == 1 and self.startingPlayer is not None and self.startingPlayer != self.activePlayer:
            errs.append(f"turn 1 belongs to the starting player: startingPlayer {self.startingPlayer} "
                        f"!= activePlayer {self.activePlayer}")
        if self.step == "UNTAP":
            errs.append("UNTAP cannot be entered: use the previous turn's END_TURN with PRIORITY_FRESH")
        if self.step == "CLEANUP" and self.enterMode != "BEGIN_STEP":
            errs.append("CLEANUP has no priority window: use BEGIN_STEP at CLEANUP, or END_TURN")
        if self.enterMode == "BEGIN_STEP" and self.stack:
            errs.append("BEGIN_STEP with a non-empty stack: a step only begins once the stack is empty")
        if self.provenance.source not in SOURCES:
            errs.append(f"provenance.source {self.provenance.source!r}")
        if self.provenance.tier is not None and self.provenance.tier not in TIERS:
            errs.append(f"provenance.tier {self.provenance.tier!r}")

        aliases = self.aliases()
        seen: set[str] = set()
        # cards each seat OWNS outside its library: its own zones, permanents it owns on either
        # battlefield (owner = the other seat when control changed), and stack spells it controls
        owned: dict[str, list[str]] = {seat: [] for seat in SEATS}
        for seat, p in self.players.items():
            owned[seat] += p.hand + p.graveyard + p.exile + p.libraryTop
            for perm in p.battlefield:
                if perm.name and not _is_token(perm):
                    owned[perm.owner or seat] += [perm.name] * perm.count
        for item in self.stack:
            if item.controller in owned and item.card:
                owned[item.controller].append(item.card)
        for seat, p in self.players.items():
            if p.decklistSource not in DECKLIST_SOURCES:
                errs.append(f"{seat}.decklistSource {p.decklistSource!r}")
            if len(p.decklist) < 40:
                errs.append(f"{seat}.decklist has {len(p.decklist)} cards; XMage needs at least 40")
            if p.handUnknown < 0:
                errs.append(f"{seat}.handUnknown < 0")
            if p.landsPlayed < 0:
                errs.append(f"{seat}.landsPlayed < 0")
            if p.librarySize is not None and p.librarySize < len(p.libraryTop):
                errs.append(f"{seat}.librarySize {p.librarySize} < libraryTop count {len(p.libraryTop)}")
            if p.manaPool is not None and not re.fullmatch(r"[WUBRGC]*", p.manaPool):
                errs.append(f"{seat}.manaPool {p.manaPool!r} (use W U B R G C)")
            pool = _multiset(p.decklist)
            short = {c: n - pool.get(c, 0) for c, n in _multiset(owned[seat]).items() if n > pool.get(c, 0)}
            if short:
                errs.append(f"{seat}: cards named in zones but missing from decklist: {short}")
            remaining = len(p.decklist) - len(owned[seat])
            if p.handUnknown > max(0, remaining):
                errs.append(f"{seat}.handUnknown {p.handUnknown} > {remaining} cards left in the library")
            for perm in p.battlefield:
                what = perm.name or perm.tokenClass or perm.token
                if not (perm.name or perm.token or perm.tokenClass):
                    errs.append(f"{seat}: battlefield entry with no name/token/tokenClass")
                if perm.attachTo and perm.attachTo not in aliases:
                    errs.append(f"{seat}: attachTo {perm.attachTo!r} names no permanent alias")
                if perm.count < 1:
                    errs.append(f"{seat}: battlefield count {perm.count}")
                if perm.damage < 0:
                    errs.append(f"{seat}: negative damage on {what}")
                if perm.id and f"{seat}:{perm.id}" in seen:
                    errs.append(f"{seat}: duplicate alias {perm.id!r}")
                if perm.id:
                    seen.add(f"{seat}:{perm.id}")
                if perm.attachTo and perm.count > 1:
                    errs.append(f"{seat}: attachTo with count > 1 on {what}")
                if any(v is None or v < 0 for v in perm.counters.values()):
                    errs.append(f"{seat}: counters on {what} must be >= 0")
                if perm.owner is not None and perm.owner not in SEATS:
                    errs.append(f"{seat}: owner {perm.owner!r} on {what}")
                if perm.owner is not None and _is_token(perm):
                    errs.append(f"{seat}: owner on a token ({what}) is not supported")

        other = "B" if self.activePlayer == "A" else "A"
        for a in self.attackers:
            if a.attacker not in aliases:
                errs.append(f"attacker {a.attacker!r} names no permanent alias")
            if not a.attacker.startswith(f"{self.activePlayer}:"):
                errs.append(f"attacker {a.attacker!r} is not controlled by the active player")
            if a.defender != f"player:{other}" and a.defender not in aliases:
                errs.append(f"defender {a.defender!r} must be player:{other} or a permanent alias")
        for b in self.blockers:
            if b.blocker not in aliases or b.attacker not in aliases:
                errs.append(f"block {b.blocker!r}->{b.attacker!r} names no permanent alias")
        steps = [st for sts in PHASE_STEPS.values() for st in sts]
        at = steps.index(self.step) if self.step in steps else -1
        if self.attackers and (self.phase != "COMBAT" or at < steps.index("DECLARE_ATTACKERS")
                               or (self.step == "DECLARE_ATTACKERS" and self.enterMode == "BEGIN_STEP")):
            errs.append("attackers need a position after attackers were declared "
                        f"(got {self.step}/{self.enterMode})")
        if self.blockers:
            if not self.attackers:
                errs.append("blockers without attackers")
            if (self.phase != "COMBAT" or at < steps.index("DECLARE_BLOCKERS")
                    or (self.step == "DECLARE_BLOCKERS" and self.enterMode == "BEGIN_STEP")):
                errs.append("blockers need a position after blockers were declared "
                            f"(got {self.step}/{self.enterMode})")
        for item in self.stack:
            if item.controller not in SEATS:
                errs.append(f"stack controller {item.controller!r}")
            for t in item.targets:
                if t not in ("player:A", "player:B") and t not in aliases:
                    errs.append(f"stack target {t!r} names no permanent alias or player")
        return errs


def _is_token(perm: Perm) -> bool:
    return bool(perm.token or perm.tokenClass)


def _multiset(xs: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def _build(cls, d: dict, **nested):
    """Construct a dataclass from a dict, ignoring unknown keys (forward compatibility)."""
    names = {f.name for f in fields(cls)}
    kw = {}
    for k, v in d.items():
        if k not in names:
            continue
        kw[k] = nested[k](v) if k in nested and v is not None else v
    return cls(**kw)


def _prune(x):
    """Drop None values and empty containers so specs stay small and readable."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            v = _prune(v)
            if v is None or (isinstance(v, (list, dict)) and not v and k not in ("players",)):
                continue
            out[k] = v
        return out
    if isinstance(x, list):
        return [_prune(v) for v in x]
    return x


def load(path) -> StateSpec:
    with open(path) as f:
        return StateSpec.from_json(f.read())


def dump(spec: StateSpec, path) -> None:
    with open(path, "w") as f:
        f.write(spec.to_json(indent=1) + "\n")
