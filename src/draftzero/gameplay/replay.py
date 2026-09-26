"""
Stream 17lands replay rows (replay_data_public.<SET>.<FMT>.csv.gz) into `Game` records.

One CSV row is one game seen from the 17lands user's side ("user") against an opponent ("oppo").
The row is not a log: for each half-turn it holds per-player event multisets (cast, drawn,
attacked, ... with no order, targets or phase) plus an end-of-turn snapshot of every visible zone.
Semantics verified in docs/008 (replay_empirics.md §3, web_17lands.md §3):

  - turn N is per player; the order is U1,O1,U2,... when on_play, else O1,U1,O2,...
  - global turn numbering (statespec): user turn N is 2N-1 on the play and 2N on the draw; the
    opponent's turn N is 2N on the play and 2N-1 on the draw
  - `num_turns` is the per-player index of the last played slot; all 30 slots exist and the ones
    past the end are zero padding (life (0, 0)): they are dropped here
  - card fields are '|'-separated grpId multisets; eot_oppo_cards_in_hand and
    oppo_turn_N_cards_drawn_or_tutored are counts
  - un-prefixed fields (lands_played, creatures_cast, cards_drawn, creatures_attacked, ...)
    describe the ACTIVE player; creatures_blocking is the non-active player's
  - opening_hand is the kept 7 before London-mulligan bottoming; the bottomed cards are inferred
    from the first snapshot (`Game.bottomed`, exact in ~99.6% of mulligan games)
  - the last slot of a game is `terminal`: the game ended during it, so its events and snapshot
    can be incomplete; never use it as a decision state

Usage:
  from draftzero.gameplay.replay import iter_games
  for g in iter_games(path, limit=1000, every=16):
      for t in g.turns: ...

  python -m draftzero.gameplay.replay --limit 20000            # parse throughput
  python -m draftzero.gameplay.replay --row 0                  # print one game's timeline

17lands public data is CC BY 4.0 (https://www.17lands.com/public_datasets).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from draftzero.gameplay.ids import DATA_DIR

csv.field_size_limit(sys.maxsize)

SIDES = ("user", "oppo")
# Per-turn fields that are counts (float in the file); every other per-turn field is an id list.
NUM_FIELDS = frozenset((
    "oppo_combat_damage_taken", "user_combat_damage_taken", "user_mana_spent", "oppo_mana_spent",
    "eot_oppo_cards_in_hand", "eot_user_life", "eot_oppo_life", "cards_drawn_or_tutored",
    "cards_drawn_count", "cards_tutored_count",
))
META_INT = ("build_index", "match_number", "game_number", "num_mulligans", "opp_num_mulligans",
            "num_turns", "user_n_games_bucket")
META_FLOAT = ("user_game_win_rate_bucket",)
META_BOOL = ("on_play", "won")
_TURN_RE = re.compile(r"^(user|oppo)_turn_(\d+)_(.+)$")


def replay_path(set_code: str = "FDN", fmt: str = "PremierDraft", data_dir=None) -> Path:
    return Path(data_dir or DATA_DIR) / f"replay_data_public.{set_code}.{fmt}.csv.gz"


def ids_of(s: str) -> list[int]:
    """'93757|95194' -> [93757, 95194]; '' -> []"""
    return [int(x) for x in s.split("|")] if s else []


class Header:
    """Column index maps for a replay CSV header (the parser is header-driven, so files with fewer
    columns, e.g. test fixtures or other sets' schemas, parse the same way)."""

    def __init__(self, cols: list[str]):
        self.cols = cols
        self.meta: dict[str, int] = {}
        self.deck: list[tuple[str, int]] = []
        self.sideboard: list[tuple[str, int]] = []
        self.turn: dict[tuple[str, int], list[tuple[str, int, bool]]] = {}
        self.max_turn = 0
        for i, c in enumerate(cols):
            m = _TURN_RE.match(c)
            if m:
                side, n, f = m.group(1), int(m.group(2)), m.group(3)
                self.turn.setdefault((side, n), []).append((f, i, f in NUM_FIELDS))
                self.max_turn = max(self.max_turn, n)
            elif c.startswith("deck_"):
                self.deck.append((c[5:], i))
            elif c.startswith("sideboard_"):
                self.sideboard.append((c[10:], i))
            elif "_total_" not in c:      # totals equal the per-turn sums in 100% of rows
                self.meta[c] = i


@dataclass
class TurnRecord:
    side: str                 # "user" or "oppo": the ACTIVE player of this half-turn
    n: int                    # that player's own turn number (1-based)
    seq: int = 0              # chronological slot index in the game (0-based)
    global_turn: int = 0      # statespec turn number (both players' turns counted)
    f: dict = field(default_factory=dict)   # field -> list[int] (id multiset) or float (count)
    played: bool = False      # False only for an empty "hole" slot inside the game (0.12% of games)
    terminal: bool = False    # the game's last slot: possibly incomplete, never a decision state

    def L(self, name: str) -> list[int]:
        v = self.f.get(name)
        return v if isinstance(v, list) else []

    def num(self, name: str, default: float = 0.0) -> float:
        v = self.f.get(name)
        return v if isinstance(v, float) else default

    def get(self, name: str, default=None):
        return self.f.get(name, default)

    @property
    def other(self) -> str:
        return "oppo" if self.side == "user" else "user"


@dataclass
class Game:
    row_index: int
    meta: dict
    deck: Counter                     # maindeck, card name -> count
    sideboard: Counter
    candidate_hands: list[list[int]]  # the 7-card hands seen during mulligans, in order
    opening_hand: list[int]           # the kept 7, before bottoming
    bottomed: list[int]               # inferred London-mulligan bottoms
    bottomed_exact: bool
    turns: list[TurnRecord]           # chronological, padding removed

    @property
    def on_play(self) -> bool:
        return bool(self.meta.get("on_play"))

    @property
    def won(self) -> bool:
        return bool(self.meta.get("won"))

    @property
    def num_turns(self) -> int:
        return int(self.meta.get("num_turns") or 0)

    def user_slot(self, n: int) -> TurnRecord | None:
        """The TurnRecord of user turn n (None if the game did not reach it)."""
        k = (2 * n - 2) if self.on_play else (2 * n - 1)
        if 0 <= k < len(self.turns) and self.turns[k].side == "user" and self.turns[k].n == n:
            return self.turns[k]
        return next((t for t in self.turns if t.side == "user" and t.n == n), None)

    def prev_slot(self, n: int) -> TurnRecord | None:
        """The half-turn before user turn n: the opponent's turn whose end-of-turn snapshot is the
        state the user's turn n starts from (None for user turn 1 on the play)."""
        t = self.user_slot(n)
        return self.turns[t.seq - 1] if t is not None and t.seq > 0 else None

    def next_slot(self, n: int) -> TurnRecord | None:
        """The opponent's half-turn right after user turn n."""
        t = self.user_slot(n)
        return self.turns[t.seq + 1] if t is not None and t.seq + 1 < len(self.turns) else None

    def user_turn_numbers(self) -> list[int]:
        return [t.n for t in self.turns if t.side == "user"]

    def decision_turns(self) -> list[int]:
        """User turns usable as decision states: played and not the game's last slot."""
        return [t.n for t in self.turns if t.side == "user" and t.played and not t.terminal]


def global_turn(side: str, n: int, on_play: bool) -> int:
    first = (side == "user") == on_play       # this side took the odd turns
    return 2 * n - 1 if first else 2 * n


def _parse_meta(row: list[str], H: Header) -> dict:
    meta = {k: row[i] for k, i in H.meta.items()}
    for k in META_INT:
        v = meta.get(k)
        if v not in (None, ""):
            meta[k] = int(float(v))
    for k in META_FLOAT:
        v = meta.get(k)
        meta[k] = float(v) if v not in (None, "") else None
    for k in META_BOOL:
        if k in meta:
            meta[k] = meta[k] == "True"
    for k in ("rank", "opp_rank"):
        if meta.get(k) in ("", "None"):
            meta[k] = None
    return meta


def parse_game(row: list[str], H: Header, row_index: int = -1) -> Game:
    meta = _parse_meta(row, H)
    if meta.get("source_row") not in (None, ""):        # an excerpt keeps the full file's row index
        row_index = int(meta.pop("source_row"))
    deck, side = Counter(), Counter()
    for name, i in H.deck:
        v = row[i]
        if v and v != "0":
            deck[name] = int(float(v))
    for name, i in H.sideboard:
        v = row[i]
        if v and v != "0":
            side[name] = int(float(v))
    cands = [ids_of(meta.pop(f"candidate_hand_{k}", "") or "") for k in range(1, 8)]
    cands = [c for c in cands if c]
    opening = ids_of(meta.pop("opening_hand", "") or "")
    on_play = bool(meta.get("on_play"))
    order = ("user", "oppo") if on_play else ("oppo", "user")
    # decode only up to num_turns (+1 guard); everything past it is zero padding
    last_n = min(H.max_turn, (meta.get("num_turns") or H.max_turn) + 1)
    turns: list[TurnRecord] = []
    for n in range(1, last_n + 1):
        for s in order:
            rec: dict = {}
            played = False
            for f, i, is_num in H.turn.get((s, n), ()):
                v = row[i]
                if is_num:
                    x = float(v) if v else 0.0
                    rec[f] = x
                    played = played or x != 0.0
                else:
                    rec[f] = [int(y) for y in v.split("|")] if v else []
                    played = played or bool(v)
            turns.append(TurnRecord(s, n, f=rec, played=played,
                                    global_turn=global_turn(s, n, on_play)))
    last = max((k for k, t in enumerate(turns) if t.played), default=-1)
    turns = turns[:last + 1]
    for k, t in enumerate(turns):
        t.seq = k
    if turns:
        turns[-1].terminal = True
    g = Game(row_index, meta, deck, side, cands, opening, [], True, turns)
    g.bottomed, g.bottomed_exact = infer_bottomed(g)
    return g


def infer_bottomed(g: Game) -> tuple[list[int], bool]:
    """London-mulligan bottoms: opening hand + first-slot draws - first-slot plays - first eot hand.
    Exact when exactly `num_mulligans` cards are unaccounted for."""
    k = int(g.meta.get("num_mulligans") or 0)
    if k == 0 or not g.turns:
        return [], k == 0
    t = g.turns[0]
    hand = Counter(g.opening_hand)
    if t.side == "user":
        hand.update(t.L("cards_drawn") + t.L("cards_tutored"))
        hand.subtract(t.L("lands_played") + t.L("creatures_cast") + t.L("non_creatures_cast")
                      + t.L("cards_discarded"))
    hand.subtract(t.L("user_instants_sorceries_cast"))
    missing = +(hand - Counter(t.L("eot_user_cards_in_hand")))
    out = sorted(missing.elements())
    return out[:k], len(out) == k


# --- I/O --------------------------------------------------------------------------------------------

def open_lines(path) -> tuple[Header, Iterator[str]]:
    """(Header, iterator over raw data lines). The file has no embedded newlines (791,160 physical
    lines = header + 791,159 rows), so rows can be skipped before CSV parsing."""
    f = gzip.open(path, "rt", newline="", encoding="utf-8")
    H = Header(next(csv.reader([f.readline()])))
    return H, f


def iter_games(path=None, limit: int | None = None, every: int = 1, start: int = 0,
               predicate: Callable[[Game], bool] | None = None, stop: int | None = None) -> Iterator[Game]:
    """Yield parsed games for rows start, start+every, ... (0-based data-row indices; `stop`
    excludes rows >= stop). `predicate(game)` filters after parsing; `limit` counts yielded games."""
    H, lines = open_lines(path or replay_path())
    yielded = 0
    try:
        for i, line in enumerate(lines):
            if stop is not None and i >= stop:
                break
            if i < start or (i - start) % every:
                continue
            g = parse_game(next(csv.reader([line])), H, i)
            if predicate is not None and not predicate(g):
                continue
            yield g
            yielded += 1
            if limit is not None and yielded >= limit:
                break
    finally:
        lines.close()


def read_games(rows, path=None) -> dict[int, Game]:
    """Parse specific data rows (0-based) -> {row: Game}. Stops reading after the last one."""
    want = set(rows)
    out: dict[int, Game] = {}
    if not want:
        return out
    H, lines = open_lines(path or replay_path())
    src = H.meta.get("source_row")          # an excerpt: match the full file's row index
    try:
        for i, line in enumerate(lines):
            if src is not None:
                row = next(csv.reader([line]))
                if int(row[src]) in want:
                    g = parse_game(row, H, i)
                    out[g.row_index] = g
            elif i in want:
                out[i] = parse_game(next(csv.reader([line])), H, i)
            if len(out) == len(want):
                break
    finally:
        lines.close()
    return out


def write_excerpt(rows, out_path, path=None, blank=("draft_id",)) -> int:
    """Write selected data rows to a small .csv.gz (test fixtures): drops totals, turn slots past the
    rows' largest num_turns and deck/sideboard columns that are empty in every kept row, and blanks
    the `blank` columns. The parser is header-driven, so the excerpt parses like the full file.
    A `source_row` column keeps each row's index in the full file (parse_game uses it as row_index)."""
    want = sorted(set(rows))
    H, lines = open_lines(path or replay_path())
    got: dict[int, list[str]] = {}
    try:
        for i, line in enumerate(lines):
            if i in want:
                got[i] = next(csv.reader([line]))
                if len(got) == len(want):
                    break
    finally:
        lines.close()
    max_n = max(int(float(r[H.meta["num_turns"]] or 0)) for r in got.values())
    keep = []
    for i, c in enumerate(H.cols):
        m = _TURN_RE.match(c)
        if m:
            if int(m.group(2)) <= max_n:
                keep.append(i)
        elif c.startswith(("deck_", "sideboard_")):
            if any(r[i] not in ("", "0") for r in got.values()):
                keep.append(i)
        elif "_total_" not in c:
            keep.append(i)
    blank_idx = {H.cols.index(b) for b in blank if b in H.cols}
    with gzip.open(out_path, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["source_row"] + [H.cols[i] for i in keep])
        for r in want:
            w.writerow([str(r)] + ["" if i in blank_idx else got[r][i] for i in keep])
    return len(keep)


def format_game(g: Game, ids) -> str:
    """Readable timeline of a game (card names instead of ids)."""
    nm = lambda xs: ", ".join(ids.name(x) for x in xs)
    short = {"cards_drawn": "draw", "cards_tutored": "tutor", "cards_discarded": "disc", "lands_played": "land",
             "creatures_cast": "crea", "non_creatures_cast": "nonc", "user_instants_sorceries_cast": "U_IS",
             "oppo_instants_sorceries_cast": "O_IS", "creatures_attacked": "att", "creatures_blocked": "blkd",
             "creatures_unblocked": "unblk", "creatures_blocking": "blkr",
             "user_creatures_killed_combat": "U_kc", "oppo_creatures_killed_combat": "O_kc",
             "user_creatures_killed_non_combat": "U_knc", "oppo_creatures_killed_non_combat": "O_knc"}
    m = g.meta
    lines = [f"row={g.row_index} on_play={g.on_play} won={g.won} num_turns={g.num_turns} "
             f"mull={m.get('num_mulligans')} opp_mull={m.get('opp_num_mulligans')} colors={m.get('main_colors')}"
             f"{'+' + m['splash_colors'] if m.get('splash_colors') else ''} vs {m.get('opp_colors')}",
             f"opening: {nm(g.opening_hand)}" + (f" | bottomed: {nm(g.bottomed)}" if g.bottomed else "")]
    for t in g.turns:
        ev = [f"{s}=[{nm(t.L(f))}]" for f, s in short.items() if t.L(f)]
        for w in ("user", "oppo"):
            if t.L(f"{w}_abilities"):
                ev.append(f"{w[0].upper()}_abil={t.L(f'{w}_abilities')}")
        if t.side == "oppo":
            ev.append(f"O_drawn#{int(t.num('cards_drawn_or_tutored'))}")
        ev.append(f"mana U{t.num('user_mana_spent'):g}/O{t.num('oppo_mana_spent'):g}")
        lines.append(f"[{t.seq:2d}] {t.side.upper()} T{t.n} (g{t.global_turn}){' END' if t.terminal else ''}: "
                     + " ".join(ev))
        lines.append(f"      life U{t.num('eot_user_life'):g} O{t.num('eot_oppo_life'):g} | U hand: "
                     f"{nm(t.L('eot_user_cards_in_hand'))} | O hand#{int(t.num('eot_oppo_cards_in_hand'))}")
        for w in ("user", "oppo"):
            lines.append(f"      {w[0].upper()} bf: L[{nm(t.L(f'eot_{w}_lands_in_play'))}] "
                         f"C[{nm(t.L(f'eot_{w}_creatures_in_play'))}] N[{nm(t.L(f'eot_{w}_non_creatures_in_play'))}]")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m draftzero.gameplay.replay",
                                 description="Parse throughput, or print one game's timeline.")
    ap.add_argument("--path", default=None, help="replay csv.gz (default data/17lands FDN PremierDraft)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--row", type=int, default=None, help="print this row's timeline")
    a = ap.parse_args(argv)
    if a.row is not None:
        from draftzero.gameplay.ids import Ids
        g = read_games([a.row], a.path)[a.row]
        print(format_game(g, Ids.load()))
        return 0
    t0 = time.time()
    n = slots = 0
    for g in iter_games(a.path, a.limit, a.every, a.start):
        n += 1
        slots += len(g.turns)
    dt = time.time() - t0
    print(f"{n} games, {slots} slots in {dt:.1f} s: {n / dt:.0f} games/s (every={a.every})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
