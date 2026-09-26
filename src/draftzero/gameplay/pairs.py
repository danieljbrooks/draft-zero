"""
Mirrored 17lands games: the same game recorded by BOTH players, as two replay rows.

13.7% of the FDN Premier rows are such halves (critique.md N1). For those games we know both
players' hands at every half-turn, both decklists and both draw sequences: ground truth for the
opponent belief model (`belief`), perfect-information encodings and a hindsight-free coaching test
set. No public key links the two rows, so they are matched on the game itself:

  key     the battlefield after the on-the-play player's 3rd turn (both players' lands and
          creatures, as sorted grpId multisets): user_turn_3 in the on-play row, oppo_turn_3 in the
          on-draw row
  match   within a key, an on-play row and an on-draw row with opposite `won`, swapped mulligan
          counts, equal `num_turns` and game_time less than 15 min apart; the closest in time wins.
          A row of either side with two or more such candidates counts as ambiguous (0 on FDN
          Premier, both sides; counted before the greedy assignment, so file order cannot hide one).
  check   an independent turn: at the on-play player's turn 5, its hand size and creatures_cast
          must equal what the other row saw (oppo_turn_5 hand count and creatures_cast)

Games that ended before the on-play player's 3rd turn, or where it had no land by then, are not
matched. The file row order is kept; the pass reads only the columns it needs (~40 s).

Output: data/gameplay/pairs_<SET>_<FMT>.jsonl, one {"row_a", "row_b", "dt_s"} per game, where
row_a is the row whose user was on the play and row_b the on-draw row (0-based data-row indices,
as in `replay.iter_games`), dt_s the game_time gap in seconds. It holds NO draft ids.
Privacy: the pair list links two anonymous events; keep it in data/gameplay/ (gitignored), never
commit or publish it, and never write draft_id pairs anywhere (critique.md Q13).

Usage:
  python -m draftzero.gameplay.pairs                       # FDN PremierDraft: write + validate
  python -m draftzero.gameplay.pairs --set FDN --format TradDraft

  from draftzero.gameplay.pairs import load_pairs, partner_map
  pairs = load_pairs()                    # [Pair(row_a, row_b, dt_s), ...]
  other = partner_map(pairs)              # row -> the row of the same game from the other side

17lands public data is CC BY 4.0 (https://www.17lands.com/public_datasets).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from draftzero.gameplay import replay
from draftzero.gameplay.ids import REPO

OUT_DIR = REPO / "data" / "gameplay"
KEY_TURN = 3          # the on-play player's turn whose end-of-turn board keys the match
CHECK_TURN = 5        # the independent validation turn
WINDOW_S = 900        # max game_time gap (s); matches near the edge validate too (264/265), use with care
_KEY_FAMS = ("eot_user_lands_in_play", "eot_user_creatures_in_play",
             "eot_oppo_lands_in_play", "eot_oppo_creatures_in_play")


@dataclass(frozen=True)
class Pair:
    row_a: int        # the row whose user was on the play
    row_b: int        # the on-draw row of the same game
    dt_s: int         # |game_time difference| in seconds


def pairs_path(set_code: str = "FDN", fmt: str = "PremierDraft", out_dir=None) -> Path:
    return Path(out_dir or OUT_DIR) / f"pairs_{set_code}_{fmt}.jsonl"


class _Cols:
    """Column indices of the few fields the matcher reads (from replay.Header)."""

    def __init__(self, H: replay.Header):
        self.n = len(H.cols)
        m = H.meta
        self.time, self.on_play, self.won = m["game_time"], m["on_play"], m["won"]
        self.mull, self.opp_mull, self.num_turns = m["num_mulligans"], m["opp_num_mulligans"], m["num_turns"]

        def turn(side: str, n: int) -> dict[str, int]:
            return {f: i for f, i, _ in H.turn.get((side, n), ())}
        self.key = {s: [turn(s, KEY_TURN)[f] for f in _KEY_FAMS] for s in replay.SIDES}
        u5, o5 = turn("user", CHECK_TURN), turn("oppo", CHECK_TURN)
        # on-play row: its own turn 5 (hand list, casts); on-draw row: the opponent's turn 5 (hand count, casts)
        self.check_play = (u5["eot_user_cards_in_hand"], u5["creatures_cast"])
        self.check_draw = (o5["eot_oppo_cards_in_hand"], o5["creatures_cast"])


def _sorted_ids(s: str) -> str:
    return "|".join(sorted(s.split("|"))) if s else ""


def _split(line: str, C: _Cols) -> list[str]:
    # data rows have no quoted fields in practice (only the header does): a plain split is 5x faster
    p = line.rstrip("\r\n").split(",")
    if len(p) != C.n:
        import csv
        p = next(csv.reader([line]))
    return p


@dataclass
class _Row:
    row: int
    t: float
    won: str
    mull: int
    opp_mull: int
    num_turns: int
    check: tuple        # (turn-5 hand size of the on-play player, its sorted creatures_cast)


def _scan(path, stop: int | None = None) -> tuple[dict, Counter]:
    """One pass: key -> ([on-play rows], [on-draw rows])."""
    H, lines = replay.open_lines(path)
    C = _Cols(H)
    groups: dict[tuple, tuple[list, list]] = defaultdict(lambda: ([], []))
    n = Counter()
    try:
        for i, line in enumerate(lines):
            if stop is not None and i >= stop:
                break
            p = _split(line, C)
            n["rows"] += 1
            on_play = p[C.on_play] == "True"
            k = [_sorted_ids(p[j]) for j in C.key["user" if on_play else "oppo"]]
            # key = (on-play player's lands, creatures, on-draw player's lands, creatures)
            key = tuple(k) if on_play else (k[2], k[3], k[0], k[1])
            if not key[0]:
                n["no_key"] += 1          # the game ended before the on-play player's 3rd land-bearing turn
                continue
            if on_play:
                h = p[C.check_play[0]]
                chk = (len(h.split("|")) if h else 0, _sorted_ids(p[C.check_play[1]]))
            else:
                h = p[C.check_draw[0]]
                chk = (int(float(h)) if h else -1, _sorted_ids(p[C.check_draw[1]]))
            # a fixed zone (UTC): a naive local-time parse jumps an hour at a DST change of the machine's zone
            t = datetime.strptime(p[C.time], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
            r = _Row(i, t, p[C.won], int(float(p[C.mull] or 0)), int(float(p[C.opp_mull] or 0)),
                     int(float(p[C.num_turns] or 0)), chk)
            groups[key][0 if on_play else 1].append(r)
    finally:
        lines.close()
    n["keys"] = len(groups)
    return groups, n


def find_pairs(path=None, window_s: int = WINDOW_S, stop: int | None = None) -> tuple[list[Pair], dict]:
    """Match mirrored rows in a replay file. Returns (pairs sorted by row_a, validation stats)."""
    t0 = time.time()
    groups, n = _scan(path or replay.replay_path(), stop)
    pairs: list[Pair] = []
    amb = Counter()
    val = Counter()

    def fits(a: _Row, b: _Row) -> bool:
        return (abs(a.t - b.t) < window_s and a.won != b.won and a.mull == b.opp_mull
                and a.opp_mull == b.mull and a.num_turns == b.num_turns)
    for plays, draws in groups.values():
        if not plays or not draws:
            continue
        ok = [[j for j, b in enumerate(draws) if fits(a, b)] for a in plays]
        # ambiguity on both sides and before the greedy `used` filter: an on-draw row claimed by two
        # on-play rows would otherwise go to the first and leave the second silently unmatched
        amb["play"] += sum(len(c) > 1 for c in ok)
        amb["draw"] += sum(k > 1 for k in Counter(j for c in ok for j in c).values())
        used: set[int] = set()
        for a, all_c in zip(plays, ok):
            cands = [j for j in all_c if j not in used]
            if not cands:
                continue
            j = min(cands, key=lambda j: abs(a.t - draws[j].t))
            used.add(j)
            b = draws[j]
            dt = int(abs(a.t - b.t))
            pairs.append(Pair(a.row, b.row, dt))
            if a.num_turns >= CHECK_TURN:
                band = "le60" if dt <= 60 else "gt60"
                val[f"{band}_n"] += 1
                val[f"{band}_hand_ok"] += a.check[0] == b.check[0]
                val[f"{band}_cast_ok"] += a.check[1] == b.check[1]
                val[f"{band}_both_ok"] += a.check == b.check
    pairs.sort(key=lambda q: q.row_a)
    dts = sorted(q.dt_s for q in pairs)
    pct = lambda q: dts[min(len(dts) - 1, int(len(dts) * q))] if dts else None
    checked = val["le60_n"] + val["gt60_n"]
    stats = {
        "rows": n["rows"], "rows_without_key": n["no_key"], "keys": n["keys"],
        "pairs": len(pairs), "paired_rows_share": round(2 * len(pairs) / max(n["rows"], 1), 5),
        "ambiguous": amb["play"] + amb["draw"], "ambiguous_play": amb["play"], "ambiguous_draw": amb["draw"],
        "dt_s": {"median": pct(0.5), "p90": pct(0.9), "p99": pct(0.99), "max": dts[-1] if dts else None,
                 "within_10s": sum(d <= 10 for d in dts), "within_60s": sum(d <= 60 for d in dts),
                 "over_60s": sum(d > 60 for d in dts), "over_300s": sum(d > 300 for d in dts)},
        "check_turn": CHECK_TURN, "checked_pairs": checked,
        "hand_size_agree": round((val["le60_hand_ok"] + val["gt60_hand_ok"]) / max(checked, 1), 6),
        "creatures_cast_agree": round((val["le60_cast_ok"] + val["gt60_cast_ok"]) / max(checked, 1), 6),
        "both_agree": round((val["le60_both_ok"] + val["gt60_both_ok"]) / max(checked, 1), 6),
        "by_gap": {k: val[k] for k in sorted(val)},
        "window_s": window_s, "seconds": round(time.time() - t0, 1),
    }
    return pairs, stats


def write_pairs(pairs: list[Pair], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for q in pairs:
            f.write(json.dumps(asdict(q)) + "\n")


def load_pairs(path=None, set_code: str = "FDN", fmt: str = "PremierDraft") -> list[Pair]:
    """Read a pairs_<SET>_<FMT>.jsonl written by `main` (default: data/gameplay/)."""
    with open(path or pairs_path(set_code, fmt)) as f:
        return [Pair(**json.loads(line)) for line in f if line.strip()]


def partner_map(pairs: list[Pair]) -> dict[int, int]:
    """row -> the other row of the same game (both directions)."""
    out = {}
    for q in pairs:
        out[q.row_a] = q.row_b
        out[q.row_b] = q.row_a
    return out


def print_stats(s: dict) -> None:
    d = s["dt_s"]
    print(f"rows {s['rows']:,} ({s['rows_without_key']:,} without a turn-{KEY_TURN} key), keys {s['keys']:,}")
    print(f"mirrored pairs {s['pairs']:,} -> {2 * s['pairs']:,} rows = {100 * s['paired_rows_share']:.2f}% of rows; "
          f"ambiguous rows {s['ambiguous']} (on-play {s['ambiguous_play']}, on-draw {s['ambiguous_draw']})")
    print(f"game_time gap s: median {d['median']} p90 {d['p90']} p99 {d['p99']} max {d['max']}; "
          f"<=10 s {d['within_10s']:,}, <=60 s {d['within_60s']:,}, >60 s {d['over_60s']:,}, >300 s {d['over_300s']:,}")
    print(f"turn-{s['check_turn']} validation on {s['checked_pairs']:,} pairs: hand size {100 * s['hand_size_agree']:.3f}%, "
          f"creatures_cast {100 * s['creatures_cast_agree']:.3f}%, both {100 * s['both_agree']:.3f}%; by gap {s['by_gap']}")
    print(f"({s['seconds']} s)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m draftzero.gameplay.pairs",
                                 description="Match mirrored 17lands replay rows (the same game from both sides).")
    ap.add_argument("--set", default="FDN")
    ap.add_argument("--format", default="PremierDraft")
    ap.add_argument("--path", default=None, help="replay csv.gz (default data/17lands/replay_data_public.<SET>.<FMT>.csv.gz)")
    ap.add_argument("--out", default=None, help="pairs jsonl (default data/gameplay/pairs_<SET>_<FMT>.jsonl)")
    ap.add_argument("--window", type=int, default=WINDOW_S, help="max game_time gap in seconds")
    ap.add_argument("--stop", type=int, default=None, help="read only the first N rows (testing)")
    a = ap.parse_args(argv)
    path = a.path or replay.replay_path(a.set, a.format)
    pairs, stats = find_pairs(path, a.window, a.stop)
    out = Path(a.out or pairs_path(a.set, a.format))
    write_pairs(pairs, out)
    stats["path"] = Path(path).name
    out.with_suffix(".summary.json").write_text(json.dumps(stats, indent=1) + "\n")
    print_stats(stats)
    print(f"wrote {out} (row indices and time gaps only) and {out.with_suffix('.summary.json').name}")
    ok = stats["ambiguous"] == 0 and stats["both_agree"] >= 0.999
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
