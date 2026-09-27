"""
17lands replay data -> StateSpecs and labels: the command-line front end of the gameplay package.

  fetch   download cards.csv, abilities.csv and the replay (and game) data from the 17lands S3
          bucket into data/17lands (skips files already there)
            python -m draftzero.gameplay.seventeenlands fetch --set FDN --format PremierDraft
  stats   one pass (full or sampled) printing sizes and the key rates of docs/008
          (replay_empirics.md §4.2 conservation, §8 flags and fidelity ladder), the StateSpec tier
          distribution, card/ability id coverage and throughput. --kinds adds the specs after each
          user turn (after: its END_TURN; blocks: the opponent's DECLARE_ATTACKERS when it attacked)
            python -m draftzero.gameplay.seventeenlands stats --every 40            # ~20k games
            python -m draftzero.gameplay.seventeenlands stats --every 40 --kinds turn after blocks
            python -m draftzero.gameplay.seventeenlands stats --every 16 --json data/gameplay/stats_every16.json
  specs   write StateSpec JSON (optionally with labels) for selected rows and user turns
            python -m draftzero.gameplay.seventeenlands specs --rows 0 4 --labels
            python -m draftzero.gameplay.seventeenlands specs --every 1000 --entry main1 \\
                --out data/gameplay/specs/FDN_PremierDraft_main1
            python -m draftzero.gameplay.seventeenlands specs --rows 4 --after --entry declare_attackers
            python -m draftzero.gameplay.seventeenlands specs --every 5000 --exact   # mirrored rows only

Specs are named <SET>_<FMT>_r<row>_u<turn>_<entry>.json (..._after_<entry> for the state after the
turn, ..._exact with the mirrored row's deck and hand) and carry no 17lands user or draft ids.
Every spec's labels["bridge"] holds the bridge request options that reach the decision its labels
describe: pass them with every bridge request (`bridge.build(spec, **spec.labels["bridge"])`).
17lands public data is CC BY 4.0 (https://www.17lands.com/public_datasets): credit 17lands in
anything built from it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from draftzero.gameplay import ids as idmod
from draftzero.gameplay import reconstruct as rc
from draftzero.gameplay import replay
from draftzero.gameplay.ids import Ids
from draftzero.gameplay.statespec import TIERS, dump

OUT_DIR = idmod.REPO / "data" / "gameplay"
# columns whose values are card ids (the rest of the per-turn fields are counts or ability ids)
_SNAPSHOT_PREFIX = "eot_"


class Stats:
    """Accumulates the research rates over a stream of games."""

    def __init__(self, ids: Ids, build_specs: bool = True, kinds=("turn",)):
        self.ids = ids
        self.build_specs = build_specs
        self.kinds = tuple(kinds)                 # turn | after | blocks (see _kind_spec)
        self.extra = {k: {"n": Counter(), "tier": Counter(), "flags": Counter(), "invalid": Counter()}
                      for k in self.kinds if k != "turn"}
        self.n = Counter()
        self.zone = defaultdict(Counter)          # (level, zone) -> {True, False}
        self.slot = defaultdict(Counter)          # (level, scope) -> {True, False}
        self.flag = Counter()
        self.ladder = Counter()
        self.tier = Counter()
        self.spec_flags = Counter()
        self.invalid = Counter()
        self.cov = Counter()
        self.t_parse = self.t_analyze = self.t_specs = 0.0

    def add(self, g: replay.Game) -> None:
        ids = self.ids
        self.n["games"] += 1
        self.n["slots"] += len(g.turns)
        t0 = time.perf_counter()
        ana = rc.analyze(g, ids)
        self.t_analyze += time.perf_counter() - t0
        for chk in ana.checks:
            for level, d in (("strict", chk.strict), ("benign", chk.benign), ("nounexp", chk.nounexp)):
                for z, ok in d.items():
                    self.zone[(level, z)][ok] += 1
                for scope in ("user_zones", "all_zones"):
                    self.slot[(level, scope)][chk.slot_ok(level, scope)] += 1
        for n, fl in ana.flags.items():
            self.n["states"] += 1
            for k, v in fl.items():
                self.flag[k] += bool(v)
            for rung, ok in rc.re_ladder(fl).items():
                self.ladder[rung] += ok
        # id coverage per occurrence: every card id in an event column / a snapshot column
        for t in g.turns:
            for f, v in t.f.items():
                if not isinstance(v, list) or not v:
                    continue
                if f.endswith("_abilities"):
                    for a in v:
                        self.cov["ability"] += 1
                        self.cov["ability_known"] += a in ids.ability_rows
                    continue
                kind = "snapshot" if f.startswith(_SNAPSHOT_PREFIX) else "event"
                for c in v:
                    self.cov[kind] += 1
                    self.cov[kind + "_mapped"] += ids.is_mapped(c)
        if self.build_specs and "turn" in self.kinds:
            t0 = time.perf_counter()
            for n in g.decision_turns():
                spec = rc.state_at_user_turn(g, n, ids=ids)
                self.n["specs"] += 1
                self.tier[spec.provenance.tier] += 1
                for f in spec.provenance.flags:
                    self.spec_flags[f] += 1
                errs = spec.validate()
                # a partial spec (hidden opponent hand) is expected; anything else is a bug
                for e in errs:
                    self.invalid[e.split(":")[0] if ":" in e else e[:40]] += 1
                self.n["invalid_specs"] += bool(errs)
            self.t_specs += time.perf_counter() - t0
        for kind, acc in self.extra.items():
            for n in g.decision_turns():
                try:
                    spec = _kind_spec(g, n, kind, ids)
                except ValueError:                # no opponent turn after it, or no attack
                    acc["n"]["skipped"] += 1
                    continue
                acc["n"]["specs"] += 1
                acc["tier"][spec.provenance.tier] += 1
                acc["flags"].update(spec.provenance.flags)
                errs = spec.validate()
                for e in errs:
                    acc["invalid"][e.split(":")[0] if ":" in e else e[:40]] += 1
                acc["n"]["invalid_specs"] += bool(errs)

    def rates(self) -> dict:
        pct = lambda c: round(100.0 * c[True] / max(1, c[True] + c[False]), 2)
        st = max(1, self.n["states"])
        out = {
            "counts": dict(self.n),
            "conservation_zone": {f"{lvl}:{z}": pct(c) for (lvl, z), c in sorted(self.zone.items())},
            "conservation_slot": {f"{lvl}:{sc}": pct(c) for (lvl, sc), c in sorted(self.slot.items())},
            "re_flags": {k: round(100.0 * v / st, 2) for k, v in sorted(self.flag.items())},
            "re_ladder": {k: round(100.0 * self.ladder[k] / st, 2) for k in rc.RE_LADDER},
            "coverage": {
                "event_card_ids_mapped": round(100.0 * self.cov["event_mapped"] / max(1, self.cov["event"]), 4),
                "snapshot_card_ids_mapped": round(100.0 * self.cov["snapshot_mapped"] / max(1, self.cov["snapshot"]), 4),
                "ability_ids_in_table": round(100.0 * self.cov["ability_known"] / max(1, self.cov["ability"]), 4),
                "occurrences": dict(self.cov)},
        }
        if self.build_specs and "turn" in self.kinds:
            ns = max(1, self.n["specs"])
            out["spec_tiers"] = {t: round(100.0 * self.tier[t] / ns, 2) for t in TIERS}
            out["spec_flags"] = {k: round(100.0 * v / ns, 2) for k, v in self.spec_flags.most_common()}
            out["spec_invalid"] = dict(self.invalid)
        for kind, acc in self.extra.items():
            ns = max(1, acc["n"]["specs"])
            out.setdefault("kinds", {})[kind] = {
                "counts": dict(acc["n"]), "spec_tiers": {t: round(100.0 * acc["tier"][t] / ns, 2) for t in TIERS},
                "spec_flags": {k: round(100.0 * v / ns, 2) for k, v in acc["flags"].most_common()},
                "spec_invalid": dict(acc["invalid"])}
        out["seconds"] = {"parse": round(self.t_parse, 3), "analyze": round(self.t_analyze, 3),
                          "specs": round(self.t_specs, 3)}
        return out


def _kind_spec(g: replay.Game, n: int, kind: str, ids: Ids, **kw):
    """turn: the start of user turn n (eot_rollover); after: the END_TURN of user turn n; blocks: the
    opponent's DECLARE_ATTACKERS after it (ValueError when there is none)."""
    if kind == "turn":
        return rc.state_at_user_turn(g, n, ids=ids, **kw)
    return rc.state_after_user_turn(g, n, "eot_rollover" if kind == "after" else "declare_attackers", ids=ids, **kw)


def collect(games, ids: Ids, build_specs: bool = True, kinds=("turn",)) -> Stats:
    st = Stats(ids, build_specs, kinds)
    it = iter(games)
    while True:
        t0 = time.perf_counter()
        g = next(it, None)
        st.t_parse += time.perf_counter() - t0
        if g is None:
            return st
        st.add(g)


def _print_stats(r: dict, wall: float) -> None:
    c = r["counts"]
    print(f"games {c.get('games', 0):,}  slots {c.get('slots', 0):,}  user-turn states {c.get('states', 0):,}"
          f"  specs {c.get('specs', 0):,}  wall {wall:.1f} s")
    sec = r["seconds"]
    if c.get("games"):
        # a spec needs its game's analysis (conservation + history walk, cached per game): the
        # end-to-end reconstruction rate counts both
        print(f"throughput: parse+read {c['games'] / max(sec['parse'], 1e-9):,.0f} games/s (incl. skipped-row "
              f"decompression), analyze {c['games'] / max(sec['analyze'], 1e-9):,.0f} games/s"
              + (f", specs {c.get('specs', 0) / max(sec['specs'] + sec['analyze'], 1e-9):,.0f} specs/s end to end "
                 f"({c.get('specs', 0) / max(sec['specs'], 1e-9):,.0f}/s once a game is analysed)" if sec["specs"] else ""))
    print("\nconservation per zone (strict / benign-only / no-unexplained), % of slots [RE §4.2]")
    zones = sorted({k.split(":", 1)[1] for k in r["conservation_zone"]})
    for z in zones:
        vals = [r["conservation_zone"].get(f"{lvl}:{z}") for lvl in ("strict", "benign", "nounexp")]
        print(f"  {z:12s} " + "  ".join("   -  " if v is None else f"{v:6.2f}" for v in vals))
    print("slot level: " + ", ".join(f"{k} {v}" for k, v in r["conservation_slot"].items()))
    print("\nRE §8 cumulative ladder (% of user-turn states determined): "
          + ", ".join(f"{k} {v}" for k, v in r["re_ladder"].items()))
    print("RE §8 flags (% of states): " + ", ".join(f"{k} {v}" for k, v in r["re_flags"].items()))
    if "spec_tiers" in r:
        print("\nStateSpec tiers (% of decision specs, T0 best): "
              + ", ".join(f"{k} {v}" for k, v in r["spec_tiers"].items()))
        print("spec flags: " + ", ".join(f"{k} {v}" for k, v in list(r["spec_flags"].items())[:30]))
        print(f"invalid specs: {c.get('invalid_specs', 0)} {r['spec_invalid'] or ''}")
    for kind, k in r.get("kinds", {}).items():
        print(f"\n{kind} specs: {k['counts']}")
        print("  tiers: " + ", ".join(f"{t} {v}" for t, v in k["spec_tiers"].items()))
        print("  flags: " + ", ".join(f"{f} {v}" for f, v in list(k["spec_flags"].items())[:30]))
        print(f"  invalid: {k['counts'].get('invalid_specs', 0)} {k['spec_invalid'] or ''}")
    print("\nid coverage: " + ", ".join(f"{k} {v}%" for k, v in r["coverage"].items() if k != "occurrences"))


def _turns(g: replay.Game, sel: str) -> list[int]:
    if sel == "decision":
        return g.decision_turns()
    if sel == "all":
        return g.user_turn_numbers()
    want = {int(x) for x in sel.split(",") if x}
    return [n for n in g.user_turn_numbers() if n in want]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m draftzero.gameplay.seventeenlands",
                                 description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download 17lands files into data/17lands")
    f.add_argument("--set", default="FDN")
    f.add_argument("--format", default="PremierDraft")
    f.add_argument("--what", nargs="+", default=["cards", "abilities", "replay", "game"], choices=sorted(idmod.URLS))
    f.add_argument("--force", action="store_true")
    for name in ("stats", "specs"):
        p = sub.add_parser(name)
        p.add_argument("--set", default="FDN")
        p.add_argument("--format", default="PremierDraft")
        p.add_argument("--path", default=None, help="replay csv.gz (default data/17lands/replay_data_public.<SET>.<FMT>.csv.gz)")
        p.add_argument("--every", type=int, default=1)
        p.add_argument("--start", type=int, default=0)
        p.add_argument("--limit", type=int, default=None)
        if name == "stats":
            p.add_argument("--no-specs", action="store_true", help="skip building a spec for every decision turn")
            p.add_argument("--kinds", nargs="+", default=["turn"], choices=["turn", "after", "blocks"],
                           help="which specs to build and grade per decision turn (default: turn)")
            p.add_argument("--json", default=None, help="also write the rates here (default: nothing)")
        else:
            p.add_argument("--rows", type=int, nargs="*", default=None, help="data-row indices (0-based)")
            p.add_argument("--turns", default="decision", help="decision (default) | all | comma list, e.g. 3,5")
            p.add_argument("--entry", choices=sorted(set(rc.ENTRIES) | set(rc.AFTER_ENTRIES)), default="eot_rollover",
                           help=f"turn start: {', '.join(rc.ENTRIES)}; with --after: {', '.join(rc.AFTER_ENTRIES)}")
            p.add_argument("--after", action="store_true",
                           help="the state after each user turn (its blocks and off-turn plays) instead of its start")
            p.add_argument("--exact", action="store_true",
                           help="mirrored rows only (data/gameplay/pairs file): the opponent's real deck and hand")
            p.add_argument("--labels", action="store_true", help="put the turn label in spec.labels")
            p.add_argument("--out", default=None, help="output dir (default data/gameplay/specs/<SET>_<FMT>)")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        for p in idmod.fetch(a.what, a.set, a.format, force=a.force):
            print(p)
        return 0
    ids = Ids.load(a.set)
    path = a.path or replay.replay_path(a.set, a.format)
    if a.cmd == "stats":
        t0 = time.time()
        st = collect(replay.iter_games(path, a.limit, a.every, a.start), ids, build_specs=not a.no_specs,
                     kinds=a.kinds)
        r = st.rates()
        r["sample"] = {"path": str(path), "every": a.every, "start": a.start, "limit": a.limit}
        _print_stats(r, time.time() - t0)
        if a.json:
            Path(a.json).parent.mkdir(parents=True, exist_ok=True)
            Path(a.json).write_text(json.dumps(r, indent=1) + "\n")
        return 0
    entries = rc.AFTER_ENTRIES if a.after else rc.ENTRIES
    if a.entry not in entries:
        ap.error(f"--entry {a.entry} needs {'no ' if a.after else ''}--after (choices: {', '.join(entries)})")
    out = Path(a.out or OUT_DIR / "specs" / f"{a.set}_{a.format}")
    out.mkdir(parents=True, exist_ok=True)
    games = (replay.read_games(a.rows, path).values() if a.rows is not None
             else replay.iter_games(path, a.limit, a.every, a.start))
    partners = {}
    if a.exact:                                  # the partner rows need a second read: hold the games
        from draftzero.gameplay.pairs import load_pairs, partner_map
        pm = partner_map(load_pairs(set_code=a.set, fmt=a.format))
        games = [g for g in games if g.row_index in pm]    # (streamed otherwise)
        other = replay.read_games([pm[g.row_index] for g in games], path)
        partners = {g.row_index: other[pm[g.row_index]] for g in games}
    t0 = time.time()
    n = bad = skipped = 0
    tiers = Counter()
    for g in games:
        for turn in _turns(g, a.turns):
            kw = dict(ids=ids, labels=a.labels, partner=partners.get(g.row_index))
            try:
                spec = (rc.state_after_user_turn(g, turn, a.entry, **kw) if a.after
                        else rc.state_at_user_turn(g, turn, a.entry, **kw))
            except ValueError:
                skipped += 1                      # no opponent turn after it, or it did not attack
                continue
            name = f"{a.set}_{a.format}_r{g.row_index}_u{turn}_{'after_' if a.after else ''}{a.entry}"
            dump(spec, out / f"{name}{'_exact' if a.exact else ''}.json")
            errs = spec.validate()
            if errs:
                bad += 1
                print(f"row {g.row_index} turn {turn}: {errs}", file=sys.stderr)
            n += 1
            tiers[spec.provenance.tier] += 1
    print(f"wrote {n} specs to {out} in {time.time() - t0:.1f} s; tiers {dict(sorted(tiers.items()))}; "
          f"{bad} failed validate(); {skipped} turns without such a state")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
