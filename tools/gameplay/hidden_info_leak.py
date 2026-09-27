#!/usr/bin/env python
"""
hidden_info_leak.py — does an agent's decision depend on cards it cannot see? (docs/009 §3.4, §7)

Each position comes as a pair of worlds with identical public information, built from two 17lands
FDN decks in assets/sample/ (A: WG deck 02871, B: UR deck 02581). A fair agent decides the same way
in both worlds, up to noise.

Positions:
  counterspell  turn 7, A in its postcombat main with Serra Angel in hand and six untapped lands;
                B has three Islands and two Mountains untapped and two cards in hand:
                  R  Refute + Island     (B can counter the Angel)
                  N  Island + Island     (B cannot)
  cantrip       turn 5, A in its precombat main with Helpful Hunter ({1}{W}, draws a card when it
                enters) and Cathar Commando ({1}{W} 3/1) in hand, and Plains + two Forests untapped
                after its land drop. A's own next card, which it cannot see, is:
                  E  Llanowar Elves      (castable with the Forest left after the Hunter)
                  L  Plains              (not playable this turn)

Deciders:
  mcts  MageZero's ComputerPlayerMCTS2 through the mzbridge `coach` op (fresh tree per seed).
        Conditions: clairvoyant = the world as built (what self-play searches); pimc = B's hand
        re-drawn from its unseen cards for each search (counterspell only: in the cantrip pair the
        hidden card is A's own library top, and a fair search simply doesn't pin it).
        `--evaluator offline` uses GameStateEvaluator3 (hand SIZE only), so a difference between
        worlds comes from the search's transitions; `--evaluator remote` uses a MageZero network
        via the mz-engine inference server (`--port`), with perfectInfo from --perfect-info.
  mad   XMage's MAD AI (ComputerPlayer7.calculateActions: "Computer - mad", and the base of
        MageZero's minimax type) run on the built game by tools/gameplay/MadProbe.java, once per
        build seed. Its search is deterministic apart from tie-breaking.

Usage:
  python tools/gameplay/hidden_info_leak.py                                  # counterspell, offline, K=8 x 1,000
  python tools/gameplay/hidden_info_leak.py --evaluator remote --port 50091 --priors binary \
      --budget 300 --conditions clairvoyant                                  # gen33 as exp #1 ran it
  python tools/gameplay/hidden_info_leak.py --position cantrip --conditions clairvoyant
  python tools/gameplay/hidden_info_leak.py --decider mad --position cantrip --k 6

Results go to data/gameplay/hidden_info_leak/ (gitignored).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from draftzero.gameplay.bridge import RUNTIME_ROOT, Bridge, build_jar  # noqa: E402

OUT_DIR = REPO / "data/gameplay/hidden_info_leak"
DECK_A = REPO / "assets/sample/decks/FDN_top_02871_WG.dck"
DECK_B = REPO / "assets/sample/decks/FDN_top_02581_UR.dck"
PROBE = Path(__file__).with_name("MadProbe.java")


def decklist(path: Path) -> list[str]:
    cards = []
    for line in path.read_text().splitlines():
        m = re.match(r"^(\d+) \[[A-Z0-9]+:\d+\] (.+)$", line.strip())
        if m:
            cards += [m.group(2)] * int(m.group(1))
    if len(cards) != 40:
        raise SystemExit(f"{path}: expected 40 cards, found {len(cards)}")
    return cards


def _players(a: dict, b: dict) -> dict:
    return {
        "A": {"name": "PlayerA", "decklist": decklist(DECK_A), "decklistSource": "exact", **a},
        "B": {"name": "PlayerB", "decklist": decklist(DECK_B), "decklistSource": "exact", **b},
    }


def counterspell_spec(b_hand: list[str]) -> dict:
    """Turn 7, A (on the play) in its postcombat main; public information identical across worlds."""
    return {
        "version": 1,
        "comment": "docs/009 counterspell leak test: A holds Serra Angel; B has 1UU open.",
        "turn": 7, "activePlayer": "A", "phase": "POSTCOMBAT_MAIN", "step": "POSTCOMBAT_MAIN",
        "enterMode": "PRIORITY_FRESH",
        "players": _players(
            {"life": 16, "landsPlayed": 1, "hand": ["Serra Angel"],
             "graveyard": ["Llanowar Elves", "Ambush Wolf", "Helpful Hunter"],
             "battlefield": [{"name": "Plains", "count": 3}, {"name": "Forest", "count": 3}]},
            {"life": 16, "landsPlayed": 0, "hand": b_hand,
             "graveyard": ["Fiery Annihilation", "Fleeting Distraction", "Bigfin Bouncer"],
             "battlefield": [{"name": "Island", "count": 3}, {"name": "Mountain", "count": 2}]}),
        "provenance": {"source": "synthetic", "ref": "tools/gameplay/hidden_info_leak.py", "tier": "T0"},
        "labels": {"decisionPlayer": "A"},
    }


def cantrip_spec(a_top: str) -> dict:
    """Turn 5, A (on the play) in its precombat main; only A's own (unseen) library top differs."""
    return {
        "version": 1,
        "comment": "docs/009 cantrip leak test: A holds Helpful Hunter + Cathar Commando, Plains + 2 Forests open.",
        "turn": 5, "activePlayer": "A", "phase": "PRECOMBAT_MAIN", "step": "PRECOMBAT_MAIN",
        "enterMode": "PRIORITY_FRESH",
        "players": _players(
            {"life": 18, "landsPlayed": 1, "hand": ["Helpful Hunter", "Cathar Commando"], "libraryTop": [a_top],
             "graveyard": ["Ambush Wolf", "Beast-Kin Ranger", "Prideful Parent", "Treetop Snarespinner"],
             "battlefield": [{"name": "Plains", "count": 1}, {"name": "Forest", "count": 2}]},
            {"life": 18, "landsPlayed": 0, "hand": ["Island", "Mountain", "Erudite Wizard"],
             "graveyard": ["Fiery Annihilation", "Bigfin Bouncer"],
             "battlefield": [{"name": "Island", "count": 2}, {"name": "Mountain", "count": 2}]}),
        "provenance": {"source": "synthetic", "ref": "tools/gameplay/hidden_info_leak.py", "tier": "T0"},
        "labels": {"decisionPlayer": "A"},
    }


POSITIONS = {
    "counterspell": {"spec": counterspell_spec,
                     "worlds": {"R": ["Refute", "Island"], "N": ["Island", "Island"]}},
    "cantrip": {"spec": cantrip_spec,
                "worlds": {"E": "Llanowar Elves", "L": "Plains"}},
}


def summarize(resp: dict) -> dict:
    agg = {a["label"]: {k: a.get(k) for k in ("meanQ", "sdQ", "visitShare", "nDet", "rank")}
           for a in resp["aggregate"]}
    return {
        "best": resp["best"],
        "aggregate": agg,
        "perDeterminizationBest": dict(Counter(d.get("best") for d in resp["determinizations"])),
        "perDeterminization": [{"best": d.get("best"),
                                "children": {c["label"]: {"N": c.get("N"), "Q": c.get("Q")} for c in d.get("children", [])},
                                "hand": d.get("sampledHands") or d.get("hands")}
                               for d in resp["determinizations"]],
        "consistent": resp.get("consistent"),
        "simsPerSec": resp.get("timing_ms", {}).get("simsPerSec"),
    }


def show(label: str, cond: str, sm: dict) -> None:
    print(f"\nworld {label} / {cond}: best = {sm['best']}   per-determinization best = {sm['perDeterminizationBest']}")
    for name, a in sorted(sm["aggregate"].items(), key=lambda kv: kv[1]["rank"]):
        sd = "-" if a["sdQ"] is None else f"{a['sdQ']:.3f}"
        q = "-" if a["meanQ"] is None else f"{a['meanQ']:+.3f}"
        print(f"   {name:24s} meanQ {q}  sdQ {sd}  visitShare {a['visitShare']:.2f}  nDet {a['nDet']}")


def run_mcts(a, pos: dict, conditions: list[str]) -> dict:
    opts: dict = {"budget": a.budget, "seed": a.seed, "decisionPlayer": "A", "timeoutSec": 600}
    if a.evaluator == "remote":
        opts["evaluator"] = {"type": "remote", "host": "127.0.0.1", "port": a.port}
        opts["perfectInfo"] = a.perfect_info
        heads = {h for h in a.priors.split(",") if h}
        opts["priors"] = {h: h in heads for h in ("priority", "target", "binary", "opponent")}
    worlds: dict = {}
    with Bridge("hidden_info_leak") as b:
        for w, arg in pos["worlds"].items():
            d = b.build(pos["spec"](arg), decisionPlayer="A")["decision"]
            print(f"[build] world {w}: {d['type']} {[x['label'] for x in d['legal']]}")
        for w, arg in pos["worlds"].items():
            s = pos["spec"](arg)
            worlds[w] = {}
            for cond in conditions:
                if cond == "clairvoyant":
                    r = b.coach_specs([s] * a.k, **opts)
                elif cond == "pimc":
                    if a.position != "counterspell":
                        raise SystemExit("pimc re-draws B's hand; it only applies to the counterspell position")
                    r = b.coach(s, determinizations=2 * a.k, resample=["B"], **opts)
                else:
                    raise SystemExit(f"unknown condition {cond}")
                worlds[w][cond] = summarize(r)
                show(f"{w} ({arg})", cond, worlds[w][cond])
    return worlds


def _runtime_db(name: str) -> Path:
    """A private working directory with its own copy of the card database (H2 opens ./db)."""
    rt = RUNTIME_ROOT / name
    (rt / "db").mkdir(parents=True, exist_ok=True)
    src, dst = REPO / "xmage/db/cards.h2.mv.db", rt / "db/cards.h2.mv.db"
    if not dst.exists() or dst.stat().st_size < src.stat().st_size or src.stat().st_mtime > dst.stat().st_mtime:
        dst.unlink(missing_ok=True)
        if subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True).returncode != 0:
            shutil.copyfile(src, dst)  # not APFS: a plain copy
    return rt


def run_mad(a, pos: dict) -> dict:
    jar = build_jar()
    rt = _runtime_db("hidden_info_leak_mad")
    final = ["--enable-final-field-mutation=ALL-UNNAMED"]
    if subprocess.run(["java", *final, "-version"], capture_output=True).returncode != 0:
        final = []  # older JDKs: the flag is unknown and not needed
    worlds: dict = {}
    with tempfile.TemporaryDirectory() as td:
        paths = {}
        for w, arg in pos["worlds"].items():
            p = Path(td) / f"{a.position}_{w}.json"
            p.write_text(json.dumps(pos["spec"](arg)))
            paths[p.name] = w
        cmd = ["java", *final, "-Xmx3g", "-cp", f"{jar}:{REPO / 'xmage/lib'}/*",
               f"-Dlog4j.configuration=file:{REPO / 'java/mzbridge/log4j.properties'}", "-Dmzb.logLevel=warn",
               f"-Dmz.actionVocab={REPO / 'assets/vocab/FDN_SPG.tsv'}",
               str(PROBE), str(a.k), str(a.skill), *[str(Path(td) / n) for n in paths]]
        r = subprocess.run(cmd, cwd=rt, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise SystemExit(f"MadProbe failed ({r.returncode}):\n{r.stderr[-3000:]}")
        rows = [json.loads(line) for line in r.stdout.splitlines() if line.startswith("{")]
    for w in pos["worlds"]:
        mine = [x for x in rows if paths[x["spec"]] == w]
        chains = Counter(" -> ".join(x["chain"]) or "(pass)" for x in mine)
        worlds[w] = {"chains": dict(chains), "rows": mine}
        print(f"\nworld {w} ({pos['worlds'][w]}) / mad: {dict(chains)}  scores {sorted({x['score'] for x in mine})}")
    return worlds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--position", choices=sorted(POSITIONS), default="counterspell")
    ap.add_argument("--decider", choices=("mcts", "mad"), default="mcts")
    ap.add_argument("--k", type=int, default=8, help="seeds per world (mcts pimc uses 2K)")
    ap.add_argument("--budget", type=int, default=1000, help="mcts: new simulations per determinization")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--evaluator", choices=("offline", "remote"), default="offline")
    ap.add_argument("--port", type=int, default=50091, help="mz-engine inference server (remote)")
    ap.add_argument("--priors", default="", help="comma list of priority,target,binary,opponent (remote)")
    ap.add_argument("--perfect-info", action=argparse.BooleanOptionalAction, default=True,
                    help="encoder perfectInfo for remote search (gen33 was trained with it on)")
    ap.add_argument("--conditions", default="clairvoyant,pimc")
    ap.add_argument("--skill", type=int, default=6, help="mad: AI skill (search depth); MageZero's minimax uses 6")
    ap.add_argument("--tag", default="", help="suffix for the results file")
    a = ap.parse_args()

    pos = POSITIONS[a.position]
    conditions = [c for c in a.conditions.split(",") if c]
    t0 = time.time()
    if a.decider == "mad":
        worlds = run_mad(a, pos)
        name = f"{a.position}_mad_skill{a.skill}_k{a.k}{a.tag}.json"
    else:
        worlds = run_mcts(a, pos, conditions)
        pi = "_pi" if a.evaluator == "remote" and a.perfect_info else ""
        prefix = "" if a.position == "counterspell" else f"{a.position}_"   # keeps the first runs' names
        name = f"{prefix}{a.evaluator}{pi}_k{a.k}_b{a.budget}_s{a.seed}{a.tag}.json"
    results = {"settings": {**vars(a), "worlds": pos["worlds"]}, "worlds": worlds,
               "seconds": round(time.time() - t0, 1)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT_DIR / name} ({results['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
