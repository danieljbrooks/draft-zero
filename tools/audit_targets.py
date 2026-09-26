"""List every bad target in a set of JVM logs, one row per case, for qualitative auditing.

    python tools/audit_targets.py runs/<run>/jvm_*.log --deck-root <dir with .dck files> \\
        --out runs/<run>/bad_targets.jsonl

A target is "bad" when a harmful effect hits the actor's own card, or a beneficial effect
hits the opponent's. Effects are classified with MageZero's own rules
(magezero.metrics.classify_effect), so rates stay comparable with experiment #1's metric.

Why this exists: in experiment #1 that metric was effectively dead. Zero targets were
classified in generations 0-32, for two reasons:
  1. Ownership. The parser decides whose card a target is from the two deck files, but with a
     different deck pair every game, draftzero.loop passed /dev/null for both, so every card
     target's owner was unknown and skipped.
  2. Oracle text. "Cast <card>" effects were classified from experiments/fdn/card_oracle.json,
     which held 79 cards and wasn't on the pods at all.
Here each game's targets are attributed on the game's own thread (threads play their games in
sequence, and a game's GAME_SUMMARY is logged on that thread), ownership comes from that
game's two decks, and oracle text from tools/fetch_oracle.py (the whole set).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from magezero.metrics import LINE_RE, NON_TARGET_PROMPT_RE, classify_effect

SUMMARY_TAG = "GAME_SUMMARY "
SIM_RE = re.compile(r"^Player: (Player[AB]) simulated ")
DCK_RE = re.compile(r"^\d+ \[[^\]]+\] (.+)$")


def deck_cards(stem: str, root: Path, cache: dict) -> set[str]:
    if stem not in cache:
        p = root / f"{stem}.dck"
        cache[stem] = {m.group(1).strip() for l in open(p) if (m := DCK_RE.match(l.strip()))} if p.exists() else set()
    return cache[stem]


def audit(logs: list[Path], deck_root: Path, oracle: dict) -> tuple[list[dict], Counter]:
    decks: dict = {}
    rows, tally = [], Counter()
    for log in logs:
        pending = defaultdict(list)          # thread -> targets of the game in progress
        actor, src = {}, {}
        for raw in open(log, errors="replace"):
            m = LINE_RE.match(raw.rstrip("\n"))
            if not m:
                continue
            ts, msg, thread = m.group(1), m.group(2), m.group(3)
            if (sm := SIM_RE.match(msg)):
                actor[thread] = sm.group(1)
            elif msg.startswith("base choose target "):
                src[thread] = msg[len("base choose target "):]
            elif msg.startswith("Targeting ") and src.get(thread):
                s, src[thread] = src[thread], None           # one source per choice
                t = msg[len("Targeting "):]
                if t != "Stop Choosing" and not NON_TARGET_PROMPT_RE.search(s):
                    pending[thread].append({"ts": ts, "actor": actor.get(thread), "src": s, "target": t})
            elif msg.startswith(SUMMARY_TAG):
                try:
                    g = json.loads(msg[len(SUMMARY_TAG):])
                except json.JSONDecodeError:
                    pending[thread].clear()
                    continue
                cards = {"PlayerA": deck_cards(g["deck_a"], deck_root, decks),
                         "PlayerB": deck_cards(g["deck_b"], deck_root, decks)}

                def owner(name: str):
                    if name in cards:
                        return name
                    a, b = name in cards["PlayerA"], name in cards["PlayerB"]
                    return "PlayerA" if a and not b else "PlayerB" if b and not a else None

                for tg in pending.pop(thread, []):
                    tally["targets"] += 1
                    who = tg["actor"]
                    if tg["src"].startswith("Cast "):
                        card = tg["src"][5:]
                        who = owner(card) or who                     # the caster owns the spell
                        effect = classify_effect((oracle.get(card) or {}).get("oracle_text", ""))
                    else:
                        effect = classify_effect(tg["src"])
                    tgt_owner = owner(tg["target"])
                    if effect is None:
                        tally["unclassified_effect"] += 1
                        continue
                    if who is None or tgt_owner is None:
                        tally["unknown_owner"] += 1                   # tokens, or a card in both decks
                        continue
                    tally["classified"] += 1
                    own = tgt_owner == who
                    if (effect == "harmful" and own) or (effect == "beneficial" and not own):
                        tally["bad"] += 1
                        rows.append({"log": log.name, "game": g.get("game"), "seed": g.get("seed"),
                                     "ts": tg["ts"], "actor": who,
                                     "actor_deck": g["deck_a"] if who == "PlayerA" else g["deck_b"],
                                     "effect": effect, "source": tg["src"][:120], "target": tg["target"],
                                     "target_owner": tgt_owner, "winner": g.get("winner")})
    return rows, tally


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--deck-root", required=True)
    ap.add_argument("--oracle", default="data/scryfall/FDN_oracle.json")
    ap.add_argument("--out", default=None, help="write one JSON row per bad target")
    a = ap.parse_args()
    oracle = json.loads(Path(a.oracle).read_text()) if Path(a.oracle).exists() else {}
    rows, t = audit([Path(p) for p in a.logs], Path(a.deck_root), oracle)
    if a.out:
        with open(a.out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    rate = f"{t['bad'] / t['classified']:.1%}" if t["classified"] else "n/a"
    print(f"targets {t['targets']}: classified {t['classified']}, bad {t['bad']} ({rate} of classified), "
          f"effect unclassifiable {t['unclassified_effect']}, owner unknown {t['unknown_owner']}")
    for r in rows[:10]:
        print(f"  [{r['actor']} {r['actor_deck']}] {r['effect']} '{r['source'][:60]}' -> {r['target']} ({r['target_owner']})")


if __name__ == "__main__":
    main()
