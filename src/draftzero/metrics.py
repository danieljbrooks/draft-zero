"""
metrics.py — health / progress metrics for self-play runs.

Moved here from the MageZero fork (danieljbrooks/MageZero mz-engine, bcc76de) when DraftZero
moved to Will's v0.2 engine: it is DraftZero's own analysis and never went upstream.

Four groups (see experiments/fdn/README.md):
  environment health  games completed/failed/timed out, game length, MCTS sims/sec, search timeouts
  behavior health     spells cast, lands, pass-with-play, missed land drops, idle turns, bad targets
  learning            written by train.py / test.py into metrics.jsonl (losses, entropy, agreement)
  strength            eval matches vs frozen baselines, per matchup, with Wilson intervals

The XMage fork logs every decision at INFO. Game threads interleave, so the parser keeps
one state machine per worker thread. Behaviour heuristics are deliberately simple and
documented next to each counter; treat them as trend lines, not ground truth.
"""
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

LINE_RE = re.compile(r"^\w+\s+(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) (.*?)\s*=>\[([^\]]+)\] (\S+)\s*$")
TURN_RE = re.compile(r"^\[(\d+):[^:]*:([A-Z_]+)\]")
CHOSE_RE = re.compile(r"^\[(\d+):[^:]*:([A-Z_]+)\]chose action:(.*?) success ratio: (\S+)")
LIFE_RE = re.compile(r"^\[(\d+):[^\]]*\]\[player PlayerA:(-?\d+)\]\[player PlayerB:(-?\d+)\]")
ACTOR_RE = re.compile(r"^\[(PlayerA|PlayerB)\], life = (-?\d+)")
SIM_RE = re.compile(r"^Player: (PlayerA|PlayerB) simulated (\d+) evaluations in ([\d.Ee-]+) seconds")
PURGE_RE = re.compile(r"^(\d+) illegals purged")
USE_ATTACK_RE = re.compile(r"^use attack with: (.*)\?: (true|false)$")
DCK_RE = re.compile(r"^(\d+) \[[^\]]+\] (.+)$")

MAIN_STEPS = {"PRECOMBAT_MAIN", "POSTCOMBAT_MAIN"}
ORACLE_PATH = Path("experiments/fdn/card_oracle.json")


# ── card knowledge ───────────────────────────────────────────

def load_deck_names(dck_path: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in Path(dck_path).read_text().splitlines():
        m = DCK_RE.match(line.strip())
        if m:
            counts[m.group(2)] = counts.get(m.group(2), 0) + int(m.group(1))
    return counts


def load_oracle() -> dict:
    return json.loads(ORACLE_PATH.read_text()) if ORACLE_PATH.exists() else {}


HARMFUL_RE = re.compile(r"deals? [^.]*damage to (?:target|any target)|destroy target|exile target (?:creature|nonland|artifact|enchantment)"
                        r"|target creature gets -|gets -\d|tap target|return target (?:creature|nonland permanent) to its owner's hand"
                        r"|gain control of target|enchanted creature (?:loses|can't|gets -|is a)", re.I)
BENEFICIAL_RE = re.compile(r"target creature (?:you control )?gets \+|\+1/\+1 counter on target|gains? (?:hexproof|indestructible|flying)"
                           r"|target creature gains", re.I)
RESTRICTED_RE = re.compile(r"target (?:creature|permanent|nonland permanent)s? (?:you control|an opponent controls|you don't control)"
                           r"|target opponent|target player", re.I)


# prompts that reuse the "choose target" log line but are not spell/ability targeting
NON_TARGET_PROMPT_RE = re.compile(r"choose cards? to discard|discard to hand size|choose which creature to block"
                                  r"|sacrifice|choose a card to|put .* on the bottom|scry|surveil", re.I)
# effects whose own cost makes you pick your own permanent (the pick is a cost, not a target)
SELF_COST_RE = re.compile(r"as an additional cost[^.]*sacrifice", re.I)


def classify_effect(text: str) -> Optional[str]:
    """'harmful' / 'beneficial' / None for single-target effects the rules don't already restrict."""
    if not text or text.lower().count("target") != 1 or RESTRICTED_RE.search(text):
        return None
    if SELF_COST_RE.search(text):
        return None
    if HARMFUL_RE.search(text):
        return "harmful"
    if BENEFICIAL_RE.search(text):
        return "beneficial"
    return None


# ── per-thread parsing state ─────────────────────────────────

def _new_game(thread: str, ts: datetime, first: str) -> dict:
    return {
        "thread": thread, "start": ts, "first_player": first, "turns": 0,
        "decisions": [],          # (turn, step, actor|None, chosen, playable list)
        "life": {"PlayerA": 20, "PlayerB": 20},
        "sims": [], "search_timeouts": 0, "search_complete": 0, "illegals_purged": 0,
        "attacks": Counter(), "targets": [],
        "pending_decision": None, "pending_playable": None, "pending_target_src": None,
        "last_sim_actor": None,
    }


def _ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f")


def parse_jvm_log(log_path: str, deck_a: str, deck_b: str, dck_a: str, dck_b: str) -> dict:
    """Parse one data-generation JVM log into per-game records plus a JVM summary."""
    names = {"PlayerA": load_deck_names(dck_a), "PlayerB": load_deck_names(dck_b)}
    oracle = load_oracle()
    games, open_games = [], {}
    summary = Counter()
    completed_order = []  # (thread, current WR) in completion order
    t_first = t_last = None

    def owner_of(card: str) -> Optional[str]:
        if card in ("PlayerA", "PlayerB"):
            return card
        in_a, in_b = card in names["PlayerA"], card in names["PlayerB"]
        return "PlayerA" if in_a and not in_b else "PlayerB" if in_b and not in_a else None

    with open(log_path, errors="replace") as f:
        for raw in f:
            m = LINE_RE.match(raw.rstrip("\n"))
            if not m:
                if "Exception" in raw and not raw.startswith("\t"):
                    summary["exception_lines"] += 1
                continue
            ts_s, msg, thread, src = m.groups()
            ts = _ts(ts_s)
            t_first = t_first or ts
            t_last = ts
            if raw.startswith("ERROR"):
                summary["error_lines"] += 1

            if msg.startswith("Player A won the die roll") or msg.startswith("Player B won the die roll"):
                open_games[thread] = _new_game(thread, ts, "PlayerA" if "Player A" in msg else "PlayerB")
                continue
            if "Game timed out" in msg:
                summary["games_timed_out"] += 1
            if msg.startswith("A game simulation failed") or msg.startswith("Caught an internal AI/Game exception") \
                    or msg.startswith("Worker thread failed"):
                summary["games_failed"] += 1
                open_games.pop(thread, None)
                continue

            g = open_games.get(thread)
            if msg.startswith("Current WR:"):
                completed_order.append((thread, float(msg.split(":")[1])))
                continue
            if g is None:
                continue

            if msg.startswith("Game #") and "completed successfully" in msg:
                g["end"] = ts
                games.append(g)
                del open_games[thread]
                continue

            if (tm := TURN_RE.match(msg)):
                g["turns"] = max(g["turns"], int(tm.group(1)))
            if (cm := CHOSE_RE.match(msg)):
                g["pending_decision"] = (int(cm.group(1)), cm.group(2), cm.group(3).strip(), g["pending_playable"] or [])
                g["pending_playable"] = None
                continue
            if msg.startswith("playable abilities: ["):
                g["pending_playable"] = [p.strip() for p in msg[len("playable abilities: ["):-1].split(", ")]
                continue
            if (am := ACTOR_RE.match(msg)):
                if g["pending_decision"]:
                    t, step, chosen, playable = g["pending_decision"]
                    g["decisions"].append((t, step, am.group(1), chosen, playable))
                    g["pending_decision"] = None
                continue
            if (lm := LIFE_RE.match(msg)):
                g["life"] = {"PlayerA": int(lm.group(2)), "PlayerB": int(lm.group(3))}
                continue
            if (sm := SIM_RE.match(msg)):
                g["last_sim_actor"] = sm.group(1)
                secs = float(sm.group(3))
                if secs > 0:
                    g["sims"].append((int(sm.group(2)), secs))
                continue
            if msg.startswith("timed out, ending search"):
                g["search_timeouts"] += 1
                continue
            if msg.startswith("required visits reached"):
                g["search_complete"] += 1
                continue
            if (pm := PURGE_RE.match(msg)):
                g["illegals_purged"] += int(pm.group(1))
                continue
            if (um := USE_ATTACK_RE.match(msg)):
                if um.group(2) == "true" and g["last_sim_actor"]:
                    g["attacks"][g["last_sim_actor"]] += 1
                continue
            if msg.startswith("base choose target "):
                g["pending_target_src"] = msg[len("base choose target "):]
                continue
            if msg.startswith("Targeting ") and g["pending_target_src"]:
                target = msg[len("Targeting "):]
                src_text = g["pending_target_src"]
                g["pending_target_src"] = None   # one source per choice; don't reuse it later
                if target == "Stop Choosing" or NON_TARGET_PROMPT_RE.search(src_text):
                    continue
                actor = g["last_sim_actor"]
                if src_text.startswith("Cast "):
                    card = src_text[5:]
                    actor = owner_of(card) or actor
                    effect = classify_effect(oracle.get(card, {}).get("oracle_text", ""))
                else:
                    effect = classify_effect(src_text)
                g["targets"].append({"actor": actor, "target_owner": owner_of(target), "effect": effect,
                                     "src": src_text[:80], "target": target})

    # winners from the global WR counter (wins = round(WR * completed so far))
    winners, wins_prev = {}, 0
    for i, (thread, wr) in enumerate(completed_order, 1):
        wins = round(wr * i)
        winners.setdefault(thread, []).append("PlayerA" if wins > wins_prev else "PlayerB")
        wins_prev = wins

    records = []
    per_thread_idx = defaultdict(int)
    for g in games:
        k = per_thread_idx[g["thread"]]
        per_thread_idx[g["thread"]] += 1
        winner = None
        life = g["life"]
        if life["PlayerA"] <= 0 < life["PlayerB"]:
            winner = "PlayerB"
        elif life["PlayerB"] <= 0 < life["PlayerA"]:
            winner = "PlayerA"
        elif k < len(winners.get(g["thread"], [])):
            winner = winners[g["thread"]][k]
        records.append(_game_record(g, winner, deck_a, deck_b))

    hours = max((t_last - t_first).total_seconds(), 1) / 3600 if t_first else None
    return {
        "log": str(log_path), "deck_a": deck_a, "deck_b": deck_b,
        "games": records,
        "summary": {
            "games_completed": len(records),
            "games_failed": summary["games_failed"],
            "games_timed_out": summary["games_timed_out"],
            "games_unfinished": len(open_games),
            "error_lines": summary["error_lines"],
            "exception_lines": summary["exception_lines"],
            "wall_hours": hours,
            "games_per_hour": len(records) / hours if hours else None,
        },
    }


def _active_player_by_turn(decisions, first_player) -> dict:
    """Turn parity: the player who plays lands in a turn is the active player. Majority vote
    over observed land plays decides whether odd turns belong to `first_player`."""
    votes = Counter()
    for t, step, actor, chosen, _ in decisions:
        if chosen.startswith("Play ") and actor:
            votes[(t % 2, actor)] += 1
    odd_owner = first_player
    other = "PlayerB" if first_player == "PlayerA" else "PlayerA"
    if votes[(1, other)] + votes[(0, first_player)] > votes[(1, first_player)] + votes[(0, other)]:
        odd_owner = other
    even_owner = "PlayerB" if odd_owner == "PlayerA" else "PlayerA"
    return {1: odd_owner, 0: even_owner}


def _game_record(g: dict, winner: Optional[str], deck_a: str, deck_b: str) -> dict:
    parity = _active_player_by_turn(g["decisions"], g["first_player"])
    per = {p: Counter() for p in ("PlayerA", "PlayerB")}
    turn_info = defaultdict(lambda: {"land_available": False, "land_played": False, "cast_available": False, "acted": False})

    for t, step, actor, chosen, playable in g["decisions"]:
        if actor is None:
            continue
        c = per[actor]
        c["priority_decisions"] += 1
        own_turn = parity[t % 2] == actor
        casts_available = any(p.startswith("Cast ") for p in playable)
        lands_available = any(p.startswith("Play ") for p in playable)
        if chosen.startswith("Play "):
            c["lands_played"] += 1
        elif chosen.startswith("Cast "):
            c["spells_cast"] += 1
        elif chosen == "Pass":
            c["passes"] += 1
        else:
            c["abilities_activated"] += 1
        if own_turn:
            ti = turn_info[(actor, t)]
            ti["land_available"] |= lands_available
            ti["cast_available"] |= casts_available
            ti["land_played"] |= chosen.startswith("Play ")
            ti["acted"] |= not (chosen == "Pass" or chosen.startswith("Play "))
            if step in MAIN_STEPS:
                c["own_main_decisions"] += 1
                # passing in your own main phase while a spell or land was castable
                if chosen == "Pass" and (casts_available or lands_available):
                    c["pass_with_play"] += 1

    for (actor, t), ti in turn_info.items():
        c = per[actor]
        c["own_turns_observed"] += 1
        if ti["land_available"] and not ti["land_played"]:
            c["missed_land_drops"] += 1        # had a land to play and never played one that turn
        if t >= 5 and ti["cast_available"] and not ti["acted"]:
            c["idle_turns"] += 1               # could cast something, did nothing but land/pass

    for tgt in g["targets"]:
        if tgt["actor"] not in per or tgt["effect"] is None or tgt["target_owner"] is None:
            continue
        c = per[tgt["actor"]]
        c["targets_classified"] += 1
        own = tgt["target_owner"] == tgt["actor"]
        if (tgt["effect"] == "harmful" and own) or (tgt["effect"] == "beneficial" and not own):
            c["bad_targets"] += 1
    for p, n in g["attacks"].items():
        per[p]["attackers_declared"] += n

    sims = sum(n for n, _ in g["sims"])
    sim_secs = sum(s for _, s in g["sims"])
    searches = g["search_timeouts"] + g["search_complete"]
    return {
        "deck_a": deck_a, "deck_b": deck_b, "thread": g["thread"],
        "first_player": g["first_player"], "winner": winner,
        "turns": g["turns"],
        "duration_s": (g["end"] - g["start"]).total_seconds(),
        "final_life": g["life"],
        "mcts_sims": sims, "mcts_sims_per_sec": sims / sim_secs if sim_secs else None,
        "search_timeout_rate": g["search_timeouts"] / searches if searches else None,
        "illegals_purged": g["illegals_purged"],
        "players": {p: dict(c) for p, c in per.items()},
    }


# ── aggregation ──────────────────────────────────────────────

def wilson(wins: int, n: int, z: float = 1.96) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if n == 0:
        return None, None, None
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def summarize_games(parsed: dict) -> dict:
    """Flatten a parse_jvm_log result into one metrics row (per player prefixes a_/b_)."""
    games = parsed["games"]
    n = len(games)
    row = dict(parsed["summary"])
    row.update(deck_a=parsed["deck_a"], deck_b=parsed["deck_b"])
    if n == 0:
        return row
    decided = [g for g in games if g["winner"]]
    wins_a = sum(g["winner"] == "PlayerA" for g in decided)
    p, lo, hi = wilson(wins_a, len(decided))
    row.update(a_winrate=p, a_winrate_lo=lo, a_winrate_hi=hi, decided_games=len(decided),
               on_play_winrate=(sum(g["winner"] == g["first_player"] for g in decided) / len(decided)) if decided else None)
    for key in ("turns", "duration_s", "mcts_sims_per_sec", "search_timeout_rate", "illegals_purged"):
        vals = [g[key] for g in games if g[key] is not None]
        row[f"mean_{key}"] = float(np.mean(vals)) if vals else None
    row["games_over_20_turns"] = sum(g["turns"] > 20 for g in games)
    for side, p_name in (("a", "PlayerA"), ("b", "PlayerB")):
        tot = Counter()
        for g in games:
            tot.update(g["players"][p_name])
        for k in ("spells_cast", "lands_played", "abilities_activated", "attackers_declared", "missed_land_drops",
                  "idle_turns", "bad_targets"):
            row[f"{side}_{k}_per_game"] = tot[k] / n
        row[f"{side}_pass_with_play_rate"] = tot["pass_with_play"] / tot["own_main_decisions"] if tot["own_main_decisions"] else None
        row[f"{side}_bad_target_rate"] = tot["bad_targets"] / tot["targets_classified"] if tot["targets_classified"] else None
        row[f"{side}_targets_classified"] = tot["targets_classified"]
    return row


def dataset_stats(files: list[str]) -> dict:
    """Cheap structural stats over HDF5 shards (no model needed)."""
    import h5py
    ActionType = _ActionType
    n_states = 0
    by_type = Counter()
    legal, one_hot, decisions = 0, 0, 0
    values, top_mass = [], []
    for path in files:
        with h5py.File(path, "r") as f:
            row = f["/row"][...]
        if row.size == 0:
            continue
        A = row.shape[1] - 4
        policy, value, is_player, atype = row[:, :A], row[:, A], row[:, A + 2] > 0.5, row[:, A + 3].astype(int)
        n_states += len(row)
        for t, name in ((ActionType.PRIORITY.value, "priority"), (ActionType.CHOOSE_TARGET.value, "target"),
                        (ActionType.CHOOSE_USE.value, "use")):
            by_type[name] += int((atype == t).sum())
        nz = (policy > 0).sum(axis=1)
        dm = nz > 0
        decisions += int(dm.sum())
        legal += int(nz[dm].sum())
        one_hot += int((nz == 1).sum())
        totals = policy[dm].sum(axis=1, keepdims=True)
        top_mass.extend((policy[dm].max(axis=1) / np.clip(totals[:, 0], 1e-8, None)).tolist())
        values.extend(value.tolist())
    v = np.asarray(values) if values else np.zeros(1)
    return {
        "files": len(files), "states": n_states, "decision_states": decisions,
        "priority_states": by_type["priority"], "target_states": by_type["target"], "use_states": by_type["use"],
        "mean_legal_actions": legal / decisions if decisions else None,
        "one_hot_policy_frac": one_hot / decisions if decisions else None,     # search collapsed to one move
        "mean_top_action_mass": float(np.mean(top_mass)) if top_mass else None,
        "value_label_mean": float(v.mean()), "value_label_abs_mean": float(np.abs(v).mean()),
        "value_label_pos_frac": float((v > 0).mean()),
    }


class _ActionType:
    """MageZero's decision-type codes (magezero.model.ActionType), kept here so reading a shard
    doesn't import torch."""
    class PRIORITY: value = 0
    class CHOOSE_TARGET: value = 3
    class CHOOSE_USE: value = 5


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(), **record}, default=str) + "\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="parse a MageZero JVM log into game metrics")
    ap.add_argument("log")
    ap.add_argument("--deck-a", required=True)
    ap.add_argument("--deck-b", required=True)
    ap.add_argument("--games-out", default=None)
    args = ap.parse_args()
    parsed = parse_jvm_log(args.log, args.deck_a, args.deck_b,
                           f"xmage/decks/{args.deck_a}.dck", f"xmage/decks/{args.deck_b}.dck")
    if args.games_out:
        Path(args.games_out).write_text("\n".join(json.dumps(g) for g in parsed["games"]) + "\n")
    print(json.dumps(summarize_games(parsed), indent=1))
