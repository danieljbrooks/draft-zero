"""Games/s through mtg-kernel's official Limited interface: the `kernel_limited_env` JSONL binary (schema 4, London
mulligans + Foundations combat + engine priority windows) driven from Python, one game at a time per process.
Policy: uniform random over the legal actions, with `activate_mana_ability` dropped while another action remains
(as dzk does), so the numbers compare with `dzk bench --bot random`. Standard library only.

    python3 mtgkernel/py/jsonl_bench.py --binary .deps/mtg-kernel/target/release/kernel_limited_env \\
        --deck assets/sample/decks/FDN_top_04956_UG.dck --deck assets/sample/decks/FDN_top_20626_WG.dck \\
        --seconds 30 --procs 1,8
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import subprocess
import time


def deck(path):
    cards = []
    for line in open(path, encoding="utf-8-sig"):
        line = line.strip()
        if not line or line.startswith(("NAME:", "SB:", "#")):
            continue
        count, rest = line.split(" ", 1)
        cards.append({"name": rest.split("] ", 1)[1] if "] " in rest else rest, "count": int(count)})
    return {"cards": cards}


def worker(args):
    binary, decks, seconds, seed = args
    proc = subprocess.Popen([binary, "--london-mulligans-v1"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, bufsize=1)
    rng = random.Random(seed)
    games = steps = errors = 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        ep = seed * 100000 + games
        d = decks if games % 2 == 0 else decks[::-1]
        req = {"request_type": "reset", "schema_version": 4, "request_id": f"r{ep}", "decks": d, "episode_id": ep,
               "env_seed": ep, "max_physical_decisions": 20000, "max_policy_steps": 20000}
        proc.stdin.write(json.dumps(req) + "\n")
        rep = json.loads(proc.stdout.readline())
        while rep["response_type"] == "decision":
            dec = rep["decision"]
            acts = dec["legal_actions"]
            real = [a for a in acts if a["semantic"]["action_kind"] != "activate_mana_ability"] or acts
            a = rng.choice(real)
            req = {"request_type": "step", "schema_version": 4, "request_id": f"s{ep}-{dec['step']}",
                   "episode_id": ep, "expected_step": dec["step"], "selected_index": a["selected_index"],
                   "selected_action_id": a["stable_id"]}
            proc.stdin.write(json.dumps(req) + "\n")
            rep = json.loads(proc.stdout.readline())
            steps += 1
        if rep["response_type"] == "error" or rep.get("terminal", {}).get("terminal_classification") != "natural":
            errors += 1
        games += 1
    proc.kill()
    return games, steps, errors, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--binary", required=True)
    ap.add_argument("--deck", action="append", required=True)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--procs", default="1")
    a = ap.parse_args()
    decks = [deck(p) for p in a.deck]
    out = []
    for n in [int(x) for x in a.procs.split(",")]:
        load0 = os.getloadavg()
        with mp.Pool(n) as pool:
            res = pool.map(worker, [(a.binary, decks, a.seconds, s + 1) for s in range(n)])
        g = sum(r[0] for r in res)
        st = sum(r[1] for r in res)
        el = max(r[3] for r in res)
        out.append({"procs": n, "games": g, "steps": st, "errors": sum(r[2] for r in res), "seconds": round(el, 2),
                    "games_per_s": g / el, "games_per_hour": 3600 * g / el, "steps_per_s": st / el,
                    "load_before": load0, "load_after": os.getloadavg()})
        print(json.dumps(out[-1]))
    print(json.dumps({"interface": "kernel_limited_env JSONL schema 4", "policy": "random (mana masked)",
                      "results": out}, indent=1))


if __name__ == "__main__":
    main()
