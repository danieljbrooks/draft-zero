"""
Play explicit deck pairs in gorge and read the results.

`dzgorge play -deckfile D -pairs-file P -single` plays each line of P (deckA, deckB, seed) once, deckA in seat 0, both
seats the same policy. This runs P in chunks, each under its own time limit (Go ignores SIGALRM; subprocess kills
with SIGKILL), keeps a killed chunk's complete games, replays the first missing game alone, and skips it if it hangs
again. Resumable: games already in OUT are not replayed.

  python -m sealed.games DECKS.jsonl PAIRS.tsv OUT.jsonl --sims 100 --workers 15
"""
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("DZ_REPO") or os.path.dirname(os.path.dirname(HERE))
GORGE = os.environ.get("GORGE_DIR", os.path.join(os.path.dirname(REPO), "ext", "gorge-sealed"))
NET = os.environ.get("PLAY_NET", os.path.join(REPO, "data/sealed/nets/B7.dzgw"))
FULL = "autopay:oppnodes:oppfull:morekinds"


def spec(sims: int, net: str = NET, cpuct: float = 0.5) -> str:
    """The game-playing bot: docs/031's full search with loop B's in-engine network; cpuct 0.5 played best there."""
    if sims <= 0:
        return "bot"
    s = f"az:sims={sims}:{FULL}"
    if net:
        s += f":topk=6:gonet={net}"
    if cpuct:
        s += f":cpuct={cpuct}"
    return s


def read_pairs(path):
    return [l.rstrip("\n").split("\t") for l in open(path) if l.strip() and not l.startswith("#")]


def _run(deckfile, pairs, pol, workers, secs, out_lines):
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as f:
        for p in pairs:
            f.write("\t".join(p) + "\n")
        pf = f.name
    outp = pf + ".jsonl"
    cmd = [os.path.join(GORGE, "bin/dzgorge"), "play", "-cards", os.path.join(GORGE, ".cards"), "-deckfile", deckfile,
           "-pairs-file", pf, "-pairs", "0", "-single", "-a", pol, "-b", pol, "-workers", str(workers), "-mulligans", "2",
           "-mull-heuristic", "-topk-max-options", "64", "-progress", "0", "-out", outp]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=secs)
    except subprocess.TimeoutExpired:
        pass
    got = []
    if os.path.exists(outp):
        raw = open(outp, "rb").read()
        raw = raw[: raw.rfind(b"\n") + 1]
        got = [json.loads(l) for l in raw.splitlines() if l.strip()]
    for p in (pf, outp, outp + ".summary.json"):
        if os.path.exists(p):
            os.remove(p)
    return got


def play(deckfile, pairs_path, out, sims=100, workers=7, chunk=None, pol=None, chunk_secs=None, log=sys.stderr):
    """Play every pair in pairs_path not yet in out; append games (one JSON line each, with `line`, the pair's line
    index) to out. Returns the number of games played now."""
    pol = pol or spec(sims)
    pairs = read_pairs(pairs_path)
    done = set()
    if os.path.exists(out):
        for l in open(out):
            if l.strip():
                done.add(json.loads(l)["line"])
    todo = [i for i in range(len(pairs)) if i not in done]
    chunk = chunk or max(workers * 8, 64)
    # about 3x a game's usual time at these settings, per wave of `workers` games, plus 10 minutes
    per_game = 0.05 if sims <= 0 else 1.5 + 0.12 * sims
    secs = chunk_secs or int(3 * per_game * chunk / workers + (120 if sims <= 0 else 600))
    t0, n = time.time(), 0
    with open(out, "a") as f:
        while todo:
            batch = todo[:chunk]
            got = _run(deckfile, [pairs[i] for i in batch], pol, workers, secs, None)
            for g in got:
                g["line"] = batch[g["pair"]]
                f.write(json.dumps(g) + "\n")
            f.flush()
            n += len(got)
            if len(got) < len(batch):          # killed: replay the first missing game alone
                miss = batch[len(got)]
                g1 = _run(deckfile, [pairs[miss]], pol, 1, 60 if sims <= 0 else max(300, int(10 * per_game)), None)
                if g1:
                    g1[0]["line"] = miss
                    f.write(json.dumps(g1[0]) + "\n"); n += 1
                else:
                    print(f"line {miss} ({pairs[miss][:2]}) did not finish; skipped", file=log)
                    f.write(json.dumps({"line": miss, "result": "skipped", "decks": pairs[miss][:2], "winner": -1}) + "\n")
                todo = todo[len(got) + 1:]
            else:
                todo = todo[len(batch):]
            el = time.time() - t0
            print(f"[{time.strftime('%H:%M:%S')}] {n} games, {n / el * 3600:.0f}/h, {len(todo)} left", file=log, flush=True)
    return n


def results(path):
    """Games from an OUT file as dicts with decks, winner (seat, -1 = none) and starter."""
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("decks"); ap.add_argument("pairs"); ap.add_argument("out")
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--policy", default=None, help="a full dzgorge policy spec instead of --sims")
    a = ap.parse_args()
    play(a.decks, a.pairs, a.out, a.sims, a.workers, a.chunk, a.policy)


if __name__ == "__main__":
    main()


def play_distributed(deckfile, pairs_path, out, sims, workers, remotes=(), log=sys.stderr, **kw):
    """play(), with a share of the pairs sent to remote machines.

    remotes: dicts {host, workers, root (the remote repo checkout), python, gorge, speed (games per worker relative to
    here, default 1)}. Each remote gets its share of the pairs not yet played, plays them with this same module over
    ssh, and its games are copied back and appended to out. Whatever a remote doesn't return (it died, its ssh
    dropped) is played here afterwards, so the result is complete either way."""
    import threading
    pairs = read_pairs(pairs_path)
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["line"] for l in open(out) if l.strip()}
    todo = [i for i in range(len(pairs)) if i not in done]
    weights = [workers] + [r["workers"] * r.get("speed", 1.0) for r in remotes]
    cuts = np.cumsum([0] + weights) / sum(weights) * len(todo)
    shares = [todo[int(cuts[k]):int(cuts[k + 1])] for k in range(len(weights))]
    results, threads = {}, []
    stop = threading.Event()   # set when every remote pair has been played here: remotes stop and return nothing

    def remote(k, r, lines):
        # Detached on the remote (tmux if r["tmux"], else nohup) and polled, so a dropped ssh doesn't lose the shard.
        # A relaunch for the same output kills the previous session first and starts a fresh shard directory.
        name = f"dzs{hashlib.md5(os.path.abspath(out).encode()).hexdigest()[:8]}k{k}"
        rd = f"{r.get('tmp', '/tmp')}/{name}-{int(time.time())}"
        sub = os.path.join(os.path.dirname(os.path.abspath(out)), f".remote{k}.tsv")
        with open(sub, "w") as f:
            for i in lines:
                f.write("\t".join(pairs[i]) + "\n")
        secs = r.get("secs", 36000)
        try:
            ssh = lambda c, t=120: subprocess.run(["ssh", "-o", "ConnectTimeout=30", r["host"], c], timeout=t,
                                                  capture_output=True, text=True)
            kill = (f"tmux kill-session -t {name} 2>/dev/null; " if r.get("tmux") else
                    f"[ -f /tmp/{name}.pid ] && kill $(cat /tmp/{name}.pid) 2>/dev/null; ")
            ssh(kill + f"mkdir -p {rd}")
            subprocess.run(["rsync", "-a", deckfile, f"{r['host']}:{rd}/decks.jsonl"], check=True, timeout=600)
            subprocess.run(["rsync", "-a", sub, f"{r['host']}:{rd}/pairs.tsv"], check=True, timeout=600)
            job = (f"cd {r['root']}/gorge && GORGE_DIR={r['gorge']} PLAY_NET={r['net']} timeout {secs} "
                   f"{r['python']} -m sealed.games {rd}/decks.jsonl {rd}/pairs.tsv {rd}/games.jsonl --sims {sims} "
                   f"--workers {r['workers']} > {rd}/play.log 2>&1; touch {rd}/done")
            if r.get("tmux"):
                ssh(f"tmux new-session -d -s {name} {shlex.quote(job)}")
            else:
                ssh(f"nohup bash -c {shlex.quote(job)} > /dev/null 2>&1 < /dev/null & echo $! > /tmp/{name}.pid")
            t0, fails = time.time(), 0
            while time.time() - t0 < secs + 600:
                if stop.wait(60):
                    ssh(kill)
                    print(f"remote {r['host']}: stopped (its pairs were played here)", file=log, flush=True)
                    return
                try:
                    st = ssh(f"test -f {rd}/done && echo DONE", 60).stdout
                    fails = 0
                    if "DONE" in st:
                        break
                except subprocess.TimeoutExpired:
                    fails += 1
                    if fails > 30:
                        raise RuntimeError("unreachable for 30 polls")
            subprocess.run(["rsync", "-a", f"{r['host']}:{rd}/games.jsonl", sub + ".games"], check=True, timeout=600)
            got = [json.loads(l) for l in open(sub + ".games") if l.strip()]
            results[k] = [(lines[g["line"]], g) for g in got if g.get("line", -1) < len(lines)]
            print(f"remote {r['host']}: {len(got)} of {len(lines)} games", file=log, flush=True)
            ssh(f"rm -rf {rd}")
        except Exception as e:  # noqa: BLE001 - any failure: the lines are replayed here
            print(f"remote {r['host']} failed: {e}", file=log, flush=True)
        finally:
            for p in (sub, sub + ".games"):
                if os.path.exists(p):
                    os.remove(p)

    for k, r in enumerate(remotes, 1):
        t = threading.Thread(target=remote, args=(k, r, shares[k]), daemon=True)
        t.start(); threads.append(t)
    local = os.path.join(os.path.dirname(os.path.abspath(out)), ".local.tsv")
    with open(local, "w") as f:
        for i in shares[0]:
            f.write("\t".join(pairs[i]) + "\n")
    lo = out + ".local"
    play(deckfile, local, lo, sims=sims, workers=workers, log=log, **kw)
    # Work stealing: while a remote is still busy, play its pairs here too, from the far end of its share, a batch
    # at a time; whichever copy of a pair finishes first is kept.
    stolen, steal_rows = set(), []
    tail = [i for k in range(1, len(weights)) for i in reversed(shares[k])]
    batch = max(workers * 8, 64) if sims > 0 else max(workers * 200, 3000)
    while any(t.is_alive() for t in threads) and len(stolen) < len(tail):
        take = [i for i in tail if i not in stolen][:batch]
        sp = out + ".steal.tsv"
        with open(sp, "w") as f:
            for i in take:
                f.write("\t".join(pairs[i]) + "\n")
        so = out + ".steal.jsonl"
        if os.path.exists(so):
            os.remove(so)
        play(deckfile, sp, so, sims=sims, workers=workers, log=log, **kw)
        for g in (json.loads(l) for l in open(so) if l.strip()):
            g["line"] = take[g["line"]]
            steal_rows.append(g)
        stolen.update(take)
        os.remove(sp); os.remove(so)
    if steal_rows:
        print(f"stole {len(steal_rows)} games from the remotes' shares", file=log, flush=True)
    if tail and len(stolen) >= len(tail):
        stop.set()
    for t in threads:
        t.join()
    have = set(done)
    with open(out, "a") as f:
        rows = [(shares[0][g["line"]], g) for g in (json.loads(l) for l in open(lo) if l.strip())]
        rows += [r for k in sorted(results) for r in results[k]]
        rows += [(g["line"], g) for g in steal_rows]
        for line, g in rows:
            if line in have:
                continue
            have.add(line)
            g["line"] = line
            f.write(json.dumps(g) + "\n")
    os.remove(lo); os.remove(local)
    return play(deckfile, pairs_path, out, sims=sims, workers=workers, log=log, **kw)   # fill whatever is missing
