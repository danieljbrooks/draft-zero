"""
Play explicit deck pairs in gorge and read the results.

`dzgorge play -deckfile D -pairs-file P -single` plays each line of P (deckA, deckB, seed) once, deckA in seat 0, both
seats the same policy. This runs P in chunks, each under its own time limit (Go ignores SIGALRM; subprocess kills
with SIGKILL), keeps a killed chunk's complete games, replays the first missing game alone, and skips it if it hangs
again. Resumable: games already in OUT are not replayed.

  python -m sealed.games DECKS.jsonl PAIRS.tsv OUT.jsonl --sims 100 --workers 15
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
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
    per_game = 1.5 + 0.12 * max(sims, 0)
    secs = chunk_secs or int(3 * per_game * chunk / workers + 600)
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
                g1 = _run(deckfile, [pairs[miss]], pol, 1, max(300, int(10 * per_game)), None)
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
