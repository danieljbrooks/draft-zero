"""Expert iteration on mtg-kernel FDN games: self-play with the current searcher, train the next network, evaluate it.

Generation k (directory WORK/g<k>):
  selfplay  `dzk play` searcher_k vs itself on the training pairs, `--data-out` -> g<k>/data
            (k = 0: `mcts:N`; k >= 1: `pmcts:N,net=<net_k>` plus --selfplay-opts, e.g. Dirichlet noise)
  train     py/train.py on the data of generations k-window+1..k (warm start from net_k) -> g<k>/train/net.dzkn
            = net_{k+1}
  parity    py/parity.py + `dzk parity` on a stratified sample of g<k>'s decisions (Rust vs PyTorch, max |delta|
            < 1e-4; on the pod this exercises its own forward kernels)
  eval      paired games of net_{k+1} on the eval pairs: `net` (greedy) and `pmcts:N` against `mcts:N` and
            `random` (--evals), on pairs/eval_v1.tsv (cycled with new shuffles past its 62 lines) and optionally the
            fixture pair (--fixture-pairs); every generation uses the same --eval-seed, so generations are compared
            on the same games

Every step is resumable and runs under a wall-clock hard kill (--timeout-* seconds; the whole process group is
killed). A step records its inputs (bot specs, the sha256 of every network and warm-start checkpoint it uses, the
data it trains on, seeds, arguments) in WORK/loop_state.json before it starts: a finished step whose inputs are
unchanged is skipped, an unfinished one is resumed (`dzk play --resume`; train.py resumes from ckpt_last.pt), and
a step whose inputs changed (a network retrained under the same path, new self-play data) has its old outputs moved
aside to <dir>.stale-<time> and is run again, which in turn changes the inputs of every step after it. The trainer
gets --max-minutes = 0.9 x --timeout-train, so it stops, evaluates and exports before the kill. Run from anywhere,
with torch available (the trainer runs with this interpreter):

    uv run --no-project --python 3.12 --with torch --with numpy python mtgkernel/py/loop.py --work runs/ei1 \\
        --gens 1 --selfplay-pairs 20 --budget 100 --threads 8 --epochs 2 \\
        --evals net:random:20,net:mcts:20,pmcts:mcts:10
"""

import argparse
import hashlib
import json
import os
import shlex
import signal
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # mtgkernel/

_sha_cache = {}


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def sha256(path):
    """sha256 of a file's content (cached by path, size and mtime); None if missing."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _sha_cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _sha_cache[key] = h.hexdigest()
    return _sha_cache[key]


def net_id(path):
    """A dzk-net-v1 file's weights identity: its header's blob sha256 (loaders check the blob against it; the
    header's export time does not count, so re-exporting the same weights changes nothing). None if missing."""
    try:
        with open(path, "rb") as f:
            if f.read(8) != b"DZKNET01":
                return sha256(path)
            (hl,) = struct.unpack("<Q", f.read(8))
            return json.loads(f.read(hl))["blob_sha256"]
    except FileNotFoundError:
        return None


def nets_of(*specs):
    """{path: weights sha256} of the network files bot specs name (net:path=F, net,path=F, pmcts:N,net=F)."""
    out = {}
    for spec in specs:
        for part in spec.replace(":", ",", 1).split(","):
            k, _, v = part.partition("=")
            if k.strip() in ("path", "net") and v:
                out[v] = net_id(v)
    return out


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


class Loop:
    def __init__(self, a):
        self.a = a
        self.work = os.path.abspath(a.work)
        os.makedirs(self.work, exist_ok=True)
        self.state_path = os.path.join(self.work, "loop_state.json")
        self.state = json.load(open(self.state_path)) if os.path.exists(self.state_path) else {"steps": {}}
        self.state["args"] = vars(a)

    def save(self):
        tmp = self.state_path + ".tmp"
        json.dump(self.state, open(tmp, "w"), indent=1)
        os.replace(tmp, self.state_path)

    def step(self, name):
        return self.state["steps"].setdefault(name, {})

    def prepare(self, name, inputs, outputs):
        """Record a step's inputs before it runs. If any of `outputs` exists but was made from other (or unrecorded)
        inputs, move it aside. Returns True if the step's existing outputs are its current inputs' own."""
        st = self.step(name)
        inputs = json.loads(json.dumps(inputs))  # as stored (tuples become lists)
        fresh = st.get("inputs") == inputs
        if not fresh:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            for o in outputs:
                if os.path.exists(o):
                    dest = f"{o}.stale-{stamp}"
                    os.rename(o, dest)
                    log(f"[{name}] inputs changed: moved {o} aside to {dest}")
            for k in [k for k in st if k != "inputs"]:
                del st[k]
            st["inputs"] = inputs
            st["inputs_digest"] = digest(inputs)
            self.save()
        return fresh

    def run(self, name, cmd, timeout, cwd=ROOT, stdout=None):
        """Run a step's command in its own process group, killed after `timeout` seconds."""
        log(f"[{name}] {shlex.join(cmd)}")
        t0 = time.time()
        out = open(stdout, "a") if stdout else None
        p = subprocess.Popen(cmd, cwd=cwd, stdout=out or None, stderr=subprocess.STDOUT if out else None,
                             start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            rc = "timeout"
        finally:
            if out:
                out.close()
        dt = round(time.time() - t0, 1)
        self.step(name).update({"cmd": cmd, "rc": rc, "seconds": dt, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        self.save()
        log(f"[{name}] rc={rc} in {dt} s")
        if rc != 0:
            raise SystemExit(f"step {name} failed (rc={rc}); rerun the same command to resume")
        return dt

    @staticmethod
    def games_in(path):
        if not os.path.exists(path):
            return 0
        with open(path, "rb") as f:
            return sum(1 for line in f if line.endswith(b"\n") and line.strip())

    def play(self, name, out_dir, bot1, bot2, pairs, decks_dir, n_pairs, seed, timeout, pair_range=None,
             data_out=None):
        a = self.a
        games = os.path.join(out_dir, "games.jsonl")
        want = 2 * (pair_range[1] - pair_range[0] if pair_range else n_pairs)
        inputs = {"bot1": bot1, "bot2": bot2, "nets": nets_of(bot1, bot2), "pairs": pairs, "decks_dir": decks_dir,
                  "n_pairs": n_pairs, "pair_range": pair_range, "seed": seed, "data_out": data_out,
                  "max_permanents": a.max_permanents, "max_game_seconds": a.max_game_seconds}
        fresh = self.prepare(name, inputs, [out_dir] + ([data_out] if data_out else []))
        if fresh and self.games_in(games) >= want and os.path.exists(os.path.join(out_dir, "summary.json")):
            log(f"[{name}] done ({want} games)")
            return
        os.makedirs(out_dir, exist_ok=True)
        cmd = [a.dzk, "play", "--decks-dir", decks_dir, "--pairs", pairs, "--bot1", bot1, "--bot2", bot2,
               "--seed", str(seed), "--threads", str(a.threads), "--out", games,
               "--max-permanents", str(a.max_permanents), "--max-game-seconds", str(a.max_game_seconds)]
        cmd += ["--pair-range", f"{pair_range[0]}:{pair_range[1]}"] if pair_range else ["--n-pairs", str(n_pairs)]
        if os.path.exists(games) and os.path.getsize(games) > 0:
            cmd.append("--resume")
        if data_out:
            cmd += ["--data-out", data_out]
        self.run(name, cmd, timeout, stdout=os.path.join(out_dir, "stderr.log"))
        s = json.load(open(os.path.join(out_dir, "summary.json")))
        self.step(name)["summary"] = {k: s.get(k) for k in (
            "games", "valid_games", "errors", "bot1_score", "ci95", "mean_turns", "mean_seconds")}
        self.step(name)["summary"]["truncated"] = s.get("terminals", {}).get("truncated")
        self.step(name)["summary"]["games_per_hour"] = s.get("this_run", {}).get("games_per_hour")
        self.save()

    def net(self, k):
        """net_k: trained after generation k-1 (None for k = 0)."""
        return None if k == 0 else os.path.join(self.work, f"g{k - 1}", "train", "net.dzkn")

    def searcher(self, k):
        a = self.a
        if k == 0:
            return f"mcts:{a.budget}{a.mcts_opts}"
        return f"pmcts:{a.budget},net={self.net(k)}{a.selfplay_opts}"

    def generation(self, k):
        a = self.a
        g = os.path.join(self.work, f"g{k}")
        os.makedirs(g, exist_ok=True)
        # ---- self-play
        bot = self.searcher(k)
        lo = (k * a.selfplay_pairs) if a.rotate_pairs else 0
        self.play(f"g{k}.selfplay", os.path.join(g, "selfplay"), bot, bot, a.train_pairs, a.train_decks_dir,
                  a.selfplay_pairs, a.seed + 1000 * k, a.timeout_selfplay, pair_range=(lo, lo + a.selfplay_pairs),
                  data_out=os.path.join(g, "data"))
        # ---- train net_{k+1}
        tdir = os.path.join(g, "train")
        name = f"g{k}.train"
        window = list(range(max(0, k - a.window + 1), k + 1))
        data = [os.path.join(self.work, f"g{j}", "data") for j in window]
        init = os.path.join(self.work, f"g{k - 1}", "train", "ckpt_best.pt") if k >= 1 and a.warm_start else None
        cmd = [sys.executable, os.path.join(HERE, "train.py"), "--data", *data, "--out", tdir,
               "--epochs", str(a.epochs), "--threads", str(a.threads), "--seed", str(a.seed + k),
               "--max-minutes", str(round(0.9 * a.timeout_train / 60, 1))]
        if init:
            cmd += ["--init", init]
        cmd += shlex.split(a.train_args)
        # (threads and the time limit do not change what is trained)
        args = [x for i, x in enumerate(cmd[2:]) if not (x in ("--threads", "--max-minutes")
                                                          or (i > 0 and cmd[2:][i - 1] in ("--threads", "--max-minutes")))]
        inputs = {"data": {f"g{j}": self.step(f"g{j}.selfplay").get("inputs_digest") for j in window},
                  "init": sha256(init) if init else None, "args": args}
        fresh = self.prepare(name, inputs, [tdir])
        if fresh and os.path.exists(os.path.join(tdir, "net.dzkn")) and os.path.exists(os.path.join(tdir, "metrics.json")):
            log(f"[{name}] done")
        else:
            os.makedirs(tdir, exist_ok=True)
            self.run(name, cmd, a.timeout_train, stdout=os.path.join(tdir, "train.log"))
            m = json.load(open(os.path.join(tdir, "metrics.json")))
            best = next((h for h in m["history"] if h["epoch"] == m["best_epoch"]), {})
            self.step(name)["metrics"] = {"best_epoch": m["best_epoch"], "select": m.get("select"),
                                          "stopped_early": m.get("stopped_early"), "train_seconds": m["train_seconds"],
                                          "train_decisions": m["train_decisions"], "val_decisions": m["val_decisions"],
                                          "val": best.get("val"), "net_sha256": m.get("net_sha256")}
            self.save()
        net = os.path.join(tdir, "net.dzkn")
        # ---- parity
        pdir = os.path.join(g, "parity")
        name = f"g{k}.parity"
        if a.parity_decisions > 0:
            fresh = self.prepare(name, {"net": net_id(net), "data": self.step(f"g{k}.selfplay").get("inputs_digest"),
                                        "decisions": a.parity_decisions}, [pdir])
            if fresh and os.path.exists(os.path.join(pdir, "parity.json")):
                log(f"[{name}] done")
            else:
                os.makedirs(pdir, exist_ok=True)
                cmd = [sys.executable, os.path.join(HERE, "parity.py"), "--net", net, "--data",
                       os.path.join(g, "data"), "--min-decisions", str(a.parity_decisions), "--out", pdir]
                self.run(name, cmd, 1800, stdout=os.path.join(pdir, "parity_py.log"))
                r = subprocess.run([a.dzk, "parity", "--net", os.path.join(pdir, "net.dzkn"), "--data",
                                    os.path.join(pdir, "decisions.dzd.gz"), "--expected",
                                    os.path.join(pdir, "expected.json")], capture_output=True, text=True)
                if r.returncode != 0:
                    raise SystemExit(f"parity failed: {r.stdout}\n{r.stderr}")
                res = json.loads(r.stdout)
                exp = json.load(open(os.path.join(pdir, "expected.json")))
                res["families"] = exp.get("families")
                res["max_actions"] = exp.get("max_actions")
                json.dump(res, open(os.path.join(pdir, "parity.json"), "w"), indent=1)
                self.step(name)["parity"] = {k2: res[k2] for k2 in (
                    "decisions", "max_abs_delta_logit", "max_abs_delta_value", "families", "max_actions")}
                self.save()
                for f in ("decisions.dzd.gz", "expected.json"):  # keep parity.json only
                    if not a.keep_parity_files:
                        os.remove(os.path.join(pdir, f))
        # ---- evaluation of net_{k+1}
        for spec in [e for e in a.evals.split(",") if e]:
            kind, opp, n = spec.split(":")
            bot1 = f"net:path={net}" if kind == "net" else f"pmcts:{a.budget},net={net}{a.eval_pmcts_opts}"
            bot2 = {"mcts": f"mcts:{a.budget}{a.mcts_opts}", "random": "random", "first": "first"}.get(opp, opp)
            tag = f"{kind}_vs_{opp}"
            self.play(f"g{k}.eval.{tag}", os.path.join(g, "eval", tag), bot1, bot2, a.eval_pairs, a.eval_decks_dir,
                      int(n), a.eval_seed, a.timeout_eval)
            if a.fixture_pairs > 0:
                self.play(f"g{k}.eval.{tag}.fixture", os.path.join(g, "eval", tag + "_fixture"), bot1, bot2,
                          a.fixture_pairs_file, a.eval_decks_dir, a.fixture_pairs, a.eval_seed, a.timeout_eval)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--gens", type=int, default=1, help="generations to run (from --start-gen)")
    ap.add_argument("--start-gen", type=int, default=0)
    ap.add_argument("--dzk", default=os.path.join(ROOT, "target", "release", "dzk"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--budget", type=int, default=100, help="simulations per decision (mcts:N, pmcts:N)")
    ap.add_argument("--mcts-opts", default="", help="extra mcts options, e.g. ',worlds=2'")
    ap.add_argument("--selfplay-opts", default=",noise=0.25,alpha=0.3,temp_moves=8,temp=1",
                    help="extra pmcts options for self-play generations >= 1")
    ap.add_argument("--eval-pmcts-opts", default="", help="extra pmcts options for the pmcts evals, e.g. ',fpu=0.2'")
    ap.add_argument("--selfplay-pairs", type=int, default=1500,
                    help="pairs per generation (2 games each; the 360-line training file is cycled with new shuffles)")
    ap.add_argument("--rotate-pairs", type=int, default=1, help="1: generation k plays pairs k*P..(k+1)*P-1")
    ap.add_argument("--train-pairs", default=os.path.join(ROOT, "decks", "gen_v1", "pairs_train.tsv"))
    ap.add_argument("--train-decks-dir", default=os.path.join(ROOT, "decks", "gen_v1"))
    ap.add_argument("--eval-pairs", default=os.path.join(ROOT, "pairs", "eval_v1.tsv"))
    ap.add_argument("--fixture-pairs-file", default=os.path.join(ROOT, "pairs", "fdn_fixture.tsv"))
    ap.add_argument("--eval-decks-dir",
                    default=os.path.join(ROOT, "decks", "gen_v1") + ":" + os.path.join(ROOT, "decks", "fixtures"))
    ap.add_argument("--evals", default="net:random:100,net:mcts:500,pmcts:mcts:250",
                    help="comma list of KIND:OPPONENT:PAIRS, KIND = net | pmcts, OPPONENT = mcts | random | a spec")
    ap.add_argument("--fixture-pairs", type=int, default=0, help="also play each eval on the fixture pair")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--eval-seed", type=int, default=7)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--window", type=int, default=2, help="train on the last W generations' data")
    ap.add_argument("--warm-start", type=int, default=1)
    ap.add_argument("--train-args", default="", help="extra train.py arguments, e.g. '--value-target mix:0.5'")
    ap.add_argument("--parity-decisions", type=int, default=300)
    ap.add_argument("--keep-parity-files", action="store_true")
    ap.add_argument("--max-permanents", type=int, default=100, help="dzk play: end a runaway game as truncated")
    ap.add_argument("--max-game-seconds", type=float, default=1800, help="dzk play: wall-clock limit per game")
    ap.add_argument("--timeout-selfplay", type=float, default=6 * 3600)
    ap.add_argument("--timeout-train", type=float, default=3 * 3600)
    ap.add_argument("--timeout-eval", type=float, default=3 * 3600)
    a = ap.parse_args()
    a.dzk = os.path.abspath(a.dzk)
    loop = Loop(a)
    for k in range(a.start_gen, a.start_gen + a.gens):
        log(f"==== generation {k}: searcher {loop.searcher(k)}")
        loop.generation(k)
    log("done")


if __name__ == "__main__":
    main()
