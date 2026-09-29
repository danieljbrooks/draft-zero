"""docs/012's hidden-information test (E1): probe pairs of worlds that look the same to the
deciding player and differ only in cards it can't see. A fair search decides the same in both.

    python tools/search_bench/leak.py run --out runs/search_bench/e1 --workers 30 [--seeds 16] [--budget 3000]
    python tools/search_bench/leak.py analyze --out runs/search_bench/e1

Each method gets its worlds the way tools/search_bench/run.py gives them on the benchmark:
clairvoyant MCTS searches the real world; PIMC and IS-MCTS search belief worlds, which
`public_worlds` draws from the position with every hidden card taken out (B's hand becomes a
count, B's decklist unknown, A's known library top dropped), so the same seed gives the same
worlds in both members of a pair. The pairs (docs/009's two positions, with extreme versions):

  counterspell      B holds Refute + Island / Island + Island; A has Serra Angel and 6 lands
  counterspell_x    B holds Refute x3 / Island x3
  cantrip           A's (unseen) library top is Llanowar Elves / Plains; A holds a cantrip creature
  cantrip_x         A's top five are Elves / Plains
  decklist          B's decklist holds three counterspells / none; B has shown only Mountains
  canary            B holds Counterspell x2, a card outside FDN that no belief can deal / Island x2

A pair passes when every option's paired Q difference has a 95% CI inside +-0.1 and the chosen
actions don't differ (Fisher p > 0.01). For the canary, no search world may hold a Counterspell.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools/gameplay"))

from hidden_info_leak import cantrip_spec, counterspell_spec  # noqa: E402

METHODS = ("clairvoyant", "pimc1", "pimc4", "ismcts")
SEARCH_SEED = 7


def _swap(deck: list[str], out: str, inn: str, n: int) -> list[str]:
    d = list(deck)
    for _ in range(n):
        d.remove(out)
        d.append(inn)
    return d


def probes() -> dict[str, dict]:
    P = {}
    P["counterspell"] = {"X": counterspell_spec(["Refute", "Island"]), "Y": counterspell_spec(["Island", "Island"])}
    x, y = counterspell_spec(["Refute"] * 3), counterspell_spec(["Island"] * 3)
    for s in (x, y):  # three Refutes in both decklists: only the hand differs
        s["players"]["B"]["decklist"] = _swap(s["players"]["B"]["decklist"], "Mountain", "Refute", 2)
    P["counterspell_x"] = {"X": x, "Y": y}
    P["cantrip"] = {"X": cantrip_spec("Llanowar Elves"), "Y": cantrip_spec("Plains")}
    x, y = cantrip_spec("Llanowar Elves"), cantrip_spec("Plains")
    x["players"]["A"]["libraryTop"] = ["Llanowar Elves"] * 5
    y["players"]["A"]["libraryTop"] = ["Plains"] * 5
    for s in (x, y):  # the same decklist in both: five Elves
        s["players"]["A"]["decklist"] = _swap(s["players"]["A"]["decklist"], "Forest", "Llanowar Elves", 2)
    P["cantrip_x"] = {"X": x, "Y": y}
    # decklist: B's hand of three is unknown, dealt from its real list, which holds four
    # counterspells (Refute x3, Essence Scatter) / none (Islands instead)
    x, y = counterspell_spec(["Island"] * 3), counterspell_spec(["Island"] * 3)
    xb, yb = x["players"]["B"], y["players"]["B"]
    xb["decklist"] = _swap(xb["decklist"], "Mountain", "Refute", 2)
    yb["decklist"] = [("Island" if c in ("Refute", "Essence Scatter") else c) for c in yb["decklist"]]
    for b in (xb, yb):
        b["hand"] = []
        b["handUnknown"] = 3
    P["decklist"] = {"X": x, "Y": y}
    x, y = counterspell_spec(["Counterspell", "Counterspell"]), counterspell_spec(["Island", "Island"])
    for s in (x, y):
        s["players"]["B"]["decklist"] = _swap(s["players"]["B"]["decklist"], "Mountain", "Counterspell", 2)
    P["canary"] = {"X": x, "Y": y}
    return P


def public_spec(spec: dict):
    """What player A can see: B's hand as a count, B's decklist unknown, A's library top unknown."""
    from draftzero.gameplay.statespec import StateSpec
    s = copy.deepcopy(spec)
    b = s["players"]["B"]
    b["handUnknown"] = len(b.get("hand") or []) + int(b.get("handUnknown") or 0)
    b["hand"] = []
    b["decklist"] = []
    b["decklistSource"] = "belief"
    s["players"]["A"]["libraryTop"] = []
    return StateSpec.from_dict(s)


def public_worlds(spec: dict, seed: int, k: int = 8) -> list[dict]:
    from draftzero.gameplay import coach as co
    worlds, info = co.plan_determinization(public_spec(spec), k=k, seed=seed)
    if worlds is None:
        raise RuntimeError(f"no belief worlds: {info}")
    return [w.to_dict() for w in worlds]


def request(method: str, real: dict, seed: int, budget: int) -> tuple[list, dict]:
    opts = {"budget": budget, "seed": 1000 + seed, "decisionPlayer": "A", "timeoutSec": 3000}
    if method == "clairvoyant":
        return [real], {**opts, "method": "tree", "worldSeeds": [5000 + seed]}
    worlds = public_worlds(real, SEARCH_SEED + seed)
    if method == "pimc1":
        return worlds[:1], {**opts, "method": "tree"}
    if method == "pimc4":
        return worlds[:4], {**opts, "method": "tree"}
    return worlds, {**opts, "method": "ismcts"}


def cmd_run(a) -> int:
    from draftzero.gameplay.bridge import BridgePool
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "probes.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            if not r.get("error"):
                done.add((r["probe"], r["world"], r["method"], r["seed"], r["budget"], r["evaluator"]))
    P = probes()
    methods = a.methods.split(",") if a.methods else list(METHODS)
    names = a.probes.split(",") if a.probes else list(P)
    jobs = [(p, w, m, s) for p in names for m in methods for s in range(a.seeds) for w in ("X", "Y")
            if (p, w, m, s, a.budget, a.evaluator) not in done]
    # IS-MCTS is the slowest: start it first
    jobs.sort(key=lambda j: (j[2] != "ismcts", j[2] != "pimc4"))
    print(f"{len(jobs)} probe searches, {a.workers} workers", flush=True)
    pool = BridgePool(a.workers, prefix=f"sb_{out.name}_", heap=a.heap, timeout=7200)
    t0 = time.time()
    try:
        with open(path, "a") as f, ThreadPoolExecutor(a.workers) as ex:
            def one(job):
                p, w, m, s = job
                row = {"probe": p, "world": w, "method": m, "seed": s, "budget": a.budget, "evaluator": a.evaluator}
                try:
                    specs, opts = request(m, P[p][w], s, a.budget)
                    if a.evaluator == "remote":
                        opts["evaluator"] = {"type": "remote", "port": int(a.port)}
                    ts = time.time()
                    r = pool.request("bench", None, specs=specs, **opts)
                    row.update(best=r.get("best"), children=r.get("children"), hiddenNames=r.get("hiddenNames"),
                               worldInfo=r.get("worldInfo"), stats=r.get("stats"), wall_s=round(time.time() - ts, 2))
                except Exception as e:  # noqa: BLE001
                    row["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
                return row
            futs = [ex.submit(one, j) for j in jobs]
            for k, fu in enumerate(as_completed(futs), 1):
                f.write(json.dumps(fu.result()) + "\n")
                f.flush()
                if k % 50 == 0:
                    print(f"  {k}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    finally:
        pool.close()
    return 0


def _fisher_2xk(a: Counter, b: Counter) -> float:
    """Exact-enough test that two samples of chosen actions share a distribution: a permutation
    test on the chi-square statistic (Fisher's exact test is 2x2 only in scipy)."""
    import random
    labels = sorted(set(a) | set(b))
    xs = [l for l in labels for _ in range(a[l])] + [l for l in labels for _ in range(b[l])]
    na = sum(a.values())

    def chi(sample_a: list, sample_b: list) -> float:
        ca, cb = Counter(sample_a), Counter(sample_b)
        n = len(sample_a) + len(sample_b)
        s = 0.0
        for l in labels:
            tot = ca[l] + cb[l]
            for c, m in ((ca[l], len(sample_a)), (cb[l], len(sample_b))):
                e = tot * m / n
                s += (c - e) ** 2 / e if e > 0 else 0.0
        return s
    obs = chi(xs[:na], xs[na:])
    if obs == 0:
        return 1.0
    rng = random.Random(0)
    hit = 0
    for _ in range(4000):
        rng.shuffle(xs)
        hit += chi(xs[:na], xs[na:]) >= obs - 1e-12
    return (hit + 1) / 4001


def cmd_analyze(a) -> int:
    rows = [json.loads(line) for line in open(Path(a.out) / "probes.jsonl")]
    rows = [r for r in rows if not r.get("error")]
    by = defaultdict(dict)
    for r in rows:
        by[(r["probe"], r["method"], r["budget"], r["evaluator"])][(r["world"], r["seed"])] = r
    report = []
    for (p, m, bud, ev), d in sorted(by.items()):
        seeds = sorted({s for (w, s) in d if ("X", s) in d and ("Y", s) in d})
        if not seeds:
            continue
        diffs = defaultdict(list)
        chosen = {"X": Counter(), "Y": Counter()}
        for s in seeds:
            qx = {c["label"]: c.get("Q") for c in d[("X", s)]["children"] or []}
            qy = {c["label"]: c.get("Q") for c in d[("Y", s)]["children"] or []}
            for lab in set(qx) & set(qy):
                if qx[lab] is not None and qy[lab] is not None:
                    diffs[lab].append(qx[lab] - qy[lab])
            chosen["X"][d[("X", s)]["best"]] += 1
            chosen["Y"][d[("Y", s)]["best"]] += 1
        worst, ok_q = None, True
        per = {}
        for lab, xs in diffs.items():
            n = len(xs)
            mu = sum(xs) / n
            sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
            half = 1.96 * sd / math.sqrt(n) if n > 1 else float("inf")
            lo, hi = mu - half, mu + half
            per[lab] = {"n": n, "dQ": round(mu, 4), "ci": [round(lo, 4), round(hi, 4)]}
            inside = lo >= -0.1 and hi <= 0.1
            ok_q &= inside
            if worst is None or abs(mu) > abs(per[worst]["dQ"]):
                worst = lab
        pval = _fisher_2xk(chosen["X"], chosen["Y"])
        canary = None
        if p == "canary":
            canary = sum("Counterspell" in (d[(w, s)].get("hiddenNames") or []) for s in seeds for w in ("X",))
        passed = ok_q and pval > 0.01 and (canary in (None, 0))
        report.append({"probe": p, "method": m, "budget": bud, "evaluator": ev, "seeds": len(seeds),
                       "pass": passed, "q_inside": ok_q, "fisher_p": round(pval, 4), "canary_hits": canary,
                       "worst": worst, "options": per, "chosen": {k: dict(v) for k, v in chosen.items()}})
    (Path(a.out) / "leak_report.json").write_text(json.dumps(report, indent=1))
    verdict = defaultdict(lambda: True)
    for r in report:
        verdict[(r["method"], r["evaluator"])] &= r["pass"]
        w = r["options"].get(r["worst"]) or {}
        print(f"{r['method']:12s} {r['probe']:15s} {r['evaluator']:7s} seeds {r['seeds']:3d}  "
              f"{'PASS' if r['pass'] else 'FAIL'}  worst {str(r['worst'])[:22]:22s} dQ {w.get('dQ')} ci {w.get('ci')}  "
              f"p {r['fisher_p']}  chosen X {r['chosen']['X']} Y {r['chosen']['Y']}"
              + (f"  canary {r['canary_hits']}" if r["canary_hits"] is not None else ""))
    print({f"{m}/{e}": ("passes" if v else "fails") for (m, e), v in verdict.items()})
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--heap", default="3g")
    r.add_argument("--seeds", type=int, default=16)
    r.add_argument("--budget", type=int, default=3000)
    r.add_argument("--methods", default=None)
    r.add_argument("--probes", default=None)
    r.add_argument("--evaluator", choices=("offline", "remote"), default="offline")
    r.add_argument("--port", default="50052")
    z = sub.add_parser("analyze")
    z.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
