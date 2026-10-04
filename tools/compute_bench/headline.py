"""docs/020's headline numbers from the compute benchmark's results (runs/compute_bench/<tag>/), one row per machine.

    python tools/compute_bench/headline.py runs/compute_bench/{3090-secure,4090-secure-64,...} --ref 3090-secure \\
        [--json rows.json]

Self-play games an hour at B simulations = workers x 3600 / (D x B x t_B):
  D      searched decisions in a game (both seats): the reference machine's finished 100-simulation games (seeded
         games replay the same on every machine, so D is shared; it's the mean of the games that finished, which
         leaves out the longest ones: a few percent high in games an hour)
  t_100  the machine's own worker-seconds a simulation in its 100-simulation games (searchSeconds / sims), at full load
  t_1000 the reference machine's mean worker-seconds a simulation on sb-v2's decisions at 1,000, times this machine's
         time on the same decisions over the reference's (sum over the decisions both ran)
  t_10000 t_1000 times the reference's ratio of 10,000 to 1,000 (flat: 64.4 against 64.0 ms on the 3090)
  workers: the games' at 100; sb's at 1,000 and 10,000 (t_1000 was measured with that many busy)
Training: the trainer's samples a second, imitation recipe (configs/compute_bench_human.yml), the best CPU thread
count and the GPU; and its share of a self-play machine's time when each game's decisions are trained on R times
(stage 6: 8 passes over the self-play rows).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import summarize  # noqa: E402

SB = "sb/decisions/ismcts-b{b}-remote-d0.99-pri-oppu-pw-netmlpd1.jsonl"


def items(d: Path, b: int) -> dict:
    p = d / SB.format(b=b)
    if not p.exists():
        return {}
    rows = (json.loads(x) for x in p.read_text().splitlines() if x.strip())
    return {r["item_id"]: r for r in rows if not r.get("error")}


def ratio(d: Path, ref: Path, b: int) -> tuple[float | None, int]:
    a, r = items(d, b), items(ref, b)
    com = [i for i in a if i in r]
    if not com:
        return None, 0
    return sum(a[i]["wall_s"] for i in com) / sum(r[i]["wall_s"] for i in com), len(com)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--ref", default="3090-secure")
    ap.add_argument("--replay", type=float, default=8.0)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    ref = next(d for d in a.dirs if d.name == a.ref)
    rs = summarize.summarize(ref)
    g_ref = rs["games_b100"]
    D = g_ref["decisions_per_game"]
    ref1000 = items(ref, 1000)
    t1000_ref = sum(x["wall_s"] for x in ref1000.values()) / sum(x["stats"]["sims"] for x in ref1000.values())
    ref10k = items(ref, 10000)
    k10 = (sum(x["wall_s"] for x in ref10k.values()) / sum(x["stats"]["sims"] for x in ref10k.values())) / t1000_ref \
        if ref10k else 1.0
    rows = []
    for d in a.dirs:
        s = summarize.summarize(d)
        si = s.get("sysinfo") or {}
        price = si.get("price_usd_hr")
        for gk in [k for k in s if k.startswith("games_b100") and isinstance(s[k], dict) and s[k].get("games")]:
            g = s[gk]
            W = g["setting"]["workers"]
            t100 = g["ms_per_sim"] / 1000
            r1000, n1000 = ratio(d, ref, 1000)
            t1000 = t1000_ref * r1000 if r1000 else None
            row = {"tag": d.name, "run": gk, "cpu": si.get("cpu_model"), "gpu": (si.get("gpu") or "").split(",")[0] or None,
                   "price": price, "vcpu_sold": si.get("pod", {}).get("RUNPOD_CPU_COUNT"), "quota": si.get("cgroup_cores"),
                   "workers": W, "server": g["setting"]["server_device"], "games_finished": g["games"],
                   "t100_ms": round(t100 * 1000, 1), "t1000_ms": round(t1000 * 1000, 1) if t1000 else None,
                   "sb1000_items": n1000, "D": D}
            # at 1,000+ the worker count is sb's (t_1000 was measured with that many busy; memory can cap it lower)
            w_sb = ((s.get("sb") or {}).get("1000") or {}).get("workers") or W
            row["workers_sb"] = w_sb
            for b, t in ((100, t100), (1000, t1000), (10000, t1000 * k10 if t1000 else None)):
                if t:
                    gph = (W if b == 100 else w_sb) * 3600 / (D * b * t)
                    row[f"g{b}"] = round(gph, 2)
                    row[f"g{b}_per_usd"] = round(gph / price, 2) if price else None
            tr = {k: v for k, v in s.items() if k.startswith("train_human") and isinstance(v, dict)}
            cpu = [v for v in tr.values() if v.get("device") == "cpu"]
            row["train_cpu"] = max((v["samples_per_s"] for v in cpu), default=None)
            row["train_gpu"] = next((v["samples_per_s"] for v in tr.values() if v.get("device") in ("cuda", "mps")), None)
            if row["train_cpu"]:
                t_game = D * a.replay / row["train_cpu"]          # seconds of the machine per game's training
                for b in (100, 1000, 10000):
                    if row.get(f"g{b}"):
                        play = 3600 / row[f"g{b}"]
                        row[f"train_share_cpu_{b}"] = round(t_game / (play + t_game), 4)
            rows.append(row)
    print(json.dumps(rows, indent=1))
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
