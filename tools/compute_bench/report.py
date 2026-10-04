"""docs/020's tables from the compute benchmark's summaries (tools/compute_bench/summarize.py, one per machine).

    python tools/compute_bench/report.py runs/compute_bench/* [--machines machines.json] [--md out.md]

Per machine, from its own measurements:
  self-play at 100 simulations   games an hour, measured (the second half of the run, both seats searching)
  at 1,000 and 10,000            the 100-simulation games scaled by sb-v2's cost per decision at each budget on the
                                 same machine: games/h(B) = games/h(100) x pod_s(100) / pod_s(B)
  training                       samples a second from the trainer (configs/compute_bench_human.yml: imitation;
                                 _soft.yml: stage 6 on self-play tables), per hour and per dollar
  self-play + training           each game's searched decisions become training rows, each trained REPLAY times
                                 (stage 6: 8 epochs over the 65% training split, plus 15% human rows: ~6 per row);
                                 on the same machine the training time comes out of the game time

--machines adds rows measured elsewhere (r1's logs, earlier experiments): a JSON list of summary-shaped dicts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EPOCH_ROWS = 10_950_000          # the full-data MLP's training decisions (configs/exp4_train_mlp.yml)
REPLAY = 6.1                     # training samples per self-play decision in stage 6's recipe


def best(v: dict | None) -> float | None:
    if not v:
        return None
    return v.get("start_window") or v.get("window") or v.get("per_game")


def machine_rows(s: dict) -> dict:
    si = s.get("sysinfo") or {}
    price = si.get("price_usd_hr")
    row = {"tag": s["tag"], "price": price, "cpu": si.get("cpu_model"), "cores": si.get("cgroup_cores"),
           "mem_gb": si.get("cgroup_mem_gb"), "gpu": (si.get("gpu") or "").split(",")[0] or None}
    games = {k: v for k, v in s.items() if k.startswith("games_b") and isinstance(v, dict) and v.get("games")}
    sb = s.get("sb") or {}
    g100 = games.get("games_b100")
    if g100:
        gph = best(g100.get("games_per_hour"))
        row.update(g100=gph, g100_n=g100["games"], dec_per_game=g100["decisions_per_game"],
                   workers=g100["setting"]["workers"], server=g100["setting"]["server_device"],
                   ms_per_sim=g100.get("ms_per_sim"), cores_busy=(g100.get("load") or {}).get("cores_busy"),
                   jvm_gb=(g100.get("load") or {}).get("jvm_rss_gb_max"), gpu_util=(g100.get("load") or {}).get("gpu_util"))
        p100 = (sb.get("100") or {}).get("pod_s_per_decision")
        for b in ("1000", "10000"):
            pb = (sb.get(b) or {}).get("pod_s_per_decision")
            if p100 and pb:
                row[f"g{b}"] = gph * p100 / pb
    for b in ("100", "1000", "10000"):
        if sb.get(b):
            row[f"sb{b}"] = sb[b].get("pod_s_per_decision")
            row[f"sb{b}_workers"] = sb[b].get("workers")
    for kind in ("human", "soft"):
        cpu = [v for k, v in s.items() if k.startswith(f"train_{kind}_cpu") and isinstance(v, dict)]
        if cpu:
            top = max(cpu, key=lambda v: v["samples_per_s"])
            row[f"train_{kind}_cpu"] = top["samples_per_s"]
            row[f"train_{kind}_cpu_threads"] = top["threads"]
        gpu = s.get(f"train_{kind}_cuda")
        if gpu:
            row[f"train_{kind}_gpu"] = gpu["samples_per_s"]
    return row


def fmt(v, nd=1):
    if v is None:
        return "–"
    if isinstance(v, float):
        if abs(v) >= 100:
            return f"{v:,.0f}"
        return f"{v:.{nd}f}"
    return str(v)


def tables(rows: list[dict]) -> str:
    out = []
    out.append("| Machine | $/hr | Cores (cgroup) | Workers | Server | Task | Simulations | Games / hour | Games / $ |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        for b in ("100", "1000", "10000"):
            gph = r.get(f"g{b}")
            if gph is None:
                continue
            gpd = gph / r["price"] if r.get("price") else None
            out.append(f"| {r['tag']} | {fmt(r.get('price'), 2)} | {fmt(r.get('cores'), 0)} | {r.get('workers')} | "
                       f"{r.get('server')} | self-play | {int(b):,} | {fmt(gph)} | {fmt(gpd) if gpd else 'free'} |")
    out.append("")
    out.append("| Machine | $/hr | Device | Task | Samples / s | Hours / epoch (10.95M) | $ / epoch | M samples / $ |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        for kind, task in (("human", "imitation (17lands)"), ("soft", "stage 6 (self-play rows)")):
            for dev in ("cpu", "gpu"):
                sps = r.get(f"train_{kind}_{dev}")
                if not sps:
                    continue
                h = EPOCH_ROWS / sps / 3600
                p = r.get("price")
                dname = f"CPU, {r.get(f'train_{kind}_cpu_threads')} threads" if dev == "cpu" else (r.get("gpu") or "GPU")
                out.append(f"| {r['tag']} | {fmt(p, 2)} | {dname} | {task} | {fmt(sps)} | {fmt(h)} | "
                           f"{fmt(h * p, 2) if p else 'free'} | {fmt(sps * 3600 / p / 1e6, 2) if p else 'free'} |")
    out.append("")
    out.append("| Machine | $/hr | Simulations | Self-play games / hour | Training s / game (same machine) | "
               "Games / hour with training | Games / $ with training | Training share |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        sps = r.get("train_soft_cpu") or r.get("train_human_cpu")
        if not sps or not r.get("dec_per_game"):
            continue
        t_game = r["dec_per_game"] * REPLAY / sps
        for b in ("100", "1000", "10000"):
            gph = r.get(f"g{b}")
            if gph is None:
                continue
            eff = 1 / (1 / gph + t_game / 3600)
            share = 1 - eff / gph
            out.append(f"| {r['tag']} | {fmt(r.get('price'), 2)} | {int(b):,} | {fmt(gph)} | {fmt(t_game)} | {fmt(eff)} | "
                       f"{fmt(eff / r['price']) if r.get('price') else 'free'} | {share:.1%} |")
    return "\n".join(out)


def paired(dirs: list[Path], ref: str | None = None) -> str:
    """Seeded games replay identically on any machine (same decks, shuffles, network and search): the same game,
    with the same number of searched decisions, finished on several machines gives each machine's worker speed on
    identical work. Ratios of game seconds to the reference machine (median over the shared games)."""
    import statistics as st
    games = {}
    for d in dirs:
        for gd in sorted(d.glob("games_b*")):
            rows = [json.loads(x) for x in (gd / "games.jsonl").read_text().splitlines()] if (gd / "games.jsonl").exists() else []
            games[f"{d.name}/{gd.name}"] = {(g["pair"], g["swap"]): g for g in rows if not g.get("error")}
    if not games:
        return ""
    ref = ref or max(games, key=lambda k: len(games[k]))
    out = [f"| Run | Games shared with {ref} | Median game-seconds ratio to it | Decisions match |", "|---|---|---|---|"]
    for k, gs in games.items():
        shared = [t for t in gs if t in games[ref]]
        same = [t for t in shared if sum(s["decisions"] for s in gs[t]["seats"].values()) ==
                sum(s["decisions"] for s in games[ref][t]["seats"].values())]
        ratios = [gs[t]["seconds"] / games[ref][t]["seconds"] for t in same]
        out.append(f"| {k} | {len(shared)} | {st.median(ratios):.2f} | {len(same)} of {len(shared)} |" if ratios
                   else f"| {k} | {len(shared)} | – | {len(same)} of {len(shared)} |")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", type=Path)
    ap.add_argument("--machines", type=Path, default=None)
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    rows = []
    for d in a.dirs:
        p = d / "summary.json"
        if p.exists():
            rows.append(machine_rows(json.loads(p.read_text())))
    if a.machines:
        rows += json.loads(a.machines.read_text())
    md = tables(rows) + "\n\n" + paired(a.dirs)
    print(md)
    if a.md:
        a.md.write_text(md + "\n")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
