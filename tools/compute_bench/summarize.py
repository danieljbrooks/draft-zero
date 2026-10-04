"""Summarise compute benchmarks (deploy/compute_bench.sh; docs/020): one machine's results directory, or a table of
several.

    python tools/compute_bench/summarize.py runs/compute_bench/<tag>            # summary.json + a printout
    python tools/compute_bench/summarize.py --table runs/compute_bench/*        # one row per machine

Games (self-play with search, one directory per setting): two throughput estimates that agree in steady state:
  start_window  workers x 3600 / the mean length of the games that started in the run's second half, every one of
                them finished (the run drains them with new games keeping the load): no warm-up, no truncation bias
  window    games finished in the run's second half / its length (inflated while the first round finishes)
  per_game  workers x 3600 / mean game seconds (biased towards short games when the run is cut early)
Search decisions (sb-v2): pod-seconds per decision as run.py measures it (wall / decisions: includes the start and
the last round's tail) and from the decisions' own times (mean worker-seconds / workers: steady state).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _mean(x):
    x = [v for v in x if v is not None]
    return statistics.fmean(x) if x else None


def load_summary(rows: list[dict], t0: float | None = None, skip_s: float = 120) -> dict:
    """Mean busy cores, memory peaks and GPU use over a phase's load samples, skipping its start."""
    if not rows:
        return {}
    start = (t0 or rows[0]["t"]) + skip_s
    steady = [r for r in rows if r["t"] >= start] or rows
    per_jvm = [r["jvm_rss_gb"] / r["jvms"] for r in steady if r.get("jvms")]
    return {"cores_busy": _mean(r.get("cores") for r in steady),
            "mem_gb_max": max((r.get("mem_gb") or 0) for r in rows),
            "anon_gb_max": max((r.get("anon_gb") or 0) for r in rows),
            "jvms": max((r.get("jvms") or 0) for r in rows),
            "jvm_rss_gb_mean": _mean(per_jvm), "jvm_rss_gb_max": max((r.get("jvm_rss_max_gb") or 0) for r in rows),
            "servers_rss_gb": max((r.get("servers_rss_gb") or 0) for r in rows),
            "gpu_util": _mean(r.get("gpu") for r in steady), "mhz": _mean(r.get("mhz") for r in steady)}


def games_summary(d: Path) -> dict:
    cfg = json.loads((d / "bench.json").read_text())
    t0 = float((d / "t0").read_text())
    t1 = float((d / "t1").read_text()) if (d / "t1").exists() else None
    games = _jsonl(d / "games.jsonl")
    ok = [g for g in games if not g.get("error")]
    W = cfg["workers"]
    # each finished game's start and finish (s after play.py started) from play.log's progress lines
    prog = [(float(m.group(1)), float(m.group(2))) for m in
            re.finditer(r"\((\d+(?:\.\d+)?) s\)\s+elapsed (\d+) s", (d / "play.log").read_text())] \
        if (d / "play.log").exists() else []
    fin = [f for _, f in prog]
    dur = (t1 - t0) if t1 else (max(fin) if fin else None)
    out = {"setting": cfg, "games": len(ok), "errors": len(games) - len(ok), "minutes": round(dur / 60, 1) if dur else None}
    if not ok:
        return out
    secs = [g["seconds"] for g in ok]
    seats = [s for g in ok for s in g["seats"].values()]
    dec = sum(s["decisions"] for s in seats)
    sims = sum(s["sims"] for s in seats)
    ssec = sum(s["searchSeconds"] for s in seats)
    out.update(
        game_s_mean=round(statistics.fmean(secs), 1), game_s_median=round(statistics.median(secs), 1),
        turns_mean=round(statistics.fmean(g["turns"] for g in ok), 1),
        no_winner=sum(g.get("winner") is None for g in ok),
        decisions_per_game=round(dec / len(ok), 1), sims_per_game=round(sims / len(ok)),
        single_option_per_game=round(sum(s["singleOption"] for s in seats) / len(ok), 1),
        net_evals_per_game=round(sum(s["netEvals"] for s in seats) / len(ok)),
        search_share=round(ssec / sum(secs), 3),            # of a game's worker time, the share spent searching
        ms_per_sim=round(1000 * ssec / sims, 2) if sims else None,   # worker time per simulation, under full load
        timeouts=sum(bool(s.get("timedOut")) for s in seats))
    rates = {"per_game": W * 3600 / statistics.fmean(secs)}
    if fin and dur:
        half = dur / 2
        n2 = sum(1 for f in fin if f >= half and f <= dur)
        rates["window"] = n2 / (dur - half) * 3600
    if prog and dur and (d / "t2").exists():
        # the start-window estimate: games that STARTED in the run's second half (past the JVMs' warm-up), all of
        # them finished (the run drains them, load kept up by new games), so long games count as often as short ones
        lens = [s_ for s_, f in prog if dur / 2 <= f - s_ <= dur]
        started = int((d / "started_by_t1").read_text()) if (d / "started_by_t1").exists() else None
        if lens:
            rates["start_window"] = W * 3600 / statistics.fmean(lens)
            out["start_window_games"] = len(lens)
            out["start_window_complete"] = started is not None and len(games) >= started
    out["games_per_hour"] = {k: round(v, 1) for k, v in rates.items()}
    out["decisions_per_s"] = round(W * dec / sum(secs), 2)
    out["sims_per_s"] = round(W * sims / sum(secs), 1)
    out["load"] = load_summary(_jsonl(d / "load.jsonl"), t0)
    return out


def sb_summary(d: Path) -> dict:
    runs = {r["run_id"]: r for r in _jsonl(d / "runs.jsonl")}
    # a run cut short by SB_MAX_MIN has decisions but no runs.jsonl row: its workers come from run.py's argv
    argv = (json.loads((d / "config.json").read_text()).get("argv") or []) if (d / "config.json").exists() else []
    w_argv = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else None
    for p in sorted((d / "decisions").glob("*.jsonl")):
        if p.stem not in runs:
            m = re.search(r"-b(\d+)-", p.stem)
            runs[p.stem] = {"run_id": p.stem, "budget": int(m.group(1)) if m else None, "workers": w_argv,
                            "wall_s": None, "pod_s_per_decision": None, "errors": None, "partial": True}
    out = {}
    for r in runs.values():
        rows = [x for x in _jsonl(d / "decisions" / f"{r['run_id']}.jsonl") if not x.get("error")]
        w = r["workers"]
        ws = [x["wall_s"] for x in rows]
        sims = [((x.get("stats") or {}).get("sims") or x.get("rootVisits") or 0) for x in rows]
        out[str(r["budget"])] = {
            "decisions": len(rows), "errors": r.get("errors"), "workers": w, "wall_s": r["wall_s"],
            "pod_s_per_decision_wall": r["pod_s_per_decision"],
            "pod_s_per_decision": round(statistics.fmean(ws) / w, 4) if ws and w else None,
            "partial": bool(r.get("partial")),
            "worker_s_per_decision": round(statistics.fmean(ws), 2) if ws else None,
            "ms_per_sim": round(1000 * sum(ws) / sum(sims), 2) if sum(sims) else None,
            "items": sorted(x["item_id"] for x in rows)[:3] + ["..."]}
    out["load"] = load_summary(_jsonl(d / "load_monitor.jsonl"))
    return out


def train_summary(d: Path) -> dict:
    s = json.loads((d / "summary.json").read_text())
    m = re.search(r"_t(\d+)$", d.name)
    cfg = s.get("config", {})
    return {"device": cfg.get("device"), "threads": int(m.group(1)) if m else None, "steps": s["step"],
            "samples": s["seen"], "train_time_s": s["train_time_s"], "samples_per_s": round(s["samples_per_s"], 1),
            "rows_per_step": round(s["seen"] / max(1, s["step"]), 1), "tables": [t["name"] for t in cfg.get("tables", [])]}


def summarize(out: Path) -> dict:
    res = {"tag": out.name}
    if (out / "sysinfo.json").exists():
        res["sysinfo"] = json.loads((out / "sysinfo.json").read_text())
    if (out / "cpuprobe.json").exists():
        res["cpuprobe"] = json.loads((out / "cpuprobe.json").read_text())
    for p in sorted(out.glob("infer_*.json")):
        res[p.stem] = [{k: r.get(k) for k in ("dtype", "batch", "evals_per_s", "ms_per_batch")}
                       for r in json.loads(p.read_text())["results"]]
    for d in sorted(out.glob("games_*")):
        if (d / "bench.json").exists():
            res[d.name] = games_summary(d)
    if (out / "sb" / "runs.jsonl").exists() or list((out / "sb" / "decisions").glob("*.jsonl")):
        res["sb"] = sb_summary(out / "sb")
    for d in sorted(out.glob("train_*")):
        if (d / "summary.json").exists():
            res[d.name] = train_summary(d)
    for p in sorted(out.glob("tstep_*.json")):
        res[p.stem] = json.loads(p.read_text())
    price = (res.get("sysinfo") or {}).get("price_usd_hr")
    if price:
        for k, v in list(res.items()):
            if k.startswith("games_") and v.get("games_per_hour"):
                gph = v["games_per_hour"].get("start_window") or v["games_per_hour"].get("window") \
                    or v["games_per_hour"]["per_game"]
                v["games_per_usd"] = round(gph / price, 1)
    return res


def table(dirs: list[Path]) -> None:
    hdr = ["tag", "cpu", "cores", "GB", "$/h", "W", "dev", "g/h b100", "g/$", "ms/sim", "sb100 pod-s", "sb1000 pod-s",
           "train cpu/s", "train gpu/s"]
    print(" | ".join(hdr))
    for d in dirs:
        if not d.is_dir():
            continue
        s = summarize(d)
        si = s.get("sysinfo", {})
        g = next((v for k, v in s.items() if k.startswith("games_b100") and isinstance(v, dict)), {})
        sb = s.get("sb", {})
        tc = max((v["samples_per_s"] for k, v in s.items() if k.startswith("train_") and v.get("device") == "cpu"), default=None)
        tg = max((v["samples_per_s"] for k, v in s.items() if k.startswith("train_") and v.get("device") == "cuda"), default=None)
        row = [s["tag"], (si.get("cpu_model") or "")[:28], si.get("cgroup_cores"), si.get("cgroup_mem_gb"),
               si.get("price_usd_hr"), (g.get("setting") or {}).get("workers"), (g.get("setting") or {}).get("server_device"),
               (g.get("games_per_hour") or {}).get("window"), g.get("games_per_usd"), g.get("ms_per_sim"),
               (sb.get("100") or {}).get("pod_s_per_decision"), (sb.get("1000") or {}).get("pod_s_per_decision"), tc, tg]
        print(" | ".join("" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in row))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args(argv)
    if a.table:
        table(a.dirs)
        return 0
    for d in a.dirs:
        s = summarize(d)
        (d / "summary.json").write_text(json.dumps(s, indent=1))
        print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
