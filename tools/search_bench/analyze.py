"""Score docs/012's first experiment: agreement with top 17lands players, per run, with
cluster-bootstrap CIs over games, pod-seconds per decision, and the frontier plot.

    python tools/search_bench/analyze.py --items data/search_bench/sb-v1 \\
        --runs runs/search_bench/e2_offline runs/search_bench/e2_network --leak runs/search_bench/e1 \\
        --out runs/search_bench/results [--plot docs/img/016-frontier]

Scores (docs/012 §2.3), each per decision type and macro-averaged over the types:
  A_set     the search's choice is in the human's label set (the headline)
  A_strict  the same, with the spell label's lenient Pass dropped
  A_soft    the share of the root's visits on the label set
References: chance (uniform over the distinct legal options) and the rule heuristic ("the
biggest spell; attack when power >= the best blocker's toughness; block when the blocker kills
the attacker and survives"), both computed here from the item's decision state.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TYPES = ("spell", "hold", "attack", "block")
METHOD_ORDER = ("clairvoyant", "pimc1", "pimc4", "ismcts")
LABEL = {"clairvoyant": "Clairvoyant MCTS", "pimc1": "PIMC, 1 world", "pimc4": "PIMC, 4 worlds",
         "ismcts": "IS-MCTS", "policy": "Policy network, no search"}


def load_items(path: Path) -> dict[str, dict]:
    with gzip.open(path / "items.jsonl.gz", "rt") as f:
        return {it["id"]: it for it in (json.loads(line) for line in f if line.strip())}


def load_rows(dirs: list[Path]) -> tuple[list[dict], list[dict]]:
    rows, runs = [], []
    for d in dirs:
        for p in sorted((d / "decisions").glob("*.jsonl")):
            for line in open(p):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not r.get("error"):
                    rows.append(r)
        if (d / "runs.jsonl").exists():
            runs += [json.loads(line) for line in open(d / "runs.jsonl") if line.strip()]
    # a decision rerun after a restart: keep the last row
    last = {}
    for r in rows:
        last[(r["run_id"], r["item_id"])] = r
    return list(last.values()), runs


# ------------------------------------------------------------------------------------ references

_MV: dict[str, float] = {}


def mana_value(name: str) -> float:
    if not _MV:
        with open(REPO / "data/17lands/cards.csv") as f:
            for row in csv.DictReader(f):
                try:
                    _MV.setdefault(row["name"], float(row["mana_value"]))
                except (TypeError, ValueError):
                    pass
    return _MV.get(name, _MV.get(name.split(" // ")[0], 0.0))


def chance(it: dict) -> float:
    legal = set(it["legal"])
    return len(set(it["label"]) & legal) / len(legal)


def _perms(state: dict, seat: str) -> list[dict]:
    return ((state or {}).get("players") or {}).get(seat, {}).get("battlefield") or []


def heuristic(it: dict) -> str | None:
    """The rule heuristic's choice at this decision (None when the item has no state dump)."""
    t, legal, st = it["type"], it["legal"], it.get("state")
    if t in ("spell", "hold"):
        casts = [lab for lab in legal if lab.startswith("Cast ")]
        if not casts:
            return "Pass"
        return max(casts, key=lambda lab: (mana_value(lab[5:]), lab))
    if st is None:
        return None
    subject = None
    text = it["decision"]["text"] or ""
    if t == "attack":
        subject = text[len("attack with: "):].rstrip("?")
        me = next((p for p in _perms(st, "A") if (p.get("x") or {}).get("name") == subject), None)
        if me is None:
            return None
        power = (me.get("x") or {}).get("power") or 0
        blockers = [(p.get("x") or {}).get("toughness") or 0 for p in _perms(st, "B") if (p.get("x") or {}).get("canBlock")]
        return "yes" if not blockers or power >= max(blockers) else "no"
    if t == "block":
        subject = text[len("choose which creature to block for "):].split(":Choose a target:")[0]
        me = next((p for p in _perms(st, "A") if (p.get("x") or {}).get("name") == subject), None)
        if me is None:
            return None
        mp, mt = (me.get("x") or {}).get("power") or 0, (me.get("x") or {}).get("toughness") or 0
        best = None
        for p in _perms(st, "B"):
            x = p.get("x") or {}
            if not x.get("attacking") or x.get("name") not in legal:
                continue
            if mp >= (x.get("toughness") or 0) and mt > (x.get("power") or 0):
                if best is None or (x.get("power") or 0) > (best.get("power") or 0):
                    best = x
        return best["name"] if best else "Stop Choosing"
    return None


# ------------------------------------------------------------------------------------ scoring

def score_row(r: dict, it: dict) -> dict:
    lab, strict = set(it["label"]), set(it["label_strict"])
    kids = r.get("children") or []
    tot = sum(c.get("N") or 0 for c in kids)
    soft = sum((c.get("N") or 0) for c in kids if c["label"] in lab) / tot if tot else None
    if r.get("method") == "policy":
        pri = {c["label"]: c.get("prior") or 0 for c in kids}
        s = sum(pri.values())
        soft = sum(v for k, v in pri.items() if k in lab) / s if s else None
    best = r.get("best")
    return {"set": float(best in lab), "strict": float(best in strict), "soft": soft}


def macro(values: list[tuple[str, float]]) -> float | None:
    by = defaultdict(list)
    for t, v in values:
        if v is not None:
            by[t].append(v)
    ms = [sum(v) / len(v) for v in by.values() if v]
    return sum(ms) / len(ms) if ms else None


def bootstrap(recs: list[dict], key: str, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the macro average, resampling games (a game's decisions go together)."""
    by_game = defaultdict(list)
    for x in recs:
        by_game[x["row"]].append(x)
    games = list(by_game)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample = [x for g in (rng.choice(games) for _ in games) for x in by_game[g]]
        m = macro([(x["type"], x[key]) for x in sample])
        if m is not None:
            vals.append(m)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def summarize(rows, runs, items, leak: dict | None) -> list[dict]:
    by_run = defaultdict(list)
    for r in rows:
        it = items.get(r["item_id"])
        if it is None:
            continue
        s = score_row(r, it)
        by_run[r["run_id"]].append({**s, "type": it["type"], "row": it["row"], "item": it["id"], "r": r})
    wall = defaultdict(float)
    ndec = defaultdict(int)
    for x in runs:
        wall[x["run_id"]] += x["wall_s"]
        ndec[x["run_id"]] += x["n"]
    out = []
    for rid, recs in sorted(by_run.items()):
        r0 = recs[0]["r"]
        per_type = {}
        for t in TYPES:
            v = [x["set"] for x in recs if x["type"] == t]
            if v:
                per_type[t] = {"n": len(v), "A_set": round(sum(v) / len(v), 4),
                               "A_soft": round(sum(x["soft"] for x in recs if x["type"] == t and x["soft"] is not None)
                                               / max(1, sum(1 for x in recs if x["type"] == t and x["soft"] is not None)), 4)}
        a_set = macro([(x["type"], x["set"]) for x in recs])
        lo, hi = bootstrap(recs, "set")
        stats = [x["r"].get("stats") or {} for x in recs]
        sims = [s.get("sims") or 0 for s in stats]
        eng = [s.get("engineSteps") or 0 for s in stats]
        e_v = sum(s.get("edgeVisits") or 0 for s in stats)
        p_v = sum(s.get("priorityEdgeVisits") or 0 for s in stats)
        t_v = sum(s.get("turnEdgeSum") or 0 for s in stats)
        method = r0["method"]
        verdict = None
        if leak is not None:
            verdict = leak.get(method)
        out.append({
            "run_id": rid, "method": method, "budget": r0["budget"], "evaluator": r0["evaluator"],
            "discount": r0.get("discount"), "unit": r0.get("unit"), "seed": r0.get("seed", 0),
            "n": len(recs), "A_set": round(a_set, 4) if a_set is not None else None, "ci": [round(lo, 4), round(hi, 4)],
            "A_strict": round(macro([(x["type"], x["strict"]) for x in recs]) or 0, 4),
            "A_soft": round(macro([(x["type"], x["soft"]) for x in recs]) or 0, 4),
            "per_type": per_type,
            "pod_s": round(wall[rid] / ndec[rid], 4) if ndec.get(rid) else None,
            "mean_wall_s": round(sum(x["r"].get("wall_s") or 0 for x in recs) / len(recs), 3),
            "sims_mean": round(sum(sims) / len(sims), 1),
            "engine_steps_per_sim": round(sum(eng) / max(1, sum(sims)), 2),
            "plies_per_action": round(e_v / p_v, 3) if p_v else None,
            "plies_per_turn": round(e_v / t_v, 3) if t_v else None,
            "consistent": round(sum(bool(x["r"].get("consistent")) for x in recs) / len(recs), 3),
            "hides": verdict,
        })
    return out


def references(items: dict, split: str | None) -> dict:
    its = [it for it in items.values() if split is None or it["split"] == split]
    ch = macro([(it["type"], chance(it)) for it in its])
    hs = [(it["type"], float(h in set(it["label"]))) for it in its if (h := heuristic(it)) is not None]
    return {"chance": round(ch, 4), "heuristic": round(macro(hs), 4) if hs else None, "n": len(its),
            "heuristic_n": len(hs),
            "chance_per_type": {t: round(sum(chance(it) for it in its if it["type"] == t)
                                         / max(1, sum(1 for it in its if it["type"] == t)), 4) for t in TYPES}}


def paired(rows, items, a: str, b: str, key: str = "set", n: int = 1000) -> dict:
    """Run a minus run b on their common decisions: the macro difference and its game-bootstrap CI."""
    A = {r["item_id"]: r for r in rows if r["run_id"] == a}
    B = {r["item_id"]: r for r in rows if r["run_id"] == b}
    common = [i for i in A if i in B and i in items]
    recs = [{"type": items[i]["type"], "row": items[i]["row"],
             "d": score_row(A[i], items[i])[key] - score_row(B[i], items[i])[key]} for i in common]
    d = macro([(x["type"], x["d"]) for x in recs])
    lo, hi = bootstrap(recs, "d", n)
    return {"a": a, "b": b, "n": len(recs), "diff": round(d, 4) if d is not None else None, "ci": [round(lo, 4), round(hi, 4)]}


# ------------------------------------------------------------------------------------ plot

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", series={"pimc4": "#2a78d6", "pimc1": "#eb6834", "ismcts": "#1baf7a"}),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", series={"pimc4": "#3987e5", "pimc1": "#d95926", "ismcts": "#199e70"}),
}


def plot(summary: list[dict], refs: dict, out_prefix: Path, title: str) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    paths = []
    main = [s for s in summary if s["method"] in METHOD_ORDER and s["discount"] == 0.99 and s["unit"] == "ply"
            and not s["seed"] and s["pod_s"]]
    ys = [s["A_set"] for s in main] + [refs["chance"]] + ([refs["heuristic"]] if refs.get("heuristic") else []) \
        + [s["A_set"] for s in summary if s["method"] == "policy" or s.get("priors")]
    lo_y = math.floor(min(ys) * 100 / 5) * 5 - 5
    hi_y = math.ceil(max(ys) * 100 / 5) * 5 + 5
    xs = [s["pod_s"] for s in main] or [0.1, 10]
    for mode, t in THEMES.items():
        plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10,
                             "xtick.color": t["muted"], "ytick.color": t["muted"],
                             "xtick.labelcolor": t["ink2"], "ytick.labelcolor": t["ink2"]})
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), dpi=160, sharex=True, sharey=True)
        fig.patch.set_facecolor(t["surface"])
        for ax, ev, ttl in zip(axes, ("offline", "remote"), ("Offline search (a heuristic scores positions)",
                                                             "Trained network: experiment #2a, gen 18")):
            ax.set_facecolor(t["surface"])
            ax.set_xscale("log")
            ax.set_xlim(min(xs) / 3, max(xs) * 4)
            ax.set_ylim(lo_y, hi_y)
            ax.grid(True, which="major", color=t["grid"], linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(t["axis"])
            ax.tick_params(which="minor", length=0)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
            ax.set_title(ttl, color=t["ink"], fontsize=11, loc="left", pad=10)
            ax.axhline(refs["chance"] * 100, color=t["muted"], linewidth=1, linestyle=(0, (4, 3)))
            ax.text(ax.get_xlim()[1] / 1.1, refs["chance"] * 100 + 0.4, "chance", color=t["ink2"], fontsize=8.5,
                    ha="right", va="bottom")
            if refs.get("heuristic") is not None:
                ax.axhline(refs["heuristic"] * 100, color=t["muted"], linewidth=1, linestyle=(0, (1, 2)))
                ax.text(ax.get_xlim()[1] / 1.1, refs["heuristic"] * 100 + 0.4, "rule heuristic, no search",
                        color=t["ink2"], fontsize=8.5, ha="right", va="bottom")
            ends = []
            for m in METHOD_ORDER:
                pts = sorted([s for s in main if s["method"] == m and s["evaluator"] == ev], key=lambda s: s["budget"])
                if not pts:
                    continue
                px, py = [s["pod_s"] for s in pts], [s["A_set"] * 100 for s in pts]
                lo = [s["A_set"] * 100 - s["ci"][0] * 100 for s in pts]
                hi = [s["ci"][1] * 100 - s["A_set"] * 100 for s in pts]
                fair = pts[0]["hides"] is True or (pts[0]["hides"] is None and m != "clairvoyant")
                c = t["series"].get(m, t["muted"]) if fair else t["muted"]
                ax.errorbar(px, py, yerr=[lo, hi], fmt="none", ecolor=c, elinewidth=1, alpha=0.5, capsize=0, zorder=2)
                ax.plot(px, py, color=c, linewidth=2, zorder=3)
                ax.plot(px, py, linestyle="none", marker="o", markersize=7,
                        markerfacecolor=c if fair else t["surface"], markeredgecolor=c if not fair else t["surface"],
                        markeredgewidth=2 if not fair else 1.5, zorder=4)
                ends.append([py[-1], px[-1], LABEL[m], t["ink"] if fair else t["ink2"]])
                if m == "pimc4":
                    for s, x_, y_ in zip(pts, px, py):
                        ax.annotate(f"{s['budget']:,}", (x_, y_), xytext=(0, -13), textcoords="offset points",
                                    color=t["ink2"], fontsize=7.5, ha="center")
            # end labels, pushed apart so none overlap (at least 1.3 points of agreement between them)
            ends.sort()
            for i in range(1, len(ends)):
                ends[i].append(None)
            ys_ = [e[0] for e in ends]
            for i in range(1, len(ys_)):
                ys_[i] = max(ys_[i], ys_[i - 1] + 1.3)
            for e, y_ in zip(ends, ys_):
                ax.annotate(e[2], (e[1], e[0]), xytext=(e[1] * 1.25, y_), textcoords="data", color=e[3], fontsize=9,
                            va="center", arrowprops=None)
            pol = [s for s in summary if s["method"] == "policy" and s["evaluator"] == ev and s["pod_s"]]
            for s in pol:
                ax.plot([s["pod_s"]], [s["A_set"] * 100], marker="s", markersize=8, color=t["ink2"], linestyle="none")
                ax.annotate("policy, no search", (s["pod_s"], s["A_set"] * 100), xytext=(0, -12),
                            textcoords="offset points", color=t["ink2"], fontsize=8.5, ha="center", va="top")
            ax.set_xlabel("Pod-seconds per decision (log scale)", color=t["ink2"])
        axes[0].set_ylabel("Agreement with top 17lands players (A_set, macro over types)", color=t["ink2"])
        handles = [Line2D([], [], color=t["series"]["ismcts"], marker="o", linewidth=2, label="passes the hidden-information test (colored, filled)"),
                   Line2D([], [], color=t["muted"], marker="o", markerfacecolor=t["surface"], linewidth=2,
                          label="fails it: reads hidden cards (gray, hollow)")]
        leg = fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
        for tx in leg.get_texts():
            tx.set_color(t["ink2"])
        fig.text(0.06, 0.955, title, color=t["ink"], fontsize=13, fontweight="bold", ha="left")
        fig.text(0.06, 0.925, "Each line: 100, 300, 1,000 and 3,000 simulations per decision, left to right; bars are "
                 "95% CIs over games. 1 pod-second = $0.00014.", color=t["ink2"], fontsize=10, ha="left")
        fig.subplots_adjust(left=0.07, right=0.93, top=0.86, bottom=0.17, wspace=0.12)
        p = Path(f"{out_prefix}-{mode}.png")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, facecolor=t["surface"])
        plt.close(fig)
        paths.append(p)
    return paths


def _style(ax, t):
    ax.set_facecolor(t["surface"])
    ax.grid(True, which="major", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
    ax.tick_params(which="minor", length=0)


def plot_types(summary, refs, items, out_prefix: Path) -> list[Path]:
    """Agreement by decision type against the budget: one panel per type, both evaluators."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    main = [s for s in summary if s["method"] in METHOD_ORDER and s["discount"] == 0.99 and s["unit"] == "ply"
            and not s["seed"] and not s.get("priors")]
    for mode, t in THEMES.items():
        fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), dpi=150, sharey=True)
        fig.patch.set_facecolor(t["surface"])
        for ax, ty in zip(axes, TYPES):
            _style(ax, t)
            ax.set_xscale("log")
            ax.set_title(f"{ty} (n = {sum(1 for it in items.values() if it['type'] == ty)})", color=t["ink"],
                         fontsize=11, loc="left")
            ax.axhline(refs["chance_per_type"][ty] * 100, color=t["muted"], linewidth=1, linestyle=(0, (4, 3)))
            for m in METHOD_ORDER:
                c = t["series"].get(m, t["muted"])
                for ev, ls in (("offline", "-"), ("remote", (0, (5, 2)))):
                    pts = sorted([s for s in main if s["method"] == m and s["evaluator"] == ev and ty in s["per_type"]],
                                 key=lambda s: s["budget"])
                    if pts:
                        ax.plot([s["budget"] for s in pts], [s["per_type"][ty]["A_set"] * 100 for s in pts], color=c,
                                linestyle=ls, marker="o", markersize=4, linewidth=1.6,
                                label=f"{LABEL[m]}, {'offline' if ev == 'offline' else 'network'}")
            ax.set_xticks([100, 300, 1000, 3000])
            ax.set_xticklabels(["100", "300", "1k", "3k"])
            ax.tick_params(colors=t["ink2"])
            ax.set_xlabel("Simulations per decision", color=t["ink2"])
        axes[0].set_ylabel("A_set (%)", color=t["ink2"])
        h, lab = axes[0].get_legend_handles_labels()
        leg = fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
        for tx in leg.get_texts():
            tx.set_color(t["ink2"])
        fig.text(0.05, 0.94, "Agreement by decision type (solid: offline search; dashed: network; gray dashes: chance)",
                 color=t["ink"], fontsize=12, fontweight="bold")
        fig.subplots_adjust(left=0.05, right=0.98, top=0.84, bottom=0.27, wspace=0.08)
        p = Path(f"{out_prefix}-{mode}.png")
        fig.savefig(p, facecolor=t["surface"])
        plt.close(fig)
        paths.append(p)
    return paths


def plot_discount(summary, out_prefix: Path) -> list[Path]:
    """E2b: agreement against the backprop discount and its unit, at 1,000 simulations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = [("1.0 / ply", 1.0, "ply"), ("0.99 / ply", 0.99, "ply"), ("0.95 / ply", 0.95, "ply"),
            ("0.9 / ply", 0.9, "ply"), ("0.95-matched\n/ action", None, "action"), ("0.95-matched\n/ turn", None, "turn")]
    rows = [s for s in summary if s["budget"] == 1000 and s["method"] in ("clairvoyant", "pimc4") and not s["seed"]
            and not s.get("priors")]
    if not any(s["discount"] != 0.99 for s in rows):
        return []
    paths = []
    for mode, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
        fig.patch.set_facecolor(t["surface"])
        _style(ax, t)
        k = 0
        for m in ("clairvoyant", "pimc4"):
            for ev, mk in (("offline", "o"), ("remote", "s")):
                xs, ys, lo, hi = [], [], [], []
                for i, (_, d, u) in enumerate(arms):
                    hit = [s for s in rows if s["method"] == m and s["evaluator"] == ev and s["unit"] == u
                           and (d is None or abs(s["discount"] - d) < 1e-9)]
                    if hit:
                        s = hit[0]
                        xs.append(i + (k - 1.5) * 0.08)
                        ys.append(s["A_set"] * 100)
                        lo.append((s["A_set"] - s["ci"][0]) * 100)
                        hi.append((s["ci"][1] - s["A_set"]) * 100)
                c = t["muted"] if m == "clairvoyant" else t["series"]["pimc4"]
                if xs:
                    ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, color=c, markersize=6, elinewidth=1, capsize=0,
                                markerfacecolor=c if ev == "offline" else t["surface"],
                                label=f"{LABEL[m]}, {'offline' if ev == 'offline' else 'network'}")
                k += 1
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([a[0] for a in arms], color=t["ink2"], fontsize=9)
        ax.tick_params(colors=t["ink2"])
        ax.set_ylabel("A_set (%), 1,000 simulations", color=t["ink2"])
        leg = ax.legend(frameon=False, fontsize=8.5, loc="lower left")
        for tx in leg.get_texts():
            tx.set_color(t["ink2"])
        ax.set_title("E2b: the backprop discount and its unit", color=t["ink"], fontsize=12, loc="left")
        fig.tight_layout()
        p = Path(f"{out_prefix}-{mode}.png")
        fig.savefig(p, facecolor=t["surface"])
        plt.close(fig)
        paths.append(p)
    return paths


def subdecision_check(rows, items) -> list[dict]:
    """E2b's free check (docs/012 §2.1): at 1,000 simulations, per discount arm, how much of the
    root's search goes to options that take one or more sub-decisions (a spell's target, the next
    attacker) before their action completes, over decisions offering both kinds of option."""
    by = defaultdict(list)
    for r in rows:
        if r.get("budget") != 1000 or r.get("method") not in ("clairvoyant", "pimc4") or r.get("seed") or r.get("priors"):
            continue
        if r["item_id"] not in items:
            continue
        kids = [c for c in r.get("children") or [] if c.get("sub") is not None and c["sub"] >= 0]
        multi = [c for c in kids if c["sub"] >= 1]
        single = [c for c in kids if c["sub"] == 0]
        if not multi or not single:
            continue
        tot = sum(c["N"] for c in kids) or 1
        best = next((c for c in kids if c["label"] == r.get("best")), None)
        by[(r["method"], r["evaluator"], r["discount"], r["unit"])].append(
            (sum(c["N"] for c in multi) / tot, float(best is not None and best["sub"] >= 1), r["item_id"]))
    out = []
    for (m, ev, d, u), xs in sorted(by.items()):
        out.append({"method": m, "evaluator": ev, "discount": d, "unit": u, "n": len(xs),
                    "share_multi": round(sum(x[0] for x in xs) / len(xs), 4),
                    "chose_multi": round(sum(x[1] for x in xs) / len(xs), 4)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default=str(REPO / "data/search_bench/sb-v1"))
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--leak", default=None, help="E1 output dir (leak_report.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--title", default="Agreement with top 17lands players against compute")
    ap.add_argument("--partial", action="store_true", help="include runs that have not finished")
    a = ap.parse_args(argv)
    items = load_items(Path(a.items))
    if a.split:
        items = {k: v for k, v in items.items() if v["split"] == a.split}
    rows, runs = load_rows([Path(d) for d in a.runs])
    leak = None
    if a.leak and (Path(a.leak) / "leak_report.json").exists():
        rep = json.loads((Path(a.leak) / "leak_report.json").read_text())
        leak = defaultdict(lambda: True)
        for r in rep:
            if r["evaluator"] == "offline":
                leak[r["method"]] &= r["pass"]
        leak = dict(leak)
    summary = summarize(rows, runs, items, leak)
    if not a.partial:
        summary = [x for x in summary if x["n"] >= len(items)]
    done = {x["run_id"] for x in summary}
    refs = references(items, None)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    contrasts = []
    for ev in ("offline", "remote"):
        for b in (100, 300, 1000, 3000, 10000):
            rid = lambda m: f"{m}-b{b}-{ev}-d0.99"
            for x, y in (("pimc1", "clairvoyant"), ("pimc4", "clairvoyant"), ("ismcts", "clairvoyant"),
                         ("pimc4", "pimc1"), ("ismcts", "pimc4"), ("ismcts", "pimc1")):
                if rid(x) in done and rid(y) in done:
                    contrasts.append(paired(rows, items, rid(x), rid(y)))
        for m in METHOD_ORDER:
            for lo, hi in ((100, 1000), (100, 3000), (1000, 3000), (300, 3000)):
                x, y = f"{m}-b{hi}-{ev}-d0.99", f"{m}-b{lo}-{ev}-d0.99"
                if x in done and y in done:
                    contrasts.append(paired(rows, items, x, y))
    for m in METHOD_ORDER:
        for b in (100, 300, 1000, 3000):
            x, y = f"{m}-b{b}-remote-d0.99", f"{m}-b{b}-offline-d0.99"
            if x in done and y in done:
                contrasts.append(paired(rows, items, x, y))
    for x in done:
        if "-s1" in x:
            base = x.replace("-s1", "")
            if base in done:
                contrasts.append(paired(rows, items, x, base))
    (out / "summary.json").write_text(json.dumps({"references": refs, "runs": summary, "contrasts": contrasts}, indent=1))
    print("references", json.dumps(refs))
    print(f"{'run':44s} {'n':>5s} {'A_set':>6s} {'95% CI':>15s} {'strict':>6s} {'soft':>6s} {'pod-s':>7s} "
          f"{'steps/sim':>9s} {'p/act':>6s} {'p/turn':>6s}  per type")
    for s in summary:
        pt = " ".join(f"{t[:2]} {v['A_set']:.2f}" for t, v in s["per_type"].items())
        print(f"{s['run_id']:44s} {s['n']:5d} {s['A_set']:6.3f} [{s['ci'][0]:.3f},{s['ci'][1]:.3f}] "
              f"{s['A_strict']:6.3f} {s['A_soft']:6.3f} {s['pod_s'] or 0:7.3f} {s['engine_steps_per_sim']:9.2f} "
              f"{s['plies_per_action'] or 0:6.2f} {s['plies_per_turn'] or 0:6.2f}  {pt}")
    sub = subdecision_check(rows, items)
    if sub:
        print("\nE2b sub-decision check (decisions offering options with and without sub-decisions, 1,000 sims):")
        for x in sub:
            print(f"  {x['method']:12s} {x['evaluator']:7s} {x['discount']:<7g} {x['unit']:7s} n {x['n']:4d}  "
                  f"visit share on multi-step options {x['share_multi']:.3f}  chose one {x['chose_multi']:.3f}")
        summ = json.loads((out / "summary.json").read_text())
        summ["subdecision_check"] = sub
        (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print("\npaired contrasts (a - b, macro A_set, 95% CI over games):")
    for c in contrasts:
        sig = "*" if c["ci"][0] > 0 or c["ci"][1] < 0 else " "
        print(f"  {sig} {c['a']:36s} - {c['b']:36s} n {c['n']:5d}  {c['diff']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]")
    if a.plot:
        for p in plot(summary, refs, Path(a.plot), a.title):
            print("plot", p)
        for p in plot_types(summary, refs, items, Path(str(a.plot).replace("frontier", "by-type"))):
            print("plot", p)
        for p in plot_discount(summary, Path(str(a.plot).replace("frontier", "discount"))):
            print("plot", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
