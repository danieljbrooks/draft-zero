"""The standard figures of a run, drawn from its metrics log (contracts §3, "draftzero-metrics/1" lines) in the docs'
look, each in a light and a dark variant (docs/033 "Metrics, plots and experiment tracking").

    report("runs/smoke", "runs/smoke/report")     -> [.../learning-light.png, .../learning-dark.png, ...]
    report(store, out)  report("hf://buckets/o/b", out)  report(lines, out)  report("metrics.jsonl", out)

learning     every learn.* metric (holdout policy cross-entropy, value log loss, top-1 agreement, AUC, ...), one
             panel each, by version (the learner's step restarts with each version's training run; lines without
             a version but with a step, a training run's own log, go by step); a series per string tag (e.g.
             split=holdout).
strength     eval.score by version, one series per opponent tag, with the 95% interval (lo/hi, else +-1.96 se).
ratings      Bradley-Terry Elo by version (eval.elo lines when the evaluator logs them, else one fit over every
             eval.score line, eval.ratings), fixed opponents as level lines; whiskers are 95% intervals.
throughput   throughput.* (games and positions an hour, ms per searched decision) over hours since the first
             report, one line per machine, plus all machines together for the hourly rates.

A figure with no data is skipped. Colours are the validated categorical slots of the docs' palette in fixed
order (light and dark steps of the same hues); an opponent keeps its colour across runs (prev, sbtactical,
nonet, bot first). Text stays in ink colours; a single series has no legend (the title names it).
matplotlib is an optional dependency (extra `plots`); it is imported only when a figure is drawn.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

BASE = frozenset({"schema", "time", "run", "component", "machine", "name", "value", "version", "step", "n", "lo",
                  "hi"})
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7",
              "series": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835",
             "series": ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767")},
}
RC = {"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9.5,
      "axes.titlesize": 10, "axes.labelsize": 9.5, "legend.fontsize": 8.5}
OPPONENTS = ("prev", "sbtactical", "nonet", "bot")          # fixed colour slots 1-4 in every figure
LEARN = ("learn.policy_ce", "learn.value_logloss", "learn.top1_visits", "learn.value_auc")
THROUGHPUT = ("throughput.games_per_hour", "throughput.positions_per_hour", "throughput.ms_per_searched")
TITLES = {
    "learn.policy_ce": "Policy cross-entropy to the search (lower is better)",
    "learn.value_logloss": "Value log loss (lower is better)",
    "learn.top1_visits": "Top-1 agreement with the search (higher is better)",
    "learn.value_auc": "Value AUC (higher is better)",
    "throughput.games_per_hour": "Games an hour",
    "throughput.positions_per_hour": "Recorded positions an hour",
    "throughput.ms_per_searched": "Milliseconds per searched decision",
}
Z95 = 1.959964


# ------------------------------------------------------------------------------------------------------- reading

def load_metrics(src: Any) -> list[dict]:
    """Metric lines from: a list of line dicts; a .jsonl file; a run directory or local store root (metrics/**.jsonl
    and metrics.jsonl); a store object (list/read_bytes: metrics/ and metrics.jsonl); or a store spec with a
    scheme ("hf://...", opened with draftzero.loop.store.open_store). Lines without a name or a finite value and
    unparsable lines are skipped."""
    if isinstance(src, (list, tuple)):
        return [x for x in src if _ok(x)]
    if hasattr(src, "list") and hasattr(src, "read_bytes"):
        return _from_store(src)
    s = str(src)
    if "://" in s:
        from draftzero.loop.store import open_store
        return _from_store(open_store(s))
    p = Path(s)
    if p.is_file():
        files = [p]
    else:
        files = sorted((p / "metrics").rglob("*.jsonl")) if (p / "metrics").is_dir() else []
        files += [p / "metrics.jsonl"] if (p / "metrics.jsonl").is_file() else []
    out: list[dict] = []
    for f in files:
        out += _parse(f.read_text(errors="replace"))
    return out


def _from_store(store) -> list[dict]:
    out: list[dict] = []
    paths = [x for x in store.list("metrics") if x.endswith(".jsonl")]
    for path in sorted(paths) + ["metrics.jsonl"]:
        b = store.read_bytes(path)
        if b:
            out += _parse(b.decode(errors="replace"))
    return out


def _parse(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        if line.strip():
            try:
                x = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _ok(x):
                out.append(x)
    return out


def _ok(x) -> bool:
    v = x.get("value") if isinstance(x, dict) else None
    return isinstance(x, dict) and isinstance(x.get("name"), str) and isinstance(v, (int, float)) \
        and not isinstance(v, bool) and math.isfinite(v)


def _num(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _version(x) -> int | None:
    m = re.fullmatch(r"v?0*(\d+)", str(x)) if x is not None else None
    return int(m[1]) if m else None


def _epoch(line: dict) -> float | None:
    """Seconds since the epoch from a line's time (ISO 8601, naive = UTC, or a number)."""
    t = line.get("time")
    if isinstance(t, (int, float)) and not isinstance(t, bool):
        return float(t)
    if isinstance(t, str):
        try:
            d = datetime.fromisoformat(t.replace("Z", "+00:00"))
        except ValueError:
            return None
        return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp()
    return None


def _series_key(line: dict) -> str:
    """A line's series within its metric: its string-valued tags ("split=holdout"), "" without any."""
    return ", ".join(f"{k}={v}" for k, v in sorted(line.items()) if k not in BASE and isinstance(v, str))


def _opponent(line: dict) -> str | None:
    for k in ("opponent", "opp", "vs"):
        if isinstance(line.get(k), str):
            return line[k]
    return None


def _interval(line: dict) -> tuple[float | None, float | None]:
    lo, hi = _num(line.get("lo")), _num(line.get("hi"))
    se = _num(line.get("se"))
    if (lo is None or hi is None) and se is not None:
        lo, hi = line["value"] - Z95 * se, line["value"] + Z95 * se
    return lo, hi


# ------------------------------------------------------------------------------------------------------ ratings

def ratings_from_metrics(lines: Iterable[dict], *, anchor: str = "sbtactical") -> dict[str, tuple[float, float]]:
    """{player: (elo, se)}: the latest eval.elo line per player when there are any (player = its "player" tag, else
    v{version}; se = its "se" tag, else from lo/hi), else one Bradley-Terry fit over every eval.score line
    (player v{version} against its opponent tag, "prev" = the version before, weighted by the games: the
    "n_games" tag, else 2 x "n_pairs", else n, else 1)."""
    lines = list(lines)
    elo = [x for x in lines if x["name"] == "eval.elo"]
    if elo:
        out = {}
        for x in elo:
            v = _version(x.get("version"))
            player = x.get("player") or (f"v{v}" if v is not None else None)
            if player is None:
                continue
            se = _num(x.get("se"))
            if se is None and _num(x.get("lo")) is not None and _num(x.get("hi")) is not None:
                se = (x["hi"] - x["lo"]) / (2 * Z95)
            out[str(player)] = (float(x["value"]), se or 0.0)
        return out
    from draftzero.eval.ratings import fit_bt, summary_triples
    rows = []
    for x in lines:
        v, opp = _version(x.get("version")), _opponent(x)
        if x["name"] != "eval.score" or v is None or opp is None:
            continue
        n = _num(x.get("n_games")) or 2 * (_num(x.get("n_pairs")) or 0) or _num(x.get("n")) or 1
        rows.append({"version": v, "opponent": opp, "score": min(1.0, max(0.0, x["value"])), "n_games": n})
    return fit_bt(summary_triples(rows), anchor=anchor)


# -------------------------------------------------------------------------------------------------------- report

def report(run_dir_or_store: Any, out_dir: str | Path, *, themes: Iterable[str] = ("light", "dark"),
           anchor: str = "sbtactical") -> list[Path]:
    """Draws the standard figures into out_dir as {learning,strength,ratings,throughput}-{light,dark}.png and
    returns the paths written (figures without data are skipped)."""
    _matplotlib()
    lines = load_metrics(run_dir_or_store)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fit = ratings_from_metrics(lines, anchor=anchor)
    written = []
    for theme in themes:
        for kind, draw in (("learning", lambda p, t: learning(lines, p, t)),
                           ("strength", lambda p, t: strength(lines, p, t)),
                           ("ratings", lambda p, t: ratings(fit, p, t, anchor=anchor)),
                           ("throughput", lambda p, t: throughput(lines, p, t))):
            p = draw(out / f"{kind}-{theme}.png", theme)
            if p is not None:
                written.append(p)
    return written


def learning(lines: Iterable[dict], path: Path, theme: str = "light") -> Path | None:
    """learn.* by version (or step, without versions), one panel per metric (at most six), a series per string
    tag."""
    learn = [x for x in lines if x["name"].startswith("learn.")]
    names = [n for n in LEARN if any(x["name"] == n for x in learn)]
    names += sorted({x["name"] for x in learn} - set(names))
    names = names[:6]
    if not names:
        return None
    keys = sorted({_series_key(x) for x in learn if x["name"] in names})
    t = THEMES[theme]
    color = {k: t["series"][i % 8] for i, k in enumerate(keys)}
    ncols = 2 if len(names) == 4 else min(3, len(names))
    with _rc(t):
        fig, axes = _figure(t, len(names), ncols, 4.2, 3.0)
        for ax, name in zip(axes, names):
            pts = [x for x in learn if x["name"] == name]
            # a version's step is that version's own training run's (it starts again at 0), so a line with a
            # version is placed by its version; only lines without one (a training run's own log) go by step
            by_step = any(_version(x.get("version")) is None for x in pts) and \
                all(_num(x.get("step")) is not None for x in pts)
            series: dict[str, dict[float, float]] = defaultdict(dict)
            for x in pts:
                xv = _num(x.get("step")) if by_step else _version(x.get("version"))
                if xv is not None:
                    series[_series_key(x)][xv] = x["value"]
            for k in sorted(series):
                xs = sorted(series[k])
                ax.plot(xs, [series[k][v] for v in xs], color=color[k], lw=2, solid_capstyle="round",
                        marker="o" if len(xs) <= 40 else None, ms=4, mec=t["surface"], mew=1, label=k or "all")
            ax.set_title(TITLES.get(name, name), loc="left")
            ax.set_xlabel("training step" if by_step else "player version")
            if not by_step:
                _integer_x(ax)
        fig.suptitle("Learning curves", x=0.01, ha="left", color=t["ink"], fontsize=11)
        if len(keys) > 1:
            _fig_legend(fig, axes)
        return _save(fig, path, t, header=True)


def strength(lines: Iterable[dict], path: Path, theme: str = "light") -> Path | None:
    """eval.score by version against each opponent, 95% whiskers, the 50% line dashed."""
    pts: dict[str, dict[int, dict]] = defaultdict(dict)
    for x in lines:
        v = _version(x.get("version"))
        if x["name"] == "eval.score" and v is not None:
            pts[_opponent(x) or "?"][v] = x                      # the latest line per (opponent, version)
    if not pts:
        return None
    t = THEMES[theme]
    opps = _order(pts)
    color = _opponent_colors(opps, t)
    with _rc(t):
        fig, (ax,) = _figure(t, 1, 1, 7.6, 4.3)
        ax.axhline(50, color=t["muted"], lw=1, ls=(0, (4, 3)), zorder=1)
        k = len(opps)
        for i, opp in enumerate(opps):
            vs = sorted(pts[opp])
            dx = (i - (k - 1) / 2) * min(0.1, 0.5 / k)          # dodge the whiskers apart
            y = [100 * pts[opp][v]["value"] for v in vs]
            lo_hi = [_interval(pts[opp][v]) for v in vs]
            err = [[max(0.0, yy - 100 * lo) if lo is not None else 0 for yy, (lo, _) in zip(y, lo_hi)],
                   [max(0.0, 100 * hi - yy) if hi is not None else 0 for yy, (_, hi) in zip(y, lo_hi)]]
            ax.errorbar([v + dx for v in vs], y, yerr=err, color=color[opp], lw=2, marker="o", ms=5,
                        mec=t["surface"], mew=1, capsize=0, elinewidth=1.2, label=opp, zorder=3)
        _integer_x(ax)
        ax.set_xlabel("player version")
        ax.set_ylabel("score (%)")
        title = "Strength by version: paired score, 95% intervals"
        ax.set_title(title + (f", against {opps[0]}" if k == 1 else ""), loc="left")
        if k > 1:
            _legend(ax)
        return _save(fig, path, t)


def ratings(fit: dict[str, tuple[float, float]], path: Path, theme: str = "light", *,
            anchor: str = "sbtactical") -> Path | None:
    """The rating ladder: versions as one line with 95% whiskers, every other player as a level line."""
    versions = sorted((_version(p), e, s) for p, (e, s) in fit.items() if re.fullmatch(r"v\d+", p))
    fixed = sorted(((p, e, s) for p, (e, s) in fit.items() if not re.fullmatch(r"v\d+", p)), key=lambda r: -r[1])
    if not versions and not fixed:
        return None
    t = THEMES[theme]
    color = _opponent_colors([p for p, _, _ in fixed], t)       # versions take slot 1, as "prev" does
    pinned = anchor if anchor in fit else next((p for p, (e, s) in fit.items() if e == 0 and s == 0), None)
    with _rc(t):
        fig, (ax,) = _figure(t, 1, 1, 7.6, 4.3)
        for p, e, _ in fixed:
            ax.axhline(e, color=color[p], lw=1.2, ls=(0, (5, 3)), label=p, zorder=2)
        if versions:
            xs = [v for v, _, _ in versions]
            ax.errorbar(xs, [e for _, e, _ in versions], yerr=[Z95 * s for _, _, s in versions], color=t["series"][0],
                        lw=2, marker="o", ms=5, mec=t["surface"], mew=1, capsize=0, elinewidth=1.2,
                        label="player versions", zorder=3)
            _integer_x(ax)
            ax.set_xlabel("player version")
        else:
            ax.set_xticks([])
        ax.set_ylabel(f"rating (Elo points; {pinned} = 0)" if pinned else "rating (Elo points)")
        ax.set_title("Rating ladder: one Bradley-Terry fit over every evaluated match", loc="left")
        if len(fixed) + bool(versions) > 1:
            _legend(ax)
        return _save(fig, path, t)


def throughput(lines: Iterable[dict], path: Path, theme: str = "light") -> Path | None:
    """throughput.* over hours since the first report, one line per machine (+ all machines for hourly rates)."""
    tp = [x for x in lines if x["name"].startswith("throughput.") and _epoch(x) is not None]
    names = [n for n in THROUGHPUT if any(x["name"] == n for x in tp)]
    names = (names + sorted({x["name"] for x in tp} - set(names)))[:4]
    if not names:
        return None
    t0 = min(_epoch(x) for x in tp)
    machines = sorted({str(x.get("machine") or x.get("component") or "?") for x in tp})
    t = THEMES[theme]
    many = len(machines) > 8
    color = {m: (t["muted"] if many else t["series"][i]) for i, m in enumerate(machines)}
    with _rc(t):
        fig, axes = _figure(t, len(names), 2 if len(names) == 4 else len(names), 4.6, 3.1)
        for ax, name in zip(axes, names):
            series: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for x in tp:
                if x["name"] == name:
                    series[str(x.get("machine") or x.get("component") or "?")].append(((_epoch(x) - t0) / 3600,
                                                                                       x["value"]))
            for i, m in enumerate(sorted(series)):
                s = sorted(series[m])
                ax.plot([a for a, _ in s], [b for _, b in s], color=color[m], lw=1.2 if many else 1.8,
                        label=("each machine" if i == 0 else None) if many else m)
            if name.endswith("_per_hour") and len(series) > 1:
                tt, total = _total(series)
                ax.plot(tt, total, color=t["ink"], lw=2.2, label="all machines")
            ax.set_title(TITLES.get(name, name), loc="left")
            ax.set_xlabel("hours since the first report")
            ax.set_ylim(bottom=0)
        fig.suptitle("Throughput", x=0.01, ha="left", color=t["ink"], fontsize=11)
        if len(machines) > 1:
            _fig_legend(fig, axes)
        return _save(fig, path, t, header=True)


def _total(series: dict[str, list[tuple[float, float]]], bins: int = 40) -> tuple[list[float], list[float]]:
    """Sum over machines of each one's rate at each of `bins` times: its latest report while that is fresher than
    three of its median report intervals; before its first report, that report for one interval (a report covers
    the chunk before it)."""
    end = max(a for s in series.values() for a, _ in s)
    start = min(a for s in series.values() for a, _ in s)
    grid = [start + (end - start) * i / (bins - 1) for i in range(bins)] if end > start else [start]
    runs = []
    for s in series.values():
        s = sorted(s)
        gaps = sorted(b[0] - a[0] for a, b in zip(s, s[1:]))
        runs.append((s, gaps[len(gaps) // 2] if gaps else math.inf))
    total = []
    for g in grid:
        acc = 0.0
        for s, gap in runs:
            done = [(a, v) for a, v in s if a <= g + 1e-12]
            if done and g - done[-1][0] <= 3 * gap:
                acc += done[-1][1]
            elif not done and s[0][0] - g <= gap:
                acc += s[0][1]
        total.append(acc)
    return grid, total


# --------------------------------------------------------------------------------------------------- the look

def _matplotlib():
    try:
        import matplotlib  # noqa: F401
    except ImportError as e:
        raise ImportError("draftzero.plots needs matplotlib: pip install 'draftzero[plots]'") from e
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # Helvetica falls back to DejaVu Sans


def _rc(t: dict):
    _matplotlib()
    import matplotlib
    return matplotlib.rc_context({**RC, "text.color": t["ink"], "axes.titlecolor": t["ink"],
                                  "axes.labelcolor": t["ink2"], "legend.labelcolor": t["ink"]})


def _figure(t: dict, n: int, ncols: int, w: float, h: float):
    from matplotlib.figure import Figure   # no pyplot: no global figure state, any backend
    nrows = math.ceil(n / ncols)
    fig = Figure(figsize=(w * ncols, h * nrows), dpi=150, facecolor=t["surface"])
    axes = [fig.add_subplot(nrows, ncols, i + 1) for i in range(n)]
    for ax in axes:
        ax.set_facecolor(t["surface"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(t["axis"])
        ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], labelsize=8.5)
        ax.grid(True, color=t["grid"], lw=0.8)
        ax.set_axisbelow(True)
        ax.xaxis.label.set_color(t["ink2"])
        ax.yaxis.label.set_color(t["ink2"])
        ax.title.set_color(t["ink"])
    return fig, axes


def _legend(ax) -> None:
    ax.legend(frameon=False, loc="best")


def _fig_legend(fig, axes) -> None:
    """One legend for a figure of panels, top right beside the figure title, clear of the data."""
    seen: dict[str, Any] = {}
    for ax in axes:
        for h, label in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(label, h)
    fig.legend(list(seen.values()), list(seen), loc="upper right", bbox_to_anchor=(0.995, 1.0), frameon=False,
               ncol=min(len(seen), 6))


def _integer_x(ax) -> None:
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _order(pts: dict) -> list[str]:
    return [o for o in OPPONENTS if o in pts] + sorted(o for o in pts if o not in OPPONENTS)


def _opponent_colors(names: list[str], t: dict) -> dict[str, str]:
    """Fixed slots for the standard opponents, the rest in sorted order after them; past eight, muted grey."""
    out, nxt = {}, len(OPPONENTS)
    for n in names:
        if n in OPPONENTS:
            out[n] = t["series"][OPPONENTS.index(n)]
    for n in sorted(x for x in names if x not in OPPONENTS):
        out[n] = t["series"][nxt] if nxt < 8 else t["muted"]
        nxt += 1
    return out


def _save(fig, path: Path, t: dict, header: bool = False) -> Path:
    """Lays out (leaving a strip for the figure title and legend when `header`) and writes the PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.1 / fig.get_figheight()) if header else None)
    fig.savefig(path, facecolor=t["surface"])
    return path
