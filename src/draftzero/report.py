"""
report.py — render a run's metrics into a single self-contained HTML dashboard.

  python -m draftzero.report runs/<run_id>            -> runs/<run_id>/dashboard.html
  python -m draftzero.report runs/<run_id> --out x.html

Moved here from the MageZero fork (mz-engine, bcc76de) with the v0.2 migration.

Reads run.json, metrics.jsonl and games.jsonl (written by runner/train/test). The page has
no external dependencies, so it can be opened locally, attached, or published as-is.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def build_series(run: dict, rows: list[dict]) -> dict:
    """Long-form points: chart id -> series name -> [{gen, v, lo, hi}]"""
    charts: dict = defaultdict(lambda: defaultdict(list))

    def add(chart, series, gen, v, lo=None, hi=None):
        if v is None or gen is None:
            return
        charts[chart][series].append({"gen": gen, "v": v, "lo": lo, "hi": hi})

    # a resumed gen re-logs its rows; keep only the latest per (kind, gen, deck/agent, baseline, epoch)
    latest = {}
    for r in rows:
        key = (r.get("kind"), r.get("gen"), r.get("deck") or r.get("agent") or r.get("deck_a"), r.get("baseline"), r.get("epoch"))
        latest[key] = r
    rows = list(latest.values())

    last_epoch = {}
    for r in rows:
        k, gen = r.get("kind"), r.get("gen")
        if k == "selfplay":
            a, b = r["deck_a"], r["deck_b"]
            add("env_games", "completed", gen, r.get("games_completed"))
            add("env_games", "failed / timed out", gen, (r.get("games_failed") or 0) + (r.get("games_timed_out") or 0))
            add("env_gph", "self-play", gen, r.get("games_per_hour"))
            add("env_turns", "mean turns", gen, r.get("mean_turns"))
            add("env_sims", "sims / sec", gen, r.get("mean_mcts_sims_per_sec"))
            add("env_timeout", "search timeout rate", gen, r.get("mean_search_timeout_rate"))
            add("env_illegal", "illegal actions purged", gen, r.get("mean_illegals_purged"))
            for side, deck in (("a", a), ("b", b)):
                for metric in ("spells_cast_per_game", "lands_played_per_game", "attackers_declared_per_game",
                               "missed_land_drops_per_game", "idle_turns_per_game", "pass_with_play_rate",
                               "bad_target_rate"):
                    add(f"beh_{metric}", deck, gen, r.get(f"{side}_{metric}"))
            add("str_selfplay", a, gen, r.get("a_winrate"), r.get("a_winrate_lo"), r.get("a_winrate_hi"))
            add("str_onplay", "on the play", gen, r.get("on_play_winrate"))
        elif k == "strength_eval":
            base = r.get("baseline", "?")
            add(f"str_eval_{base}", r.get("agent", r.get("deck_a")), gen,
                r.get("a_winrate"), r.get("a_winrate_lo"), r.get("a_winrate_hi"))
        elif k == "train_epoch":
            last_epoch[(r["deck"], gen)] = r  # keep the final epoch of each gen
        elif k == "eval_prev_model":
            d = r["deck"]
            add("learn_prior_agree", d, gen, r.get("priority_A_acc"))
            add("learn_prior_value_corr", d, gen, r.get("value_corr"))
            add("learn_prior_target_agree", d, gen, r.get("choose_target_acc"))
        elif k == "dataset":
            d = r["deck"]
            add("data_states", d, gen, r.get("states"))
            add("data_onehot", d, gen, r.get("one_hot_policy_frac"))
            add("data_legal", d, gen, r.get("mean_legal_actions"))
        elif k == "gen_timing":
            add("env_gen_minutes", "whole generation", gen, (r.get("gen_seconds") or 0) / 60)
            add("env_gen_minutes", "self-play", gen, (r.get("generate_seconds") or 0) / 60)
        elif k == "resources":
            # CPU is charted against the cgroup quota, not the host: a container sees every
            # core on the machine, so raw percentages understate use by ~6x on a rented pod.
            add("res_cpu", "CPU mean (of quota)", gen, r.get("cpu_quota_percent_mean"))
            add("res_cpu", "CPU peak (of quota)", gen, r.get("cpu_quota_percent_max"))
            add("res_gpu", "GPU mean", gen, r.get("gpu_util_mean"))
            add("res_gpu", "GPU peak", gen, r.get("gpu_util_max"))
            add("res_mem", "container RAM", gen, r.get("container_mem_gb_mean"))
            add("res_mem", "container RAM peak", gen, r.get("container_mem_gb_max"))
            add("res_mem", "VRAM used", gen, r.get("gpu_mem_used_gb_mean"))

    for (deck, gen), r in sorted(last_epoch.items(), key=lambda x: x[0][1]):
        add("learn_value_loss", f"{deck} train", gen, r.get("train_value_loss"))
        add("learn_value_loss", f"{deck} held-out", gen, r.get("val_value_loss"))
        add("learn_policy_loss", deck, gen, r.get("train_priority_A_loss"))
        add("learn_entropy_target", deck, gen, r.get("val_priority_A_target_entropy"))
        add("learn_entropy_pred", deck, gen, r.get("val_priority_A_pred_entropy"))
        add("learn_grad", deck, gen, r.get("grad_norm"))
    return {c: {s: sorted(p, key=lambda x: x["gen"]) for s, p in ss.items()} for c, ss in charts.items()}


SECTIONS = [
    ("Actual strength", "Win rates with 95% Wilson intervals. Eval games put each new checkpoint against opponents that never change, so a rising line means real improvement, not both decks drifting together.", [
        ("str_eval_offline", "vs. search without a network", "win rate", "pct"),
        ("str_eval_frozen:0", "vs. opponent's frozen gen-0 net", "win rate", "pct"),
        ("str_eval_minimax", "vs. scripted minimax AI", "win rate", "pct"),
        ("str_selfplay", "Self-play matchup (both nets current)", "win rate", "pct"),
        ("str_onplay", "Player on the play wins", "rate", "pct"),
    ]),
    ("Learning", "Does the network absorb what search finds? Agreement is last gen's net predicting this gen's MCTS choices on fresh games.", [
        ("learn_prior_agree", "Priority agreement: prev net vs new search", "top-1 accuracy", "pct"),
        ("learn_prior_target_agree", "Target agreement: prev net vs new search", "top-1 accuracy", "pct"),
        ("learn_prior_value_corr", "Value prediction vs outcome (prev net)", "correlation", "num"),
        ("learn_value_loss", "Value loss", "MSE", "num"),
        ("learn_policy_loss", "Priority policy loss (train)", "weighted KL", "num"),
        ("learn_entropy_target", "Entropy of MCTS priority targets", "nats", "num"),
        ("learn_entropy_pred", "Entropy of network priority policy", "nats", "num"),
        ("learn_grad", "Mean gradient norm", "L2", "num"),
        ("data_states", "Training states collected", "states", "int"),
        ("data_onehot", "Decisions where search picked one move only", "fraction", "pct"),
        ("data_legal", "Legal actions per decision", "actions", "num"),
    ]),
    ("Behavior health", "Heuristics parsed from game logs. Early agents fail by doing nothing, so spells up and idle turns / passes-with-plays down is the healthy direction.", [
        ("beh_spells_cast_per_game", "Spells cast per game", "spells", "num"),
        ("beh_lands_played_per_game", "Lands played per game", "lands", "num"),
        ("beh_attackers_declared_per_game", "Attackers declared per game", "attackers", "num"),
        ("beh_pass_with_play_rate", "Own main phase: passed with a castable card", "rate", "pct"),
        ("beh_missed_land_drops_per_game", "Missed land drops per game", "turns", "num"),
        ("beh_idle_turns_per_game", "Idle turns (turn 5+, castable, did nothing)", "turns", "num"),
        ("beh_bad_target_rate", "Bad targets (removal on own / pump on theirs)", "rate", "pct"),
    ]),
    ("Environment health", "Is the simulator producing games reliably and fast enough?", [
        ("env_games", "Games per generation", "games", "int"),
        ("env_gph", "Throughput", "games / hour", "num"),
        ("env_gen_minutes", "Wall time per generation", "minutes", "num"),
        ("env_turns", "Game length", "turns (both players)", "num"),
        ("env_sims", "MCTS simulations per second", "sims / sec", "num"),
        ("env_timeout", "Searches cut off by timeout", "rate", "pct"),
        ("env_illegal", "Illegal actions purged per game", "count", "num"),
        ("res_cpu", "CPU utilization (share of quota)", "percent", "num"),
        ("res_gpu", "GPU utilization", "percent", "num"),
        ("res_mem", "Memory in use", "GB", "num"),
    ]),
]


def render(run_dir: Path, out: Path) -> Path:
    run = json.loads((run_dir / "run.json").read_text())
    rows = _load_jsonl(run_dir / "metrics.jsonl")
    games = _load_jsonl(run_dir / "games.jsonl")
    series = build_series(run, rows)
    snap = run.get("run_config_snapshot", {})
    baselines = snap.get("eval", {}).get("baselines", [])
    decks = [run["primary"]["deck"]] + [o["deck"] for o in run.get("opponents", [])]
    selfplay_games = [g for g in games if g.get("kind") == "selfplay"]
    notes_file = run_dir / "notes.md"
    notes = [l.strip("- ").strip() for l in notes_file.read_text().splitlines() if l.strip()] if notes_file.exists() else []
    payload = {
        "title": f"{' vs '.join(decks)}",
        "run_id": run_dir.name,
        "stage": run.get("stage"), "current_gen": run.get("current_gen"),
        "completed": run.get("completed_at") is not None,
        "generations": snap.get("generations"), "games_per_gen": snap.get("games_per_gen"),
        "decks": decks,
        "sections": [{"title": t, "blurb": b, "charts": [{"id": c, "title": ct, "unit": u, "fmt": f} for c, ct, u, f in cs
                                                         if not c.startswith("str_eval_") or c[len("str_eval_"):] in baselines or c in series]}
                     for t, b, cs in SECTIONS],
        "series": series,
        "totals": {
            "selfplay_games": len(selfplay_games),
            "eval_games": sum(1 for g in games if g.get("kind") == "strength_eval"),
            "gens_done": len(run.get("gens", {})),
        },
        "rows": rows,
        "notes": notes,
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(payload, default=str).replace("</", "<\\/"))
    out.write_text(html)
    return out


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MageZero Training Health</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
  --good: #0ca30c; --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 28px 16px 60px; }
header h1 { font-size: 22px; margin: 0 0 4px; font-weight: 650; }
header p { margin: 0; color: var(--text-secondary); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 20px 0 8px; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.tile .k { color: var(--text-secondary); font-size: 12px; }
.tile .v { font-size: 22px; font-weight: 600; margin-top: 2px; }
.tile .s { color: var(--muted); font-size: 12px; }
section { margin-top: 30px; }
section h2 { font-size: 17px; margin: 0 0 2px; font-weight: 650; }
section > p { margin: 0 0 12px; color: var(--text-secondary); max-width: 780px; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 0 0 12px; color: var(--text-secondary); font-size: 12px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 16px; height: 2px; border-radius: 1px; display: inline-block; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(340px, 100%), 1fr)); gap: 12px; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px 8px; position: relative; min-width: 0; }
.card h3 { font-size: 13px; font-weight: 600; margin: 0; }
.card .unit { color: var(--muted); font-size: 11px; margin-bottom: 4px; }
.card svg { width: 100%; height: 150px; display: block; overflow: visible; }
.empty { height: 150px; display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: 12px; }
.tip { position: absolute; pointer-events: none; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 8px; padding: 6px 8px; font-size: 12px; box-shadow: 0 4px 14px rgba(0,0,0,.12); display: none; z-index: 2; white-space: nowrap; }
.tip .row { display: flex; align-items: center; gap: 6px; }
.tip .row b { font-variant-numeric: tabular-nums; }
.tip .row i { width: 12px; height: 2px; display: inline-block; }
.tip .g { color: var(--muted); margin-bottom: 2px; }
details { margin-top: 30px; }
summary { cursor: pointer; color: var(--text-secondary); }
.tablewrap { overflow-x: auto; margin-top: 10px; background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; }
table { border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 8px; border-bottom: 1px solid var(--grid); text-align: left; white-space: nowrap; }
th { color: var(--text-secondary); font-weight: 600; position: sticky; top: 0; background: var(--surface-1); }
.status { display: inline-flex; align-items: center; gap: 6px; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
</style>
</head>
<body>
<main>
  <header>
    <h1 id="title"></h1>
    <p id="subtitle"></p>
  </header>
  <div class="tiles" id="tiles"></div>
  <div id="sections"></div>
  <details>
    <summary>All metric rows (table view)</summary>
    <div class="tablewrap" id="table"></div>
  </details>
</main>
<script>
const D = __DATA__;
const $ = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const NS = "http://www.w3.org/2000/svg";
const S = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };

// color follows the entity: decks keep one slot everywhere; other series take the next free slots
const slotOf = {}; let nextSlot = 1;
function colorFor(name) {
  const deck = D.decks.find(d => name === d || name.startsWith(d + " "));
  const key = deck || name;
  if (!(key in slotOf)) slotOf[key] = Math.min(nextSlot++, 3);
  return `var(--series-${slotOf[key]})`;
}
D.decks.forEach(colorFor);
const dashed = name => / held-out$/.test(name) || name === "self-play" || name === "failed / timed out";

function fmt(v, f) {
  if (v == null || Number.isNaN(v)) return "–";
  if (f === "pct") return (v * 100).toFixed(1) + "%";
  if (f === "int") return Math.round(v).toLocaleString();
  return Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3);
}

document.getElementById("title").textContent = `MageZero self-play: ${D.title}`;
document.getElementById("subtitle").textContent =
  `Run ${D.run_id} · ${D.completed ? "complete" : `in progress: gen ${D.current_gen}, stage ${D.stage}`}`;

const tiles = document.getElementById("tiles");
function tile(k, v, s) { const t = $("div", "tile"); t.append($("div", "k", k), $("div", "v", v)); if (s) t.append($("div", "s", s)); tiles.append(t); }
tile("Generations done", `${D.totals.gens_done} / ${D.generations ?? "?"}`, `${D.games_per_gen ?? "?"} self-play games per gen`);
tile("Self-play games", D.totals.selfplay_games.toLocaleString());
tile("Eval games", D.totals.eval_games.toLocaleString());
const gph = (D.series.env_gph || {})["self-play"];
tile("Throughput", gph ? fmt(gph[gph.length - 1].v, "num") : "–", "games / hour, latest gen");
for (const base of ["offline", "frozen:0"]) {
  const s = D.series["str_eval_" + base] || {};
  for (const deck of D.decks) {
    const pts = s[deck]; if (!pts || !pts.length) continue;
    const p = pts[pts.length - 1];
    tile(`${deck} vs ${base === "offline" ? "no-net search" : "frozen gen 0"}`, fmt(p.v, "pct"),
         `gen ${p.gen} · 95% CI ${fmt(p.lo, "pct")}–${fmt(p.hi, "pct")}`);
  }
}

function chart(card, spec) {
  const data = D.series[spec.id] || {};
  const names = Object.keys(data).filter(n => data[n].length);
  if (!names.length) { card.append($("div", "empty", "no data yet")); return; }
  const W = 320, H = 150, m = { l: 40, r: 10, t: 8, b: 20 };
  const all = names.flatMap(n => data[n]);
  const gens = [...new Set(all.map(p => p.gen))].sort((a, b) => a - b);
  const g0 = gens[0], g1 = Math.max(gens[gens.length - 1], g0 + 1);
  let lo = Math.min(...all.map(p => p.lo ?? p.v)), hi = Math.max(...all.map(p => p.hi ?? p.v));
  if (spec.fmt === "pct") { lo = Math.min(lo, 0); hi = Math.max(hi, spec.id.startsWith("str_") ? 1 : hi); }
  if (lo > 0 && lo < hi * 0.6) lo = 0;
  if (hi === lo) { hi = lo + 1; }
  const pad = (hi - lo) * 0.06; hi += pad; if (lo !== 0) lo -= pad;
  if (spec.id.startsWith("str_") && spec.fmt === "pct") { lo = 0; hi = 1; }
  const x = g => m.l + (g - g0) / (g1 - g0) * (W - m.l - m.r);
  const y = v => H - m.b - (v - lo) / (hi - lo) * (H - m.t - m.b);
  const svg = S("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", role: "img", "aria-label": spec.title });
  for (let i = 0; i <= 3; i++) {
    const v = lo + (hi - lo) * i / 3, yy = y(v);
    svg.append(S("line", { x1: m.l, x2: W - m.r, y1: yy, y2: yy, stroke: i === 0 ? "var(--axis)" : "var(--grid)", "stroke-width": 1, "vector-effect": "non-scaling-stroke" }));
    const t = S("text", { x: m.l - 6, y: yy + 3, "text-anchor": "end", "font-size": 10, fill: "var(--muted)" }); t.textContent = fmt(v, spec.fmt); svg.append(t);
  }
  if (spec.id.startsWith("str_") && spec.fmt === "pct") {
    svg.append(S("line", { x1: m.l, x2: W - m.r, y1: y(0.5), y2: y(0.5), stroke: "var(--axis)", "stroke-dasharray": "3 3", "stroke-width": 1, "vector-effect": "non-scaling-stroke" }));
  }
  const step = Math.max(1, Math.ceil(gens.length / 8));
  gens.forEach((g, i) => { if (i % step) return; const t = S("text", { x: x(g), y: H - 4, "text-anchor": "middle", "font-size": 10, fill: "var(--muted)" }); t.textContent = `g${g}`; svg.append(t); });
  for (const n of names) {
    const pts = data[n], c = colorFor(n);
    for (const p of pts) if (p.lo != null) {  // error bar per gen; the band only shows with 2+ gens
      svg.append(S("line", { x1: x(p.gen), x2: x(p.gen), y1: y(p.lo), y2: y(p.hi), stroke: c, "stroke-opacity": 0.45, "stroke-width": 2, "vector-effect": "non-scaling-stroke" }));
    }
    if (pts.some(p => p.lo != null)) {
      const band = pts.map(p => `${x(p.gen)},${y(p.hi)}`).concat(pts.slice().reverse().map(p => `${x(p.gen)},${y(p.lo)}`));
      svg.append(S("polygon", { points: band.join(" "), fill: c, "fill-opacity": 0.12 }));
    }
    svg.append(S("polyline", { points: pts.map(p => `${x(p.gen)},${y(p.v)}`).join(" "), fill: "none", stroke: c, "stroke-width": 2,
      "stroke-dasharray": dashed(n) ? "5 3" : "none", "vector-effect": "non-scaling-stroke", "stroke-linejoin": "round" }));
    for (const p of pts) svg.append(S("circle", { cx: x(p.gen), cy: y(p.v), r: 3, fill: c, stroke: "var(--surface-1)", "stroke-width": 1.5 }));
  }
  const cross = S("line", { y1: m.t, y2: H - m.b, stroke: "var(--axis)", "stroke-width": 1, "vector-effect": "non-scaling-stroke", visibility: "hidden" });
  svg.append(cross);
  card.append(svg);
  if (names.length > 1) {
    const lg = $("div", "legend");
    for (const n of names) { const s = $("span"); const i = $("i"); i.style.background = colorFor(n); s.append(i, document.createTextNode(n)); lg.append(s); }
    lg.style.margin = "4px 0 0"; card.append(lg);
  }
  const tip = $("div", "tip"); card.append(tip);
  const show = (evt) => {
    const r = svg.getBoundingClientRect();
    const px = (evt.clientX - r.left) / r.width * W;
    const g = gens.reduce((a, b) => Math.abs(x(b) - px) < Math.abs(x(a) - px) ? b : a);
    cross.setAttribute("x1", x(g)); cross.setAttribute("x2", x(g)); cross.setAttribute("visibility", "visible");
    tip.replaceChildren($("div", "g", `generation ${g}`));
    for (const n of names) {
      const p = data[n].find(q => q.gen === g); if (!p) continue;
      const row = $("div", "row"); const i = $("i"); i.style.background = colorFor(n);
      const val = $("b", null, fmt(p.v, spec.fmt));
      row.append(i, val, document.createTextNode(" " + n + (p.lo != null ? ` (${fmt(p.lo, spec.fmt)}–${fmt(p.hi, spec.fmt)})` : "")));
      tip.append(row);
    }
    tip.style.display = "block";
    const cr = card.getBoundingClientRect();
    let left = evt.clientX - cr.left + 12; if (left + tip.offsetWidth > cr.width) left = evt.clientX - cr.left - tip.offsetWidth - 12;
    tip.style.left = Math.max(0, left) + "px"; tip.style.top = (evt.clientY - cr.top + 12) + "px";
  };
  svg.addEventListener("pointermove", show);
  svg.addEventListener("pointerleave", () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); });
}

if (D.notes && D.notes.length) {
  const sec = $("section"); sec.append($("h2", null, "Run notes"));
  const ul = document.createElement("ul"); ul.style.cssText = "margin:6px 0 0;padding-left:18px;color:var(--text-secondary);max-width:820px";
  for (const n of D.notes) { const li = document.createElement("li"); li.style.marginBottom = "4px"; li.textContent = n; ul.append(li); }
  sec.append(ul); document.getElementById("sections").append(sec);
}
const root = document.getElementById("sections");
for (const sec of D.sections) {
  const el = $("section"); el.append($("h2", null, sec.title), $("p", null, sec.blurb));
  const grid = $("div", "grid");
  for (const spec of sec.charts) {
    const card = $("div", "card"); card.append($("h3", null, spec.title), $("div", "unit", spec.unit));
    chart(card, spec); grid.append(card);
  }
  el.append(grid); root.append(el);
}

// table view: every metric row, flattened
const keys = [...new Set(D.rows.flatMap(r => Object.keys(r)))].filter(k => k !== "ts" && k !== "log");
const order = ["kind", "gen", "deck", "deck_a", "deck_b", "agent", "baseline", "epoch"];
keys.sort((a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99));
const table = $("table"); const thead = $("tr"); keys.forEach(k => thead.append($("th", null, k))); table.append(thead);
for (const r of D.rows) {
  const tr = $("tr");
  keys.forEach(k => { const v = r[k]; tr.append($("td", null, v == null ? "" : typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(4)) : typeof v === "object" ? JSON.stringify(v) : String(v))); });
  table.append(tr);
}
document.getElementById("table").append(table);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rd = Path(args.run_dir)
    print(render(rd, Path(args.out) if args.out else rd / "dashboard.html"))
