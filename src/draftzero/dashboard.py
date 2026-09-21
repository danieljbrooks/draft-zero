"""dashboard.py — the format-knowledge page.

MageZero's report.py renders the generic training charts (throughput, losses, win rates,
resource use). It has no idea what a draft format is, and should not: whole-format
statistics are DraftZero's concern, so DraftZero renders them itself rather than patching
a template it does not own.

The page is standalone -- no build step, no CDN, no dependency on MageZero's internal
chart helpers, which would silently break the next time that template changes.

    python -m draftzero.dashboard runs/<run_id>
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from draftzero import stats as dzstats

OUT_NAME = "format.html"


def build_payload(run_dir: Path, meta_path: Optional[Path] = None,
                  reference_path: Optional[Path] = None) -> Optional[dict]:
    """Format statistics for a run, or None when the run has no games yet."""
    run_dir = Path(run_dir)
    games_file = run_dir / "games.jsonl"
    if not games_file.exists():
        return None
    games = [json.loads(l) for l in games_file.read_text().splitlines() if l.strip()]
    if not games:
        return None

    run = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").exists() else {}
    snap = run.get("run_config_snapshot", {})
    st = snap.get("stats", {})
    meta_path = Path(meta_path or snap.get("pools", {}).get("meta") or "assets/decks.tsv")
    reference_path = Path(reference_path or snap.get("reference") or "assets/reference/FDN_gih.json")

    meta = dzstats.load_deck_meta(meta_path) if meta_path.exists() else {}
    ref = dzstats.load_reference(reference_path if reference_path.exists() else None)
    s = dzstats.summarize(games, meta, ref, st.get("window_gens", 10), st.get("min_card_games", 15))
    s["run_id"] = run_dir.name
    s["stage"] = run.get("stage")
    s["current_gen"] = run.get("current_gen")
    s["completed"] = run.get("completed_at") is not None
    s["has_reference"] = bool(ref.get("cards"))
    return s


def render(run_dir: Path, out: Optional[Path] = None, index: bool = True, **kw) -> Optional[Path]:
    payload = build_payload(run_dir, **kw)
    if payload is None:
        return None
    run_dir = Path(run_dir)
    out = Path(out or run_dir / OUT_NAME)
    html = TEMPLATE.replace("__DATA__", json.dumps(payload, default=str).replace("</", "<\\/"))
    out.write_text(html)
    if index and out.parent == run_dir:
        write_index(run_dir, payload)
    return out


def write_index(run_dir: Path, payload: dict) -> Path:
    """A landing page linking both dashboards.

    MageZero renders dashboard.html and cannot know this page exists, so without an index
    the format view is only reachable by knowing its filename.
    """
    run_dir = Path(run_dir)
    sg = payload.get("selfplay_games", {})
    state = "complete" if payload.get("completed") else \
        f"in progress: gen {payload.get('current_gen')}, stage {payload.get('stage')}"
    idx = run_dir / "index.html"
    idx.write_text(INDEX_TEMPLATE
                   .replace("__RUN__", str(payload.get("run_id", run_dir.name)))
                   .replace("__STATE__", state)
                   # network covers gens 1+, heuristic is gen 0; "recent" is a WINDOW INSIDE
                   # network, so summing all three would count those games twice
                   .replace("__GAMES__", f"{sg.get('network', 0) + sg.get('heuristic', 0):,}")
                   .replace("__DECKS__", f"{payload.get('decks_seen', 0):,}"))
    return idx


INDEX_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__RUN__</title>
<style>
:root{--bg:#fff;--panel:#f7f7f8;--ink:#16161a;--muted:#6b6b76;--line:#e3e3e8;--accent:#3b6fd4}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#121216;--panel:#1a1a20;--ink:#ececf1;--muted:#9a9aa6;--line:#2a2a33;--accent:#7aa2f7}}
:root[data-theme="dark"]{--bg:#121216;--panel:#1a1a20;--ink:#ececf1;--muted:#9a9aa6;
  --line:#2a2a33;--accent:#7aa2f7}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:640px;margin:0 auto;padding:48px 16px}
h1{font-size:21px;margin:0 0 2px} p.sub{color:var(--muted);margin:0 0 24px}
a.card{display:block;background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:16px 18px;margin-bottom:10px;text-decoration:none;color:inherit}
a.card:hover{border-color:var(--accent)}
a.card .t{font-weight:600;color:var(--accent)} a.card .d{color:var(--muted);font-size:12.5px;margin-top:3px}
</style></head>
<body><div class="wrap">
<h1>__RUN__</h1>
<p class="sub">__STATE__ · __GAMES__ self-play games · __DECKS__ distinct decks</p>
<a class="card" href="dashboard.html">
  <div class="t">Training health</div>
  <div class="d">Win rates, losses, throughput, CPU/GPU/RAM. Rendered by MageZero.</div></a>
<a class="card" href="format.html">
  <div class="t">Format knowledge</div>
  <div class="d">Card GIH win rates against 17lands, rank correlation, deck colour records.</div></a>
</div></body></html>
"""


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Format knowledge</title>
<style>
:root{
  --bg:#ffffff; --panel:#f7f7f8; --ink:#16161a; --muted:#6b6b76; --line:#e3e3e8; --grid:#ededf1;
  --accent:#3b6fd4; --accent2:#c2410c; --good:#15803d;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#121216; --panel:#1a1a20; --ink:#ececf1; --muted:#9a9aa6; --line:#2a2a33; --grid:#24242c;
  --accent:#7aa2f7; --accent2:#f0883e; --good:#4ade80;
}}
:root[data-theme="dark"]{
  --bg:#121216; --panel:#1a1a20; --ink:#ececf1; --muted:#9a9aa6; --line:#2a2a33; --grid:#24242c;
  --accent:#7aa2f7; --accent2:#f0883e; --good:#4ade80;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:22px;margin:0 0 2px} h2{font-size:16px;margin:32px 0 6px}
h3{font-size:13px;margin:0;font-weight:600}
p.sub{color:var(--muted);margin:0 0 4px} p.blurb{color:var(--muted);margin:0 0 14px;max-width:72ch}
a{color:var(--accent)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:14px 0}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile .k{color:var(--muted);font-size:12px} .tile .v{font-size:23px;font-weight:600;margin-top:2px}
.tile .s{color:var(--muted);font-size:11px;margin-top:3px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.unit{color:var(--muted);font-size:11px;margin-bottom:6px}
svg{width:100%;height:auto;display:block;overflow:visible}
.tablewrap{max-height:430px;overflow:auto;border:1px solid var(--line);border-radius:10px;margin-top:8px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:6px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{position:sticky;top:0;background:var(--panel);cursor:pointer;font-weight:600;z-index:1}
tbody tr:hover{background:var(--grid)}
.empty{color:var(--muted);padding:22px 0;text-align:center}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:11px;margin-top:6px}
.sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:middle}
@media (max-width:640px){ .wrap{padding:20px 16px 48px} h1{font-size:19px} }
</style>
</head>
<body>
<div class="wrap">
  <h1 id="title">Format knowledge</h1>
  <p class="sub" id="subtitle"></p>
  <p class="sub"><a href="dashboard.html">← training health dashboard</a></p>
  <div id="root"></div>
</div>
<script>
const D = __DATA__;

const $ = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c;
  if (txt != null) e.textContent = txt; return e; };
const S = (t, a) => { const e = document.createElementNS("http://www.w3.org/2000/svg", t);
  for (const k in a) e.setAttribute(k, a[k]); return e; };
const pct = v => v == null ? "–" : (v * 100).toFixed(1) + "%";
const num = (v, d = 2) => v == null ? "–" : v.toFixed(d);
const root = document.getElementById("root");

document.getElementById("subtitle").textContent =
  `Run ${D.run_id} · ${D.completed ? "complete" : `in progress: gen ${D.current_gen}, stage ${D.stage}`}`;

// ── headline tiles ───────────────────────────────────────────
const tiles = $("div", "tiles");
const tile = (k, v, s) => { const t = $("div", "tile");
  t.append($("div", "k", k), $("div", "v", v)); if (s) t.append($("div", "s", s));
  tiles.append(t); };
const r = D.gih_rho, sg = D.selfplay_games;
tile("GIH rank corr., network", num(r.network), `${r.network_cards} cards · ${sg.network} self-play games`);
tile(`GIH rank corr., last ${D.window_gens} gens`, num(r.recent), `${r.recent_cards} cards · ${sg.recent} games`);
tile("GIH rank corr., heuristic gen 0", num(r.heuristic), `${r.heuristic_cards} cards · ${sg.heuristic} games`);
tile("Distinct decks played", (D.decks_seen || 0).toLocaleString(), "drawn at random from the pools");
root.append(tiles);

root.append($("p", "blurb",
  `Does the agent value cards and colours the way strong human players do? GIH ("game in hand") win rate is the ` +
  `win rate in games where a player drew the card, as 17lands defines it. These come from self-play, so the average ` +
  `is 50% by construction and only the ranking is comparable: ρ is the Spearman rank correlation with 17lands' GIH WR ` +
  `over cards seen in at least ${D.min_card_games} games. Gen 0 is heuristic search with no network — the baseline for ` +
  `how much card sense search alone has. Expect slow movement: each card needs hundreds of games.`));

// ── line chart: rho by generation ────────────────────────────
function lineChart(host, series) {
  const W = 480, H = 210, m = { l: 44, r: 12, t: 10, b: 26 };
  const all = [].concat(...series.map(s => s.points));
  if (!all.length) { host.append($("div", "empty", "no data yet")); return; }
  const gs = all.map(p => p.gen), vs = all.map(p => p.v);
  const x0 = Math.min(...gs), x1 = Math.max(...gs, x0 + 1);
  let y0 = Math.min(...vs, 0), y1 = Math.max(...vs, 0.1);
  const padY = (y1 - y0) * 0.12 || 0.05; y0 -= padY; y1 += padY;
  const X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
  const Y = v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
  const svg = S("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
                         "aria-label": "GIH rank correlation by generation" });
  for (let i = 0; i <= 4; i++) {
    const v = y0 + (y1 - y0) * i / 4;
    svg.append(S("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: "var(--grid)",
                           "vector-effect": "non-scaling-stroke" }));
    const t = S("text", { x: m.l - 7, y: Y(v) + 3.5, "text-anchor": "end", "font-size": 10,
                          fill: "var(--muted)" });
    t.textContent = v.toFixed(2); svg.append(t);
  }
  if (y0 < 0 && y1 > 0) svg.append(S("line", { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0),
    stroke: "var(--muted)", "stroke-dasharray": "3 3", "vector-effect": "non-scaling-stroke" }));
  const ticks = [...new Set([x0, Math.round((x0 + x1) / 2), x1])];
  for (const g of ticks) {
    const t = S("text", { x: X(g), y: H - 8, "text-anchor": "middle", "font-size": 10, fill: "var(--muted)" });
    t.textContent = `gen ${g}`; svg.append(t);
  }
  series.forEach((s, i) => {
    const col = i === 0 ? "var(--accent)" : "var(--accent2)";
    const pts = s.points.slice().sort((a, b) => a.gen - b.gen);
    if (pts.length > 1) {
      svg.append(S("path", { d: pts.map((p, j) => `${j ? "L" : "M"}${X(p.gen)},${Y(p.v)}`).join(" "),
        fill: "none", stroke: col, "stroke-width": 2, "vector-effect": "non-scaling-stroke" }));
    }
    for (const p of pts) {
      const c = S("circle", { cx: X(p.gen), cy: Y(p.v), r: 2.8, fill: col });
      const tt = S("title", {}); tt.textContent = `gen ${p.gen}: ρ=${p.v.toFixed(3)} (${p.cards} cards)`;
      c.append(tt); svg.append(c);
    }
  });
  host.append(svg);
  const leg = $("div", "legend");
  series.forEach((s, i) => {
    const e = $("span", null, null);
    const sw = $("span", "sw"); sw.style.background = i === 0 ? "var(--accent)" : "var(--accent2)";
    e.append(sw, document.createTextNode(s.name)); leg.append(e);
  });
  host.append(leg);
}

// ── scatter: agent vs 17lands ────────────────────────────────
function scatter(host) {
  const useNet = D.selfplay_games.network > 0;
  const key = useNet ? "wr" : "heuristic_wr", nkey = useNet ? "games" : "heuristic_games";
  const pts = (D.cards || []).filter(c => c[nkey] >= D.min_card_games && c.ref_wr != null && c[key] != null);
  host.append($("div", "unit",
    `${pts.length} cards with ≥${D.min_card_games} games · ${useNet ? "network" : "heuristic gen 0"} · hover for names`));
  if (!pts.length) { host.append($("div", "empty", "not enough games per card yet")); return; }
  const W = 480, H = 210, m = { l: 44, r: 12, t: 10, b: 30 };
  const xs = pts.map(p => p.ref_wr), ys = pts.map(p => p[key]);
  const x0 = Math.min(...xs) - .01, x1 = Math.max(...xs) + .01;
  const y0 = Math.min(...ys, .35) - .01, y1 = Math.max(...ys, .65) + .01;
  const X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
  const Y = v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
  const svg = S("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
                         "aria-label": "card GIH win rate, agent versus 17lands" });
  for (let i = 0; i <= 3; i++) {
    const v = y0 + (y1 - y0) * i / 3;
    svg.append(S("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: "var(--grid)",
                           "vector-effect": "non-scaling-stroke" }));
    const t = S("text", { x: m.l - 7, y: Y(v) + 3.5, "text-anchor": "end", "font-size": 10,
                          fill: "var(--muted)" });
    t.textContent = pct(v); svg.append(t);
  }
  for (const v of [x0 + .01, (x0 + x1) / 2, x1 - .01]) {
    const t = S("text", { x: X(v), y: H - 12, "text-anchor": "middle", "font-size": 10, fill: "var(--muted)" });
    t.textContent = pct(v); svg.append(t);
  }
  // y=x: where the agent would agree with 17lands exactly. The axes cover very different
  // ranges (agent per-card WR is far noisier), so this looks shallow and needs labelling or
  // it reads as a fitted trend line.
  const lo = Math.max(x0, y0), hi = Math.min(x1, y1);
  if (hi > lo) {
    svg.append(S("line", { x1: X(lo), y1: Y(lo), x2: X(hi), y2: Y(hi),
      stroke: "var(--muted)", "stroke-dasharray": "4 4", "vector-effect": "non-scaling-stroke" }));
    const lbl = S("text", { x: X(hi) - 3, y: Y(hi) - 5, "text-anchor": "end", "font-size": 9,
                            fill: "var(--muted)" });
    lbl.textContent = "agrees with 17lands"; svg.append(lbl);
  }
  for (const p of pts) {
    const c = S("circle", { cx: X(p.ref_wr), cy: Y(p[key]), r: 2.6,
                            fill: "var(--accent)", "fill-opacity": .65 });
    const tt = S("title", {});
    tt.textContent = `${p.card}: agent ${pct(p[key])} (${p[nkey]} games), 17lands ${pct(p.ref_wr)}`;
    c.append(tt); svg.append(c);
  }
  svg.append((() => { const t = S("text", { x: (m.l + W - m.r) / 2, y: H - 1, "text-anchor": "middle",
    "font-size": 10, fill: "var(--muted)" }); t.textContent = "x: 17lands GIH WR · y: agent GIH WR"; return t; })());
  host.append(svg);
}

// ── sortable table ───────────────────────────────────────────
function table(host, cols, rows, initial) {
  if (!rows.length) { host.append($("div", "empty", "nothing to show yet")); return; }
  const wrap = $("div", "tablewrap"), t = $("table"), head = $("tr"), body = document.createElement("tbody");
  let sortKey = initial || cols[0].k, asc = false;
  const draw = () => {
    const rs = rows.slice().sort((a, b) => {
      const x = a[sortKey], y = b[sortKey];
      if (typeof x === "string" || typeof y === "string")
        return String(x).localeCompare(String(y)) * (asc ? 1 : -1);
      return ((x ?? -Infinity) - (y ?? -Infinity)) * (asc ? 1 : -1);
    });
    body.replaceChildren(...rs.map(rw => {
      const tr = $("tr");
      cols.forEach(c => tr.append($("td", null, c.f ? c.f(rw) : (rw[c.k] ?? "–"))));
      return tr;
    }));
  };
  cols.forEach(c => {
    const th = $("th", null, c.h);
    th.onclick = () => { asc = sortKey === c.k ? !asc : false; sortKey = c.k; draw(); };
    head.append(th);
  });
  t.append(head, body); wrap.append(t); host.append(wrap); draw();
}

// ── assemble ─────────────────────────────────────────────────
const grid = $("div", "grid");

const c1 = $("div", "card");
c1.append($("h3", null, "GIH rank correlation with 17lands"), $("div", "unit", "Spearman ρ · cumulative from gen 1"));
const byName = {};
for (const p of (D.rho_series || [])) {
  if (p.v == null) continue;
  (byName[p.series] = byName[p.series] || []).push(p);
}
lineChart(c1, Object.entries(byName).map(([name, points]) => ({ name, points })));
grid.append(c1);

const c2 = $("div", "card");
c2.append($("h3", null, "Card GIH win rate vs 17lands"));
scatter(c2);
grid.append(c2);
root.append(grid);

if (!D.has_reference) {
  root.append($("p", "blurb",
    "No 17lands reference loaded, so ρ and the comparison columns are empty. " +
    "Build one with `python -m draftzero.reference`."));
}

root.append($("h2", null, "Cards"));
root.append($("p", "blurb",
  `Agent GIH win rate per card against the 17lands reference. Δ is agent minus 17lands: ` +
  `strongly negative means the agent under-uses a card humans win with. Click a header to sort.`));
table(root, [
  { k: "card", h: "Card" },
  { k: "games", h: "Games", f: r => r.games || 0 },
  { k: "wr", h: "Agent GIH", f: r => pct(r.wr) },
  { k: "ref_wr", h: "17lands GIH", f: r => pct(r.ref_wr) },
  { k: "delta", h: "Δ", f: r => r.delta == null ? "–" : (r.delta > 0 ? "+" : "") + (r.delta * 100).toFixed(1) },
  { k: "heuristic_wr", h: "Gen 0", f: r => pct(r.heuristic_wr) },
], (D.cards || []).filter(c => (c.games || 0) >= D.min_card_games || (c.heuristic_games || 0) >= D.min_card_games)
   .map(c => ({ ...c, delta: (c.wr != null && c.ref_wr != null) ? c.wr - c.ref_wr : null })), "games");

root.append($("h2", null, "Deck colours"));
root.append($("p", "blurb",
  "Win rate by main colours, with 95% Wilson intervals. Self-play averages 50% overall, so read these " +
  "as which colour pairs the agent handles better than others, not as absolute strength."));
table(root, [
  { k: "colors", h: "Colours" },
  { k: "games", h: "Games" },
  { k: "wr", h: "Win rate", f: r => pct(r.wr) },
  { k: "ci", h: "95% CI", f: r => (r.lo == null ? "–" : `${pct(r.lo)} – ${pct(r.hi)}`) },
  { k: "ref_wr", h: "17lands", f: r => pct(r.ref_wr) },
  { k: "heuristic_wr", h: "Gen 0", f: r => pct(r.heuristic_wr) },
], (D.colors || []), "games");
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="render the format-knowledge page for a run")
    ap.add_argument("run_dir")
    ap.add_argument("--out")
    ap.add_argument("--meta")
    ap.add_argument("--reference")
    a = ap.parse_args()
    p = render(Path(a.run_dir), Path(a.out) if a.out else None,
               meta_path=Path(a.meta) if a.meta else None,
               reference_path=Path(a.reference) if a.reference else None)
    if p is None:
        print("no games yet; nothing rendered", file=sys.stderr)
        raise SystemExit(1)
    print(p)
