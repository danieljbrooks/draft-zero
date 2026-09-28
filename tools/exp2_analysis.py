"""
exp2_analysis.py — the numbers and figures behind docs/013 (experiment #2, run 1 against run 2).

Inputs are each run's own files, as the watchdog pushes them to Hugging Face:
games.jsonl and metrics.jsonl under <run dir>/. Deck lists come from the 17lands top-player
pool (data/deckgen), and the 17lands reference from the public game file (data/17lands).

    python tools/exp2_analysis.py reference            # 17lands GP / GIH / GNS / IWD per card (~80 s)
    python tools/exp2_analysis.py stats --run1 <dir> --run2 <dir> [--gens 1-12] [--boot 200]
    python tools/exp2_analysis.py figures --run1 <dir> --run2 <dir> --out docs/img [--plasticity <json>]

Card stats follow the run dashboards: both sides of a self-play game, only the current network's
side of a league game (player A). GIH = in hand (drawn, opening hand included); GNS = in the
deck, never seen; IWD = GIH - GNS; GP = in the deck. Basic lands are excluded.
17lands public data is CC BY 4.0 (https://www.17lands.com/public_datasets).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DECKS = REPO / "data" / "deckgen" / "FDN_PremierDraft_wr60" / "top_player_FDN_decks"
GAME_DATA = REPO / "data" / "17lands" / "game_data_public.FDN.PremierDraft.csv.gz"
CARDS_CSV = REPO / "data" / "17lands" / "cards.csv"
REF_OUT = REPO / "assets" / "reference" / "FDN_card_stats.json"
BASICS = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
STATS = ("gih_wr", "gns_wr", "iwd", "gp_wr")


# ── 17lands reference ─────────────────────────────────────────────

def build_reference(path: Path = GAME_DATA, out: Path = REF_OUT) -> dict:
    f = gzip.open(path, "rt")
    r = csv.reader(f)
    H = next(r)
    cards = [h[5:] for h in H if h.startswith("deck_")]
    di = [H.index("deck_" + c) for c in cards]
    oi = [H.index("opening_hand_" + c) for c in cards]
    dr = [H.index("drawn_" + c) for c in cards]
    wi = H.index("won")
    n = len(cards)
    acc = {k: np.zeros((n, 2)) for k in ("gp", "gih", "gns")}
    games = 0
    for row in r:
        a = np.fromiter((row[i] != "0" for i in di), bool, n)
        if not a.any():
            continue
        seen = np.fromiter(((row[i] != "0") or (row[j] != "0") for i, j in zip(oi, dr)), bool, n)
        w = row[wi] == "True"
        for k, m in (("gp", a), ("gih", a & seen), ("gns", a & ~seen)):
            acc[k][m, 0] += 1
            acc[k][m, 1] += w
        games += 1
    ref = {}
    for i, c in enumerate(cards):
        o = {}
        for k, v in acc.items():
            o[f"{k}_n"] = int(v[i, 0])
            o[f"{k}_wr"] = v[i, 1] / v[i, 0] if v[i, 0] else None
        o["iwd"] = o["gih_wr"] - o["gns_wr"] if o["gih_wr"] is not None and o["gns_wr"] is not None else None
        ref[c] = o
    out.write_text(json.dumps({"games": games, "source": path.name, "cards": ref}))
    return {"games": games, "cards": ref}


def load_reference() -> dict:
    return json.loads(REF_OUT.read_text())["cards"]


def card_info() -> dict:
    return {r["name"]: r for r in csv.DictReader(open(CARDS_CSV)) if r.get("expansion") == "FDN"}


# ── runs ──────────────────────────────────────────────────────────

_deck_cache: dict = {}


def deck(stem: str) -> set:
    if stem not in _deck_cache:
        cards = set()
        for line in open(DECKS / f"{stem}.dck"):
            m = re.match(r"\d+ \[[^\]]+\] (.+)", line.strip())
            if m:
                cards.add(m.group(1))
        _deck_cache[stem] = cards - BASICS
    return _deck_cache[stem]


def read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in open(path) if line.strip()]


def selfplay(games: list, lo: int = 1, hi: int = 10 ** 6) -> list:
    return [g for g in games if g["kind"] == "selfplay" and lo <= g["gen"] <= hi]


def _sides(g):
    for s in ("a", "b") if g["agent_side"] == "both" else ("a",):
        yield g[f"deck_{s}"], set(g.get(f"drawn_{s}") or []), g["winner"] == s.upper()


def card_stats(games: list) -> dict:
    acc = defaultdict(lambda: {"gp": [0, 0], "gih": [0, 0], "gns": [0, 0]})
    for g in games:
        for d, drawn, won in _sides(g):
            for c in deck(d):
                a = acc[c]
                a["gp"][0] += 1
                a["gp"][1] += won
                k = "gih" if c in drawn else "gns"
                a[k][0] += 1
                a[k][1] += won
    out = {}
    for c, a in acc.items():
        o = {f"{k}_n": v[0] for k, v in a.items()}
        o.update({f"{k}_wr": (v[1] / v[0] if v[0] else None) for k, v in a.items()})
        o["iwd"] = o["gih_wr"] - o["gns_wr"] if o["gih_wr"] is not None and o["gns_wr"] is not None else None
        out[c] = o
    return out


def spearman(x, y):
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def rho(st: dict, ref: dict, info: dict, key: str, min_n: int = 30, rarity: str | None = "common"):
    """Spearman rank correlation of a run's per-card stat with 17lands' (cards with >= min_n games
    in hand, and >= min_n not seen for GNS / IWD)."""
    xs, ys = [], []
    for c, o in st.items():
        if c not in ref or ref[c].get(key) is None or o.get(key) is None:
            continue
        if o["gih_n"] < min_n or (key in ("gns_wr", "iwd") and o["gns_n"] < min_n):
            continue
        if rarity and info.get(c, {}).get("rarity") != rarity:
            continue
        xs.append(o[key])
        ys.append(ref[c][key])
    return spearman(xs, ys), len(xs)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None, None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def metrics_rows(path: Path, kind: str) -> dict:
    """gen -> the last metrics row of that kind."""
    out = {}
    for d in read_jsonl(path):
        if d.get("kind") == kind:
            out[d["gen"]] = d
    return out


def league_by_gen(games: list) -> dict:
    out = defaultdict(lambda: [0, 0])
    for g in games:
        if g["kind"] == "selfplay" and g["agent_side"] == "a":
            out[g["gen"]][0] += g["winner"] == "A"
            out[g["gen"]][1] += 1
    return dict(out)


def evals(games: list) -> dict:
    out = defaultdict(lambda: [0, 0])
    for g in games:
        if g["kind"] == "strength_eval":
            out[g["gen"]][0] += g["winner"] == "A"
            out[g["gen"]][1] += 1
    return dict(out)


def rho_by_gen(games: list, ref: dict, info: dict, top: int, rarity="common", min_cards: int = 20) -> dict:
    """Cumulative GIH rho over gens 1..g, for each g (None while fewer than min_cards cards qualify)."""
    out = {}
    for g in range(1, top + 1):
        r, n = rho(card_stats(selfplay(games, 1, g)), ref, info, "gih_wr", rarity=rarity)
        out[g] = (r if n >= min_cards else None, n)
    return out


# ── stats command ─────────────────────────────────────────────────

def cmd_stats(a) -> None:
    ref, info = load_reference(), card_info()
    lo, hi = (int(x) for x in a.gens.split("-"))
    G = {n: selfplay(read_jsonl(Path(d) / "games.jsonl"), lo, hi) for n, d in (("run1", a.run1), ("run2", a.run2))}
    st = {n: card_stats(v) for n, v in G.items()}
    rng = np.random.default_rng(0)
    print(f"self-play games, gens {lo}-{hi}: " + ", ".join(f"{n} {len(v)}" for n, v in G.items()))
    for key in STATS:
        (r1, n1), (r2, n2) = rho(st["run1"], ref, info, key), rho(st["run2"], ref, info, key)
        d = []
        for _ in range(a.boot):
            rr = []
            for n in ("run1", "run2"):
                gs = G[n]
                idx = rng.integers(0, len(gs), len(gs))
                rr.append(rho(card_stats([gs[i] for i in idx]), ref, info, key)[0])
            if None not in rr:
                d.append(rr[1] - rr[0])
        ci = (np.percentile(d, 2.5), np.percentile(d, 97.5)) if d else (float("nan"),) * 2
        print(f"{key:7s} run1 {r1:+.3f} ({n1}) run2 {r2:+.3f} ({n2}) gap 95% [{ci[0]:+.2f}, {ci[1]:+.2f}]")


# ── figures ───────────────────────────────────────────────────────

# the reference categorical palette (dataviz skill), in its fixed order
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "normal", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "xtick.color": INK2, "ytick.color": INK2, "font.size": 9.5, "legend.frameon": False,
        "legend.fontsize": 8.5, "lines.linewidth": 2, "lines.markersize": 5,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _series(ax, xs, ys, color, label, **kw):
    ax.plot(xs, ys, color=color, label=label, marker=kw.pop("marker", "o"), **kw)


def fig_run2(run2: Path, out: Path, plasticity: dict | None, human_nonpass: dict | None) -> None:
    import matplotlib.pyplot as plt
    _style(plt)
    ref, info = load_reference(), card_info()
    m = run2 / "metrics.jsonl"
    tr, ho = metrics_rows(m, "train_epoch"), metrics_rows(m, "eval_prev_model")
    games = [g for g in read_jsonl(run2 / "games.jsonl") if g["gen"] <= max(tr)]   # completed generations only
    ds, sp = metrics_rows(m, "dataset"), metrics_rows(m, "selfplay")
    ha = {d["gen"] if d["agreement_checkpoint"] != "gen0" else -1: d
          for d in read_jsonl(m) if d.get("kind") == "human_agreement"}
    top = max(tr)
    fig, axs = plt.subplots(3, 3, figsize=(15, 12.5))
    fig.suptitle(f"Exp #2 run 2 (imitation start): learning curves, gens 0–{top} "
                 "(RTX 3090, budget 300, λ 0.95, priors off)", fontsize=13, color=INK)

    ax = axs[0, 0]
    g = sorted(tr)
    for key, c, lab in (("train_value_loss", C2, "value"), ("train_priority_A_loss", C1, "priority policy"),
                        ("train_choose_target_loss", C3, "target policy"), ("train_choose_use_loss", C4, "yes/no (attack, block)")):
        _series(ax, g, [tr[x][key] for x in g], c, lab)
    ax.set_title("Training loss (last epoch per generation)")
    ax.set_xlabel("generation")
    ax.set_ylim(bottom=0)
    ax.legend()

    ax = axs[0, 1]
    g = sorted(x for x in ho if x >= 1)
    for key, c, lab in (("value_loss", C2, "value"), ("priority_A_loss", C1, "priority policy"),
                        ("choose_target_loss", C3, "target policy"), ("choose_use_loss", C4, "yes/no")):
        _series(ax, g, [ho[x][key] for x in g], c, lab)
    ax.set_title("Held out: previous net on this generation's new games")
    ax.set_xlabel("generation (gen 0 omitted: the pretrained net's priority loss was 1.60)")
    ax.set_ylim(bottom=0)
    ax.legend()

    ax = axs[0, 2]
    lg = league_by_gen(games)
    gg = sorted(lg)
    p = [wilson(*lg[x]) for x in gg]
    ax.errorbar(gg, [q[0] for q in p], yerr=[[q[0] - q[1] for q in p], [q[2] - q[0] for q in p]],
                color=C1, marker="o", capsize=3, lw=2, label="league: net vs older nets")
    ev = evals(games)
    pe = {x: wilson(*ev[x]) for x in ev}
    ax.errorbar([x + 0.15 for x in pe], [q[0] for q in pe.values()],
                yerr=[[q[0] - q[1] for q in pe.values()], [q[2] - q[0] for q in pe.values()]],
                color=C2, marker="D", ms=7, capsize=3, ls="none", label="eval: net vs raw search @300 (200 games)")
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax.set_ylim(0.2, 0.9)
    ax.set_title("Strength (95% intervals)")
    ax.set_xlabel("generation")
    ax.set_ylabel("win rate")
    ax.legend(loc="upper left")

    ax = axs[1, 0]
    rc = rho_by_gen(games, ref, info, top, "common")
    ra = rho_by_gen(games, ref, info, top, None)
    g = sorted(rc)
    _series(ax, g, [rc[x][0] if rc[x][0] is not None else np.nan for x in g], C3, "commons (≥ 30 games in hand)")
    _series(ax, g, [ra[x][0] if ra[x][0] is not None else np.nan for x in g], INK2, "all cards (≥ 30 games)",
            marker="s", ls="--")
    for x in g:
        if rc[x][0] is not None:
            ax.annotate(str(rc[x][1]), (x, rc[x][0]), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=7, color=INK2)
    ax.axhline(0.28, color=C2, lw=1.2, ls=":", label="exp #1 commons, gens 10+ (0.28)")
    ax.set_title("17lands agreement (numbers: commons counted)")
    ax.set_xlabel("generations included (1..g), self-play + league")
    ax.set_ylabel("Spearman ρ vs 17lands GIH WR")
    ax.legend(loc="lower right")

    ax = axs[1, 1]
    g = sorted(ds)
    _series(ax, g, [ds[x]["value_label_abs_median"] for x in g], C2, "median |value label|")
    _series(ax, g, [ds[x]["value_label_near0_frac"] for x in g], C4, "share with |v| < 0.1", marker="s")
    _series(ax, g, [ds[x]["one_hot_policy_frac"] for x in g], C1, "policy targets one-hot", marker="^", ls=":")
    ax.set_ylim(0, 1)
    ax.set_title("Training targets")
    ax.set_xlabel("generation")
    ax.legend()

    ax = axs[1, 2]
    g = sorted(ds)
    hist = np.array([ds[x]["value_label_hist"] for x in g], float)
    hist = hist / hist.sum(1, keepdims=True)
    edges = np.linspace(-1, 1, 11)
    im = ax.imshow(hist.T, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=0.16,
                   extent=(g[0] - 0.5, g[-1] + 0.5, -1, 1))
    ax.set_title("Value labels per generation (share of states per bin)")
    ax.set_xlabel("generation")
    ax.set_ylabel("value label")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.ax.tick_params(colors=INK2)
    cb.outline.set_edgecolor(GRID)

    ax = axs[2, 0]
    g = sorted(sp)
    ax.bar(g, [sp[x]["games_per_hour"] for x in g], color=[GRID if x == 0 else C1 for x in g],
           edgecolor=SURF, linewidth=2)
    for x in g:
        ax.annotate(f"{sp[x]['games_per_hour']:.0f}", (x, sp[x]["games_per_hour"]), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7.5, color=INK2)
    ax.set_title("Throughput: games per hour of play (one shared inference server)")
    ax.set_xlabel("generation")

    ax = axs[2, 1]
    xs = sorted(ha)
    lab = ["start" if x == -1 else str(x) for x in xs]
    pos = list(range(len(xs)))
    _series(ax, pos, [ha[x]["top1"] for x in xs], C1, "top-1 in the human's plays (all options)")
    _series(ax, pos, [ha[x]["value_auc"] for x in xs], C2, "value AUC vs human game results", marker="s")
    if human_nonpass:
        pts = [(lab.index(k), v) for k, v in human_nonpass.items() if k in lab]
        _series(ax, [p_[0] for p_ in pts], [p_[1] for p_ in pts], C3, "top-1 among non-Pass plays (chance 48%)",
                marker="^")
    ax.set_xticks(pos, lab)
    ax.set_ylim(0, 1)
    ax.set_title("Human agreement, 3,000 held-out 17lands decisions")
    ax.set_xlabel("network (start = pretrained; n = trained at gen n)")
    ax.legend(loc="center right")

    ax = axs[2, 2]
    if plasticity:
        style = {"fresh": (INK2, "fresh network"), "pretrained": (C2, "run 2's pretrained start"),
                 "run1gen0": (C1, "run 1's gen 0"), "shrinkperturb": (C3, "pretrained, shrink and perturb")}
        for n, ep in plasticity.items():
            c, lab = style.get(n, (C4, n))
            _series(ax, list(range(len(ep))), [np.nan if v is None else v for v in ep], c, lab)
        ax.set_xticks([0, 1, 2])
        ax.set_title("Plasticity test: held-out loss after each epoch")
        ax.set_xlabel("epochs on run 2 gens 8–11 (0 = before training)")
        ax.set_ylabel("total loss on gen 12")
        ax.legend()
    else:
        ax.set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "013-run2-learning-curves.png", dpi=110)


def fig_compare(run1: Path, run2: Path, out: Path) -> None:
    import matplotlib.pyplot as plt
    _style(plt)
    ref, info = load_reference(), card_info()
    R = {"run 1 (heuristic start)": (run1, C1), "run 2 (imitation start)": (run2, C2)}
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Exp #2: run 1 against run 2", fontsize=13, color=INK)
    tops = {}
    for name, (d, c) in R.items():
        m = d / "metrics.jsonl"
        ho, tr = metrics_rows(m, "eval_prev_model"), metrics_rows(m, "train_epoch")
        games = [g for g in read_jsonl(d / "games.jsonl") if g["gen"] <= max(tr)]
        g = sorted(x for x in ho if x >= 1)
        _series(axs[0, 0], g, [ho[x]["avg_total_loss"] for x in g], c, name)
        g = sorted(tr)
        _series(axs[0, 1], g, [tr[x]["train_value_loss"] for x in g], c, name)
        lg = league_by_gen(games)
        gg = sorted(lg)
        cum = np.cumsum([lg[x][0] for x in gg]), np.cumsum([lg[x][1] for x in gg])
        p = [wilson(int(k), int(n)) for k, n in zip(*cum)]
        axs[1, 0].plot(gg, [q[0] for q in p], color=c, marker="o", label=f"{name}, league (cumulative)")
        axs[1, 0].fill_between(gg, [q[1] for q in p], [q[2] for q in p], color=c, alpha=0.12, lw=0)
        ev = evals(games)
        pe = {x: wilson(*ev[x]) for x in ev}
        off = -0.12 if c == C1 else 0.12
        axs[1, 0].errorbar([x + off for x in pe], [q[0] for q in pe.values()],
                           yerr=[[q[0] - q[1] for q in pe.values()], [q[2] - q[0] for q in pe.values()]],
                           color=c, marker="D", ms=7, capsize=3, ls="none", mec=SURF,
                           label=f"{name}, eval vs raw search")
        top = max(x for x in tr)
        tops[name] = top
        rc = rho_by_gen(games, ref, info, top, "common")
        g = sorted(rc)
        _series(axs[1, 1], g, [rc[x][0] if rc[x][0] is not None else np.nan for x in g], c, name)
    axs[0, 0].set_title("Held-out loss: previous net on new games (total)")
    axs[0, 1].set_title("Training value loss (last epoch)")
    axs[1, 0].set_title("Strength: league (cumulative, 95% band) and evals")
    axs[1, 0].axhline(0.5, color=INK2, lw=0.8, ls=":")
    axs[1, 0].set_ylim(0.3, 0.75)
    axs[1, 0].set_ylabel("win rate")
    axs[1, 1].set_title("17lands GIH ρ, commons (gens 1..g)")
    axs[1, 1].axhline(0.28, color=INK2, lw=1, ls=":", label="exp #1 commons, gens 10+ (0.28)")
    axs[1, 1].set_ylabel("Spearman ρ")
    for ax in axs.flat:
        ax.set_xlabel("generation")
        ax.legend(fontsize=8)
    axs[0, 1].set_ylim(bottom=0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "013-run1-vs-run2.png", dpi=110)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("reference")
    s = sub.add_parser("stats")
    s.add_argument("--run1", required=True)
    s.add_argument("--run2", required=True)
    s.add_argument("--gens", default="1-1000")
    s.add_argument("--boot", type=int, default=200)
    f = sub.add_parser("figures")
    f.add_argument("--run1", required=True)
    f.add_argument("--run2", required=True)
    f.add_argument("--out", default=str(REPO / "docs" / "img"))
    f.add_argument("--plasticity", default=None, help="json: start -> [held-out loss per epoch]")
    f.add_argument("--human-nonpass", default=None, help="json: network label -> non-Pass top-1")
    a = ap.parse_args()
    if a.cmd == "reference":
        r = build_reference()
        print(r["games"], "games,", len(r["cards"]), "cards ->", REF_OUT)
    elif a.cmd == "stats":
        cmd_stats(a)
    else:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        pl = json.loads(Path(a.plasticity).read_text()) if a.plasticity else None
        hn = json.loads(Path(a.human_nonpass).read_text()) if a.human_nonpass else None
        fig_run2(Path(a.run2), out, pl, hn)
        fig_compare(Path(a.run1), Path(a.run2), out)


if __name__ == "__main__":
    main()
