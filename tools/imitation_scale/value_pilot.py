"""docs/017's value pilot: how well do three leaf evaluators predict the results of held-out human
games? On #2b's clean test turn starts (docs/011), the same positions scored by

  heuristic   offline MageZero's GameStateEvaluator3.evaluateNormalized (the bridge's encode op with
              heuristic=true, after rebuilding each position from its 17lands row);
  #2b         the human-pretrained start's value head (behaviour cloning + game results);
  #2a gen 18  experiment #2a's last self-play value head.

Two stages, on the laptop:

    MZ_XMAGE_DIR=<v0.2 bundle>/xmage MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv \\
      python tools/imitation_scale/value_pilot.py heuristic --tables <run2 worktree>/data/gameplay/imitation
    python tools/imitation_scale/value_pilot.py score --tables ... --net 2b=<gen0.pt.gz> --net 2a=<gen18.pt.gz>
    python tools/imitation_scale/value_pilot.py figure          # docs/img/017-value-pilot-{light,dark}.png

The heuristic stage also checks that each rebuilt position encodes to exactly the features stored in
the table, so all three evaluators see the same states.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from draftzero.gameplay import imitation as im

OUT = Path("data/imitation_scale/value_pilot")


def _work(task):
    """One game: rebuild each requested turn start, encode it, score it with the heuristic."""
    from draftzero.gameplay import reconstruct as rc
    from draftzero.gameplay.replay import parse_game
    row, line, items = task
    g = parse_game(next(csv.reader([line])), im._W["H"], row)
    out = []
    for pos, turn, feats in items:
        rec = {"pos": pos, "row": row, "turn": turn}
        try:
            spec = rc.state_at_user_turn(g, turn, labels=True, ids=im._W["ids"])
            r = im._bridge().encode(spec, perfectInfo=False, heuristic=True, **dict(spec.labels.get("bridge") or {}))
            f = np.asarray(r.get("features") or [], dtype=np.int32)
            rec.update(heuristic=r.get("heuristic"), same_state=bool(np.array_equal(f, feats)))
        except Exception as e:  # noqa: BLE001 - a failed rebuild is a data point
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
        out.append(rec)
    return out


def stage_heuristic(tables: Path, workers: int, limit: int | None) -> None:
    from draftzero.gameplay.replay import open_lines, replay_path
    t = im.load_table(tables / "h5" / "turnstart_test.h5")
    keep = np.load(tables / "turnstart_test_clean_idx.npy")
    if limit:
        keep = keep[:limit]
    rows, turns = t["meta/row"][keep], t["meta/turn"][keep]
    by_row: dict[int, list] = {}
    for pos, r, n in zip(keep, rows, turns):
        a, b = t["offsets"][pos], t["offsets"][pos + 1]
        by_row.setdefault(int(r), []).append((int(pos), int(n), t["indices"][a:b]))
    H, lines = open_lines(replay_path())

    def tasks():
        for i, line in enumerate(lines):
            if i in by_row:
                yield (i, line, by_row[i])

    OUT.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    res, t0 = [], time.monotonic()
    with ctx.Pool(workers, initializer=im._worker_init,
                  initargs=(H.cols, str(OUT / "runtime"), "2500m", ctx.Value("i", 0))) as pool:
        for k, out in enumerate(pool.imap_unordered(_work, tasks(), chunksize=2), 1):
            res.extend(out)
            if k % 200 == 0:
                print(f"{k} games, {len(res)} positions, {time.monotonic() - t0:.0f} s", flush=True)
    lines.close()
    (OUT / "heuristic.jsonl").write_text("".join(json.dumps(r) + "\n" for r in res))
    ok = [r for r in res if r.get("heuristic") is not None]
    print(f"{len(res)} positions, {len(ok)} scored, {sum(r['same_state'] for r in ok)} with identical features, "
          f"{time.monotonic() - t0:.0f} s")


def _auc_ci(v, y, games, n_boot=300, seed=1):
    """AUC with a bootstrap over games."""
    u, inv = np.unique(games, return_inverse=True)
    groups = [np.flatnonzero(inv == k) for k in range(len(u))]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        ix = np.concatenate([groups[k] for k in rng.integers(0, len(u), len(u))])
        boots.append(im.auc_score(v[ix], y[ix]))
    return im.auc_score(v, y), float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def _paired_auc_diff(a, b, y, games, n_boot=300, seed=2):
    u, inv = np.unique(games, return_inverse=True)
    groups = [np.flatnonzero(inv == k) for k in range(len(u))]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        ix = np.concatenate([groups[k] for k in rng.integers(0, len(u), len(u))])
        d.append(im.auc_score(a[ix], y[ix]) - im.auc_score(b[ix], y[ix]))
    return im.auc_score(a, y) - im.auc_score(b, y), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


def _logloss_cv(v, y, games):
    """Log-loss after a logistic (Platt) fit, two folds split by game: each evaluator on its best scale."""
    from scipy.optimize import minimize
    fold = (np.unique(games, return_inverse=True)[1] % 2).astype(bool)
    ll = np.empty(len(v))
    for f in (False, True):
        tr, te = fold == f, fold != f

        def nll(w):
            z = w[0] * v[tr] + w[1]
            return np.mean(np.logaddexp(0, z) - y[tr] * z)
        w = minimize(nll, [1.0, 0.0]).x
        z = w[0] * v[te] + w[1]
        ll[te] = np.logaddexp(0, z) - y[te] * z
    return float(ll.mean())


def stage_score(tables: Path, nets: dict[str, Path]) -> None:
    heur = [json.loads(x) for x in (OUT / "heuristic.jsonl").read_text().splitlines()]
    heur = [h for h in heur if h.get("heuristic") is not None and h["same_state"]]
    pos = np.array(sorted(h["pos"] for h in heur))
    hv = {h["pos"]: h["heuristic"] for h in heur}
    t = im.subset_table(im.load_table(tables / "h5" / "turnstart_test.h5"), pos)
    y = (t["z"] > 0).astype(np.float64)
    games = t["meta/row"].astype(np.int64)
    turn = t["meta/turn"].astype(np.int64)
    V = {"heuristic": np.array([hv[p] for p in pos], dtype=np.float64)}
    im._torch_env()
    for name, path in nets.items():
        model, vocab = im.load_checkpoint(path)
        p = im.predict(model, vocab, t)
        V[name] = p["v"].astype(np.float64)
        print(f"{name}: {p['mapped_share']:.1%} of features in its vocab")
    out = {"n": int(len(y)), "games": int(len(np.unique(games))), "base_rate": float(y.mean()), "evaluators": {}}
    buckets = [(1, 2), (3, 4), (5, 6), (7, 9), (10, 99)]
    first = {}                                   # each game's earliest scored turn, for the trend measure
    for i in np.argsort(turn, kind="stable"):
        first.setdefault(games[i], i)
    late = np.array([turn[i] >= 5 and turn[first[games[i]]] <= 2 for i in range(len(y))])
    for name, v in V.items():
        e = {"auc": _auc_ci(v, y, games), "log_loss_platt": _logloss_cv(v, y, games),
             "by_turn": {f"{a}-{b}": {"n": int(((turn >= a) & (turn <= b)).sum()),
                                     "auc": im.auc_score(v[(turn >= a) & (turn <= b)], y[(turn >= a) & (turn <= b)])}
                         for a, b in buckets}}
        # trend: the change since the game's first position (turn 1-2), at turns 5+. A game's deck and
        # who is on the play are constant, so they cancel; what's left is how the position moved.
        dv = np.array([v[i] - v[first[games[i]]] for i in range(len(y))])
        e["trend_auc"] = _auc_ci(dv[late], y[late], games[late])
        e["trend_n"] = int(late.sum())
        out["evaluators"][name] = e
    names = list(V)
    out["paired_auc"] = {f"{a} - {b}": _paired_auc_diff(V[a], V[b], y, games)
                         for k, a in enumerate(names) for b in names[k + 1:]}
    from scipy.stats import spearmanr
    out["spearman"] = {f"{a} ~ {b}": float(spearmanr(V[a], V[b]).statistic)
                       for k, a in enumerate(names) for b in names[k + 1:]}
    # the networks on replayed mid-turn priority decisions of the same clean games (never trained on)
    rp = im.load_table(tables / "h5" / "replay_priority_test.h5")
    rp = im.subset_table(rp, np.flatnonzero(np.isin(rp["meta/row"], np.unique(games))))
    yr = (rp["z"] > 0).astype(np.float64)
    out["midturn"] = {"n": int(len(yr)), "games": int(len(np.unique(rp["meta/row"])))}
    for name, path in nets.items():
        model, vocab = im.load_checkpoint(path)
        out["midturn"][name] = _auc_ci(im.predict(model, vocab, rp)["v"].astype(np.float64), yr,
                                       rp["meta/row"].astype(np.int64))
    (OUT / "results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", heuristic="#52514e", **{"2a": "#2a78d6", "2b": "#eb6834"}),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", heuristic="#c3c2b7", **{"2a": "#3987e5", "2b": "#d95926"}),
}
LABELS = {"heuristic": "Heuristic (hand-written)", "2a": "#2a gen 18 (self-play)", "2b": "#2b (behaviour cloning)"}
# docs/016 §7: PIMC with 1 world, 1,000 simulations, priors off; balanced score and 95% CI
SB_V1 = {"heuristic": (0.597, 0.571, 0.626), "2a": (0.604, 0.576, 0.635), "2b": (0.599, 0.568, 0.633)}


def stage_figure(out_prefix: Path = Path("docs/img/017-value-pilot")) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = json.loads((OUT / "results.json").read_text())
    order = ["heuristic", "2a", "2b"]
    buckets = list(r["evaluators"]["heuristic"]["by_turn"])
    for mode, t in THEMES.items():
        plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10})
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150, gridspec_kw={"width_ratios": [1.5, 1]})
        fig.patch.set_facecolor(t["surface"])
        for ax in (a1, a2):
            ax.set_facecolor(t["surface"])
            ax.tick_params(colors=t["ink2"], length=0)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(t["axis"])
            ax.grid(color=t["grid"], linewidth=0.8, zorder=0)
        x = np.arange(len(buckets))
        for k in order:
            e = r["evaluators"][k]
            ys = [e["by_turn"][b]["auc"] for b in buckets]
            a1.plot(x, ys, color=t[k], linewidth=2, marker="o", markersize=7, markeredgecolor=t["surface"],
                    markeredgewidth=2, zorder=3)
            a1.annotate(f"{LABELS[k]}: {e['auc'][0]:.3f}", (x[-1], ys[-1]), xytext=(8, {"heuristic": -3, "2a": -12, "2b": 4}[k]),
                        textcoords="offset points", color=t["ink"], fontsize=8.5, va="center")
        a1.set_xticks(x, [b.replace("-99", "+") for b in buckets])
        a1.set_xlim(-0.3, len(buckets) + 1.9)
        a1.set_xlabel("The player's turn number", color=t["ink2"])
        a1.set_ylabel("AUC predicting who won (0.5 = chance)", color=t["ink2"])
        a1.set_title("Offline: predicting the results of held-out human games", color=t["ink"], fontsize=10.5, loc="left")
        ys = np.arange(len(order))[::-1]
        for y, k in zip(ys, order):
            m, lo, hi = SB_V1[k]
            a2.plot([lo, hi], [y, y], color=t[k], linewidth=2, zorder=3)
            a2.plot([m], [y], marker="o", markersize=8, color=t[k], markeredgecolor=t["surface"], markeredgewidth=2, zorder=4)
            a2.annotate(f"{m:.3f}", (hi, y), xytext=(6, 0), textcoords="offset points", color=t["ink"], fontsize=9, va="center")
        a2.set_yticks(ys, [LABELS[k] for k in order])
        for lab in a2.get_yticklabels():
            lab.set_color(t["ink"])
        a2.set_ylim(-0.6, len(order) - 0.4)
        a2.set_xlim(0.55, 0.66)
        a2.set_xlabel("Balanced agreement (0.50 = constant)", color=t["ink2"])
        a2.set_title("In search: the same heads at the leaves", color=t["ink"], fontsize=10.5, loc="left")
        fig.text(0.012, 0.955, "The behaviour-cloned value head predicts results better, but doesn't search better",
                 color=t["ink"], fontsize=12.5, fontweight="bold", ha="left")
        fig.text(0.012, 0.9, f"Left: {r['n']:,} held-out turn starts from {r['games']:,} games (#2b's clean test split). "
                 "Right: experiment #3's 1,000 decisions, PIMC with 1 world, 1,000 simulations, priors off; 95% CIs.",
                 color=t["ink2"], fontsize=8.5, ha="left")
        fig.subplots_adjust(left=0.07, right=0.97, top=0.8, bottom=0.14, wspace=0.75)
        p = Path(f"{out_prefix}-{mode}.png")
        fig.savefig(p, facecolor=t["surface"])
        plt.close(fig)
        print(p)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("heuristic")
    h.add_argument("--tables", type=Path, default=im.OUT_DIR)
    h.add_argument("--workers", type=int, default=3)
    h.add_argument("--limit", type=int)
    s = sub.add_parser("score")
    s.add_argument("--tables", type=Path, default=im.OUT_DIR)
    s.add_argument("--net", action="append", default=[], help="name=checkpoint")
    sub.add_parser("figure")
    a = ap.parse_args(argv)
    if a.cmd == "heuristic":
        stage_heuristic(a.tables, a.workers, a.limit)
    elif a.cmd == "score":
        stage_score(a.tables, {k: Path(v) for k, v in (x.split("=", 1) for x in a.net)})
    else:
        stage_figure()


if __name__ == "__main__":
    main()
