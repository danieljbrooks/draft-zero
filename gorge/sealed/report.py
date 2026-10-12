"""
Tables and figures for docs/032.

  python -m sealed.report curves RUN_DIR [RUN_DIR ...]      # per-generation table (+ docs/img/032-curves-light.png)
  python -m sealed.report cards ENS.pt [--top 25]           # self-play card values against 17lands' GIH win rate
  python -m sealed.report roles ENS.pt                      # cards whose learned roles moved from the printed card
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

from . import cards as cardsmod

REPO = cardsmod.REPO


def curves(runs, fig=None):
    rows = []
    for r in runs:
        for m in sorted(glob.glob(os.path.join(r, "gen*/metrics.json")), key=lambda p: int(p.split("gen")[-1].split("/")[0])):
            j = json.load(open(m))
            ag, vh = j["agreement"], j.get("vs_human", {})
            rows.append((os.path.basename(r.rstrip("/")), j["gen"], j["sims"], ag["overlap"], ag["spells"],
                         ag["main_colours_match"], ag["lands_mean"], j["train"]["logloss"], j["train"]["base_logloss"],
                         vh.get("winrate"), vh.get("games")))
    print("| Run | Gen | Play bot | Cards shared /40 | Spells shared | Main colours match | Lands | Held-out loss (base) | Vs human decks |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        bot = "heuristic" if r[2] == 0 else f"{r[2]} sims"
        vh = f"{100 * r[9]:.1f}% of {r[10]:,}" if r[9] is not None else ""
        print(f"| {r[0]} | {r[1]} | {bot} | {r[3]:.1f} | {r[4]:.1f} | {100 * r[5]:.0f}% | {r[6]:.1f} | "
              f"{r[7]:.4f} ({r[8]:.4f}) | {vh} |")
    if fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for run in dict.fromkeys(r[0] for r in rows):
            rr = [r for r in rows if r[0] == run]
            ax[0].plot([r[1] for r in rr], [r[3] for r in rr], marker="o", label=run)
            pts = [(r[1], 100 * r[9]) for r in rr if r[9] is not None]
            if pts:
                ax[1].plot(*zip(*pts), marker="o", label=run)
        ax[0].set_title("Cards shared with the human deck (of 40)"); ax[0].set_xlabel("generation")
        ax[1].set_title("Win rate against human decks (%)"); ax[1].set_xlabel("generation")
        ax[1].axhline(50, color="grey", lw=0.8, ls="--")
        for a in ax:
            a.grid(alpha=0.3); a.spines[["top", "right"]].set_visible(False)
        ax[1].legend(frameon=False, fontsize=8)
        f.tight_layout(); f.savefig(fig, dpi=150)


def marginal(ens, C, n_decks=200):
    """Each non-land card's marginal value: the mean change in the ensemble's deck score when it replaces one random
    non-land card of a built val deck whose colours can cast it (decks from the ensemble's own builder)."""
    from . import build as Bd, decks as D
    rng = np.random.default_rng(0)
    pools = D.load_pools("sealed", "val")[:n_decks]
    with __import__("torch").no_grad():
        qq = np.mean([m.quality().numpy() for m in ens.members], 0)
    obj = Bd.Objective(ens, "mean")
    decks = [Bd.build(p["pool_v"], obj, qq, n_refine=2) for p in pools]
    cols = [((d[:, None] * C.pips).sum(0) > 0) for d in decks]
    out = np.full(len(C), np.nan)
    base = obj(np.array(decks))
    for c in np.flatnonzero(C.spell):
        rows, idx = [], []
        for k, d in enumerate(decks):
            if C.colors[c].any() and not (C.colors[c] <= cols[k]).all():
                continue
            nl = np.flatnonzero((d > 0) & C.spell)
            o = rng.choice(nl)
            e = d.copy(); e[o] -= 1; e[c] += 1
            rows.append(e); idx.append(k)
        if rows:
            out[c] = float(np.mean(obj(np.array(rows)) - base[idx]))
    return out


def cards(ens_path, top=25):
    import torch
    from . import model as M
    C = cardsmod.load()
    ens = M.Ensemble.load(ens_path)
    q = marginal(ens, C)
    gih = json.load(open(os.path.join(REPO, "assets/reference/FDN_card_stats.json")))["cards"]
    g = np.array([gih.get(n, {}).get("gih_wr", np.nan) for n in C.names])
    ok = C.spell & ~np.isnan(g) & ~np.isnan(q)
    from scipy.stats import spearmanr
    rho = spearmanr(q[ok], g[ok]).correlation
    idx = np.flatnonzero(ok)
    rq = {i: r + 1 for r, i in enumerate(idx[np.argsort(-q[idx])])}
    rg = {i: r + 1 for r, i in enumerate(idx[np.argsort(-g[idx])])}
    print(f"Spearman correlation of self-play marginal card value with 17lands GIH win rate, {ok.sum()} spells: {rho:.3f}\n")
    print("| Card | Rarity | 17lands GIH WR | Rank by 17lands | Rank by self-play |")
    print("|---|---|---|---|---|")
    for i in idx[np.argsort(-g[idx])][:top]:
        print(f"| {C.names[i]} | {C.rarity[i]} | {100 * g[i]:.1f}% | {rg[i]} | {rq[i]} |")
    print("\nThe self-play model's top cards:\n")
    print("| Card | Rarity | Rank by self-play | Rank by 17lands |")
    print("|---|---|---|---|")
    for i in idx[np.argsort(-q[idx])][:top]:
        print(f"| {C.names[i]} | {C.rarity[i]} | {rq[i]} | {rg[i]} |")
    return rho


def roles(ens_path, top=15):
    import torch
    from . import model as M
    C = cardsmod.load()
    ens = M.Ensemble.load(ens_path)
    with torch.no_grad():
        R = [m.roles() for m in ens.members]
    land = np.mean([r[0].numpy() for r in R], 0)
    mv = np.mean([(r[1].numpy() * np.arange(r[1].shape[1])).sum(1) for r in R], 0)
    crea = np.mean([r[2].numpy() for r in R], 0)
    print("| Card | Printed: land? mana value / creature? | Learned: land share, mana value, creature share |")
    print("|---|---|---|")
    move = np.abs(land - C.is_land) + np.abs(mv - np.minimum(C.mv, 7)) / 7 + np.abs(crea - C.is_creature)
    for i in np.argsort(-move)[:top]:
        print(f"| {C.names[i]} | {'land' if C.is_land[i] else 'spell'}, {C.mv[i]:.0f}, "
              f"{'creature' if C.is_creature[i] else '-'} | {land[i]:.2f}, {mv[i]:.1f}, {crea[i]:.2f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["curves", "cards", "roles"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--fig", default=None)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    if a.what == "curves":
        curves(a.paths, a.fig)
    elif a.what == "cards":
        cards(a.paths[0], a.top)
    else:
        roles(a.paths[0], a.top)


if __name__ == "__main__":
    main()
