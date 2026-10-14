"""
Secondary measures: how human-like a builder's decks are, beyond card-for-card agreement.

  python -m sealed.secondary EVAL_DIR [--games GEN_DIR ...]

From EVAL_DIR/built.jsonl (the builder's decks for the first N val pools, as evalbuild writes them):
- main colours: the share of decks per colour pair, builder against the humans (total variation distance);
- per card: how often the card is played when it is in the pool, builder against the humans (rank correlation);
- deck shape: lands, creatures, average mana value of the spells, colours.
With --games: each card's games-in-hand win rate in those self-play games, against 17lands' (rank correlation).
"""
import argparse
import collections
import json
import os

import numpy as np
from scipy.stats import spearmanr

from . import decks as D
from . import selfplay as S

C = D.C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir")
    ap.add_argument("--kind", default="sealed")
    ap.add_argument("--games", nargs="*", default=[])
    a = ap.parse_args()
    built = [C.vec(json.loads(l)) for l in open(os.path.join(a.eval_dir, "built.jsonl"))]
    pools = D.load_pools(a.kind, "val")[: len(built)]
    human = [p["ref_v"] for p in pools]
    out = {"pools": len(built)}

    def main_cols(d):
        return "".join(sorted(D.deck_colors(d).split("+")[0], key="WUBRG".index)) or "C"
    cb = collections.Counter(main_cols(d) for d in built)
    ch = collections.Counter(main_cols(d) for d in human)
    keys = sorted(set(cb) | set(ch), key=lambda k: -ch.get(k, 0))
    tv = 0.5 * sum(abs(cb[k] / len(built) - ch[k] / len(human)) for k in keys)
    out["colour_tv_distance"] = round(tv, 3)
    out["colours"] = {k: [round(cb[k] / len(built), 3), round(ch[k] / len(human), 3)] for k in keys[:12]}

    # per-card play rate when in the pool
    inpool = np.zeros(len(C)); pb = np.zeros(len(C)); ph = np.zeros(len(C))
    for d, h, p in zip(built, human, pools):
        pv = p["pool_v"]
        m = (pv > 0) & ~C.is_basic
        inpool += m
        pb += m * (np.minimum(d, pv) / np.maximum(pv, 1))
        ph += m * (np.minimum(h, pv) / np.maximum(pv, 1))
    ok = inpool >= 10
    rb, rh = pb[ok] / inpool[ok], ph[ok] / inpool[ok]
    out["card_play_rate_rho"] = round(float(spearmanr(rb, rh).correlation), 3)
    names = np.array(C.names)[ok]
    diff = rb - rh
    out["played_more_than_humans"] = [(n, round(float(x), 2), round(float(y), 2))
                                      for n, x, y in sorted(zip(names, rb, rh), key=lambda t: -(t[1] - t[2]))[:12]]
    out["played_less_than_humans"] = [(n, round(float(x), 2), round(float(y), 2))
                                      for n, x, y in sorted(zip(names, rb, rh), key=lambda t: t[1] - t[2])[:12]]

    def shape(ds):
        return {"lands": round(float(np.mean([d[C.is_land].sum() for d in ds])), 2),
                "creatures": round(float(np.mean([d[C.is_creature].sum() for d in ds])), 2),
                "spell_mv": round(float(np.mean([(d * C.mv)[C.spell].sum() / max(d[C.spell].sum(), 1) for d in ds])), 2),
                "nonbasic_lands": round(float(np.mean([d[C.is_land & ~C.is_basic].sum() for d in ds])), 2),
                "noncreature_spells": round(float(np.mean([d[C.spell & ~C.is_creature].sum() for d in ds])), 2)}
    out["shape_builder"], out["shape_humans"] = shape(built), shape(human)

    if a.games:
        gih = json.load(open(os.path.join(D.cardsmod.REPO, "assets/reference/FDN_card_stats.json")))["cards"]
        w = np.zeros(len(C)); n = np.zeros(len(C))
        for g in S.load_games([os.path.join(x, "games.jsonl") for x in a.games]):
            if g.get("winner", -1) not in (0, 1):
                continue
            for s in (0, 1):
                for c in set(g.get("in_hand", [[], []])[s]):
                    i = C.index.get(c)
                    if i is not None:
                        n[i] += 1; w[i] += g["winner"] == s
        ref = np.array([gih.get(x, {}).get("gih_wr", np.nan) for x in C.names])
        ok = C.spell & (n >= 200) & ~np.isnan(ref)
        out["selfplay_gih_vs_17lands_rho"] = round(float(spearmanr(w[ok] / n[ok], ref[ok]).correlation), 3)
        out["selfplay_gih_cards"] = int(ok.sum())
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(a.eval_dir, "secondary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
