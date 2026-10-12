"""The search-depth test's decks and pairs: on val pools, the human reference deck (h), a random deck-shaped build (r)
and the 17lands GIH heuristic (g), crossed over pool pairs (i, j): h_i vs r_j and r_j vs h_i on one seed, and the same
for (j, i). Writes data/sealed/depth/{decks.jsonl, hr.tsv, hg.tsv}."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sealed import decks as D

C = D.C
out = os.path.join(D.cardsmod.REPO, "data/sealed/depth")
val = D.load_pools("sealed", "val")
gih = json.load(open(os.path.join(D.cardsmod.REPO, "assets/reference/FDN_card_stats.json")))["cards"]
rating = np.array([gih.get(n, {}).get("gih_wr", np.nan) for n in C.names])
rng = np.random.default_rng(31)
dk = {}
for k, p in enumerate(val):
    dk[f"h{k}"] = p["ref_v"]; dk[f"r{k}"] = D.random_build(p["pool_v"], rng); dk[f"g{k}"] = D.rating_build(p["pool_v"], rating)
D.write_decks(os.path.join(out, "decks.jsonl"), dk)
for other in "rg":
    with open(os.path.join(out, f"h{other}.tsv"), "w") as f:
        for i in range(0, len(val), 2):
            j = i + 1
            for a, b in ((i, j), (j, i)):
                seed = int(rng.integers(1, 2**62))
                f.write(f"h{a}\t{other}{b}\t{seed}\n{other}{b}\th{a}\t{seed}\n")
