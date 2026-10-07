"""Collect docs/027's numbers from mirrored run directories into the figures' JSON inputs and one table.

    python gorge/dzg_collect.py RESULTS_DIR      # RESULTS_DIR/<machine>/runs/... mirrored by rsync

Reads every *.summary.json (dzgorge play) and curves.jsonl/config.json (dzg.train) under RESULTS_DIR and
writes RESULTS_DIR/{speed,scores_g1,gens}.json plus RESULTS_DIR/table.tsv (one row per game run).
"""
from __future__ import annotations

import glob
import json
import os
import sys

MACHINES = {"dz3090": "RTX 3090 (community)", "dzpod": "RTX 3080 Ti (community)", "r1": "r1 (2x RTX PRO 6000)"}


def runs(root):
    out = []
    for p in sorted(glob.glob(os.path.join(root, "*", "runs", "**", "*.jsonl.summary.json"), recursive=True)):
        rel = os.path.relpath(p, root)
        machine = rel.split(os.sep)[0]
        s = json.load(open(p))
        s["_machine"], s["_path"] = machine, rel
        s["_name"] = os.path.basename(p)[: -len(".jsonl.summary.json")]
        s["_dir"] = os.path.basename(os.path.dirname(p))
        out.append(s)
    return out


def main(root):
    rs = runs(root)
    with open(os.path.join(root, "table.tsv"), "w") as f:
        f.write("machine\tdir\tname\ta\tb\tgames\ta_score\tgames_per_hour\taz_ms_per_searched\tworkers\n")
        for s in rs:
            f.write("\t".join(str(x) for x in (s["_machine"], s["_dir"], s["_name"], s.get("a"), s.get("b"), s.get("games"),
                                               round(s.get("a_score", 0), 4), round(s.get("games_per_hour", 0)),
                                               round(s.get("az_ms_per_searched", 0) or 0, 1), s.get("workers"))) + "\n")
    # games an hour: the bench directories
    speed = []
    for s in rs:
        if not s["_dir"].startswith("bench-"):
            continue
        net, mode, sims = s["_name"].rsplit("-", 2)
        mach = "3090" if "3090" in s["_dir"] else "3080 Ti"
        speed.append({"machine": mach, "net": net, "mode": mode, "sims": int(sims), "games": s.get("games"),
                      "games_per_hour": s.get("games_per_hour"), "ms_per_searched": s.get("az_ms_per_searched"),
                      "series": f"{mach}, {'no network' if net == 'none' else net}, "
                                f"{'self-play' if mode == 'selfplay' else 'search vs bot'}"})
    json.dump(speed, open(os.path.join(root, "speed.json"), "w"), indent=1)
    # generation-1 arms and the loops' generations
    g1 = [{"dir": s["_dir"], "arm": s["_name"], "score": s["a_score"], "games": s["games"],
           "games_per_hour": s.get("games_per_hour"), "machine": s["_machine"]}
          for s in rs if s["_dir"].startswith("g1-")]
    json.dump(g1, open(os.path.join(root, "scores_g1.json"), "w"), indent=1)
    gens = {}
    for s in rs:
        d = s["_dir"]
        if d.startswith("eval-g") and "loop" in s["_path"]:
            loop = [x for x in s["_path"].split(os.sep) if x.startswith("loop")][0]
            gens.setdefault(loop, []).append({"gen": int(d[len("eval-g"):]), "arm": s["_name"], "score": s["a_score"],
                                              "games": s["games"]})
    json.dump(gens, open(os.path.join(root, "gens.json"), "w"), indent=1)
    print(f"{len(rs)} game runs; {len(speed)} speed rows; {len(g1)} gen-1 arms; loops {sorted(gens)}")


if __name__ == "__main__":
    main(sys.argv[1])
