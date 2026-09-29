"""Why does search agree with top 17lands players only about half the time? Breakdowns of the
first experiment's decisions (docs/016 §8).

    python tools/search_bench/diagnose.py --items data/search_bench/sb-v1 --runs runs/search_bench/e2_offline \\
        runs/search_bench/e2_network ... --out runs/search_bench/diagnosis
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import load_items, load_rows, mana_value  # noqa: E402

FOCUS = ["pimc1-b3000-offline-d0.99", "pimc1-b3000-remote-d0.99", "clairvoyant-b3000-offline-d0.99",
         "ismcts-b3000-offline-d0.99", "pimc1-b100-offline-d0.99", "policy-b0-remote"]


def cast_name(label: str) -> str | None:
    return label[5:] if label and label.startswith("Cast ") else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--policy-imit", default=None, help="dir of the #2b start's policy run (its run id collides)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    items = {k: v for k, v in load_items(Path(a.items)).items() if v["split"] == "test"}
    rows, _ = load_rows([Path(d) for d in a.runs])
    by = defaultdict(dict)
    for r in rows:
        if r["item_id"] in items:
            by[r["run_id"]][r["item_id"]] = r
    if a.policy_imit:
        irows, _ = load_rows([Path(a.policy_imit)])
        for r in irows:
            if r["item_id"] in items:
                by["imit-policy"][r["item_id"]] = r
    out = {}
    lines = []

    def P(s=""):
        lines.append(s)
        print(s)

    # 1. label composition
    P("== 1. Labels")
    for t in ("spell", "hold", "attack", "block"):
        its = [it for it in items.values() if it["type"] == t]
        if t == "spell":
            lenient = sum("Pass" in it["label"] and "Pass" not in it["label_strict"] for it in its)
            sizes = Counter(len(it["label_strict"]) for it in its)
            P(f"spell: {len(its)} items; Pass counted leniently in {lenient} ({lenient / len(its):.0%}); "
              f"human cast-set sizes {dict(sorted(sizes.items()))}; mean legal options {sum(it['n_options'] for it in its) / len(its):.2f}")
        elif t == "attack":
            yes = sum(it["label"] == ["yes"] for it in its)
            P(f"attack: {len(its)} items; the human attacked with the creature in {yes} ({yes / len(its):.0%})")
        elif t == "block":
            nob = sum(it["label"] == ["Stop Choosing"] for it in its)
            P(f"block: {len(its)} items; the human did not block with it in {nob} ({nob / len(its):.0%})")
        else:
            P(f"hold: {len(its)} items; label Pass; mean legal options {sum(it['n_options'] for it in its) / len(its):.2f}")

    # 2. what each run does instead
    P("\n== 2. Choices by type (focus runs)")
    for rid in FOCUS + (["imit-policy"] if a.policy_imit else []):
        R = by.get(rid)
        if not R:
            continue
        P(f"-- {rid}")
        res = {}
        # spell / hold: pass vs cast, and mana value relative to the human's
        for t in ("spell", "hold"):
            c = Counter()
            for iid, r in R.items():
                it = items[iid]
                if it["type"] != t:
                    continue
                best = r.get("best") or ""
                lab = set(it["label"])
                strict = set(it["label_strict"])
                if t == "hold" and best == "Pass":
                    c["agree (pass)"] += 1
                elif best in strict:
                    c["agree (a human cast)"] += 1
                elif best == "Pass":
                    c["pass, counted (human attacked)" if "Pass" in lab else "pass, human cast"] += 1
                elif cast_name(best):
                    if t == "hold":
                        c["cast, human held"] += 1
                    else:
                        hm = max((mana_value(cast_name(x) or "") for x in strict), default=0)
                        sm = mana_value(cast_name(best))
                        c["cast another: cheaper than the human's" if sm < hm else
                          "cast another: dearer" if sm > hm else "cast another: same mana value"] += 1
                else:
                    c["activate / land / other"] += 1
            n = sum(1 for iid in R if items[iid]["type"] == t)
            res[t] = {k: round(v / n, 3) for k, v in c.items()}
            P(f"   {t:6s} " + "; ".join(f"{k} {v / n:.0%}" for k, v in c.most_common()))
        # attack 2x2
        m = Counter()
        for iid, r in R.items():
            it = items[iid]
            if it["type"] == "attack":
                m[(it["label"][0], r.get("best"))] += 1
        n = sum(m.values())
        if n:
            P(f"   attack human yes: search yes {m[('yes', 'yes')] / n:.0%} no {m[('yes', 'no')] / n:.0%} | "
              f"human no: search yes {m[('no', 'yes')] / n:.0%} no {m[('no', 'no')] / n:.0%}  "
              f"(search attacks {(m[('yes', 'yes')] + m[('no', 'yes')]) / n:.0%}, humans {(m[('yes', 'yes')] + m[('yes', 'no')]) / n:.0%})")
            res["attack"] = {f"{h}->{s}": round(v / n, 3) for (h, s), v in m.items()}
        # block
        b = Counter()
        for iid, r in R.items():
            it = items[iid]
            if it["type"] != "block":
                continue
            h_block = it["label"][0] != "Stop Choosing"
            s_block = (r.get("best") or "Stop Choosing") != "Stop Choosing"
            same = r.get("best") in it["label"]
            b[("human blocks" if h_block else "human no block", "search blocks" if s_block else "search no block",
               "same" if same else "diff")] += 1
        n = sum(b.values())
        if n:
            P("   block " + "; ".join(f"{h}/{s}/{d} {v / n:.0%}" for (h, s, d), v in sorted(b.items())))
            res["block"] = {"/".join(k): round(v / n, 3) for k, v in b.items()}
        out[rid] = res

    # 3. near ties: how far the human's option is from the search's best, in the search's own values
    P("\n== 3. Disagreements: near ties or confident? (Q of the search's choice minus the best Q among the human's options)")
    for rid in FOCUS[:5]:
        R = by.get(rid)
        if not R:
            continue
        gaps, shares = [], []
        for iid, r in R.items():
            it = items[iid]
            if r.get("best") in set(it["label"]):
                continue
            q = {c["label"]: c.get("Q") for c in r.get("children") or []}
            n = {c["label"]: c.get("N") or 0 for c in r.get("children") or []}
            tot = sum(n.values()) or 1
            hq = [q[l] for l in it["label"] if q.get(l) is not None]
            if q.get(r.get("best")) is None or not hq:
                continue
            gaps.append(q[r["best"]] - max(hq))
            shares.append(sum(n.get(l, 0) for l in it["label"]) / tot)
        gaps.sort()
        k = len(gaps)
        if k:
            P(f"   {rid:34s} n {k:4d}  median gap {gaps[k // 2]:.3f}  <0.02: {sum(g < 0.02 for g in gaps) / k:.0%}  "
              f"<0.05: {sum(g < 0.05 for g in gaps) / k:.0%}  >0.15: {sum(g > 0.15 for g in gaps) / k:.0%}  "
              f"mean visit share on the human's options {sum(shares) / k:.0%}")
            out.setdefault("near_ties", {})[rid] = {"n": k, "median_gap": round(gaps[k // 2], 4),
                                                   "lt_0.02": round(sum(g < 0.02 for g in gaps) / k, 3),
                                                   "lt_0.05": round(sum(g < 0.05 for g in gaps) / k, 3)}

    # 4. horizon: how far ahead the trees look
    P("\n== 4. Horizon (mean over decisions): leaf depth in plies and turns crossed per simulation")
    for rid in sorted(by):
        if not rid.endswith("-d0.99") or "policy" in rid:
            continue
        R = by[rid]
        st = [r.get("stats") or {} for r in R.values()]
        sims = sum(s.get("sims") or 0 for s in st) or 1
        P(f"   {rid:34s} plies/sim {sum(s.get('edgeVisits') or 0 for s in st) / sims:5.2f}  "
          f"turns crossed/sim {sum(s.get('turnEdgeSum') or 0 for s in st) / sims:4.2f}  max depth (mean) "
          f"{sum(s.get('maxDepth') or 0 for s in st) / max(1, len(st)):5.1f}")

    # 5. do the searches agree with each other?
    P("\n== 5. Agreement between runs (same choice, over common decisions)")
    pairs = [("pimc1-b3000-offline-d0.99", "clairvoyant-b3000-offline-d0.99"),
             ("pimc1-b3000-offline-d0.99", "ismcts-b3000-offline-d0.99"),
             ("pimc1-b3000-offline-d0.99", "pimc1-b3000-remote-d0.99"),
             ("pimc1-b3000-offline-d0.99", "pimc1-b1000-offline-d0.99"),
             ("pimc4-b1000-offline-d0.99", "pimc4-b1000-offline-d0.99-s1"),
             ("pimc4-b1000-remote-d0.99", "pimc4-b1000-remote-d0.99-s1"),
             ("pimc1-b3000-offline-d0.99", "policy-b0-remote"),
             ("pimc1-b3000-offline-d0.99", "imit-policy"),
             ("policy-b0-remote", "imit-policy")]
    for x, y in pairs:
        if x in by and y in by:
            common = [i for i in by[x] if i in by[y]]
            same = sum(by[x][i].get("best") == by[y][i].get("best") for i in common)
            P(f"   {x:34s} ~ {y:34s} {same / len(common):.0%} of {len(common)}")

    # 6. how many decisions does any search get right?
    P("\n== 6. Decisions no search gets right (over every E2 run at 0.99 per ply)")
    runs = [rid for rid in by if rid.endswith("-d0.99") and "policy" not in rid and "-s1" not in rid]
    hit = Counter()
    for iid, it in items.items():
        k = sum(by[rid].get(iid, {}).get("best") in set(it["label"]) for rid in runs if iid in by[rid])
        hit[(it["type"], "never" if k == 0 else "always" if k == len(runs) else "some")] += 1
    for t in ("spell", "hold", "attack", "block"):
        n = sum(v for (tt, _), v in hit.items() if tt == t)
        P(f"   {t:6s} never {hit[(t, 'never')] / n:.0%}  some {hit[(t, 'some')] / n:.0%}  always {hit[(t, 'always')] / n:.0%}  "
          f"({len(runs)} runs)")
        out.setdefault("coverage", {})[t] = {k: round(hit[(t, k)] / n, 3) for k in ("never", "some", "always")}
    if a.policy_imit and "imit-policy" in by:
        both = Counter()
        for iid, it in items.items():
            s_ok = any(by[rid].get(iid, {}).get("best") in set(it["label"]) for rid in runs)
            i_ok = by["imit-policy"].get(iid, {}).get("best") in set(it["label"])
            both[(it["type"], s_ok, i_ok)] += 1
        P("   the human-pretrained policy on decisions no search gets right:")
        for t in ("spell", "hold", "attack", "block"):
            never = both[(t, False, True)] + both[(t, False, False)]
            if never:
                P(f"     {t:6s} {both[(t, False, True)]} of {never} right ({both[(t, False, True)] / never:.0%})")

    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "diagnosis.json").write_text(json.dumps(out, indent=1))
    (Path(a.out) / "diagnosis.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def act_split(items: dict, R: dict) -> dict:
    """Split agreement into (1) act or not: balanced accuracy of 'does something' (cast, attack,
    block) against 'passes', over spell+hold, attack and block decisions; and (2) what, given that
    both acted: the spell among the human's casts, the attacker blocked."""
    out = {}
    def acted(t, label):
        if t in ("spell", "hold"):
            return label != "Pass"
        if t == "attack":
            return label == "yes"
        return label != "Stop Choosing"
    for group, types in (("cast", ("spell", "hold")), ("attack", ("attack",)), ("block", ("block",))):
        tp = fn = tn = fp = 0
        what_ok = what_n = 0
        for iid, r in R.items():
            it = items[iid]
            if it["type"] not in types or r.get("best") is None:
                continue
            if it["type"] == "spell":
                h_act = True  # the human cast a spell this turn
            else:
                h_act = acted(it["type"], it["label"][0])
            s_act = acted(it["type"], r["best"])
            tp += h_act and s_act
            fn += h_act and not s_act
            fp += (not h_act) and s_act
            tn += (not h_act) and (not s_act)
            if h_act and s_act and it["type"] in ("spell", "block"):
                what_n += 1
                what_ok += r["best"] in set(it["label_strict"] if it["type"] == "spell" else it["label"])
        rec_act = tp / (tp + fn) if tp + fn else None
        rec_pass = tn / (tn + fp) if tn + fp else None
        out[group] = {"bal_acc": round((rec_act + rec_pass) / 2, 3) if rec_act is not None and rec_pass is not None else None,
                      "acts": round((tp + fp) / max(1, tp + fp + tn + fn), 3), "human_acts": round((tp + fn) / max(1, tp + fp + tn + fn), 3),
                      "what": round(what_ok / what_n, 3) if what_n else None, "what_n": what_n}
    return out

