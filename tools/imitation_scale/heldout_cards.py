"""The held-out-cards test (docs/022 §4.4): does a network play a card it never saw in training? Train the GNN and the
MLP on the same games with every game that shows a few chosen cards removed, then score both on test decisions that
involve those cards, against the same networks trained without the removal.

    python tools/imitation_scale/heldout_cards.py candidates --out data/imitation_graph/heldout
    python tools/imitation_scale/heldout_cards.py choose --out data/imitation_graph/heldout [--cards "A,B,C"]
    python tools/imitation_scale/heldout_cards.py eval CKPT [CKPT ...] --cards data/imitation_graph/heldout/cards.json \\
        --tables-dir data/imitation_graph/h5 --out runs/gnn/heldout [--split test]

candidates: every card's share of the top players' training games it appears in (the player's decklist, or any card
    either player was seen to have: cast, played, on the battlefield, in a graveyard...), by rarity and type.
choose: the held-out cards (given, or picked automatically: `--k` cards appearing in `--lo`..`--hi` of the games, of
    different types and colours) and exclude_games.npy, every training game where one of them appears. The trainers
    take it as `exclude_games` (after `fraction`, so both networks lose the same games).
eval: for each checkpoint (flat or graph), on the split's priority and target tables: top-1 when the human acted and
    the set NLL, on the decisions where a held-out card is a legal option ("involved"), where the human's move is a
    held-out card ("chosen"), and on the rest ("control").

17lands public data is CC BY 4.0 (https://www.17lands.com/public_datasets).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
TOP = 0.60
BASICS = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
PRIORITY = ("turnstart", "replay_priority", "opp_priority")
TARGET = ("replay_target", "opp_block")


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(f"imitation_scale_{name}", REPO / "tools" / "imitation_scale" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def game_cards(g, ids) -> set[str]:
    """Every non-basic card name a game shows: the player's decklist and any card id in its turn records."""
    names = set(g.deck)
    for t in g.turns:
        for v in t.f.values():
            if isinstance(v, list):
                for x in v:
                    if ids.is_mapped(x):
                        try:
                            if not ids.is_token(x):
                                names.add(ids.name(x))
                        except Exception:     # noqa: BLE001 - an id that isn't a card
                            pass
    return {n for n in names if n and n not in BASICS}


def scan(limit: int | None = None) -> tuple[list[int], list[set[str]], Counter]:
    """(row, cards) of every top player's training game, and how many games each card appears in."""
    from draftzero.gameplay.ids import Ids
    from draftzero.gameplay.replay import open_lines, parse_game, replay_path
    splits = np.load(REPO / "data" / "imitation_scale" / "row_split_exp4.npy")
    ids = Ids.load()
    H, lines = open_lines(replay_path())
    wr = H.cols.index("user_game_win_rate_bucket")
    rows, cards, count = [], [], Counter()
    try:
        for i, line in enumerate(lines):
            if splits[i] != 0:
                continue
            v = line.split(",")[wr]
            if not v or float(v) < TOP:
                continue
            g = parse_game(next(csv.reader([line])), H, i)
            c = game_cards(g, ids)
            rows.append(i)
            cards.append(c)
            count.update(c)
            if limit and len(rows) >= limit:
                break
    finally:
        lines.close()
    return rows, cards, count


def card_meta(names) -> dict:
    """Types, colour identity and rarity from 17lands' cards.csv (FDN's printing where a name has several)."""
    meta = {}
    with open(REPO / "data" / "17lands" / "cards.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["name"] in names and (r["name"] not in meta or r["expansion"] == "FDN"):
                meta[r["name"]] = {"types": r["types"], "colors": r["color_identity"], "rarity": r["rarity"]}
    return {n: meta.get(n, {}) for n in names}


def choose(rows, cards, count, k: int, lo: float, hi: float, given: list[str] | None) -> list[str]:
    n = len(rows)
    if given:
        return given
    meta = card_meta(count)
    cand = sorted((c for c, m in count.items() if lo <= m / n <= hi), key=lambda c: -count[c])
    picked, kinds = [], set()
    for c in cand:
        t = str(meta.get(c, {}).get("types") or "")
        kind = next((x for x in ("Creature", "Instant", "Sorcery", "Enchantment", "Artifact") if x in t), "Other")
        col = str(meta.get(c, {}).get("colors"))
        if (kind, col) in kinds:
            continue
        picked.append(c)
        kinds.add((kind, col))
        if len(picked) >= k:
            break
    return picked


def mentions(label: str, name: str) -> bool:
    return re.search(r"(^|Cast |Play )" + re.escape(name) + r"($|[ :,])", label) is not None


def evaluate(ckpts: list[str], cards: list[str], tables_dir: Path, split: str, rows_cap: int, out: Path) -> list[dict]:
    from draftzero.gameplay import graph_supervised as gs
    from draftzero.gameplay import supervised as sv
    re_ = _tool("rare_eval")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sel, groups, data = {}, {}, {}
    for t in PRIORITY + TARGET:
        p = tables_dir / f"{t}_{split}.h5"
        with h5py.File(p, "r") as f:
            n = len(f["offsets"]) - 1
            labs = [json.loads(x) for x in f["legal_labels_json"].asstr()[:]]
            lidx = [json.loads(x) for x in f["legal_label_idx_json"].asstr()[:]]
            sp, si = f["set_indptr"][:], f["set_idx"][:]
        inv = np.array([any(mentions(lab, c) for lab in L for c in cards) for L in labs])
        chosen = np.array([any(mentions(lab, c) for j, lab in enumerate(L) for c in cards
                               if lidx[r][j] in set(si[sp[r]:sp[r + 1]].tolist())) for r, L in enumerate(labs)])
        rng = np.random.default_rng(0)
        ctrl = np.flatnonzero(~inv)
        ctrl = np.sort(rng.choice(ctrl, min(len(ctrl), rows_cap), replace=False))
        s = np.sort(np.concatenate([np.flatnonzero(inv), ctrl]))
        sel[t] = s
        groups[t] = {"involved": inv[s], "chosen": chosen[s], "control": ~inv[s]}
    results = []
    for ck in ckpts:
        try:
            model, vocab, edge_vocab, _ = gs.load_checkpoint(ck, dev)
            graph = True
        except ValueError:
            model, vocab, _ = sv.load_any_checkpoint(ck, device=dev)
            graph = False
        rec = {"checkpoint": ck, "graph": graph, "cards": cards, "split": split, "groups": {}}
        acc = {g: [[], []] for g in ("involved", "chosen", "control")}
        for t, s in sel.items():
            rows = re_.load_rows(tables_dir, t, 0, split=split, sel=s) if not graph else None
            head = "priority" if t in PRIORITY else "target"
            if graph:
                lg = re_.graph_logits(model, vocab, edge_vocab, tables_dir, t, s, dev, split=split)
                with h5py.File(tables_dir / f"{t}_{split}.h5", "r") as f:
                    lp, li = f["legal_indptr"][:], f["legal_idx"][:]
                    sp, si = f["set_indptr"][:], f["set_idx"][:]
                legal = [li[lp[r]:lp[r + 1]] for r in s]
                hum = [si[sp[r]:sp[r + 1]] for r in s]
            else:
                lg = re_.logits_for(model, vocab, [r[0] for r in rows], head, dev)
                legal, hum = [r[1] for r in rows], [r[2] for r in rows]
            for j in range(len(s)):
                top1, nll = re_.score(lg[j], legal[j], hum[j])
                for g, m in groups[t].items():
                    if m[j]:
                        if top1 is not None:
                            acc[g][0].append(top1)
                        if nll is not None:
                            acc[g][1].append(nll)
        for g, (t1, nl) in acc.items():
            rec["groups"][g] = {"n_acted": len(t1), "top1_nonpass": float(np.mean(t1)) if t1 else None,
                                "n": len(nl), "set_nll": float(np.mean(nl)) if nl else None}
        out.mkdir(parents=True, exist_ok=True)
        name = Path(ck).parent.name + "-" + Path(ck).name.split(".")[0]
        (out / f"{name}.json").write_text(json.dumps(rec, indent=1))
        print(name, json.dumps(rec["groups"]), flush=True)
        results.append(rec)
        del model
        torch.cuda.empty_cache()
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--limit", type=int)
    h = sub.add_parser("choose")
    h.add_argument("--out", type=Path, required=True)
    h.add_argument("--cards", default=None, help="comma-separated card names (default: picked automatically)")
    h.add_argument("--k", type=int, default=5)
    h.add_argument("--lo", type=float, default=0.03)
    h.add_argument("--hi", type=float, default=0.08)
    h.add_argument("--limit", type=int)
    e = sub.add_parser("eval")
    e.add_argument("checkpoints", nargs="+")
    e.add_argument("--cards", type=Path, required=True)
    e.add_argument("--tables-dir", type=Path, required=True)
    e.add_argument("--split", default="test")
    e.add_argument("--control-rows", type=int, default=5000, help="control rows a table (all involved rows are used)")
    e.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "eval":
        cards = json.loads(a.cards.read_text())["cards"]
        evaluate(a.checkpoints, cards, a.tables_dir, a.split, a.control_rows, a.out)
        return 0
    rows, cards, count = scan(a.limit)
    n = len(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "candidates":
        meta = card_meta(count)
        with open(a.out / "candidates.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["card", "games", "share", "types", "colors", "rarity"])
            for name, k in count.most_common():
                m = meta.get(name, {})
                w.writerow([name, k, round(k / n, 4), m.get("types"), m.get("colors"), m.get("rarity")])
        print(f"{n} games, {len(count)} cards -> {a.out / 'candidates.csv'}")
        return 0
    held = choose(rows, cards, count, a.k, a.lo, a.hi, [x.strip() for x in a.cards.split(",")] if a.cards else None)
    unknown = [x for x in held if x not in count]
    if unknown:
        raise SystemExit(f"cards in no training game: {unknown}")
    hs = set(held)
    ex = np.asarray([r for r, c in zip(rows, cards) if c & hs], np.int64)
    np.save(a.out / "exclude_games.npy", ex)
    info = {"cards": held, "shares": {x: round(count[x] / n, 4) for x in held}, "train_games": n,
            "excluded_games": int(len(ex)), "excluded_share": round(len(ex) / n, 4)}
    (a.out / "cards.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
