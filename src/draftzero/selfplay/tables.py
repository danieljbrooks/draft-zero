"""Game batches -> training rows (docs/021 §1.3).

One row per searched decision of a training seat (records.pack's "train"), with:
  the policy target   the search's visits over the options, normalised (soft)
  the value target    TD(lambda): the result blended backwards with the search's root values, MageZero's labels
  held out            a hash of the game puts ~heldout_share of games aside for the offline checks

GNN (graph_rows, GraphRows): the graph and each option's nodes, as graph_tables lays out the imitation tables,
kept per batch as one .npz the trainer concatenates into its window. A yes/no question's options are [no, yes]
with no nodes (the use head reads them); a row whose option has no node in the graph is dropped.
MLP (flat_tables): tools/imitation_scale/selfplay_tables.py's soft tables, for supervised.py.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np

from draftzero.selfplay import records
from draftzero.selfplay.config import REPO

ATYPE = {"PRIORITY": 0, "CHOOSE_TARGET": 3, "CHOOSE_USE": 5}
ARRAYS = ("ids", "values", "child", "parent", "label", "opt_node")             # concatenated, with pointers
PER_ROW = ("graph_type", "z", "z_td", "game", "turn", "heuristic", "version", "heldout", "n_opt")
PER_OPT = ("opt_p", "opt_q", "opt_played")


def td_targets(q: np.ndarray, result: float, lam: float) -> np.ndarray:
    """MageZero's labels: G = result; backwards over the seat's decisions, G = lam * G + (1 - lam) * q_i."""
    out = np.empty(len(q), np.float32)
    g = float(result)
    for i in range(len(q) - 1, -1, -1):
        g = lam * g + (1 - lam) * float(q[i])
        out[i] = g
    return out


def game_key(sg: dict) -> int:
    """A stable 63-bit id for a seat's game: the value cap counts positions per game."""
    g = sg["game"]
    s = f"{sg.get('run')}|{sg.get('machine')}|{sg.get('chunk')}|{g.get('pair')}|{g.get('swap')}|{sg.get('seat')}"
    return int.from_bytes(hashlib.sha1(s.encode()).digest()[:8], "big") >> 1


def is_heldout(sg: dict, share: float) -> bool:
    g = sg["game"]
    s = f"heldout|{sg.get('run')}|{sg.get('machine')}|{sg.get('chunk')}|{g.get('pair')}"
    return int.from_bytes(hashlib.sha1(s.encode()).digest()[:8], "big") / 2 ** 64 < share


# ------------------------------------------------------------------------------------------------ GNN

def graph_rows(batch: Path, lam: float = 0.99, heldout_share: float = 0.05) -> dict:
    """A batch's training rows as flat numpy arrays (concatenated graphs and options, with pointers)."""
    from draftzero.gameplay import graph_tables as gt
    parts = {k: [] for k in ARRAYS + PER_ROW + PER_OPT + ("nlen", "elen", "olen")}
    dropped = 0
    for sg in records.read(batch):
        if not sg.get("train"):          # the starting network's seat in a game against it
            continue
        recs = [r for r in sg["records"] if r.get("graph") and r["type"] in ATYPE and len(r["visits"]) >= 2
                and sum(r["visits"]) > 0]
        if not recs:
            continue
        res = sg.get("result")
        zt = td_targets(np.asarray([r["q"] for r in recs], np.float64), 0.0 if res is None else res, lam)
        if res is None:
            zt[:] = np.nan
        gk, held = game_key(sg), is_heldout(sg, heldout_share)
        for r, t in zip(recs, zt):
            g = gt.decode(r["graph"])
            use = g["type"] == ATYPE["CHOOSE_USE"]
            olen = g["opt_len"]
            if len(olen) != len(r["visits"]) or (use and len(olen) != 2) or (not use and (olen == 0).any()):
                dropped += 1
                continue
            v = np.asarray(r["visits"], np.float64)
            parts["ids"].append(g["ids"])
            parts["values"].append(g["values"])
            parts["child"].append(g["child"])
            parts["parent"].append(g["parent"])
            parts["label"].append(g["label"])
            parts["opt_node"].append(g["opt_node"])
            parts["nlen"].append(len(g["ids"]))
            parts["elen"].append(len(g["child"]))
            parts["olen"].append(olen)
            parts["opt_p"].append((v / v.sum()).astype(np.float32))
            q = r.get("opt_q") or [None] * len(v)
            parts["opt_q"].append(np.asarray([np.nan if x is None else x for x in q], np.float32))
            played = np.zeros(len(v), bool)
            if r.get("played") is not None and 0 <= r["played"] < len(v):
                played[r["played"]] = True
            parts["opt_played"].append(played)
            parts["graph_type"].append(g["type"])
            parts["z"].append(np.nan if res is None else float(res))
            parts["z_td"].append(float(t))
            parts["game"].append(gk)
            parts["turn"].append(int(r.get("turn", 0)))
            parts["heuristic"].append(float(r.get("heuristic", np.nan)))
            parts["version"].append(int(sg["version"]))
            parts["heldout"].append(held)
            parts["n_opt"].append(len(v))
    return _pack(parts, dropped)


def _pack(parts: dict, dropped: int) -> dict:
    cat = lambda k, dt: np.concatenate(parts[k]).astype(dt, copy=False) if parts[k] else np.zeros(0, dt)  # noqa: E731

    def ptr(lens):
        p = np.zeros(len(lens) + 1, np.int64)
        np.cumsum(np.asarray(lens, np.int64), out=p[1:])
        return p
    olens = cat("olen", np.int64)
    out = {"ids": cat("ids", np.int32), "values": cat("values", np.int16), "child": cat("child", np.uint16),
           "parent": cat("parent", np.uint16), "label": cat("label", np.int32), "opt_node": cat("opt_node", np.int32),
           "node_ptr": ptr(parts["nlen"]), "edge_ptr": ptr(parts["elen"]), "opt_ptr": ptr(olens),
           "row_opt_ptr": ptr(parts["n_opt"]), "opt_p": cat("opt_p", np.float32), "opt_q": cat("opt_q", np.float32),
           "opt_played": cat("opt_played", bool),
           "graph_type": np.asarray(parts["graph_type"], np.int8), "z": np.asarray(parts["z"], np.float32),
           "z_td": np.asarray(parts["z_td"], np.float32), "game": np.asarray(parts["game"], np.int64),
           "turn": np.asarray(parts["turn"], np.int32), "heuristic": np.asarray(parts["heuristic"], np.float32),
           "version": np.asarray(parts["version"], np.int32), "heldout": np.asarray(parts["heldout"], bool),
           "dropped": np.asarray(dropped, np.int64)}
    return out


def save(rows: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **rows)
    tmp.replace(path)


def load(path: Path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def concat(parts: list[dict]) -> dict:
    """Several batches' rows as one (pointers shifted)."""
    if not parts:
        return _pack({k: [] for k in ARRAYS + PER_ROW + PER_OPT + ("nlen", "elen", "olen")}, 0)
    out = {}
    for k in ("ids", "values", "child", "parent", "label", "opt_node", "opt_p", "opt_q", "opt_played", "graph_type",
              "z", "z_td", "game", "turn", "heuristic", "version", "heldout"):
        out[k] = np.concatenate([p[k] for p in parts])
    for k in ("node_ptr", "edge_ptr", "opt_ptr", "row_opt_ptr"):
        segs, base = [np.zeros(1, np.int64)], 0
        for p in parts:
            segs.append(p[k][1:] + base)
            base += int(p[k][-1])
        out[k] = np.concatenate(segs)
    out["dropped"] = np.asarray(sum(int(p["dropped"]) for p in parts), np.int64)
    return out


def n_rows(rows: dict) -> int:
    return len(rows["graph_type"])


def select(rows: dict, mask: np.ndarray) -> dict:
    """The rows where `mask` is true (pointers rebuilt)."""
    from draftzero.gameplay import graph_net as gn
    r = np.flatnonzero(mask)
    out = {k: rows[k][r] for k in ("graph_type", "z", "z_td", "game", "turn", "heuristic", "version", "heldout")}
    for arrs, ptr in ((("ids", "values"), "node_ptr"), (("child", "parent", "label"), "edge_ptr")):
        p, *xs = gn.gather(rows[ptr], r, *(rows[a] for a in arrs))
        out[ptr] = p
        out.update(zip(arrs, xs))
    p, op, oq, opl = gn.gather(rows["row_opt_ptr"], r, rows["opt_p"], rows["opt_q"], rows["opt_played"])
    out.update(row_opt_ptr=p, opt_p=op, opt_q=oq, opt_played=opl)
    opt_ids = np.repeat(rows["row_opt_ptr"][r], np.diff(p)) + (np.arange(p[-1]) - np.repeat(p[:-1], np.diff(p)))
    op2, on = gn.gather(rows["opt_ptr"], opt_ids, rows["opt_node"])
    out.update(opt_ptr=op2, opt_node=on, dropped=rows["dropped"])
    return out


def to_graph_table(rows: dict, vocab, edge_vocab, name: str = "selfplay"):
    """The rows as graph_supervised's GraphTable (mapped to the network's vocabularies), plus each option's
    target probability and each row's value target for the self-play loss (gnn_train)."""
    from draftzero.gameplay import graph_net as gn
    from draftzero.gameplay import graph_supervised as gs
    rp = rows["row_opt_ptr"]
    p = rows["opt_p"]
    row = np.repeat(np.arange(len(rp) - 1), np.diff(rp))
    best = p >= np.maximum.reduceat(p, rp[:-1])[row] if len(p) else np.zeros(0, bool)   # the most visited (ties too)
    is_pass = np.zeros(len(best), bool)
    legal_indptr, legal_idx, set_indptr, set_idx = gs._labels_view(rp, best, is_pass)
    n = len(rows["graph_type"])
    t = gs.GraphTable(name=name, kind="soft", split="train", spec={"name": name, "kind": "soft"},
                      file_rows=np.arange(n), game=rows["game"], turn=rows["turn"], z=rows["z"],
                      w=np.ones(n, np.float32), node_type=gn.node_types(rows["ids"]), node_id=rows["ids"].copy(),
                      node_value=rows["values"], node_ptr=rows["node_ptr"], edge_child=rows["child"],
                      edge_parent=rows["parent"], edge_label=rows["label"].copy(), edge_ptr=rows["edge_ptr"],
                      opt_node=rows["opt_node"], opt_ptr=rows["opt_ptr"], row_opt_ptr=rp, opt_in_set=best,
                      opt_is_pass=is_pass, graph_type=rows["graph_type"], legal_indptr=legal_indptr,
                      legal_idx=legal_idx, set_indptr=set_indptr, set_idx=set_idx)
    gs.map_table(t, vocab, edge_vocab)
    t.aux["opt_p"] = rows["opt_p"]
    t.aux["z_td"] = rows["z_td"]
    return t


# ------------------------------------------------------------------------------------------------ MLP

def _selfplay_tables():
    path = REPO / "tools" / "imitation_scale" / "selfplay_tables.py"
    spec = importlib.util.spec_from_file_location("selfplay_tables", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def flat_tables(batches: list[Path], out_dir: Path, lam: float = 0.99, heldout_share: float = 0.05,
                prefix: str = "selfplay") -> dict:
    """The flat MLP's soft tables (supervised.py's `soft` kind) from batch files: <prefix>_train.h5 and
    <prefix>_val.h5 (the held-out games)."""
    st = _selfplay_tables()
    train, val = [], []
    for b in batches:
        for sg in records.read(b):
            if not sg.get("train"):
                continue
            line = {"pair": sg["game"]["pair"], "swap": sg["game"]["swap"], "seat": sg["seat"],
                    "result": sg.get("result"), "records": [r for r in sg["records"] if r.get("features")]}
            rows = st.rows_of(line, lam)
            gk = game_key(sg)
            for r in rows:
                r["game"] = gk
            (val if is_heldout(sg, heldout_share) else train).extend(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    st.write_table(train, out_dir / f"{prefix}_train.h5")
    st.write_table(val, out_dir / f"{prefix}_val.h5")
    return {"train_rows": len(train), "val_rows": len(val)}
