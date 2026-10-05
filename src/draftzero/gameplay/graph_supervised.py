"""Imitation training for MageZero's graph network (graph_net.NetGraph) on experiment #4's decisions
(docs/022): the GNN's counterpart of supervised.py, on the same rows, losses and measures.

    python -m draftzero.gameplay.graph_supervised train --config configs/gnn_train.yml --out runs/gnn/main \\
        --tables-dir data/imitation_graph/h5
    python -m draftzero.gameplay.graph_supervised train --out runs/gnn/main --resume
    python -m draftzero.gameplay.graph_supervised sweep --spec configs/gnn_sweep.yml --out runs/gnn/sweep \\
        --tables-dir data/imitation_graph/h5
    python -m draftzero.gameplay.graph_supervised evaluate --checkpoint runs/gnn/main/best.pt.gz --split test \\
        --tables-dir data/imitation_graph/h5
    python -m draftzero.gameplay.graph_supervised bench --tables-dir data/imitation_graph/h5 [--checkpoint ...]

Data: each table is a flat table and its graph file (`build.py tables --graph`), row for row. Labels,
weights, results and metadata come from the flat table through supervised.py's own readers; the
state and the legal options come from the graph file (graph_tables). The rows are the ones
supervised.py would train and validate on with the same `fraction`, `subset_seed`, `val_rows` and
`val_seed`, so the two trainers' measures compare row for row.

What carries over from supervised.py unchanged: the set NLL over the legal options (a priority row's
label is the set of plays the player still had that turn), one-hot targets and blocks, the passivity
fix (act_weights), value_per_game positions a game, TD(lambda) value targets recomputed from the
network, the cosine schedule, and every validation measure (supervised.evaluate, fed the GNN's
per-option scores).

What differs, because the network differs:
  * a legal option is a set of graph nodes (one ability node per copy of a card, a target's node,
    Pass's node) and its logit is the log-sum-exp of their scores under the head its decision type
    reads; the softmax runs over the row's legal options, as for the flat networks (MageZero's own
    graph trainer instead softmaxes over every node of the head's types, legal or not);
  * "attack with X?" is a target choice between Stop Choosing (no) and the defending player (yes),
    read by the target head, as on the graph-encoder branch; its measures are still the binary
    table's (accuracy, AUC of P(yes));
  * the leaf and edge-label vocabularies keep ids seen in more than vocab_k training states
    (MageZero's graph rule, k = 10), and typed nodes have no vocab rows (they are their type).
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gzip
import hashlib
import io
import json
import math
import queue
import shutil
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from draftzero.gameplay import graph_net as gn
from draftzero.gameplay import graph_tables as gt
from draftzero.gameplay import supervised as sv

DEFAULT_TABLES = [{"name": "turnstart", "kind": "priority_set"},
                  {"name": "replay_priority", "kind": "priority_set"},
                  {"name": "opp_priority", "kind": "priority_set"},
                  {"name": "replay_attack", "kind": "binary"},
                  {"name": "replay_target", "kind": "target"},
                  {"name": "opp_block", "kind": "target"}]
DEFAULTS: dict[str, Any] = {
    # data (as supervised.py)
    "tables_dir": None,
    "tables": DEFAULT_TABLES,
    "fraction": 1.0, "subset_seed": 0,
    "exclude_games": None,         # a .npy of games (meta/row) dropped from training after `fraction` (docs/022 §4.4)
    "val_rows": 20000, "val_seed": 12345,
    "vocab_k": 10,                 # leaves seen in <= k training states get no row (MageZero's graph rule)
    "edge_vocab_k": 10,            # the same for edge labels
    "data_cache": None,            # a directory: the mapped graph arrays as .npy, memory-mapped on load
    "stream_cache": False,         # build data_cache a chunk of rows at a time (vocab pass, then mapping), never
                                   # holding a whole table: all the rows in ~40 GB of RAM (docs/024)
    "cache_chunk_rows": 300_000,   # rows a chunk when streaming
    # model
    "arch": dict(gn.ARCH_DEFAULT),
    # losses (supervised.py's, experiment #4's final recipes)
    "policy_weight": 1.0, "binary_weight": 1.0, "target_weight": 1.0,
    "value_weight": 0.5,
    "value_per_game": 16,
    "value_target": "td",          # result | td
    "td_lambda": 0.99,
    "td_start_epochs": 0.25,
    "td_refresh_epochs": 1.0,
    "act_weights": {"opp_priority": 3.0, "replay_priority": 3.0},
    # optimisation
    "lr": 1e-4, "warmup_steps": 3000, "lr_schedule": "cosine", "lr_min_frac": 0.1,
    "wsd_decay_frac": 0.2,         # lr_schedule wsd: warm up, hold the peak, cosine down over this last share of the
                                   # steps. Resuming with a larger max_epochs before the decay starts extends the hold
    "emb_lr_mult": 1.0,            # the leaf and edge-label embeddings' learning rate = lr x this
    "emb_init_std": None,          # re-draw the leaf, edge-label, type and value embeddings and CLS from N(0, std);
                                   # null: upstream's N(0, 1) (experiment #4's transformer trained well only at 0.02)
    "weight_decay": 0.0, "grad_clip": 1.0,
    "ema_decay": None,
    "batch_rows": 64,              # states per batch (MageZero's graph trainer: 64)
    "eval_batch_rows": 128,
    "amp": "auto", "device": "auto", "seed": 0,
    "prefetch": 3,
    # budget, evaluation, checkpoints
    "time_budget_s": 3600,
    "max_steps": None, "max_epochs": 1,
    "eval_every_s": None, "eval_every_steps": None, "eval_at_start": True,
    "ckpt_every_s": 1800, "latest_every_s": 600,
}
DATA_KEYS = ("tables_dir", "tables", "fraction", "subset_seed", "val_rows", "val_seed", "vocab_k", "edge_vocab_k",
             "exclude_games", "data_cache", "stream_cache", "cache_chunk_rows")
KIND_CODE = {"priority_set": 0, "priority_onehot": 0, "target": 1, "binary": 2}


# ================================================================================================
# configuration
# ================================================================================================

def resolve_config(*layers: dict | None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    for layer in layers:
        for k, v in (layer or {}).items():
            if k not in cfg:
                raise KeyError(f"unknown config key {k}")
            cfg[k] = {**cfg[k], **v} if k == "arch" and isinstance(v, dict) else copy.deepcopy(v)
    cfg["arch"] = gn.full_arch(cfg["arch"])
    cfg["tables"] = [sv._table_spec(t, i) for i, t in enumerate(cfg["tables"])]
    bad = [t["name"] for t in cfg["tables"] if t["kind"] not in KIND_CODE]
    if bad:
        raise ValueError(f"tables {bad}: graph training takes kinds {sorted(KIND_CODE)}")
    if cfg["tables_dir"] is None:
        cfg["tables_dir"] = "data/imitation_graph/h5"
    for k, ok in {"value_target": ("result", "td"), "lr_schedule": ("constant", "cosine", "wsd"),
                  "amp": ("auto", "bf16", "fp16", "off")}.items():
        if cfg[k] not in ok:
            raise ValueError(f"{k} must be one of {ok}, not {cfg[k]!r}")
    if cfg["lr_schedule"] == "cosine" and not (cfg["max_steps"] or cfg["max_epochs"]):
        raise ValueError("lr_schedule cosine needs max_steps or max_epochs")
    return cfg


# ================================================================================================
# data
# ================================================================================================

@dataclass
class GraphTable:
    """One split of one table: the flat table's labels (supervised._read_labels) and the graph
    file's states and options, for rows `file_rows`. legal/set CSRs index the row's options with
    Pass as 0 (supervised.PASS_IDX) and the others 1.., so supervised's measures run on them."""
    name: str
    kind: str
    split: str
    spec: dict
    file_rows: np.ndarray
    game: np.ndarray
    turn: np.ndarray
    z: np.ndarray
    w: np.ndarray
    node_type: np.ndarray            # int8 [nodes]
    node_id: np.ndarray              # int32 [nodes]: raw feature id, replaced by node_row once mapped
    node_value: np.ndarray           # int16 [nodes]
    node_ptr: np.ndarray             # int64 [n+1]
    edge_child: np.ndarray           # uint16 [edges], local to the row
    edge_parent: np.ndarray
    edge_label: np.ndarray           # int32 [edges]: raw label, replaced by edge_row once mapped
    edge_ptr: np.ndarray
    opt_node: np.ndarray             # int32 [option nodes], local to the row
    opt_ptr: np.ndarray              # int64 [options+1]
    row_opt_ptr: np.ndarray          # int64 [n+1]
    opt_in_set: np.ndarray           # bool [options]
    opt_is_pass: np.ndarray          # bool [options]
    graph_type: np.ndarray           # int8 [n]
    legal_indptr: np.ndarray
    legal_idx: np.ndarray
    set_indptr: np.ndarray
    set_idx: np.ndarray
    y: np.ndarray | None = None
    lk: np.ndarray | None = None
    lk_names: list = field(default_factory=list)
    aux: dict = field(default_factory=dict)
    zv: np.ndarray | None = None
    mapped: bool = False
    notes: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.node_ptr) - 1


def _labels_view(row_opt_ptr: np.ndarray, in_set: np.ndarray, is_pass: np.ndarray):
    """legal and set CSRs over the options: Pass 0, the others 1.. in their order."""
    n_opt = len(in_set)
    row = np.repeat(np.arange(len(row_opt_ptr) - 1), np.diff(row_opt_ptr))
    pos = np.arange(n_opt) - row_opt_ptr[:-1][row] if n_opt else np.zeros(0, np.int64)
    legal = np.where(is_pass, sv.PASS_IDX, pos + 1).astype(np.int32)
    cs = np.r_[0, np.cumsum(in_set.astype(np.int64))]
    set_ptr = cs[row_opt_ptr].astype(np.int64)
    return row_opt_ptr.astype(np.int64), legal, set_ptr, legal[in_set.astype(bool)]


def _read_graph(path: Path, sel: np.ndarray, block: int = 1 << 15) -> dict:
    """Rows `sel` (sorted) of a graph file, a block of rows at a time."""
    import h5py
    with h5py.File(path, "r") as f:
        ptrs = {k: f[k][:] for k in ("node_ptr", "edge_ptr", "row_opt_ptr", "opt_ptr")}
        gtype, missing = f["graph_type"][:], f["missing"][:]
        meta_row = f["meta/row"][:]
        n = len(ptrs["node_ptr"]) - 1
        parts: dict[str, list] = {k: [] for k in ("ids", "values", "child", "parent", "label", "opt_node",
                                                  "in_set", "is_pass", "nlen", "elen", "olen", "onlen")}
        for a in range(0, n, block):
            b = min(n, a + block)
            lo, hi = np.searchsorted(sel, [a, b])
            if lo == hi:
                continue
            pick = sel[lo:hi] - a
            npt = ptrs["node_ptr"][a:b + 1] - ptrs["node_ptr"][a]
            ept = ptrs["edge_ptr"][a:b + 1] - ptrs["edge_ptr"][a]
            rop = ptrs["row_opt_ptr"][a:b + 1] - ptrs["row_opt_ptr"][a]
            o0, o1 = ptrs["row_opt_ptr"][a], ptrs["row_opt_ptr"][b]
            opp = ptrs["opt_ptr"][o0:o1 + 1] - ptrs["opt_ptr"][o0]
            sl = lambda k, p, q: f[k][p:q]                          # noqa: E731
            nodes = (sl("node_ids", ptrs["node_ptr"][a], ptrs["node_ptr"][b]),
                     sl("node_values", ptrs["node_ptr"][a], ptrs["node_ptr"][b]))
            edges = (sl("edge_child", ptrs["edge_ptr"][a], ptrs["edge_ptr"][b]),
                     sl("edge_parent", ptrs["edge_ptr"][a], ptrs["edge_ptr"][b]),
                     sl("edge_label", ptrs["edge_ptr"][a], ptrs["edge_ptr"][b]))
            opts = (f["opt_in_set"][o0:o1], f["opt_is_pass"][o0:o1])
            onodes = f["opt_node"][ptrs["opt_ptr"][o0]:ptrs["opt_ptr"][o1]]
            _, ids, vals = gn.gather(npt, pick, *nodes)
            _, ch, pa, lb = gn.gather(ept, pick, *edges)
            _, ins, isp = gn.gather(rop, pick, *opts)
            olens = np.diff(rop)[pick]
            opt_ids = np.repeat(rop[pick] - np.r_[0, np.cumsum(olens)[:-1]], olens) + np.arange(olens.sum())
            _, on = gn.gather(opp, opt_ids, onodes)
            for k, v in (("ids", ids), ("values", vals), ("child", ch), ("parent", pa), ("label", lb),
                         ("opt_node", on), ("in_set", ins), ("is_pass", isp)):
                parts[k].append(v)
            parts["nlen"].append(np.diff(npt)[pick])
            parts["elen"].append(np.diff(ept)[pick])
            parts["olen"].append(np.diff(rop)[pick])
            parts["onlen"].append(np.diff(opp)[opt_ids])
    cat = lambda k, dt: np.concatenate(parts[k]).astype(dt, copy=False) if parts[k] else np.zeros(0, dt)  # noqa: E731

    def ptr(k):
        lens = cat(k, np.int64)
        p = np.zeros(len(lens) + 1, np.int64)
        np.cumsum(lens, out=p[1:])
        return p
    return {"ids": cat("ids", np.int32), "values": cat("values", np.int16), "node_ptr": ptr("nlen"),
            "child": cat("child", np.uint16), "parent": cat("parent", np.uint16), "label": cat("label", np.int32),
            "edge_ptr": ptr("elen"), "opt_node": cat("opt_node", np.int32), "opt_ptr": ptr("onlen"),
            "row_opt_ptr": ptr("olen"), "in_set": cat("in_set", bool), "is_pass": cat("is_pass", bool),
            "graph_type": gtype[sel].astype(np.int8), "missing": missing[sel], "meta_row": meta_row[sel]}


def load_table(cfg: dict, spec: dict, split: str, sel: np.ndarray, log=print) -> GraphTable:
    import h5py
    flat = sv.table_path(cfg, spec, split)
    gpath = gt.graph_path(flat)
    if not gpath.exists():
        raise FileNotFoundError(f"{gpath}: no graph file beside {flat.name} (build.py build/tables --graph)")
    with h5py.File(flat, "r") as f:
        lab = sv._read_labels(f, spec, sel)
    g = _read_graph(gpath, np.asarray(sel, np.int64))
    if not np.array_equal(g["meta_row"], lab["game"]):
        raise ValueError(f"{gpath}: rows don't line up with {flat.name} (meta/row differs)")
    legal_indptr, legal_idx, set_indptr, set_idx = _labels_view(g["row_opt_ptr"], g["in_set"], g["is_pass"])
    w = lab["w"].copy()
    if (g["missing"] > 0).any():
        w[g["missing"] > 0] = 0.0
    t = GraphTable(name=spec["name"], kind=spec["kind"], split=split, spec=spec, file_rows=np.asarray(sel),
                   game=lab["game"], turn=lab["turn"], z=lab["z"], w=w,
                   node_type=gn.node_types(g["ids"]), node_id=g["ids"], node_value=g["values"], node_ptr=g["node_ptr"],
                   edge_child=g["child"], edge_parent=g["parent"], edge_label=g["label"], edge_ptr=g["edge_ptr"],
                   opt_node=g["opt_node"], opt_ptr=g["opt_ptr"], row_opt_ptr=g["row_opt_ptr"],
                   opt_in_set=g["in_set"], opt_is_pass=g["is_pass"], graph_type=g["graph_type"],
                   legal_indptr=legal_indptr, legal_idx=legal_idx, set_indptr=set_indptr, set_idx=set_idx,
                   y=lab.get("y"), lk=lab.get("lk"), lk_names=lab.get("lk_names") or [], zv=lab.get("zv"),
                   notes=lab["notes"])
    if (g["missing"] > 0).any():
        t.notes.append(f"{int((g['missing'] > 0).sum())} rows have an option with no graph node: weight 0")
    if spec["kind"] == "binary" and len(t.y):
        # options are [no, yes]: the graph's label flags must agree with the flat y
        yes = t.opt_in_set[t.row_opt_ptr[:-1] + 1]
        if not np.array_equal(yes.astype(np.int64), t.y):
            raise ValueError(f"{gpath}: the yes/no labels don't match {flat.name}'s y")
    return t


def _id_state_counts(ids: np.ndarray, ptr: np.ndarray, keep: np.ndarray | None = None,
                     chunk_rows: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """(distinct ids, how many rows each occurs in), counting an id once per row; `keep` masks items."""
    out_ids, out_cnt = [], []
    n = len(ptr) - 1
    for a in range(0, n, chunk_rows):
        b = min(n, a + chunk_rows)
        x = ids[ptr[a]:ptr[b]].astype(np.int64) - np.iinfo(np.int32).min
        row = np.repeat(np.arange(b - a, dtype=np.int64), np.diff(ptr[a:b + 1]))
        if keep is not None:
            m = keep[ptr[a]:ptr[b]]
            x, row = x[m], row[m]
        pairs = np.unique(row << 32 | x)
        u, c = np.unique(pairs & 0xFFFFFFFF, return_counts=True)
        out_ids.append(u)
        out_cnt.append(c)
    if not out_ids:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    u = np.concatenate(out_ids)
    c = np.concatenate(out_cnt)
    order = np.argsort(u, kind="stable")
    u, c = u[order], c[order]
    starts = np.flatnonzero(np.r_[True, u[1:] != u[:-1]])
    return u[starts] + np.iinfo(np.int32).min, np.add.reduceat(c, starts)


def build_vocabs(tables: list[GraphTable], k: int, k_edge: int):
    """Leaf and edge-label vocabs from the training tables: ids seen in more than k states."""
    from magezero.vocab import FeatureVocab
    leaf_ids, leaf_cnt, lab_ids, lab_cnt = [], [], [], []
    for t in tables:
        u, c = _id_state_counts(t.node_id, t.node_ptr, keep=t.node_type == gn.NodeType.LEAF)
        leaf_ids.append(u)
        leaf_cnt.append(c)
        u, c = _id_state_counts(t.edge_label, t.edge_ptr)
        lab_ids.append(u)
        lab_cnt.append(c)

    mk = lambda ids: FeatureVocab(ids, feature_hash_bins=gn.GLOBAL_MAX, hash_version=2)   # noqa: E731
    return mk(_kept(leaf_ids, leaf_cnt, k)), mk(_kept(lab_ids, lab_cnt, k_edge))


def _kept(ids: list, cnt: list, k: int) -> np.ndarray:
    """The ids whose summed state counts exceed k, ascending (the vocab rule)."""
    if not ids:
        return np.zeros(0, np.int64)
    u, c = np.concatenate(ids), np.concatenate(cnt)
    order = np.argsort(u, kind="stable")
    u, c = u[order], c[order]
    starts = np.flatnonzero(np.r_[True, u[1:] != u[:-1]]) if len(u) else np.zeros(0, np.int64)
    tot = np.add.reduceat(c, starts) if len(u) else c
    return u[starts][tot > k] if len(u) else u


def _compact(vocab, edge_vocab) -> tuple:
    """The dtypes of a mapped table's node rows and edge labels: the smallest that hold the vocabs
    (int16 and int8 for MageZero's 2-3k leaves and ~20 labels: 2.5 of the 7 bytes a node and 3 of the 8
    an edge, ~20 GB on all the training rows)."""
    nd = np.int16 if len(vocab) < np.iinfo(np.int16).max else np.int32
    ed = np.int8 if len(edge_vocab) + 1 < np.iinfo(np.int8).max else np.int32
    return nd, ed


def map_table(t: GraphTable, vocab, edge_vocab) -> None:
    """Raw ids -> embedding rows, in place: node_id becomes the leaf row (-1 for typed nodes and
    leaves outside the vocab), edge_label the label row + 1 (0 outside the vocab)."""
    if t.mapped:
        return
    nd, ed = _compact(vocab, edge_vocab)
    t.node_id = np.where(t.node_type == gn.NodeType.LEAF, vocab.lookup(t.node_id), -1).astype(nd)
    t.edge_label = (edge_vocab.lookup(t.edge_label) + 1).astype(ed)
    t.mapped = True


_ARRAYS = ("node_type", "node_id", "node_value", "node_ptr", "edge_child", "edge_parent", "edge_label", "edge_ptr",
           "opt_node", "opt_ptr", "row_opt_ptr", "opt_in_set", "opt_is_pass", "graph_type")


def _cache_dir(cfg: dict, spec: dict, split: str, sel: np.ndarray, vocab, edge_vocab) -> Path | None:
    if not cfg["data_cache"]:
        return None
    p = gt.graph_path(sv.table_path(cfg, spec, split))
    st = p.stat()
    h = hashlib.sha1(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}|g2".encode())
    for a in (np.asarray(sel, np.int64), np.asarray(vocab.ids, np.int64), np.asarray(edge_vocab.ids, np.int64)):
        h.update(a.tobytes())
    return Path(cfg["data_cache"]) / f"graph_{spec['name']}_{split}_{h.hexdigest()[:16]}"


def _save_cache(d: Path, t: GraphTable) -> None:
    tmp = d.with_name(d.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for k in _ARRAYS:
        np.save(tmp / f"{k}.npy", getattr(t, k))
    if d.exists():
        shutil.rmtree(d)
    tmp.rename(d)


def _load_cache(d: Path, t: GraphTable) -> None:
    for k in _ARRAYS:
        setattr(t, k, np.load(d / f"{k}.npy", mmap_mode="r"))
    t.mapped = True


def _chunks(sel: np.ndarray, rows: int):
    for a in range(0, len(sel), rows):
        yield sel[a:a + rows]


def stream_vocabs(cfg: dict, sels: dict, log=print):
    """build_vocabs over the training rows `sels` ({table: rows}), read a chunk at a time; the same
    vocabs (ids are counted once a state either way). Cached in data_cache as vocab_<key>.npz."""
    from magezero.vocab import FeatureVocab
    h = hashlib.sha1(f"{cfg['vocab_k']}|{cfg['edge_vocab_k']}|v1".encode())
    for s in cfg["tables"]:
        p = gt.graph_path(sv.table_path(cfg, s, "train"))
        st = p.stat()
        h.update(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode())
        h.update(np.asarray(sels[s["name"]], np.int64).tobytes())
    path = Path(cfg["data_cache"]) / f"vocab_{h.hexdigest()[:16]}.npz"
    mk = lambda ids: FeatureVocab(ids, feature_hash_bins=gn.GLOBAL_MAX, hash_version=2)   # noqa: E731
    if path.exists():
        z = np.load(path)
        return mk(z["leaf"]), mk(z["edge"])
    t0 = time.monotonic()
    leaf_ids, leaf_cnt, lab_ids, lab_cnt = [], [], [], []
    for s in cfg["tables"]:
        gpath = gt.graph_path(sv.table_path(cfg, s, "train"))
        for ch in _chunks(np.asarray(sels[s["name"]], np.int64), cfg["cache_chunk_rows"]):
            g = _read_graph(gpath, ch)
            u, c = _id_state_counts(g["ids"], g["node_ptr"], keep=gn.node_types(g["ids"]) == gn.NodeType.LEAF)
            leaf_ids.append(u)
            leaf_cnt.append(c)
            u, c = _id_state_counts(g["label"], g["edge_ptr"])
            lab_ids.append(u)
            lab_cnt.append(c)
        log(f"graph_supervised: vocab pass: {s['name']} ({time.monotonic() - t0:.0f} s)")
    leaf, edge = _kept(leaf_ids, leaf_cnt, cfg["vocab_k"]), _kept(lab_ids, lab_cnt, cfg["edge_vocab_k"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, leaf=leaf, edge=edge)
    tmp.replace(path)
    return mk(leaf), mk(edge)


def _stream_cache(gpath: Path, sel: np.ndarray, vocab, edge_vocab, d: Path, chunk_rows: int, log=print) -> None:
    """A table's mapped graph arrays (rows `sel`) written into cache dir `d` a chunk at a time, as
    preallocated .npy files: the same arrays map_table + _save_cache write, never all in memory."""
    import h5py
    with h5py.File(gpath, "r") as f:
        npt, ept, rop, opp = (f[k][:] for k in ("node_ptr", "edge_ptr", "row_opt_ptr", "opt_ptr"))
    sel = np.asarray(sel, np.int64)
    n = len(sel)
    N, E = int((npt[sel + 1] - npt[sel]).sum()), int((ept[sel + 1] - ept[sel]).sum())
    O, ON = int((rop[sel + 1] - rop[sel]).sum()), int((opp[rop[sel + 1]] - opp[rop[sel]]).sum())
    nd, ed = _compact(vocab, edge_vocab)
    shapes = {"node_type": (np.int8, N), "node_id": (nd, N), "node_value": (np.int16, N), "node_ptr": (np.int64, n + 1),
              "edge_child": (np.uint16, E), "edge_parent": (np.uint16, E), "edge_label": (ed, E),
              "edge_ptr": (np.int64, n + 1), "opt_node": (np.int32, ON), "opt_ptr": (np.int64, O + 1),
              "row_opt_ptr": (np.int64, n + 1), "opt_in_set": (bool, O), "opt_is_pass": (bool, O),
              "graph_type": (np.int8, n)}
    assert set(shapes) == set(_ARRAYS)
    tmp = d.with_name(d.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    mm = {k: np.lib.format.open_memmap(tmp / f"{k}.npy", mode="w+", dtype=dt, shape=(size,))
          for k, (dt, size) in shapes.items()}
    for k in ("node_ptr", "edge_ptr", "opt_ptr", "row_opt_ptr"):
        mm[k][0] = 0
    r = nn = e = o = on = 0
    t0 = time.monotonic()
    for ch in _chunks(sel, chunk_rows):
        g = _read_graph(gpath, ch)
        k, cn, ce, co, con = len(ch), len(g["ids"]), len(g["child"]), len(g["in_set"]), len(g["opt_node"])
        types = gn.node_types(g["ids"])
        mm["node_type"][nn:nn + cn] = types
        mm["node_id"][nn:nn + cn] = np.where(types == gn.NodeType.LEAF, vocab.lookup(g["ids"]), -1).astype(nd)
        mm["node_value"][nn:nn + cn] = g["values"]
        mm["node_ptr"][r + 1:r + k + 1] = g["node_ptr"][1:] + nn
        mm["edge_child"][e:e + ce] = g["child"]
        mm["edge_parent"][e:e + ce] = g["parent"]
        mm["edge_label"][e:e + ce] = (edge_vocab.lookup(g["label"]) + 1).astype(ed)
        mm["edge_ptr"][r + 1:r + k + 1] = g["edge_ptr"][1:] + e
        mm["opt_node"][on:on + con] = g["opt_node"]
        mm["opt_ptr"][o + 1:o + co + 1] = g["opt_ptr"][1:] + on
        mm["row_opt_ptr"][r + 1:r + k + 1] = g["row_opt_ptr"][1:] + o
        mm["opt_in_set"][o:o + co] = g["in_set"]
        mm["opt_is_pass"][o:o + co] = g["is_pass"]
        mm["graph_type"][r:r + k] = g["graph_type"]
        r, nn, e, o, on = r + k, nn + cn, e + ce, o + co, on + con
        for a in mm.values():      # write back now: dirty pages count against a container's memory limit (docs/024)
            a.flush()
    if (r, nn, e, o, on) != (n, N, E, O, ON):
        raise RuntimeError(f"{gpath}: streamed {(r, nn, e, o, on)} against {(n, N, E, O, ON)}")
    for a in mm.values():
        a.flush()
    del mm
    if d.exists():
        shutil.rmtree(d)
    tmp.rename(d)
    log(f"graph_supervised: cached {gpath.name}: {n} rows, {N} nodes, {E} edges ({time.monotonic() - t0:.0f} s)")


def load_table_cached(cfg: dict, spec: dict, split: str, sel: np.ndarray, vocab, edge_vocab, log=print) -> GraphTable:
    """load_table + map_table for stream_cache: the labels from the flat table, the graph arrays
    memory-mapped from data_cache (streamed into it first if they aren't there)."""
    import h5py
    flat = sv.table_path(cfg, spec, split)
    gpath = gt.graph_path(flat)
    if not gpath.exists():
        raise FileNotFoundError(f"{gpath}: no graph file beside {flat.name} (build.py build/tables --graph)")
    sel = np.asarray(sel, np.int64)
    with h5py.File(flat, "r") as f:
        lab = sv._read_labels(f, spec, sel)
    with h5py.File(gpath, "r") as f:
        missing, meta_row = f["missing"][:][sel], f["meta/row"][:][sel]
    if not np.array_equal(meta_row, lab["game"]):
        raise ValueError(f"{gpath}: rows don't line up with {flat.name} (meta/row differs)")
    cd = _cache_dir(cfg, spec, split, sel, vocab, edge_vocab)
    if not (cd / "graph_type.npy").exists():
        _stream_cache(gpath, sel, vocab, edge_vocab, cd, cfg["cache_chunk_rows"], log)
    a = {k: np.load(cd / f"{k}.npy", mmap_mode="r") for k in _ARRAYS}
    legal_indptr, legal_idx, set_indptr, set_idx = _labels_view(np.asarray(a["row_opt_ptr"]), np.asarray(a["opt_in_set"]),
                                                                np.asarray(a["opt_is_pass"]))
    w = lab["w"].copy()
    if (missing > 0).any():
        w[missing > 0] = 0.0
    t = GraphTable(name=spec["name"], kind=spec["kind"], split=split, spec=spec, file_rows=sel,
                   game=lab["game"], turn=lab["turn"], z=lab["z"], w=w,
                   node_type=a["node_type"], node_id=a["node_id"], node_value=a["node_value"], node_ptr=a["node_ptr"],
                   edge_child=a["edge_child"], edge_parent=a["edge_parent"], edge_label=a["edge_label"],
                   edge_ptr=a["edge_ptr"], opt_node=a["opt_node"], opt_ptr=a["opt_ptr"], row_opt_ptr=a["row_opt_ptr"],
                   opt_in_set=a["opt_in_set"], opt_is_pass=a["opt_is_pass"], graph_type=a["graph_type"],
                   legal_indptr=legal_indptr, legal_idx=legal_idx, set_indptr=set_indptr, set_idx=set_idx,
                   y=lab.get("y"), lk=lab.get("lk"), lk_names=lab.get("lk_names") or [], zv=lab.get("zv"),
                   notes=lab["notes"], mapped=True)
    if (missing > 0).any():
        t.notes.append(f"{int((missing > 0).sum())} rows have an option with no graph node: weight 0")
    if spec["kind"] == "binary" and len(t.y):
        yes = np.asarray(t.opt_in_set)[t.row_opt_ptr[:-1] + 1]
        if not np.array_equal(yes.astype(np.int64), t.y):
            raise ValueError(f"{gpath}: the yes/no labels don't match {flat.name}'s y")
    return t


@dataclass
class Data:
    vocab: Any
    edge_vocab: Any
    train: list
    val: list
    info: dict


def load_data(cfg: dict, *, vocabs=None, splits: tuple = ("train", "val"), log=print) -> Data:
    """The training rows (a `fraction` of the training games) and the validation rows (whole games,
    `val_rows` per table), chosen exactly as supervised.load_data chooses them; the leaf and edge
    vocabs built on the training rows (unless given)."""
    cfg = resolve_config(cfg)
    t0 = time.monotonic()
    specs = cfg["tables"]
    info: dict[str, Any] = {"tables": {}}
    train, val = [], []
    if "train" in splits:
        games = {s["name"]: sv._games_of(sv.table_path(cfg, s, "train")) for s in specs}
        allg = np.unique(np.concatenate(list(games.values())))
        keep_g = allg
        if cfg["fraction"] < 1:
            rng = np.random.default_rng(cfg["subset_seed"])
            keep_g = np.sort(rng.choice(allg, max(1, int(round(cfg["fraction"] * len(allg)))), replace=False))
        keep_g = sv.drop_excluded(keep_g, cfg["exclude_games"], log)
        sel = {s["name"]: np.flatnonzero(np.isin(games[s["name"]], keep_g)) for s in specs}
        info.update(train_games=int(len(keep_g)), train_games_all=int(len(allg)))
        if cfg["stream_cache"]:
            if not cfg["data_cache"]:
                raise ValueError("stream_cache needs data_cache (a directory for the memory-mapped arrays)")
            if vocabs is None:
                vocabs = stream_vocabs(cfg, sel, log)
                log(f"graph_supervised: vocabs {len(vocabs[0])} leaves (k={cfg['vocab_k']}), {len(vocabs[1])} edge "
                    f"labels ({time.monotonic() - t0:.0f} s)")
            train = [load_table_cached(cfg, s, "train", sel[s["name"]], *vocabs, log=log) for s in specs]
        else:
            train = [load_table(cfg, s, "train", sel[s["name"]], log=log) for s in specs]
    for split in [x for x in splits if x != "train"]:
        games = {s["name"]: sv._games_of(sv.table_path(cfg, s, split)) for s in specs}
        vs = sv._val_selection(games, cfg["val_rows"], cfg["val_seed"])
        if cfg["stream_cache"] and vocabs is not None:
            val += [load_table_cached(cfg, s, split, vs[s["name"]], *vocabs, log=log) for s in specs]
        else:
            val += [load_table(cfg, s, split, vs[s["name"]], log=log) for s in specs]
    if vocabs is None:
        if not train:
            raise ValueError("no training rows to build the vocabs on: pass vocabs (a checkpoint's)")
        vocabs = build_vocabs(train, cfg["vocab_k"], cfg["edge_vocab_k"])
        log(f"graph_supervised: vocabs {len(vocabs[0])} leaves (k={cfg['vocab_k']}), {len(vocabs[1])} edge labels "
            f"({time.monotonic() - t0:.0f} s)")
    vocab, edge_vocab = vocabs
    for t in train + val:
        cd = _cache_dir(cfg, t.spec, t.split, t.file_rows, vocab, edge_vocab)
        if cd is not None and (cd / "graph_type.npy").exists():
            _load_cache(cd, t)
            continue
        map_table(t, vocab, edge_vocab)
        if cd is not None:
            _save_cache(cd, t)
            _load_cache(cd, t)
    for t in train + val:
        m = min(len(t.node_type), 50_000_000)       # a memory-mapped table: its first nodes are enough
        leaves = np.asarray(t.node_type[:m]) == gn.NodeType.LEAF
        info["tables"][f"{t.name}_{t.split}"] = {
            "rows": t.n, "games": int(len(np.unique(t.game))),
            "nodes_per_row": round(float(t.node_ptr[-1]) / max(t.n, 1), 1),
            "edges_per_row": round(float(t.edge_ptr[-1]) / max(t.n, 1), 1),
            "leaves_in_vocab": round(float((np.asarray(t.node_id[:m])[leaves] >= 0).mean()) if leaves.any() else 1.0, 4)}
    info.update(vocab_rows=len(vocab), edge_vocab_rows=len(edge_vocab), load_s=round(time.monotonic() - t0, 1))
    log(f"graph_supervised: data loaded in {info['load_s']} s: " + ", ".join(
        f"{k} {v['rows']} rows" for k, v in info["tables"].items()))
    return Data(vocab, edge_vocab, train, val, info)


def with_act_weights(tables: list[GraphTable], weights: dict, log=print) -> list[GraphTable]:
    """supervised.with_act_weights for graph tables: x f on rows whose label holds a non-Pass play."""
    unknown = set(weights) - {t.name for t in tables}
    if unknown:
        raise ValueError(f"act_weights names tables not in the data: {sorted(unknown)}")
    out = []
    for t in tables:
        f = weights.get(t.name)
        if f is None or f == 1:
            out.append(t)
            continue
        a = sv.acted_rows(t)
        t2 = copy.copy(t)
        t2.w = np.where(a, t.w * float(f), t.w).astype(np.float32)
        out.append(t2)
        log(f"graph_supervised: {t.name}: policy weight x{f:g} on {int(a.sum())} of {t.n} rows where the human acted")
    return out


# ================================================================================================
# batches
# ================================================================================================

def make_batch(tables: list[GraphTable], rows_by_table: list[tuple[int, np.ndarray]], pin: bool = False) -> dict:
    """The network's input and the option layout for rows of several tables ([(table index, rows)]).
    Batch-wide indices throughout; `pos` maps each batch row back to (table, row)."""
    parts = {k: [] for k in ("ntype", "nrow", "nval", "nlen", "ec", "ep", "el", "orow", "onode", "onopt",
                             "oidx", "oin", "gtype", "tab", "row")}
    node_base = row_base = opt_base = 0
    for ti, r in rows_by_table:
        t = tables[ti]
        r = np.asarray(r, np.int64)
        nptr, ntype, nrow, nval = gn.gather(t.node_ptr, r, t.node_type, t.node_id, t.node_value)
        eptr, ec, ep, el = gn.gather(t.edge_ptr, r, t.edge_child, t.edge_parent, t.edge_label)
        shift = nptr[:-1] + node_base
        erow = np.repeat(np.arange(len(r)), np.diff(eptr))
        optr, oin = gn.gather(t.row_opt_ptr, r, t.opt_in_set)
        orow = np.repeat(np.arange(len(r)), np.diff(optr))
        opt_ids = np.repeat(t.row_opt_ptr[r] - optr[:-1], np.diff(optr)) + np.arange(optr[-1])
        onptr, onode = gn.gather(t.opt_ptr, opt_ids, t.opt_node)
        onopt = np.repeat(np.arange(len(opt_ids)), np.diff(onptr))
        parts["ntype"].append(ntype)
        parts["nrow"].append(nrow)
        parts["nval"].append(nval)
        parts["nlen"].append(np.diff(nptr))
        parts["ec"].append(ec.astype(np.int64) + shift[erow])
        parts["ep"].append(ep.astype(np.int64) + shift[erow])
        parts["el"].append(el)
        parts["orow"].append(orow + row_base)
        parts["onode"].append(onode.astype(np.int64) + shift[orow[onopt]])
        parts["onopt"].append(onopt + opt_base)
        parts["oidx"].append(np.arange(len(opt_ids)) - optr[:-1][orow])
        parts["oin"].append(oin)
        parts["gtype"].append(t.graph_type[r])
        parts["tab"].append(np.full(len(r), ti, np.int64))
        parts["row"].append(r)
        node_base += int(nptr[-1])
        row_base += len(r)
        opt_base += len(opt_ids)
    T = lambda k, dt: torch.from_numpy(np.ascontiguousarray(np.concatenate(parts[k]).astype(dt, copy=False)))  # noqa: E731
    offs = np.zeros(row_base + 1, np.int64)
    np.cumsum(np.concatenate(parts["nlen"]), out=offs[1:])
    graphs = gn.Graphs(T("ntype", np.int64), T("nrow", np.int64), T("nval", np.int64), torch.from_numpy(offs),
                       T("ec", np.int64), T("ep", np.int64), T("el", np.int64))
    b = {"graphs": graphs, "opt": gn.OptionBatch(T("orow", np.int64), T("onode", np.int64), T("onopt", np.int64), opt_base),
         "opt_index": T("oidx", np.int64), "opt_in": T("oin", bool), "graph_type": T("gtype", np.int64),
         "tab": np.concatenate(parts["tab"]), "row": np.concatenate(parts["row"])}
    if pin:
        b["graphs"] = gn.Graphs(*(x.pin_memory() for x in b["graphs"]))
    return b


def to_device(b: dict, dev) -> dict:
    out = dict(b)
    out["graphs"] = b["graphs"].to(dev)
    out["opt"] = b["opt"].to(dev)
    for k in ("opt_index", "opt_in", "graph_type"):
        out[k] = b[k].to(dev, non_blocking=True)
    return out


def row_losses(out: gn.Out, b: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Per batch row: the set NLL over its legal options (inf where its label set is empty), and
    each option's logit."""
    ob = b["opt"]
    B = b["graph_type"].shape[0]
    lg = gn.option_logits(out, ob, b["graph_type"], b["opt_index"])
    lg = torch.where(torch.isfinite(lg), lg, torch.full_like(lg, -1e4))
    lse_all = gn.segment_logsumexp(lg, ob.opt_row, B)
    m = b["opt_in"].nonzero().squeeze(1)
    lse_set = gn.segment_logsumexp(lg[m], ob.opt_row[m], B)
    return lse_all - lse_set, lg


class Sampler:
    """Every training row once an epoch, in a fresh random order (mixed tables), batch_rows at a
    time; resumable from (epoch, position)."""

    def __init__(self, tables: list[GraphTable], batch_rows: int, seed: int):
        self.tables, self.B, self.seed = tables, batch_rows, seed
        self.off = np.r_[0, np.cumsum([t.n for t in tables])].astype(np.int64)
        self.epoch, self.pos = 0, 0
        self._perm = None

    @property
    def rows(self) -> int:
        return int(self.off[-1])

    def _order(self) -> np.ndarray:
        if self._perm is None or self._perm[0] != self.epoch:
            self._perm = (self.epoch, np.random.default_rng([self.seed, 0x47, self.epoch]).permutation(self.rows))
        return self._perm[1]

    def next(self) -> tuple[np.ndarray, int]:
        order = self._order()
        if self.pos >= len(order):
            self.epoch, self.pos = self.epoch + 1, 0
            order = self._order()
        g = order[self.pos:self.pos + self.B]
        ep = self.epoch
        self.pos += len(g)
        return np.sort(g), ep

    def split(self, g: np.ndarray) -> list[tuple[int, np.ndarray]]:
        ti = np.searchsorted(self.off, g, side="right") - 1
        return [(int(i), g[ti == i] - self.off[i]) for i in np.unique(ti)]

    def state(self) -> dict:
        return {"epoch": self.epoch, "pos": self.pos}

    def load(self, st: dict) -> None:
        self.epoch, self.pos = int(st["epoch"]), int(st["pos"])


class Prefetcher:
    def __init__(self, sampler: Sampler, depth: int, pin: bool):
        self.s, self.pin = sampler, pin
        self.q: queue.Queue = queue.Queue(maxsize=max(1, depth))
        self.stop = threading.Event()
        self.err: BaseException | None = None
        self.th = None
        if depth > 0:
            self.th = threading.Thread(target=self._run, daemon=True)
            self.th.start()

    def _one(self):
        g, ep = self.s.next()
        b = make_batch(self.s.tables, self.s.split(g), pin=self.pin)
        b["g"], b["epoch"], b["state"] = g, ep, self.s.state()
        return b

    def _run(self):
        try:
            while not self.stop.is_set():
                b = self._one()
                while not self.stop.is_set():
                    try:
                        self.q.put(b, timeout=0.2)
                        break
                    except queue.Full:
                        continue
        except BaseException as e:      # noqa: BLE001
            self.err = e

    def get(self) -> dict:
        if self.th is None:
            return self._one()
        while True:
            if self.err is not None:
                raise self.err
            try:
                return self.q.get(timeout=0.5)
            except queue.Empty:
                continue

    def close(self):
        self.stop.set()
        if self.th is not None:
            self.th.join(timeout=10)


# ================================================================================================
# evaluation
# ================================================================================================

@torch.no_grad()
def table_outputs(model, t: GraphTable, dev, dtype, batch_rows: int = 128) -> dict:
    """supervised.table_outputs' fields for a graph table: vx, per-row option scores (aligned with
    legal_idx) and set NLL, and for binary tables P(yes) and its cross-entropy."""
    out: dict[str, Any] = {"vx": np.zeros(t.n, np.float32), "scores": [None] * t.n, "nll": np.full(t.n, np.nan)}
    if t.kind == "binary":
        out["p_yes"], out["ce"] = np.zeros(t.n), np.zeros(t.n)
    for a in range(0, t.n, batch_rows):
        r = np.arange(a, min(t.n, a + batch_rows))
        b = to_device(make_batch([t], [(0, r)]), dev)
        with sv._autocast(dev, dtype):
            o = model(b["graphs"])
        nll, lg = row_losses(o, b)
        out["vx"][r] = o.value_x.float().cpu().numpy()
        out["nll"][r] = nll.float().cpu().numpy()
        lg = lg.float().cpu().numpy().astype(np.float64)
        rop = t.row_opt_ptr
        base = rop[r[0]]
        for j, row in enumerate(r):
            out["scores"][row] = lg[rop[row] - base:rop[row + 1] - base]
        if t.kind == "binary":
            for j, row in enumerate(r):
                s = out["scores"][row]
                p = np.exp(s - s.max())
                p /= p.sum()
                out["p_yes"][row] = p[1]
                out["ce"][row] = -math.log(max(p[int(t.y[row])], 1e-12))
    out["nll"][~np.isfinite(out["nll"])] = np.nan
    return out


def evaluate(model, tables: list[GraphTable], cfg: dict, dev, dtype=None) -> dict:
    """supervised.evaluate's measures, every one computed by its code from the GNN's outputs."""
    return sv.evaluate(model, tables, {**cfg, "eval_batch_rows": cfg["eval_batch_rows"]}, dev, dtype,
                       outputs=lambda m, t: table_outputs(m, t, dev, dtype, cfg["eval_batch_rows"]))


@torch.no_grad()
def values(model, tables: list[GraphTable], dev, dtype, batch_rows: int = 256) -> np.ndarray:
    """v = tanh(x) for every row of `tables`, in order (the TD targets' input)."""
    was = model.training
    model.eval()
    out = []
    for ti, t in enumerate(tables):
        for a in range(0, t.n, batch_rows):
            r = np.arange(a, min(t.n, a + batch_rows))
            b = make_batch([t], [(0, r)])
            with sv._autocast(dev, dtype):
                out.append(model(b["graphs"].to(dev)).value.float().cpu().numpy())
    if was:
        model.train()
    return np.concatenate(out) if out else np.zeros(0, np.float32)


# ================================================================================================
# checkpoints
# ================================================================================================

def vocab_state(v) -> dict:
    """MageZero's graph vocab format (vocab.py on graph-encoder: format 2, hash version 2)."""
    return {"format_version": 2, "ids": torch.from_numpy(np.asarray(v.ids, np.int64).copy()),
            "feature_hash_bins": gn.GLOBAL_MAX, "hash_algorithm": "xmage_feature_hash", "hash_version": 2}


def save_weights(path: Path, model, vocab, edge_vocab, info: dict) -> None:
    """gzip'd torch.save in MageZero's graph checkpoint layout (model_state_dict, feature_vocab,
    edge_vocab), so its own server loads a default-shaped network; plus arch and our run info."""
    buf = io.BytesIO()
    torch.save({"epoch": info.get("epoch", 0), "model_state_dict": model.state_dict(),
                "feature_vocab": vocab_state(vocab), "edge_vocab": vocab_state(edge_vocab),
                "arch": model.arch, "graph_supervised": sv._jsonable(info),
                "upstream_loadable": gn.upstream_loadable(model.arch)}, buf)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=1) as f:
        f.write(buf.getvalue())
    tmp.replace(path)


def load_checkpoint(path, device="cpu"):
    """(model in eval mode, leaf vocab, edge vocab, meta) from a graph checkpoint (ours or MageZero's)."""
    from magezero.vocab import FeatureVocab
    p = str(path)
    if p.startswith("hf://"):
        from draftzero.gameplay.imitation_net import resolve
        p = str(resolve(p))
    opener = gzip.open if p.endswith(".gz") else open
    with opener(p, "rb") as f:
        ck = torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)
    if "edge_vocab" not in ck:
        raise ValueError(f"{path} is not a graph checkpoint (no edge vocab)")
    vocab = FeatureVocab.from_state_dict(ck["feature_vocab"])
    edge_vocab = FeatureVocab.from_state_dict(ck["edge_vocab"])
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck["model"]
    model = gn.NetGraph(len(vocab), len(edge_vocab), ck.get("arch"))
    model.load_state_dict(sd)
    return model.to(device).eval(), vocab, edge_vocab, {"arch": model.arch, "info": ck.get("graph_supervised")}


# ================================================================================================
# training
# ================================================================================================

class EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    @contextlib.contextmanager
    def swapped(self, model):
        live = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        try:
            yield
        finally:
            model.load_state_dict(live)


class Trainer:
    def __init__(self, cfg: dict, data: Data, out: Path, log=print, resume: bool = False):
        self.cfg, self.data, self.out, self.log = cfg, data, out, log
        out.mkdir(parents=True, exist_ok=True)
        self.dev = sv.pick_device(cfg["device"])
        self.dtype = sv.amp_dtype(cfg["amp"], self.dev)
        torch.manual_seed(cfg["seed"])
        self.train_tables = with_act_weights(data.train, cfg["act_weights"], log=log)
        self.model = gn.NetGraph(len(data.vocab), len(data.edge_vocab), cfg["arch"]).to(self.dev)
        if cfg["emb_init_std"] is not None:
            with torch.no_grad():
                for e in (self.model.embedding, self.model.edge_embedding, self.model.type_embedding,
                          self.model.value_embedding):
                    e.weight.normal_(0.0, float(cfg["emb_init_std"]))
                self.model.edge_embedding.weight[0].zero_()          # the unknown label stays the padding row
                self.model.cls.normal_(0.0, float(cfg["emb_init_std"]))
        emb = [p for n, p in self.model.named_parameters() if n in ("embedding.weight", "edge_embedding.weight")]
        rest = [p for n, p in self.model.named_parameters() if n not in ("embedding.weight", "edge_embedding.weight")]
        kw = {"fused": True} if self.dev.type == "cuda" else {}
        self.opt = torch.optim.AdamW([{"params": rest, "lr_mult": 1.0},
                                      {"params": emb, "lr_mult": float(cfg["emb_lr_mult"])}],
                                     lr=cfg["lr"], weight_decay=cfg["weight_decay"], **kw)
        self.ema = EMA(self.model, float(cfg["ema_decay"])) if cfg["ema_decay"] else None
        self.sampler = Sampler(self.train_tables, cfg["batch_rows"], cfg["seed"])
        self.epoch_rows = self.sampler.rows
        steps_per_epoch = max(1, math.ceil(self.epoch_rows / cfg["batch_rows"]))
        self.total_steps = cfg["max_steps"] or (int(cfg["max_epochs"] * steps_per_epoch) if cfg["max_epochs"] else None)
        cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)   # noqa: E731
        tr = self.train_tables
        self.g_game = cat([t.game for t in tr], np.int64)
        self.g_turn = cat([t.turn for t in tr], np.int32)
        self.g_phase = cat([np.full(t.n, t.spec["phase"]) for t in tr], np.int32)
        self.g_z = cat([t.z if t.zv is None else t.zv for t in tr], np.float32)
        self.g_ok = cat([np.isfinite(t.z) & bool(t.spec["value"]) for t in tr], bool)
        self.g_w = cat([t.w for t in tr], np.float32)
        self.g_kw = cat([np.full(t.n, {"priority_set": cfg["policy_weight"], "priority_onehot": cfg["policy_weight"],
                                       "target": cfg["target_weight"], "binary": cfg["binary_weight"]}[t.kind])
                         for t in tr], np.float32)
        self.targets = self.g_z.copy()
        self.vmask_epoch, self.vmask = -1, None
        self.step, self.seen, self.train_s, self.td_refreshes = 0, 0, 0.0, 0
        self.next_td = cfg["td_start_epochs"] if cfg["value_target"] == "td" else None
        self.best = {"policy": math.inf, "value": math.inf, "combined": math.inf}
        self.best_eval: dict = {}
        self.sessions = 0
        if resume and (out / "latest.pt").exists():
            self._load_latest()

    # -- schedule and value targets ----------------------------------------------------------------
    def lr_at(self, step: int) -> float:
        c = self.cfg
        if step < c["warmup_steps"]:
            return c["lr"] * (step + 1) / c["warmup_steps"]
        if c["lr_schedule"] == "constant" or not self.total_steps:
            return c["lr"]
        lo = c["lr"] * c["lr_min_frac"]
        if c["lr_schedule"] == "wsd":
            start = max(c["warmup_steps"], int(self.total_steps * (1 - c["wsd_decay_frac"])))
            if step < start:
                return c["lr"]
            p = min(1.0, (step - start) / max(1, self.total_steps - start))
        else:
            p = min(1.0, (step - c["warmup_steps"]) / max(1, self.total_steps - c["warmup_steps"]))
        return lo + (c["lr"] - lo) * 0.5 * (1 + math.cos(math.pi * p))

    def value_mask(self, epoch: int) -> np.ndarray:
        if self.vmask_epoch != epoch:
            k = self.cfg["value_per_game"]
            self.vmask = (sv.value_subsample_mask(self.g_game, k, self.cfg["seed"], epoch, self.g_ok)
                          if k else self.g_ok.copy())
            self.vmask_epoch = epoch
        return self.vmask

    def refresh_td(self) -> None:
        t0 = time.monotonic()
        v = values(self.model, self.train_tables, self.dev, self.dtype)
        self.targets = sv.td_lambda_targets(self.g_game, self.g_turn, self.g_phase, v, self.g_z,
                                            self.cfg["td_lambda"]).astype(np.float32)
        self.td_refreshes += 1
        self.log(f"graph_supervised: TD({self.cfg['td_lambda']}) targets from the network ({time.monotonic() - t0:.0f} s)")

    # -- one step ----------------------------------------------------------------------------------
    def train_step(self, b: dict) -> dict:
        cfg, dev = self.cfg, self.dev
        g = b["g"]
        w = torch.from_numpy(self.g_w[g] * self.g_kw[g]).to(dev)
        vm_np = self.value_mask(b["epoch"])[g]
        vm = torch.from_numpy(vm_np).to(dev)
        tgt = torch.from_numpy(self.targets[g]).to(dev)
        bd = to_device(b, dev)
        lr = self.lr_at(self.step)
        for grp in self.opt.param_groups:
            grp["lr"] = lr * grp["lr_mult"]
        with sv._autocast(dev, self.dtype):
            out = self.model(bd["graphs"])
        with sv._fp32(dev, self.dtype):
            nll, _ = row_losses(out, bd)
            ok = (w > 0) & torch.isfinite(nll)
            pol = (torch.where(ok, nll, torch.zeros_like(nll)) * w).sum() / ok.sum().clamp(min=1)
            if vm.any():
                vl = sv.value_bce_from_logit(out.value_x[vm], tgt[vm]).mean()
            else:
                vl = out.value_x.sum() * 0
            loss = pol + cfg["value_weight"] * vl
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg["grad_clip"])
        self.opt.step()
        if self.ema is not None:
            self.ema.update(self.model)
        self.step += 1
        self.seen += len(g)
        return {"loss": loss.item(), "policy": pol.item(), "value": vl.item(), "lr": lr}

    # -- evaluation and checkpoints -----------------------------------------------------------------
    def do_eval(self) -> dict:
        t0 = time.monotonic()
        ctx = self.ema.swapped(self.model) if self.ema is not None else contextlib.nullcontext()
        with ctx:
            m = evaluate(self.model, self.data.val, self.cfg, self.dev, self.dtype)
            rec = {"step": self.step, "seen": self.seen, "epochs": round(self.seen / max(1, self.epoch_rows), 4),
                   "train_s": round(self.train_s, 1), **m}
            for key, metric in (("policy", "select/policy_loss"), ("value", "select/value_logloss"),
                                ("combined", "select/combined")):
                v = m.get(metric)
                if v is not None and v < self.best[key]:
                    self.best[key] = v
                    self.best_eval[key] = rec
                    name = {"policy": "best_policy", "value": "best_value", "combined": "best"}[key]
                    save_weights(self.out / f"{name}.pt.gz", self.model, self.data.vocab, self.data.edge_vocab,
                                 self._info(rec))
        rec["eval_s"] = round(time.monotonic() - t0, 1)
        with open(self.out / "evals.jsonl", "a") as f:
            f.write(json.dumps(sv._jsonable(rec)) + "\n")
        self.log(f"graph_supervised: step {self.step} ({rec['epochs']:.3f} epochs): set NLL {m.get('policy/set_nll')}, "
                 f"top-1 acted {m.get('policy/top1_nonpass')}, attack acc {m.get('binary/acc')}, "
                 f"value AUC {m.get('value/auc')} ({rec['eval_s']} s)")
        return rec

    def _info(self, rec: dict | None = None) -> dict:
        return {"step": self.step, "seen": self.seen, "epoch": self.seen / max(1, self.epoch_rows),
                "train_time_s": self.train_s, "metrics": rec, "config": self.cfg, "data": self.data.info}

    def save_latest(self) -> None:
        st = {"model": self.model.state_dict(), "opt": self.opt.state_dict(),
              "ema": self.ema.shadow if self.ema is not None else None, "sampler": self.sampler.state(),
              "step": self.step, "seen": self.seen, "train_s": self.train_s, "td_refreshes": self.td_refreshes,
              "next_td": self.next_td, "targets": self.targets, "best": self.best, "best_eval": self.best_eval,
              "vocab": vocab_state(self.data.vocab), "edge_vocab": vocab_state(self.data.edge_vocab),
              "config": self.cfg, "sessions": self.sessions, "rng": torch.get_rng_state()}
        tmp = self.out / "latest.pt.tmp"
        torch.save(st, tmp)
        tmp.replace(self.out / "latest.pt")

    def _load_latest(self) -> None:
        # onto the CPU first: the vocab ids and the RNG state must stay there; the model's and the optimizer's
        # load_state_dict move theirs to the parameters' device, and the EMA shadow is moved below
        st = torch.load(self.out / "latest.pt", map_location="cpu", weights_only=False)
        if not np.array_equal(np.asarray(st["vocab"]["ids"]), np.asarray(self.data.vocab.ids)):
            raise ValueError("latest.pt's leaf vocab differs from this data's: load the data with its vocabs")
        self.model.load_state_dict(st["model"])
        self.opt.load_state_dict(st["opt"])
        if self.ema is not None and st["ema"] is not None:
            self.ema.shadow = {k: v.to(self.dev) for k, v in st["ema"].items()}
        self.sampler.load(st["sampler"])
        for k in ("step", "seen", "train_s", "td_refreshes", "next_td", "targets", "best", "best_eval", "sessions"):
            setattr(self, k, st[k])
        torch.set_rng_state(st["rng"])
        self.log(f"graph_supervised: resumed at step {self.step} ({self.seen / max(1, self.epoch_rows):.3f} epochs)")

    # -- the loop ---------------------------------------------------------------------------------
    def run(self) -> dict:
        cfg = self.cfg
        self.sessions += 1
        (self.out / "config.json").write_text(json.dumps(sv._jsonable(cfg), indent=1))
        stop_flag = {"stop": None}

        def on_term(*_):
            stop_flag["stop"] = "sigterm"
        old = signal.signal(signal.SIGTERM, on_term) if threading.current_thread() is threading.main_thread() else None
        if cfg["eval_at_start"] and self.step == 0:
            self.do_eval()
        pf = Prefetcher(self.sampler, cfg["prefetch"], pin=self.dev.type == "cuda")
        last_eval = last_ckpt = last_latest = time.monotonic()
        last_eval_step = self.step
        losses, t_start = [], time.monotonic()
        stop = None
        try:
            while True:
                epochs = self.seen / max(1, self.epoch_rows)
                if self.total_steps and self.step >= self.total_steps:
                    stop = "max_steps" if cfg["max_steps"] else "max_epochs"
                elif cfg["max_epochs"] and epochs >= cfg["max_epochs"]:
                    stop = "max_epochs"
                elif self.train_s >= cfg["time_budget_s"]:
                    stop = "time_budget"
                elif stop_flag["stop"]:
                    stop = stop_flag["stop"]
                if stop:
                    break
                if self.next_td is not None and epochs >= self.next_td:
                    self.refresh_td()
                    self.next_td += cfg["td_refresh_epochs"]
                t0 = time.monotonic()
                b = pf.get()
                losses.append(self.train_step(b))
                self.train_s += time.monotonic() - t0
                now = time.monotonic()
                if (cfg["eval_every_steps"] and self.step - last_eval_step >= cfg["eval_every_steps"]) or \
                        (cfg["eval_every_s"] and now - last_eval >= cfg["eval_every_s"]):
                    tl = {k: float(np.mean([x[k] for x in losses])) for k in ("loss", "policy", "value")}
                    self.log(f"graph_supervised: train {tl} lr {losses[-1]['lr']:.2e} "
                             f"({self.seen / max(1e-9, self.train_s):.0f} states/s)")
                    losses = []
                    self.do_eval()
                    last_eval, last_eval_step = time.monotonic(), self.step
                if cfg["ckpt_every_s"] and now - last_ckpt >= cfg["ckpt_every_s"]:
                    (self.out / "ckpt").mkdir(exist_ok=True)
                    save_weights(self.out / "ckpt" / f"step{self.step:08d}.pt.gz", self.model, self.data.vocab,
                                 self.data.edge_vocab, self._info())
                    last_ckpt = now
                if cfg["latest_every_s"] and now - last_latest >= cfg["latest_every_s"]:
                    self.save_latest()
                    last_latest = now
        finally:
            pf.close()
            if old is not None:
                signal.signal(signal.SIGTERM, old)
        self.save_latest()
        final = self.do_eval() if self.step != last_eval_step or not self.best_eval else None
        save_weights(self.out / "final.pt.gz", self.model, self.data.vocab, self.data.edge_vocab, self._info(final))
        summary = {"out": str(self.out), "stop": stop, "step": self.step, "seen": self.seen,
                   "epochs": round(self.seen / max(1, self.epoch_rows), 4), "train_time_s": round(self.train_s, 1),
                   "samples_per_s": round(self.seen / max(1e-9, self.train_s), 1), "td_refreshes": self.td_refreshes,
                   "sessions": self.sessions, "best": self.best, "best_policy_eval": self.best_eval.get("policy"),
                   "best_value_eval": self.best_eval.get("value"), "best_combined_eval": self.best_eval.get("combined"),
                   "params": gn.parameter_counts(self.model), "upstream_loadable": gn.upstream_loadable(self.model.arch),
                   "data": self.data.info, "config": self.cfg, "wall_s": round(time.monotonic() - t_start, 1)}
        (self.out / "summary.json").write_text(json.dumps(sv._jsonable(summary), indent=1))
        return summary


# ================================================================================================
# speed
# ================================================================================================

def bench(model, tables: list[GraphTable], dev, dtype, *, batches=(1, 8, 32, 128), seconds: float = 5.0,
          threads: int | None = None) -> dict:
    """States a second for forward passes at each batch size, on real rows (eval mode), and the time
    to build the batch on the CPU."""
    if threads:
        torch.set_num_threads(threads)
    t = max(tables, key=lambda x: x.n)
    model.eval()
    out = {"device": str(dev), "dtype": str(dtype), "threads": torch.get_num_threads(), "table": t.name}
    rng = np.random.default_rng(0)
    for B in batches:
        r = np.sort(rng.choice(t.n, min(B, t.n), replace=False))
        b = make_batch([t], [(0, r)])
        g = b["graphs"].to(dev)
        with torch.no_grad(), sv._autocast(dev, dtype):
            for _ in range(2):
                model(g)
            sv._sync(dev)
            n, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < seconds:
                model(g)
                n += 1
            sv._sync(dev)
            dt = (time.perf_counter() - t0) / n
        t1 = time.perf_counter()
        for _ in range(5):
            make_batch([t], [(0, r)])
        out[f"b{B}"] = {"states_per_s": round(len(r) / dt, 1), "ms_per_batch": round(dt * 1000, 2),
                        "batch_build_ms": round((time.perf_counter() - t1) / 5 * 1000, 2)}
    return out


def bench_train(trainer: Trainer, steps: int = 50) -> dict:
    pf = Prefetcher(trainer.sampler, 2, pin=trainer.dev.type == "cuda")
    try:
        for _ in range(3):
            trainer.train_step(pf.get())
        sv._sync(trainer.dev)
        t0, n = time.perf_counter(), 0
        for _ in range(steps):
            b = pf.get()
            trainer.train_step(b)
            n += len(b["g"])
        sv._sync(trainer.dev)
        dt = time.perf_counter() - t0
    finally:
        pf.close()
    return {"steps": steps, "states_per_s": round(n / dt, 1), "s_per_step": round(dt / steps, 4)}


# ================================================================================================
# sweeps and the CLI
# ================================================================================================

def run_sweep(spec_path: Path, out: Path, overrides: dict, only: list | None = None, log=print) -> list:
    """A sweep: `base` (and base_config), then each of `runs` ({name, ...overrides}) trained in turn on
    data loaded once; a run with summary.json is read back, one with latest.pt resumes."""
    import yaml
    spec = yaml.safe_load(Path(spec_path).read_text())
    base = sv.load_config_file(spec.get("base_config")) if spec.get("base_config") else {}
    base = {**base, **(spec.get("base") or {})}
    cfg0 = resolve_config(base, overrides)
    data = load_data(cfg0, log=log)
    rows = []
    for r in spec["runs"]:
        r = dict(r)
        name = r.pop("name")
        if only and name not in only:
            continue
        bad = [k for k in r if k in DATA_KEYS and r[k] != cfg0[k]]
        if bad:
            raise ValueError(f"run {name} changes data keys {bad}: put them in base")
        cfg = resolve_config(base, overrides, r)
        d = out / "runs" / name
        if (d / "summary.json").exists():
            s = json.loads((d / "summary.json").read_text())
        else:
            log(f"graph_supervised: sweep run {name}")
            s = Trainer(cfg, data, d, log=log, resume=True).run()
        bp = s.get("best_policy_eval") or {}
        bv = s.get("best_value_eval") or {}
        rows.append({"name": name, "overrides": r, "steps": s["step"], "epochs": s["epochs"],
                     "train_s": s["train_time_s"], "samples_per_s": s["samples_per_s"],
                     "policy_set_nll": bp.get("policy/set_nll"), "policy_top1_nonpass": bp.get("policy/top1_nonpass"),
                     "binary_acc": bp.get("binary/acc"), "opp_block_top1": bp.get("opp_block/top1"),
                     "replay_target_top1": bp.get("replay_target/top1"), "opp_pass_top1": bp.get("opp_priority/pass_top1"),
                     "value_auc": bv.get("value/auc"), "value_logloss": bv.get("value/logloss")})
        (out / "sweep.json").write_text(json.dumps(sv._jsonable({"spec": str(spec_path), "runs": rows}), indent=1))
    return rows


def _cfg_from_args(a) -> dict:
    layers = []
    if getattr(a, "resume", False) and (Path(a.out) / "config.json").exists():
        layers.append(json.loads((Path(a.out) / "config.json").read_text()))
    if getattr(a, "config", None):
        layers.append(sv.load_config_file(a.config))
    over = sv.parse_overrides(getattr(a, "set", None))
    if getattr(a, "tables_dir", None):
        over["tables_dir"] = a.tables_dir
    if getattr(a, "device", None):
        over["device"] = a.device
    layers.append(over)
    return resolve_config(*layers)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "sweep", "evaluate", "bench"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path)
        p.add_argument("--tables-dir")
        p.add_argument("--set", action="append", help="KEY=VALUE (YAML), e.g. arch.passes=1")
        p.add_argument("--device")
        if name == "train":
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--resume", action="store_true")
        if name == "sweep":
            p.add_argument("--spec", type=Path, required=True)
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--only", help="comma-separated run names")
        if name in ("evaluate", "bench"):
            p.add_argument("--checkpoint")
            p.add_argument("--split", default="val")
            p.add_argument("--json", type=Path)
        if name == "bench":
            p.add_argument("--seconds", type=float, default=5.0)
            p.add_argument("--train-steps", type=int, default=50)
            p.add_argument("--cpu-threads", default="1", help="comma-separated CPU thread counts for CPU inference")
    a = ap.parse_args(argv)
    if a.cmd == "train":
        cfg = _cfg_from_args(a)
        data = load_data(cfg)
        print(json.dumps(sv._jsonable(Trainer(cfg, data, a.out, resume=a.resume).run()), indent=1)[:4000])
    elif a.cmd == "sweep":
        over = sv.parse_overrides(a.set)
        if a.tables_dir:
            over["tables_dir"] = a.tables_dir
        rows = run_sweep(a.spec, a.out, over, a.only.split(",") if a.only else None)
        print(json.dumps(sv._jsonable(rows), indent=1))
    elif a.cmd == "evaluate":
        cfg = _cfg_from_args(a)
        dev = sv.pick_device(cfg["device"])
        model, vocab, edge_vocab, meta = load_checkpoint(a.checkpoint, dev)
        data = load_data(cfg, vocabs=(vocab, edge_vocab), splits=(a.split,))
        res = {"checkpoint": a.checkpoint, "split": a.split,
               **evaluate(model, data.val, cfg, dev, sv.amp_dtype(cfg["amp"], dev)), "data": data.info}
        if a.json:
            a.json.write_text(json.dumps(sv._jsonable(res), indent=1))
        print(json.dumps(sv._jsonable(res), indent=1))
    else:
        cfg = _cfg_from_args(a)
        dev = sv.pick_device(cfg["device"])
        dtype = sv.amp_dtype(cfg["amp"], dev)
        if a.checkpoint:
            model, vocab, edge_vocab, _ = load_checkpoint(a.checkpoint, dev)
            data = load_data(cfg, vocabs=(vocab, edge_vocab))
        else:
            data = load_data(cfg)
            model = gn.NetGraph(len(data.vocab), len(data.edge_vocab), cfg["arch"]).to(dev)
        res = {"params": gn.parameter_counts(model), "data": data.info,
               "infer": bench(model, data.val or data.train, dev, dtype, seconds=a.seconds)}
        cpu_model = copy.deepcopy(model).cpu()
        res["infer_cpu"] = [bench(cpu_model, data.val or data.train, torch.device("cpu"), None, seconds=a.seconds,
                                  batches=(1, 32), threads=int(n)) for n in str(a.cpu_threads).split(",")]
        if a.train_steps and data.train:
            import tempfile
            tr = Trainer({**cfg, "eval_at_start": False}, data, Path(tempfile.mkdtemp(prefix="gnn_bench_")),
                         log=lambda *x, **k: None)
            res["train"] = bench_train(tr, a.train_steps)
        if a.json:
            a.json.write_text(json.dumps(sv._jsonable(res), indent=1))
        print(json.dumps(sv._jsonable(res), indent=1))


if __name__ == "__main__":
    main()
