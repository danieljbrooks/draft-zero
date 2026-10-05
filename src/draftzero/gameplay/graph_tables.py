"""Graph encodings of imitation decisions, for MageZero's graph network (docs/022).

The bridge's `graph` option (java/mzbridge GraphRecord) returns each decision state as MageZero's
graph encoder sees it (the vendored copy of WillWroble/mage graph-encoder in
java/mzbridge/src/org/draftzero/mzbridge/graph): nodes (a feature id and a numeric value each),
edges child -> parent with a hashed label, and the legal options as graph nodes. Option j of a
decision is the list of nodes that stand for the j-th entry of the decision's `legal` list: one
ability node per copy of a card for a play, the target's node for a target, Stop Choosing's node,
the defending player's node for "attack" (attacks are target choices on the graph-encoder branch).

`build.py tables --graph` writes a graph file beside each flat table, row for row:

    <table>_<split>.graph.h5     (same rows, in the same order, as <table>_<split>.h5)
      node_ids      int32  [nodes]     node feature id (typed nodes: the hash of their type name)
      node_values   int16  [nodes]     numeric value (0 for non-numeric nodes)
      node_ptr      int64  [N+1]
      edge_child    uint16 [edges]     node index local to the row's graph
      edge_parent   uint16 [edges]
      edge_label    int32  [edges]     hashed edge label
      edge_ptr      int64  [N+1]
      opt_node      int32  [option nodes]  node index local to the row's graph
      opt_ptr       int64  [options+1]     option -> its nodes
      row_opt_ptr   int64  [N+1]           row -> its options (the flat row's legal labels, in order)
      opt_in_set    uint8  [options]       1 if the option is in the human's label set
      opt_is_pass   uint8  [options]       1 for Pass
      graph_type    int8   [N]             MageZero ActionType ordinal: 0 PRIORITY, 3 CHOOSE_TARGET, 5 CHOOSE_USE
      missing       int16  [N]             options with no node (the row's policy can't score them)
      meta/row, meta/turn  int64 [N]       copied from the flat table, to check the alignment on load

Per-row labels, weights, value targets and metadata stay in the flat table.
"""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

GRAPH_SUFFIX = ".graph.h5"
ACTION_TYPES = {"PRIORITY": 0, "CHOOSE_NUM": 1, "BLANK": 2, "CHOOSE_TARGET": 3, "MAKE_CHOICE": 4, "CHOOSE_USE": 5}
# the pointer arrays of a graph file: build.merge_h5 shifts them when it concatenates parts
GRAPH_PTRS = ("node_ptr", "edge_ptr", "row_opt_ptr", "opt_ptr")
ENCODER_PIN = "WillWroble/mage@e4afc9c77ba7e4a6dbc24966cbdc6dc0819e0bff"   # java/mzbridge/graph_sync.sh


def _ints(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), "<i4")


def decode(g: dict | None) -> dict | None:
    """A bridge `graph` record -> compact numpy arrays (what the build shards keep)."""
    if not g:
        return None
    opts = g.get("options") or []
    child, parent = _ints(g["edge_child"]), _ints(g["edge_parent"])
    ids = _ints(g["ids"]).astype(np.int32)
    if len(ids) > 65535:
        raise ValueError(f"a graph of {len(ids)} nodes: edge indices no longer fit in uint16")
    return {"type": ACTION_TYPES[g["type"]],
            "ids": ids,
            "values": np.clip(_ints(g["values"]), -32768, 32767).astype(np.int16),
            "child": child.astype(np.uint16), "parent": parent.astype(np.uint16),
            "label": _ints(g["edge_label"]).astype(np.int32),
            "opt_node": np.asarray([k for o in opts for k in o], np.int32),
            "opt_len": np.asarray([len(o) for o in opts], np.int32),
            "missing": int(g.get("missing") or 0)}


def option_labels(legal: list[str], chosen_set) -> tuple[np.ndarray, np.ndarray]:
    """(in_set, is_pass) per legal label."""
    s = set(chosen_set or [])
    return (np.asarray([x in s for x in legal], np.uint8), np.asarray([x == "Pass" for x in legal], np.uint8))


def write(path: Path, graphs: list[dict], in_set: list[np.ndarray], is_pass: list[np.ndarray],
          meta_row, meta_turn) -> None:
    """One graph file (the module docstring's layout) for rows whose decoded graphs are `graphs`, with
    each row's per-option label flags. Every graph must have one option group per label flag."""
    import h5py
    n = len(graphs)
    for i, (g, a, b) in enumerate(zip(graphs, in_set, is_pass)):
        if g is None:
            raise ValueError(f"row {i}: no graph (build with --graph)")
        if len(g["opt_len"]) != len(a) or len(a) != len(b):
            raise ValueError(f"row {i}: {len(g['opt_len'])} option groups for {len(a)} legal labels")

    def ptr(lens) -> np.ndarray:
        p = np.zeros(len(lens) + 1, np.int64)
        p[1:] = np.cumsum(lens)
        return p

    def cat(key, dtype) -> np.ndarray:
        parts = [g[key] for g in graphs]
        return np.concatenate(parts).astype(dtype, copy=False) if parts else np.zeros(0, dtype)
    opt_len = cat("opt_len", np.int64)
    gz = dict(compression="gzip", compression_opts=1)
    with h5py.File(path, "w") as f:
        f.create_dataset("node_ids", data=cat("ids", np.int32), **gz)
        f.create_dataset("node_values", data=cat("values", np.int16), **gz)
        f.create_dataset("node_ptr", data=ptr([len(g["ids"]) for g in graphs]))
        f.create_dataset("edge_child", data=cat("child", np.uint16), **gz)
        f.create_dataset("edge_parent", data=cat("parent", np.uint16), **gz)
        f.create_dataset("edge_label", data=cat("label", np.int32), **gz)
        f.create_dataset("edge_ptr", data=ptr([len(g["child"]) for g in graphs]))
        f.create_dataset("opt_node", data=cat("opt_node", np.int32), **gz)
        f.create_dataset("opt_ptr", data=ptr(opt_len))
        f.create_dataset("row_opt_ptr", data=ptr([len(g["opt_len"]) for g in graphs]))
        f.create_dataset("opt_in_set", data=np.concatenate(in_set).astype(np.uint8) if n else np.zeros(0, np.uint8))
        f.create_dataset("opt_is_pass", data=np.concatenate(is_pass).astype(np.uint8) if n else np.zeros(0, np.uint8))
        f.create_dataset("graph_type", data=np.asarray([g["type"] for g in graphs], np.int8))
        f.create_dataset("missing", data=np.asarray([g["missing"] for g in graphs], np.int16))
        f.create_dataset("meta/row", data=np.asarray(meta_row, np.int64))
        f.create_dataset("meta/turn", data=np.asarray(meta_turn, np.int64))
        f.attrs["layout"] = "draftzero.gameplay.graph_tables (MageZero graph encoder, docs/022)"
        f.attrs["encoder"] = ENCODER_PIN


def graph_path(flat_path: Path) -> Path:
    """<table>_<split>.h5 -> <table>_<split>.graph.h5"""
    return flat_path.with_name(flat_path.name[:-3] + GRAPH_SUFFIX)


def sizes(path: Path) -> dict:
    """Rows, nodes, edges and options of a graph file, with per-row means and bytes on disk."""
    import h5py
    with h5py.File(path, "r") as f:
        n = f["node_ptr"].shape[0] - 1
        nodes, edges = int(f["node_ptr"][-1]), int(f["edge_ptr"][-1])
        options = int(f["row_opt_ptr"][-1])
        miss = f["missing"][:]
    return {"rows": n, "nodes": nodes, "edges": edges, "options": options,
            "nodes_per_row": nodes / max(n, 1), "edges_per_row": edges / max(n, 1),
            "rows_missing_an_option": int((miss > 0).sum()), "bytes": path.stat().st_size,
            "bytes_per_row": path.stat().st_size / max(n, 1)}
