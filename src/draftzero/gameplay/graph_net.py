"""MageZero's graph network (Will Wroble, WillWroble/MageZero branch graph-encoder), for DraftZero's
imitation training and search (docs/022).

The network is MageZero's `NetGraph` (src/magezero/model.py at de225045, 2026-10-04), copied here
rather than imported: that branch's modules import each other as top-level names (`from vocab import
...`), and DraftZero pins MageZero v0.2 for the flat networks. The code below is the upstream model with
four changes, none of which changes what a default network computes:

  * its sizes are constructor arguments (`arch`), so a sweep can vary them; the defaults are
    upstream's constants, and a default network's state dict is upstream's, so it loads in
    MageZero's own graph server (`load_checkpoint` there);
  * `leaf_dropout`: drop a share of leaf edges while training, the graph analogue of the flat
    networks' token dropout (0 = upstream);
  * `local_depth`: LocalLayers per type within each pass (1 = upstream). Each extra layer re-attends
    from the node's updated embedding to the same children; extra layers live in `local_extra`, so a
    depth-1 network's state dict is upstream's;
  * `forward` also returns the value head's input to its Tanh (`value_x`), which the imitation
    trainer's cross-entropy needs, as supervised.value_logit does for the flat networks.

Input: the graph encoder's state graph (graph_tables), nodes typed ROOT, PLAYER, ZONE, STACK_OBJECT,
PERMANENT, CARD, ABILITY or LEAF, edges child -> parent with a label:

  embeddings: typed node = type embedding; leaf = leaf vocab row + numeric value bucket
  local layers: `passes` bottom-up passes, each `local_depth` LocalLayers per type, run in order
      ABILITY -> CARD -> PERMANENT -> STACK_OBJECT -> ZONE -> PLAYER -> ROOT
      every node attends over [itself, children + edge label embedding]
  global layers: TransformerEncoder over [CLS, the state's internal nodes]
      ├── MLP per node  priority   (read at ABILITY nodes: a play is its ability's node; Pass has one)
      ├── MLP per node  target     (read at CARD / PERMANENT / STACK_OBJECT / PLAYER nodes; Stop Choosing is a CARD)
      ├── MLP on CLS    choose_use (false, true)
      └── MLP on CLS    value      (tanh, -1..1)
"""
from __future__ import annotations

import math
from enum import IntEnum
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

UPSTREAM = "WillWroble/MageZero@de225045c82a2ddc107b99ac833e21ab6d566279 (graph-encoder)"
GLOBAL_MAX = 2 ** 31 - 1

# upstream's constants (model.py): the default architecture
ARCH_DEFAULT = {"type": "graph", "d_model": 512, "heads": 4, "ff": 1024, "dropout": 0.25, "passes": 2,
                "global_layers": 2, "head_hidden": 256, "leaf_dropout": 0.0, "local_depth": 1}

# numeric value buckets: exact -5..20 (P/T, counters, mana, costs, small counts), then 21-25, 26-30,
# 31-40, 41-60, 61+ (life totals, library and zone sizes); everything below -5 is one bucket
VALUE_BOUNDS = list(range(-5, 22)) + [26, 31, 41, 61]


class NodeType(IntEnum):
    """Java FeatureGraph.Node.Type, plus the root."""
    ROOT = 0
    PLAYER = 1
    ZONE = 2
    STACK_OBJECT = 3
    PERMANENT = 4
    CARD = 5
    ABILITY = 6
    LEAF = 7


STAGES = (NodeType.ABILITY, NodeType.CARD, NodeType.PERMANENT, NodeType.STACK_OBJECT,
          NodeType.ZONE, NodeType.PLAYER, NodeType.ROOT)
PRIORITY_TYPES = (NodeType.ABILITY,)
TARGET_TYPES = (NodeType.CARD, NodeType.PERMANENT, NodeType.STACK_OBJECT, NodeType.PLAYER)


# ------------------------------------------------------------------------------------------------
# ids: the Java encoder's hash (upstream vocab.py), to tell typed nodes from leaves
# ------------------------------------------------------------------------------------------------

_MASK = (1 << 64) - 1
_GOLDEN = 0x9E3779B185EBCA87


def _mix64(z: int) -> int:
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def feature_id(name: str) -> int:
    """The id XMage stores for a leaf name or edge label: StateEncoder.indexFor(hash64(name))."""
    data = name.encode("utf-8")
    h = _mix64(_GOLDEN ^ (len(data) * _GOLDEN & _MASK))
    full = len(data) // 8 * 8
    for i in range(0, full, 8):
        h ^= _mix64(int.from_bytes(data[i:i + 8], "little"))
        h = (((h << 27 | h >> 37) & _MASK) * _GOLDEN + 0x165667B19E3779F9) & _MASK
    h ^= _mix64(int.from_bytes(data[full:], "little"))
    h = ((h ^ (h >> 33)) * 0xFF51AFD7ED558CCD) & _MASK
    h = ((h ^ (h >> 33)) * 0xC4CEB9FE1A85EC53) & _MASK
    h ^= h >> 33
    if h == 1 << 63:  # Long.MIN_VALUE: Java's -h overflows and the remainder keeps the sign
        return -((1 << 63) % GLOBAL_MAX)
    return (h if h < 1 << 63 else (1 << 64) - h) % GLOBAL_MAX


# a typed node's id is the hash of its type name and the root's id is 0; every other node is a leaf
TYPE_IDS = {0: NodeType.ROOT}
for _t in NodeType:
    if _t not in (NodeType.ROOT, NodeType.LEAF):
        TYPE_IDS[feature_id(_t.name)] = _t


def node_types(ids: np.ndarray) -> np.ndarray:
    types = np.full(ids.shape, NodeType.LEAF, dtype=np.int8)
    for type_id, t in TYPE_IDS.items():
        types[ids == type_id] = t
    return types


def full_arch(arch: dict | None) -> dict:
    a = {**ARCH_DEFAULT, **(arch or {})}
    unknown = set(a) - set(ARCH_DEFAULT)
    if unknown:
        raise ValueError(f"unknown graph arch keys {sorted(unknown)} (known: {sorted(ARCH_DEFAULT)})")
    if a["type"] != "graph":
        raise ValueError(f"arch.type {a['type']!r}: graph_net builds only 'graph'")
    if a["d_model"] % a["heads"]:
        raise ValueError(f"d_model {a['d_model']} is not a multiple of heads {a['heads']}")
    return a


def upstream_loadable(arch: dict) -> bool:
    """A network MageZero's own graph server loads as it is (upstream's constants)."""
    a = full_arch(arch)
    return all(a[k] == ARCH_DEFAULT[k] for k in ("d_model", "heads", "ff", "passes", "global_layers", "head_hidden",
                                                 "local_depth"))


# ------------------------------------------------------------------------------------------------
# the network
# ------------------------------------------------------------------------------------------------

class Graphs(NamedTuple):
    """A batch of state graphs, concatenated: the network's input."""
    node_type: torch.Tensor     # [N] NodeType
    node_row: torch.Tensor      # [N] leaf vocab row; -1 for typed nodes and leaves outside the vocab
    node_value: torch.Tensor    # [N] numeric value (0 for non-numeric)
    node_offsets: torch.Tensor  # [B+1] first node of each state
    edge_child: torch.Tensor    # [E] node index (batch-wide)
    edge_parent: torch.Tensor   # [E] node index (batch-wide)
    edge_label: torch.Tensor    # [E] edge vocab row + 1; 0 for labels outside the vocab

    def to(self, device):
        return Graphs(*(t.to(device, non_blocking=True) for t in self))


class Out(NamedTuple):
    priority: torch.Tensor      # [N] per-node priority score (-inf at leaves)
    target: torch.Tensor        # [N] per-node target score (-inf at leaves)
    use: torch.Tensor           # [B, 2] choose_use logits (false, true)
    value: torch.Tensor         # [B] tanh(value_x)
    value_x: torch.Tensor       # [B] the value head's input to its Tanh


def graph_index(offsets: torch.Tensor) -> torch.Tensor:
    """State of each node, from a batch's node offsets."""
    return torch.repeat_interleave(torch.arange(offsets.numel() - 1, device=offsets.device), offsets.diff())


def segment_logsumexp(x: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    """Log-sum-exp over the rows of x that share an index in seg (values 0..n-1); returns n rows
    (-inf for an index with no row)."""
    index = seg.view(-1, *([1] * (x.dim() - 1))).expand_as(x)
    with torch.no_grad():  # the max only stabilizes; the result and its gradient don't depend on it
        m = x.new_full((n, *x.shape[1:]), float("-inf")).scatter_reduce(0, index, x, "amax")
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    return m + x.new_zeros((n, *x.shape[1:])).index_add(0, seg, (x - m[seg]).exp()).log()


class LocalLayer(nn.TransformerEncoderLayer):
    """A pre-norm transformer layer run over each node's [node, children...] sequence, keeping only
    the node's output token (upstream's LocalLayer)."""

    def __init__(self, d_model: int, heads: int, ff: int, dropout: float):
        super().__init__(d_model, heads, ff, dropout, activation="gelu", batch_first=True, norm_first=True)

    def forward(self, node: torch.Tensor, child: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        """node: [S, d] this stage's nodes; child: [E, d] child tokens (child embedding + edge label
        embedding); seg: [E] position in `node` of each child's parent. Returns the nodes' new embeddings."""
        S, d = node.shape
        H, dh = self.self_attn.num_heads, self.self_attn.head_dim
        w, b = self.self_attn.in_proj_weight, self.self_attn.in_proj_bias
        q, k, v = F.linear(self.norm1(node), w, b).chunk(3, dim=-1)
        k_child, v_child = F.linear(self.norm1(child), w[d:], b[d:]).chunk(2, dim=-1)
        seg = torch.cat([torch.arange(S, device=seg.device), seg])  # each node also attends to itself
        k = torch.cat([k, k_child]).view(-1, H, dh)
        v = torch.cat([v, v_child]).view(-1, H, dh)
        score = (q.reshape(S, H, dh)[seg] * k).sum(-1).float() / math.sqrt(dh)  # [S+E, H]
        attn = (score - segment_logsumexp(score, seg, S)[seg]).exp()
        attn = F.dropout(attn, self.self_attn.dropout, self.training)
        out = torch.zeros(S, H, dh, device=node.device).index_add_(
            0, seg, (attn.to(v.dtype).unsqueeze(-1) * v).float())
        node = node + self.dropout1(self.self_attn.out_proj(out.view(S, d).to(node.dtype)))
        return node + self._ff_block(self.norm2(node))


def _mlp(d: int, hidden: int, out_dim: int, *tail: nn.Module) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, out_dim), *tail)


class NetGraph(nn.Module):
    def __init__(self, num_embeddings: int = 0, num_edge_labels: int = 0, arch: dict | None = None):
        super().__init__()
        a = self.arch = full_arch(arch)
        d = a["d_model"]
        self.embedding = nn.Embedding(num_embeddings, d)                         # leaves
        self.edge_embedding = nn.Embedding(num_edge_labels + 1, d, padding_idx=0)  # row 0: unknown label
        self.type_embedding = nn.Embedding(len(NodeType), d)
        self.value_embedding = nn.Embedding(len(VALUE_BOUNDS) + 1, d)
        self.register_buffer("value_bounds", torch.tensor(VALUE_BOUNDS), persistent=False)
        self.local_layers = nn.ModuleList(
            nn.ModuleDict({t.name: LocalLayer(d, a["heads"], a["ff"], a["dropout"]) for t in STAGES})
            for _ in range(a["passes"]))
        if a["local_depth"] < 1:
            raise ValueError(f"local_depth {a['local_depth']}: at least 1")
        if a["local_depth"] > 1:  # the layers after each type's first, per pass (absent at depth 1: upstream's layout)
            self.local_extra = nn.ModuleList(
                nn.ModuleDict({t.name: nn.ModuleList(LocalLayer(d, a["heads"], a["ff"], a["dropout"])
                                                     for _ in range(a["local_depth"] - 1)) for t in STAGES})
                for _ in range(a["passes"]))
        self.cls = nn.Parameter(torch.randn(d))
        global_layer = nn.TransformerEncoderLayer(d, a["heads"], a["ff"], a["dropout"], activation="gelu",
                                                  batch_first=True, norm_first=True)
        self.global_layers = nn.TransformerEncoder(global_layer, a["global_layers"], norm=nn.LayerNorm(d),
                                                   enable_nested_tensor=False)
        h = a["head_hidden"]
        self.priority_head = _mlp(d, h, 1)
        self.target_head = _mlp(d, h, 1)
        self.binary_head = _mlp(d, h, 2)
        self.value_head = _mlp(d, h, 1, nn.Tanh())

    def forward(self, g: Graphs) -> Out:
        B, N = g.node_offsets.numel() - 1, g.node_type.numel()

        # initial embeddings: typed nodes their type, leaves their name and numeric value
        leaves = (g.node_row >= 0).nonzero().squeeze(1)
        value = torch.bucketize(g.node_value[leaves], self.value_bounds, right=True)
        h = self.type_embedding(g.node_type).index_add(
            0, leaves, self.embedding(g.node_row[leaves]) + self.value_embedding(value))

        # leaves outside the vocab have no embedding, so their edges are dropped (and, while
        # training, a leaf_dropout share of the others)
        child_leaf = g.node_type[g.edge_child] == NodeType.LEAF
        keep = ~child_leaf | (g.node_row[g.edge_child] >= 0)
        if self.training and self.arch["leaf_dropout"] > 0:
            keep = keep & ~(child_leaf & (torch.rand(keep.shape, device=keep.device) < self.arch["leaf_dropout"]))
        child, parent, label = g.edge_child[keep], g.edge_parent[keep], g.edge_label[keep]

        # one stage per type: every node of that type, its children, and each child's parent position
        parent_type = g.node_type[parent]
        stages = []
        for t in STAGES:
            nodes = (g.node_type == t).nonzero().squeeze(1)
            if len(nodes):
                edges = (parent_type == t).nonzero().squeeze(1)
                stages.append((t.name, nodes, child[edges], label[edges], torch.searchsorted(nodes, parent[edges])))

        # local layers, bottom-up passes: children earlier in the order arrive updated in this pass,
        # children of the same type or later in the order (attachments, targets, linked exile) from the last one
        extra = getattr(self, "local_extra", None)
        for p, layers in enumerate(self.local_layers):
            for name, nodes, c, lbl, seg in stages:
                kids = h[c] + self.edge_embedding(lbl)
                x = layers[name](h[nodes], kids, seg).to(h.dtype)
                for layer in (extra[p][name] if extra is not None else ()):
                    x = layer(x, kids, seg).to(h.dtype)
                h = h.index_copy(0, nodes, x)

        # global layers: self-attention over [CLS, internal nodes], one padded sequence per state
        internal = (g.node_type != NodeType.LEAF).nonzero().squeeze(1)
        gi = graph_index(g.node_offsets)[internal]
        counts = torch.bincount(gi, minlength=B)
        pos = torch.arange(len(internal), device=h.device) - (counts.cumsum(0) - counts)[gi] + 1
        x = h.new_zeros(B, int(counts.max()) + 1, h.shape[1])
        x[:, 0] = self.cls.to(h.dtype)
        x[gi, pos] = h[internal]
        pad = torch.arange(x.shape[1], device=h.device) > counts.unsqueeze(1)
        x = self.global_layers(x, src_key_padding_mask=pad)
        cls, z = x[:, 0], x[gi, pos]

        def per_node(head):
            return torch.full((N,), float("-inf"), device=h.device).index_copy(
                0, internal, head(z).squeeze(-1).float())

        vx = self.value_head[:-1](cls).squeeze(-1).float()
        return Out(per_node(self.priority_head), per_node(self.target_head), self.binary_head(cls).float(),
                   torch.tanh(vx), vx)


def parameter_counts(model: NetGraph) -> dict:
    out = {"total": 0}
    for name, p in model.named_parameters():
        part = name.split(".")[0]
        out[part] = out.get(part, 0) + p.numel()
        out["total"] += p.numel()
    return out


# ------------------------------------------------------------------------------------------------
# batches: graph rows -> the network's input, and options -> logits
# ------------------------------------------------------------------------------------------------

def gather(ptr: np.ndarray, sel: np.ndarray, *arrays: np.ndarray) -> tuple:
    """The CSR slices of rows `sel` (ptr over `arrays`), concatenated, and their new ptr."""
    lens = (ptr[sel + 1] - ptr[sel]).astype(np.int64)
    new = np.zeros(len(sel) + 1, np.int64)
    np.cumsum(lens, out=new[1:])
    pos = np.repeat(ptr[sel] - new[:-1], lens) + np.arange(new[-1])
    return (new,) + tuple(a[pos] for a in arrays)


class OptionBatch(NamedTuple):
    """Per option of a batch: its row, its nodes (batch-wide) and which node belongs to which option."""
    opt_row: torch.Tensor       # [O] row of each option
    node: torch.Tensor          # [K] batch-wide node index of each option node
    node_opt: torch.Tensor      # [K] option of each option node
    n_opt: int

    def to(self, device):
        return OptionBatch(self.opt_row.to(device), self.node.to(device), self.node_opt.to(device), self.n_opt)


def option_logits(out: Out, ob: OptionBatch, graph_type: torch.Tensor, opt_index: torch.Tensor) -> torch.Tensor:
    """[O] logit of each option: log-sum-exp of its nodes' scores under the head its row's decision type
    reads (priority at PRIORITY rows, target at CHOOSE_TARGET rows; two copies of a card are one option,
    so their probabilities add), or the choose_use logit at CHOOSE_USE rows (options [no, yes]).
    opt_index: [O] each option's position within its row."""
    gt = graph_type[ob.opt_row[ob.node_opt]]
    s = torch.where(gt == 0, out.priority[ob.node], out.target[ob.node])
    lg = segment_logsumexp(s.float(), ob.node_opt, ob.n_opt)
    is_use = graph_type[ob.opt_row] == 5
    if is_use.any():
        u = out.use.float()[ob.opt_row, opt_index.clamp(max=1)]
        lg = torch.where(is_use, u, lg)
    return lg
