"""Three networks over gorge's entity encoding, one interface: forward(batch) -> (opt_scores [no] f32,
value_logit [B] f32).

  mlp          DeepSets: per-card vectors, per-group sum and max pools, residual MLP blocks
               (DraftZero's BagMLPNet analogue; gorge's own network is this shape, one layer deep)
  transformer  an entity transformer over [CLS, DENSE, SPARSE, cards]
  gnn          MageZero's hierarchical graph network (graph_net.NetGraph analogue): a bottom-up pass
               of local attention over root -> players -> zones -> cards -> leaves, then global
               transformer layers over the internal nodes

Options never attend to each other (SPEC §1): an option's score depends only on its state and itself.
One nn.Embedding(16384, d_emb) table is shared by the state's sparse bag, the cards' identity rows and
the options' hashed rows, as in gorge's network; option slots have their own 128-row table.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from . import packfmt as pf

DEFAULTS = {
    "mlp": {"d": 384, "ff": 1024, "layers": 3, "d_emb": 128, "card_d": 256, "head_hidden": 384,
            "policy_layers": 2, "value_layers": 1, "dropout": 0.0, "emb_std": 0.05},
    "transformer": {"d": 192, "heads": 6, "layers": 4, "ff": 768, "d_emb": 128, "head_hidden": 192,
                    "policy_layers": 2, "value_layers": 1, "dropout": 0.0, "emb_std": 0.05},
    "gnn": {"d": 128, "heads": 4, "global_layers": 2, "ff": 512, "passes": 1, "d_emb": 128, "head_hidden": 128,
            "policy_layers": 2, "value_layers": 1, "dropout": 0.0, "emb_std": 0.05},
}
ARCHS = tuple(DEFAULTS)


def full_config(arch: str, config: dict | None = None) -> dict:
    if arch not in DEFAULTS:
        raise ValueError(f"unknown arch {arch!r} (known: {list(DEFAULTS)})")
    c = {**DEFAULTS[arch], **(config or {})}
    unknown = set(c) - set(DEFAULTS[arch])
    if unknown:
        raise ValueError(f"unknown {arch} config keys {sorted(unknown)} (known: {sorted(DEFAULTS[arch])})")
    if "heads" in c and c["d"] % c["heads"]:
        raise ValueError(f"d {c['d']} is not a multiple of heads {c['heads']}")
    return c


# ------------------------------------------------------------------------------------------------
# shared pieces
# ------------------------------------------------------------------------------------------------

def bag(weight: torch.Tensor, rows: torch.Tensor, vals: torch.Tensor, off: torch.Tensor) -> torch.Tensor:
    """sum_j weight[rows[j]] * vals[j] per bag; off has n+1 entries; an empty bag is 0."""
    return F.embedding_bag(rows, weight, off, mode="sum", per_sample_weights=vals.to(weight.dtype),
                           include_last_offset=True)


def segment_logsumexp(x: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    """Log-sum-exp over the rows of x sharing an index in seg (0..n-1); -inf for an empty segment."""
    index = seg.view(-1, *([1] * (x.dim() - 1))).expand_as(x)
    with torch.no_grad():
        m = x.new_full((n, *x.shape[1:]), float("-inf")).scatter_reduce(0, index, x, "amax")
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    return m + x.new_zeros((n, *x.shape[1:])).index_add(0, seg, (x - m[seg]).exp()).log()


def _mlp(d_in: int, hidden: int, layers: int, out: int = 1, zero_last: bool = True) -> nn.Sequential:
    mods, d = [], d_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    last = nn.Linear(d, out)
    if zero_last:   # a uniform policy and a 50% value at init
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    return nn.Sequential(*mods, last)


def _ref(h: torch.Tensor, idx: torch.Tensor, none: torch.Tensor) -> torch.Tensor:
    """Rows h[idx] (idx = batch-wide card index), the learned `none` vector where idx < 0."""
    if h.shape[0] == 0:
        return none.to(h.dtype if h.is_floating_point() else none.dtype).expand(idx.shape[0], -1)
    g = h[idx.clamp(min=0)]
    return torch.where((idx >= 0).unsqueeze(1), g, none.to(g.dtype))


class _MLPBlock(nn.Module):
    """Pre-norm residual block: x + fc2(relu(fc1(norm(x)))) (supervised._MLPBlock)."""

    def __init__(self, d: int, ff: int, p: float):
        super().__init__()
        self.norm, self.fc1, self.fc2, self.drop = nn.LayerNorm(d), nn.Linear(d, ff), nn.Linear(ff, d), nn.Dropout(p)

    def forward(self, x):
        return x + self.drop(self.fc2(F.relu(self.fc1(self.norm(x)))))


class _SABlock(nn.Module):
    """Pre-norm transformer encoder layer with a key padding mask (keep: True = a real token)."""

    def __init__(self, d: int, heads: int, ff: int, p: float):
        super().__init__()
        self.h, self.p = heads, p
        self.n1, self.qkv, self.out = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.n2, self.ff1, self.ff2 = nn.LayerNorm(d), nn.Linear(d, ff), nn.Linear(ff, d)
        self.drop = nn.Dropout(p)

    def forward(self, x, keep):
        B, T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :],
                                           dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(B, T, d)))
        return x + self.drop(self.ff2(F.gelu(self.ff1(self.n2(x)))))


class _CrossBlock(nn.Module):
    """Option queries [B, O, d] attend over the state's tokens [B, T, d]; queries never see each other."""

    def __init__(self, d: int, heads: int, ff: int, p: float):
        super().__init__()
        self.h, self.p = heads, p
        self.nq, self.nkv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.kv, self.out = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.n2, self.ff1, self.ff2 = nn.LayerNorm(d), nn.Linear(d, ff), nn.Linear(ff, d)
        self.drop = nn.Dropout(p)

    def forward(self, q_in, x, keep):
        B, O, d = q_in.shape
        T = x.shape[1]
        q = self.q(self.nq(q_in)).view(B, O, self.h, d // self.h).transpose(1, 2)
        k, v = self.kv(self.nkv(x)).view(B, T, 2, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :],
                                           dropout_p=self.p if self.training else 0.0)
        h = q_in + self.drop(self.out(a.transpose(1, 2).reshape(B, O, d)))
        return h + self.drop(self.ff2(F.gelu(self.ff1(self.n2(h)))))


class _OptionInput(nn.Module):
    """o = bag(slots) + proj(bag(hashed rows via the shared table)) + Linear(opt_dense) + bot * bot_vec."""

    def __init__(self, d: int, d_emb: int, emb_std: float):
        super().__init__()
        self.slots = nn.Embedding(pf.SLOT_ROWS, d)
        nn.init.normal_(self.slots.weight, std=emb_std)
        self.hproj = nn.Linear(d_emb, d, bias=False)
        self.dense = nn.Linear(pf.OPT_DENSE_W, d)
        self.bot = nn.Parameter(torch.randn(d) * 0.02)

    def forward(self, b, table: torch.Tensor) -> torch.Tensor:
        return (bag(self.slots.weight, b.os_row, b.os_val, b.os_off) + self.hproj(bag(table, b.oh_row, b.oh_val, b.oh_off))
                + self.dense(b.opt_dense) + b.opt_bot.unsqueeze(1) * self.bot)


class _Net(nn.Module):
    arch: str = ""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.table = nn.Embedding(pf.TABLE_ROWS, config["d_emb"])
        nn.init.normal_(self.table.weight, std=config["emb_std"])

    def parameter_counts(self) -> dict:
        out = {"total": 0}
        for name, p in self.named_parameters():
            part = name.split(".")[0]
            out[part] = out.get(part, 0) + p.numel()
            out["total"] += p.numel()
        return out

    def _option_scores(self, b, h_state, x, keep, card_h):
        """Shared by transformer and gnn: queries from o and the ea/eb tokens cross-attend over the
        state's final tokens x [B, T, d]; score = MLP([h_state ‖ q' ‖ h_ea ‖ h_eb])."""
        d = x.shape[-1]
        if b.no == 0:
            return x.new_zeros(0, dtype=torch.float32)
        ha = _ref(card_h, b.opt_ea, self.none_a)
        hb = _ref(card_h, b.opt_eb, self.none_b)
        q = (self.opt_in(b, self.table.weight) + self.ea_proj(ha) + self.eb_proj(hb)).float()
        qp = q.new_zeros(b.B, max(b.Omax, 1), d)
        qp[b.opt_state, b.opt_pos] = q
        qp = self.cross(qp, x, keep)
        q2 = qp[b.opt_state, b.opt_pos]
        z = torch.cat([h_state[b.opt_state].float(), q2.float(), ha.float(), hb.float()], 1)
        return self.policy(z).squeeze(1).float()


# ------------------------------------------------------------------------------------------------
# mlp
# ------------------------------------------------------------------------------------------------

class MLPNet(_Net):
    """DeepSets: card c = relu(W2 relu(Linear(raw) + proj(bag(identity)))); per group the 0.25-scaled sum
    and the max of c; s = Linear(dense) + proj(bag(sparse)) + Linear(pools) -> residual MLP blocks;
    option o = option input + Linear(c_ea) + Linear(c_eb); score = MLP([s ‖ o]); value = MLP(s)."""
    arch = "mlp"

    def __init__(self, config: dict):
        super().__init__(config)
        c = config
        d, e, cd = c["d"], c["d_emb"], c["card_d"]
        self.card_raw = nn.Linear(pf.RAW_W, cd)
        self.card_rows = nn.Linear(e, cd, bias=False)
        self.card_phi = nn.Linear(cd, cd)
        self.dense_in = nn.Linear(pf.DENSE_W, d)
        self.sparse_proj = nn.Linear(e, d, bias=False)
        self.pool_proj = nn.Linear(pf.GROUPS * 2 * cd, d)
        self.blocks = nn.ModuleList(_MLPBlock(d, c["ff"], c["dropout"]) for _ in range(c["layers"]))
        self.norm = nn.LayerNorm(d)
        self.opt_in = _OptionInput(d, e, c["emb_std"])
        self.ea_proj, self.eb_proj = nn.Linear(cd, d), nn.Linear(cd, d)
        self.none_a = nn.Parameter(torch.randn(cd) * 0.02)
        self.none_b = nn.Parameter(torch.randn(cd) * 0.02)
        self.opt_norm = nn.LayerNorm(d)
        self.policy = _mlp(2 * d, c["head_hidden"], c["policy_layers"])
        self.value = _mlp(d, c["head_hidden"], c["value_layers"])

    def forward(self, b):
        B, w = b.B, self.table.weight
        c = F.relu(self.card_raw(b.card_raw) + self.card_rows(bag(w, b.cr_row, b.cr_val, b.cr_off)))
        c = F.relu(self.card_phi(c))                                     # [nc, cd] >= 0
        cd = c.shape[1]
        seg = b.card_state * pf.GROUPS + b.card_group
        sum_pool = c.new_zeros(B * pf.GROUPS, cd).index_add(0, seg, c) * 0.25
        max_pool = c.new_zeros(B * pf.GROUPS, cd).scatter_reduce(0, seg.unsqueeze(1).expand_as(c), c, "amax",
                                                                 include_self=False)
        pools = torch.cat([sum_pool.view(B, pf.GROUPS * cd), max_pool.view(B, pf.GROUPS * cd)], 1)
        s = self.dense_in(b.dense) + self.sparse_proj(bag(w, b.sp_row, b.sp_val, b.sp_off)) + self.pool_proj(pools)
        for blk in self.blocks:
            s = blk(s)
        s = self.norm(s)
        value = self.value(s).squeeze(1).float()
        if b.no == 0:
            return s.new_zeros(0, dtype=torch.float32), value
        o = self.opt_in(b, w) + self.ea_proj(_ref(c, b.opt_ea, self.none_a)) + self.eb_proj(_ref(c, b.opt_eb, self.none_b))
        o = self.opt_norm(o)
        scores = self.policy(torch.cat([s[b.opt_state].to(o.dtype), o], 1)).squeeze(1).float()
        return scores, value


# ------------------------------------------------------------------------------------------------
# transformer
# ------------------------------------------------------------------------------------------------

class TransformerNet(_Net):
    """Tokens [CLS, DENSE, SPARSE, cards (Linear(raw) + proj(bag(identity)) + group embedding)], pre-norm
    encoder layers with a padding mask; value = MLP(h_CLS); options cross-attend over the final tokens."""
    arch = "transformer"
    N_SPECIAL = 3

    def __init__(self, config: dict):
        super().__init__(config)
        c = config
        d, e = c["d"], c["d_emb"]
        self.cls = nn.Parameter(torch.randn(d) * 0.02)
        self.dense_in = nn.Linear(pf.DENSE_W, d)
        self.sparse_proj = nn.Linear(e, d, bias=False)
        self.card_raw = nn.Linear(pf.RAW_W, d)
        self.card_rows = nn.Linear(e, d, bias=False)
        self.group_emb = nn.Embedding(pf.GROUPS, d)
        nn.init.normal_(self.group_emb.weight, std=0.02)
        self.blocks = nn.ModuleList(_SABlock(d, c["heads"], c["ff"], c["dropout"]) for _ in range(c["layers"]))
        self.norm = nn.LayerNorm(d)
        self.opt_in = _OptionInput(d, e, c["emb_std"])
        self.ea_proj, self.eb_proj = nn.Linear(d, d), nn.Linear(d, d)
        self.none_a = nn.Parameter(torch.randn(d) * 0.02)
        self.none_b = nn.Parameter(torch.randn(d) * 0.02)
        self.cross = _CrossBlock(d, c["heads"], c["ff"], c["dropout"])
        self.policy = _mlp(4 * d, c["head_hidden"], c["policy_layers"])
        self.value = _mlp(d, c["head_hidden"], c["value_layers"])

    def forward(self, b):
        B, w = b.B, self.table.weight
        d, S = self.cls.shape[0], self.N_SPECIAL
        special = torch.stack([self.cls.expand(B, d).float(), self.dense_in(b.dense).float(),
                               self.sparse_proj(bag(w, b.sp_row, b.sp_val, b.sp_off)).float()], 1)
        cards = (self.card_raw(b.card_raw) + self.card_rows(bag(w, b.cr_row, b.cr_val, b.cr_off))
                 + self.group_emb(b.card_group)).float()
        cp = cards.new_zeros(B, b.Cmax, d)
        cp[b.card_state, b.card_pos] = cards
        x = torch.cat([special, cp], 1)
        keep = torch.cat([b.card_mask.new_ones(B, S), b.card_mask], 1)
        for blk in self.blocks:
            x = blk(x, keep)
        x = self.norm(x)
        h_cls = x[:, 0]
        value = self.value(h_cls).squeeze(1).float()
        card_h = x[b.card_state, S + b.card_pos]
        return self._option_scores(b, h_cls, x, keep, card_h), value


# ------------------------------------------------------------------------------------------------
# gnn
# ------------------------------------------------------------------------------------------------

# node types
ROOT, ME, OPP, Z_MYBF, Z_OPPBF, Z_HAND, Z_STACK, CARD = range(8)
# edge types
E_IDENT, E_RAW, E_DENSE, E_SPARSE, E_CARD_ZONE, E_ZONE_PLAYER, E_STACK_ROOT, E_PLAYER_ROOT = range(8)
N_FIXED = 7   # per state: root, me, opponent, 4 zones (group g -> zone node 3 + g)


class LocalLayer(nn.Module):
    """A pre-norm transformer layer over each node's [node, children...], keeping the node's output
    (graph_net.LocalLayer): segment softmax over children grouped by parent, index_add of values."""

    def __init__(self, d: int, heads: int, ff: int, p: float):
        super().__init__()
        self.h, self.p = heads, p
        self.n1, self.qkv, self.out = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.n2, self.ff1, self.ff2 = nn.LayerNorm(d), nn.Linear(d, ff), nn.Linear(ff, d)
        self.drop = nn.Dropout(p)

    def forward(self, node: torch.Tensor, child: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        """node [S, d] (f32); child [E, d] child tokens; seg [E] parent position in `node`."""
        S, d = node.shape
        H, dh = self.h, d // self.h
        q, k, v = self.qkv(self.n1(node)).chunk(3, -1)
        kc, vc = F.linear(self.n1(child), self.qkv.weight[d:], self.qkv.bias[d:]).chunk(2, -1)
        seg_all = torch.cat([torch.arange(S, device=node.device), seg])
        k = torch.cat([k, kc.to(k.dtype)]).view(-1, H, dh)
        v = torch.cat([v, vc.to(v.dtype)]).view(-1, H, dh)
        score = (q.float().reshape(S, H, dh)[seg_all] * k.float()).sum(-1) / math.sqrt(dh)   # [S+E, H], f32
        attn = (score - segment_logsumexp(score, seg_all, S)[seg_all]).exp()
        attn = F.dropout(attn, self.p, self.training)
        out = torch.zeros(S, H, dh, device=node.device).index_add_(0, seg_all, attn.unsqueeze(-1) * v.float())
        node = node + self.drop(self.out(out.view(S, d))).float()
        return node + self.drop(self.ff2(F.gelu(self.ff1(self.n2(node))))).float()


class GraphNet(_Net):
    """root -> players (me, opponent) -> zones (my battlefield and hand under me, the opponent's
    battlefield under the opponent, the stack under the root) -> cards -> leaves (one per identity row,
    table[row] * value, plus a Linear(raw) leaf); the root's leaves are DENSE and each sparse row.
    A bottom-up pass of local layers (card, zone, player, root), then global pre-norm transformer
    layers over [root, players, zones, cards]; value = MLP(root); options cross-attend over the
    global layer's nodes."""
    arch = "gnn"

    def __init__(self, config: dict):
        super().__init__(config)
        c = config
        d, e, hd, ff, p = c["d"], c["d_emb"], c["heads"], c["ff"], c["dropout"]
        self.node_type = nn.Embedding(8, d)
        self.edge_type = nn.Embedding(8, d)
        nn.init.normal_(self.node_type.weight, std=0.02)
        nn.init.normal_(self.edge_type.weight, std=0.02)
        self.leaf_proj = nn.Linear(e, d, bias=False)
        self.raw_leaf = nn.Linear(pf.RAW_W, d)
        self.dense_leaf = nn.Linear(pf.DENSE_W, d)
        self.local = nn.ModuleList(nn.ModuleDict({t: LocalLayer(d, hd, ff, p) for t in ("card", "zone", "player", "root")})
                                   for _ in range(c["passes"]))
        self.blocks = nn.ModuleList(_SABlock(d, hd, ff, p) for _ in range(c["global_layers"]))
        self.norm = nn.LayerNorm(d)
        self.opt_in = _OptionInput(d, e, c["emb_std"])
        self.ea_proj, self.eb_proj = nn.Linear(d, d), nn.Linear(d, d)
        self.none_a = nn.Parameter(torch.randn(d) * 0.02)
        self.none_b = nn.Parameter(torch.randn(d) * 0.02)
        self.cross = _CrossBlock(d, hd, ff, p)
        self.policy = _mlp(4 * d, c["head_hidden"], c["policy_layers"])
        self.value = _mlp(d, c["head_hidden"], c["value_layers"])
        # zones under the players: my battlefield and hand under me (0), the opponent's battlefield under them (1)
        # (buffers, so a forward pass makes no host-to-device copy)
        self.register_buffer("zone_ids", torch.tensor([0, 2, 1]), persistent=False)
        self.register_buffer("zone_player", torch.tensor([0, 0, 1]), persistent=False)

    def forward(self, b):
        B, nc, w, dev = b.B, b.nc, self.table.weight, b.dense.device
        et = self.edge_type.weight
        nt = self.node_type.weight
        d = nt.shape[1]
        # leaves (fixed): card identity rows and raw, the root's DENSE and sparse rows
        ident = (self.leaf_proj(w[b.cr_row] * b.cr_val.unsqueeze(1).to(w.dtype)) + et[E_IDENT]).float()
        raw = (self.raw_leaf(b.card_raw) + et[E_RAW]).float()
        dense = (self.dense_leaf(b.dense) + et[E_DENSE]).float()
        sparse = (self.leaf_proj(w[b.sp_row] * b.sp_val.unsqueeze(1).to(w.dtype)) + et[E_SPARSE]).float()
        card_seg = torch.cat([b.cr_card, torch.arange(nc, device=dev)])
        card_leaves = torch.cat([ident, raw])
        # structure: zone of each card, player of each zone, the root's children
        zone_of_card = b.card_state * 4 + b.card_group                         # into [B*4]
        st = torch.arange(B, device=dev)
        player_zone = (st.unsqueeze(1) * 4 + self.zone_ids).reshape(-1)         # [3B] zones under players
        player_of_zone = (st.unsqueeze(1) * 2 + self.zone_player).reshape(-1)
        stack_zone = st * 4 + 3
        root_seg = torch.cat([st.repeat_interleave(2), st, st, b.sp_state])   # players, stack, DENSE, sparse
        # initial typed node embeddings
        h_card = nt[CARD].float().expand(nc, d)
        h_zone = nt[Z_MYBF:Z_STACK + 1].float().repeat(B, 1)                    # [B*4]
        h_player = nt[ME:OPP + 1].float().repeat(B, 1)                          # [B*2]
        h_root = nt[ROOT].float().expand(B, d)
        for layers in self.local:
            h_card = layers["card"](h_card, card_leaves, card_seg)
            h_zone = layers["zone"](h_zone, h_card + et[E_CARD_ZONE].float(), zone_of_card)
            h_player = layers["player"](h_player, h_zone[player_zone] + et[E_ZONE_PLAYER].float(), player_of_zone)
            kids = torch.cat([h_player + et[E_PLAYER_ROOT].float(), h_zone[stack_zone] + et[E_STACK_ROOT].float(),
                              dense, sparse])
            h_root = layers["root"](h_root, kids, root_seg)
        # global layers over [root, me, opp, 4 zones, cards]
        fixed = torch.cat([h_root.unsqueeze(1), h_player.view(B, 2, d), h_zone.view(B, 4, d)], 1)
        cp = h_card.new_zeros(B, b.Cmax, d)
        cp[b.card_state, b.card_pos] = h_card
        x = torch.cat([fixed, cp], 1)
        keep = torch.cat([b.card_mask.new_ones(B, N_FIXED), b.card_mask], 1)
        for blk in self.blocks:
            x = blk(x, keep)
        x = self.norm(x)
        root = x[:, 0]
        value = self.value(root).squeeze(1).float()
        card_h = x[b.card_state, N_FIXED + b.card_pos]
        return self._option_scores(b, root, x, keep, card_h), value


# ------------------------------------------------------------------------------------------------
# building, saving, loading; candidate logits
# ------------------------------------------------------------------------------------------------

_CLASSES = {"mlp": MLPNet, "transformer": TransformerNet, "gnn": GraphNet}


def build(arch: str, config: dict | None = None) -> _Net:
    return _CLASSES[arch](full_config(arch, config))


def save(model: _Net, path, meta: dict | None = None) -> None:
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"format": "dzg-model-1", "arch": model.arch, "config": dict(model.config), "state_dict": sd,
                "meta": meta or {}}, path)


def load(path, device="cpu") -> tuple[_Net, dict]:
    """(model in eval mode on device, the checkpoint dict)."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = build(ck["arch"], ck["config"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck


def candidate_logits(opt_scores: torch.Tensor, b) -> tuple[torch.Tensor, torch.Tensor]:
    """(logit [ncand], log-probability [ncand]): a candidate's logit is the sum of its options' scores
    (an empty candidate scores 0); log-softmax within each state."""
    s = opt_scores.float()
    z = s.new_zeros(b.ncand).index_add(0, b.co_cand, s[b.co_opt])
    return z, z - segment_logsumexp(z, b.cand_state, b.B)[b.cand_state]


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def autocast(device: torch.device, amp: str):
    """bf16 autocast on CUDA only (a no-op context elsewhere)."""
    import contextlib
    if amp == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
