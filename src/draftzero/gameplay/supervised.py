"""
supervised.py — experiment #4's supervised trainer (docs/017 §6.2–6.4): a MageZero network trained
on top players' 17lands decisions at scale, a hyperparameter sweep, and an inference-speed check.

pretrain.py (experiment #2b) stays as it was. This module reuses imitation.py's data helpers (the
masked softmax and set NLL, `_batch`, `trunk`, `length_batches`, `dense_masks`, `policy_rows`,
`auc_score`, `new_net`, `load_checkpoint`) and adds what a multi-hour, multi-table run needs:

  tables     any list of HDF5 tables in imitation.write_h5's layout, each with a kind and a
             sampling weight (configuration, below). Per-row `weight` is honoured (policy losses);
             `meta/label_status` 2 (unreachable) rows train the value only. Features are streamed
             from disk and mapped through the feature vocab block by block, so the raw ids are
             never all in memory.
  vocab      MageZero's ignore-list rule (kept_feature_ids, k = 10) on the training rows, computed
             by a streaming equivalent (`FeatureStats`) that needs memory per distinct feature, not
             per token, so it scales to ~7M decisions. v0.2's hash width.
  model      default: MageZero v0.2's NetTransformer exactly as imitation.new_net builds it (2
             layers, d = 512, ff 1,024, 4 heads, 1,024-wide policy heads, tanh value), so the
             checkpoint loads in MageZero's inference server unchanged. Sweep options: layers,
             width (ff = 2 x width), transformer or MLP (mean-pooled EmbeddingBag + residual MLP
             blocks, same five heads: MageZero v0.2 has no embedding-bag `Net`), dropouts.
  losses     player priority head: set NLL over the masked legal softmax (one-hot labels are the
             |S| = 1 case); binary head: cross-entropy, slot 1 = yes; target head: set NLL over the
             legal targets; value: binary cross-entropy on P(win) = (1 + v) / 2 = sigmoid(2x),
             computed from the pre-tanh logit x (default), or MSE on v (#2b's loss). Options: the
             value loss on k random positions per game per epoch; TD(lambda) value targets along
             each game; auxiliary heads (turns left, final life difference) kept outside
             model_state_dict, so MageZero still loads the network.
  training   AdamW (Adam at weight_decay 0) with warm-up, constant or cosine LR; bf16 autocast on
             CUDA (fp16 + GradScaler where bf16 is missing), fp32 on MPS / CPU; token-budgeted,
             length-bucketed batches, one table per batch, tables interleaved deterministically in
             their configured shares; a background thread builds the next batches. The default
             batch sizes are pretrain.py's (64 rows up to 768 tokens, ~(768 / L)^2 x 64 above);
             on the 16 GB laptop use 16 (max_batch_rows, batch_tokens, batch_attn). Time-, step- or
             epoch-budgeted; validation every N minutes or steps; plateau stop on the combined
             validation measure.
  validation per table (and by label kind where a table has several): priority top-1 in the human's
             set, non-Pass top-1 (docs/013 §2.4: the best non-Pass option is among the human's plays,
             on rows with a play and >= 2 non-Pass options), top-3, set NLL, top-1 on Pass labels;
             binary accuracy / AUC; target top-1 / set NLL; value log-loss, MSE, AUC and AUC by the
             player's turn (1-2, 3-4, 5-6, 7-9, 10+). Pooled: policy/*, binary/*, value/*. Selection:
             best policy = the training mix of set NLL / cross-entropy, best value = value log-loss,
             combined = policy + value_weight x value. #2b's checkpoint reproduces docs/011's 73.1%
             top-1 and docs/013's 72.7% non-Pass top-1 with these.
  checkpoints  (out dir) latest.pt (everything needed to resume: weights, optimizer, scaler,
             sampler position, RNG states, counters, best scores, TD targets) every
             `latest_every_s` and at every evaluation; best_policy / best_value / best (combined)
             .pt.gz and ckpt/step*.pt.gz every `ckpt_every_s`, all weights-only in MageZero's
             format (model_state_dict + feature_vocab, gzip) plus `arch` and run info;
             final.pt.gz; evals.jsonl (every evaluation, flat keys, easy to plot); summary.json.

    python -m draftzero.gameplay.supervised train --out runs/exp4/main --config configs/exp4_train.yml
    python -m draftzero.gameplay.supervised train --out runs/exp4/main --resume
    python -m draftzero.gameplay.supervised train --out runs/exp4/main --resume --set time_budget_s=50000
    python -m draftzero.gameplay.supervised sweep --spec configs/exp4_sweep.yml --out runs/exp4/sweep
    python -m draftzero.gameplay.supervised bench-speed --checkpoint runs/exp4/main/best.pt.gz
    python -m draftzero.gameplay.supervised bench-speed --config configs/exp4_train.yml --vocab-rows 60000
    python -m draftzero.gameplay.supervised evaluate --checkpoint <ckpt.pt.gz> --split test

Configuration: YAML or JSON, merged over DEFAULTS (unknown keys are an error); `--set a.b=v`
overrides any key on the command line (values parsed as YAML). Tables:

    tables_dir: /workspace/data/gameplay/imitation/h5     # files are <name>_<split>.h5
    tables:
      - {name: turnstart,       kind: priority_set,   weight: 1.0}
      - {name: replay_priority, kind: priority_set,   weight: 1.0}
      - {name: replay_attack,   kind: binary,         weight: 1.0}
      - {name: block,           kind: target,         weight: 1.0}
      - {name: timing,          kind: priority_onehot, weight: 0.5,
         label_kind_names: {0: exact, 1: imputed}}
    arch: {type: transformer, layers: 2, width: 512}

  kind      priority_set (legal + set CSR, player priority head), priority_onehot (the same head;
            a `chosen_idx` column if the table has one, else its set CSR), binary (`y`, or the
            policy slots of /row; binary head), target (legal + set CSR, target head), soft
            (search visit distributions: below)
  weight    sampling weight: a table's share of training samples is weight x rows
  group     a name in `group_shares`: the group's tables take that fixed share of the samples
            (split among them by weight x rows); tables without a group share the rest
  value     whether its rows train and score the value head (default true)
  value_column  the column the value head trains on (default z, the game result); evaluation always
            scores against z
  phase     order of its positions within a turn, for TD(lambda) (default: list position)

Stage 6 (docs/017 §6.6): training on the network's own games from tools/imitation_scale/
selfplay_tables.py, anchored to the human-trained network.

  soft tables     legal CSR, set CSR + `set_p` (the search's visit distribution), `meta/atype` per
                  row (0 player priority head, 3 target head, 5 binary head); loss: cross-entropy
                  against the visit distribution over the legal options; `z_td` as value_column
  init_checkpoint start from this network: its weights, shape and feature vocab
  kl_weight       x KL(pi_ref || pi) on the soft tables' rows, pi_ref the frozen reference network
                  (`kl_ref`, default init_checkpoint) on the full state, computed once at the start;
                  annealed linearly to kl_weight_end over the run (max_steps, max_epochs or the time
                  budget). The value is anchored only through the human tables' rows
  freeze          [trunk, policy] trains the value head alone
  collapse        on human priority tables: the network's Pass rate (top-1 Pass where Pass is legal)
                  against the humans', and policy entropy; on soft tables: cross-entropy and top-1
                  agreement with the search, entropy, KL to the reference, value against results

17lands public data is CC BY 4.0 (https://www.17lands.com/public_datasets).
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import gzip
import hashlib
import io
import json
import math
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from draftzero.gameplay import imitation as im

im._torch_env()                    # PYTORCH_ENABLE_MPS_FALLBACK must be set before torch is imported

import torch                       # noqa: E402
import torch.nn.functional as F    # noqa: E402
from torch import nn               # noqa: E402

GLOBAL_MAX = 2 ** 31 - 1           # MageZero v0.2's feature hash width (model.GLOBAL_MAX, pretrain.py)
PASS_IDX = 0                       # assets/vocab/FDN_SPG.tsv: "A 0 Pass"
TABLE_KINDS = ("priority_set", "priority_onehot", "binary", "target", "soft")
POLICY_KINDS = ("priority_set", "priority_onehot", "target")      # tables with legal / set CSR
PRIORITY_KINDS = ("priority_set", "priority_onehot")
SOFT_HEADS = {0: "priority", 3: "target", 5: "binary"}             # soft tables' meta/atype -> head
FREEZABLE = ("trunk", "policy", "value")
AUX_TARGETS = {"turns_left": 10.0, "life_diff": 20.0,             # name -> scale the head predicts in
               "result": 1.0}  # the game's result on every row: a throwaway head that feeds the trunk the
                               # outcome signal while the value head keeps value_per_game positions a game
TURN_BUCKETS = (((1, 2), "1-2"), ((3, 4), "3-4"), ((5, 6), "5-6"), ((7, 9), "7-9"), ((10, 10 ** 9), "10+"))

ARCH_DEFAULT = {"type": "transformer", "layers": 2, "width": 512, "ff": None, "heads": 4, "head_hidden": 256,
                "policy_width": im.A_DIM, "dropout": 0.1, "pool_dropout": 0.2,
                "norm_first": False,   # pre-LayerNorm layers and a final LayerNorm (MageZero's are post-LN)
                "ffn": "relu",         # relu (MageZero's) | swiglu (gated, 2/3 x ff hidden units: about the same size) | gelu (MLP)
                "pool": "mean",        # mean (MageZero's) | attn (a learned query attends over the tokens)
                "value_tower": False,  # the value head on its own embedding + layers (value_layers, default layers)
                "value_layers": None,
                "value_tower_type": "transformer",   # transformer | mlp (a pooled EmbeddingBag + MLP blocks: cheap)
                "value_detach": False,  # the value head reads the shared features with the gradient stopped
                "mlp_norm": "layer",    # the MLP's blocks: layer | batch (BatchNorm1d) | none
                "bag_mode": "mean"}     # the MLP's token pooling: mean | sum | max
DEFAULT_TABLES = [{"name": "turnstart", "kind": "priority_set"},
                  {"name": "replay_priority", "kind": "priority_set"},
                  {"name": "replay_attack", "kind": "binary"}]
TABLE_DEFAULTS = {"name": None, "kind": None, "weight": 1.0, "value": True, "dir": None, "phase": None,
                  "label_kind_names": None, "value_column": None, "group": None}
DEFAULTS: dict[str, Any] = {
    # data
    "tables_dir": None,            # default: data/gameplay/imitation/h5
    "tables": DEFAULT_TABLES,
    "fraction": 1.0,               # share of the training GAMES used (meta/row), drawn with subset_seed
    "subset_seed": 0,
    "val_rows": 5000,              # per validation table, whole games, drawn with val_seed (None = all)
    "val_seed": 12345,
    "vocab_k": 10,                 # MageZero's ignore rule: drop features in <= k training states
    "vocab_max_rows": None,        # build the vocab on a random sample of this many training rows
    "data_cache": None,            # a directory: mapped tables are cached there (fast resume)
    # model
    "arch": ARCH_DEFAULT,
    # losses
    "policy_weight": 1.0, "binary_weight": 1.0, "target_weight": 1.0,
    "value_loss": "bce",           # bce: cross-entropy on P(win) = (1 + v) / 2; mse: (v - z)^2 as #2b
    "value_weight": 0.5,
    "value_per_game": None,        # k: value loss on only k random positions per game per epoch
    "value_target": "result",      # result | td
    "td_lambda": 0.95,
    "td_start_epochs": 0.25,       # TD targets replace the result after this many epochs ...
    "td_refresh_epochs": 1.0,      # ... and are recomputed from the network this often
    "aux_targets": [],             # turns_left, life_diff
    "aux_weight": 0.1,
    # stage 6: from a trained network, anchored to it (docs/017 §6.6)
    "init_checkpoint": None,       # start from this network (weights, shape, feature vocab)
    "kl_weight": 0.0,              # x KL(pi_ref || pi) on soft tables' rows ...
    "kl_weight_end": None,         # ... annealed linearly to this over the run (null: constant)
    "kl_ref": None,                # the reference network (default: init_checkpoint)
    "freeze": [],                  # any of trunk, policy, value: kept as they are
    "group_shares": {},            # table group -> fixed share of the training samples
    "act_weights": {},             # table name -> policy-loss weight multiplier on its training rows where the human
                                   # acted (a non-Pass play in the label set): against the networks' passivity
    # optimisation
    "lr": 3e-4, "warmup_steps": 300, "lr_schedule": "constant", "lr_min_frac": 0.1,
    "emb_init_std": None,          # embedding rows' initial std (None: MageZero's N(0, 1) keyed rows; scaled otherwise)
    "weight_decay": 0.0, "grad_clip": 1.0,
    "token_dropout": 0.3,
    "batch_tokens": 64 * 768,      # padded tokens per batch: rows <= batch_tokens // padded length L ...
    "batch_attn": 64 * 768 ** 2,   # ... and rows <= batch_attn // L^2 (attention memory; null: no cap) ...
    "max_batch_rows": 64,          # ... and <= this. The defaults are pretrain.py's batch_for exactly
    "eval_batch_rows": 64,         # imitation.length_batches base (rows shrink with L^2 above 768 tokens)
    "amp": "auto",                 # auto (bf16 on CUDA, fp16 if no bf16; off elsewhere) | bf16 | fp16 | off
    "device": "auto",
    "seed": 0,
    "prefetch": 3,                 # batches built ahead by a background thread (0 = inline)
    # budget, evaluation, checkpoints
    "time_budget_s": 3600,         # training time (evaluation and checkpoint time excluded)
    "max_steps": None, "max_epochs": None,
    "eval_every_s": 600, "eval_every_steps": None, "eval_at_start": True,
    "ckpt_every_s": 1800, "latest_every_s": 600,
    "patience_epochs": 1.0,        # stop when the combined measure has not improved for this long
    "min_delta": 1e-4,
}
DATA_KEYS = ("tables_dir", "tables", "fraction", "subset_seed", "val_rows", "val_seed", "vocab_k",
             "vocab_max_rows", "init_checkpoint")


# ================================================================================================
# configuration
# ================================================================================================

def full_arch(arch: dict | None, materialize: bool = True) -> dict:
    """The arch with every key. ff null means 2 x width: configs keep it null (so a sweep run that
    changes only the width gets ff = 2 x its width), models and checkpoints get the number."""
    a = {**ARCH_DEFAULT, **(arch or {})}
    unknown = set(a) - set(ARCH_DEFAULT)
    if unknown:
        raise KeyError(f"unknown arch keys {sorted(unknown)}")
    if a["type"] not in ("transformer", "mlp"):
        raise ValueError(f"arch.type must be transformer or mlp, not {a['type']!r}")
    if a["ff"] is None and materialize:
        a["ff"] = 2 * int(a["width"])
    for k in ("layers", "width", "ff", "heads", "head_hidden", "policy_width", "value_layers"):
        if a[k] is not None:
            a[k] = int(a[k])
    if a["ffn"] not in ("relu", "swiglu", "gelu") or a["pool"] not in ("mean", "attn"):
        raise ValueError(f"arch.ffn must be relu, swiglu or gelu and arch.pool mean or attn: {a['ffn']!r}, {a['pool']!r}")
    if a["type"] == "transformer" and a["ffn"] == "gelu":
        raise ValueError("arch.ffn gelu is for the MLP only")
    if a["mlp_norm"] not in ("layer", "batch", "none") or a["bag_mode"] not in ("mean", "sum", "max"):
        raise ValueError(f"arch.mlp_norm must be layer, batch or none and arch.bag_mode mean, sum or max: "
                         f"{a['mlp_norm']!r}, {a['bag_mode']!r}")
    if a["value_tower_type"] not in ("transformer", "mlp"):
        raise ValueError(f"arch.value_tower_type must be transformer or mlp, not {a['value_tower_type']!r}")
    return a


def extended_transformer(a: dict) -> bool:
    """A transformer TransformerNetX builds: any option beyond depth, width and pre-LN."""
    return a["type"] == "transformer" and (a["ffn"] != "relu" or a["pool"] != "mean" or bool(a["value_tower"]))


def emb_width(arch: dict) -> int:
    """The pooled embedding's width: two towers' worth with a value tower."""
    a = full_arch(arch)
    return a["width"] * (2 if a["value_tower"] and a["type"] == "transformer" else 1)


def magezero_loadable(arch: dict | None) -> bool:
    """True when MageZero v0.2's server (`NetTransformer(len(vocab))`) loads the checkpoint as it is:
    the default shape. Dropout rates don't matter (no weights)."""
    a = full_arch(arch)
    return (a["type"] == "transformer" and a["layers"] == 2 and a["width"] == 512 and a["ff"] == 1024
            and a["heads"] == 4 and a["head_hidden"] == 256 and a["policy_width"] == im.A_DIM
            and not a["norm_first"] and not extended_transformer(a) and not a["value_detach"])


def _table_spec(t: dict, i: int) -> dict:
    s = {**TABLE_DEFAULTS, **t}
    unknown = set(s) - set(TABLE_DEFAULTS)
    if unknown:
        raise KeyError(f"unknown table keys {sorted(unknown)} in {t}")
    if not s["name"] or s["kind"] not in TABLE_KINDS:
        raise ValueError(f"a table needs a name and a kind in {TABLE_KINDS}: {t}")
    if s["phase"] is None:
        s["phase"] = i
    s["weight"] = float(s["weight"])
    return s


def _merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in out:
            raise KeyError(f"unknown config key {path}{k}")
        if isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v, f"{path}{k}.") if k == "arch" else {**out[k], **v}
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_config(*layers: dict | None) -> dict:
    """DEFAULTS, then each layer merged on top (nested for `arch`, `tables` replaced whole)."""
    cfg = copy.deepcopy(DEFAULTS)
    for layer in layers:
        cfg = _merge(cfg, layer or {})
    cfg["arch"] = full_arch(cfg["arch"], materialize=False)
    cfg["tables"] = [_table_spec(t, i) for i, t in enumerate(cfg["tables"])]
    if cfg["tables_dir"] is None:
        cfg["tables_dir"] = str(im.OUT_DIR / "h5")
    checks = {"value_loss": ("bce", "mse"), "value_target": ("result", "td"), "lr_schedule": ("constant", "cosine"),
              "amp": ("auto", "bf16", "fp16", "off")}
    for k, ok in checks.items():
        if cfg[k] not in ok:
            raise ValueError(f"{k} must be one of {ok}, not {cfg[k]!r}")
    bad = [a for a in cfg["aux_targets"] if a not in AUX_TARGETS]
    if bad:
        raise ValueError(f"unknown aux_targets {bad}; known: {list(AUX_TARGETS)}")
    if cfg["lr_schedule"] == "cosine" and not (cfg["max_steps"] or cfg["max_epochs"]):
        raise ValueError("lr_schedule cosine needs max_steps or max_epochs")
    bad = [x for x in cfg["freeze"] if x not in FREEZABLE]
    if bad or len(set(cfg["freeze"])) == len(FREEZABLE):
        raise ValueError(f"freeze takes some of {FREEZABLE} (not all), not {cfg['freeze']}")
    if cfg["value_target"] == "td" and any(t["value_column"] not in (None, "z") for t in cfg["tables"]):
        raise ValueError("value_target td recomputes every table's value targets: drop value_column")
    groups = {t["group"] for t in cfg["tables"] if t["group"] is not None}
    if groups - set(cfg["group_shares"]):
        raise ValueError(f"table groups {sorted(groups - set(cfg['group_shares']))} have no group_shares entry")
    if groups and (sum(cfg["group_shares"][g] for g in groups) >= 1
                   or all(t["group"] is not None for t in cfg["tables"])):
        raise ValueError("group_shares must leave a share for the tables without a group")
    if (cfg["kl_weight"] or cfg["kl_weight_end"]) and not (cfg["kl_ref"] or cfg["init_checkpoint"]):
        raise ValueError("kl_weight needs a reference network: kl_ref or init_checkpoint")
    return cfg


def parse_overrides(items: list[str] | None) -> dict:
    """['arch.layers=4', 'lr=1e-3'] -> {'arch': {'layers': 4}, 'lr': 0.001} (values as YAML)."""
    import yaml
    out: dict = {}
    for it in items or []:
        key, eq, val = it.partition("=")
        if not eq:
            raise ValueError(f"--set wants key=value, got {it!r}")
        d = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        v = yaml.safe_load(val)
        d[parts[-1]] = float(v) if isinstance(v, str) and _is_float(v) else v
    return out


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_config_file(path: Path | None) -> dict:
    import yaml
    return (yaml.safe_load(Path(path).read_text()) or {}) if path else {}


def _jsonable(x):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    return json.loads(json.dumps(x, default=conv))


# ================================================================================================
# device and precision
# ================================================================================================

def pick_device(name: str = "auto") -> torch.device:
    return im.device() if name in (None, "auto") else torch.device(name)


def amp_dtype(amp: str, dev: torch.device):
    if amp == "off":
        return None
    if amp == "auto":
        if dev.type != "cuda":
            return None
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[amp]


def _autocast(dev, dtype):
    return torch.autocast(device_type=dev.type, dtype=dtype) if dtype is not None else contextlib.nullcontext()


def _fp32(dev, dtype):
    """The heads and losses run in fp32 even under autocast: they are cheap, and bf16 logits tie."""
    return torch.autocast(device_type=dev.type, enabled=False) if dtype is not None else contextlib.nullcontext()


def _sync(dev) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


# ================================================================================================
# models
# ================================================================================================

def _head(d_in: int, hidden: int, out: int, tanh: bool = False) -> nn.Sequential:
    mods = [nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, out)]
    return nn.Sequential(*mods, nn.Tanh()) if tanh else nn.Sequential(*mods)


class _Detach(nn.Module):
    def forward(self, x):
        return x.detach()


class _HeadsMixin:
    def _make_heads(self, d: int, a: dict) -> None:
        h, p = a["head_hidden"], a["policy_width"]
        self.player_priority_head = _head(d, h, p)
        self.opponent_priority_head = _head(d, h, p)
        self.target_head = _head(d, h, p)
        self.binary_head = _head(d, h, 2)
        self.value_head = _head(d, h, 1, tanh=True)
        if a["value_detach"]:          # the value loss trains its head only, never the shared features
            self.value_head = nn.Sequential(_Detach(), *self.value_head)

    def _outputs(self, emb):
        return (self.player_priority_head(emb), self.opponent_priority_head(emb), self.target_head(emb),
                self.binary_head(emb), self.value_head(emb).squeeze(-1))


class TransformerNet(_HeadsMixin, nn.Module):
    """NetTransformer at any depth / width: the same attribute names, the same five outputs from
    forward(indices, offsets) (bag starts), so imitation.trunk / heads work on it. Only the
    default shape is built as MageZero's own class (build_model)."""

    def __init__(self, num_embeddings: int, arch: dict):
        super().__init__()
        a = full_arch(arch)
        d = a["width"]
        self.input_dropout = 0.0
        self.embedding = nn.Embedding(num_embeddings, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=a["heads"], dim_feedforward=a["ff"],
                                           dropout=a["dropout"], batch_first=True, norm_first=bool(a["norm_first"]))
        # pre-LN stacks end with a LayerNorm (the residual stream is otherwise unnormalised)
        self.transformer = nn.TransformerEncoder(layer, num_layers=a["layers"], enable_nested_tensor=False,
                                                 norm=nn.LayerNorm(d) if a["norm_first"] else None)
        self.embedding_dropout = nn.Dropout(a["pool_dropout"])
        self._make_heads(d, a)

    def forward(self, indices, offsets):
        return self._outputs(im.trunk(self, indices, offsets))


def _norm1d(kind: str, d: int) -> nn.Module:
    return nn.LayerNorm(d) if kind == "layer" else nn.BatchNorm1d(d) if kind == "batch" else nn.Identity()


class _MLPBlock(nn.Module):
    """A pre-norm residual block: x + drop(fc2(act(fc1(norm(x))))). SwiGLU keeps about the size with
    2/3 x ff gated hidden units."""

    def __init__(self, d: int, ff: int, p: float, ffn: str = "relu", norm: str = "layer"):
        super().__init__()
        self.ffn = ffn
        self.norm = _norm1d(norm, d)
        h = max(1, 2 * ff // 3) if ffn == "swiglu" else ff
        self.fc1 = nn.Linear(d, 2 * h if ffn == "swiglu" else h)
        self.fc2 = nn.Linear(h, d)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h = self.fc1(self.norm(x))
        if self.ffn == "swiglu":
            a, b = h.chunk(2, -1)
            h = F.silu(a) * b
        else:
            h = F.gelu(h) if self.ffn == "gelu" else F.relu(h)
        return x + self.drop(self.fc2(h))


class BagMLPNet(_HeadsMixin, nn.Module):
    """The MLP arm: a mean-pooled EmbeddingBag (MageZero's pre-transformer `Net` summed a bag; v0.2
    has no such class), `layers` residual MLP blocks, the same five heads."""

    def __init__(self, num_embeddings: int, arch: dict):
        super().__init__()
        a = full_arch(arch)
        d = a["width"]
        self.embedding = nn.EmbeddingBag(num_embeddings, d, mode=a["bag_mode"])
        self.blocks = nn.ModuleList([_MLPBlock(d, a["ff"], a["dropout"], a["ffn"], a["mlp_norm"])
                                     for _ in range(a["layers"])])
        self.norm = _norm1d(a["mlp_norm"], d) if a["layers"] else nn.Identity()
        self.embedding_dropout = nn.Dropout(a["pool_dropout"])
        self._make_heads(d, a)

    def encode(self, indices, offsets):
        x = self.embedding(indices % self.embedding.num_embeddings, offsets)
        for b in self.blocks:
            x = b(x)
        return self.embedding_dropout(self.norm(x))

    def forward(self, indices, offsets):
        return self._outputs(self.encode(indices, offsets))


def _padded(indices, offsets, num_embeddings: int, pad_to: int | None = None):
    """(padded ids [B, L], mask [B, L]) as imitation.trunk pads: to `pad_to`, else the next bucket."""
    indices = indices % num_embeddings
    ends = torch.cat([offsets[1:], torch.tensor([indices.shape[0]], device=offsets.device)])
    lengths = ends - offsets
    max_len = pad_to or im.bucket_len(int(lengths.max().item()))
    lengths = lengths.clamp(max=max_len)
    ar = torch.arange(max_len, device=indices.device)
    mask = ar.unsqueeze(0) < lengths.unsqueeze(1)
    pos = offsets.unsqueeze(1) + ar.unsqueeze(0)
    return torch.where(mask, indices[pos.clamp(max=indices.shape[0] - 1)], torch.zeros_like(pos)), mask


class _Block(nn.Module):
    """A transformer layer, post- or pre-LN, with a ReLU or SwiGLU feed-forward."""

    def __init__(self, d: int, heads: int, ff: int, p: float, norm_first: bool, swiglu: bool):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        h = max(1, int(2 * ff / 3)) if swiglu else ff
        self.w1, self.w2 = nn.Linear(d, h), nn.Linear(h, d)
        self.w3 = nn.Linear(d, h) if swiglu else None
        self.drop = nn.Dropout(p)
        self.norm_first = norm_first

    def _ff(self, x):
        h = F.silu(self.w1(x)) * self.w3(x) if self.w3 is not None else F.relu(self.w1(x))
        return self.w2(self.drop(h))

    def _sa(self, x, pad):
        return self.attn(x, x, x, key_padding_mask=pad, need_weights=False)[0]

    def forward(self, x, pad):
        if self.norm_first:
            x = x + self.drop(self._sa(self.norm1(x), pad))
            return x + self.drop(self._ff(self.norm2(x)))
        x = self.norm1(x + self.drop(self._sa(x, pad)))
        return self.norm2(x + self.drop(self._ff(x)))


class _AttnPool(nn.Module):
    """A learned query attends over the state's tokens (Set Transformer's PMA with one seed)."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, mask):
        q = self.query.expand(x.shape[0], -1, -1)
        return self.norm(self.attn(q, x, x, key_padding_mask=~mask, need_weights=False)[0].squeeze(1))


class _Tower(nn.Module):
    """Embedding, transformer layers and pooling: a state's tokens to one vector."""

    def __init__(self, num_embeddings: int, a: dict, layers: int):
        super().__init__()
        d = a["width"]
        self.embedding = nn.Embedding(num_embeddings, d)
        self.blocks = nn.ModuleList([_Block(d, a["heads"], a["ff"], a["dropout"], bool(a["norm_first"]),
                                            a["ffn"] == "swiglu") for _ in range(layers)])
        self.final_norm = nn.LayerNorm(d) if a["norm_first"] else nn.Identity()
        self.pool = _AttnPool(d, a["heads"]) if a["pool"] == "attn" else None
        self.dropout = nn.Dropout(a["pool_dropout"])

    def forward(self, indices, offsets, pad_to=None):
        padded, mask = _padded(indices, offsets, self.embedding.num_embeddings, pad_to)
        x = self.embedding(padded)
        for b in self.blocks:
            x = b(x, ~mask)
        x = self.final_norm(x)
        if self.pool is not None:
            p = self.pool(x, mask)
        else:
            p = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp(min=1).unsqueeze(-1).float()
        return self.dropout(p)


class _BagTower(nn.Module):
    """The MLP arm's trunk as a tower (a mean-pooled EmbeddingBag and residual MLP blocks): a value
    tower at a fraction of a transformer tower's cost."""

    def __init__(self, num_embeddings: int, a: dict, layers: int):
        super().__init__()
        d = a["width"]
        self.embedding = nn.EmbeddingBag(num_embeddings, d, mode=a["bag_mode"])
        self.blocks = nn.ModuleList([_MLPBlock(d, a["ff"], a["dropout"], "relu" if a["ffn"] == "swiglu" else a["ffn"],
                                               a["mlp_norm"]) for _ in range(layers)])
        self.norm = _norm1d(a["mlp_norm"], d) if layers else nn.Identity()
        self.dropout = nn.Dropout(a["pool_dropout"])

    def forward(self, indices, offsets, pad_to=None):
        x = self.embedding(indices % self.embedding.num_embeddings, offsets)
        for b in self.blocks:
            x = b(x)
        return self.dropout(self.norm(x))


class _Slice(nn.Module):
    def __init__(self, a: int, b: int):
        super().__init__()
        self.a, self.b = a, b

    def forward(self, x):
        return x[..., self.a:self.b]


class TransformerNetX(_HeadsMixin, nn.Module):
    """Transformer variants beyond MageZero's layer (docs/018, stage 2's second round): a SwiGLU
    feed-forward, attention pooling, and a value tower (the value head on its own embedding and
    layers, so its loss doesn't shape the policy's features). The pooled embedding is the towers'
    outputs side by side, and each head reads its own part, so the trainer's code is unchanged."""

    def __init__(self, num_embeddings: int, arch: dict):
        super().__init__()
        a = full_arch(arch)
        d = a["width"]
        self.tower = _Tower(num_embeddings, a, a["layers"])
        self.embedding = self.tower.embedding           # the policy tower's: embedding init, embed_dim
        if not a["value_tower"]:
            self.value_tower = None
        elif a["value_tower_type"] == "mlp":
            self.value_tower = _BagTower(num_embeddings, a, 2 if a["value_layers"] is None else a["value_layers"])
        else:
            self.value_tower = _Tower(num_embeddings, a, a["value_layers"] or a["layers"])
        self._make_heads(d, a)
        if self.value_tower is not None:
            for name in ("player_priority_head", "opponent_priority_head", "target_head", "binary_head"):
                setattr(self, name, nn.Sequential(_Slice(0, d), *getattr(self, name)))
            self.value_head = nn.Sequential(_Slice(d, 2 * d), *self.value_head)

    def encode(self, indices, offsets, pad_to=None):
        e = self.tower(indices, offsets, pad_to)
        if self.value_tower is not None:
            e = torch.cat([e, self.value_tower(indices, offsets, pad_to)], -1)
        return e

    def forward(self, indices, offsets):
        return self._outputs(self.encode(indices, offsets))


class AuxHeads(nn.Module):
    """Auxiliary regression heads on the pooled embedding. Saved under `aux_state_dict`, outside
    model_state_dict, so MageZero's server still loads the network."""

    def __init__(self, d: int, names: list[str], hidden: int = 256):
        super().__init__()
        self.names = list(names)
        self.net = _head(d, hidden, len(self.names))

    def forward(self, emb):
        return self.net(emb)


def _set_dropouts(m, a: dict) -> None:
    m.embedding_dropout.p = a["pool_dropout"]
    for layer in m.transformer.layers:
        layer.dropout.p = layer.dropout1.p = layer.dropout2.p = a["dropout"]
        layer.self_attn.dropout = a["dropout"]


def build_model(arch: dict, num_embeddings: int, vocab=None, emb_std: float | None = None):
    """The network for `arch`. The default shape is imitation.new_net's NetTransformer itself.
    `vocab` initialises the embedding rows as pretrain.py did (magezero.vocab.initial_rows, N(0, 1)),
    scaled to `emb_std` when given."""
    a = full_arch(arch)
    if magezero_loadable(a):
        m = im.new_net(num_embeddings, a["policy_width"])
        _set_dropouts(m, a)
    elif extended_transformer(a):
        m = TransformerNetX(num_embeddings, a)
    elif a["type"] == "transformer":
        m = TransformerNet(num_embeddings, a)
    else:
        m = BagMLPNet(num_embeddings, a)
    if vocab is not None:
        from magezero.vocab import initial_rows
        with torch.no_grad():
            rows = torch.as_tensor(initial_rows(vocab.ids, a["width"]))
            rows = rows * float(emb_std) if emb_std is not None else rows
            m.embedding.weight.copy_(rows)
            if getattr(m, "value_tower", None) is not None:
                m.value_tower.embedding.weight.copy_(rows)
    return m


def infer_arch(sd: dict) -> dict:
    """The arch of a checkpoint saved without one (e.g. pretrain.py's): a transformer sized from
    its weights. The head count is not in the weights; MageZero's is 4."""
    layers = {int(k.split(".")[2]) for k in sd if k.startswith("transformer.layers.")}
    if not layers:
        raise ValueError("checkpoint has no `arch` and is not a transformer")
    return full_arch({"type": "transformer", "layers": max(layers) + 1, "width": sd["embedding.weight"].shape[1],
                      "ff": sd["transformer.layers.0.linear1.weight"].shape[0],
                      "head_hidden": sd["player_priority_head.0.weight"].shape[0],
                      "policy_width": sd["player_priority_head.2.weight"].shape[0],
                      "norm_first": "transformer.norm.weight" in sd})


def encode(model, idx, off, pad_to: int | None = None):
    """Pooled embedding [B, width] (training-mode dropouts follow model.training). `pad_to` (the
    batch's padded length, known on the CPU) spares imitation.trunk a GPU sync to find it."""
    if isinstance(model, BagMLPNet):
        return model.encode(idx, off)
    if isinstance(model, TransformerNetX):
        return model.encode(idx, off, pad_to)
    return im.trunk(model, idx, off, pad_to)


def value_logit(model, emb):
    """x with v = tanh(x): the value head without its Tanh."""
    return model.value_head[:-1](emb).squeeze(-1)


def embed_dim(model) -> int:
    return model.embedding.embedding_dim


# ================================================================================================
# losses
# ================================================================================================

def policy_nll(logits, legal, sset):
    """-log sum_{a in S and L} softmax_L(logits)[a] per row (inf where S and L share nothing). A
    one-hot label is the |S| = 1 case: the NLL of the chosen action."""
    return im.set_nll(im.masked_log_softmax(logits.float(), legal), sset & legal)


def value_bce_from_logit(x, z):
    """Cross-entropy of P(win) = (1 + tanh x) / 2 = sigmoid(2x) against (1 + z) / 2 (z in [-1, 1];
    soft for TD targets), from the logit: stable where tanh saturates."""
    return F.binary_cross_entropy_with_logits(2 * x.float(), (1 + z.float()) / 2, reduction="none")


def value_mse(x, z):
    return (torch.tanh(x.float()) - z.float()) ** 2


def soft_logits(model, emb, at):
    """[B, P] logits of soft-table rows, each from its decision type's head (SOFT_HEADS): player
    priority, target, or the binary head's two logits in columns 0 (no) and 1 (yes)."""
    pri = model.player_priority_head(emb)
    out = torch.where((at == 3)[:, None], model.target_head(emb), pri)
    bn = F.pad(model.binary_head(emb), (0, pri.shape[1] - 2), value=float("-inf"))
    return torch.where((at == 5)[:, None], bn, out)


def soft_ce(logits, legal, target):
    """(-sum_a target(a) log softmax_L(logits)(a) per row, the masked log-probabilities)."""
    lsm = im.masked_log_softmax(logits.float(), legal)
    return -torch.where(target > 0, target * lsm, torch.zeros_like(lsm)).sum(1), lsm


def value_subsample_mask(games: np.ndarray, k: int, seed: int, epoch: int,
                         eligible: np.ndarray | None = None) -> np.ndarray:
    """k positions per game (meta/row), drawn afresh each epoch; games with <= k eligible positions
    keep all of them. Deterministic in (seed, epoch)."""
    games = np.asarray(games)
    n = len(games)
    eligible = np.ones(n, bool) if eligible is None else np.asarray(eligible, bool)
    key = np.random.default_rng([int(seed), 0x5641, int(epoch)]).random(n)
    key[~eligible] = np.inf                       # ineligible rows rank last in their game
    order = np.lexsort((key, games))
    g = games[order]
    start = np.r_[True, g[1:] != g[:-1]]
    first = np.maximum.accumulate(np.where(start, np.arange(n), 0))
    rank = np.arange(n) - first
    mask = np.zeros(n, bool)
    mask[order] = rank < k
    return mask & eligible


def td_lambda_targets(games: np.ndarray, turns: np.ndarray, phase: np.ndarray, v: np.ndarray, z: np.ndarray,
                      lam: float) -> np.ndarray:
    """TD(lambda) value targets along each game: positions ordered by (turn, phase, input order);
    G_last = z, G_t = (1 - lam) v_{t+1} + lam G_{t+1}, with v the network's own value (in [-1, 1])
    at the game's next position. lam = 1 gives z everywhere, lam = 0 the next position's value."""
    n = len(games)
    if n == 0:
        return np.zeros(0, np.float32)
    order = np.lexsort((np.arange(n), phase, turns, games))
    g, vs, zs = np.asarray(games)[order], np.asarray(v, np.float64)[order], np.asarray(z, np.float64)[order]
    last = np.r_[g[1:] != g[:-1], True]
    # depth from the end of the game, per position
    end_idx = np.flatnonzero(last)
    game_end = np.repeat(end_idx, np.diff(np.r_[-1, end_idx]))
    depth = game_end - np.arange(n)
    G = np.empty(n)
    G[last] = zs[last]
    for d in range(1, int(depth.max()) + 1):
        pos = np.flatnonzero(depth == d)
        G[pos] = (1 - lam) * vs[pos + 1] + lam * G[pos + 1]
    out = np.empty(n, np.float32)
    out[order] = G
    return out


# ================================================================================================
# feature vocab (streaming kept_feature_ids)
# ================================================================================================

def _aggregate(ids, h1, h2, cnt):
    order = np.argsort(ids, kind="stable")
    s = ids[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]]) if len(s) else np.zeros(0, np.int64)
    if not len(s):
        return s, h1[:0], h2[:0], cnt[:0]
    return (s[starts], np.add.reduceat(h1[order], starts), np.add.reduceat(h2[order], starts),
            np.add.reduceat(cnt[order], starts))


class FeatureStats:
    """magezero.vocab.kept_feature_ids without the [states x features] matrix: per distinct feature
    id, the number of states it is active in and two 64-bit sums of per-state random keys. Two
    features active in exactly the same states have equal (count, sums); different state sets
    collide with probability ~2^-128. So kept() returns the same ids (drop ids in <= k states; of
    ids with identical state sets keep the smallest) in memory proportional to distinct features."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng([int(seed), 0x70CAB])
        self.ids = np.zeros(0, np.int64)
        self.h1 = np.zeros(0, np.uint64)
        self.h2 = np.zeros(0, np.uint64)
        self.cnt = np.zeros(0, np.int64)
        self.parts: list = []
        self.pending = 0
        self.n_states = 0

    def add(self, indices: np.ndarray, lengths: np.ndarray) -> None:
        indices = np.asarray(indices, np.int64)
        lengths = np.asarray(lengths, np.int64)
        n = len(lengths)
        top = np.iinfo(np.uint64).max
        k1 = self.rng.integers(0, top, size=n, dtype=np.uint64, endpoint=True)
        k2 = self.rng.integers(0, top, size=n, dtype=np.uint64, endpoint=True)
        bags = np.repeat(np.arange(n, dtype=np.int64), lengths)
        if len(indices) > 1:                          # set semantics: a repeat within a state counts once
            rising = indices[1:] > indices[:-1]
            ends = np.cumsum(lengths)[:-1] - 1
            rising[ends[(ends >= 0) & (ends < len(rising))]] = True
            if not rising.all():
                o = np.lexsort((indices, bags))
                bags, indices = bags[o], indices[o]
                keep = np.r_[True, (bags[1:] != bags[:-1]) | (indices[1:] != indices[:-1])]
                bags, indices = bags[keep], indices[keep]
        self.parts.append(_aggregate(indices, k1[bags], k2[bags], np.ones(len(indices), np.int64)))
        self.n_states += n
        self.pending += len(self.parts[-1][0])
        if self.pending > max(8_000_000, 2 * len(self.ids)):
            self._merge()

    def _merge(self) -> None:
        if not self.parts:
            return
        cat = lambda i, own: np.concatenate([own] + [p[i] for p in self.parts])   # noqa: E731
        self.ids, self.h1, self.h2, self.cnt = _aggregate(cat(0, self.ids), cat(1, self.h1), cat(2, self.h2),
                                                          cat(3, self.cnt))
        self.parts, self.pending = [], 0

    def kept(self, k: int = 10) -> np.ndarray:
        self._merge()
        m = self.cnt > k
        ids, cnt, h1, h2 = self.ids[m], self.cnt[m], self.h1[m], self.h2[m]
        order = np.lexsort((ids, h2, h1, cnt))       # identical state sets adjacent, smallest id first
        c, a, b = cnt[order], h1[order], h2[order]
        first = np.r_[True, (c[1:] != c[:-1]) | (a[1:] != a[:-1]) | (b[1:] != b[:-1])] if len(c) else c.astype(bool)
        return np.sort(ids[order][first])


# ================================================================================================
# tables
# ================================================================================================

@dataclass
class Table:
    """One split of one table, its rows mapped through the feature vocab."""
    name: str
    kind: str
    split: str
    spec: dict
    file_rows: np.ndarray            # row numbers in the HDF5 file
    rows: np.ndarray                 # mapped features (vocab rows), CSR with ptr
    ptr: np.ndarray
    game: np.ndarray                 # meta/row: the 17lands game
    turn: np.ndarray                 # meta/turn: the player's turn number
    z: np.ndarray                    # game result +1 / -1 (nan = unknown)
    w: np.ndarray                    # policy-loss weight per row (file `weight`; 0 if unreachable)
    legal_indptr: np.ndarray | None = None
    legal_idx: np.ndarray | None = None
    set_indptr: np.ndarray | None = None
    set_idx: np.ndarray | None = None
    y: np.ndarray | None = None      # binary label (1 = yes)
    lk: np.ndarray | None = None     # label kind code per row ...
    lk_names: list = field(default_factory=list)   # ... and its names
    aux: dict = field(default_factory=dict)        # aux target name -> float32 (nan = unknown)
    zv: np.ndarray | None = None     # the value-training target (value_column) where it is not z
    set_p: np.ndarray | None = None  # soft: probability per set entry (the search's visit share)
    atype: np.ndarray | None = None  # soft: meta/atype per row (SOFT_HEADS)
    ref_p: np.ndarray | None = None  # soft: the reference network's probability per legal entry
    mapped_share: float = 1.0
    notes: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.ptr) - 1


def table_path(cfg: dict, spec: dict, split: str) -> Path:
    return Path(spec.get("dir") or cfg["tables_dir"]) / f"{spec['name']}_{split}.h5"


def _sub_csr(indptr: np.ndarray, idx: np.ndarray, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lens = (indptr[sel + 1] - indptr[sel]).astype(np.int64)
    ptr = np.r_[0, np.cumsum(lens)].astype(np.int64)
    pos = np.repeat(indptr[sel] - ptr[:-1], lens) + np.arange(ptr[-1])
    return idx[pos], ptr


def _iter_blocks(f, off: np.ndarray, sel: np.ndarray, block: int = 8192):
    """(selected rows, their raw feature ids concatenated, lengths) block by block for sorted `sel`."""
    ds = f["indices"]
    n = len(off) - 1
    for a in range(0, n, block):
        b = min(n, a + block)
        lo, hi = np.searchsorted(sel, [a, b])
        if lo == hi:
            continue
        pick = sel[lo:hi]
        chunk = ds[off[a]:off[b]]
        lens = (off[pick + 1] - off[pick]).astype(np.int64)
        if hi - lo == b - a:
            idx = chunk
        else:
            p = np.r_[0, np.cumsum(lens)]
            idx = chunk[np.repeat(off[pick] - off[a] - p[:-1], lens) + np.arange(p[-1])]
        yield pick, idx.astype(np.int64), lens


def _games_of(path: Path) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as f:
        n = len(f["offsets"]) - 1
        return f["meta/row"][:].astype(np.int64) if "meta/row" in f else np.arange(n, dtype=np.int64)


def _read_labels(f, spec: dict, sel: np.ndarray) -> dict:
    kind = spec["kind"]
    n = len(f["offsets"]) - 1
    has = lambda k: k in f                                   # noqa: E731
    out: dict[str, Any] = {"notes": []}
    out["game"] = (f["meta/row"][:].astype(np.int64) if has("meta/row") else np.arange(n, dtype=np.int64))[sel]
    out["turn"] = (f["meta/turn"][:].astype(np.int32) if has("meta/turn") else np.zeros(n, np.int32))[sel]
    if has("z"):
        z = f["z"][:].astype(np.float32)
    elif has("meta/won"):
        z = np.where(f["meta/won"][:] > 0, 1.0, -1.0).astype(np.float32)
    else:
        z = np.full(n, np.nan, np.float32)
        out["notes"].append("no z / meta/won: no value target")
    out["z"] = z[sel]
    w = f["weight"][:].astype(np.float32) if has("weight") else np.ones(n, np.float32)
    if has("meta/label_status"):
        w = np.where(f["meta/label_status"][:] == 2, 0.0, w).astype(np.float32)
    out["w"] = w[sel]
    vc = spec.get("value_column")
    if vc not in (None, "z"):
        out["zv"] = f[vc][:].astype(np.float32)[sel]
    if kind == "soft":
        out["legal_idx"], out["legal_indptr"] = _sub_csr(f["legal_indptr"][:], f["legal_idx"][:], sel)
        sp = f["set_indptr"][:]
        out["set_idx"], out["set_indptr"] = _sub_csr(sp, f["set_idx"][:], sel)
        out["set_p"], _ = _sub_csr(sp, f["set_p"][:].astype(np.float32), sel)
        out["atype"] = f["meta/atype"][:][sel].astype(np.int8)
        unknown = set(np.unique(out["atype"]).tolist()) - set(SOFT_HEADS)
        if unknown:
            raise ValueError(f"soft table rows of meta/atype {sorted(unknown)}: no head for them")
    if kind in POLICY_KINDS:
        out["legal_idx"], out["legal_indptr"] = _sub_csr(f["legal_indptr"][:], f["legal_idx"][:], sel)
        if kind == "priority_onehot" and has("chosen_idx"):
            ch = f["chosen_idx"][:][sel].astype(np.int32)
            out["set_indptr"] = np.r_[0, np.cumsum(ch >= 0)].astype(np.int64)
            out["set_idx"] = ch[ch >= 0]
        else:
            if kind == "priority_onehot":
                out["notes"].append("priority_onehot without chosen_idx: its set CSR is the label")
            out["set_idx"], out["set_indptr"] = _sub_csr(f["set_indptr"][:], f["set_idx"][:], sel)
    if kind == "binary":
        if has("y"):
            out["y"] = f["y"][:][sel].astype(np.int64)
        else:
            out["y"] = f["row"][:, :2][sel].argmax(1).astype(np.int64)
            out["notes"].append("binary label from /row's policy slots (no y)")
    names = None
    if has("replay_label_kind"):
        raw = np.asarray(f["replay_label_kind"].asstr()[:], dtype=object)[sel]
        names = sorted(set(raw.tolist()))
        lut = {s: i for i, s in enumerate(names)}
        out["lk"] = np.asarray([lut[s] for s in raw], np.int16)
    elif has("label_kind"):
        codes = f["label_kind"][:][sel].astype(np.int64)
        given = {int(k): str(v) for k, v in (spec.get("label_kind_names") or {0: "exact", 1: "set"}).items()}
        uniq = np.unique(codes)
        names = [given.get(int(c), str(int(c))) for c in uniq]
        out["lk"] = np.searchsorted(uniq, codes).astype(np.int16)
    out["lk_names"] = names or []
    aux = {}
    if has("meta/num_turns"):
        aux["turns_left"] = (f["meta/num_turns"][:][sel] - out["turn"]).astype(np.float32)
    if has("meta/final_life_diff"):
        aux["life_diff"] = f["meta/final_life_diff"][:][sel].astype(np.float32)
    out["aux"] = aux
    return out


def _cache_key(path: Path, sel: np.ndarray, vocab, kind: str) -> str:
    st = path.stat()
    h = hashlib.sha1()
    h.update(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{kind}|v2".encode())
    h.update(np.asarray(sel, np.int64).tobytes())
    h.update(np.asarray(vocab.ids, np.int64).tobytes())
    return h.hexdigest()[:20]


def load_table(path: Path, spec: dict, split: str, sel: np.ndarray, vocab, *, cache: Path | None = None,
               log=print) -> Table:
    """Rows `sel` (sorted) of one HDF5 table: labels in memory, features mapped block by block."""
    import h5py
    sel = np.asarray(sel, np.int64)
    if cache is not None:
        kind_key = f"{spec['kind']}|{spec.get('value_column')}"
        cp = Path(cache) / f"{spec['name']}_{split}_{_cache_key(path, sel, vocab, kind_key)}.npz"
        if cp.exists():
            z = np.load(cp, allow_pickle=False)
            d = {k: z[k] for k in z.files}
            meta = json.loads(str(d.pop("__meta__")))
            aux = {k[5:]: d.pop(k) for k in list(d) if k.startswith("aux__")}
            return Table(name=spec["name"], kind=spec["kind"], split=split, spec=spec, aux=aux,
                         lk_names=meta["lk_names"], mapped_share=meta["mapped_share"], notes=meta["notes"],
                         **{k: (v if v.shape != () else None) for k, v in d.items()})
    dtype = np.uint16 if len(vocab) <= np.iinfo(np.uint16).max else np.int32
    with h5py.File(path, "r") as f:
        lab = _read_labels(f, spec, sel)
        off = f["offsets"][:]
        parts, lens, raw = [], [], 0
        for _, idx, ln in _iter_blocks(f, off, sel):
            r, p = vocab.map_csr(idx, np.r_[0, np.cumsum(ln)])
            parts.append(r.astype(dtype))
            lens.append(np.diff(p))
            raw += len(idx)
    rows = np.concatenate(parts) if parts else np.zeros(0, dtype)
    ptr = np.r_[0, np.cumsum(np.concatenate(lens) if lens else [])].astype(np.int64)
    t = Table(name=spec["name"], kind=spec["kind"], split=split, spec=spec, file_rows=sel, rows=rows, ptr=ptr,
              game=lab["game"], turn=lab["turn"], z=lab["z"], w=lab["w"],
              legal_indptr=lab.get("legal_indptr"), legal_idx=lab.get("legal_idx"),
              set_indptr=lab.get("set_indptr"), set_idx=lab.get("set_idx"), y=lab.get("y"), lk=lab.get("lk"),
              lk_names=lab["lk_names"], aux=lab["aux"], zv=lab.get("zv"), set_p=lab.get("set_p"),
              atype=lab.get("atype"), mapped_share=float(len(rows) / max(1, raw)), notes=lab["notes"])
    for note in t.notes:
        log(f"supervised: {spec['name']}_{split}: {note}")
    if cache is not None:
        Path(cache).mkdir(parents=True, exist_ok=True)
        arrs = {k: getattr(t, k) for k in ("file_rows", "rows", "ptr", "game", "turn", "z", "w", "legal_indptr",
                                            "legal_idx", "set_indptr", "set_idx", "y", "lk", "zv", "set_p", "atype")}
        arrs = {k: (np.asarray(0) if v is None else v) for k, v in arrs.items()}
        arrs.update({f"aux__{k}": v for k, v in t.aux.items()})
        arrs["__meta__"] = np.asarray(json.dumps({"lk_names": t.lk_names, "mapped_share": t.mapped_share,
                                                  "notes": t.notes}))
        tmp = cp.with_suffix(".tmp.npz")
        np.savez(tmp, **arrs)
        os.replace(tmp, cp)
    return t


@dataclass
class Data:
    vocab: Any
    train: list
    val: list
    info: dict

    def __post_init__(self):
        tr = self.train
        self.goff = np.r_[0, np.cumsum([t.n for t in tr])].astype(np.int64)
        cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)   # noqa: E731
        self.g_game = cat([t.game for t in tr], np.int64)
        self.g_turn = cat([t.turn for t in tr], np.int32)
        self.g_phase = cat([np.full(t.n, t.spec["phase"]) for t in tr], np.int32)
        self.g_z = cat([t.z if t.zv is None else t.zv for t in tr], np.float32)     # the value-training targets
        self.g_value_ok = cat([np.isfinite(t.z) & np.isfinite(t.z if t.zv is None else t.zv) & bool(t.spec["value"])
                               for t in tr], bool)

    @property
    def epoch_rows(self) -> int:
        return int(self.goff[-1])


def _val_selection(games_by_table: dict, cap: int | None, seed: int) -> dict:
    """Whole games, in one random order shared by every table, until each table has `cap` rows."""
    allg = np.unique(np.concatenate(list(games_by_table.values()))) if games_by_table else np.zeros(0, np.int64)
    rank = np.empty(len(allg), np.int64)
    rank[np.random.default_rng(seed).permutation(len(allg))] = np.arange(len(allg))
    out = {}
    for name, g in games_by_table.items():
        r = rank[np.searchsorted(allg, g)]
        if cap is None or len(g) <= cap:
            out[name] = np.arange(len(g))
            continue
        cnt = np.bincount(r, minlength=len(allg))
        thr = int(np.searchsorted(np.cumsum(cnt), cap)) + 1
        out[name] = np.flatnonzero(r < thr)
    return out


def _turns_left_proxy(tables: list) -> None:
    """turns_left where the table has no meta/num_turns: the game's last recorded player turn in the
    loaded tables minus this turn (turn starts are recorded for every non-terminal turn, so this is
    the turns left before the final one; approximate)."""
    if not tables:
        return
    games = np.concatenate([t.game for t in tables])
    turns = np.concatenate([t.turn for t in tables])
    u, inv = np.unique(games, return_inverse=True)
    mx = np.full(len(u), -1, np.int64)
    np.maximum.at(mx, inv, turns)
    for t in tables:
        if "turns_left" not in t.aux:
            t.aux["turns_left"] = (mx[np.searchsorted(u, t.game)] - t.turn).astype(np.float32)
            t.notes.append("turns_left = last recorded turn of the game - turn (proxy: no meta/num_turns)")


def acted_rows(t: Table) -> np.ndarray:
    """Rows whose label set holds a non-Pass play (the human acted)."""
    if t.set_indptr is None:
        return np.zeros(t.n, bool)
    cs = np.r_[0, np.cumsum(t.set_idx != PASS_IDX)]
    return (cs[t.set_indptr[1:]] - cs[t.set_indptr[:-1]]) > 0


def with_act_weights(data: Data, weights: dict, log=print) -> Data:
    """A copy of `data` whose named training tables weight their acted rows' policy loss by
    weights[name] (the shared tables, and every validation table, are left as they are)."""
    unknown = set(weights) - {t.name for t in data.train}
    if unknown:
        raise ValueError(f"act_weights names tables not in the data: {sorted(unknown)}")
    train = []
    for t in data.train:
        f = weights.get(t.name)
        if f is None or f == 1:
            train.append(t)
            continue
        a = acted_rows(t)
        train.append(replace(t, w=np.where(a, t.w * float(f), t.w).astype(np.float32)))
        log(f"supervised: {t.name}: policy weight x{f:g} on {int(a.sum())} of {t.n} rows where the human acted")
    return Data(vocab=data.vocab, train=train, val=data.val, info=data.info)


def ensure_aux(tables: list, names: list, log=print) -> None:
    """The derived aux targets `names` asks for, on every table that lacks them: turns_left (a proxy, see
    _turns_left_proxy) and result (the game's result). A sweep loads its data once, with the base config,
    so a run whose own config adds aux targets gets them here (before this, such runs trained an aux
    head with no targets and so no loss)."""
    if "turns_left" in names and any("turns_left" not in t.aux for t in tables):
        _turns_left_proxy([t for t in tables if t.split == "train"])
        _turns_left_proxy([t for t in tables if t.split != "train"])
    if "result" in names:
        for t in tables:
            if "result" not in t.aux:
                t.aux["result"] = t.z.astype(np.float32)
    for a in names:
        if not any(a in t.aux for t in tables):
            log(f"supervised: WARNING aux target {a} is in no table (life_diff needs a meta/final_life_diff column): "
                f"its head gets no loss")


def load_data(cfg: dict, *, vocab=None, splits: tuple = ("train", "val"), log=print) -> Data:
    """The training rows (a `fraction` of the training games), the feature vocab built on them
    (unless given), and the validation rows (whole games, `val_rows` per table), for every table."""
    import h5py
    from magezero.vocab import FeatureVocab
    cfg = resolve_config(cfg)
    t0 = time.monotonic()
    if vocab is None and cfg["init_checkpoint"]:
        vocab = checkpoint_vocab(cfg["init_checkpoint"])
        log(f"supervised: feature vocab of {cfg['init_checkpoint']} ({len(vocab)} features)")
    specs = cfg["tables"]
    cache = cfg["data_cache"]
    info: dict[str, Any] = {"tables": {}}
    train, val = [], []
    if "train" in splits:
        games = {s["name"]: _games_of(table_path(cfg, s, "train")) for s in specs}
        allg = np.unique(np.concatenate(list(games.values())))
        keep_g = allg
        if cfg["fraction"] < 1:
            rng = np.random.default_rng(cfg["subset_seed"])
            keep_g = np.sort(rng.choice(allg, max(1, int(round(cfg["fraction"] * len(allg)))), replace=False))
        sel = {s["name"]: np.flatnonzero(np.isin(games[s["name"]], keep_g)) for s in specs}
        info.update(train_games=int(len(keep_g)), train_games_all=int(len(allg)))
        if vocab is None:
            stats = FeatureStats(seed=cfg["subset_seed"])
            total = sum(len(v) for v in sel.values())
            vsel = dict(sel)
            if cfg["vocab_max_rows"] and total > cfg["vocab_max_rows"]:
                rng = np.random.default_rng([cfg["subset_seed"], 1])
                vsel = {k: np.sort(rng.choice(v, int(round(len(v) * cfg["vocab_max_rows"] / total)), replace=False))
                        for k, v in sel.items()}
            for s in specs:
                with h5py.File(table_path(cfg, s, "train"), "r") as f:
                    off = f["offsets"][:]
                    for _, idx, ln in _iter_blocks(f, off, vsel[s["name"]]):
                        stats.add(idx, ln)
            vocab = FeatureVocab(stats.kept(cfg["vocab_k"]), feature_hash_bins=GLOBAL_MAX)
            info.update(vocab_rows_built_on=int(stats.n_states))
            log(f"supervised: vocab {len(vocab)} features (k={cfg['vocab_k']}) from {stats.n_states} training "
                f"states ({time.monotonic() - t0:.0f} s)")
        for s in specs:
            train.append(load_table(table_path(cfg, s, "train"), s, "train", sel[s["name"]], vocab, cache=cache, log=log))
    for split in [x for x in splits if x != "train"]:
        games = {s["name"]: _games_of(table_path(cfg, s, split)) for s in specs}
        vs = _val_selection(games, cfg["val_rows"], cfg["val_seed"])
        for s in specs:
            val.append(load_table(table_path(cfg, s, split), s, split, vs[s["name"]], vocab, cache=cache, log=log))
    ensure_aux(train + val, cfg["aux_targets"], log=log)
    for t in train + val:
        info["tables"][f"{t.name}_{t.split}"] = {"rows": t.n, "games": int(len(np.unique(t.game))),
                                                 "mean_len": round(float(np.diff(t.ptr).mean()) if t.n else 0, 1),
                                                 "mapped_share": round(t.mapped_share, 4)}
    info.update(vocab_rows=len(vocab), load_s=round(time.monotonic() - t0, 1))
    log(f"supervised: data loaded in {info['load_s']} s: " + ", ".join(
        f"{k} {v['rows']} rows" for k, v in info["tables"].items()))
    return Data(vocab=vocab, train=train, val=val, info=info)


# ================================================================================================
# batches
# ================================================================================================

def _bucket_lens(lengths: np.ndarray) -> np.ndarray:
    b = np.asarray(im.BUCKETS)
    return b[np.minimum(np.searchsorted(b, lengths, side="left"), len(b) - 1)]


def table_shares(tables: list, cfg: dict) -> np.ndarray:
    """Each training table's share of the samples: weight x rows, normalised; a table group in
    `group_shares` takes its fixed share (split among its tables by weight x rows), the tables
    without a group split the rest."""
    raw = np.array([t.spec["weight"] * t.n for t in tables], np.float64)
    groups = [t.spec.get("group") for t in tables]
    out = np.zeros(len(tables))
    rest = 1.0 - sum(cfg["group_shares"][g] for g in set(groups) if g is not None)
    for g in set(groups):
        m = np.array([x == g for x in groups])
        tot = raw[m].sum()
        if tot > 0:
            out[m] = raw[m] / tot * (rest if g is None else cfg["group_shares"][g])
    return out / out.sum()


def epoch_length(tables: list, share: np.ndarray) -> int:
    """Samples per epoch: every row once, or with table groups, one pass over the tables without
    a group (stage 6: the self-play rows, with the human rows mixed in at their share)."""
    m = np.array([t.spec.get("group") is None for t in tables])
    n = np.array([t.n for t in tables], np.float64)
    if m.all() or not share[m].sum():
        return int(n.sum())
    return max(1, int(round(n[m].sum() / share[m].sum())))


def _dense_probs(indptr: np.ndarray, idx: np.ndarray, p: np.ndarray, sel: np.ndarray, width: int) -> np.ndarray:
    m = np.zeros((len(sel), width), np.float32)
    for j, r in enumerate(sel):
        a, b = indptr[r], indptr[r + 1]
        np.add.at(m[j], idx[a:b], p[a:b])
    return m


class Sampler:
    """Deterministic, resumable batch order. Each table runs its own epochs: a fresh permutation
    per epoch (seeded by seed, table, epoch), rows grouped by padded length bucket, cut into
    token-budgeted batches, batch order shuffled. Tables are interleaved so each gets weight x rows
    of the samples (the table furthest behind its share goes next). Token dropout draws from the
    sampler's own generator, so state_dict() after a batch reproduces everything that follows."""

    def __init__(self, data: Data, cfg: dict):
        self.data, self.cfg = data, cfg
        self.seed = int(cfg["seed"])
        self.td = float(cfg["token_dropout"])
        self.P = cfg["arch"]["policy_width"]
        self.buckets = [_bucket_lens((np.diff(t.ptr) * (1 - self.td)).astype(np.int64)) for t in data.train]
        self.share = table_shares(data.train, cfg)
        self.epoch_len = epoch_length(data.train, self.share)
        T = len(data.train)
        self.state = {"epoch": [0] * T, "cursor": [0] * T, "drawn": [0] * T, "seen": 0, "batches": 0}
        self.rng = np.random.default_rng([self.seed, 0xD20])
        self._batches: dict = {}
        self._vmask: tuple | None = None

    def rows_for(self, L: int) -> int:
        """Rows in a batch padded to L tokens: the token budget, the attention budget (memory grows
        with L^2: a 64 x 2,048 batch took ~20 GB on the 16 GB laptop) and the row cap."""
        cap = [int(self.cfg["max_batch_rows"]), int(self.cfg["batch_tokens"]) // int(L)]
        if self.cfg["batch_attn"]:
            cap.append(int(self.cfg["batch_attn"]) // int(L) ** 2)
        return max(1, min(cap))

    def epoch_batches(self, ti: int, e: int) -> list:
        key = (ti, e)
        if key not in self._batches:
            self._batches = {k: v for k, v in self._batches.items() if k[0] != ti}
            rng = np.random.default_rng([self.seed, 0xBA7C, ti, e])
            perm = rng.permutation(self.data.train[ti].n)
            bk = self.buckets[ti][perm]
            out = []
            for L in np.unique(bk):
                rows = perm[bk == L]
                step = self.rows_for(int(L))
                out.extend(rows[i:i + step] for i in range(0, len(rows), step))
            self._batches[key] = [out[i] for i in rng.permutation(len(out))]
        return self._batches[key]

    def next(self) -> tuple[int, np.ndarray]:
        s = self.state
        drawn = np.asarray(s["drawn"], np.float64)
        ok = self.share > 0
        ratio = np.where(ok, drawn / np.where(ok, self.share, 1), np.inf)
        ti = int(np.argmin(ratio))
        bl = self.epoch_batches(ti, s["epoch"][ti])
        sel = np.sort(bl[s["cursor"][ti]])
        s["cursor"][ti] += 1
        if s["cursor"][ti] >= len(bl):
            s["epoch"][ti] += 1
            s["cursor"][ti] = 0
        return ti, sel

    def value_mask(self, epoch: int) -> tuple[np.ndarray, float]:
        """(rows whose value loss counts this epoch, their share of the eligible rows)."""
        d, k = self.data, self.cfg["value_per_game"]
        if self._vmask is None or self._vmask[0] != epoch:
            if k:
                m = value_subsample_mask(d.g_game, int(k), self.seed, epoch, d.g_value_ok)
                frac = float(m.sum() / max(1, d.g_value_ok.sum()))
            else:
                m, frac = d.g_value_ok, 1.0
            self._vmask = (epoch, m, frac)
        return self._vmask[1], self._vmask[2]

    def make_batch(self, ti: int, sel: np.ndarray, pin: bool = False) -> dict:
        t = self.data.train[ti]
        s = self.state
        vepoch = s["seen"] // max(1, self.epoch_len)
        idx, off = im._batch(t.rows, t.ptr, sel, "cpu", self.td, self.rng)
        lens = np.diff(np.r_[off.numpy(), idx.numel()])
        gidx = self.data.goff[ti] + sel
        vm, frac = self.value_mask(int(vepoch))
        b = {"ti": ti, "n": len(sel), "idx": idx, "off": off, "pad": im.bucket_len(int(lens.max())),
             "gidx": torch.from_numpy(gidx),
             "w": torch.from_numpy(t.w[sel]), "vmask": torch.from_numpy(vm[gidx]),
             "vfrac": frac, "veligible": int(self.data.g_value_ok[gidx].sum())}
        if t.kind in POLICY_KINDS:
            b["L"] = torch.from_numpy(im.dense_masks(t.legal_indptr, t.legal_idx, sel, self.P))
            b["S"] = torch.from_numpy(im.dense_masks(t.set_indptr, t.set_idx, sel, self.P))
        if t.kind == "soft":
            b["L"] = torch.from_numpy(im.dense_masks(t.legal_indptr, t.legal_idx, sel, self.P))
            b["Pt"] = torch.from_numpy(_dense_probs(t.set_indptr, t.set_idx, t.set_p, sel, self.P))
            b["at"] = torch.from_numpy(t.atype[sel].astype(np.int64))
            if t.ref_p is not None:
                b["R"] = torch.from_numpy(_dense_probs(t.legal_indptr, t.legal_idx, t.ref_p, sel, self.P))
        if t.kind == "binary":
            b["y"] = torch.from_numpy(t.y[sel])
        names = self.cfg["aux_targets"]
        if names:
            b["aux"] = torch.from_numpy(np.stack([t.aux.get(a, np.full(t.n, np.nan, np.float32))[sel]
                                                  for a in names], 1).astype(np.float32))
        s["seen"] += len(sel)
        s["drawn"][ti] += len(sel)
        s["batches"] += 1
        b["state"] = self.state_dict()
        if pin:
            for k, v in b.items():
                if isinstance(v, torch.Tensor):
                    b[k] = v.pin_memory()
        return b

    def state_dict(self) -> dict:
        return {**copy.deepcopy(self.state), "rng": self.rng.bit_generator.state}

    def load_state_dict(self, d: dict) -> None:
        d = copy.deepcopy(d)
        self.rng.bit_generator.state = d.pop("rng")
        self.state = d


class Prefetcher:
    """Builds batches on a background thread (`depth` ahead); depth 0 builds them inline."""

    def __init__(self, sampler: Sampler, depth: int, pin: bool):
        self.s, self.depth, self.pin = sampler, depth, pin
        self.q: queue.Queue = queue.Queue(maxsize=max(1, depth))
        self.stop = threading.Event()
        self.err: BaseException | None = None
        self.th = None
        if depth > 0:
            self.th = threading.Thread(target=self._run, daemon=True)
            self.th.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                b = self.s.make_batch(*self.s.next(), pin=self.pin)
                while not self.stop.is_set():
                    try:
                        self.q.put(b, timeout=0.2)
                        break
                    except queue.Full:
                        continue
        except BaseException as e:      # noqa: BLE001 - re-raised in the consumer
            self.err = e

    def get(self) -> dict:
        if self.th is None:
            return self.s.make_batch(*self.s.next(), pin=self.pin)
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
def table_outputs(model, t: Table, dev, dtype, *, batch_rows: int = 128, aux: AuxHeads | None = None) -> dict:
    """Per row of a table: value logit, and by kind the legal-option logits (policy_rows' input) and
    set NLL, or P(yes) and cross-entropy; aux predictions in their units."""
    P = model.player_priority_head[-1].out_features
    out: dict[str, Any] = {"vx": np.zeros(t.n, np.float32)}
    if t.kind in POLICY_KINDS + ("soft",):
        out["scores"] = [None] * t.n
        out["nll"] = np.full(t.n, np.nan)
    if t.kind == "binary":
        out["p_yes"] = np.zeros(t.n)
        out["ce"] = np.zeros(t.n)
    if aux is not None:
        out["aux"] = np.zeros((t.n, len(aux.names)), np.float32)
    for sel in im.length_batches(np.diff(t.ptr), batch_rows):
        idx, off = im._batch(t.rows, t.ptr, sel, dev)
        with _autocast(dev, dtype):
            emb = encode(model, idx, off)
        with _fp32(dev, dtype):
            emb = emb.float()
            out["vx"][sel] = value_logit(model, emb).cpu().numpy()
            if t.kind == "soft":
                L = torch.as_tensor(im.dense_masks(t.legal_indptr, t.legal_idx, sel, P), device=dev)
                Pt = torch.as_tensor(_dense_probs(t.set_indptr, t.set_idx, t.set_p, sel, P), device=dev)
                logits = soft_logits(model, emb, torch.as_tensor(t.atype[sel].astype(np.int64), device=dev))
                out["nll"][sel] = soft_ce(logits, L, Pt)[0].cpu().numpy()
                lg = logits.cpu().numpy()
                for j, r in enumerate(sel):
                    out["scores"][r] = lg[j, t.legal_idx[t.legal_indptr[r]:t.legal_indptr[r + 1]]].astype(np.float64)
            if t.kind in POLICY_KINDS:
                head = model.target_head if t.kind == "target" else model.player_priority_head
                logits = head(emb)
                L = torch.as_tensor(im.dense_masks(t.legal_indptr, t.legal_idx, sel, P), device=dev)
                S = torch.as_tensor(im.dense_masks(t.set_indptr, t.set_idx, sel, P), device=dev)
                out["nll"][sel] = policy_nll(logits, L, S).cpu().numpy()
                lg = logits.cpu().numpy()
                for j, r in enumerate(sel):
                    out["scores"][r] = lg[j, t.legal_idx[t.legal_indptr[r]:t.legal_indptr[r + 1]]].astype(np.float64)
            if t.kind == "binary":
                bl = model.binary_head(emb)
                y = torch.as_tensor(t.y[sel], device=dev)
                out["p_yes"][sel] = torch.softmax(bl, -1)[:, 1].cpu().numpy()
                out["ce"][sel] = F.cross_entropy(bl, y, reduction="none").cpu().numpy()
            if aux is not None:
                scale = torch.tensor([AUX_TARGETS[a] for a in aux.names], device=dev)
                out["aux"][sel] = (aux(emb) * scale).cpu().numpy()
    return out


def _policy_rows_ex(t: Table, scores: list, drop_pass: bool) -> dict:
    """Per-row top-1 / top-3 / NLL with imitation.policy_rows' semantics (ties broken in
    expectation) for rows with a usable label. drop_pass removes Pass from the legal options and
    the label: the non-Pass top-1 of docs/013 §2.4 and docs/014 §2.5 (the network's best non-Pass
    option is among the human's plays), scored where the human made a play and at least two
    non-Pass options were legal (chance 48% on #2b's test rows)."""
    n = t.n
    li_parts, s_parts, sc_parts, ls = [], [], [], np.full(n, 2, np.int32)
    for r in range(n):
        li = t.legal_idx[t.legal_indptr[r]:t.legal_indptr[r + 1]]
        S = t.set_idx[t.set_indptr[r]:t.set_indptr[r + 1]]
        sc = scores[r]
        if drop_pass:
            keep = li != PASS_IDX
            li, sc, S = li[keep], sc[keep], S[S != PASS_IDX]
        li_parts.append(li)
        s_parts.append(S)
        sc_parts.append(sc)
        if t.w[r] > 0 and len(li) >= (2 if drop_pass else 1) and np.isin(S, li).any():
            ls[r] = 1
    lp = np.r_[0, np.cumsum([len(x) for x in li_parts])]
    sp = np.r_[0, np.cumsum([len(x) for x in s_parts])]
    view = {"legal_indptr": lp, "legal_idx": np.concatenate(li_parts) if n else np.zeros(0, np.int32),
            "set_indptr": sp, "set_idx": np.concatenate(s_parts) if n else np.zeros(0, np.int32),
            "meta/label_status": ls}
    pr = im.policy_rows(view, sc_parts, probabilistic=not drop_pass)
    full = {k: np.full(n, np.nan) for k in ("top1", "top3", "nll")}
    rows = pr["rows"].astype(np.int64)
    for k in ("top1", "top3", "nll"):
        if k in pr and len(rows):
            full[k][rows] = pr[k]
    chance = np.full(n, np.nan)
    for r in rows:
        nl = lp[r + 1] - lp[r]
        chance[r] = np.isin(view["legal_idx"][lp[r]:lp[r + 1]], view["set_idx"][sp[r]:sp[r + 1]]).sum() / nl
    full["chance"] = chance
    return full


def _mean(x) -> float | None:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def _softmax(s: np.ndarray) -> np.ndarray:
    e = np.exp(s - s.max())
    return e / e.sum()


def _entropy(p: np.ndarray) -> float:
    q = p[p > 0]
    return float(-(q * np.log(q)).sum())


def collapse_rows(t: Table, scores: list) -> dict:
    """Per row of a human priority table (nan where it does not apply): whether the network's top
    choice is Pass (rows where Pass and something else are legal), whether the human only passed
    there, and the entropy of the network's distribution over the legal options. A self-trained
    policy drifting into passivity shows here first (docs/016: the search's passivity)."""
    net_pass, hum_pass, ent = (np.full(t.n, np.nan) for _ in range(3))
    for r in range(t.n):
        li = t.legal_idx[t.legal_indptr[r]:t.legal_indptr[r + 1]]
        if len(li) < 2 or t.w[r] <= 0:
            continue
        p = _softmax(scores[r])
        ent[r] = _entropy(p)
        if PASS_IDX in li:
            net_pass[r] = float(li[int(np.argmax(p))] == PASS_IDX)
            hum_pass[r] = float(np.array_equal(t.set_idx[t.set_indptr[r]:t.set_indptr[r + 1]], [PASS_IDX]))
    return {"pass_top1": net_pass, "pass_human": hum_pass, "entropy": ent}


def priority_metrics(t: Table, o: dict) -> dict:
    a = _policy_rows_ex(t, o["scores"], drop_pass=False)
    b = _policy_rows_ex(t, o["scores"], drop_pass=True)
    c = collapse_rows(t, o["scores"])
    pass_only = np.array([np.array_equal(t.set_idx[t.set_indptr[r]:t.set_indptr[r + 1]], [PASS_IDX])
                          for r in range(t.n)], bool)

    def block(m):
        return {"n": int(np.isfinite(a["top1"][m]).sum()), "top1": _mean(a["top1"][m]), "top3": _mean(a["top3"][m]),
                "set_nll": _mean(a["nll"][m]), "n_nonpass": int(np.isfinite(b["top1"][m]).sum()),
                "top1_nonpass": _mean(b["top1"][m]), "chance_nonpass": _mean(b["chance"][m]),
                "n_pass_rows": int((np.isfinite(a["top1"]) & pass_only & m).sum()),
                "top1_pass_rows": _mean(a["top1"][m & pass_only]), "pass_top1": _mean(c["pass_top1"][m]),
                "pass_human": _mean(c["pass_human"][m]), "entropy": _mean(c["entropy"][m])}

    out = block(np.ones(t.n, bool))
    if t.lk is not None and len(t.lk_names) > 1:
        for code, name in enumerate(t.lk_names):
            for k, v in block(t.lk == code).items():
                out[f"{name}/{k}"] = v
    out["_rows"] = (a, b, c)
    return out


def soft_metrics(t: Table, o: dict) -> dict:
    """A soft table (the network's own searched decisions): cross-entropy against the search's
    visit distribution, top-1 among the search's most-visited options, the network's entropy and
    the search's, KL(reference || network) where a reference is loaded, and Pass at the top
    where Pass is legal (network, search); overall and by decision type."""
    n = t.n
    top1, ent, sent, kl, npass, spass = (np.full(n, np.nan) for _ in range(6))
    for r in range(n):
        a, b = t.legal_indptr[r], t.legal_indptr[r + 1]
        li = t.legal_idx[a:b]
        S = t.set_idx[t.set_indptr[r]:t.set_indptr[r + 1]]
        sp = t.set_p[t.set_indptr[r]:t.set_indptr[r + 1]]
        if len(li) < 2 or not len(S):
            continue
        p = _softmax(o["scores"][r])
        top = set(S[sp >= sp.max() - 1e-6].tolist())
        best = int(li[int(np.argmax(p))])
        top1[r] = float(best in top)
        ent[r] = _entropy(p)
        sent[r] = _entropy(sp.astype(np.float64))
        if t.ref_p is not None:
            rp = t.ref_p[a:b].astype(np.float64)
            m = rp > 0
            kl[r] = float((rp[m] * (np.log(rp[m]) - np.log(np.maximum(p[m], 1e-12)))).sum())
        if t.atype[r] == 0 and PASS_IDX in li:
            npass[r] = float(best == PASS_IDX)
            spass[r] = float(top == {PASS_IDX})
    out = {"n": int(np.isfinite(top1).sum()), "ce": _mean(o["nll"]), "top1_search": _mean(top1),
           "entropy": _mean(ent), "search_entropy": _mean(sent), "pass_top1": _mean(npass),
           "pass_search": _mean(spass)}
    if t.ref_p is not None:
        out["kl_ref"] = _mean(kl)
    for code, name in SOFT_HEADS.items():
        m = t.atype == code
        out.update({f"{name}/n": int(np.isfinite(top1[m]).sum()), f"{name}/top1_search": _mean(top1[m]),
                    f"{name}/ce": _mean(o["nll"][m])})
    return out


def target_metrics(t: Table, o: dict) -> dict:
    a = _policy_rows_ex(t, o["scores"], drop_pass=False)
    out = {"n": int(np.isfinite(a["top1"]).sum()), "top1": _mean(a["top1"]), "top3": _mean(a["top3"]),
           "set_nll": _mean(a["nll"]), "chance": _mean(a["chance"])}
    if t.lk is not None and len(t.lk_names) > 1:
        for c, name in enumerate(t.lk_names):
            m = t.lk == c
            out.update({f"{name}/n": int(np.isfinite(a["top1"][m]).sum()), f"{name}/top1": _mean(a["top1"][m])})
    return out


def binary_metrics(y: np.ndarray, p: np.ndarray, ce: np.ndarray) -> dict:
    if not len(y):
        return {"n": 0}
    return {"n": int(len(y)), "acc": float(((p > 0.5).astype(int) == y).mean()),
            "auc": im.auc_score(p, y.astype(np.float64)), "ce": float(ce.mean())}


def value_metrics(vx: np.ndarray, z: np.ndarray, turn: np.ndarray) -> dict:
    """Value head against game results: log-loss and accuracy of P(win) = sigmoid(2x), MSE of v =
    tanh(x) against z (#2b's loss), calibration error (ten P(win) bins), AUC overall and by the
    player's turn."""
    ok = np.isfinite(z)
    vx, z, turn = vx[ok].astype(np.float64), z[ok], turn[ok]
    if not len(z):
        return {"n": 0}
    won = (z > 0).astype(np.float64)
    p = np.clip(1 / (1 + np.exp(-2 * vx)), 1e-7, 1 - 1e-7)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[bins == k].mean() - won[bins == k].mean()) * (bins == k).sum() for k in range(10)
              if (bins == k).any()) / len(z)
    out = {"n": int(len(z)), "logloss": float(-(won * np.log(p) + (1 - won) * np.log(1 - p)).mean()),
           "mse": float(((np.tanh(vx) - z) ** 2).mean()), "acc": float(((vx > 0) == (won > 0)).mean()),
           "auc": im.auc_score(vx, won), "ece": float(ece)}
    for (lo, hi), lab in TURN_BUCKETS:
        m = (turn >= lo) & (turn <= hi)
        out[f"auc_t{lab}"] = im.auc_score(vx[m], won[m]) if m.sum() >= 2 else None
        out[f"n_t{lab}"] = int(m.sum())
    return out


def evaluate(model, tables: list, cfg: dict, dev, dtype=None, *, aux: AuxHeads | None = None,
             shares: dict | None = None) -> dict:
    """Every validation measure, flat: '<table>/<metric>' per table (by label kind where tables
    have more than one), 'policy/*' pooled over the priority tables, 'binary/*', 'value/*' pooled
    over the value tables, and the selection measures 'select/policy_loss' (the training mix of
    set NLL / cross-entropy), 'select/value_logloss' and 'select/combined'."""
    was = model.training
    model.eval()
    if aux is not None:
        aux.eval()
    res: dict[str, Any] = {}
    pol_parts, vx_all, z_all, turn_all, by, bp, bce = [], [], [], [], [], [], []
    pri_rows = []
    kw = {"priority_set": cfg["policy_weight"], "priority_onehot": cfg["policy_weight"],
          "target": cfg["target_weight"], "binary": cfg["binary_weight"], "soft": cfg["policy_weight"]}
    for t in tables:
        if t.n == 0:
            continue
        o = table_outputs(model, t, dev, dtype, batch_rows=cfg["eval_batch_rows"], aux=aux)
        if t.kind in PRIORITY_KINDS:
            m = priority_metrics(t, o)
            pri_rows.append(m.pop("_rows"))
            loss = m["set_nll"]
        elif t.kind == "soft":
            m = soft_metrics(t, o)
            loss = m["ce"]
        elif t.kind == "target":
            m = target_metrics(t, o)
            loss = m["set_nll"]
        else:
            m = binary_metrics(t.y, o["p_yes"], o["ce"])
            by.append(t.y)
            bp.append(o["p_yes"])
            bce.append(o["ce"])
            loss = m.get("ce")
        if loss is not None:
            share = (shares or {}).get(t.name, t.n)
            pol_parts.append((share, kw[t.kind] * loss))
        if t.spec["value"]:
            vm = value_metrics(o["vx"], t.z, t.turn)
            m.update({"value_logloss": vm.get("logloss"), "value_auc": vm.get("auc"), "value_ece": vm.get("ece")})
            vx_all.append(o["vx"])
            z_all.append(t.z)
            turn_all.append(t.turn)
        if aux is not None:
            for j, name in enumerate(aux.names):
                tgt = t.aux.get(name)
                if tgt is not None and np.isfinite(tgt).any():
                    ok = np.isfinite(tgt)
                    m[f"aux_{name}_mae"] = float(np.abs(o["aux"][ok, j] - tgt[ok]).mean())
        res.update({f"{t.name}/{k}": v for k, v in m.items()})
    if pri_rows:
        a = {k: np.concatenate([x[0][k] for x in pri_rows]) for k in ("top1", "top3", "nll")}
        b = {k: np.concatenate([x[1][k] for x in pri_rows]) for k in ("top1", "chance")}
        c = {k: np.concatenate([x[2][k] for x in pri_rows]) for k in ("pass_top1", "pass_human", "entropy")}
        res.update({"policy/top1": _mean(a["top1"]), "policy/top3": _mean(a["top3"]), "policy/set_nll": _mean(a["nll"]),
                    "policy/top1_nonpass": _mean(b["top1"]), "policy/chance_nonpass": _mean(b["chance"]),
                    "policy/n": int(np.isfinite(a["top1"]).sum()), "policy/pass_top1": _mean(c["pass_top1"]),
                    "policy/pass_human": _mean(c["pass_human"]), "policy/entropy": _mean(c["entropy"])})
    if by:
        res.update({f"binary/{k}": v for k, v in binary_metrics(np.concatenate(by), np.concatenate(bp),
                                                               np.concatenate(bce)).items()})
    if vx_all:
        res.update({f"value/{k}": v for k, v in value_metrics(np.concatenate(vx_all), np.concatenate(z_all),
                                                             np.concatenate(turn_all)).items()})
    sw = sum(s for s, _ in pol_parts)
    res["select/policy_loss"] = float(sum(s * l for s, l in pol_parts) / sw) if sw else None
    res["select/value_logloss"] = res.get("value/logloss")
    res["select/combined"] = ((res["select/policy_loss"] or 0.0)
                              + cfg["value_weight"] * (res["select/value_logloss"] or 0.0))
    if was:
        model.train()
        if aux is not None:
            aux.train()
    if dev.type == "mps":
        torch.mps.empty_cache()
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items()}


# ================================================================================================
# checkpoints
# ================================================================================================

def _atomic_torch_save(obj, path: Path, compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if compress:
        buf = io.BytesIO()
        torch.save(obj, buf)
        with gzip.open(tmp, "wb", compresslevel=1) as f:
            f.write(buf.getvalue())
    else:
        torch.save(obj, tmp)
    os.replace(tmp, path)


def save_weights(path: Path, model, vocab, arch: dict, info: dict | None = None, aux: AuxHeads | None = None) -> None:
    """MageZero's checkpoint format (what server.init and train.py --checkpoint read:
    model_state_dict + feature_vocab, gzip), plus `arch` (rebuilds non-default networks:
    load_any_checkpoint) and `supervised` (run info, plain Python types). Auxiliary heads go under
    aux_state_dict, outside model_state_dict."""
    a = full_arch(arch)
    obj = {"epoch": 0, "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
           "feature_vocab": vocab.state_dict(), "arch": a,
           "supervised": _jsonable({**(info or {}), "magezero_loadable": magezero_loadable(a)})}
    if aux is not None:
        obj["aux_state_dict"] = {k: v.detach().cpu() for k, v in aux.state_dict().items()}
        obj["aux_targets"] = list(aux.names)
    _atomic_torch_save(obj, Path(path), compress=True)


def _load_ck(path: Path) -> dict:
    from magezero.model import load_model
    path = Path(path)
    return load_model(str(path)) if path.suffix == ".gz" else torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_vocab(path: Path):
    """The feature vocab saved with a checkpoint."""
    from magezero.vocab import FeatureVocab
    ck = _load_ck(path)
    return FeatureVocab.from_state_dict(ck["feature_vocab"] if "feature_vocab" in ck else ck["vocab"])


def checkpoint_arch(path: Path) -> dict:
    ck = _load_ck(path)
    return full_arch(ck["arch"]) if ck.get("arch") else infer_arch(ck.get("model_state_dict") or ck["model"])


@torch.no_grad()
def set_reference(ref_model, tables: list, cfg: dict, dev, dtype) -> None:
    """t.ref_p on every soft table: the reference network's distribution over each row's legal
    options (its head for the row's decision type), on the full state, aligned with legal_idx."""
    ref_model.eval()
    for t in tables:
        if t.kind == "soft" and t.n:
            o = table_outputs(ref_model, t, dev, dtype, batch_rows=cfg["eval_batch_rows"])
            t.ref_p = np.concatenate([_softmax(s) for s in o["scores"]]).astype(np.float32)


def load_reference(cfg: dict, tables: list, vocab, dev, dtype, log=print) -> bool:
    """The KL reference (kl_ref, default init_checkpoint) on the soft tables, when there are both."""
    path = cfg["kl_ref"] or cfg["init_checkpoint"]
    if not path or not any(t.kind == "soft" for t in tables):
        return False
    ref, rvocab, _ = load_any_checkpoint(path, dev)
    if not np.array_equal(rvocab.ids, vocab.ids):
        raise ValueError(f"the reference network {path} has another feature vocab than the data's")
    t0 = time.monotonic()
    set_reference(ref, tables, cfg, dev, dtype)
    log(f"supervised: reference policy of {path} on the soft tables ({time.monotonic() - t0:.0f} s)")
    del ref
    return True


def param_part(name: str) -> str:
    """trunk, policy or value: the part of the network a parameter belongs to (`freeze`)."""
    head = name.split(".")[0]
    if head == "value_head":
        return "value"
    if head in ("player_priority_head", "opponent_priority_head", "target_head", "binary_head"):
        return "policy"
    return "trunk"


def load_any_checkpoint(path: Path, device="cpu"):
    """(model, vocab, meta) for any checkpoint this module or pretrain.py wrote (or latest.pt):
    rebuilds the network from its saved `arch` (default when absent), loads aux heads if present.
    meta: arch, aux (AuxHeads or None), info, magezero_loadable."""
    from magezero.vocab import FeatureVocab
    ck = _load_ck(path)
    sd = ck.get("model_state_dict") or ck["model"]
    arch = full_arch(ck["arch"]) if ck.get("arch") else infer_arch(sd)
    vocab = FeatureVocab.from_state_dict(ck["feature_vocab"] if "feature_vocab" in ck else ck["vocab"])
    model = build_model(arch, sd["embedding.weight"].shape[0])
    model.load_state_dict(sd)
    aux = None
    asd = ck.get("aux_state_dict") or ck.get("aux")
    if asd:
        names = ck.get("aux_targets") or ck.get("config", {}).get("aux_targets")
        aux = AuxHeads(emb_width(arch), names)
        aux.load_state_dict(asd)
        aux.to(device).eval()
    return model.to(device).eval(), vocab, {"arch": arch, "aux": aux, "info": ck.get("supervised"),
                                            "magezero_loadable": magezero_loadable(arch)}


# ================================================================================================
# training
# ================================================================================================

class _Log:
    def __init__(self, path: Path | None, echo=print):
        self.path, self.echo = path, echo

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        if self.echo is not None:
            self.echo(line)
        if self.path is not None:
            with open(self.path, "a") as f:
                f.write(line + "\n")


def _model_keys(cfg: dict) -> dict:
    return {k: cfg[k] for k in DATA_KEYS + ("arch", "seed", "token_dropout", "batch_tokens", "batch_attn",
                                             "max_batch_rows", "aux_targets", "value_per_game", "freeze")}


class Trainer:
    """One training run in `out`. See the module docstring for what it writes."""

    def __init__(self, cfg: dict, out: Path, *, resume: bool = False, data: Data | None = None, log=print):
        im._torch_env()
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = _Log(self.out / "train.log", log)
        self.cfg = cfg = resolve_config(cfg)
        if cfg["init_checkpoint"]:             # its shape; this config's dropouts
            cfg["arch"] = {**checkpoint_arch(cfg["init_checkpoint"]),
                           **{k: cfg["arch"][k] for k in ("dropout", "pool_dropout")}}
        latest = None
        if resume:
            if not (self.out / "latest.pt").exists():
                raise FileNotFoundError(f"--resume: no {self.out / 'latest.pt'}")
            latest = torch.load(self.out / "latest.pt", map_location="cpu", weights_only=False)
            was = _model_keys(resolve_config(latest["config"]))
            now = _model_keys(cfg)
            diff = sorted(k for k in now if json.dumps(now[k], sort_keys=True) != json.dumps(was[k], sort_keys=True))
            if diff:
                raise ValueError(f"--resume with changed data / model settings {diff}: start a new run instead")
        elif (self.out / "latest.pt").exists():
            raise FileExistsError(f"{self.out} already has a run (latest.pt): pass --resume or a new --out")
        self.dev = pick_device(cfg["device"])
        self.dtype = amp_dtype(cfg["amp"], self.dev)
        if self.dev.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(cfg["seed"])
        if latest is not None:
            from magezero.vocab import FeatureVocab
            vocab = FeatureVocab.from_state_dict(latest["vocab"])
            if data is not None and not np.array_equal(data.vocab.ids, vocab.ids):
                raise ValueError("--resume: the data's feature vocab differs from the run's")
        else:
            vocab = data.vocab if data is not None else None
        self.data = data or load_data(cfg, vocab=vocab, log=self.log)
        ensure_aux(self.data.train + self.data.val, cfg["aux_targets"], log=self.log)
        if cfg["act_weights"]:
            self.data = with_act_weights(self.data, cfg["act_weights"], log=self.log)
        self.vocab = self.data.vocab
        init = cfg["init_checkpoint"] if latest is None else None
        self.model = build_model(cfg["arch"], len(self.vocab), vocab=None if (latest or init) else self.vocab,
                                 emb_std=cfg["emb_init_std"]).to(self.dev)
        self.aux = AuxHeads(emb_width(cfg["arch"]), cfg["aux_targets"]).to(self.dev) if cfg["aux_targets"] else None
        if init:
            m0, v0, meta0 = load_any_checkpoint(init)
            if not np.array_equal(v0.ids, self.vocab.ids):
                raise ValueError(f"init_checkpoint {init}: the data's feature vocab differs from the checkpoint's")
            self.model.load_state_dict(m0.state_dict())
            if self.aux is not None and meta0["aux"] is not None and meta0["aux"].names == self.aux.names:
                self.aux.load_state_dict(meta0["aux"].state_dict())
            del m0
            self.log(f"supervised: starting from {init}")
        for name, p in self.model.named_parameters():
            p.requires_grad_(param_part(name) not in cfg["freeze"])
        params = [p for p in self.model.parameters() if p.requires_grad] + (list(self.aux.parameters()) if self.aux else [])
        self.opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.dtype == torch.float16 and self.dev.type == "cuda")
        self.sampler = Sampler(self.data, cfg)
        self.epoch_len = self.sampler.epoch_len
        self.shares = {t.name: float(s) for t, s in zip(self.data.train, self.sampler.share)}
        self.has_ref = load_reference(cfg, self.data.train + self.data.val, self.vocab, self.dev, self.dtype, self.log)
        self.c = {"step": 0, "seen": 0, "train_time_s": 0.0, "sessions": 0, "last_eval_t": 0.0, "last_ckpt_t": 0.0,
                  "last_eval_step": -1, "td_next_seen": None, "td_refreshes": 0, "stop": None}
        self.best = {"policy": [math.inf, None], "value": [math.inf, None], "combined": [math.inf, None, 0]}
        self.td = None
        self.sampler_state = self.sampler.state_dict()
        if latest is not None:
            self._restore(latest)
        self.c["sessions"] += 1
        (self.out / "config.json").write_text(json.dumps(_jsonable(cfg), indent=1))
        self.log(f"supervised: {'resumed at step ' + str(self.c['step']) if latest else 'new run'}; device {self.dev}, "
                 f"amp {self.dtype}, arch {full_arch(cfg['arch'])} (MageZero-loadable: {magezero_loadable(cfg['arch'])}), "
                 f"{sum(p.numel() for p in self.model.parameters()):,} parameters, {self.data.epoch_rows} training rows"
                 + (f" (epoch = {self.epoch_len} samples)" if self.epoch_len != self.data.epoch_rows else "")
                 + (f"; frozen: {cfg['freeze']}" if cfg["freeze"] else "")
                 + (f"; KL weight {cfg['kl_weight']} -> {cfg['kl_weight_end']}" if self.has_ref else ""))

    # ---------------------------------------------------------------------------------- state
    def _restore(self, ck: dict) -> None:
        self.model.load_state_dict(ck["model"])
        if self.aux is not None and ck.get("aux"):
            self.aux.load_state_dict(ck["aux"])
        self.opt.load_state_dict(ck["opt"])
        if ck.get("scaler"):
            self.scaler.load_state_dict(ck["scaler"])
        self.sampler.load_state_dict(ck["sampler"])
        self.sampler_state = ck["sampler"]
        self.c.update(ck["counters"])
        self.c["stop"] = None
        self.best = ck["best"]
        self.td = None if ck.get("td") is None else np.asarray(ck["td"], np.float32)
        torch.set_rng_state(ck["torch_rng"])
        if self.dev.type == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        ev = self.out / "evals.jsonl"
        if ev.exists():                                      # drop evaluations after the saved state
            keep = [ln for ln in ev.read_text().splitlines() if ln.strip() and json.loads(ln)["step"] <= self.c["step"]]
            ev.write_text("".join(x + "\n" for x in keep))

    def save_latest(self) -> None:
        ck = {"model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
              "aux": {k: v.detach().cpu() for k, v in self.aux.state_dict().items()} if self.aux else None,
              "opt": self.opt.state_dict(), "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
              "sampler": self.sampler_state, "counters": dict(self.c), "best": self.best,
              "td": None if self.td is None else torch.from_numpy(self.td),
              "torch_rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state_all() if self.dev.type == "cuda" else None,
              "vocab": self.vocab.state_dict(), "arch": self.cfg["arch"], "config": _jsonable(self.cfg),
              "aux_targets": self.cfg["aux_targets"]}
        _atomic_torch_save(ck, self.out / "latest.pt", compress=False)

    def _info(self, metrics: dict | None = None) -> dict:
        return {"step": self.c["step"], "seen": self.c["seen"], "epoch": round(self.c["seen"] / self.epoch_len, 4),
                "train_time_s": round(self.c["train_time_s"], 1), "metrics": metrics or {},
                "config": self.cfg, "data": self.data.info}

    def save(self, name: str, metrics: dict | None = None) -> Path:
        p = self.out / name
        save_weights(p, self.model, self.vocab, self.cfg["arch"], self._info(metrics), self.aux)
        return p

    # ---------------------------------------------------------------------------------- schedule
    def progress(self) -> float:
        """How far through the run (0 to 1): by max_steps, else max_epochs, else the time budget."""
        cfg, c = self.cfg, self.c
        if cfg["max_steps"]:
            p = c["step"] / cfg["max_steps"]
        elif cfg["max_epochs"]:
            p = c["seen"] / (cfg["max_epochs"] * self.epoch_len)
        elif cfg["time_budget_s"]:
            p = c["train_time_s"] / cfg["time_budget_s"]
        else:
            p = 0.0
        return min(1.0, max(0.0, p))

    def kl_at(self) -> float:
        a, b = self.cfg["kl_weight"], self.cfg["kl_weight_end"]
        return float(a if b is None else a + (b - a) * self.progress())

    def lr_at(self, step: int, seen: int) -> float:
        cfg = self.cfg
        lr = cfg["lr"] * min(1.0, (step + 1) / max(1, cfg["warmup_steps"]))
        if cfg["lr_schedule"] == "cosine":
            prog = step / cfg["max_steps"] if cfg["max_steps"] else seen / (cfg["max_epochs"] * self.epoch_len)
            prog = min(1.0, max(0.0, prog))
            lr *= cfg["lr_min_frac"] + (1 - cfg["lr_min_frac"]) * 0.5 * (1 + math.cos(math.pi * prog))
        return lr

    # ---------------------------------------------------------------------------------- TD targets
    @torch.no_grad()
    def refresh_td(self) -> None:
        d, cfg = self.data, self.cfg
        t0 = time.monotonic()
        v = np.zeros(d.epoch_rows, np.float32)
        was = self.model.training
        self.model.eval()
        for ti, t in enumerate(d.train):
            if not t.spec["value"]:
                continue
            for sel in im.length_batches(np.diff(t.ptr), cfg["eval_batch_rows"]):
                idx, off = im._batch(t.rows, t.ptr, sel, self.dev)
                with _autocast(self.dev, self.dtype):
                    emb = encode(self.model, idx, off)
                with _fp32(self.dev, self.dtype):
                    v[d.goff[ti] + sel] = torch.tanh(value_logit(self.model, emb.float())).cpu().numpy()
        if was:
            self.model.train()
        ok = d.g_value_ok
        td = np.full(d.epoch_rows, np.nan, np.float32)
        td[ok] = td_lambda_targets(d.g_game[ok], d.g_turn[ok], d.g_phase[ok], v[ok], d.g_z[ok], cfg["td_lambda"])
        self.td = td
        self.c["td_refreshes"] += 1
        self.log(f"supervised: TD(lambda={cfg['td_lambda']}) targets refreshed at step {self.c['step']} "
                 f"({time.monotonic() - t0:.0f} s; mean |target| {np.nanmean(np.abs(td)):.3f})")

    # ---------------------------------------------------------------------------------- one step
    def value_targets(self):
        """The value targets of every training row, on the device (the game result, or the TD
        targets once refreshed): indexing them there avoids a host-to-device copy per step."""
        src = self.td if self.td is not None else self.data.g_z
        if getattr(self, "_vt", None) is None or self._vt[0] is not src:
            self._vt = (src, torch.as_tensor(src, device=self.dev))
        return self._vt[1]

    def step(self, b: dict) -> dict:
        cfg, dev, m = self.cfg, self.dev, self.model
        t = self.data.train[b["ti"]]
        nb = self.dev.type == "cuda"
        idx, off = b["idx"].to(dev, non_blocking=nb), b["off"].to(dev, non_blocking=nb)
        w = b["w"].to(dev, non_blocking=nb)
        with _autocast(dev, self.dtype):
            emb = encode(m, idx, off, b["pad"])
        parts = {}
        with _fp32(dev, self.dtype):
            emb = emb.float()
            if t.kind == "soft":
                L, Pt = b["L"].to(dev, non_blocking=nb), b["Pt"].to(dev, non_blocking=nb)
                at = b["at"].to(dev, non_blocking=nb)
                ce, lsm = soft_ce(soft_logits(m, emb, at), L, Pt)
                hw = torch.where(at == 3, cfg["target_weight"], torch.where(at == 5, cfg["binary_weight"],
                                                                            cfg["policy_weight"])) * w
                nv = (w > 0).sum().clamp(min=1)
                lp = (ce * hw).sum() / nv
                parts["soft"] = lp
                kw = 1.0
                if "R" in b:              # KL(pi_ref || pi) = CE(pi_ref, pi) - H(pi_ref)
                    R = b["R"].to(dev, non_blocking=nb)
                    pos = R > 0
                    kl = (-torch.where(pos, R * lsm, torch.zeros_like(lsm)).sum(1)
                          + torch.where(pos, R * torch.log(R.clamp(min=1e-30)), torch.zeros_like(R)).sum(1))
                    lk = (kl * hw).sum() / nv
                    parts["kl"] = lk
                    klw = self.kl_at()
                    if klw:
                        lp = lp + klw * lk
            elif t.kind in POLICY_KINDS:
                L, S = b["L"].to(dev, non_blocking=nb), b["S"].to(dev, non_blocking=nb)
                head = m.target_head if t.kind == "target" else m.player_priority_head
                nll = policy_nll(head(emb), L, S)
                valid = (w > 0) & (S & L).any(1)
                lp = torch.where(valid, nll * w, torch.zeros_like(nll)).sum() / valid.sum().clamp(min=1)
                kw = cfg["target_weight"] if t.kind == "target" else cfg["policy_weight"]
                parts["target" if t.kind == "target" else "priority"] = lp
            else:
                y = b["y"].to(dev, non_blocking=nb)
                ce = F.cross_entropy(m.binary_head(emb), y, reduction="none")
                lp = (ce * w).sum() / (w > 0).sum().clamp(min=1)
                kw = cfg["binary_weight"]
                parts["binary"] = lp
            loss = kw * lp
            vx = value_logit(m, emb)
            tgt = self.value_targets()[b["gidx"].to(dev, non_blocking=nb)]
            vm = b["vmask"].to(dev, non_blocking=nb) & torch.isfinite(tgt)
            tgt = torch.where(vm, tgt, torch.zeros_like(tgt))
            per = value_bce_from_logit(vx, tgt) if cfg["value_loss"] == "bce" else value_mse(vx, tgt)
            denom = max(1.0, b["veligible"] * b["vfrac"])
            lv = torch.where(vm, per, torch.zeros_like(per)).sum() / denom
            parts["value"] = lv
            loss = loss + cfg["value_weight"] * lv
            if self.aux is not None:
                at = b["aux"].to(dev, non_blocking=nb)
                scale = torch.tensor([AUX_TARGETS[a] for a in self.aux.names], device=dev)
                ok = torch.isfinite(at)
                pred = self.aux(emb)
                la = torch.where(ok, F.smooth_l1_loss(pred, torch.nan_to_num(at) / scale, reduction="none"),
                                 torch.zeros_like(pred)).sum() / ok.sum().clamp(min=1)
                parts["aux"] = la
                loss = loss + cfg["aux_weight"] * la
        lr = self.lr_at(self.c["step"], self.c["seen"])
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if cfg["grad_clip"]:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_([p for g in self.opt.param_groups for p in g["params"]], cfg["grad_clip"])
        self.scaler.step(self.opt)
        self.scaler.update()
        self.c["step"] += 1
        self.c["seen"] = int(b["state"]["seen"])
        self.sampler_state = b["state"]
        return {"loss": loss, **parts, "lr": lr}

    # ---------------------------------------------------------------------------------- evaluation
    def do_eval(self, running: dict, t_since: float, n_since: int) -> dict:
        t0 = time.monotonic()
        self.c["last_eval_t"] = self.c["train_time_s"]       # the eval schedule restarts from here, resumed or not
        ev = evaluate(self.model, self.data.val, self.cfg, self.dev, self.dtype, aux=self.aux, shares=self.shares)
        rec = {"step": self.c["step"], "seen": self.c["seen"], "epoch": round(self.c["seen"] / self.epoch_len, 4),
               "train_time_s": round(self.c["train_time_s"], 1), "time": round(time.time(), 1),
               "lr": self.lr_at(self.c["step"], self.c["seen"]),
               "samples_per_s": round(n_since / t_since, 1) if t_since > 0 else None}
        if self.has_ref:
            rec["kl_weight"] = round(self.kl_at(), 5)
        rec.update({f"train/{k}": round(float(v[0]) / max(1, v[1]), 5) for k, v in running.items()})
        rec.update(ev)
        flags = []
        pl, vl, cb = ev["select/policy_loss"], ev["select/value_logloss"], ev["select/combined"]
        if pl is not None and pl < self.best["policy"][0]:
            self.best["policy"] = [pl, self.c["step"]]
            self.save("best_policy.pt.gz", ev)
            flags.append("policy")
        if vl is not None and vl < self.best["value"][0]:
            self.best["value"] = [vl, self.c["step"]]
            self.save("best_value.pt.gz", ev)
            flags.append("value")
        if cb is not None and cb < self.best["combined"][0] - self.cfg["min_delta"]:
            self.best["combined"] = [cb, self.c["step"], self.c["seen"]]
            self.save("best.pt.gz", ev)
            flags.append("combined")
        rec["best"] = flags
        rec["eval_s"] = round(time.monotonic() - t0, 1)
        with open(self.out / "evals.jsonl", "a") as f:
            f.write(json.dumps(_jsonable(rec)) + "\n")
        self.c["last_eval_step"] = self.c["step"]
        g = lambda k: ev.get(k)                                # noqa: E731
        self.log(f"supervised: step {self.c['step']} epoch {rec['epoch']:.3f} | policy top1 {g('policy/top1')} "
                 f"non-Pass {g('policy/top1_nonpass')} set NLL {g('policy/set_nll')} | binary acc {g('binary/acc')} "
                 f"AUC {g('binary/auc')} | value logloss {g('value/logloss')} AUC {g('value/auc')} | "
                 f"combined {cb} {('best: ' + ','.join(flags)) if flags else ''} ({rec['eval_s']} s)")
        for t in self.data.val:
            if t.kind == "soft" and t.n:
                self.log(f"supervised:   {t.name}: CE {g(t.name + '/ce')} top-1 with the search "
                         f"{g(t.name + '/top1_search')} KL to the reference {g(t.name + '/kl_ref')} Pass at the top "
                         f"{g(t.name + '/pass_top1')} (search {g(t.name + '/pass_search')}) | value AUC "
                         f"{g(t.name + '/value_auc')} | human rows: Pass at the top {g('policy/pass_top1')} "
                         f"(humans {g('policy/pass_human')}) entropy {g('policy/entropy')}")
        return rec

    # ---------------------------------------------------------------------------------- loop
    def run(self) -> dict:
        cfg, c = self.cfg, self.c
        stop = {"flag": None}
        old = {}

        def on_signal(signum, _frame):
            stop["flag"] = f"signal {signum}"
        if threading.current_thread() is threading.main_thread():
            for s in (signal.SIGTERM, signal.SIGINT):
                old[s] = signal.signal(s, on_signal)
        pf = Prefetcher(self.sampler, int(cfg["prefetch"]), pin=self.dev.type == "cuda")
        running: dict[str, list] = {}
        sess0 = time.monotonic()
        paused = 0.0
        base_time = c["train_time_s"]
        last_latest = time.monotonic()
        since_t, since_n = c["train_time_s"], c["seen"]
        epoch_rows = self.epoch_len
        if cfg["value_target"] == "td" and c["td_next_seen"] is None:
            c["td_next_seen"] = int(cfg["td_start_epochs"] * epoch_rows)
        try:
            if cfg["eval_at_start"] and c["step"] == 0 and c["last_eval_step"] < 0:
                p0 = time.monotonic()
                self.do_eval({}, 0, 0)
                self.save_latest()
                paused += time.monotonic() - p0
            while True:
                now = time.monotonic()
                c["train_time_s"] = base_time + (now - sess0) - paused
                reason = stop["flag"]
                if reason is None and cfg["time_budget_s"] is not None and c["train_time_s"] >= cfg["time_budget_s"]:
                    reason = "time budget"
                if reason is None and cfg["max_steps"] is not None and c["step"] >= cfg["max_steps"]:
                    reason = "max steps"
                if reason is None and cfg["max_epochs"] is not None and c["seen"] >= cfg["max_epochs"] * epoch_rows:
                    reason = "max epochs"
                if reason is None and cfg["patience_epochs"] and self.best["combined"][1] is not None and \
                        c["seen"] - self.best["combined"][2] >= cfg["patience_epochs"] * epoch_rows:
                    reason = "plateau"
                if reason:
                    c["stop"] = reason
                    break
                if c["td_next_seen"] is not None and c["seen"] >= c["td_next_seen"]:
                    self.refresh_td()
                    c["td_next_seen"] = c["seen"] + max(1, int(cfg["td_refresh_epochs"] * epoch_rows))
                b = pf.get()
                out = self.step(b)
                for k, v in out.items():          # summed on the device: no GPU sync per step
                    if k != "lr":
                        r = running.setdefault(k, [0.0, 0])
                        r[0] = r[0] + v.detach()
                        r[1] += 1
                if c["step"] % 200 == 0 and not bool(torch.isfinite(running["loss"][0])):
                    c["stop"] = "non-finite loss"
                    break
                tt = base_time + (time.monotonic() - sess0) - paused
                due = (cfg["eval_every_steps"] and c["step"] % cfg["eval_every_steps"] == 0) or \
                      (cfg["eval_every_s"] and tt - c["last_eval_t"] >= cfg["eval_every_s"])
                if due:
                    p0 = time.monotonic()
                    c["train_time_s"] = tt
                    self.do_eval(running, tt - since_t, c["seen"] - since_n)
                    running, since_t, since_n = {}, tt, c["seen"]
                    self.save_latest()
                    last_latest = time.monotonic()
                    paused += time.monotonic() - p0
                if cfg["ckpt_every_s"] and tt - c["last_ckpt_t"] >= cfg["ckpt_every_s"]:
                    p0 = time.monotonic()
                    c["last_ckpt_t"] = tt
                    self.save(f"ckpt/step{c['step']:08d}.pt.gz")
                    paused += time.monotonic() - p0
                if cfg["latest_every_s"] and time.monotonic() - last_latest >= cfg["latest_every_s"]:
                    p0 = time.monotonic()
                    c["train_time_s"] = tt
                    self.save_latest()
                    last_latest = time.monotonic()
                    paused += time.monotonic() - p0
        except KeyboardInterrupt:
            c["stop"] = "interrupted"
        finally:
            pf.close()
            for s, h in old.items():
                signal.signal(s, h)
        last = None
        if c["stop"] == "interrupted" or str(c["stop"]).startswith("signal"):
            self.save_latest()        # a pod being stopped: keep the state, skip the evaluation
            self.log(f"supervised: stopped ({c['stop']}) at step {c['step']}; latest.pt saved, resume with --resume")
            return self.summary()
        if c["last_eval_step"] != c["step"] and c["stop"] != "non-finite loss":
            last = self.do_eval(running, c["train_time_s"] - since_t, c["seen"] - since_n)
        self.save_latest()
        self.save("final.pt.gz", last)
        summary = self.summary()
        (self.out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=1))
        self.log(f"supervised: stopped ({c['stop']}) at step {c['step']}, {c['seen']} samples, "
                 f"{c['seen'] / self.epoch_len:.2f} epochs, {c['train_time_s']:.0f} s of training")
        return summary

    def summary(self) -> dict:
        evs = read_evals(self.out / "evals.jsonl")
        at = lambda step: next((e for e in evs if e["step"] == step), None)   # noqa: E731
        tr = [e["samples_per_s"] for e in evs if e.get("samples_per_s")]
        return {"out": str(self.out), "stop": self.c["stop"], "step": self.c["step"], "seen": self.c["seen"],
                "epochs": round(self.c["seen"] / self.epoch_len, 3), "train_time_s": round(self.c["train_time_s"], 1),
                "samples_per_s": round(float(np.median(tr)), 1) if tr else None,
                "td_refreshes": self.c["td_refreshes"], "sessions": self.c["sessions"], "best": self.best, "best_policy_eval": at(self.best["policy"][1]),
                "best_value_eval": at(self.best["value"][1]), "best_combined_eval": at(self.best["combined"][1]),
                "last_eval": evs[-1] if evs else None, "magezero_loadable": magezero_loadable(self.cfg["arch"]),
                "data": self.data.info, "config": self.cfg}


def read_evals(path: Path) -> list[dict]:
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def train(cfg: dict, out: Path, *, resume: bool = False, data: Data | None = None, log=print) -> dict:
    """Train (or resume) one run in `out`; returns summary.json's contents."""
    return Trainer(cfg, out, resume=resume, data=data, log=log).run()


# ================================================================================================
# inference speed
# ================================================================================================

def _bench_inputs(B: int, lengths: np.ndarray, rows: int, rng) -> tuple:
    ln = lengths[rng.integers(0, len(lengths), B)]
    idx = torch.as_tensor(rng.integers(0, rows, int(ln.sum())), dtype=torch.long)
    off = torch.as_tensor(np.r_[0, np.cumsum(ln)[:-1]], dtype=torch.long)
    return idx, off


@torch.no_grad()
def bench_model(model, dev, *, batches=(1, 8, 32, 128), lengths: np.ndarray | None = None, seconds: float = 2.0,
                dtypes: list | None = None, paths: list | None = None, seed: int = 0, log=None) -> list[dict]:
    """Evaluations per second at each batch size and precision on `dev`. Path 'forward' is the
    network's own forward (for the default network, MageZero's NetTransformer.forward: what its
    server runs, with a per-row padding loop, padded to the batch's longest state); 'trunk' pads in
    one vectorised step to imitation's length buckets + the five heads. Outputs are copied to the
    CPU each batch, as the server does. States are drawn at random from `lengths` (a real table's
    mapped lengths, ideally): a batch costs what its longest state costs, so large batches of
    mixed lengths pay for padding (each result records its mean padded length)."""
    model = model.to(dev).eval()
    rows = model.embedding.num_embeddings
    lengths = np.asarray(lengths if lengths is not None else
                         np.clip(np.random.default_rng(seed).normal(760, 150, 4096), 64, 3000).astype(np.int64))
    if dtypes is None:
        dtypes = ["fp32"] + (["fp16", "bf16"] if dev.type == "cuda" and torch.cuda.is_bf16_supported()
                             else ["fp16"] if dev.type in ("cuda", "mps") else [])
    if paths is None:
        paths = ["forward"] if isinstance(model, BagMLPNet) else ["forward", "trunk"]
    dt = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
    rng = np.random.default_rng(seed)
    res = []
    for dname in dtypes:
        for path in paths:
            for B in batches:
                try:
                    inputs = [_bench_inputs(B, lengths, rows, rng) for _ in range(4)]

                    def run(i):
                        idx, off = inputs[i % len(inputs)]
                        idx, off = idx.to(dev), off.to(dev)
                        with _autocast(dev, dt[dname]):
                            if path == "forward":
                                outs = model(idx, off)
                            else:
                                emb = encode(model, idx, off)
                                outs = (model.player_priority_head(emb), model.opponent_priority_head(emb),
                                        model.target_head(emb), model.binary_head(emb), model.value_head(emb))
                        return [o.float().cpu() for o in outs]
                    for i in range(3):
                        run(i)
                    _sync(dev)
                    n, t0 = 0, time.perf_counter()
                    while time.perf_counter() - t0 < seconds or n < 3:
                        run(n)
                        n += 1
                    el = time.perf_counter() - t0
                    longest = [int(torch.diff(torch.cat([o, torch.tensor([len(i)])])).max()) for i, o in inputs]
                    pad = [x if path == "forward" else im.bucket_len(x) for x in longest]
                    r = {"dtype": dname, "path": path, "batch": B, "evals_per_s": round(B * n / el, 1),
                         "ms_per_batch": round(1000 * el / n, 2), "padded_len": round(float(np.mean(pad)))}
                except Exception as e:                     # noqa: BLE001 - an unsupported dtype is a result
                    r = {"dtype": dname, "path": path, "batch": B, "error": f"{type(e).__name__}: {e}"[:200]}
                res.append(r)
                if log:
                    log(f"bench: {r}")
    return res


def bench_report(model, dev, *, arch: dict, batches, lengths=None, seconds=2.0, dtypes=None, log=print) -> dict:
    name = torch.cuda.get_device_name(0) if dev.type == "cuda" else dev.type
    res = bench_model(model, dev, batches=batches, lengths=lengths, seconds=seconds, dtypes=dtypes, log=log)
    return {"device": str(dev), "device_name": name, "arch": full_arch(arch), "magezero_loadable": magezero_loadable(arch),
            "vocab_rows": model.embedding.num_embeddings,
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "mean_tokens": None if lengths is None else round(float(np.mean(lengths)), 1), "results": res}


# ================================================================================================
# sweep
# ================================================================================================

SWEEP_METRICS = [  # (key in the run summary, panel title, higher is better, format)
    ("policy_top1_nonpass", "Priority: non-Pass top-1 in the human's plays", True, "{:.1%}"),
    ("policy_set_nll", "Priority: set NLL", False, "{:.3f}"),
    ("binary_auc", "Attack yes/no: AUC", True, "{:.3f}"),
    ("value_logloss", "Value: log-loss of P(win)", False, "{:.4f}"),
    ("value_auc", "Value: AUC against results", True, "{:.3f}"),
    ("infer_eps", "Inference: evaluations / s (batch 32)", True, "{:,.0f}"),
]


def _run_row(name: str, spec: dict, summ: dict, speed: dict | None) -> dict:
    bp = summ.get("best_policy_eval") or {}
    bv = summ.get("best_value_eval") or {}
    row = {"name": name, "role": spec.get("role", ""), "seed": summ["config"]["seed"],
           "overrides": json.dumps({k: v for k, v in spec.items() if k not in ("name", "role")}, sort_keys=True),
           "magezero_loadable": summ["magezero_loadable"], "steps": summ["step"], "epochs": summ["epochs"],
           "train_time_s": summ["train_time_s"], "train_samples_per_s": summ["samples_per_s"], "stop": summ["stop"],
           "policy_top1_nonpass": bp.get("policy/top1_nonpass"), "policy_top1": bp.get("policy/top1"),
           "policy_set_nll": bp.get("policy/set_nll"), "binary_auc": bp.get("binary/auc"),
           "binary_acc": bp.get("binary/acc"), "value_logloss": bv.get("value/logloss"),
           "value_auc": bv.get("value/auc"), "value_mse": bv.get("value/mse"),
           "best_policy_step": summ["best"]["policy"][1], "best_value_step": summ["best"]["value"][1]}
    for k, v in bv.items():
        if k.startswith("value/auc_t"):
            row[k.replace("/", "_")] = v
    soft = [t["name"] for t in summ["config"]["tables"] if t["kind"] == "soft"]
    if soft:                       # stage 6: collapse measures at the run's best combined evaluation
        be = summ.get("best_combined_eval") or summ.get("last_eval") or {}
        for k in ("policy/pass_top1", "policy/pass_human", "policy/entropy", "policy/top1", "value/auc"):
            row[f"best_{k.replace('/', '_')}"] = be.get(k)
        for name in soft:
            for k in ("ce", "top1_search", "kl_ref", "entropy", "pass_top1", "pass_search", "value_auc",
                      "value_logloss", "value_ece"):
                row[f"{name}_{k}"] = be.get(f"{name}/{k}")
    for r in (speed or {}).get("results", []):
        if "evals_per_s" in r and r["path"] == "forward":
            row[f"infer_eps_{r['dtype']}_b{r['batch']}"] = r["evals_per_s"]
    fast = [r for r in (speed or {}).get("results", []) if r.get("batch") == 32 and "evals_per_s" in r
            and r["path"] == "forward"]
    row["infer_eps"] = max((r["evals_per_s"] for r in fast), default=None)
    row["infer_device"] = (speed or {}).get("device_name")
    return row


def _load_sweep_spec(spec_path: Path, overrides: dict | None) -> tuple[dict, dict, list]:
    spec = load_config_file(spec_path)
    bc = spec.get("base_config")
    if bc and not Path(bc).exists() and (Path(spec_path).parent / bc).exists():
        bc = Path(spec_path).parent / bc
    base = resolve_config(load_config_file(bc) if bc else {}, spec.get("base") or {}, overrides or {})
    runs = spec["runs"]
    names = [r["name"] for r in runs]
    if len(set(names)) != len(names):
        raise ValueError("sweep run names must be unique")
    return spec, base, runs


def run_sweep(spec_path: Path, out: Path, *, overrides: dict | None = None, only: list | None = None,
              bench_seconds: float = 2.0, follow: bool = False, follow_idle_s: float = 0.0, log=print) -> list[dict]:
    """Every config of a sweep spec on the same data (one load: the same training-game subset,
    feature vocab and validation rows), each with the same budget; a run that already finished is
    read back, one with a latest.pt is resumed. Writes sweep.json, sweep.csv and sweep-{light,dark}.png.

    spec: {base_config?: a config file, base: {config keys shared by every run}, runs: [{name, role?,
    <config overrides>}]}; base_config is relative to the spec's directory or the working directory.
    role 'default' marks the default config's seeds (the plot's noise band).

    follow: re-read the spec before every run and take the first run not yet done, in the spec's
    order, so runs can be added or reordered while the sweep goes (the data stay loaded; the data
    settings may not change). With none left it waits up to follow_idle_s for more, and stops early
    once <out>/STOP exists."""
    spec, base, runs = _load_sweep_spec(spec_path, overrides)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lg = _Log(out / "sweep.log", log)
    holder: dict = {}
    rows: dict = {}

    def one(r: dict, base: dict) -> dict:
        over = {k: v for k, v in r.items() if k not in ("name", "role")}
        cfg = resolve_config(base, over)
        changed = [k for k in DATA_KEYS if json.dumps(cfg[k], sort_keys=True) != json.dumps(base[k], sort_keys=True)]
        if changed:
            raise ValueError(f"sweep run {r['name']} changes data settings {changed}: put them in base")
        rd = out / "runs" / r["name"]
        if (rd / "summary.json").exists() and (rd / "speed.json").exists():
            lg(f"sweep: {r['name']} already done")
            summ = json.loads((rd / "summary.json").read_text())
            speed = json.loads((rd / "speed.json").read_text())
        else:
            if "data" not in holder:
                holder["data"] = load_data(base, log=lg)
            data = holder["data"]
            if not (rd / "summary.json").exists():
                lg(f"sweep: run {r['name']}: {over}")
                summ = train(cfg, rd, resume=(rd / "latest.pt").exists(), data=data, log=log)
            else:
                summ = json.loads((rd / "summary.json").read_text())
            ck = rd / "best.pt.gz" if (rd / "best.pt.gz").exists() else rd / "final.pt.gz"
            model, _, meta = load_any_checkpoint(ck)
            dev = pick_device(cfg["device"])
            lens = np.diff(data.val[0].ptr) if data.val and data.val[0].n else None
            speed = bench_report(model, dev, arch=meta["arch"], batches=(1, 8, 32, 128), lengths=lens,
                                 seconds=bench_seconds, log=None)
            (rd / "speed.json").write_text(json.dumps(speed, indent=1))
            del model
        return _run_row(r["name"], r, summ, speed)

    def write(runs_now: list) -> None:
        order = [r["name"] for r in runs_now]
        _write_sweep(out, [rows[n] for n in order if n in rows] + [v for k, v in rows.items() if k not in order], base)

    if not follow:
        for r in runs:
            if only and r["name"] not in only:
                continue
            rows[r["name"]] = one(r, base)
            write(runs)
    else:
        idle_since = None
        while True:
            spec, base2, runs = _load_sweep_spec(spec_path, overrides)
            changed = [k for k in DATA_KEYS if json.dumps(base2[k], sort_keys=True) != json.dumps(base[k], sort_keys=True)]
            if changed:
                raise ValueError(f"the spec's data settings changed ({changed}): start a new sweep")
            base = base2
            todo = [r for r in runs if r["name"] not in rows and (not only or r["name"] in only)]
            if (out / "STOP").exists():
                lg("sweep: STOP file found; stopping")
                break
            if not todo:
                idle_since = idle_since or time.monotonic()
                if time.monotonic() - idle_since >= follow_idle_s:
                    break
                time.sleep(30)
                continue
            idle_since = None
            r = todo[0]
            rows[r["name"]] = one(r, base)
            write(runs)
    paths = plot_sweep([rows[r["name"]] for r in runs if r["name"] in rows], out / "sweep",
                       title=spec.get("title") or "Supervised sweep")
    lg(f"sweep: {len(rows)} runs -> {out / 'sweep.json'}, {out / 'sweep.csv'}, " + ", ".join(map(str, paths)))
    return list(rows.values())


def _write_sweep(out: Path, rows: list, base: dict) -> None:
    (out / "sweep.json").write_text(json.dumps(_jsonable({"base": base, "runs": rows}), indent=1))
    keys = []
    for r in rows:
        keys.extend(k for k in r if k not in keys)
    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


THEMES = {   # tools/search_bench/analyze.py's THEMES (docs figures), with this plot's two series
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", band="#e9e8e2", default="#2a78d6", variant="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", band="#34342f", default="#3987e5", variant="#d95926"),
}


def plot_sweep(rows: list, out_prefix: Path, title: str = "Supervised sweep") -> list[Path]:
    """One panel per key measure, configs on the vertical axis (in spec order), the default
    config's seeds as a shaded noise band (their range) with its mean as a line. Light and dark."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    metrics = [m for m in SWEEP_METRICS if any(r.get(m[0]) is not None for r in rows)]
    if not rows or not metrics:
        return []
    names = [r["name"] for r in rows]
    y = np.arange(len(rows))[::-1]
    ncol = min(3, len(metrics))
    nrow = math.ceil(len(metrics) / ncol)
    paths = []
    for mode, t in THEMES.items():
        plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10,
                             "xtick.color": t["muted"], "ytick.color": t["muted"],
                             "xtick.labelcolor": t["ink2"], "ytick.labelcolor": t["ink2"]})
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 0.42 * len(rows) * nrow + 1.6 * nrow + 1.2),
                                 dpi=150, squeeze=False, sharey=True)
        fig.patch.set_facecolor(t["surface"])
        for ax in axes.flat[len(metrics):]:
            ax.set_visible(False)
        for ax, (key, ttl, higher, fmt) in zip(axes.flat, metrics):
            ax.set_facecolor(t["surface"])
            ax.grid(True, axis="x", color=t["grid"], linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(t["axis"])
            ax.tick_params(axis="y", length=0)
            vals = [r.get(key) for r in rows]
            dflt = [v for r, v in zip(rows, vals) if r.get("role") == "default" and v is not None]
            if len(dflt) >= 2:
                ax.axvspan(min(dflt), max(dflt), color=t["band"], zorder=0, linewidth=0)
            if dflt:
                ax.axvline(float(np.mean(dflt)), color=t["default"], linewidth=1, alpha=0.6, zorder=1)
            for yi, r, v in zip(y, rows, vals):
                if v is None:
                    continue
                c = t["default"] if r.get("role") == "default" else t["variant"]
                ax.plot([v], [yi], marker="o", markersize=8, color=c, markeredgecolor=t["surface"],
                        markeredgewidth=1.5, linestyle="none", zorder=3)
                ax.annotate(fmt.format(v), (v, yi), xytext=(7, 0), textcoords="offset points", va="center",
                            fontsize=8, color=t["ink2"])
            fin = [v for v in vals if v is not None]
            if fin:
                lo, hi = min(fin + dflt), max(fin + dflt)
                pad = (hi - lo) * 0.25 or abs(hi) * 0.05 or 1
                ax.set_xlim(lo - pad * 0.4, hi + pad)
            dev_name = next((r.get("infer_device") for r in rows if r.get("infer_device")), None)
            if key == "infer_eps" and dev_name:
                ttl = ttl.replace("(batch 32)", f"(batch 32, {dev_name})")
            ax.set_title(ttl + ("  (higher is better)" if higher else "  (lower is better)"), color=t["ink"],
                         fontsize=10, loc="left", pad=8)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _, f=fmt: f.format(v).replace(".0%", "%")))
        for ax in axes[:, 0]:
            ax.set_yticks(y)
            ax.set_yticklabels(names)
            ax.set_ylim(-0.7, len(rows) - 0.3)
        handles = [Line2D([], [], color=t["default"], marker="o", linestyle="none", markersize=8,
                          label="default config (each seed)"),
                   Line2D([], [], color=t["variant"], marker="o", linestyle="none", markersize=8,
                          label="one setting changed"),
                   Patch(color=t["band"], label="range of the default's seeds (noise band)")]
        leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
        for tx in leg.get_texts():
            tx.set_color(t["ink2"])
        fig.text(0.02, 0.985, title, color=t["ink"], fontsize=13, fontweight="bold", ha="left", va="top")
        fig.text(0.02, 0.985 - 0.35 / fig.get_figheight(), "Policy measures at each run's best-policy evaluation, "
                 "value measures at its best-value evaluation; same games, budget and validation rows for every run.",
                 color=t["ink2"], fontsize=9, ha="left", va="top")
        fig.tight_layout(rect=(0, 0.6 / fig.get_figheight(), 1, 1 - 0.75 / fig.get_figheight()))
        p = Path(f"{out_prefix}-{mode}.png")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, facecolor=t["surface"])
        plt.close(fig)
        paths.append(p)
    return paths


# ================================================================================================
# CLI
# ================================================================================================

def _cfg_from_args(args, saved: dict | None = None) -> dict:
    layers = [saved or {}, load_config_file(getattr(args, "config", None))]
    over = parse_overrides(getattr(args, "set", None))
    if getattr(args, "tables_dir", None):
        over["tables_dir"] = str(args.tables_dir)
    if getattr(args, "budget", None) is not None:
        over["time_budget_s"] = args.budget
    if getattr(args, "device", None):
        over["device"] = args.device
    return resolve_config(*layers, over)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m draftzero.gameplay.supervised", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--config", type=Path, default=None, help="YAML / JSON config (merged over DEFAULTS)")
        p.add_argument("--tables-dir", type=Path, default=None)
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key")
        p.add_argument("--device", default=None)

    t = sub.add_parser("train", help="train (or --resume) one run")
    common(t)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--budget", type=float, default=None, help="total training seconds (time_budget_s)")
    t.add_argument("--resume", action="store_true", help="continue the run in --out from latest.pt")
    s = sub.add_parser("sweep", help="every config of a sweep spec on the same data")
    s.add_argument("--spec", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--tables-dir", type=Path, default=None)
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a base key")
    s.add_argument("--only", default=None, help="comma-separated run names")
    s.add_argument("--bench-seconds", type=float, default=2.0)
    s.add_argument("--plot-only", action="store_true", help="redraw the plot from sweep.json")
    s.add_argument("--follow", action="store_true", help="re-read the spec before each run (add runs as it goes)")
    s.add_argument("--follow-idle", type=float, default=0.0, help="with --follow: seconds to wait for new runs")
    b = sub.add_parser("bench-speed", help="evaluations / s of a checkpoint or config")
    b.add_argument("--checkpoint", type=Path, default=None)
    b.add_argument("--config", type=Path, default=None)
    b.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    b.add_argument("--vocab-rows", type=int, default=14496, help="embedding rows for --config (#2b: 14,496)")
    b.add_argument("--table", type=Path, default=None, help="an HDF5 table: sample its state lengths")
    b.add_argument("--tokens", type=float, default=760, help="mean state length without --table")
    b.add_argument("--batches", default="1,8,32,128")
    b.add_argument("--dtypes", default=None, help="fp32,fp16,bf16 (default: what the device has)")
    b.add_argument("--seconds", type=float, default=3.0)
    b.add_argument("--device", default=None)
    b.add_argument("--json", type=Path, default=None)
    e = sub.add_parser("evaluate", help="the validation measures of a checkpoint on a split")
    common(e)
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--split", default="val")
    e.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    im._torch_env()

    if args.cmd == "train":
        saved = None
        if args.resume and (args.out / "config.json").exists():
            saved = json.loads((args.out / "config.json").read_text())
        cfg = _cfg_from_args(args, saved)
        summ = train(cfg, args.out, resume=args.resume)
        print(json.dumps({k: summ[k] for k in ("stop", "step", "seen", "epochs", "train_time_s", "best")}, indent=1))
    elif args.cmd == "sweep":
        if args.plot_only:
            d = json.loads((args.out / "sweep.json").read_text())
            print(plot_sweep(d["runs"], args.out / "sweep"))
            return 0
        over = parse_overrides(args.set)
        if args.tables_dir:
            over["tables_dir"] = str(args.tables_dir)
        run_sweep(args.spec, args.out, overrides=over, only=args.only.split(",") if args.only else None,
                  follow=args.follow, follow_idle_s=args.follow_idle,
                  bench_seconds=args.bench_seconds)
    elif args.cmd == "bench-speed":
        dev = pick_device(args.device or "auto")
        if args.checkpoint:
            model, vocab, meta = load_any_checkpoint(args.checkpoint)
            arch = meta["arch"]
        else:
            cfg = resolve_config(load_config_file(args.config), parse_overrides(args.set))
            arch = cfg["arch"]
            torch.manual_seed(0)
            model = build_model(arch, args.vocab_rows)
        lengths = None
        if args.table:                # the state lengths of the table's first 2,000 rows (mapped, given a vocab)
            import h5py
            with h5py.File(args.table, "r") as f:
                off = f["offsets"][:2001].astype(np.int64)
                lengths = np.diff(off)
                if args.checkpoint:
                    _, p = vocab.map_csr(f["indices"][off[0]:off[-1]].astype(np.int64), off - off[0])
                    lengths = np.maximum(np.diff(p), 1)
        elif args.tokens:
            lengths = np.clip(np.random.default_rng(0).normal(args.tokens, args.tokens * 0.2, 4096), 16,
                              3000).astype(np.int64)
        rep = bench_report(model, dev, arch=arch, batches=[int(x) for x in args.batches.split(",")], lengths=lengths,
                           seconds=args.seconds, dtypes=args.dtypes.split(",") if args.dtypes else None, log=print)
        if args.json:
            args.json.write_text(json.dumps(rep, indent=1))
        print(f"{rep['device_name']}: {rep['parameters']:,} parameters, MageZero-loadable {rep['magezero_loadable']}")
        for r in rep["results"]:
            print(f"  {r['dtype']:5s} {r['path']:8s} batch {r['batch']:4d}: " +
                  (f"{r['evals_per_s']:10.1f} evals/s  {r['ms_per_batch']:8.2f} ms/batch  padded to {r['padded_len']}"
                   if "evals_per_s" in r else r["error"]))
    else:
        model, vocab, meta = load_any_checkpoint(args.checkpoint)
        saved = (meta["info"] or {}).get("config")
        cfg = _cfg_from_args(args, {k: v for k, v in (saved or {}).items() if k in DEFAULTS})
        dev = pick_device(cfg["device"])
        data = load_data(cfg, vocab=vocab, splits=(args.split,))
        ref = cfg["kl_ref"] or cfg["init_checkpoint"]
        if ref and Path(ref).exists():
            try:
                load_reference(cfg, data.val, vocab, dev, amp_dtype(cfg["amp"], dev))
            except ValueError as e:
                print(f"supervised: no KL to the reference: {e}", file=sys.stderr)
        res = evaluate(model.to(dev), data.val, cfg, dev, amp_dtype(cfg["amp"], dev), aux=meta["aux"])
        res = {"checkpoint": str(args.checkpoint), "split": args.split, "data": data.info, **res}
        if args.json:
            args.json.write_text(json.dumps(_jsonable(res), indent=1))
        print(json.dumps(_jsonable(res), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
