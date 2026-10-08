"""The self-play run's settings: one YAML file read by every role (docs/021 §3).

    run: gnn-selfplay-1
    network: gnn                       # gnn (MageZero's graph network) or mlp (experiment #4's flat MLP)
    start: runs/gnn/full_r1/best_policy.pt.gz
    play:     {simulations: 1000, ...} # the search, exploration and the opponent mix
    worker:   {workers: auto, ...}     # how a machine runs its games
    trainer:  {reuse: 4, ...}          # when and how the trainer trains and publishes
    eval:     {every_versions: 2, ...} # evaluation jobs and their sequential test

Unknown keys are errors, so a typo can't silently fall back to a default. Paths are relative to the repo root.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]

DEFAULTS: dict[str, Any] = {
    "run": "selfplay",
    "network": "gnn",                       # gnn | mlp
    "start": None,                          # the starting checkpoint (version 0): a path or hf://
    "deck_root": "data/deckgen/FDN_PremierDraft_wr60/top_player_FDN_decks",
    "play": {
        "method": "pimc",                   # pimc | ismcts (docs/019 §4.6: PIMC)
        "simulations": 1000,
        "root_noise": 0.25,                 # Dirichlet noise at the root (self-play only); 0 turns it off
        "root_noise_alpha": 0.3,
        "sample_turns": 3,                  # each seat plays by visit counts for its first N turns
        "max_turns": 50,
        "pool": "data/pools/train.txt",     # self-play decks
        "past_share": 0.2,                  # share of chunks against the starting network (Will: not only itself)
        "closed_decklists": True,           # guess the opponent's deck (the belief service)
        "search_timeout": 1800.0,
        "game_timeout": 43200.0,
    },
    "worker": {
        "workers": "auto",                  # game workers (JVMs): auto = from the machine's cores and memory
        "heap": "3g",
        "servers": "auto",                  # network server processes: auto = 2 per 8 cores (min 1, max 4)
        "device": "auto",                   # where the network runs: auto (its GPU if any), cpu, cuda
        "pairs_per_chunk": "auto",          # deck pairs (2 games each) a chunk: auto = enough for every worker
        "poll_s": 120,                      # how often an idle worker re-reads control.json
        "base_port": 50200,                 # network servers take base_port + 10 x slot; belief at base_port - 30
        "price_per_hour": 0.0,              # for the controller's spend estimate (heartbeats)
    },
    "trainer": {
        "min_new_games": 200,               # publish a version once this many new games are in...
        "publish_every_s": 3600,            # ...and this long has passed since the last (or 2x min_new_games arrived)
        "window_games": 5000,               # the newest games trained on
        "reuse": 4.0,                       # training samples per new position (KataGo's cap: 4)
        "batch_rows": 64,
        "lr": 3e-5,
        "warmup_steps": 100,
        "grad_clip": 1.0,
        "weight_decay": 0.0,
        "value_weight": 0.5,
        "value_per_game": 16,               # positions a game that carry the value loss (all share one result)
        "td_lambda": 0.99,
        "kl_weight": 1.0,                   # x KL(pi_start || pi) on self-play positions...
        "kl_weight_end": 0.1,               # ...annealed to this over kl_anneal_versions
        "kl_anneal_versions": 20,
        "human_share": 0.0,                 # share of each batch from the human tables (0: none)
        "human_tables_dir": None,           # the imitation tables (graph files beside them for the GNN)
        "human_fraction": 0.1,              # how much of the human training games to load
        "human_data_cache": None,           # graph_supervised's memory-mapped cache of all the human rows (fast)
        "human_value_weight": None,         # x the value loss on human rows (None: value_weight)
        "policy_weight": 1.0,               # x the self-play policy loss (0: value only)          docs/028's arms:
        "policy_target": "visits",          # visits | cq (the behaviour policy tilted by the search's values)
        "cq_scale": 10.0,                   # cq's tilt per unit of value above the root's
        "train_only": None,                 # parameter-name prefixes to train (None: all)
        "value_head_init": "keep",          # keep | reset | shrink (shrink and perturb the value head)
        "kl_mask_use": False,               # no KL on yes/no rows (the start's use head never trained)
        "seed": None,
        "heldout_share": 0.05,              # self-play games kept out of training, for the offline checks
        "keep_versions": 3,                 # weights kept besides every 10th version
        "full_every_versions": 6,           # the trainer's full state (optimizer too) to the store this often
        "device": "auto",
        "lease_s": 1800,
        "poll_s": 60,
        "max_versions": None,               # stop after this many new versions (smoke tests)
    },
    "eval": {
        "every_versions": 2,                # a head-to-head against the starting network every N versions
        "simulations": 100,
        "pairs": 500,                       # deck pairs (2 games each): the sequential test's cap
        "shard_pairs": 10,
        "pool": "data/pools/eval.txt",
        "seed": 20261006,
        "opponent": "start",                # start (version 0) | heuristic
        "heuristic_simulations": 100,
        "sprt_p0": 0.50, "sprt_p1": 0.55, "sprt_alpha": 0.05, "sprt_beta": 0.05,
        "late_factor": 3.0,                 # reassign a shard after this many x its expected time
        "shard_expected_s": 2400,
        "max_share": 0.15,                  # of a worker's chunks
    },
    "controller": {"poll_s": 60, "live_s": 7200, "status_every_s": 600},   # status: one commit, at most this often
}

NETWORKS = ("gnn", "mlp")


def _merge(base: dict, over: dict, where: str) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in base:
            raise ValueError(f"unknown setting {where}{k}")
        if isinstance(base[k], dict) and base[k] and isinstance(v, dict):
            out[k] = _merge(base[k], v, f"{where}{k}.")
        else:
            out[k] = v
    return out


def load(path: str | Path | None = None, **overrides) -> dict:
    """The settings: the defaults, then the YAML at `path`, then keyword overrides (nested dicts merge)."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        import yaml
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text()) or {}, "")
    cfg = _merge(cfg, overrides, "")
    check(cfg)
    return cfg


def check(cfg: dict) -> None:
    if cfg["network"] not in NETWORKS:
        raise ValueError(f"network must be one of {NETWORKS}, not {cfg['network']!r}")
    if not cfg["start"]:
        raise ValueError("start: the starting checkpoint is required")
    if cfg["play"]["method"] not in ("pimc", "ismcts"):
        raise ValueError("play.method must be pimc or ismcts")
    if int(cfg["play"]["simulations"]) < 1:
        raise ValueError("play.simulations must be >= 1")
    if not 0 <= float(cfg["play"]["past_share"]) <= 1:
        raise ValueError("play.past_share must be in [0, 1]")
    if not 0 <= float(cfg["trainer"]["human_share"]) < 1:
        raise ValueError("trainer.human_share must be in [0, 1)")
    if float(cfg["trainer"]["human_share"]) > 0 and not cfg["trainer"]["human_tables_dir"]:
        raise ValueError("trainer.human_share needs trainer.human_tables_dir")
    if cfg["eval"]["opponent"] not in ("start", "heuristic"):
        raise ValueError("eval.opponent must be start or heuristic")


def path(p: str | Path) -> Path:
    """A setting's path, relative to the repo root unless absolute."""
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def bot(cfg: dict, simulations: int | None = None) -> str:
    """play.py's bot name for this run's network searching at `simulations`."""
    name = "gnn" if cfg["network"] == "gnn" else "il_bc"
    return f"{name}@{int(simulations or cfg['play']['simulations'])}"
