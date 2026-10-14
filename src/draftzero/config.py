"""config.py — the run config (interface 8, contracts §3): one YAML file per run, read by every role.

    run: e1-baseline-mlp            # the run id (required)
    extends: base/player.yml        # merged (deep) first: dicts merge, lists and scalars replace, {} clears a
                                    # free dict (actor.pcr.by_kind, player.config)
    actor: {sims: 200}              # everything else as in configs/base/player.yml

The YAML mirrors the nested dataclasses below (RunConfig and one class per section). Unknown keys are errors
(ConfigError, naming the dotted path and the file), so a typo cannot silently fall back to a default. Values are
converted to the field's type: YAML reads `1e-4` as text (a YAML float needs a dot), and "1e-4" becomes 1e-4 here.
`extends:` is a path relative to the file, else to one of its parent folders (so configs/runs/x.yml can say
`base/player.yml`), else to the repo's configs/; it may be a list (merged in order) and may chain.

`dump` writes the whole merged config (no `extends`), and `config_hash` fingerprints its canonical JSON, so the
store's config.yml reloads to the same hash. `player.config` overrides the architecture's defaults for a network
built from random init (a warm start keeps its checkpoint's config): its keys are checked at load against
draftzero.nets.models.DEFAULTS[player.arch] and each value takes its default's type ("5e-2" is 0.05). It merges
like any dict, and a key set to null is dropped (the arch's default), so a run that changes `player.arch` sets
the old arch's other keys to null (`card_d: null` for a transformer that extends base/player.yml).

Plan changes mid-run: only LIVE paths may change. `apply_change` validates one change, the controller records it
in events.jsonl ({kind: "set", path, old, new, at_version, ...}), and `effective` replays the events onto the
launched config: the config in effect for version N is the launched one plus every event with at_version <= N.
The MACHINE paths (checkouts, data, the store) may differ between a run's machines and are not part of that record.

    python -m draftzero.config configs/smoke.yml     prints the merged config and its hash
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import difflib
import functools
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import types
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Union

import yaml

REPO = Path(os.environ.get("DZ_REPO") or Path(__file__).resolve().parents[2])
CONFIGS = REPO / "configs"

MODES = ("async", "sync")
ARCHS = ("mlp", "transformer", "gnn")             # only mlp runs inside the engine (.dzgw); see nets/
POLICY_TARGETS = ("visits", "cq")
WINDOW_KINDS = ("fixed", "katago")                # learner.window.kind
SEARCH_KEYS = ("autopay", "oppnodes", "oppfull", "morekinds")      # draftzero.engine.specs.FULL; any of FLAG_KEYS
# decision kinds the fork's playout-cap randomization takes a per-kind probability for (dzgorge kinds= names)
PCR_KINDS = ("priority", "attackers", "blockers", "target", "modes", "optional", "choose", "multitarget")
OPPONENT_RE = re.compile(r"^(prev|sbtactical|bot|nonet|v\d+|file:.+\.dzgw)$")
RUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
PLAYER_MLP = {"d": 128, "ff": 256, "layers": 1, "d_emb": 32, "card_d": 64, "head_hidden": 128}   # loop F's width-128 MLP


class ConfigError(ValueError):
    """A config that does not fit the schema; `path` is the dotted key at fault (None for whole-file problems)."""

    def __init__(self, message: str, path: str | None = None):
        super().__init__(message)
        self.path = path


# ------------------------------------------------------------------------------------------------ the schema

@dataclass
class BudgetConfig:
    """When the run stops (the first limit reached); None = no limit."""
    max_versions: int | None = None
    max_hours: float | None = None
    max_games: int | None = None


@dataclass
class EngineConfig:
    """The gorge checkout and the engine flags every game uses (draftzero.engine.Gorge)."""
    name: str = "gorge"
    root: str = "/work/gorge-fork"                # checkout with bin/dzgorge (engine/gorge/build.sh) and .cards
    census: bool = True                          # paid-mana census; False = -no-paid-mana-census
    max_turns: int = 100
    max_intents: int = 20000
    mulligans: int = 2
    mull_heuristic: bool = True
    topk_max_options: int = 64


@dataclass
class DataConfig:
    """The deck pool in gorge format (decks/*.json + pool.tsv) and its splits."""
    decks: str = "data/gorge/decks"
    pool: str = "data/gorge/pool.tsv"
    train_split: str = "train"
    eval_split: str = "eval"


@dataclass
class PlayerConfig:
    arch: str = "mlp"
    config: dict[str, Any] = field(default_factory=lambda: dict(PLAYER_MLP))
    init: str | None = None                      # None: version 0 is the search without a net; or a .pt path / hf://


@dataclass
class BeliefConfig:
    """The corpus :closedlist draws the opponent's hidden decklist from; None = the run's own decks/pool."""
    decks: str | None = None
    pool: str | None = None
    split: str = "train"


@dataclass
class SearchConfig:
    keys: list[str] = field(default_factory=lambda: list(SEARCH_KEYS))
    topk: int | None = 6
    cpuct: float | None = 0.25
    closedlist: bool = True
    belief: BeliefConfig = field(default_factory=BeliefConfig)


@dataclass
class PcrConfig:
    """Playout-cap randomization (fork 5fd039e): with probability `prob` (or by_kind[kind]) a decision gets the full
    search (actor.sims), with root noise, and is recorded; otherwise `cheap_sims`, no noise, argmax, not recorded.
    prob 0 = off. Renders as PolicySpec(pcr=prob, cheap=cheap_sims, pcrk=tuple(sorted(by_kind.items())))."""
    prob: float = 0.0
    cheap_sims: int = 40
    by_kind: dict[str, float] = field(default_factory=dict)


@dataclass
class ActorConfig:
    sims: int = 100
    explore: bool = True
    record_every: int = 1
    chunk_games: int = 50
    chunk_secs: int = 3600                       # a chunk's hard timeout
    workers: int = 2                             # dzgorge -workers per actor process
    pcr: PcrConfig = field(default_factory=PcrConfig)


@dataclass
class WindowConfig:
    """The learner's training window. fixed: the newest learner.window_positions positions. katago: KataGo's
    growing window over the N positions ingested so far, N while N <= c, then c * (1 + beta * ((N / c)^alpha - 1) /
    alpha), at most learner.window_positions (draftzero.loop.learner.katago_window)."""
    kind: str = "fixed"
    c: int = 150000
    alpha: float = 0.75
    beta: float = 0.4


@dataclass
class LearnerConfig:
    device: str = "cpu"
    min_new_positions: int = 20000               # train once this many new recorded positions arrived (async) ...
    window_positions: int = 300000               # ... on the newest this-many positions
    epochs: float = 2.0
    steps_per_position: float | None = None      # if set: max_steps = ceil(steps_per_position * new / batch)
    batch: int = 512
    lr: float = 1.0e-4
    warmup: int = 100
    policy_target: str = "visits"
    value_blend: float = 1.0                     # value target (1-x) game outcome + x root value
    warm_start: bool = True                      # continue from the current version's weights
    threads: int = 4
    surprise_weight: float = 0.0                 # KataGo policy surprise weighting (nets.train); 0 = off
    value_surprise: float = 0.0                  # value surprise weighting (--value-surprise); 0 = off
    ema_decay: float = 0.0                       # weight moving average, published when better; 0 = off
    window: WindowConfig = field(default_factory=WindowConfig)


@dataclass
class EvalConfig:
    every_versions: int = 1
    sims: int = 100
    pairs: int = 100                             # paired games: 2 games a pair
    seed: int = 3131
    workers: int = 2
    opponents: list[str] = field(default_factory=lambda: ["prev", "sbtactical"])
    # prev | v<N> | sbtactical | bot | nonet (the search without a net) | file:/path.dzgw


@dataclass
class TrackingConfig:
    trackio: bool = False                        # mirror metrics into Trackio (draftzero.tracking); optional extra
    project: str = "draftzero"


@dataclass
class SyncConfig:
    games_per_version: int = 200                 # sync mode only


@dataclass
class RunConfig:
    run: str                                     # the run id (required)
    code: str = "dev"                            # release tag the run was launched from, or "dev"
    seed: int = 1
    mode: str = "async"
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    data: DataConfig = field(default_factory=DataConfig)
    player: PlayerConfig = field(default_factory=PlayerConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    store: str = "runs/smoke"                    # local path | hf://buckets/<owner>/<bucket> | ssh://host/path
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    paused: bool = False                         # live: actors idle while true


# the dotted paths a plan change may set mid-run (a path below one of these is live too: actor.pcr.by_kind.target)
LIVE: frozenset[str] = frozenset({
    "actor.sims", "actor.pcr.prob", "actor.pcr.cheap_sims", "actor.pcr.by_kind", "actor.record_every",
    "actor.chunk_games",
    "learner.lr", "learner.window_positions", "learner.min_new_positions", "learner.steps_per_position",
    "learner.epochs",
    "eval.every_versions", "eval.pairs", "eval.sims",
    "budget.max_versions", "budget.max_hours", "budget.max_games",
    "paused",
})


# the paths that name things on one machine (a checkout, data, the init checkpoint, the store, the tracker): a run's
# machines may differ in them, so each role's own config file supplies them; every other field is the run's
MACHINE: frozenset[str] = frozenset({
    "engine.root", "data.decks", "data.pool", "search.belief.decks", "search.belief.pool", "player.init", "store",
    "tracking.trackio", "tracking.project",
})


# ------------------------------------------------------------------------------------------------ types

_NONE_WORDS = ("null", "~")
_TRUE = ("true", "yes", "on", "1")
_FALSE = ("false", "no", "off", "0")


def _hints(cls) -> dict[str, Any]:
    return typing.get_type_hints(cls)


def _optional_of(tp) -> Any | None:
    """X for `X | None`, else None."""
    if typing.get_origin(tp) in (Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return None


def _bad(path: str, want: str, v) -> ConfigError:
    return ConfigError(f"{path or 'config'}: expected {want}, got {v!r}", path)


def _parse_text(v: str):
    try:
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v


def _coerce(v, tp, path: str):
    """`v` converted to the type `tp` (a schema annotation), or ConfigError naming `path`."""
    inner = _optional_of(tp)
    if inner is not None:
        if v is None or (isinstance(v, str) and v.strip().lower() in _NONE_WORDS):
            return None
        return _coerce(v, inner, path)
    if is_dataclass(tp):
        if v is None:
            v = {}
        elif isinstance(v, str):
            v = _parse_text(v)
        elif is_dataclass(v):
            v = dataclasses.asdict(v)
        if not isinstance(v, Mapping):
            raise _bad(path, "a mapping", v)
        return _build(tp, v, path)
    origin = typing.get_origin(tp)
    if origin is list:
        (item,) = typing.get_args(tp)
        if isinstance(v, str):
            v = _parse_text(v)
        if not isinstance(v, (list, tuple)):
            raise _bad(path, "a list", v)
        return [_coerce(x, item, f"{path}[{i}]") for i, x in enumerate(v)]
    if origin is dict:
        _, vt = typing.get_args(tp)
        if isinstance(v, str):
            v = _parse_text(v)
        if v is None:
            return {}
        if not isinstance(v, Mapping):
            raise _bad(path, "a mapping", v)
        return {str(k): _coerce(x, vt, f"{path}.{k}") for k, x in v.items()}
    if tp is Any:
        return copy.deepcopy(v)
    if tp is bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.strip().lower() in _TRUE + _FALSE:
            return v.strip().lower() in _TRUE
        raise _bad(path, "true or false", v)
    if tp is int:
        if isinstance(v, bool):
            raise _bad(path, "an integer", v)
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            s = v.strip().replace("_", "")
            try:
                return int(s)
            except ValueError:
                try:
                    v = float(s)
                except ValueError:
                    raise _bad(path, "an integer", v) from None
        if isinstance(v, float) and math.isfinite(v) and v.is_integer():
            return int(v)
        raise _bad(path, "an integer", v)
    if tp is float:
        if isinstance(v, bool):
            raise _bad(path, "a number", v)
        if isinstance(v, str):
            try:
                v = float(v.strip().replace("_", ""))
            except ValueError:
                raise _bad(path, "a number", v) from None
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return float(v)
        raise _bad(path, "a finite number", v)
    if tp is str:
        if isinstance(v, str):
            return v
        raise _bad(path, "a string", v)
    raise TypeError(f"config.py: unsupported schema type {tp!r} at {path}")


def _unknown(cls, key: str, prefix: str, where: str = "") -> ConfigError:
    names = [f.name for f in fields(cls)]
    near = difflib.get_close_matches(key, names, n=1)
    hint = f" (did you mean {prefix}{near[0]}?)" if near else ""
    return ConfigError(f"{where}unknown key {prefix}{key}{hint}", f"{prefix}{key}")


def _build(cls, d: Mapping, path: str):
    prefix = f"{path}." if path else ""
    names = {f.name for f in fields(cls)}
    for k in d:
        if k not in names:
            raise _unknown(cls, str(k), prefix)
    hints, kw = _hints(cls), {}
    for f in fields(cls):
        if f.name in d:
            kw[f.name] = _coerce(d[f.name], hints[f.name], prefix + f.name)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"missing required key {prefix}{f.name}", prefix + f.name)
    obj = cls(**kw)
    if cls is PlayerConfig:
        obj.config = _player_config(obj.arch, obj.config, prefix + "config")
    return obj


@functools.lru_cache(maxsize=1)
def _arch_defaults() -> dict[str, dict[str, Any]]:
    """draftzero.nets.models.DEFAULTS (each arch's config keys and default values), read from the module's source
    unless it is loaded: it imports torch, which loading a config (in every actor, say) should not."""
    m = sys.modules.get("draftzero.nets.models")
    if m is not None:
        return m.DEFAULTS
    spec = importlib.util.find_spec("draftzero.nets.models")
    for node in ast.parse(Path(spec.origin).read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "DEFAULTS" for t in node.targets):
            return ast.literal_eval(node.value)
    from draftzero.nets import models
    return models.DEFAULTS


def _player_config(arch: str, config: dict, path: str) -> dict:
    """player.config checked against the arch's keys, each value converted to its default's type ("5e-2" is
    0.05); a key set to null is dropped (the arch's default). An unknown arch is left to check()."""
    defaults = _arch_defaults().get(arch)
    if defaults is None:
        return config
    out = {}
    for k, v in config.items():
        if v is None or (isinstance(v, str) and v.strip().lower() in _NONE_WORDS):
            continue
        if k not in defaults:
            near = difflib.get_close_matches(k, list(defaults), n=1)
            raise ConfigError(f"unknown key {path}.{k} for a {arch}"
                              + (f" (did you mean {path}.{near[0]}?)" if near else "")
                              + f"; its keys: {', '.join(defaults)}", f"{path}.{k}")
        out[k] = _coerce(v, type(defaults[k]), f"{path}.{k}")
    return out


def _check_keys(raw: Mapping, cls, prefix: str, where: str) -> None:
    """Unknown keys in one file, before merging, so the error names the file that has the typo."""
    hints = _hints(cls)
    for k, v in raw.items():
        if k not in hints:
            raise _unknown(cls, str(k), prefix, where)
        tp = _optional_of(hints[k]) or hints[k]
        if is_dataclass(tp) and isinstance(v, Mapping):
            _check_keys(v, tp, f"{prefix}{k}.", where)


# ------------------------------------------------------------------------------------------------ loading

def _merge(base: Mapping, over: Mapping, tp=None) -> dict:
    """`over` deep-merged onto `base`: mappings merge key by key; anything else (lists too) replaces. With the
    schema type `tp` of `base`, an empty mapping for a free dict field (player.config, actor.pcr.by_kind) clears
    it, where it would otherwise change nothing."""
    hints = _hints(tp) if is_dataclass(tp) else {}
    out = copy.deepcopy(dict(base))
    for k, v in over.items():
        ftp = _optional_of(hints.get(k)) or hints.get(k)
        clear = isinstance(v, Mapping) and not v and typing.get_origin(ftp) is dict
        if isinstance(out.get(k), Mapping) and isinstance(v, Mapping) and not clear:
            out[k] = _merge(out[k], v, ftp)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _expand(overrides: Mapping[str, Any]) -> dict:
    """{"actor.sims": 16} -> {"actor": {"sims": 16}}; nested mappings pass through."""
    out: dict = {}
    for k, v in overrides.items():
        node = out
        *head, last = str(k).split(".")
        for part in head:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"override {k}: {part} is set to a value and a section at once", str(k))
        node[last] = _merge(node[last], v) if isinstance(node.get(last), Mapping) and isinstance(v, Mapping) else v
    return out


def _find_base(ref: str, here: Path | None) -> Path:
    p = Path(ref).expanduser()
    if p.is_absolute():
        if p.is_file():
            return p.resolve()
    else:
        dirs = [here, *here.parents] if here is not None else [Path.cwd()]
        for d in [*dirs, CONFIGS]:
            if (d / p).is_file():
                return (d / p).resolve()
    raise ConfigError(f"extends: {ref} not found (looked next to {here or Path.cwd()}, in its parent folders and "
                      f"in {CONFIGS})", "extends")


def _raw(data, src: str, here: Path | None, seen: tuple[Path, ...]) -> dict:
    """The file's mapping with its `extends:` chain merged underneath it."""
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"{src}: a config is a YAML mapping, not {type(data).__name__}")
    data = dict(data)
    ext = data.pop("extends", None)
    _check_keys(data, RunConfig, "", f"{src}: ")
    merged: dict = {}
    for ref in ([] if ext is None else [ext] if isinstance(ext, (str, Path)) else list(ext)):
        bp = _find_base(str(ref), here)
        if bp in seen:
            raise ConfigError(f"{src}: extends cycle through {bp}", "extends")
        base = yaml.safe_load(bp.read_text())
        merged = _merge(merged, _raw(base, str(bp), bp.parent, (*seen, bp)), RunConfig)
    return _merge(merged, data, RunConfig)


def from_dict(d: Mapping, overrides: Mapping[str, Any] | None = None) -> RunConfig:
    """A RunConfig from a plain mapping (to_dict's shape); `extends` is resolved against the working directory."""
    raw = _raw(d, "config", None, ())
    if overrides:
        raw = _merge(raw, _expand(overrides), RunConfig)
    cfg = _coerce(raw, RunConfig, "")
    check(cfg)
    return cfg


def loads(text: str, overrides: Mapping[str, Any] | None = None, *, base_dir: str | Path | None = None) -> RunConfig:
    """A RunConfig from YAML text (e.g. a store's config.yml); `extends` resolves against base_dir (default cwd)."""
    here = Path(base_dir).resolve() if base_dir is not None else Path.cwd()
    raw = _raw(yaml.safe_load(text), "config", here, ())
    if overrides:
        raw = _merge(raw, _expand(overrides), RunConfig)
    cfg = _coerce(raw, RunConfig, "")
    check(cfg)
    return cfg


def load(path: str | Path, overrides: Mapping[str, Any] | None = None) -> RunConfig:
    """The config at `path` (its `extends:` chain merged first), then `overrides` ({"actor.sims": 16} or nested),
    converted and checked. Overrides are launch-time settings: any key, live or not."""
    p = Path(path).expanduser().resolve()
    try:
        data = yaml.safe_load(p.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: not valid YAML: {e}") from None
    raw = _raw(data, str(path), p.parent, (p,))
    if overrides:
        raw = _merge(raw, _expand(overrides), RunConfig)
    try:
        cfg = _coerce(raw, RunConfig, "")
        check(cfg)
    except ConfigError as e:
        raise ConfigError(f"{path}: {e}", e.path) from None
    return cfg


# ------------------------------------------------------------------------------------------------ output

def to_dict(cfg: RunConfig) -> dict:
    """The config as plain dicts and lists, in schema order (what config.yml holds)."""
    return dataclasses.asdict(cfg)


class _Section(dict):
    """A schema section: dumped as a block mapping (lists and free dicts are dumped inline)."""


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(_Section, lambda d, v: d.represent_mapping("tag:yaml.org,2002:map", v, flow_style=False))
_Dumper.add_representer(dict, lambda d, v: d.represent_mapping("tag:yaml.org,2002:map", v, flow_style=True))
_Dumper.add_representer(list, lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", v, flow_style=True))


def _sectioned(obj) -> _Section:
    return _Section((f.name, _sectioned(v) if is_dataclass(v := getattr(obj, f.name)) else copy.deepcopy(v))
                    for f in fields(obj))


def dump(cfg: RunConfig) -> str:
    """The whole config as YAML (no `extends`), in schema order: load(dump) gives the same config_hash."""
    head = f"# run config {cfg.run}: config_hash {config_hash(cfg)} (draftzero.config.dump)\n"
    return head + yaml.dump(_sectioned(cfg), Dumper=_Dumper, sort_keys=False, width=120, allow_unicode=False)


def config_hash(cfg: RunConfig) -> str:
    """sha256 of the config's canonical JSON (sorted keys, no spaces), first 16 hex digits."""
    s = json.dumps(to_dict(cfg), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def flatten(cfg: RunConfig | Mapping) -> dict[str, Any]:
    """{dotted path: value} for every leaf field (dict-typed fields such as actor.pcr.by_kind stay whole)."""
    out: dict[str, Any] = {}

    def walk(obj, prefix: str) -> None:
        for f in fields(obj):
            v = getattr(obj, f.name)
            if is_dataclass(v):
                walk(v, f"{prefix}{f.name}.")
            else:
                out[prefix + f.name] = copy.deepcopy(v)

    walk(cfg if is_dataclass(cfg) else from_dict(cfg), "")
    return out


def diff(a: RunConfig, b: RunConfig) -> dict[str, tuple[Any, Any]]:
    """{path: (a's value, b's value)} for the leaf fields that differ."""
    fa, fb = flatten(a), flatten(b)
    return {k: (fa[k], fb[k]) for k in fa if fa[k] != fb[k]}


def get(cfg: RunConfig, path: str):
    """The value at a dotted path (a section comes back as a dict); ConfigError if there is none."""
    node: Any = to_dict(cfg)
    for i, part in enumerate(path.split(".")):
        if not isinstance(node, Mapping) or part not in node:
            raise ConfigError(f"unknown key {'.'.join(path.split('.')[:i + 1])}", path)
        node = node[part]
    return node


def resolve_path(p: str | Path) -> Path:
    """A config path (data.decks, data.pool, store, ...): absolute as is, else relative to the repo root when
    running from a checkout ($DZ_REPO overrides), else to the working directory."""
    p = Path(p).expanduser()
    if p.is_absolute():
        return p
    return (REPO / p) if (REPO / "pyproject.toml").is_file() else Path.cwd() / p


# ------------------------------------------------------------------------------------------------ checks

def check(cfg: RunConfig) -> None:
    """Rules across fields and ranges; ConfigError naming the first path that breaks one."""
    def need(ok: bool, path: str, what: str) -> None:
        if not ok:
            raise ConfigError(f"{path}: {what}", path)

    need(bool(RUN_RE.match(cfg.run)), "run", f"a run id is letters, digits, '.', '_' and '-' ({cfg.run!r})")
    need(bool(cfg.code.strip()), "code", "a release tag or 'dev'")
    need(cfg.mode in MODES, "mode", f"one of {MODES}, not {cfg.mode!r}")
    for k in ("max_versions", "max_hours", "max_games"):
        v = getattr(cfg.budget, k)
        need(v is None or v > 0, f"budget.{k}", "null or > 0")
    e = cfg.engine
    need(e.name == "gorge", "engine.name", f"only gorge is supported, not {e.name!r}")
    need(bool(e.root), "engine.root", "the gorge checkout")
    for k in ("max_turns", "max_intents", "topk_max_options"):
        need(getattr(e, k) >= 1, f"engine.{k}", ">= 1")
    need(e.mulligans >= 0, "engine.mulligans", ">= 0")
    need(cfg.player.arch in ARCHS, "player.arch", f"one of {ARCHS}, not {cfg.player.arch!r}")
    for k, v in cfg.player.config.items():
        need(isinstance(v, (int, float, str, bool)) or v is None, f"player.config.{k}", "a scalar")
    pc = {**_arch_defaults().get(cfg.player.arch, {}), **cfg.player.config}
    if "heads" in pc:
        need(pc["heads"] >= 1 and pc["d"] % pc["heads"] == 0, "player.config.heads",
             f"divides player.config.d ({pc['d']})")
    init = cfg.player.init
    need(init is None or bool(init.strip()), "player.init", "null, a path or hf://owner/repo[@rev]/file")
    if init is not None and init.startswith("hf://"):
        from draftzero.hub import parse
        try:
            ok = parse(init).path is not None
        except ValueError:
            ok = False
        need(ok, "player.init", f"hf://owner/repo[@rev]/file, not {init!r}")
    s = cfg.search
    from draftzero.engine.specs import FLAG_KEYS        # what PolicySpec.keys takes (a light module)
    for i, k in enumerate(s.keys):
        need(k in FLAG_KEYS, f"search.keys[{i}]", f"a bare engine flag, one of {', '.join(FLAG_KEYS)} (not {k!r})")
    need(s.topk is None or s.topk >= 1, "search.topk", "null or >= 1")
    need(s.cpuct is None or s.cpuct > 0, "search.cpuct", "null or > 0")
    a = cfg.actor
    for k in ("sims", "record_every", "chunk_games", "chunk_secs", "workers"):
        need(getattr(a, k) >= 1, f"actor.{k}", ">= 1")
    p = a.pcr
    need(0.0 <= p.prob <= 1.0, "actor.pcr.prob", "in [0, 1]")
    need(p.cheap_sims >= 1, "actor.pcr.cheap_sims", ">= 1")
    for k, v in p.by_kind.items():
        need(k in PCR_KINDS, f"actor.pcr.by_kind.{k}", f"a decision kind, one of {PCR_KINDS}")
        need(0.0 <= v <= 1.0, f"actor.pcr.by_kind.{k}", "in [0, 1]")
    need(not p.by_kind or p.prob > 0, "actor.pcr.by_kind",
         "needs actor.pcr.prob > 0 (the full-search share of the other kinds; 0 turns playout-cap randomization off)")
    need(p.prob == 0 or p.cheap_sims <= a.sims, "actor.pcr.cheap_sims", f"at most actor.sims ({a.sims})")
    lr = cfg.learner
    need(lr.min_new_positions >= 1, "learner.min_new_positions", ">= 1")
    need(lr.window_positions >= lr.min_new_positions, "learner.window_positions",
         f"at least learner.min_new_positions ({lr.min_new_positions})")
    need(lr.epochs > 0, "learner.epochs", "> 0")
    need(lr.steps_per_position is None or lr.steps_per_position > 0, "learner.steps_per_position", "null or > 0")
    need(lr.batch >= 1, "learner.batch", ">= 1")
    need(lr.lr > 0, "learner.lr", "> 0")
    need(lr.warmup >= 0, "learner.warmup", ">= 0")
    need(lr.policy_target in POLICY_TARGETS, "learner.policy_target", f"one of {POLICY_TARGETS}")
    need(0.0 <= lr.value_blend <= 1.0, "learner.value_blend", "in [0, 1]")
    need(lr.threads >= 1, "learner.threads", ">= 1")
    need(0.0 <= lr.surprise_weight <= 1.0, "learner.surprise_weight", "in [0, 1] (0 = off)")
    need(0.0 <= lr.value_surprise <= 1.0, "learner.value_surprise", "in [0, 1] (0 = off)")
    need(0.0 <= lr.ema_decay < 1.0, "learner.ema_decay", "in [0, 1) (0 = off)")
    w = lr.window
    need(w.kind in WINDOW_KINDS, "learner.window.kind", f"one of {WINDOW_KINDS}, not {w.kind!r}")
    need(w.c >= 1, "learner.window.c", ">= 1")
    need(0.0 < w.alpha <= 1.0, "learner.window.alpha", "in (0, 1]")
    need(0.0 <= w.beta <= 1.0, "learner.window.beta", "in [0, 1]")
    ev = cfg.eval
    for k in ("every_versions", "sims", "pairs", "workers"):
        need(getattr(ev, k) >= 1, f"eval.{k}", ">= 1")
    for i, o in enumerate(ev.opponents):
        need(bool(OPPONENT_RE.match(o)), f"eval.opponents[{i}]",
             f"prev, v<N>, sbtactical, bot, nonet or file:<path>.dzgw, not {o!r}")
    st = cfg.store
    need(bool(st.strip()), "store", "a local path, hf://buckets/<owner>/<bucket> or ssh://host/path")
    if st.startswith("hf://"):
        need(bool(re.match(r"^hf://buckets/[^/\s]+/[^/\s]+/?$", st)), "store", f"hf://buckets/<owner>/<bucket>, not {st!r}")
    if st.startswith("ssh://"):
        need(bool(re.match(r"^ssh://[^/\s]+/.+", st)), "store", f"ssh://host/path, not {st!r}")
    need(bool(cfg.tracking.project.strip()), "tracking.project", "a project name")
    need(cfg.sync.games_per_version >= 1, "sync.games_per_version", ">= 1")


# ------------------------------------------------------------------------------------------------ plan changes

def _sections(cls, prefix: str = "") -> dict[str, list[str]]:
    """{section path: its leaf paths} for every nested section."""
    out: dict[str, list[str]] = {}
    leaves: list[str] = []
    for name, tp in _hints(cls).items():
        tp = _optional_of(tp) or tp
        if is_dataclass(tp):
            sub = _sections(tp, f"{prefix}{name}.")
            out.update(sub)
            leaves += sub[f"{prefix}{name}"]
        else:
            leaves.append(prefix + name)
    out[prefix.rstrip(".")] = leaves
    return out


_LIVE_SECTIONS = frozenset(k for k, leaves in _sections(RunConfig).items() if k and all(x in LIVE for x in leaves))


def is_live(path: str) -> bool:
    """True if a plan change may set `path`: a LIVE path, a key below one (actor.pcr.by_kind.target), or a whole
    section whose every field is live (budget, actor.pcr)."""
    parts = path.split(".")
    return path in _LIVE_SECTIONS or any(".".join(parts[:i]) in LIVE for i in range(1, len(parts) + 1))


def _set(cfg: RunConfig, path: str, value) -> RunConfig:
    """A new config with `path` set to `value` (converted and checked); `cfg` is untouched."""
    parts = path.split(".")
    if not path or not all(parts):
        raise ConfigError(f"bad path {path!r}", path)
    d = to_dict(cfg)
    node, tp = d, RunConfig
    for i, part in enumerate(parts):
        here, last = ".".join(parts[:i + 1]), i == len(parts) - 1
        if is_dataclass(tp):
            hints = _hints(tp)
            if part not in hints:
                raise _unknown(tp, part, ".".join(parts[:i]) + "." if i else "")
            ftp = _optional_of(hints[part]) or hints[part]
            if last:
                if is_dataclass(ftp):           # a whole section: merge, so unnamed fields keep their values
                    v = _parse_text(value) if isinstance(value, str) else value
                    if not isinstance(v, Mapping):
                        raise _bad(here, "a mapping", value)
                    node[part] = _merge(node[part] or {}, v, ftp)
                else:
                    node[part] = value
            else:
                node, tp = node[part], ftp
        elif typing.get_origin(tp) is dict and last:
            if value is None:
                node.pop(part, None)
            else:
                node[part] = value
        else:
            raise ConfigError(f"unknown key {here}", path)
    new = _coerce(d, RunConfig, "")
    check(new)
    return new


def apply_change(cfg: RunConfig, path: str, value) -> RunConfig:
    """`cfg` with one plan change applied (a new object; `cfg` is untouched). The path must be live (is_live) and
    the value must fit its type; text values are read as YAML ("3e-4", "true", "null", "{prob: 0.5}"). A section
    merges into the current one; setting a by_kind entry to None removes it, and an empty by_kind ({}, alone or in
    an actor.pcr section) clears it. ConfigError otherwise."""
    if not is_live(path):
        get(cfg, path)                           # an unknown path says so first
        raise ConfigError(f"{path} is fixed for the run; live paths: {', '.join(sorted(LIVE))}", path)
    return _set(cfg, path, value)


def effective(cfg: RunConfig, events: Iterable[dict], upto_version: int | None = None) -> RunConfig:
    """The launched config with the plan's "set" events replayed in order. With upto_version, only the events in
    effect for that version (at_version <= upto_version; an event without at_version always counts). Events are
    replayed without the live check (they were checked when made), so narrowing LIVE later cannot break a replay."""
    for ev in events:
        if ev.get("kind", "set") != "set":
            continue
        at = ev.get("at_version")
        if upto_version is not None and at is not None and int(at) > upto_version:
            continue
        cfg = _set(cfg, ev["path"], ev["new"])
    return cfg


def main(argv: list[str] | None = None) -> int:
    import sys
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("usage: python -m draftzero.config CONFIG.yml [path=value ...]", file=sys.stderr)
        return 2
    over = dict(a.split("=", 1) for a in args[1:])
    print(dump(load(args[0], over)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
