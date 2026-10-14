"""cli.py — the `dz` command (contracts §7). `dz <command> --help` describes each one.

    dz version                                  the installed DraftZero version
    dz manifest [CONFIG]                        what defines a run here: code, engine build, cards, machine (alias:
                                                provenance); with CONFIG also the config, its hash and the belief corpus
    dz run CONFIG [--store S] [--actors N] [--max-hours H] [--work-dir DIR] [--set PATH=VALUE ...]
                                                every role of a run on this machine (local mode)
    dz actor|learner|evaluator|controller --config C [--store S] [--machine M]
                                                one role, for runs over several machines (draftzero.loop.cli)
    dz plan STORE set PATH=VALUE ... | list     a live config change, applied by the controller
    dz status STORE [--json]                    the controller's status page
    dz stop STORE                               ask the controller to stop the run
    dz report STORE|RUN_DIR [--out DIR]         the standard figures (needs the `plots` extra)
    dz models pull hf://owner/repo[@rev]/path   a published model file, into the local Hugging Face cache
    dz play --a SPEC --b SPEC --pairs N         paired games between two gorge policy specs, and A's score
    dz eval MODEL --opponents a,b [--pairs N]   a network (.dzgw, hf:// file, or a run's version) against opponents
    dz engine build|version|coverage            build dzgorge at engine/gorge/GORGE_REF; its build record; deck support
    dz pod up|down|list|balance|offers          RunPod pods for relay actors (up needs --max-price and --max-hours)
    dz relay CONFIG --host NAME ROOT [SSH] ...  move shards and versions between the run's store and actor hosts

Commands that play games (run, play, eval, engine version|coverage, manifest) read the engine and the deck pool from
a config: CONFIG, or --config (default configs/base/player.yml), with --set PATH=VALUE overrides at launch, e.g.
--set engine.root=/path/to/gorge --set data.decks=assets/sample/decks --set data.pool=assets/sample/pool.tsv.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

LOOP_COMMANDS = ("actor", "learner", "evaluator", "controller", "status", "stop", "plan")
LOCAL_POLL_S = 10.0     # dz run on a folder: looking costs nothing, and an idle role backs off to 4x this
SAMPLE_DATA = "--set data.decks=assets/sample/decks --set data.pool=assets/sample/pool.tsv"


class CliError(Exception):
    """A user-facing error: printed as `dz <command>: <message>`, exit code 2."""


def _user_errors() -> tuple[type[BaseException], ...]:
    """Errors that mean bad input or a missing file rather than a bug: printed without a traceback."""
    out: list[type[BaseException]] = [FileNotFoundError]
    try:
        from draftzero.config import ConfigError
        out.append(ConfigError)
    except ImportError:
        pass
    try:
        from draftzero.engine.gorge import GorgeError
        out.append(GorgeError)
    except ImportError:
        pass
    mod = sys.modules.get("draftzero.workers.runpod")
    if mod is not None:
        out.append(mod.RunPodError)
    return tuple(out)


# ------------------------------------------------------------------------------------------------ config and engine

def _overrides(sets: list[str] | None) -> dict:
    """["engine.root=/x", "actor.sims=16"] -> {"engine.root": "/x", "actor.sims": 16} (values read as JSON/YAML)."""
    from draftzero.loop.control import parse_value
    out = {}
    for s in sets or []:
        k, eq, v = s.partition("=")
        if not eq or not k.strip():
            raise CliError(f"--set {s!r}: want PATH=VALUE, e.g. --set engine.root=/work/gorge-fork")
        out[k.strip()] = parse_value(v)
    return out


def base_config() -> Path:
    from draftzero import config
    return config.CONFIGS / "base" / "player.yml"


def load_config(path: str | Path | None, sets: list[str] | None = None):
    """The RunConfig at `path` (default configs/base/player.yml) with --set overrides."""
    from draftzero import config
    p = Path(path) if path else base_config()
    if not p.is_file():
        raise CliError(f"no config at {p}" + ("" if path else " (outside a source checkout, pass --config)"))
    return config.load(p, _overrides(sets) or None)


def engine_problems(cfg, *, data: bool = True) -> list[str]:
    """What is missing to play games with cfg's engine and deck pool, each with the fix."""
    from draftzero.engine.gorge import Gorge
    g = Gorge.from_config(cfg)
    out = []
    if not g.binary.is_file():
        out.append(f"no dzgorge at {g.binary}: build it with `dz engine build --dir {g.root}`, or point the "
                   "config's engine.root at a build (--set engine.root=DIR)")
    elif not g.cards.is_dir():
        out.append(f"no card corpus at {g.cards}: `dz engine build --dir {g.root}` fetches and compiles it")
    if data and not (g.decks.is_dir() and g.pool.is_file()):
        out.append(f"no deck pool at {g.decks} and {g.pool}: build it (tools/extract_decks.py, then "
                   f"python -m draftzero.data.decks) or play the 80 sample decks: {SAMPLE_DATA}")
    return out


def make_gorge(cfg, *, data: bool = True):
    """Gorge.from_config(cfg), or CliError saying what is missing."""
    problems = engine_problems(cfg, data=data)
    if problems:
        raise CliError("; ".join(problems))
    from draftzero.engine.gorge import Gorge
    return Gorge.from_config(cfg)


# ------------------------------------------------------------------------------------------------ version, manifest

def cmd_version(a) -> int:
    from draftzero import __version__
    print(__version__)
    return 0


def cmd_manifest(a) -> int:
    from draftzero import manifest
    cfg = load_config(a.config, a.set) if a.config else None
    gorge = None
    if cfg is not None and not engine_problems(cfg, data=False):
        from draftzero.engine.gorge import Gorge
        gorge = Gorge.from_config(cfg)
    print(json.dumps(manifest.collect(cfg, gorge=gorge), indent=2, default=str))
    return 0


# ------------------------------------------------------------------------------------------------ run

def _code_warning(cfg) -> str | None:
    """A warning when the config names a release (code: v0.2.0) this checkout isn't at, clean."""
    code = getattr(cfg, "code", "dev")
    if code == "dev":
        return None
    from draftzero import manifest
    repo = manifest._checkout()
    if repo is None:
        return None
    state = manifest.git_state(repo)
    tag = manifest._git(repo, "describe", "--tags", "--exact-match", "HEAD")
    if tag == code and state.get("dirty") is False:
        return None
    where = f"{(state.get('commit') or 'unknown')[:9]}" + (", dirty" if state.get("dirty") else "")
    return (f"code: {code}, but this checkout is at {tag or where}: the run is not publishable as {code} "
            "(fine for development; launch published runs from the release tag on a clean tree)")


def cmd_run(a) -> int:
    from draftzero import config
    from draftzero.loop import cli as loop_cli
    from draftzero.loop.actor import check_playable
    sets = _overrides(a.set)
    cfg = config.load(a.config, sets or None)
    store = a.store or cfg.store
    if store.startswith("ssh://"):
        raise CliError(f"store {store}: an ssh:// store is served by a relay (`dz relay`); run the roles against a "
                       "local folder or an hf://buckets/ store")
    check_playable(cfg)
    if not a.no_check:
        problems = engine_problems(cfg)
        if problems:
            raise CliError("; ".join(problems))
    warn = _code_warning(cfg)
    if warn:
        print(f"dz run: warning: {warn}", file=sys.stderr)
    wd = Path(a.work_dir) if a.work_dir else Path("runs") / cfg.run / "work"
    cfg_path = Path(a.config)
    if sets:                                             # the roles read one file: the config with the overrides
        wd.mkdir(parents=True, exist_ok=True)
        cfg_path = wd / "config.yml"
        cfg_path.write_text(config.dump(cfg))
    argv = ["local", "--config", str(cfg_path), "--store", str(store), "--actors", str(a.actors),
            "--work-dir", str(wd)]
    if a.max_hours is not None:
        argv += ["--max-hours", str(a.max_hours)]
    if a.machine:
        argv += ["--machine", a.machine]
    if a.allow_config_change:
        argv += ["--allow-config-change"]
    poll = a.poll if a.poll is not None else (None if _is_remote(store) else LOCAL_POLL_S)
    if poll is not None:
        argv += ["--poll", str(poll)]
    print(f"dz run: {cfg.run} ({config.config_hash(cfg)}), store {store}, {a.actors} actor(s), logs in {wd}/logs; "
          f"watch with `dz status {store}`, end with `dz stop {store}`", flush=True)
    rc = loop_cli.main(argv)
    why = {0: "the run stopped", 1: "a role failed (see its log)", 2: f"--max-hours {a.max_hours} reached",
           130: "interrupted", 143: "terminated (SIGTERM)", 129: "hung up (SIGHUP)"}.get(rc, f"exit {rc}")
    print(f"dz run: {why}; `dz status {store}`, `dz report {store}`", flush=True)
    return rc


def cmd_loop(a) -> int:            # only reached through main's shortcut; kept for completeness
    from draftzero.loop import cli as loop_cli
    return loop_cli.main([a.cmd, *a.rest], prog="dz")


# ------------------------------------------------------------------------------------------------ report, models

def _is_remote(spec: str) -> bool:
    return "://" in str(spec)


def cmd_report(a) -> int:
    src = a.src
    if not _is_remote(src) and not Path(src).exists():
        raise CliError(f"{src}: no such store, run folder or metrics file")
    out = Path(a.out) if a.out else (Path(src) / "report" if not _is_remote(src) and Path(src).is_dir()
                                     else Path("report"))
    themes = ("light", "dark") if a.theme == "both" else (a.theme,)
    try:
        from draftzero.plots import standard
        paths = standard.report(src, out, themes=themes, anchor=a.anchor)
    except ImportError as e:
        if "matplotlib" not in str(e):
            raise
        msg = str(e) if "pip install" in str(e) else f"{e}: pip install 'draftzero[plots]'"
        print(f"dz report: {msg}", file=sys.stderr)
        return 2
    if not paths:
        print(f"dz report: no metrics in {src} yet (nothing drawn)", file=sys.stderr)
        return 1
    for p in paths:
        print(p)
    return 0


def cmd_models(a) -> int:
    from draftzero import hub
    for uri in a.uri:
        if not hub.is_hub(uri):
            raise CliError(f"{uri}: want hf://<owner>/<repo>[@<revision>]/<path>")
        print(hub.resolve(uri, cache_dir=a.cache_dir))
    return 0


# ------------------------------------------------------------------------------------------------ play and eval

def _spec(s: str, sims: int | None) -> str:
    """A policy spec string, checked (PolicySpec.parse), with sims replaced on a search kind when --sims is given."""
    from draftzero.engine.specs import KINDS, SEARCH_KINDS, PolicySpec
    try:
        p = PolicySpec.parse(s)
    except ValueError as e:
        raise CliError(str(e)) from None
    if p.kind not in KINDS:
        raise CliError(f"policy {s!r}: kind {p.kind!r} is not one of {', '.join(KINDS)}")
    if sims is not None and p.kind in SEARCH_KINDS:
        import dataclasses
        return dataclasses.replace(p, sims=int(sims)).render()
    return s


def _fmt_score(s: dict) -> str:
    if s.get("score") is None:
        return "no complete pair"
    return (f"{s['score']:.3f} ± {1.96 * (s.get('se') or 0):.3f} (95% [{s['lo']:.3f}, {s['hi']:.3f}]) over "
            f"{s['n_pairs']} pairs")


def _out_dir(out: str | None, prefix: str):
    """(directory, temporary?) for games: --out, else a temporary directory removed at the end."""
    if out:
        p = Path(out).absolute()
        p.mkdir(parents=True, exist_ok=True)
        return p, False
    return Path(tempfile.mkdtemp(prefix=prefix)), True


def cmd_play(a) -> int:
    from draftzero.eval.paired import score_pairs
    from draftzero.loop.actor import belief
    cfg = load_config(a.config, a.set)
    gorge = make_gorge(cfg)
    sa, sb = _spec(a.a, a.sims), _spec(a.b, a.sims)
    split = a.split or cfg.data.eval_split
    seed = cfg.eval.seed if a.seed is None else a.seed
    out, tmp = _out_dir(a.out, "dz-play-")
    try:
        t0 = time.time()
        res = gorge.play(sa, sb, pairs=a.pairs, seed=seed, out_dir=out, split=split,
                         workers=a.workers or cfg.eval.workers, belief=belief(cfg),
                         timeout_s=a.timeout or float(cfg.actor.chunk_secs))
        wall = time.time() - t0
        s = score_pairs(res.games, hero_side="A", paired=sa != sb)
        rec = {"a": sa, "b": sb, "pairs": a.pairs, "seed": seed, "split": split, "games": len(res.games),
               "skipped": res.skipped, "wall_s": round(wall, 1), "score_a": s,
               "games_path": None if tmp else str(res.games_path)}
        if a.json:
            print(json.dumps(rec, indent=1, default=str))
        else:
            print(f"A {sa}\nB {sb}")
            print(f"{len(res.games)} games on the {split} split, seed {seed}, in {wall:.0f} s"
                  + (f"; pairs skipped after a hang: {res.skipped}" if res.skipped else ""))
            print(f"A's score {_fmt_score(s)}: A won {s['wins']}, B won {s['losses']}, {s['draws']} draws, "
                  f"{s['stalls']} stalls, {s['errors']} errors")
            if not tmp:
                print(f"games: {res.games_path}")
    finally:
        if tmp:
            shutil.rmtree(out, ignore_errors=True)
    return 0


_STORE_VERSION = re.compile(r"^(?P<store>.+?)(?:@(?P<v>v?\d+|latest))?$")


def _model_source(model: str):
    """(store or None, version or None, local .dzgw path or None) for dz eval's MODEL argument:
    a .dzgw file, an hf:// model file, or a run's store (folder or hf://buckets/...) with @vN (default @latest)."""
    from draftzero import hub
    if model.startswith("hf://") and not model.startswith("hf://buckets/"):
        return None, None, hub.resolve(model)
    if not model.startswith("hf://") and Path(model).is_file():
        return None, None, Path(model).absolute()
    m = _STORE_VERSION.match(model)
    spec, v = m["store"], m["v"]
    from draftzero.loop import control as ctl
    from draftzero.loop.store import open_store
    if not spec.startswith("hf://") and not (Path(spec) / ctl.CONFIG).is_file():
        raise CliError(f"{model}: not a .dzgw file, an hf:// model file or a run store (a folder with config.yml)")
    store = open_store(spec)
    if v in (None, "latest"):
        latest = store.read_json(ctl.latest_path())
        if latest is None:
            raise CliError(f"{spec}: no version published yet ({ctl.latest_path()} is missing)")
        version = int(latest["version"])
    else:
        version = int(v.lstrip("v"))
    return store, version, None


def _store_config(store, a):
    """The run's own config (the store's config.yml) unless --config is given; --set applies either way."""
    from draftzero import config
    from draftzero.loop import control as ctl
    if a.config:
        return load_config(a.config, a.set)
    raw = store.read_bytes(ctl.CONFIG)
    if raw is None:
        return load_config(None, a.set)
    return config.loads(raw.decode(), _overrides(a.set) or None)


def _version_spec(cfg, store, version: int, sims: int, wd: Path):
    from draftzero.loop import control as ctl
    from draftzero.loop.actor import policy_spec
    _, files = ctl.fetch_version(store, version, wd / "versions" / f"v{version:04d}")
    return policy_spec(cfg, sims=sims, explore=False, gonet=files.get("model.dzgw"))


def opponent_spec(cfg, name: str, sims: int, *, store=None, version: int | None = None, wd: Path | None = None):
    """dz eval's opponents: the run config's names (prev, v<N>, sbtactical, bot, nonet, file:<path>.dzgw) plus
    random, a .dzgw path, or any gorge policy spec (e.g. "az:sims=100:autopay")."""
    from draftzero.loop.actor import policy_spec
    if name in ("prev",) or re.fullmatch(r"v\d+", name):
        if store is None:
            raise CliError(f"opponent {name}: needs MODEL to be a run's store (STORE[@vN])")
        v = version - 1 if name == "prev" else int(name[1:])
        if v < 0:
            raise CliError(f"opponent {name}: version {version} has no previous version")
        return _version_spec(cfg, store, v, sims, wd)
    if name == "nonet":
        return policy_spec(cfg, sims=sims, explore=False, gonet=None)
    if name in ("sbtactical", "bot", "random"):
        return name
    path = name[len("file:"):] if name.startswith("file:") else name
    if path.endswith(".dzgw"):
        p = Path(path).absolute()
        if not p.is_file():
            raise CliError(f"opponent {name}: no such file")
        return policy_spec(cfg, sims=sims, explore=False, gonet=p)
    return _spec(name, None)


def _label(name: str) -> str:
    s = re.sub(r"-{2,}", "-", re.sub(r"[^A-Za-z0-9._-]+", "-", name)).strip("-")
    return s[:60] or "opponent"


def cmd_eval(a) -> int:
    from draftzero.eval.paired import score_pairs
    from draftzero.loop.actor import belief, check_playable, policy_spec
    store, version, net = _model_source(a.model)
    cfg = _store_config(store, a) if store is not None else load_config(a.config, a.set)
    check_playable(cfg)
    gorge = make_gorge(cfg)
    sims = a.sims or cfg.eval.sims
    pairs = a.pairs or cfg.eval.pairs
    seed = cfg.eval.seed if a.seed is None else a.seed
    split = cfg.data.eval_split
    opponents = [o.strip() for o in a.opponents.split(",") if o.strip()]
    if not opponents:
        raise CliError("--opponents: name at least one (e.g. sbtactical,nonet)")
    out, tmp = _out_dir(a.out, "dz-eval-")
    results = []
    try:
        hero = (_version_spec(cfg, store, version, sims, out) if store is not None
                else policy_spec(cfg, sims=sims, explore=False, gonet=net))
        name = f"v{version:04d} of {a.model.split('@')[0]}" if store is not None else str(net)
        if not a.json:
            print(f"dz eval: {name}\n  {hero.render()}\n  {pairs} pairs (2 games each) against each opponent, "
                  f"{sims} simulations, {split} split, seed {seed}", flush=True)
        for opp in opponents:
            b = opponent_spec(cfg, opp, sims, store=store, version=version, wd=out)
            t0 = time.time()
            res = gorge.play(hero, b, pairs=pairs, seed=seed, out_dir=out / "games" / _label(opp), split=split,
                             workers=a.workers or cfg.eval.workers, belief=belief(cfg),
                             timeout_s=a.timeout or float(cfg.actor.chunk_secs))
            s = score_pairs(res.games, hero_side="A")
            results.append({"opponent": opp, "spec": b if isinstance(b, str) else b.render(),
                            "wall_s": round(time.time() - t0, 1), "skipped": res.skipped, **s})
            if not a.json:
                print(f"  vs {opp}: {_fmt_score(s)} ({s['wins']}-{s['losses']}-{s['draws'] + s['stalls']} W-L-D, "
                      f"{s['errors']} errors, {time.time() - t0:.0f} s)", flush=True)
        summary = {"model": a.model, "version": version, "spec": hero.render(), "sims": sims, "pairs": pairs,
                   "seed": seed, "split": split, "results": results}
        if not tmp:
            (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
        if a.json:
            print(json.dumps(summary, indent=1, default=str))
        else:
            print()
            print(f"{'opponent':<24} {'score':>6} {'95% interval':>16} {'pairs':>5} {'W-L-D':>9} {'errors':>6}")
            for r in results:
                sc = "-" if r["score"] is None else f"{r['score']:.3f}"
                ci = "-" if r["score"] is None else f"[{r['lo']:.3f}, {r['hi']:.3f}]"
                wld = f"{r['wins']}-{r['losses']}-{r['draws'] + r['stalls']}"
                print(f"{r['opponent'][:24]:<24} {sc:>6} {ci:>16} {r['n_pairs']:>5} {wld:>9} {r['errors']:>6}")
            print()
            print(json.dumps(summary, default=str))
    finally:
        if tmp:
            shutil.rmtree(out, ignore_errors=True)
    return 0


# ------------------------------------------------------------------------------------------------ engine

def cmd_engine(a) -> int:
    from draftzero import config
    if a.engine_cmd == "build":
        script = config.REPO / "engine" / "gorge" / "build.sh"
        if not script.is_file():
            raise CliError(f"no {script}: `dz engine build` runs from a source checkout")
        env = dict(os.environ)
        target = a.dir or (load_config(a.config, a.set).engine.root if a.config else None) or env.get("GORGE_DIR")
        default = config.REPO.parent / "ext" / "gorge-dz"
        root = Path(target).expanduser().absolute() if target else default
        env["GORGE_DIR"] = str(root)
        print(f"dz engine build: bash {script} with GORGE_DIR={root}", flush=True)
        rc = subprocess.run(["bash", str(script)], env=env).returncode
        if rc == 0:
            base = load_config(None).engine.root if base_config().is_file() else None
            if base and Path(base) != root:
                print(f"dz engine build: configs/base/player.yml has engine.root {base}; point your config at this "
                      f"build (engine: {{root: {root}}}) or pass --set engine.root={root}")
        return rc
    if a.engine_cmd == "version":
        from draftzero.engine.gorge import Gorge
        if a.root:
            g = Gorge(Path(a.root), Path("."), Path("."))
        else:
            g = Gorge.from_config(load_config(a.config, a.set))
        v = g.version()
        print(json.dumps(v, indent=2, default=str))
        if not g.binary.is_file():
            print(f"dz engine version: no dzgorge at {g.binary}", file=sys.stderr)
            return 1
        return 0
    g = make_gorge(load_config(a.config, a.set))
    cov = g.coverage()
    if a.json:
        print(json.dumps(cov, indent=1))
        return 0
    print(f"{cov['supported_decks']} of {cov['decks']} decks in {g.pool} play only supported cards; "
          f"{cov['unsupported_cards']} of {cov['cards']} distinct cards unsupported; {cov['failed_decks']} decks "
          "failed to resolve")
    worst = sorted(cov["unsupported"].items(), key=lambda kv: -kv[1]["decks"])
    for card, info in worst[:a.top]:
        print(f"  {card}: in {info['decks']} decks, missing {info['missing']}")
    if len(worst) > a.top:
        print(f"  ... and {len(worst) - a.top} more (--json lists all)")
    return 0


# ------------------------------------------------------------------------------------------------ pods

def _bundle(a):
    from draftzero.workers import runpod as rp
    if a.no_bundle:
        return None
    if a.config:
        from draftzero.engine.gorge import Gorge
        g = Gorge.from_config(load_config(a.config, a.set))
        return rp.Bundle(engine_root=g.root, decks=g.decks, pool=g.pool)
    return rp.Bundle()


def cost_guard(spec, max_hours: float, balance: str) -> str:
    worst = spec.max_price * max_hours
    what = f"GPU {spec.gpu}" if spec.gpu else f"CPU {spec.cpu} x {spec.vcpu} vCPU"
    return (f"pod {spec.name}: {what} on {spec.cloud.title()} Cloud\n"
            f"  cost guard: at most ${spec.max_price:.3f}/h (a quote above it is refused, a pod billing above it is\n"
            f"  terminated) for at most {max_hours:g} h from its create, the wait for SSH included (it is armed to\n"
            f"  terminate itself then, even if this machine dies; one whose SSH is not up by then is removed): at\n"
            f"  most ${worst:.2f} in all, more only if RunPod's API refuses the terminate (the pod retries it for\n"
            f"  30 min). Balance {balance}. `dz pod down ID` stops the billing sooner.")


def cmd_pod(a) -> int:
    from draftzero.workers import runpod as rp
    api = rp.RunPodAPI()
    if a.pod_cmd == "balance":
        b = api.balance()
        print(f"balance ${float(b.get('clientBalance') or 0):.2f}, spending ${float(b.get('currentSpendPerHr') or 0):.3f}/h"
              f", spend limit {b.get('spendLimit')}")
        return 0
    if a.pod_cmd == "list":
        pods = api.list()
        if a.json:
            print(json.dumps([rp.brief(p) for p in pods], indent=1))
        elif not pods:
            print("no pods")
        for p in ([] if a.json else pods):
            b = rp.brief(p)
            print(f"{b['id']}  {b['name'] or '-':<16} {b['desiredStatus'] or '-':<10} ${b['costPerHr'] or 0}/h "
                  f"{b['vcpuCount'] or '?'} vCPU  {b['ssh'] or 'no ssh yet'}")
        return 0
    if a.pod_cmd == "offers":
        for x in api.offers(a.min_vcpu, max_price=a.max_price, community_only=not a.secure):
            print(f"{x['cloud']:9} {x['offer'][:30]:30} ${x['price']:<6} {x['vcpu']:>3} vCPU {x['ram'] or 0:>4} GB "
                  f"${x['usd_per_vcpu_hr']:.4f}/vCPU-h {x['stock']}")
        return 0
    if a.pod_cmd == "down":         # try every pod: one failure must not leave the later ones billing
        ids, failed = list(a.ids), []
        for name in a.name or []:
            try:
                ids += [p["id"] for p in api.find(name)]
            except Exception as e:                       # noqa: BLE001 - reported below
                failed.append(f"--name {name}")
                print(f"dz pod down: listing pods named {name!r} failed: {e}", file=sys.stderr)
        ids = list(dict.fromkeys(ids))
        if not ids and not failed:
            raise CliError("no pod to remove: give pod ids or --name NAME")
        for pid in ids:
            try:
                api.terminate(pid)
                print(f"terminated {pid}")
            except Exception as e:                       # noqa: BLE001 - reported below
                failed.append(pid)
                print(f"dz pod down: {pid}: {e}", file=sys.stderr)
        try:
            left = api.list()
            print(f"{len(left)} pod(s) left" + (": " + ", ".join(p.get("id", "?") for p in left) if left else ""))
        except Exception as e:                           # noqa: BLE001
            print(f"dz pod down: listing the pods left failed ({e}): check `dz pod list`", file=sys.stderr)
        if failed:
            print(f"dz pod down: FAILED for {', '.join(failed)}: those pods may still be billing; check `dz pod list`",
                  file=sys.stderr)
            return 1
        return 0
    # up
    spec = rp.PodSpec(name=a.name, max_price=a.max_price, gpu=a.gpu, cpu=a.cpu, vcpu=a.vcpu, min_vcpu=a.min_vcpu,
                      image=a.image or rp.IMAGE, datacenters=tuple(a.datacenter or ()),
                      cloud="SECURE" if a.secure else "COMMUNITY", allow_secure=a.secure)
    try:
        spec.validate()
    except ValueError as e:
        raise CliError(str(e)) from None
    if not (0 < a.max_hours <= rp.MAX_ARM_HOURS):
        raise CliError(f"--max-hours must be in (0, {rp.MAX_ARM_HOURS:g}]")
    files = _bundle(a)
    try:
        bal = f"${float(api.balance().get('clientBalance') or 0):.2f}"
    except Exception as e:                               # noqa: BLE001 - the create checks the balance again
        bal = f"unknown ({type(e).__name__})"
    print(cost_guard(spec, a.max_hours, bal), flush=True)
    if not a.yes:
        if not sys.stdin.isatty():
            raise CliError("not confirmed: pass --yes to rent the pod without a prompt")
        if input("Rent it? [y/N] ").strip().lower() not in ("y", "yes"):
            print("not rented")
            return 1
    pod = rp.up(api, spec, max_hours=a.max_hours, files=files, setup=not a.no_setup,
                keep_on_failure=a.keep_on_failure)
    print(json.dumps(rp.brief(pod.info) | {"id": pod.id, "ssh": pod.target.spec}, indent=1, default=str))
    print(f"pod {pod.id} is up and armed to terminate itself {a.max_hours:g} h after its create. Remove it with "
          f"`dz pod down {pod.id}`; "
          + ("its actors run from " + files.remote_repo + " (`dz relay` pulls their shards)" if files else
             "nothing was copied to it (--no-bundle)"))
    return 0


# ------------------------------------------------------------------------------------------------ relay

def cmd_relay(a) -> int:
    from draftzero import config
    from draftzero.loop import relay
    cfg = config.load(a.config)
    store, upstream = a.store, a.upstream
    if store is None:
        if _is_remote(cfg.store):
            raise CliError(f"the config's store is {cfg.store}: give the relay a local folder of its own "
                           "(--store DIR); the config's store becomes its upstream")
        store = cfg.store
    elif upstream is None and _is_remote(cfg.store):
        upstream = cfg.store
    if _is_remote(store):
        raise CliError(f"--store {store}: the relay's store is a local folder")
    argv = ["--store", str(store), "--interval", str(a.interval)]
    for h in a.host or []:
        argv += ["--host", *h]
    if a.hosts:
        argv += ["--hosts", a.hosts]
    if upstream:
        argv += ["--upstream", upstream]
    if a.once:
        argv.append("--once")
    return relay.main(argv)


# ------------------------------------------------------------------------------------------------ parser

def _add_config(p: argparse.ArgumentParser, default_note: str = "default configs/base/player.yml") -> None:
    p.add_argument("--config", default=None, help=f"run config for the engine, deck pool and search ({default_note})")
    p.add_argument("--set", action="append", default=[], metavar="PATH=VALUE",
                   help="override a config field (repeatable), e.g. --set engine.root=/path/to/gorge")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dz", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    def add(name, help, **kw):
        p = sub.add_parser(name, help=help, description=kw.pop("description", help), **kw)
        return p

    add("version", "print the installed DraftZero version").set_defaults(func=cmd_version)
    for name in ("manifest", "provenance"):
        p = add(name, "print what defines a run here: code, engine build, card lock, machine"
                      + ("" if name == "manifest" else " (= dz manifest)"),
                description="Print a run manifest (draftzero.manifest.collect): DraftZero's version and commit, the "
                            "gorge pin, the engine build and card lock, python/torch/numpy and the machine; with "
                            "CONFIG also the config, its hash and the :closedlist belief corpus.")
        p.add_argument("config", nargs="?", default=None, metavar="CONFIG")
        p.add_argument("--set", action="append", default=[], metavar="PATH=VALUE", help="override a config field")
        p.set_defaults(func=cmd_manifest)

    p = add("run", "start every role of a run on this machine (local mode)",
            description="Start the controller, the learner, one evaluator and N actors on this machine against one "
                        "store (python -m draftzero.loop.cli local). Returns when the run stops: 0 at its budget or "
                        "`dz stop`, 1 when a role failed, 2 at --max-hours, 130 on Ctrl-C, 143 on SIGTERM, 129 on "
                        "SIGHUP (the roles are stopped either way). Starting it again on the same store resumes the "
                        "run, on the store's config.yml plus events.jsonl: CONFIG (with --set) may then differ from "
                        "it only in machine paths, unless --allow-config-change.")
    p.add_argument("config", metavar="CONFIG")
    p.add_argument("--store", default=None, help="a folder or hf://buckets/<owner>/<bucket> (default: the config's)")
    p.add_argument("--actors", type=int, default=1, help="actor processes (default 1)")
    p.add_argument("--max-hours", type=float, default=None, help="stop the run after this many hours")
    p.add_argument("--work-dir", default=None, help="local caches and role logs (default runs/<run>/work)")
    p.add_argument("--machine", default=None, help="prefix of this machine's role names (default local)")
    p.add_argument("--poll", type=float, default=None, help="seconds between the roles' looks at the store "
                                                              f"(default {LOCAL_POLL_S:g} on a folder, 60 on a bucket)")
    p.add_argument("--set", action="append", default=[], metavar="PATH=VALUE",
                   help="override a config field at launch (repeatable); the store's config.yml records it")
    from draftzero.loop.cli import ALLOW_HELP
    p.add_argument("--allow-config-change", action="store_true", help=ALLOW_HELP)
    p.add_argument("--no-check", action="store_true", help="skip the engine and deck-pool checks")
    p.set_defaults(func=cmd_run)

    from draftzero.loop.cli import HELP as LOOP_HELP
    for name in LOOP_COMMANDS:          # parsed by draftzero.loop.cli (main's shortcut); listed here for --help
        p = add(name, LOOP_HELP[name], add_help=False)
        p.add_argument("rest", nargs=argparse.REMAINDER)
        p.set_defaults(func=cmd_loop)

    p = add("report", "draw a run's standard figures (needs the plots extra: matplotlib)",
            description="Draw learning curves, strength by version, ratings and throughput from a run's metrics "
                        "into PNGs ({learning,strength,ratings,throughput}-{light,dark}.png).")
    p.add_argument("src", metavar="STORE|RUN_DIR", help="a store (folder or hf://buckets/...), a run folder or a "
                                                         "metrics .jsonl")
    p.add_argument("--out", default=None, help="output folder (default <STORE>/report, or ./report)")
    p.add_argument("--theme", choices=("light", "dark", "both"), default="both")
    p.add_argument("--anchor", default="sbtactical", help="the player fixed at 0 Elo")
    p.set_defaults(func=cmd_report)

    p = add("models", "published models: dz models pull hf://...")
    msub = p.add_subparsers(dest="models_cmd", required=True, metavar="pull")
    q = msub.add_parser("pull", help="download model files into the local Hugging Face cache and print their paths",
                        description="Download hf://<owner>/<repo>[@<revision>]/<path> files (a repo without a path: "
                                    "the whole repo) and print the local paths. Private repos need HF_TOKEN or "
                                    "`hf auth login`.")
    q.add_argument("uri", nargs="+", metavar="URI")
    q.add_argument("--cache-dir", default=None)
    p.set_defaults(func=cmd_models)

    p = add("play", "paired games between two policy specs, and A's score",
            description="Play paired games between two gorge policy specs (dzgorge play; each pair twice with the "
                        "seats swapped, once when A == B) and print A's paired score. Specs as in "
                        "draftzero.engine.specs, e.g. 'az:sims=100:autopay:oppnodes:oppfull:morekinds:closedlist:"
                        "topk=6:gonet=/path/model.dzgw', sbtactical, bot, random.")
    p.add_argument("--a", required=True, metavar="SPEC", help="policy A (the one scored)")
    p.add_argument("--b", required=True, metavar="SPEC", help="policy B")
    p.add_argument("--pairs", type=int, required=True, help="deck pairs")
    p.add_argument("--sims", type=int, default=None, help="set the simulations of both search specs")
    p.add_argument("--split", default=None, help="deck pool split (default: the config's eval split)")
    p.add_argument("--seed", type=int, default=None, help="default: the config's eval.seed")
    p.add_argument("--workers", type=int, default=None, help="games in parallel (default: the config's eval.workers)")
    p.add_argument("--timeout", type=float, default=None, help="seconds before a hung run is killed "
                                                               "(default: actor.chunk_secs)")
    p.add_argument("--out", default=None, help="keep the games here (default: a temporary folder)")
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    _add_config(p)
    p.set_defaults(func=cmd_play)

    p = add("eval", "a network against opponents in paired games: a table and JSON",
            description="Play MODEL against each opponent in paired games on the eval split (the run evaluator's "
                        "protocol: no exploration, :closedlist and the search keys of the config) and print a table "
                        "and the JSON summary. MODEL: a .dzgw file, an hf://<owner>/<repo>[@rev]/<file>.dzgw, or a "
                        "run's store STORE[@vN] (default its latest version; its config.yml supplies the search and "
                        "engine unless --config is given). Opponents: prev, v<N> (with a store), nonet (the same "
                        "search without a network), sbtactical, bot, random, file:<path>.dzgw or a .dzgw path, or "
                        "any policy spec.")
    p.add_argument("model", metavar="MODEL")
    p.add_argument("--opponents", required=True, help="comma-separated, e.g. sbtactical,nonet")
    p.add_argument("--pairs", type=int, default=None, help="pairs per opponent (default: the config's eval.pairs)")
    p.add_argument("--sims", type=int, default=None, help="simulations (default: the config's eval.sims)")
    p.add_argument("--seed", type=int, default=None, help="default: the config's eval.seed")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--out", default=None, help="keep games and summary.json here (default: a temporary folder)")
    p.add_argument("--json", action="store_true", help="print only the JSON summary")
    _add_config(p, "default: the store's config.yml, else configs/base/player.yml")
    p.set_defaults(func=cmd_eval)

    p = add("engine", "the gorge engine: build, version, coverage")
    esub = p.add_subparsers(dest="engine_cmd", required=True, metavar="build|version|coverage")
    q = esub.add_parser("build", help="build dzgorge at engine/gorge/GORGE_REF (needs git, make, Go 1.25+)",
                        description="Run engine/gorge/build.sh: clone or fetch the fork at GORGE_REF into the gorge "
                                    "folder, fetch and compile the card corpus if missing, build bin/dzgorge and "
                                    "write bin/dzgorge.build.json. The folder: --dir, else --config's engine.root, "
                                    "else $GORGE_DIR, else ext/gorge-dz next to the repo.")
    q.add_argument("--dir", default=None, help="the gorge checkout to build in")
    _add_config(q, "its engine.root is the folder")
    q = esub.add_parser("version", help="print the engine's build record (fork commit, binary SHA-256, card lock)")
    q.add_argument("--root", default=None, help="a gorge checkout (instead of the config's engine.root)")
    _add_config(q)
    q = esub.add_parser("coverage", help="which decks of the pool play only supported cards (dzgorge coverage)")
    q.add_argument("--json", action="store_true", help="the whole record as JSON")
    q.add_argument("--top", type=int, default=20, help="unsupported cards to list (default 20)")
    _add_config(q)
    p.set_defaults(func=cmd_engine)

    p = add("pod", "RunPod pods for relay actors: up, down, list, balance, offers",
            description="Rent and remove RunPod pods (draftzero.workers.runpod). The API key is read from "
                        "/root/.runpod/env or ~/.runpod/env (RUNPOD_API_KEY). Pods bill until `dz pod down`.")
    psub = p.add_subparsers(dest="pod_cmd", required=True, metavar="up|down|list|balance|offers")
    q = psub.add_parser("up", help="rent a pod, arm its self-destruct, copy the engine, decks and repo, bootstrap it",
                        description="Create a pod, wait for SSH, arm it to terminate itself --max-hours after its "
                                    "create, then (unless --no-bundle) copy dzgorge, the card corpus, the deck pool "
                                    "and the repo and bootstrap a venv. Any failure after the create removes the pod. "
                                    "Prints the cost guard and asks before renting (--yes skips the question).")
    q.add_argument("--name", required=True, help="pod name (unique: an existing name is refused)")
    g = q.add_mutually_exclusive_group(required=True)
    g.add_argument("--gpu", default=None, help="GPU type id from `dz pod offers` (bought for its vCPUs)")
    g.add_argument("--cpu", default=None, help="CPU flavor (cpu3c, cpu5c, ...): Secure Cloud only, needs --secure "
                                               "and --datacenter")
    q.add_argument("--max-price", type=float, required=True, help="$/h cap: required")
    q.add_argument("--max-hours", type=float, required=True,
                   help="the pod's whole life, from its create: it terminates itself then (required)")
    q.add_argument("--min-vcpu", type=int, default=8, help="GPU pods: vCPU floor (default 8)")
    q.add_argument("--vcpu", type=int, default=8, help="CPU pods: vCPUs (default 8)")
    q.add_argument("--datacenter", action="append", default=[], help="pin to a data center (repeatable)")
    q.add_argument("--secure", action="store_true", help="allow Secure Cloud (costs more; CPU pods need it)")
    q.add_argument("--image", default=None, help="container image (default RunPod's base image)")
    q.add_argument("--no-bundle", action="store_true", help="only the armed pod: copy nothing")
    q.add_argument("--no-setup", action="store_true", help="copy the bundle but skip the bootstrap")
    q.add_argument("--keep-on-failure", action="store_true",
                   help="keep the pod when a step after arming it fails (one not armed yet is always removed)")
    q.add_argument("--yes", action="store_true", help="rent without asking")
    _add_config(q, "its engine.root and deck pool are copied; default /work/gorge-fork and data/gorge")
    q = psub.add_parser("down", help="terminate pods (billing stops only here)",
                        description="Terminate every pod given (by id or --name), even when one fails, then list the "
                                    "pods left. Exit 1 when any terminate failed (those pods may still be billing).")
    q.add_argument("ids", nargs="*", metavar="ID")
    q.add_argument("--name", action="append", default=[], help="every pod with this name (repeatable)")
    q = psub.add_parser("list", help="the account's pods")
    q.add_argument("--json", action="store_true")
    psub.add_parser("balance", help="the account balance and current spend")
    q = psub.add_parser("offers", help="in-stock GPU pods, most vCPU per dollar first")
    q.add_argument("--min-vcpu", type=int, default=8)
    q.add_argument("--max-price", type=float, default=None)
    q.add_argument("--secure", action="store_true", help="also list Secure Cloud")
    p.set_defaults(func=cmd_pod)

    p = add("relay", "move shards and versions between the run's store and actor hosts over rsync",
            description="Run the relay (draftzero.loop.relay) for CONFIG's run: push config.yml, control.json and "
                        "the latest versions to each host's <ROOT>/store and pull complete shards, heartbeats and "
                        "metrics back, every --interval seconds until the run stops. The store is the config's "
                        "(a local folder); when that is an hf://buckets/ store, give the relay a folder with "
                        "--store and the bucket becomes its upstream.")
    p.add_argument("config", metavar="CONFIG")
    p.add_argument("--host", nargs="+", action="append", default=[], metavar="NAME ROOT [SSH]",
                   help='one actor host (repeatable): NAME ROOT ["user@host -p PORT"]; no SSH: ROOT is a local path')
    p.add_argument("--hosts", default=None, help="a JSON or YAML list of {name, root, ssh} instead of --host")
    p.add_argument("--store", default=None, help="the relay's local store folder (default: the config's store)")
    p.add_argument("--upstream", default=None, help="the run's real store when --store is the relay's mirror")
    p.add_argument("--interval", type=float, default=60, help="seconds between syncs (default 60)")
    p.add_argument("--once", action="store_true", help="one sync, then exit")
    p.set_defaults(func=cmd_relay)
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in LOOP_COMMANDS:
        from draftzero.loop import cli as loop_cli
        return loop_cli.main(argv, prog="dz")
    a = parser().parse_args(argv)
    try:
        return int(a.func(a) or 0)
    except CliError as e:
        print(f"dz {a.cmd}: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(f"dz {a.cmd}: interrupted", file=sys.stderr)
        return 130
    except Exception as e:                               # noqa: BLE001 - known user errors without a traceback
        if isinstance(e, _user_errors()):
            print(f"dz {a.cmd}: {e}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
