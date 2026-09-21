"""
loop.py — train one network on the whole FDN format (see README.md).

Every game draws both decks at random from the training pool (~28k top-player decks). Each
generation:

  1. play      gen 0: heuristic search (no network) on both sides.
               gen 1+: ~70% self-play (both sides use the current net, both sides are training
               data), ~20% vs a random older checkpoint and ~10% vs gen 0's net (only the current
               net's side is training data). Games run in chunks, so the dashboard refreshes as
               they finish rather than once per generation.
  2. train     on the newest `replay_states` states (older shards move to archive/).
  3. eval      the new checkpoint plays paired games on held-out eval-pool decks against fixed
               baselines: deck X vs Y, then Y vs X, the agent always player A.

Generations are open-ended (`generations: null`): stop the process whenever. `--resume` continues
the newest unfinished run, redoing the interrupted stage.

  PYTHONPATH=src python -m draftzero.loop --config configs/laptop.yml
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from draftzero import dashboard
from draftzero import paths as dzpaths
from draftzero import provenance
from draftzero import stats as gstats
from magezero import metrics
from magezero.resources import ResourceMonitor
from magezero.runner import (OPPONENT_PORT, PRIMARY_PORT, TMP_DIR, launch_jvm, refresh_dashboard, run_test,
                             run_train, start_server, stop_server)
from magezero.util.config import GenSettings, load_curriculum, resolve_gen

RUNS_DIR = Path("runs")

DEFAULTS = {
    "model": "FDN_generalist",
    "version": 1,
    "target_games": None,   # stop once this many games have been played (null = open-ended)
    "base_game_yml": "configs/game.yml",
    "curriculum": "configs/curriculum.yml",
    # Pool files hold deck STEMS; deck_root says where the .dck files actually live on
    # this machine. Override with MZ_DECK_DIR so one config works on every worker.
    "deck_root": "data/decks",
    "pools": {"train": "data/pools/train.txt",
              "eval": "data/pools/eval.txt",
              "meta": "assets/decks.tsv"},
    "reference": "assets/reference/FDN_gih.json",
    "generations": None,          # None: run until stopped
    "bootstrap_games": 96,        # gen 0, heuristic search on both sides
    "games_per_gen": 48,
    "chunk_games": 12,            # games per JVM run; metrics refresh after each
    "mix": {"self": 0.7, "past": 0.2, "gen0": 0.1},
    "replay_states": 40000,
    "epochs_bootstrap": 2,
    "epochs": 1,
    "eval": {"baselines": ["offline", "minimax"], "pairs": 1, "window_gens": 6},
    "stats": {"window_gens": 10, "min_card_games": 15},
    "jvm": {"heap": "8g", "threads": 3, "search_budget": 200, "timeout_ms": 8000, "max_minutes": 30},
}


def load_config(path: str) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for k, v in raw.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


# ── run state ────────────────────────────────────────────────

class Run:
    def __init__(self, cfg: dict, run_dir: Path):
        self.cfg, self.dir = cfg, run_dir
        self.model, self.version = cfg["model"], cfg["version"]
        self.data = Path("data") / self.model / f"ver{self.version}"
        self.models = Path("models") / self.model / f"ver{self.version}"
        self.meta = gstats.load_deck_meta(Path(cfg["pools"]["meta"]))
        self.deck_root = Path(os.environ.get("MZ_DECK_DIR") or cfg["deck_root"])
        # the JVM needs real paths; the pool files on disk stay machine-independent
        self.eval_pool = [str(dzpaths.deck_path(s, self.deck_root))
                          for s in dzpaths.read_pool(Path(cfg["pools"]["eval"]))]
        self.train_pool_file = str(dzpaths.resolve_pool(
            dzpaths.read_pool(Path(cfg["pools"]["train"])), self.deck_root,
            run_dir / ".pools" / "train.txt"))

    @property
    def state(self) -> dict:
        return json.loads((self.dir / "run.json").read_text())

    def update(self, **fields) -> dict:
        s = self.state
        s.update(fields)
        (self.dir / "run.json").write_text(json.dumps(s, indent=2))
        return s

    def next_session(self) -> int:
        sid = self.state.get("next_session", 0)
        self.update(next_session=sid + 1)
        return sid

    def has_checkpoint(self, name: str = "model") -> bool:
        return (self.models / f"{name}.pt.gz").exists()


def find_active(model: str) -> Optional[Path]:
    if not RUNS_DIR.exists():
        return None
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        f = d / "run.json"
        if f.exists():
            r = json.loads(f.read_text())
            if r.get("generalist") and r["primary"]["deck"] == model \
                    and not r.get("completed_at") and not r.get("abandoned_at"):
                return d
    return None


def create_run(cfg: dict) -> Path:
    run_dir = RUNS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True)
    state = {
        "generalist": True,
        "created_at": datetime.now().isoformat(),
        "primary": {"deck": cfg["model"], "version": cfg["version"]},
        "opponents": [],
        "run_config_snapshot": {**cfg, "generations": cfg["generations"] or "open-ended",
                                "games_per_gen": cfg["games_per_gen"]},
        "env": {k: v for k, v in os.environ.items() if k.startswith("MZ_") or k.startswith("PYTORCH_")},
        "provenance": provenance.collect(),
        "current_gen": 0, "stage": "play", "next_session": 0, "gens": {},
    }
    (run_dir / "run.json").write_text(json.dumps(state, indent=2))
    return run_dir


# ── game.yml ─────────────────────────────────────────────────

def _player(p: dict, pool: str, mode: str, out: Path, ptype: str, offline: bool, settings: GenSettings, jvm: dict):
    p["deckPath"] = ""
    p["deck_pool"] = str(Path(pool).resolve())
    p["deck_pool_mode"] = mode
    p["output_file"] = str(out.resolve())
    p["type"] = ptype
    p["mcts"]["offline_mode"] = offline
    p["mcts"]["td_discount"] = settings.td_discount
    p["mcts"]["search_budget"] = jvm["search_budget"]
    p["mcts"]["timeout_ms"] = jvm["timeout_ms"]
    p["priors"]["prior_temperature"] = settings.prior_temperature
    for head in ("binary", "priority", "target", "opponent"):
        p["priors"][head] = bool(getattr(settings.priors, head)) and not offline


def game_yml(run: Run, name: str, settings: GenSettings, games: int, *,
             pool_a: str, pool_b: str, mode: str,
             out_a: Path, out_b: Path, offline_a: bool, offline_b: bool,
             type_b: str = "mcts", b_port: int = PRIMARY_PORT) -> str:
    cfg = yaml.safe_load(Path(run.cfg["base_game_yml"]).read_text())
    jvm = run.cfg["jvm"]
    _player(cfg["player_a"], pool_a, mode, out_a, "mcts", offline_a, settings, jvm)
    _player(cfg["player_b"], pool_b, mode, out_b, type_b, offline_b, settings, jvm)
    cfg["training"]["games"] = games
    cfg["training"]["threads"] = min(jvm["threads"], games)
    cfg["training"]["max_minutes"] = jvm["max_minutes"]
    cfg["server"]["port"] = PRIMARY_PORT
    cfg["server"]["opponent_port"] = b_port   # self-play: both players query the same server
    out = TMP_DIR / f"game_{name}.yml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(out)


# ── one JVM job ──────────────────────────────────────────────

def play_job(run: Run, gen: int, name: str, yml: str, kind: str, opponent: str, agent_side: str) -> list[dict]:
    """Run one JVM; append its games to games.jsonl. Returns the parsed per-game behavior records."""
    log_path = run.dir / f"jvm_gen{gen}_{name}.log"
    t0 = time.time()
    try:
        launch_jvm(yml, log_path, run.cfg["jvm"]["heap"])
    except Exception as e:
        print(f"[gen {gen}] {name} FAILED: {e}")
    summaries = gstats.parse_summaries(log_path)
    for s in summaries:
        metrics.append_jsonl(run.dir / "games.jsonl", {
            "kind": kind, "gen": gen, "job": name, "opponent": opponent, "agent_side": agent_side,
            "deck_a": s["deck_a"], "deck_b": s["deck_b"],
            "colors_a": run.meta.get(s["deck_a"], {}).get("main_colors"),
            "colors_b": run.meta.get(s["deck_b"], {}).get("main_colors"),
            "first": s.get("first"), "winner": s.get("winner"), "turns": s.get("turns"),
            "drawn_a": s.get("drawn_a"), "drawn_b": s.get("drawn_b"), "seed": s.get("seed"),
        })
    try:
        parsed = metrics.parse_jvm_log(str(log_path), "A", "B", os.devnull, os.devnull)
        records = parsed["games"]
    except Exception as e:
        print(f"[metrics] WARNING could not parse {log_path}: {e}")
        records = []
    wins = sum(s.get("winner") == "A" for s in summaries)
    print(f"[gen {gen}] {name}: {len(summaries)} games in {(time.time() - t0) / 60:.1f} min, A won {wins}")
    refresh(run)
    return [{**r, "_seconds": time.time() - t0} for r in records]


def refresh(run: Run) -> None:
    try:
        rows = [json.loads(l) for l in (run.dir / "games.jsonl").read_text().splitlines() if l.strip()] \
            if (run.dir / "games.jsonl").exists() else []
        recs = gstats.deck_records(rows)
        with open(run.dir / "deck_records.tsv", "w") as f:
            f.write("deck\tmain_colors\tgames\twins\n")
            for d, (n, w) in sorted(recs.items()):
                f.write(f"{d}\t{run.meta.get(d, {}).get('main_colors', '?')}\t{n}\t{w}\n")
    except Exception as e:
        print(f"[metrics] WARNING deck records failed: {e}")
    refresh_dashboard(run.dir)          # MageZero: generic training health
    try:
        dashboard.render(run.dir)       # DraftZero: format knowledge
    except Exception as e:
        # a rendering problem must never take down a run that is generating games fine
        print(f"[metrics] WARNING format dashboard failed: {e}")


# ── play ─────────────────────────────────────────────────────

def plan_games(run: Run, gen: int, rng: random.Random) -> list[dict]:
    """Jobs for one generation: dicts with kind, opponent (label), checkpoint for player B, games."""
    cfg = run.cfg
    chunk = cfg["chunk_games"]

    def chunks(total, **job):
        out = []
        while total > 0:
            out.append({**job, "games": min(chunk, total)})
            total -= chunk
        return out

    if gen == 0 or not run.has_checkpoint():
        return chunks(cfg["bootstrap_games"], opponent="heuristic", checkpoint=None)
    n = cfg["games_per_gen"]
    mix = cfg["mix"]
    # gen N plays with the net trained at gen N-1; "older" means gens 1..N-2 (gen 0 has its own share)
    older = [g for g in range(1, gen - 1) if run.has_checkpoint(f"gen{g}")]
    n_gen0 = round(n * mix["gen0"]) if gen >= 2 and run.has_checkpoint("gen0") else 0
    n_past = round(n * mix["past"]) if older else 0
    if not older and n_gen0:
        n_gen0 = round(n * (mix["gen0"] + mix["past"]))
    jobs = chunks(n - n_past - n_gen0, opponent="self", checkpoint=None)
    if n_past:
        k = rng.choice(older)
        jobs += chunks(n_past, opponent=f"gen{k}", checkpoint=f"gen{k}")
    if n_gen0:
        jobs += chunks(n_gen0, opponent="gen0", checkpoint="gen0")
    return jobs


def play_generation(run: Run, gen: int, settings: GenSettings) -> dict:
    rng = random.Random(f"{run.dir.name}-{gen}")
    jobs = plan_games(run, gen, rng)
    testing = run.data / "testing"
    scratch = run.dir / "scratch"
    testing.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    offline = jobs[0]["opponent"] == "heuristic"
    print(f"[gen {gen}] plan: " + ", ".join(f"{j['games']}×{j['opponent']}" for j in jobs))

    selfplay_records, league = [], []
    t0 = time.time()
    server = None
    try:
        if not offline:
            server = start_server(run.model, run.version, PRIMARY_PORT, run.dir)
        for i, job in enumerate(jobs):
            sid = run.next_session()
            both = job["checkpoint"] is None
            out_a = testing / f"session{sid}_A_{job['opponent']}.hdf5"
            out_b = (testing if both else scratch) / f"session{sid}_B_{job['opponent']}.hdf5"
            name = f"{i:02d}_{job['opponent']}"
            yml = game_yml(run, f"gen{gen}_{name}", settings, job["games"],
                           pool_a=run.train_pool_file, pool_b=run.train_pool_file, mode="random",
                           out_a=out_a, out_b=out_b, offline_a=offline, offline_b=offline,
                           b_port=PRIMARY_PORT if both else OPPONENT_PORT)
            frozen = None
            try:
                if not both:
                    frozen = start_server(run.model, run.version, OPPONENT_PORT, run.dir, checkpoint=job["checkpoint"])
                recs = play_job(run, gen, name, yml, "selfplay", job["opponent"], "both" if both else "a")
            finally:
                if frozen:
                    stop_server(frozen)
            (selfplay_records if both else league).extend(recs)
            if not both:
                for f in scratch.glob(f"session{sid}_B_*"):
                    f.unlink()
    finally:
        if server:
            stop_server(server)

    # one behavior / environment row per generation, over the games where both sides are the agent
    hours = (time.time() - t0) / 3600
    if selfplay_records:
        row = {"kind": "selfplay", "gen": gen, "primary_offline": offline,
               **metrics.summarize_games({"deck_a": run.model, "deck_b": "B side", "games": selfplay_records,
                                          "summary": {"games_completed": len(selfplay_records)}})}
        row["games_per_hour"] = (len(selfplay_records) + len(league)) / hours if hours else None
        metrics.append_jsonl(run.dir / "metrics.jsonl", row)
    games = read_games(run)
    if not games:
        print(f"[gen {gen}] WARNING: no games recorded this generation - check the jvm log")
    for opp in sorted({g["opponent"] for g in games if g["gen"] == gen and g["agent_side"] == "a"}):
        # all older checkpoints share one series; gen 0 has its own
        label = "gen0" if opp == "gen0" else "older"
        gg = [g for g in games if g["gen"] == gen and g["opponent"] == opp and g["agent_side"] == "a"]
        wins = sum(g["winner"] == "A" for g in gg)
        p, lo, hi = metrics.wilson(wins, len(gg))
        metrics.append_jsonl(run.dir / "metrics.jsonl", {"kind": "league", "gen": gen, "baseline": label,
                                                         "opponent": opp, "games": len(gg), "a_winrate": p,
                                                         "a_winrate_lo": lo, "a_winrate_hi": hi})
    return {"play_seconds": time.time() - t0, "jobs": jobs}


# ── train ────────────────────────────────────────────────────

def _states(path: Path) -> int:
    import h5py
    try:
        with h5py.File(path, "r") as f:
            return int(f["/row"].shape[0])
    except Exception:
        return 0


def train_generation(run: Run, gen: int) -> dict:
    t0 = time.time()
    testing, training, archive = (run.data / s for s in ("testing", "training", "archive"))
    new_files = sorted(testing.glob("*.hdf5"))
    metrics.append_jsonl(run.dir / "metrics.jsonl", {"kind": "dataset", "gen": gen, "deck": run.model,
                                                     **metrics.dataset_stats([str(p) for p in new_files])})
    if run.has_checkpoint():
        run.update(stage="eval_prev")
        run_test(run.model, run.version, run.dir, gen)
    training.mkdir(parents=True, exist_ok=True)
    for f in new_files:
        shutil.move(str(f), str(training / f.name))

    # replay window: newest shards first until replay_states is reached; the rest go to archive/
    shards = sorted(training.glob("*.hdf5"), key=lambda p: int(p.name.split("_")[0][7:]), reverse=True)
    kept = 0
    archive.mkdir(parents=True, exist_ok=True)
    for f in shards:
        if kept >= run.cfg["replay_states"]:
            shutil.move(str(f), str(archive / f.name))
        else:
            kept += _states(f)
    run.update(stage="train")
    use_ckpt = run.has_checkpoint()
    run_train(run.model, run.version, run.cfg["epochs"] if use_ckpt else run.cfg["epochs_bootstrap"],
              use_checkpoint=use_ckpt, run_dir=run.dir, gen=gen, dense_vocab=True)
    return {"train_seconds": time.time() - t0, "replay_states": kept}


# ── eval ─────────────────────────────────────────────────────

def eval_generation(run: Run, gen: int, settings: GenSettings) -> dict:
    """Paired games on held-out decks: for each pair (X, Y) the agent plays X vs Y, then Y vs X."""
    t0 = time.time()
    ev = run.cfg["eval"]
    scratch = run.dir / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    server = start_server(run.model, run.version, PRIMARY_PORT, run.dir, checkpoint=f"gen{gen}")
    try:
        for baseline in ev["baselines"]:
            rng = random.Random(f"eval-{gen}-{baseline}")
            pool_a, pool_b = [], []
            for _ in range(ev["pairs"]):
                x, y = rng.sample(run.eval_pool, 2)
                pool_a += [x, y]
                pool_b += [y, x]
            fa, fb = TMP_DIR / f"eval_{baseline}_A.txt", TMP_DIR / f"eval_{baseline}_B.txt"
            fa.parent.mkdir(parents=True, exist_ok=True)
            fa.write_text("\n".join(pool_a) + "\n")
            fb.write_text("\n".join(pool_b) + "\n")
            name = f"eval_{baseline}"
            yml = game_yml(run, f"gen{gen}_{name}", settings, len(pool_a),
                           pool_a=str(fa), pool_b=str(fb), mode="sequential",
                           out_a=scratch / f"{name}_A.hdf5", out_b=scratch / f"{name}_B.hdf5",
                           offline_a=False, offline_b=True,
                           type_b="minimax" if baseline == "minimax" else "mcts", b_port=OPPONENT_PORT)
            play_job(run, gen, name, yml, "strength_eval", baseline, "a")
    finally:
        stop_server(server)
        shutil.rmtree(scratch, ignore_errors=True)

    # rolling window: a few games per gen are too noisy alone
    games = read_games(run)
    lo_gen = gen - ev["window_gens"] + 1
    for baseline in ev["baselines"]:
        this = [g for g in games if g["kind"] == "strength_eval" and g["opponent"] == baseline and g["gen"] == gen]
        win = [g for g in games if g["kind"] == "strength_eval" and g["opponent"] == baseline and lo_gen <= g["gen"] <= gen]
        wins = sum(g["winner"] == "A" for g in win)
        p, lo, hi = metrics.wilson(wins, len(win))
        metrics.append_jsonl(run.dir / "metrics.jsonl", {
            "kind": "strength_eval", "gen": gen, "agent": run.model, "agent_checkpoint": f"gen{gen}",
            "baseline": baseline, "window_gens": ev["window_gens"], "window_games": len(win),
            "a_winrate": p, "a_winrate_lo": lo, "a_winrate_hi": hi,
            "gen_games": len(this), "gen_wins": sum(g["winner"] == "A" for g in this)})
        print(f"[eval] gen {gen} vs {baseline}: {sum(g['winner'] == 'A' for g in this)}/{len(this)} this gen, "
              f"{wins}/{len(win)} over the last {ev['window_gens']} gens")
    return {"eval_seconds": time.time() - t0}


# ── main loop ────────────────────────────────────────────────

STAGES = ["play", "train", "eval"]


def read_games(run: "Run") -> list[dict]:
    """Every game recorded so far. Empty when the JVM produced nothing (bad deck paths,
    a crashed generation), which must not take the whole run down with a traceback."""
    f = run.dir / "games.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def games_played(run: "Run") -> int:
    """Total games recorded so far, across every generation of this run."""
    return len(read_games(run))


def main(cfg: dict, resume: Optional[bool]) -> None:
    active = find_active(cfg["model"])
    if active and resume is None:
        resume = input(f"Active run found: {active.name}. Resume? [Y/n] ").strip().lower() in ("", "y", "yes")
    if active and resume:
        run = Run(cfg, active)
        gen, stage = run.state["current_gen"], run.state["stage"]
        if stage not in STAGES:
            stage = "play"
        if stage == "eval_prev":
            stage = "train"
        print(f"[run] resuming {active.name} at gen {gen}, stage {stage}")
        if stage == "play":
            # redo the generation's games: drop its half-written data and logged games
            for f in (run.data / "testing").glob("*.hdf5"):
                f.unlink()
            gpath = run.dir / "games.jsonl"
            if gpath.exists():
                keep = [l for l in gpath.read_text().splitlines() if l.strip() and json.loads(l)["gen"] < gen]
                gpath.write_text("".join(l + "\n" for l in keep))
    else:
        if active:
            json_path = active / "run.json"
            s = json.loads(json_path.read_text())
            s["abandoned_at"] = datetime.now().isoformat()
            json_path.write_text(json.dumps(s, indent=2))
        run = Run(cfg, create_run(cfg))
        gen, stage = 0, "play"
    print(f"[run] {run.dir}")
    curriculum = load_curriculum(cfg["curriculum"])

    resmon = ResourceMonitor(interval=float(cfg.get("resource_interval_s", 15))).start()

    while True:
        if cfg["generations"] is not None and gen >= cfg["generations"]:
            break
        target = cfg.get("target_games")
        if target:
            played = games_played(run)
            if played >= target:
                print(f"[run] target reached: {played} games >= {target}")
                break
            print(f"[run] {played}/{target} games played")
        print(f"\n========== GEN {gen} ==========")
        settings = resolve_gen(curriculum, gen)
        record = run.state["gens"].get(str(gen), {})
        record.update(settings={"td_discount": settings.td_discount, "prior_temperature": settings.prior_temperature,
                                "priors": vars(settings.priors)})
        t0 = time.time()
        if stage == "play":
            run.update(current_gen=gen, stage="play")
            record.update(play_generation(run, gen, settings))
            stage = "train"
        if stage == "train":
            run.update(current_gen=gen, stage="train")
            record.update(train_generation(run, gen))
            stage = "eval"
        if stage == "eval":
            run.update(current_gen=gen, stage="eval")
            if run.cfg["eval"]["baselines"]:
                record.update(eval_generation(run, gen, settings))
        record["gen_seconds"] = record.get("gen_seconds", 0) + time.time() - t0
        record["completed_at"] = datetime.now().isoformat()
        gens = run.state["gens"]
        gens[str(gen)] = record
        run.update(gens=gens, current_gen=gen + 1, stage="play")
        metrics.append_jsonl(run.dir / "metrics.jsonl", {
            "kind": "gen_timing", "gen": gen, "gen_seconds": record["gen_seconds"],
            "generate_seconds": record.get("play_seconds"), "eval_seconds": record.get("eval_seconds")})
        metrics.append_jsonl(run.dir / "metrics.jsonl",
                             {"kind": "resources", "gen": gen, **resmon.summary()})
        refresh(run)
        gen, stage = gen + 1, "play"

    resmon.stop()
    run.update(completed_at=datetime.now().isoformat(), stage="done",
               games_played=games_played(run))
    print(f"\n✓ Run complete: {run.dir.name} ({games_played(run)} games)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/laptop.yml")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--resume", dest="resume", action="store_true", default=None)
    g.add_argument("--fresh", dest="resume", action="store_false")
    args = ap.parse_args()
    main(load_config(args.config), args.resume)
