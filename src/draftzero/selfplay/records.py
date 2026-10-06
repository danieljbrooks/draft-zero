"""Game batches (docs/021 §3.2): a worker's finished games, one gzipped JSON-lines file per upload.

A worker plays a chunk of games with tools/imitation_scale/play.py, which writes games.jsonl (one line per game)
and records/p<pair>[s].jsonl.gz (each game's two seats, with their searched decisions when --record). pack()
turns a chunk's output into one batch, one line per seat per game:

    {"run", "machine", "chunk", "version": the version that played this seat, "train": its records are
     training data (the current version's seats; not the starting network's in a game against it),
     "game": {pair, swap, deck1, deck2, game_seed, budget, method, winner, turns, seconds, error},
     "seat": "A"|"B", "bot", "result": 1 | -1 | null, "records": [...]}

Each record is BenchPlayer's (play.py --record): the decision type, turn, the options and the search's visits,
backed-up values and priors, the option played, the root value, the heuristic's score, and the state: the graph
and each option's nodes for the GNN ("graph"), the feature ids for the flat MLP ("features").
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path


def pack(chunk_dir: Path, out: Path, *, run: str, machine: str, chunk: str, version: int,
         opponent_version: int | None = None) -> dict:
    """One batch file from a play.py chunk. `opponent_version`: bot2's version when it isn't `version` (bot1
    plays `version`; play.py seats bot1 in A unless the game is swapped). Returns the batch's summary."""
    chunk_dir = Path(chunk_dir)
    games = {}
    gfile = chunk_dir / "games.jsonl"
    if gfile.exists():
        for line in gfile.read_text().splitlines():
            if line.strip():
                g = json.loads(line)
                games[(g["pair"], g["swap"])] = g
    lines, summary = [], {"games": 0, "errors": 0, "seat_games": 0, "train_seat_games": 0, "decisions": 0,
                          "no_winner": 0, "seconds": 0.0}
    for (pair, swap), g in sorted(games.items()):
        summary["seconds"] += float(g.get("seconds") or 0)
        if g.get("error"):
            summary["errors"] += 1
            continue
        summary["games"] += 1
        summary["no_winner"] += g.get("winner") is None
        meta = {k: g.get(k) for k in ("pair", "swap", "deck1", "deck2", "game_seed", "budget", "method", "winner",
                                      "turns", "seconds")}
        rec_file = chunk_dir / "records" / f"p{pair:05d}{'s' if swap else ''}.jsonl.gz"
        seats = {}
        if rec_file.exists():
            with gzip.open(rec_file, "rt") as f:
                for ln in f:
                    if ln.strip():
                        s = json.loads(ln)
                        seats[s["seat"]] = s
        for seat in ("A", "B"):
            s = seats.get(seat)
            if s is None:
                continue
            bot1_here = (seat == "A") != bool(swap)
            v = version if bot1_here or opponent_version is None else opponent_version
            train = v == version
            lines.append(json.dumps({"run": run, "machine": machine, "chunk": chunk, "version": v, "train": train,
                                     "game": meta, "seat": seat, "bot": s.get("bot"), "result": s.get("result"),
                                     "records": s.get("records") or []}) + "\n")
            summary["seat_games"] += 1
            summary["train_seat_games"] += train
            summary["decisions"] += len(s.get("records") or [])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with gzip.open(tmp, "wt") as f:
        f.write("".join(lines))
    tmp.replace(out)
    summary.update(bytes=out.stat().st_size, packed=time.time())
    return summary


def read(path: Path):
    """The seat-games of a batch file, one dict each."""
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def batch_name(machine: str, chunk: str) -> str:
    return f"{machine}-{chunk}.jsonl.gz"


def eval_results(chunk_dir: Path) -> list[dict]:
    """An evaluation shard's games (play.py's games.jsonl lines), for upload."""
    gfile = Path(chunk_dir) / "games.jsonl"
    if not gfile.exists():
        return []
    return [json.loads(x) for x in gfile.read_text().splitlines() if x.strip()]
