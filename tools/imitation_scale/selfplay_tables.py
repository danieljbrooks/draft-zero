"""Stage 6's self-play records (tools/imitation_scale/play.py --record) -> trainer tables (docs/017
§6.6): <out>/selfplay_{train,val,test}.h5, the `soft` table kind of draftzero.gameplay.supervised.

    python tools/imitation_scale/selfplay_tables.py --records runs/exp4/stage6/play/records \\
        --out data/imitation_scale/h5

--records takes record files (play.py --record writes one per game) or directories of them.

Every deck pair goes whole to one split (both of its games): 65% train, 10% validation (early
stopping) and 25% test, the held-out evaluation set of stage 6b, never trained on.

One row per searched decision of one seat:
  indices / offsets   the state as the search's root encoded it: the seat's view, the opponent's hand
                      hidden
  legal CSR           every legal option's action index
  set CSR + set_p     the policy target: the search's visit counts over the options, normalised
  meta/atype          the decision type: 0 PRIORITY (player priority head), 3 CHOOSE_TARGET (target
                      head), 5 CHOOSE_USE (binary head, options 0 no / 1 yes)
  z                   the game's result from the seat, +1 / -1 (nan without a winner): what the
                      value is scored against
  z_td                the value target: MageZero's TD(lambda) over the seat's decisions, backwards
                      from the result, each step blending in the search's root value (lambda 0.95,
                      ParallelDataGenerator.generateLabeledStatesForGame); nan without a winner
  heuristic           GameStateEvaluator3's score of the position from the seat (nan in older
                      records): the baseline stage 6b compares the value head with
  meta/row            the game (pair x 2 + seat order); meta/seat 0 A / 1 B; meta/pair the deck pair
  meta/turn           the player's turn number as the human tables count it, (game turn + 1) // 2;
  meta/game_turn      the game's turn number
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ATYPE = {"PRIORITY": 0, "CHOOSE_TARGET": 3, "CHOOSE_USE": 5}
SPLITS = ("train", "val", "test")
SALT = "draftzero-exp4-selfplay-v1"


def split_of(pair: int, run: str = "") -> int:
    h = int.from_bytes(hashlib.sha1(f"{SALT}|{run}|{pair}".encode()).digest()[:8], "big") / 2 ** 64
    return 0 if h < 0.65 else (1 if h < 0.75 else 2)


def td_targets(q: np.ndarray, result: float, lam: float) -> np.ndarray:
    """MageZero's labels: G = result; backwards, G = lam * G + (1 - lam) * q_i, label_i = G."""
    out = np.empty(len(q), np.float32)
    g = float(result)
    for i in range(len(q) - 1, -1, -1):
        g = lam * g + (1 - lam) * float(q[i])
        out[i] = g
    return out


def rows_of(line: dict, lam: float) -> list[dict]:
    """The usable rows of one seat's game: decisions with a head and 2+ options, in order."""
    res = line.get("result")
    recs = [r for r in line["records"] if r["type"] in ATYPE and len(r["legal"]) >= 2 and sum(r["visits"]) > 0]
    recs = [r for r in recs if r["type"] != "CHOOSE_USE" or max(r["legal"]) <= 1]
    if not recs:
        return []
    zt = td_targets(np.asarray([r["q"] for r in recs], np.float64), 0.0 if res is None else res, lam)
    if res is None:
        zt[:] = np.nan
    game = int(line["pair"]) * 2 + int(bool(line["swap"]))
    out = []
    for r, t in zip(recs, zt):
        v = np.asarray(r["visits"], np.float64)
        keep = v > 0
        out.append({"features": np.asarray(r["features"], np.int32), "legal": np.asarray(r["legal"], np.int32),
                    "set": np.asarray(r["legal"], np.int32)[keep], "p": (v[keep] / v.sum()).astype(np.float32),
                    "atype": ATYPE[r["type"]], "z": np.nan if res is None else float(res), "z_td": float(t),
                    "heuristic": float(r.get("heuristic", np.nan)),
                    "game": game, "turn": (int(r["turn"]) + 1) // 2, "game_turn": int(r["turn"]), "seat": 0 if line["seat"] == "A" else 1,
                    "pair": int(line["pair"])})
    return out


def heuristic_auc(rows: list[dict]) -> float | None:
    """The heuristic's AUC against the seat's result over decided games' rows (the value head's
    baseline, docs/017 §6.6 stage 6b)."""
    h = np.asarray([r["heuristic"] for r in rows], np.float64)
    z = np.asarray([r["z"] for r in rows], np.float64)
    ok = np.isfinite(h) & np.isfinite(z)
    if ok.sum() < 2 or len(np.unique(z[ok])) < 2:
        return None
    from scipy.stats import rankdata
    rk, won = rankdata(h[ok]), z[ok] > 0
    return float((rk[won].sum() - won.sum() * (won.sum() + 1) / 2) / (won.sum() * (~won).sum()))


def write_table(rows: list[dict], path: Path) -> None:
    import h5py

    def csr(key):
        ptr = np.r_[0, np.cumsum([len(r[key]) for r in rows])].astype(np.int64)
        data = np.concatenate([r[key] for r in rows]) if rows else np.zeros(0)
        return ptr, data
    path.parent.mkdir(parents=True, exist_ok=True)
    off, ind = csr("features")
    lptr, lidx = csr("legal")
    sptr, sidx = csr("set")
    _, sp = csr("p")
    with h5py.File(path, "w") as f:
        f.create_dataset("indices", data=ind.astype(np.int32), compression="gzip", compression_opts=1)
        f.create_dataset("offsets", data=off)
        f.create_dataset("legal_indptr", data=lptr)
        f.create_dataset("legal_idx", data=lidx.astype(np.int32))
        f.create_dataset("set_indptr", data=sptr)
        f.create_dataset("set_idx", data=sidx.astype(np.int32))
        f.create_dataset("set_p", data=sp.astype(np.float32))
        f.create_dataset("z", data=np.asarray([r["z"] for r in rows], np.float32))
        f.create_dataset("z_td", data=np.asarray([r["z_td"] for r in rows], np.float32))
        f.create_dataset("heuristic", data=np.asarray([r["heuristic"] for r in rows], np.float32))
        f.create_dataset("weight", data=np.ones(len(rows), np.float32))
        for k in ("atype", "turn", "game_turn", "seat", "pair"):
            f.create_dataset(f"meta/{k}", data=np.asarray([r[k] for r in rows], np.int32))
        f.create_dataset("meta/row", data=np.asarray([r["game"] for r in rows], np.int64))
        f.attrs["layout"] = "soft policy targets (set CSR + set_p) over legal CSR; draftzero tools/imitation_scale"


def record_files(paths: list[Path]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        out.extend(sorted(p.glob("*.jsonl.gz")) if p.is_dir() else [p])
    return out


def build(records: list[Path], out: Path, lam: float = 0.95, run: str = "", prefix: str = "selfplay") -> dict:
    by = {s: [] for s in SPLITS}
    stats, types = Counter(), Counter()
    for path in record_files(records):
        with gzip.open(path, "rt") as f:
            for ln in f:
                if not ln.strip():
                    continue
                line = json.loads(ln)
                types.update(r["type"] for r in line["records"])
                rows = rows_of(line, lam)
                stats["seat_games"] += 1
                stats["rows"] += len(rows)
                by[SPLITS[split_of(int(line["pair"]), run)]].extend(rows)
    out_stats = {"seat_games": stats["seat_games"], "decisions": sum(types.values()), "rows": stats["rows"],
                 "decisions_by_type": dict(types)}
    for s in SPLITS:
        write_table(by[s], out / f"{prefix}_{s}.h5")
        out_stats[s] = {"rows": len(by[s]), "games": len({r["game"] for r in by[s]}),
                        "by_type": dict(Counter(r["atype"] for r in by[s])),
                        "heuristic_auc": heuristic_auc(by[s])}
    return out_stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--td-lambda", type=float, default=0.95)
    ap.add_argument("--run", default="", help="salts the pair -> split hash (several runs' pairs share numbers)")
    ap.add_argument("--prefix", default="selfplay")
    a = ap.parse_args(argv)
    print(json.dumps(build(a.records, a.out, a.td_lambda, a.run, a.prefix), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
