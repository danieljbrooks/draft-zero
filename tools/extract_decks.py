"""
Extract limited decks from a 17lands public game-data file.

Each row of the game data is one game and carries the player's full maindeck (`deck_<card>`
columns) plus the player's overall win-rate bucket. A deck is one (draft_id, build_index): the
same draft, same build. We keep decks whose player is at or above --min-winrate.

  python tools/extract_decks.py --set FDN --format PremierDraft --min-winrate 0.60

Data: https://www.17lands.com/public_datasets (downloaded and cached under data/17lands/).
17lands public data is licensed CC BY 4.0: credit 17lands in anything built from it.
Outputs (under data/deckgen/<set>_<format>_wr<min>/):
  decks.jsonl   one deck per line: ids, player buckets, colors, games/wins in the sample, cards
  summary.txt   counts by color pair, deck size, and the most-played cards
  top_player_<set>_decks/*.dck   (with --dck) XMage deck files, one per deck

XMage card numbers come from XMage's own compiled set classes (via javap), because XMage's
collector numbers differ from Scryfall's for some cards. Pass the set classes to search in order,
e.g. Foundations plus SpecialGuests for FDN's bonus-sheet cards.

  python tools/extract_decks.py --set FDN --dck --xmage-jar xmage/lib/mage-sets-1.4.58.jar \
      --xmage-sets Foundations,SpecialGuests
"""
import re
import shutil
import subprocess
import zipfile
import argparse
import csv
import gzip
import json
import os
import sys
import urllib.request
from collections import Counter, defaultdict

URL = "https://17lands-public.s3.amazonaws.com/analysis_data/game_data/game_data_public.{set}.{format}.csv.gz"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# downloads and the XMage card-number cache (gitignored, regenerable)
CACHE = os.path.join(REPO, "data", "17lands")


def fetch(set_code: str, fmt: str) -> str:
    path = os.path.join(CACHE, f"game_data_public.{set_code}.{fmt}.csv.gz")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = URL.format(set=set_code, format=fmt)
        print(f"downloading {url}", file=sys.stderr)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def find_javap() -> str:
    for cand in (os.path.join(os.environ.get("JAVA_HOME", ""), "bin", "javap"), shutil.which("javap")):
        if cand and os.path.exists(cand):
            return cand
    sys.exit("javap not found: set JAVA_HOME to a JDK")


def xmage_card_numbers(jar: str, set_classes: list[str]) -> dict[str, tuple[str, str]]:
    """card name -> (XMage set code, collector number), searching set classes in order."""
    cache = os.path.join(CACHE, "xmage_" + "_".join(set_classes) + ".json")
    if os.path.exists(cache) and (not os.path.exists(jar) or os.path.getmtime(cache) >= os.path.getmtime(jar)):
        # no jar on this machine: trust the cache rather than fail (a stale cache is only
        # possible if the jar changed, and then it is present to compare against)
        return {k: tuple(v) for k, v in json.load(open(cache)).items()}
    if not os.path.exists(jar):
        sys.exit(f"XMage set jar not found: {jar}\n"
                 f"Pass --xmage-jar, or point the repo's `xmage` symlink at an XMage build.")
    javap = find_javap()
    out: dict[str, tuple[str, str]] = {}
    tmp = os.path.join(CACHE, "_classes")
    with zipfile.ZipFile(jar) as z:
        for cls in set_classes:
            member = f"mage/sets/{cls}.class"
            z.extract(member, tmp)
            text = subprocess.run([javap, "-c", "-p", os.path.join(tmp, member)],
                                  capture_output=True, text=True, check=True).stdout.splitlines()
            strings = [re.search(r"// String (.*)$", l) for l in text]
            set_code = [m.group(1) for m in strings if m][1]   # super("Set Name", "CODE", ...)
            for i, m in enumerate(strings[:-1]):
                if not m:
                    continue
                name = m.group(1).replace("\\'", "'").replace('\\"', '"')
                nxt = text[i + 1]
                num = re.search(r"(?:sipush|bipush)\s+(\d+)$", nxt) or re.search(r"iconst_(\d)$", nxt) \
                    or re.search(r'// String (\w+)$', nxt)
                if num and name not in out and name[:1].isupper():
                    out[name] = (set_code, num.group(1))
    shutil.rmtree(tmp, ignore_errors=True)
    json.dump(out, open(cache, "w"), indent=0, sort_keys=True)
    return out


def write_dck(decks, out_dir: str, set_code: str, numbers: dict) -> tuple[int, Counter]:
    """One XMage .dck per deck. Returns (written, missing card counts)."""
    os.makedirs(out_dir, exist_ok=True)
    missing = Counter()
    written = 0
    for i, d in enumerate(decks, 1):
        unknown = [c for c in d["cards"] if c not in numbers]
        if unknown:
            missing.update(unknown)
            continue
        colors = d["main_colors"] + (f"+{d['splash_colors']}" if d["splash_colors"] else "")
        name = f"Top player {set_code} deck {i:05d} {colors or 'C'} (player WR {d['user_game_win_rate_bucket']:.2f})"
        lines = [f"NAME:{name}"]
        lines += [f"{k} [{numbers[c][0]}:{numbers[c][1]}] {c}" for c, k in sorted(d["cards"].items())]
        fname = f"{set_code}_top_{i:05d}_{(colors or 'C').replace('+', 'p')}.dck"
        with open(os.path.join(out_dir, fname), "w") as f:
            f.write("\n".join(lines) + "\n")
        d["dck_file"] = fname
        written += 1
    return written, missing


def bucket(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract(path: str, min_winrate: float, min_player_games: int):
    csv.field_size_limit(sys.maxsize)
    decks = {}
    games = Counter()
    wins = Counter()
    rows_seen = rows_kept = 0
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        deck_cols = [(i, name[len("deck_"):]) for i, name in enumerate(header) if name.startswith("deck_")]
        for row in reader:
            rows_seen += 1
            wr = bucket(row[col["user_game_win_rate_bucket"]])
            n = bucket(row[col["user_n_games_bucket"]])
            if wr is None or wr < min_winrate or (n or 0) < min_player_games:
                continue
            rows_kept += 1
            key = (row[col["draft_id"]], row[col["build_index"]])
            games[key] += 1
            wins[key] += row[col["won"]] == "True"
            if key in decks:
                continue
            cards = {name: int(row[i]) for i, name in deck_cols if row[i] not in ("", "0")}
            decks[key] = {
                "draft_id": key[0],
                "build_index": int(key[1]),
                "expansion": row[col["expansion"]],
                "event_type": row[col["event_type"]],
                "draft_time": row[col["draft_time"]],
                "rank": row[col["rank"]],
                "main_colors": row[col["main_colors"]],
                "splash_colors": row[col["splash_colors"]],
                "user_game_win_rate_bucket": wr,
                "user_n_games_bucket": n,
                "maindeck_size": sum(cards.values()),
                "cards": dict(sorted(cards.items())),
            }
    for key, d in decks.items():
        d["games_in_sample"] = games[key]
        d["wins_in_sample"] = wins[key]
    return list(decks.values()), rows_seen, rows_kept


def summarize(decks, rows_seen, rows_kept, args) -> str:
    lines = [f"{args.set} {args.format}: {rows_seen:,} game rows, {rows_kept:,} from players with "
             f"win-rate bucket >= {args.min_winrate:.2f}; {len(decks):,} unique decks (draft_id + build_index)", ""]
    colors = Counter(d["main_colors"] + (f"+{d['splash_colors']}" if d["splash_colors"] else "") for d in decks)
    lines.append("decks by colors (main+splash):")
    lines += [f"  {c or '(none)':10s} {n:6,}" for c, n in colors.most_common(25)]
    sizes = Counter(d["maindeck_size"] for d in decks)
    lines += ["", "maindeck sizes: " + ", ".join(f"{s}: {n:,}" for s, n in sorted(sizes.items()))]
    wr = Counter(d["user_game_win_rate_bucket"] for d in decks)
    lines += ["player win-rate buckets: " + ", ".join(f"{b:.2f}: {n:,}" for b, n in sorted(wr.items()))]
    played = Counter()
    for d in decks:
        for name, k in d["cards"].items():
            played[name] += k
    lines += ["", "most-played cards (total copies across decks):"]
    lines += [f"  {n:7,}  {name}" for name, n in played.most_common(30)]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", default="FDN")
    ap.add_argument("--format", default="PremierDraft")
    ap.add_argument("--min-winrate", type=float, default=0.60, help="minimum user_game_win_rate_bucket")
    ap.add_argument("--min-player-games", type=int, default=0, help="minimum user_n_games_bucket")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dck", action="store_true", help="also write XMage .dck files")
    ap.add_argument("--xmage-jar", default=os.path.join(REPO, "xmage", "lib", "mage-sets-1.4.58.jar"))
    ap.add_argument("--xmage-sets", default="Foundations,SpecialGuests",
                    help="XMage set classes to look up card numbers in, in priority order")
    args = ap.parse_args()

    path = fetch(args.set, args.format)
    decks, rows_seen, rows_kept = extract(path, args.min_winrate, args.min_player_games)
    out = args.out or os.path.join(REPO, "data", "deckgen", f"{args.set}_{args.format}_wr{int(args.min_winrate * 100)}")
    os.makedirs(out, exist_ok=True)
    decks.sort(key=lambda d: (d["draft_time"], d["draft_id"], d["build_index"]))
    if args.dck:
        numbers = xmage_card_numbers(args.xmage_jar, args.xmage_sets.split(","))
        dck_dir = os.path.join(out, f"top_player_{args.set}_decks")
        written, missing = write_dck(decks, dck_dir, args.set, numbers)
        print(f"wrote {written:,} .dck files to {dck_dir}")
        if missing:
            print(f"skipped {len(decks) - written:,} decks with cards not in {args.xmage_sets}: "
                  + ", ".join(f"{c} ({n})" for c, n in missing.most_common()))
    with open(os.path.join(out, "decks.jsonl"), "w") as f:
        for d in decks:
            f.write(json.dumps(d) + "\n")
    summary = summarize(decks, rows_seen, rows_kept, args)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary)
    print(f"wrote {out}/decks.jsonl")


if __name__ == "__main__":
    main()
