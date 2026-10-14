"""paths.py — where the package finds its data, and keeping machine-specific paths out of versioned data.

Decks are named by stem (``FDN_top_00001_WB``), never by path, in every versioned file: gorge reads
``<decks dir>/<stem>.json`` (``draftzero.data.pools``), so a pool moves between machines unchanged. Absolute paths
in pool files broke every game on the first RunPod run, twice: once because the paths were the laptop's, and again
when an rsync from the laptop restored them.

    GORGE_DECKS, GORGE_POOL    the full FDN pool in gorge format (data/gorge/, gitignored; python -m draftzero.data.decks)
    SAMPLE_DECKS, SAMPLE_POOL  the 80-deck sample shipped in assets/sample (a fresh clone's smoke runs)

The ``.dck`` helpers below (``read_pool`` with legacy full paths, ``resolve_pool``, ``build_pools``) are the XMage
era's stem lists (``assets/sample/train.txt``); gorge does not need them.

``REPO`` is the checkout whose ``assets/`` and ``data/`` the package reads: ``$DZ_REPO`` if set (a frozen copy of
the code running against another checkout), else this source tree (``src/draftzero/data`` -> the repo root).
"""
import os
from pathlib import Path
from typing import Iterable

REPO = Path(os.environ.get("DZ_REPO") or Path(__file__).resolve().parents[3])
ASSETS = REPO / "assets"
GORGE_DECKS = REPO / "data" / "gorge" / "decks"
GORGE_POOL = REPO / "data" / "gorge" / "pool.tsv"
SAMPLE_DECKS = ASSETS / "sample" / "decks"
SAMPLE_POOL = ASSETS / "sample" / "pool.tsv"
DECK_SUFFIX = ".json"      # a gorge deck file
SUFFIX = ".dck"            # an XMage deck file (legacy pool files)


def gorge_deck_path(stem: str, decks_dir: Path = GORGE_DECKS) -> Path:
    """Where gorge looks for a deck: ``<decks_dir>/<stem>.json`` (a legacy path or ``.dck`` name is reduced to its stem)."""
    return Path(decks_dir) / f"{Path(stem).stem if stem.endswith((SUFFIX, DECK_SUFFIX)) or '/' in stem else stem}{DECK_SUFFIX}"


def read_pool(pool_file: Path) -> list[str]:
    """Deck stems from a pool file. Tolerates legacy files holding full paths."""
    out = []
    for line in Path(pool_file).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(Path(line).stem if ("/" in line or line.endswith(SUFFIX)) else line)
    return out


def deck_path(stem: str, deck_root: Path) -> Path:
    return Path(deck_root) / f"{Path(stem).stem}{SUFFIX}"


def resolve_pool(stems: Iterable[str], deck_root: Path, out_file: Path) -> Path:
    """Write absolute deck paths for the JVM. Fails loudly rather than letting the JVM
    report a bare 'deck size=0' for every game."""
    stems = list(stems)
    paths = [deck_path(s, deck_root) for s in stems]
    missing = [p for p in paths[:200] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of the first {min(len(paths), 200)} decks missing under {deck_root}, "
            f"e.g. {missing[0]}. Set deck_root (or MZ_DECK_DIR) to the directory holding the .dck files."
        )
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(str(p) for p in paths) + "\n")
    return out_file


def normalize_pool_file(pool_file: Path) -> int:
    """Rewrite a pool file in place to bare stems. Returns how many lines changed."""
    p = Path(pool_file)
    before = p.read_text().splitlines()
    after = read_pool(p)
    if after != [b.strip() for b in before if b.strip()]:
        p.write_text("\n".join(after) + "\n")
        return sum(1 for a, b in zip(after, [b.strip() for b in before if b.strip()]) if a != b)
    return 0


def build_pools(meta_tsv: Path, out_dir: Path) -> dict[str, int]:
    """Generate train/eval pool files from the split column of decks.tsv.

    The split is the experiment definition, so decks.tsv is the versioned source of truth
    and the pool files are derived. That keeps one small reviewable file under git instead
    of 31.5k paths that go stale the moment they leave this machine.
    """
    rows = Path(meta_tsv).read_text().splitlines()
    header = rows[0].split("\t")
    buckets: dict[str, list[str]] = {}
    for r in rows[1:]:
        if not r.strip():
            continue
        vals = dict(zip(header, r.split("\t")))
        buckets.setdefault(vals["split"], []).append(vals["deck"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, stems in buckets.items():
        stems.sort()
        (out_dir / f"{split}.txt").write_text("\n".join(stems) + "\n")
        counts[split] = len(stems)
    return counts
