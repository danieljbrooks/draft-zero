"""
dedupe_persist.py — store the watchdog's persist mirror once, on pods where it shares the disk.

On a pod with no network volume, `--persist data/persist` is the same disk as the files it
mirrors, so every checkpoint, replay shard and log is stored twice. Exp #2 run 2's 40 GB disk would
have filled at about hour 31 that way (docs/013 §2.7).

This replaces each mirror file with a hard link to its identical source. The watchdog's
`rsync -rlptD` skips a file whose size and modification time match, so the links survive later
syncs. Deletes nothing. Only files unchanged for 10 minutes are linked, and never model.pt.gz,
which training rewrites in place.

    python tools/dedupe_persist.py <run name> [--repo /root/draft-zero]      (e.g. hourly)
"""
import argparse
import os
import time
from pathlib import Path


def dedupe(repo: Path, run: str, min_age_s: float = 600) -> tuple[int, int]:
    persist = repo / "data" / "persist"
    pairs = [(persist / "models", repo / "models"), (persist / run, repo / "runs" / run),
             (persist / "data", repo / "data")]
    now, linked, saved = time.time(), 0, 0
    for dst_dir, src_dir in pairs:
        for dst in dst_dir.rglob("*"):
            if not dst.is_file() or dst.is_symlink() or dst.name == "model.pt.gz":
                continue
            src = src_dir / dst.relative_to(dst_dir)
            if not src.is_file() or src.is_symlink() or persist in src.parents:
                continue
            s, d = src.stat(), dst.stat()
            if (s.st_ino == d.st_ino or s.st_size != d.st_size or int(s.st_mtime) != int(d.st_mtime)
                    or now - s.st_mtime < min_age_s):
                continue
            tmp = dst.with_name(dst.name + ".dzlink")
            os.link(src, tmp)
            os.replace(tmp, dst)
            linked += 1
            saved += s.st_size
    return linked, saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run directory name, e.g. 2026-09-27_03-42-21")
    ap.add_argument("--repo", type=Path, default=Path("/root/draft-zero"))
    a = ap.parse_args()
    n, b = dedupe(a.repo, a.run)
    print(f"linked {n} files, freed {b / 2 ** 30:.1f} GB")
