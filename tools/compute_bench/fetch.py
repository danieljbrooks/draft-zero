"""Fetch the compute benchmark's results from the HF repo (exp4/bench/runs/<tag>/) into runs/compute_bench/<tag>/.

    python tools/compute_bench/fetch.py                 # every tag
    python tools/compute_bench/fetch.py 3090-secure     # some
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "danbrooks/draftzero-checkpoints"
ROOT = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    tags = (argv if argv is not None else sys.argv[1:]) or None
    tok = os.environ.get("HF_TOKEN")
    files = HfApi(token=tok).list_repo_files(REPO_ID)
    have = sorted({f.split("/")[3] for f in files if f.startswith("exp4/bench/runs/") and f.count("/") >= 4})
    for tag in tags or have:
        if tag not in have:
            print(f"{tag}: not on HF yet")
            continue
        d = snapshot_download(REPO_ID, allow_patterns=[f"exp4/bench/runs/{tag}/**"], token=tok)
        src, dst = Path(d) / "exp4/bench/runs" / tag, ROOT / "runs/compute_bench" / tag
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"{tag}: {sum(1 for _ in dst.rglob('*'))} files in {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
