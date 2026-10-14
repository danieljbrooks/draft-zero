"""Store a pack's card_raw (≈95% of its bytes) as float16, in place: halves the disk and RAM a game takes.

    python gorge/recipe/shrink.py PACK_DIR [...]

The values are small multiples of 0.05-0.25 (53 distinct in stage 0); float16 keeps them within 2.5e-4 relative.
dzg.data casts card_raw to float32 when it builds a batch, so nothing else changes.
"""
import os
import sys
from pathlib import Path

import numpy as np

for root in sys.argv[1:]:
    for f in sorted(Path(root).glob("*/card_raw.npy")) or sorted(Path(root).glob("card_raw.npy")):
        a = np.load(f, mmap_mode="r")
        if a.dtype == np.float16:
            continue
        tmp = f.with_suffix(".tmp.npy")
        np.save(tmp, np.asarray(a, np.float16))
        os.replace(tmp, f)
        print(f, a.nbytes >> 20, "MB ->", f.stat().st_size >> 20, "MB")
