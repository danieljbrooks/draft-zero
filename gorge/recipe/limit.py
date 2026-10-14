"""Records in a pack's first N games (records are in game order): python gorge/recipe/limit.py PACK N [N ...]"""
import sys

import numpy as np

sys.path.insert(0, "gorge")
from dzg import data as D  # noqa: E402

ds = D.ShardSet([sys.argv[1]])
g = ds.array("game")
# the game index restarts per shard; count distinct games in record order across shards
change = np.ones(len(g), bool)
change[1:] = g[1:] != g[:-1]
ordinal = np.cumsum(change)          # 1-based game ordinal of each record
for n in map(int, sys.argv[2:]):
    print(int(np.searchsorted(ordinal, n, side="right")))
