"""
Measures actual peak RSS of ItemItemCF.fit() at MovieLens-25M catalog
scale (62,423 items, 162,541 users) using synthetic interaction data,
since the real ml-25m.zip is not downloadable from this environment's
network allowlist.

Usage:
    python scripts/benchmark_item_item_cf_memory.py
"""

import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from src.models.baseline import ItemItemCF

N_ITEMS = 62423
N_USERS = 162541
N_INTERACTIONS = 13_750_000

rng = np.random.default_rng(0)
item_ids = rng.integers(1, N_ITEMS + 1, size=N_INTERACTIONS, dtype=np.int32)
user_ids = rng.integers(1, N_USERS + 1, size=N_INTERACTIONS, dtype=np.int32)
timestamps = np.arange(N_INTERACTIONS, dtype=np.int64)

train = pl.DataFrame({
    "userId": user_ids,
    "movieId": item_ids,
    "timestamp": timestamps,
}).unique(subset=["userId", "movieId"])

print(f"items: {train['movieId'].n_unique():,}")
print(f"users: {train['userId'].n_unique():,}")
print(f"interactions: {train.height:,}")

start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
t0 = time.time()

model = ItemItemCF(top_k=50, block_size=2000)
model.fit(train)

t1 = time.time()
end_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

print(f"fit time: {t1 - t0:.1f}s")
print(f"peak RSS before fit: {start_rss / 1024:.1f} MB")
print(f"peak RSS after fit: {end_rss / 1024:.1f} MB")
print(f"peak RSS delta: {(end_rss - start_rss) / 1024:.1f} MB")
