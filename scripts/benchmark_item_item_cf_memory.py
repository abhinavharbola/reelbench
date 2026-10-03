import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import resource
except ImportError:
    sys.exit("this benchmark needs the unix-only resource module; run it on Linux or macOS")

import numpy as np
import polars as pl

from src.models.baseline import ItemItemCF

N_ITEMS = 31195
N_USERS = 162541
N_INTERACTIONS = 13_750_000
POPULARITY_EXPONENT = 1.0
USER_ACTIVITY_EXPONENT = 0.8


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return peak / divisor


def zipf_probabilities(n: int, exponent: float) -> np.ndarray:
    weights = 1.0 / np.arange(1, n + 1) ** exponent
    return weights / weights.sum()


rng = np.random.default_rng(0)
item_ids = rng.choice(N_ITEMS, size=N_INTERACTIONS, p=zipf_probabilities(N_ITEMS, POPULARITY_EXPONENT)).astype(np.int32) + 1
user_ids = rng.choice(N_USERS, size=N_INTERACTIONS, p=zipf_probabilities(N_USERS, USER_ACTIVITY_EXPONENT)).astype(np.int32) + 1
timestamps = np.arange(N_INTERACTIONS, dtype=np.int64)

train = pl.DataFrame({
    "userId": user_ids,
    "movieId": item_ids,
    "timestamp": timestamps,
}).unique(subset=["userId", "movieId"])

print(f"items: {train['movieId'].n_unique():,}")
print(f"users: {train['userId'].n_unique():,}")
print(f"interactions: {train.height:,}")

start_rss = peak_rss_mb()
t0 = time.time()

model = ItemItemCF(top_k=50, block_size=2000)
model.fit(train)

t1 = time.time()
end_rss = peak_rss_mb()

print(f"fit time: {t1 - t0:.1f}s")
print(f"peak RSS before fit: {start_rss:.1f} MB")
print(f"peak RSS after fit: {end_rss:.1f} MB")
print(f"peak RSS delta: {end_rss - start_rss:.1f} MB")
