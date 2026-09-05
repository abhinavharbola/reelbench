"""
Baseline models: popularity and item-item collaborative filtering.

Item-item CF note (this fixes the RAM issue flagged during review): a naive
dense item-item similarity matrix over MovieLens 25M's ~62k items is
~15GB at float32, which blows the 16GB budget before anything else is
loaded. This implementation computes similarity as a sparse matrix in
row-blocks and keeps only the top-K neighbors per item, so memory stays
bounded by (num_items * top_k) instead of (num_items ** 2).
"""

from collections import defaultdict

import numpy as np
import polars as pl
from scipy import sparse


class PopularityModel:
    """Ranks items by raw positive-interaction count in train. Same
    ranked list for every user, filtered by what that user has already
    seen."""

    def __init__(self):
        self.ranked_items: list[int] = []
        self._seen_by_user: dict[int, set] = {}

    def fit(self, train: pl.DataFrame) -> None:
        # sort by (count desc, movieId asc): group_by() alone doesn't
        # guarantee row order for ties, so without a secondary key here
        # the relative rank of equally-popular movies changes from run to
        # run on identical data, which silently shifts ndcg@k/map@k/
        # diversity for any k beyond the point where counts stop being
        # unique -- confirmed by running the same seeded pipeline twice
        # and diffing the resulting comparison_table.csv.
        counts = (
            train.group_by("movieId")
            .agg(pl.len().alias("count"))
            .sort(["count", "movieId"], descending=[True, False])
        )
        self.ranked_items = counts["movieId"].to_list()
        self._seen_by_user = (
            train.group_by("userId")
            .agg(pl.col("movieId"))
            .to_dict(as_series=False)
        )
        self._seen_by_user = {
            uid: set(items)
            for uid, items in zip(self._seen_by_user["userId"], self._seen_by_user["movieId"])
        }

    def recommend(self, user_id: int, k: int) -> list[int]:
        seen = self._seen_by_user.get(user_id, set())
        out = []
        for item in self.ranked_items:
            if item not in seen:
                out.append(item)
                if len(out) == k:
                    break
        return out


class ItemItemCF:
    """
    Cosine-similarity item-item CF, sparse and top-K bounded.

    fit():
      1. build a binary (items x users) sparse interaction matrix
      2. L2-normalize each item's row -> cosine similarity reduces to a dot product
      3. compute similarity in row-blocks (not all at once) and keep only
         the top_k highest-similarity neighbors per item, discard the rest
         immediately -> peak memory is bounded, never O(num_items^2)

    recommend(): score candidate items as the sum of similarities to items
    the user has already interacted with, weighted by neighbor rank.
    """

    def __init__(self, top_k: int = 50, block_size: int = 2000):
        self.top_k = top_k
        self.block_size = block_size
        self.item_ids: np.ndarray | None = None
        self.item_id_to_idx: dict[int, int] = {}
        self.neighbors: dict[int, list[tuple[int, float]]] = {}
        self._seen_by_user: dict[int, set] = {}

    def fit(self, train: pl.DataFrame) -> None:
        item_ids = train["movieId"].unique().sort().to_list()
        user_ids = train["userId"].unique().sort().to_list()
        self.item_ids = np.array(item_ids)
        self.item_id_to_idx = {item: idx for idx, item in enumerate(item_ids)}
        user_id_to_idx = {user: idx for idx, user in enumerate(user_ids)}

        rows = train["movieId"].replace_strict(self.item_id_to_idx).to_numpy()
        cols = train["userId"].replace_strict(user_id_to_idx).to_numpy()
        data = np.ones(len(rows), dtype=np.float32)

        item_user = sparse.csr_matrix(
            (data, (rows, cols)), shape=(len(item_ids), len(user_ids))
        )

        norms = np.sqrt(item_user.multiply(item_user).sum(axis=1)).A1
        norms[norms == 0] = 1.0
        row_normalizer = sparse.diags(1.0 / norms)
        item_user_normalized = row_normalizer @ item_user

        n_items = item_user_normalized.shape[0]
        self.neighbors = {}
        for start in range(0, n_items, self.block_size):
            end = min(start + self.block_size, n_items)
            block = item_user_normalized[start:end]
            sim_block = block @ item_user_normalized.T  # (block_size x n_items), sparse

            sim_block = sim_block.tocsr()
            for local_idx in range(sim_block.shape[0]):
                global_idx = start + local_idx
                row = sim_block.getrow(local_idx)
                if row.nnz == 0:
                    self.neighbors[item_ids[global_idx]] = []
                    continue
                candidate_idx = row.indices
                candidate_val = row.data
                order = np.argsort(-candidate_val)
                top = []
                for o in order:
                    neighbor_idx = candidate_idx[o]
                    if neighbor_idx == global_idx:
                        continue
                    top.append((item_ids[neighbor_idx], float(candidate_val[o])))
                    if len(top) == self.top_k:
                        break
                self.neighbors[item_ids[global_idx]] = top

        self._seen_by_user = {
            uid: set(items)
            for uid, items in zip(
                *train.group_by("userId").agg(pl.col("movieId")).to_dict(as_series=False).values()
            )
        }

    def recommend(self, user_id: int, k: int) -> list[int]:
        seen = self._seen_by_user.get(user_id, set())
        if not seen:
            return []

        scores: dict[int, float] = defaultdict(float)
        for seed_item in seen:
            for neighbor_item, sim in self.neighbors.get(seed_item, []):
                if neighbor_item in seen:
                    continue
                scores[neighbor_item] += sim

        # secondary movieId tiebreak for the same determinism reason as
        # PopularityModel.fit() above -- accumulated scores can tie exactly
        # when two candidates share the same set of contributing neighbors.
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return [item for item, _ in ranked[:k]]


