from collections import defaultdict

import numpy as np
import polars as pl
from scipy import sparse


class PopularityModel:
    def __init__(self):
        self.ranked_items: list[int] = []
        self._seen_by_user: dict[int, set] = {}

    def fit(self, train: pl.DataFrame) -> None:
        counts = (
            train.group_by("movieId")
            .agg(pl.len().alias("count"))
            .sort(["count", "movieId"], descending=[True, False])
        )
        self.ranked_items = counts["movieId"].to_list()
        grouped = train.group_by("userId").agg(pl.col("movieId")).to_dict(as_series=False)
        self._seen_by_user = {
            uid: set(items) for uid, items in zip(grouped["userId"], grouped["movieId"])
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
            sim_block = block @ item_user_normalized.T

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

        grouped = train.group_by("userId").agg(pl.col("movieId")).to_dict(as_series=False)
        self._seen_by_user = {
            uid: set(items) for uid, items in zip(grouped["userId"], grouped["movieId"])
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

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return [item for item, _ in ranked[:k]]
