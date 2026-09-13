"""
ALS / BPR matrix factorization baseline, via the `implicit` library, CPU.

Both algorithms in `implicit` expect a (users x items) sparse matrix of
confidence/interaction weights and natively support filtering out items the
user has already interacted with at recommend-time — no manual seen-item
bookkeeping needed here, unlike the other baselines.
"""

import numpy as np
import polars as pl
from scipy import sparse

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking


class MatrixFactorizationModel:
    """Wraps implicit's ALS or BPR behind the same fit/recommend interface
    used by the other baselines, so the evaluation harness treats all
    approaches identically."""

    def __init__(self, method: str = "als", factors: int = 64, iterations: int = 15, regularization: float = 0.01):
        if method not in ("als", "bpr"):
            raise ValueError("method must be 'als' or 'bpr'")
        self.method = method
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization

        self.model = None
        self.user_id_to_idx: dict[int, int] = {}
        self.idx_to_item_id: dict[int, int] = {}
        self.user_items: sparse.csr_matrix | None = None

    def fit(self, train: pl.DataFrame) -> None:
        user_ids = train["userId"].unique().sort().to_list()
        item_ids = train["movieId"].unique().sort().to_list()
        self.user_id_to_idx = {u: i for i, u in enumerate(user_ids)}
        item_id_to_idx = {m: i for i, m in enumerate(item_ids)}
        self.idx_to_item_id = {i: m for m, i in item_id_to_idx.items()}

        rows = train["userId"].replace_strict(self.user_id_to_idx).to_numpy()
        cols = train["movieId"].replace_strict(item_id_to_idx).to_numpy()
        data = np.ones(len(rows), dtype=np.float32)

        self.user_items = sparse.csr_matrix(
            (data, (rows, cols)), shape=(len(user_ids), len(item_ids))
        )

        if self.method == "als":
            self.model = AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
                random_state=42,
            )
        else:
            self.model = BayesianPersonalizedRanking(
                factors=self.factors,
                iterations=self.iterations,
                random_state=42,
            )

        self.model.fit(self.user_items)

    def recommend(self, user_id: int, k: int) -> list[int]:
        if user_id not in self.user_id_to_idx:
            return []  # cold user, not covered by this model -> handled separately

        user_idx = self.user_id_to_idx[user_id]
        item_indices, _scores = self.model.recommend(
            user_idx,
            self.user_items[user_idx],
            N=k,
            filter_already_liked_items=True,
        )
        return [self.idx_to_item_id[i] for i in item_indices]
