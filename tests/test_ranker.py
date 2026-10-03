import pytest

pytest.importorskip("lightgbm")
import polars as pl

from src.ranking.features import FEATURE_COLUMNS
from src.ranking.ranker import build_training_table, rank_candidates, train_ranker


def make_features(user_id: int, n: int = 6) -> pl.DataFrame:
    rows = []
    for i in range(n):
        row = {"userId": user_id, "movieId": 100 + i}
        for j, column in enumerate(FEATURE_COLUMNS):
            row[column] = float(i if j == 0 else (i * (j + 1)) % 5)
        rows.append(row)
    return pl.DataFrame(rows)


def test_training_table_labels_positives():
    table = build_training_table({1: make_features(1), 2: make_features(2)}, {1: {105}, 2: {100}})
    assert table.filter((pl.col("userId") == 1) & (pl.col("movieId") == 105))["label"][0] == 1
    assert table.filter((pl.col("userId") == 1) & (pl.col("movieId") == 100))["label"][0] == 0


def test_train_ranker_rejects_empty_and_unlabeled_tables():
    with pytest.raises(ValueError):
        train_ranker(pl.DataFrame())
    table = build_training_table({1: make_features(1)}, {1: set()})
    with pytest.raises(ValueError):
        train_ranker(table)


def test_rank_candidates_returns_sorted_top_k_and_handles_empty():
    per_user = {u: make_features(u) for u in range(1, 21)}
    positives = {u: {105} for u in range(1, 21)}
    ranker = train_ranker(build_training_table(per_user, positives), num_boost_round=10)

    ranked = rank_candidates(ranker, make_features(99), top_k=3)
    assert len(ranked) == 3
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)
    assert rank_candidates(ranker, pl.DataFrame(), top_k=3) == []
