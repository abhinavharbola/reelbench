import polars as pl
import pytest

from src.data.split import assert_no_leakage, compute_global_cutoff, temporal_split


def make_synthetic_interactions() -> pl.DataFrame:
    """
    3 users, timestamps 0..99 (day-resolution ints for readability).

    user 1: 20 interactions spread 0..90   (has post-cutoff activity)
    user 2: 20 interactions spread 5..95   (has post-cutoff activity)
    user 3: only 2 interactions, both early -> below min_train_interactions
    """
    rows = []
    for uid, ts_list in [
        (1, list(range(0, 91, 5))),   # 19 points, 0..90
        (2, list(range(5, 96, 5))),   # 19 points, 5..95
        (3, [1, 2]),
    ]:
        for i, ts in enumerate(ts_list):
            rows.append({"userId": uid, "movieId": 100 + i, "timestamp": ts})
    return pl.DataFrame(rows).cast({"userId": pl.Int32, "movieId": pl.Int32, "timestamp": pl.Int64})


def test_cutoff_is_global_not_per_user():
    interactions = make_synthetic_interactions()
    cutoff = compute_global_cutoff(interactions, test_quantile=0.7)

    split = temporal_split(interactions, cutoff_timestamp=cutoff, min_train_interactions=5)

    # every train row for every user must be strictly before the SAME cutoff
    assert (split.train["timestamp"] < cutoff).all()
    # every test row for every user must be at/after the SAME cutoff
    assert (split.test["timestamp"] >= cutoff).all()


def test_no_leakage_assertion_passes_on_valid_split():
    interactions = make_synthetic_interactions()
    split = temporal_split(interactions, cutoff_timestamp=50, min_train_interactions=5)
    assert_no_leakage(split)  # should not raise


def test_low_activity_user_excluded():
    interactions = make_synthetic_interactions()
    split = temporal_split(interactions, cutoff_timestamp=50, min_train_interactions=5)

    train_users = set(split.train["userId"].unique().to_list())
    assert 3 not in train_users  # user 3 has only 2 interactions total


def test_max_test_per_user_cap_is_earliest_first():
    interactions = make_synthetic_interactions()
    split = temporal_split(
        interactions, cutoff_timestamp=50, max_test_per_user=3, min_train_interactions=5
    )

    user1_test = split.test.filter(pl.col("userId") == 1).sort("timestamp")
    assert user1_test.height == 3
    # earliest-first among post-cutoff points: 50, 55, 60
    assert user1_test["timestamp"].to_list() == [50, 55, 60]


def test_assert_no_leakage_catches_corrupted_split():
    interactions = make_synthetic_interactions()
    split = temporal_split(interactions, cutoff_timestamp=50, min_train_interactions=5)

    # corrupt: inject a future-dated row into train
    corrupted_train = pl.concat(
        [split.train, pl.DataFrame({"userId": [1], "movieId": [999], "timestamp": [999]})
         .cast({"userId": pl.Int32, "movieId": pl.Int32, "timestamp": pl.Int64})]
    )
    corrupted = split.__class__(train=corrupted_train, test=split.test, cutoff_timestamp=split.cutoff_timestamp)

    with pytest.raises(AssertionError):
        assert_no_leakage(corrupted)
