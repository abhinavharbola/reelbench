import polars as pl
import pytest

from src.data.split import SplitResult, assert_no_leakage, compute_global_cutoff, temporal_split


def make_synthetic_interactions() -> pl.DataFrame:
    rows = []
    for uid, ts_list in [
        (1, list(range(0, 91, 5))),
        (2, list(range(5, 96, 5))),
        (3, [1, 2]),
    ]:
        for i, ts in enumerate(ts_list):
            rows.append({"userId": uid, "movieId": 100 + i, "timestamp": ts})
    return pl.DataFrame(rows).cast({"userId": pl.Int32, "movieId": pl.Int32, "timestamp": pl.Int64})


def test_cutoff_is_global_not_per_user():
    interactions = make_synthetic_interactions()
    cutoff = compute_global_cutoff(interactions, test_quantile=0.7)

    split = temporal_split(interactions, cutoff_timestamp=cutoff, min_train_interactions=5)

    assert (split.train["timestamp"] < cutoff).all()
    assert (split.test["timestamp"] >= cutoff).all()


def test_no_leakage_assertion_passes_on_valid_split():
    interactions = make_synthetic_interactions()
    split = temporal_split(interactions, cutoff_timestamp=50, min_train_interactions=5)
    assert_no_leakage(split)


def test_low_activity_user_excluded():
    interactions = make_synthetic_interactions()
    split = temporal_split(interactions, cutoff_timestamp=50, min_train_interactions=5)

    train_users = set(split.train["userId"].unique().to_list())
    assert 3 not in train_users


def test_max_test_per_user_cap_is_earliest_first():
    interactions = make_synthetic_interactions()
    split = temporal_split(
        interactions, cutoff_timestamp=50, max_test_per_user=3, min_train_interactions=5
    )

    user1_test = split.test.filter(pl.col("userId") == 1).sort("timestamp")
    assert user1_test.height == 3
    assert user1_test["timestamp"].to_list() == [50, 55, 60]


def test_per_user_split_looks_valid_but_leaks_globally():
    user1 = [{"userId": 1, "movieId": 100 + i, "timestamp": t} for i, t in enumerate(range(0, 11))]
    user2 = [{"userId": 2, "movieId": 999, "timestamp": t} for t in range(20, 31)]
    interactions = pl.DataFrame(user1 + user2).cast(
        {"userId": pl.Int32, "movieId": pl.Int32, "timestamp": pl.Int64}
    )

    def per_user_leave_last_n(df: pl.DataFrame, n: int = 2):
        train_parts, test_parts = [], []
        for uid in df["userId"].unique().to_list():
            u = df.filter(pl.col("userId") == uid).sort("timestamp")
            train_parts.append(u.head(u.height - n))
            test_parts.append(u.tail(n))
        return pl.concat(train_parts), pl.concat(test_parts)

    per_user_train, per_user_test = per_user_leave_last_n(interactions, n=2)

    for uid in interactions["userId"].unique().to_list():
        u_train_max = per_user_train.filter(pl.col("userId") == uid)["timestamp"].max()
        u_test_min = per_user_test.filter(pl.col("userId") == uid)["timestamp"].min()
        assert u_train_max < u_test_min

    user1_test_start = per_user_test.filter(pl.col("userId") == 1)["timestamp"].min()
    leaked_rows = per_user_train.filter(pl.col("timestamp") >= user1_test_start)
    assert leaked_rows.height > 0

    fake_split = SplitResult(train=per_user_train, test=per_user_test, cutoff_timestamp=user1_test_start)
    with pytest.raises(AssertionError):
        assert_no_leakage(fake_split)

    real_split = temporal_split(interactions, cutoff_timestamp=user1_test_start, min_train_interactions=1)
    assert real_split.train["userId"].unique().to_list() == [1]
    assert_no_leakage(real_split)


def test_assert_no_leakage_catches_corrupted_split():
    interactions = make_synthetic_interactions()
    split = temporal_split(interactions, cutoff_timestamp=50, min_train_interactions=5)

    corrupted_train = pl.concat(
        [split.train, pl.DataFrame({"userId": [1], "movieId": [999], "timestamp": [999]})
         .cast({"userId": pl.Int32, "movieId": pl.Int32, "timestamp": pl.Int64})]
    )
    corrupted = split.__class__(train=corrupted_train, test=split.test, cutoff_timestamp=split.cutoff_timestamp)

    with pytest.raises(AssertionError):
        assert_no_leakage(corrupted)
