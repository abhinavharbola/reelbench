import polars as pl

from src.data.split import carve_ranker_supervision_split
from src.ranking.features import build_feature_context


def make_synthetic_train() -> pl.DataFrame:
    rows = []
    for uid in (1, 2):
        for ts in range(0, 51, 5):
            rows.append({"userId": uid, "movieId": 100 + uid, "timestamp": ts})
    for uid in (1, 2):
        for ts in (60, 65, 70, 75, 80, 85, 90):
            rows.append({"userId": uid, "movieId": 999, "timestamp": ts})
    return pl.DataFrame(rows).cast({"userId": pl.Int32, "movieId": pl.Int32, "timestamp": pl.Int64})


def test_ranker_feature_context_excludes_label_window_popularity():
    train = make_synthetic_train()

    ranker_split = carve_ranker_supervision_split(train, val_quantile=0.5, min_seen_interactions=1)
    assert ranker_split.cutoff_timestamp < train["timestamp"].max()

    leaky_context = build_feature_context(train, item_genres={})
    fixed_context = build_feature_context(ranker_split.train, item_genres={})

    leaky_popularity = leaky_context["item_popularity"].get(999, 0)
    fixed_popularity = fixed_context["item_popularity"].get(999, 0)

    assert leaky_popularity == 14
    assert fixed_popularity < leaky_popularity
    assert fixed_popularity == sum(
        1 for row in ranker_split.train.iter_rows(named=True) if row["movieId"] == 999
    )


def test_ranker_feature_context_uses_explicit_reference_timestamp():
    train = make_synthetic_train()
    ranker_split = carve_ranker_supervision_split(train, val_quantile=0.5, min_seen_interactions=1)

    context = build_feature_context(
        ranker_split.train, item_genres={}, reference_timestamp=ranker_split.cutoff_timestamp
    )
    stats = context["user_stats"][1]

    max_ts_for_user = ranker_split.train.filter(pl.col("userId") == 1)["timestamp"].max()
    expected_days = max(ranker_split.cutoff_timestamp - max_ts_for_user, 0) / 86400.0
    assert stats["days_since_last_interaction"] == expected_days


def test_feature_context_defaults_to_input_max_timestamp_when_no_reference_given():
    train = make_synthetic_train()
    context = build_feature_context(train, item_genres={})
    last_ts = train.filter(pl.col("userId") == 1)["timestamp"].max()
    expected_days = max(train["timestamp"].max() - last_ts, 0) / 86400.0
    assert context["user_stats"][1]["days_since_last_interaction"] == expected_days
