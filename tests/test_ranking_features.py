import polars as pl

from src.data.split import carve_ranker_supervision_split
from src.ranking.features import build_feature_context


def make_synthetic_train() -> pl.DataFrame:
    """
    2 users, 3 items, timestamps 0..99.

    Item 999 only gets watched in the *late* half of train (timestamps
    60..90), by several users -- simulating an item whose popularity
    spikes only inside what carve_ranker_supervision_split() will carve
    out as the label window.
    """
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

    assert leaky_popularity == 14  # all 14 watches of item 999 across both users
    assert fixed_popularity < leaky_popularity
    assert fixed_popularity == sum(
        1 for row in ranker_split.train.iter_rows(named=True) if row["movieId"] == 999
    )


def test_ranker_feature_context_reference_timestamp_matches_its_own_input():
    train = make_synthetic_train()
    ranker_split = carve_ranker_supervision_split(train, val_quantile=0.5, min_seen_interactions=1)

    context = build_feature_context(ranker_split.train, item_genres={})
    stats = context["user_stats"][1]

    reference_timestamp = ranker_split.train.select(pl.col("timestamp").max()).item()
    max_ts_for_user = ranker_split.train.filter(pl.col("userId") == 1)["timestamp"].max()
    expected_days = max(reference_timestamp - max_ts_for_user, 0) / 86400.0
    assert stats["days_since_last_interaction"] == expected_days
