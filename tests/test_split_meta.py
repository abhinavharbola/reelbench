import polars as pl

from src.data.split import (
    carve_ranker_supervision_split,
    load_cutoff,
    ranker_train_view,
    save_split_meta,
)


def make_train() -> pl.DataFrame:
    rows = []
    for uid in (1, 2, 3):
        for ts in range(0, 100, 5):
            rows.append({"userId": uid, "movieId": 100 + (ts % 7), "timestamp": ts})
    return pl.DataFrame(rows).cast({"userId": pl.Int32, "movieId": pl.Int32, "timestamp": pl.Int64})


def test_ranker_train_view_matches_carved_split_train():
    train = make_train()
    view = ranker_train_view(train)
    carved = carve_ranker_supervision_split(train).train
    assert view.sort(["userId", "timestamp"]).equals(carved.sort(["userId", "timestamp"]))
    assert view.height < train.height


def test_split_meta_round_trip_and_fallback(tmp_path):
    train = make_train()
    assert load_cutoff(tmp_path, train) == int(train["timestamp"].max())
    save_split_meta(tmp_path, 12345)
    assert load_cutoff(tmp_path, train) == 12345
