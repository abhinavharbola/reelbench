import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

RANKER_SPLIT_SUFFIX = "_rs"
SPLIT_META_FILENAME = "split_meta.json"


@dataclass
class SplitResult:
    train: pl.DataFrame
    test: pl.DataFrame
    cutoff_timestamp: int


def compute_global_cutoff(interactions: pl.DataFrame, test_quantile: float = 0.9) -> int:
    cutoff = interactions.select(pl.col("timestamp").quantile(test_quantile)).item()
    return int(cutoff)


def temporal_split(
    interactions: pl.DataFrame,
    cutoff_timestamp: int | None = None,
    test_quantile: float = 0.9,
    max_test_per_user: int = 10,
    min_train_interactions: int = 5,
) -> SplitResult:
    if cutoff_timestamp is None:
        cutoff_timestamp = compute_global_cutoff(interactions, test_quantile)

    train = interactions.filter(pl.col("timestamp") < cutoff_timestamp)
    test_pool = interactions.filter(pl.col("timestamp") >= cutoff_timestamp)

    train_counts = train.group_by("userId").agg(pl.len().alias("n_train"))
    eligible_users = train_counts.filter(pl.col("n_train") >= min_train_interactions)["userId"].to_list()

    train = train.filter(pl.col("userId").is_in(eligible_users))

    test = (
        test_pool.filter(pl.col("userId").is_in(eligible_users))
        .sort(["userId", "timestamp"])
        .group_by("userId", maintain_order=True)
        .head(max_test_per_user)
    )

    return SplitResult(train=train, test=test, cutoff_timestamp=cutoff_timestamp)


def build_user_seen_items(interactions: pl.DataFrame) -> dict[int, set]:
    by_user = interactions.group_by("userId").agg(pl.col("movieId")).to_dict(as_series=False)
    return {uid: set(items) for uid, items in zip(by_user["userId"], by_user["movieId"])}


def carve_ranker_supervision_split(
    train: pl.DataFrame,
    val_quantile: float = 0.9,
    min_seen_interactions: int = 1,
) -> SplitResult:
    return temporal_split(
        train,
        test_quantile=val_quantile,
        max_test_per_user=10_000,
        min_train_interactions=min_seen_interactions,
    )


def assert_no_leakage(split: SplitResult) -> None:
    max_train_ts = split.train.select(pl.col("timestamp").max()).item()
    min_test_ts = split.test.select(pl.col("timestamp").min()).item()

    if max_train_ts is not None and max_train_ts >= split.cutoff_timestamp:
        raise AssertionError("train contains interactions at/after the cutoff")
    if min_test_ts is not None and min_test_ts < split.cutoff_timestamp:
        raise AssertionError("test contains interactions before the cutoff")

    train_users = set(split.train["userId"].unique().to_list())
    test_users = set(split.test["userId"].unique().to_list())
    if not test_users.issubset(train_users):
        raise AssertionError("test contains users absent from train")


def ranker_train_view(train: pl.DataFrame) -> pl.DataFrame:
    return carve_ranker_supervision_split(train).train


def save_split_meta(processed_dir: Path, cutoff_timestamp: int) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / SPLIT_META_FILENAME).write_text(json.dumps({"cutoff_timestamp": int(cutoff_timestamp)}))


def load_cutoff(processed_dir: Path, train: pl.DataFrame) -> int:
    path = processed_dir / SPLIT_META_FILENAME
    if path.exists():
        return int(json.loads(path.read_text())["cutoff_timestamp"])
    return int(train.select(pl.col("timestamp").max()).item())
