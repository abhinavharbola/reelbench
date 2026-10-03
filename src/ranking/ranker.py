from pathlib import Path

import lightgbm as lgb
import polars as pl

from src.ranking.features import FEATURE_COLUMNS


def build_training_table(
    per_user_features: dict[int, pl.DataFrame],
    positive_items: dict[int, set],
) -> pl.DataFrame:
    tables = []
    for uid, feats in per_user_features.items():
        if feats.height == 0:
            continue
        positives = positive_items.get(uid, set())
        labeled = feats.with_columns(
            pl.col("movieId").is_in(positives).cast(pl.Int8).alias("label")
        )
        tables.append(labeled)
    return pl.concat(tables) if tables else pl.DataFrame()


def train_ranker(training_table: pl.DataFrame, num_boost_round: int = 200) -> lgb.Booster:
    if training_table.height == 0:
        raise ValueError("ranker training table is empty: no candidates were generated for any user")
    if int(training_table["label"].sum()) == 0:
        raise ValueError("ranker training table has no positive labels: retrieved candidates never hit a held-out item")
    training_table = training_table.sort("userId")

    group_sizes = (
        training_table.group_by("userId", maintain_order=True)
        .agg(pl.len().alias("n"))["n"]
        .to_list()
    )

    X = training_table.select(FEATURE_COLUMNS).to_numpy()
    y = training_table["label"].to_numpy()

    train_set = lgb.Dataset(X, label=y, group=group_sizes, feature_name=FEATURE_COLUMNS)

    params = {
        "objective": "lambdarank",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
    }

    return lgb.train(params, train_set, num_boost_round=num_boost_round)


def rank_candidates(model: lgb.Booster, features: pl.DataFrame, top_k: int) -> list[tuple[int, float]]:
    if features.height == 0:
        return []
    X = features.select(FEATURE_COLUMNS).to_numpy()
    scores = model.predict(X)
    ranked = features.with_columns(pl.Series("score", scores)).sort("score", descending=True)
    top = ranked.head(top_k)
    return list(zip(top["movieId"].to_list(), top["score"].to_list()))


def save_model(model: lgb.Booster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def load_model(path: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))
