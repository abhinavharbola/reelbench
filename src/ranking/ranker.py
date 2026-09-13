"""
LightGBM re-ranker on top of retrieved candidates. Uses LambdaRank
(learning-to-rank), grouped by user, so training directly optimizes ranking
quality rather than pointwise classification.
"""

from pathlib import Path

import lightgbm as lgb
import polars as pl

from src.ranking.features import FEATURE_COLUMNS


def build_training_table(
    per_user_features: dict[int, pl.DataFrame],  # userId -> build_features_for_candidates() output
    positive_items: dict[int, set],  # userId -> ground-truth positive movieIds (from a held-out slice of train)
) -> pl.DataFrame:
    """Concatenates per-user feature tables and attaches a binary label:
    1 if the candidate is one of that user's known positives, else 0.

    Skips any user whose feature table came back empty (height 0, e.g. a
    retrieval call that returned zero candidates -- can happen for a
    malformed embedding such as a NaN vector, which FAISS surfaces as an
    all-invalid-index result). One bad embedding shouldn't crash training
    for every other user; the skip is silent in the return value but the
    caller can compare `len(per_user_features)` to the row count if it
    wants to know how many users were dropped."""
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
    """training_table must be sorted/grouped by userId (LightGBM ranking
    requires contiguous groups) with FEATURE_COLUMNS + label columns present."""
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
        "metric": "ndcg",
        "ndcg_eval_at": [10, 20],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
    }

    return lgb.train(params, train_set, num_boost_round=num_boost_round)


def rank_candidates(model: lgb.Booster, features: pl.DataFrame, top_k: int) -> list[tuple[int, float]]:
    """features: build_features_for_candidates() output for one user.
    Returns [(movieId, model_score), ...] ranked by predicted relevance,
    highest first.

    Returns the ranker's own predicted score, not the raw
    embedding_similarity feature that went into it -- the ranking order
    comes from this LightGBM prediction (a combination of similarity,
    popularity, recency, genre match, and user stats), so a caller that
    displayed embedding_similarity as "the score" would show a value that
    doesn't correspond to the order the list is actually sorted in. This
    was previously an actual bug: src/serving/app.py and
    ui/data_access.py both looked up embedding_similarity per movieId
    after ranking and labeled it "score", which does not
    necessarily decrease monotonically down the list."""
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
