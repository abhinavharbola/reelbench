import polars as pl

from src.eval.metrics import evaluate_all


def evaluate_model(
    name: str,
    model,
    test: pl.DataFrame,
    catalog_size: int,
    item_genres: dict,
    ks: tuple[int, ...] = (10, 20),
) -> dict:
    test_by_user = test.group_by("userId").agg(pl.col("movieId")).to_dict(as_series=False)
    user_ids = test_by_user["userId"]
    relevant_lists = [set(items) for items in test_by_user["movieId"]]

    top_k = max(ks)
    all_recommended = [model.recommend(uid, k=top_k) for uid in user_ids]

    metrics = evaluate_all(all_recommended, relevant_lists, catalog_size, item_genres, ks=ks)
    metrics["model"] = name
    return metrics


def describe_test_population(train: pl.DataFrame, test: pl.DataFrame) -> dict:
    train_items = set(train["movieId"].unique().to_list())
    n_interactions = test.height
    reachable = test.filter(pl.col("movieId").is_in(train_items)).height
    return {
        "eval_users": test["userId"].n_unique(),
        "train_users": train["userId"].n_unique(),
        "test_interactions": n_interactions,
        "test_interactions_in_train_catalog": reachable,
        "reachable_fraction": reachable / n_interactions if n_interactions else 0.0,
    }


def format_population(population: dict) -> str:
    return (
        f"evaluated users: {population['eval_users']:,} of {population['train_users']:,} train users, "
        f"test interactions: {population['test_interactions']:,}, "
        f"of which {population['reachable_fraction']:.1%} involve items present in the train catalog"
    )
