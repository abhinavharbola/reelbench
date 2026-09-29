import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src.data.split import assert_no_leakage, temporal_split
from src.eval.metrics import evaluate_all
from src.eval.results_table import upsert_results
from src.eval.tracking import log_model_run
from src.models.baseline import ItemItemCF, PopularityModel
from src.models.mf import MatrixFactorizationModel
from src.ranking.features import build_item_genre_map

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = PROCESSED_DIR / "models"
RESULTS_DIR = Path("results")
K_VALUES = (10, 20)
TOP_K_FOR_RECS = max(K_VALUES)


def evaluate_model(name: str, model, test: pl.DataFrame, catalog_size: int, item_genres: dict) -> dict:
    test_by_user = (
        test.group_by("userId")
        .agg(pl.col("movieId"))
        .to_dict(as_series=False)
    )
    user_ids = test_by_user["userId"]
    relevant_lists = [set(items) for items in test_by_user["movieId"]]

    all_recommended = [model.recommend(uid, k=TOP_K_FOR_RECS) for uid in user_ids]

    metrics = evaluate_all(all_recommended, relevant_lists, catalog_size, item_genres, ks=K_VALUES)
    metrics["model"] = name
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mf-method", choices=["als", "bpr"], default="als")
    parser.add_argument("--mf-factors", type=int, default=64)
    parser.add_argument("--mf-iterations", type=int, default=15)
    return parser.parse_args()


def main():
    args = parse_args()

    interactions = pl.read_parquet(PROCESSED_DIR / "interactions.parquet")
    movies = pl.read_parquet(PROCESSED_DIR / "movies.parquet")
    item_genres = build_item_genre_map(movies)

    split = temporal_split(interactions)
    assert_no_leakage(split)
    catalog_size = split.train["movieId"].n_unique()
    print(f"train: {split.train.height:,} rows, test: {split.test.height:,} rows, "
          f"cutoff: {split.cutoff_timestamp}, catalog size: {catalog_size:,}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    split.train.write_parquet(PROCESSED_DIR / "train.parquet")
    split.test.write_parquet(PROCESSED_DIR / "test.parquet")

    results = []

    pop = PopularityModel()
    pop.fit(split.train)
    pop_metrics = evaluate_model("popularity", pop, split.test, catalog_size, item_genres)
    results.append(pop_metrics)
    with log_model_run("popularity", params={}, metrics=pop_metrics):
        pass
    with open(MODELS_DIR / "popularity.pkl", "wb") as f:
        pickle.dump(pop, f)
    print("popularity done")

    cf = ItemItemCF(top_k=50)
    cf.fit(split.train)
    cf_metrics = evaluate_model("item_item_cf", cf, split.test, catalog_size, item_genres)
    results.append(cf_metrics)
    with log_model_run("item_item_cf", params={"top_k": 50}, metrics=cf_metrics):
        pass
    with open(MODELS_DIR / "item_item_cf.pkl", "wb") as f:
        pickle.dump(cf, f)
    print("item-item CF done")

    mf = MatrixFactorizationModel(method=args.mf_method, factors=args.mf_factors, iterations=args.mf_iterations)
    mf.fit(split.train)
    mf_metrics = evaluate_model(args.mf_method, mf, split.test, catalog_size, item_genres)
    results.append(mf_metrics)
    with log_model_run(args.mf_method, params={"factors": args.mf_factors, "iterations": args.mf_iterations}, metrics=mf_metrics):
        pass
    with open(MODELS_DIR / f"{args.mf_method}.pkl", "wb") as f:
        pickle.dump(mf, f)
    print(f"{args.mf_method.upper()} done")

    RESULTS_DIR.mkdir(exist_ok=True)
    table_path = RESULTS_DIR / "comparison_table.csv"
    owned_models = ["popularity", "item_item_cf", args.mf_method]
    combined = upsert_results(table_path, results, owned_models)

    print(combined)
    print(f"\nwritten to {table_path}")
    print(f"models written to {MODELS_DIR}")


if __name__ == "__main__":
    main()
