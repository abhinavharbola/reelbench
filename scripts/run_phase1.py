import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src.config import DATA_DIR as PROCESSED_DIR
from src.config import MODELS_DIR, RESULTS_DIR
from src.data.split import assert_no_leakage, save_split_meta, temporal_split
from src.eval.harness import describe_test_population, evaluate_model, format_population
from src.eval.results_table import upsert_results
from src.eval.tracking import log_model_run
from src.models.baseline import ItemItemCF, PopularityModel
from src.models.mf import MatrixFactorizationModel
from src.ranking.features import build_item_genre_map

K_VALUES = (10, 20)


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
    save_split_meta(PROCESSED_DIR, split.cutoff_timestamp)
    print(format_population(describe_test_population(split.train, split.test)))

    results = []

    pop = PopularityModel()
    pop.fit(split.train)
    pop_metrics = evaluate_model("popularity", pop, split.test, catalog_size, item_genres, ks=K_VALUES)
    results.append(pop_metrics)
    with log_model_run("popularity", params={}, metrics=pop_metrics):
        pass
    with open(MODELS_DIR / "popularity.pkl", "wb") as f:
        pickle.dump(pop, f)
    print("popularity done")

    cf = ItemItemCF(top_k=50)
    cf.fit(split.train)
    cf_metrics = evaluate_model("item_item_cf", cf, split.test, catalog_size, item_genres, ks=K_VALUES)
    results.append(cf_metrics)
    with log_model_run("item_item_cf", params={"top_k": 50}, metrics=cf_metrics):
        pass
    with open(MODELS_DIR / "item_item_cf.pkl", "wb") as f:
        pickle.dump(cf, f)
    print("item-item CF done")

    mf = MatrixFactorizationModel(method=args.mf_method, factors=args.mf_factors, iterations=args.mf_iterations)
    mf.fit(split.train)
    mf_metrics = evaluate_model(args.mf_method, mf, split.test, catalog_size, item_genres, ks=K_VALUES)
    results.append(mf_metrics)
    with log_model_run(args.mf_method, params={"factors": args.mf_factors, "iterations": args.mf_iterations}, metrics=mf_metrics):
        pass
    with open(MODELS_DIR / f"{args.mf_method}.pkl", "wb") as f:
        pickle.dump(mf, f)
    print(f"{args.mf_method.upper()} done")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table_path = RESULTS_DIR / "comparison_table.csv"
    owned_models = ["popularity", "item_item_cf", args.mf_method]
    combined = upsert_results(table_path, results, owned_models)

    print(combined)
    print(f"\nwritten to {table_path}")
    print(f"models written to {MODELS_DIR}")


if __name__ == "__main__":
    main()
