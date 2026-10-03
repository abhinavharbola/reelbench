import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src.config import DATA_DIR, RESULTS_DIR
from src.data.split import build_user_seen_items, load_cutoff
from src.eval.harness import describe_test_population, evaluate_model, format_population
from src.eval.results_table import upsert_results
from src.eval.tracking import log_model_run
from src.ranking.features import build_feature_context, build_item_genre_map
from src.ranking.pipeline import EMBEDDING_MODELS, NORMALIZE_BY_MODEL, EmbeddingRankerPipeline

K_VALUES = (10, 20)


def main():
    train_path = DATA_DIR / "train.parquet"
    test_path = DATA_DIR / "test.parquet"
    movies_path = DATA_DIR / "movies.parquet"
    table_path = RESULTS_DIR / "comparison_table.csv"

    for p in [train_path, test_path, movies_path]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run scripts/run_phase1.py first.")

    train = pl.read_parquet(train_path)
    test = pl.read_parquet(test_path)
    movies = pl.read_parquet(movies_path)

    catalog_size = train["movieId"].n_unique()
    item_genres = build_item_genre_map(movies)
    feature_context = build_feature_context(
        train, item_genres, reference_timestamp=load_cutoff(DATA_DIR, train)
    )
    seen_by_user = build_user_seen_items(train)
    print(format_population(describe_test_population(train, test)))

    new_rows = []
    for prefix in EMBEDDING_MODELS:
        try:
            pipeline = EmbeddingRankerPipeline.from_artifacts(
                DATA_DIR, prefix, NORMALIZE_BY_MODEL[prefix], feature_context, seen_by_user, train
            )
        except FileNotFoundError as exc:
            print(f"skipping {prefix}: {exc}. Run scripts/build_ui_artifacts.py first.")
            continue

        metrics = evaluate_model(prefix, pipeline, test, catalog_size, item_genres, ks=K_VALUES)
        new_rows.append(metrics)
        with log_model_run(prefix, params={"stage": "retrieval+ranking"}, metrics=metrics):
            pass
        print(f"{prefix} done: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "model"))

    if not new_rows:
        print("no pipeline models scored, nothing to write.")
        return

    scored = [row["model"] for row in new_rows]
    combined = upsert_results(table_path, new_rows, scored)
    print(combined)
    print(f"\nwritten to {table_path}")


if __name__ == "__main__":
    main()
