import json
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from src.data.personas import curate_personas
from src.data.split import (
    RANKER_SPLIT_SUFFIX,
    assert_no_leakage,
    build_user_seen_items,
    carve_ranker_supervision_split,
    save_split_meta,
    temporal_split,
)
from src.eval.harness import evaluate_model
from src.models.baseline import ItemItemCF, PopularityModel
from src.models.mf import MatrixFactorizationModel
from src.ranking.build import build_model_artifacts
from src.ranking.features import build_feature_context, build_item_genre_map
from src.ranking.pipeline import EMBEDDING_MODELS, NORMALIZE_BY_MODEL, EmbeddingRankerPipeline

random.seed(42)
np.random.seed(42)

DATA_DIR = Path("data/demo")
RESULTS_DIR = Path("results/demo")
MODELS_DIR = DATA_DIR / "models"

ADJECTIVES = ["Silent", "Midnight", "Broken", "Last", "Hidden", "Distant", "Golden", "Crooked", "Endless", "Quiet"]
NOUNS = ["Harbor", "Signal", "Garden", "Machine", "River", "Letter", "Orchard", "Station", "Mirror", "Horizon"]
GENRES_POOL = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Horror", "Thriller", "Documentary", "Animation"]

N_USERS = 400
N_ITEMS = 350
K_VALUES = (10, 20)


def generate_synthetic_data():
    movies_rows = []
    for item_id in range(1, N_ITEMS + 1):
        title = f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)} ({1970 + item_id % 55})"
        genres = "|".join(random.sample(GENRES_POOL, random.randint(1, 3)))
        movies_rows.append({"movieId": item_id, "title": title, "genres": genres})
    movies = pl.DataFrame(movies_rows).cast({"movieId": pl.Int32})

    item_weights = np.random.pareto(1.5, N_ITEMS) + 0.1
    item_weights = item_weights / item_weights.sum()

    interactions = []
    for user_id in range(1, N_USERS + 1):
        n = random.randint(15, 60)
        items = np.random.choice(range(1, N_ITEMS + 1), size=n, replace=False, p=item_weights)
        for i, item_id in enumerate(sorted(items)):
            interactions.append({"userId": user_id, "movieId": int(item_id), "timestamp": 1_600_000_000 + i * 86400 + random.randint(0, 3600)})

    interactions_df = pl.DataFrame(interactions).cast({"userId": pl.Int32, "movieId": pl.Int32, "timestamp": pl.Int64})
    return interactions_df, movies


def build_mock_neural_embeddings(als_model: MatrixFactorizationModel, item_ids: list[int], user_ids: list[int]):
    item_idx_map = {v: k for k, v in als_model.idx_to_item_id.items()}

    item_embs, valid_item_ids = [], []
    for item_id in item_ids:
        if item_id in item_idx_map:
            base = als_model.model.item_factors[item_idx_map[item_id]]
            item_embs.append(base + np.random.normal(0, 0.15, size=base.shape))
            valid_item_ids.append(item_id)

    user_embs, valid_user_ids = [], []
    for user_id in user_ids:
        if user_id in als_model.user_id_to_idx:
            base = als_model.model.user_factors[als_model.user_id_to_idx[user_id]]
            user_embs.append(base + np.random.normal(0, 0.15, size=base.shape))
            valid_user_ids.append(user_id)

    item_df = pl.DataFrame({"movieId": valid_item_ids, "embedding": [e.tolist() for e in item_embs]})
    user_df = pl.DataFrame({"userId": valid_user_ids, "embedding": [e.tolist() for e in user_embs]})
    return item_df, user_df


def write_mock_embeddings(prefix: str, train: pl.DataFrame) -> None:
    als = MatrixFactorizationModel(method="als", factors=32, iterations=15)
    als.fit(train)
    item_ids = train["movieId"].unique().sort().to_list()
    user_ids = train["userId"].unique().sort().to_list()
    item_df, user_df = build_mock_neural_embeddings(als, item_ids, user_ids)
    item_df.write_parquet(DATA_DIR / f"{prefix}_item_embeddings.parquet")
    user_df.write_parquet(DATA_DIR / f"{prefix}_user_embeddings.parquet")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    interactions, movies = generate_synthetic_data()
    interactions.write_parquet(DATA_DIR / "interactions.parquet")
    movies.write_parquet(DATA_DIR / "movies.parquet")

    split = temporal_split(interactions, test_quantile=0.85, min_train_interactions=5)
    split.train.write_parquet(DATA_DIR / "train.parquet")
    split.test.write_parquet(DATA_DIR / "test.parquet")
    save_split_meta(DATA_DIR, split.cutoff_timestamp)
    print(f"train: {split.train.height}, test: {split.test.height}")

    catalog_size = split.train["movieId"].n_unique()
    item_genres = build_item_genre_map(movies)

    results = []

    pop = PopularityModel()
    pop.fit(split.train)
    results.append(evaluate_model("popularity", pop, split.test, catalog_size, item_genres, ks=K_VALUES))
    with open(MODELS_DIR / "popularity.pkl", "wb") as f:
        pickle.dump(pop, f)

    cf = ItemItemCF(top_k=30)
    cf.fit(split.train)
    results.append(evaluate_model("item_item_cf", cf, split.test, catalog_size, item_genres, ks=K_VALUES))
    with open(MODELS_DIR / "item_item_cf.pkl", "wb") as f:
        pickle.dump(cf, f)

    als = MatrixFactorizationModel(method="als", factors=32, iterations=15)
    als.fit(split.train)
    results.append(evaluate_model("als", als, split.test, catalog_size, item_genres, ks=K_VALUES))
    with open(MODELS_DIR / "als.pkl", "wb") as f:
        pickle.dump(als, f)

    print("baselines done, building demo embedding-based models")

    ranker_split = carve_ranker_supervision_split(split.train, val_quantile=0.8)
    assert_no_leakage(ranker_split)
    ranker_seen_by_user = build_user_seen_items(ranker_split.train)
    ranker_positives = build_user_seen_items(ranker_split.test)
    full_seen_by_user = build_user_seen_items(split.train)

    eval_feature_context = build_feature_context(
        split.train, item_genres, reference_timestamp=split.cutoff_timestamp
    )
    ranker_feature_context = build_feature_context(
        ranker_split.train, item_genres, reference_timestamp=ranker_split.cutoff_timestamp
    )

    for prefix in EMBEDDING_MODELS:
        write_mock_embeddings(prefix, split.train)
        write_mock_embeddings(f"{prefix}{RANKER_SPLIT_SUFFIX}", ranker_split.train)

        build_model_artifacts(
            DATA_DIR, prefix, ranker_split, ranker_feature_context,
            ranker_seen_by_user, ranker_positives, pool_size=30, num_boost_round=50,
        )

        pipeline = EmbeddingRankerPipeline.from_artifacts(
            DATA_DIR, prefix, NORMALIZE_BY_MODEL[prefix], eval_feature_context, full_seen_by_user, split.train
        )
        results.append(evaluate_model(prefix, pipeline, split.test, catalog_size, item_genres, ks=K_VALUES))
        print(f"{prefix} demo model done")

    table = pl.DataFrame(results)
    table = table.select(["model"] + [c for c in table.columns if c != "model"])
    table.write_csv(RESULTS_DIR / "comparison_table.csv")
    print(table)

    personas = curate_personas(split.train, movies)
    (DATA_DIR / "personas.json").write_text(json.dumps(personas, indent=2))
    print(f"{len(personas)} personas cached")

    print("\ndemo artifacts ready. run:")
    print("  RECSYS_DATA_DIR=data/demo RECSYS_RESULTS_DIR=results/demo streamlit run ui/app.py")


if __name__ == "__main__":
    main()
