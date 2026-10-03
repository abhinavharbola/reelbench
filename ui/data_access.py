import json
import pickle

import polars as pl
import streamlit as st

from src.config import DATA_DIR, MODELS_DIR, RESULTS_DIR
from src.data.split import build_user_seen_items, load_cutoff
from src.ranking.features import build_feature_context, build_item_genre_map
from src.ranking.pipeline import EMBEDDING_MODELS, NORMALIZE_BY_MODEL, EmbeddingRankerPipeline

BASELINE_FILES = [
    ("popularity", "popularity.pkl"),
    ("item_item_cf", "item_item_cf.pkl"),
    ("als", "als.pkl"),
    ("bpr", "bpr.pkl"),
]


@st.cache_data
def load_comparison_table() -> pl.DataFrame | None:
    path = RESULTS_DIR / "comparison_table.csv"
    if not path.exists():
        return None
    return pl.read_csv(path)


@st.cache_data
def load_personas() -> list[dict]:
    path = DATA_DIR / "personas.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


@st.cache_data
def load_movie_lookup() -> dict[int, tuple[str, str]]:
    path = DATA_DIR / "movies.parquet"
    if not path.exists():
        return {}
    movies = pl.read_parquet(path, columns=["movieId", "title", "genres"])
    return {row["movieId"]: (row["title"], row["genres"]) for row in movies.iter_rows(named=True)}


@st.cache_data
def load_user_id_index() -> dict | None:
    path = DATA_DIR / "train.parquet"
    if not path.exists():
        return None
    ids = pl.read_parquet(path, columns=["userId"])["userId"].unique().to_list()
    if not ids:
        return None
    return {"ids": set(ids), "min": min(ids), "max": max(ids), "count": len(ids)}


@st.cache_resource
def load_model_registry() -> dict:
    registry: dict = {"_errors": {}}

    for name, filename in BASELINE_FILES:
        path = MODELS_DIR / filename
        if path.exists():
            with open(path, "rb") as f:
                registry[name] = pickle.load(f)

    train_path = DATA_DIR / "train.parquet"
    movies_path = DATA_DIR / "movies.parquet"
    if train_path.exists() and movies_path.exists():
        train = pl.read_parquet(train_path)
        movies = pl.read_parquet(movies_path)
        item_genres = build_item_genre_map(movies)
        feature_context = build_feature_context(
            train, item_genres, reference_timestamp=load_cutoff(DATA_DIR, train)
        )
        seen_by_user = build_user_seen_items(train)

        for name in EMBEDDING_MODELS:
            try:
                registry[name] = EmbeddingRankerPipeline.from_artifacts(
                    DATA_DIR, name, NORMALIZE_BY_MODEL[name], feature_context, seen_by_user, train
                )
            except FileNotFoundError:
                continue
            except ValueError as exc:
                registry["_errors"][name] = str(exc)

    return registry


def get_recommendations(model_name: str, user_id: int, k: int = 10) -> list[dict]:
    registry = load_model_registry()
    movie_lookup = load_movie_lookup()
    if model_name not in registry or not movie_lookup:
        return []

    model = registry[model_name]
    if model_name in EMBEDDING_MODELS:
        pairs = model.recommend_scored(user_id, k)
    else:
        pairs = [(movie_id, None) for movie_id in model.recommend(user_id, k)]

    out = []
    for movie_id, score in pairs:
        entry = movie_lookup.get(movie_id)
        if entry is None:
            continue
        out.append({"movieId": movie_id, "title": entry[0], "genres": entry[1], "score": score})
    return out
