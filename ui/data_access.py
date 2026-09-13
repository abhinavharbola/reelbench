"""
Data access layer for the Streamlit UI. Reads only local cached artifacts
(parquet, pickled models, FAISS index, LightGBM model, results CSV) -- no
live external calls, no recomputation of metrics in the UI, matching the
project spec's explicit constraints on this screen.

Falls back to nothing gracefully: if an artifact is missing, callers get
None/[] rather than a crash, and app.py surfaces a clear "run this script
first" message instead of a stack trace.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import streamlit as st

from src.data.split import build_user_seen_items
from src.ranking.features import build_feature_context, build_features_for_candidates, build_item_genre_map
from src.ranking.ranker import load_model as load_ranker_model
from src.ranking.ranker import rank_candidates
from src.retrieval.faiss_index import FaissRetriever

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
MODELS_DIR = DATA_DIR / "models"


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
def load_movies() -> pl.DataFrame | None:
    path = DATA_DIR / "movies.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


@st.cache_data
def load_all_user_ids() -> list[int]:
    path = DATA_DIR / "train.parquet"
    if not path.exists():
        return []
    train = pl.read_parquet(path)
    return sorted(train["userId"].unique().to_list())


class EmbeddingRankerRecommender:
    """Shared inference path for the two embedding-based models (two-tower,
    SASRec): FAISS retrieval over cached embeddings, then LightGBM
    re-ranking. Mirrors src/serving/app.py's logic, exposed here as a
    reusable class since the UI needs it for two separate models."""

    def __init__(self, user_embeddings_path: Path, index_path: Path, ranker_path: Path):
        self.retriever = FaissRetriever()
        self.retriever.load(index_path)
        self.ranker = load_ranker_model(ranker_path)

        user_emb = pl.read_parquet(user_embeddings_path)
        self.user_emb_lookup = dict(zip(user_emb["userId"].to_list(), user_emb["embedding"].to_list()))

    def recommend(self, user_id: int, k: int, feature_context: dict, seen_items: set | None = None) -> list[tuple[int, float]]:
        if user_id not in self.user_emb_lookup:
            return []
        user_vec = np.array(self.user_emb_lookup[user_id], dtype=np.float32)
        candidates = self.retriever.query(user_vec, top_n=max(k * 5, 50))

        features = build_features_for_candidates(user_id, candidates, **feature_context, seen_items=seen_items)
        # rank_candidates returns the ranker's own predicted score, which
        # is what actually determined this order -- not the raw FAISS
        # embedding similarity, which doesn't necessarily decrease
        # monotonically down the ranked list (see ranker.py).
        return rank_candidates(self.ranker, features, top_k=k)


@st.cache_resource
def load_model_registry() -> dict:
    """Loads whatever model artifacts exist on disk. Missing artifacts are
    simply omitted from the returned dict -- the recommendations screen
    only offers side-by-side comparison for models that are actually
    trained and cached."""
    registry = {}

    for name, filename in [
        ("popularity", "popularity.pkl"),
        ("item_item_cf", "item_item_cf.pkl"),
        ("als", "als.pkl"),
        ("bpr", "bpr.pkl"),
    ]:
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
        registry["_feature_context"] = build_feature_context(train, item_genres)
        # every user's full train history -- so the two embedding-based
        # models never recommend something the user has already watched,
        # matching popularity/item-item-CF/ALS which already exclude it
        registry["_seen_by_user"] = build_user_seen_items(train)

        for name, prefix in [("two_tower", "two_tower"), ("sasrec", "sasrec")]:
            item_emb_path = DATA_DIR / f"{prefix}_item_embeddings.parquet"
            user_emb_path = DATA_DIR / f"{prefix}_user_embeddings.parquet"
            index_path = DATA_DIR / "faiss_index" / f"{prefix}_items.index"
            ranker_path = DATA_DIR / "ranker" / f"{prefix}_ranker.txt"
            if all(p.exists() for p in [item_emb_path, user_emb_path, index_path, ranker_path]):
                registry[name] = EmbeddingRankerRecommender(user_emb_path, index_path, ranker_path)

    return registry


def get_recommendations(model_name: str, user_id: int, k: int = 10) -> list[dict]:
    """Uniform output regardless of model: [{movieId, title, genres, score}, ...]"""
    registry = load_model_registry()
    movies = load_movies()
    if model_name not in registry or movies is None:
        return []

    model = registry[model_name]
    if model_name in ("two_tower", "sasrec"):
        seen = registry.get("_seen_by_user", {}).get(user_id, set())
        pairs = model.recommend(user_id, k, registry["_feature_context"], seen_items=seen)
    else:
        item_ids = model.recommend(user_id, k)
        # popularity and item-item CF are rank-based, not similarity-scored --
        # None signals the UI to show rank instead of a fabricated score
        pairs = [(mid, None) for mid in item_ids]

    out = []
    for movie_id, score in pairs:
        row = movies.filter(pl.col("movieId") == movie_id)
        if row.height == 0:
            continue
        out.append({
            "movieId": movie_id,
            "title": row["title"][0],
            "genres": row["genres"][0],
            "score": score,
        })
    return out
