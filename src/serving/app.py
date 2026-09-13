"""
FastAPI serving layer, CPU-only. Reads only local cached artifacts (FAISS
index, LightGBM ranker, embedding parquet files) -- no live external API
calls at request time, ever, per the project spec.

Run: uvicorn src.serving.app:app --reload
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data.split import build_user_seen_items
from src.ranking.features import build_feature_context, build_features_for_candidates, build_item_genre_map
from src.ranking.ranker import load_model, rank_candidates
from src.retrieval.faiss_index import FaissRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recsys.serving")

ARTIFACTS_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

_state: dict = {}


def load_artifacts() -> None:
    """All artifacts loaded once at startup, held in memory -- no per-request
    disk or network I/O beyond what's already resident."""
    logger.info("loading cached artifacts")

    train = pl.read_parquet(ARTIFACTS_DIR / "train.parquet")
    movies = pl.read_parquet(ARTIFACTS_DIR / "movies.parquet")

    retriever = FaissRetriever()
    retriever.load(ARTIFACTS_DIR / "faiss_index" / "items.index")

    ranker = load_model(ARTIFACTS_DIR / "ranker" / "lightgbm_ranker.txt")

    user_embeddings = pl.read_parquet(ARTIFACTS_DIR / "two_tower_user_embeddings.parquet")
    user_emb_lookup = dict(zip(user_embeddings["userId"].to_list(), user_embeddings["embedding"].to_list()))

    item_genres = build_item_genre_map(movies)
    feature_context = build_feature_context(train, item_genres)

    # optional: content-similarity index over cold-start (Gemini) item
    # embeddings, built by build_serving_artifacts.py if that cache
    # exists. A separate index, not merged with the two-tower one -- the
    # two embedding spaces have no shared dimension or training signal.
    cold_start_index_path = ARTIFACTS_DIR / "faiss_index" / "cold_start_items.index"
    cold_start_retriever = None
    if cold_start_index_path.exists():
        cold_start_retriever = FaissRetriever()
        cold_start_retriever.load(cold_start_index_path)

    _state.update({
        "train": train,
        "movies": movies,
        "retriever": retriever,
        "ranker": ranker,
        "user_emb_lookup": user_emb_lookup,
        **feature_context,
        # every user's full train history, so /recommend never surfaces a
        # movie the user has already watched -- the other 4 approaches in
        # this project already do this internally, this brings the
        # embedding+ranker path served here to the same standard
        "seen_by_user": build_user_seen_items(train),
        "cold_start_retriever": cold_start_retriever,
    })
    logger.info("artifacts loaded, serving ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield
    _state.clear()


app = FastAPI(title="MovieLens Recommender", lifespan=lifespan)


class RecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 10
    retrieval_pool_size: int = 100


class RecommendationItem(BaseModel):
    movie_id: int
    title: str
    genres: str
    score: float


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[RecommendationItem]


class SimilarItemsResponse(BaseModel):
    movie_id: int
    similar: list[RecommendationItem]


@app.get("/health")
def health():
    return {"status": "ok", "artifacts_loaded": len(_state) > 0}


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: RecommendationRequest):
    if req.user_id not in _state["user_emb_lookup"]:
        raise HTTPException(status_code=404, detail=f"user {req.user_id} not found in cached embeddings")

    user_vec = np.array(_state["user_emb_lookup"][req.user_id], dtype=np.float32)
    candidates = _state["retriever"].query(user_vec, top_n=req.retrieval_pool_size)

    features = build_features_for_candidates(
        req.user_id,
        candidates,
        _state["item_popularity"],
        _state["item_recency"],
        _state["user_stats"],
        _state["user_genre_profiles"],
        _state["item_genres"],
        seen_items=_state["seen_by_user"].get(req.user_id, set()),
    )
    ranked = rank_candidates(_state["ranker"], features, top_k=req.top_n)

    movies = _state["movies"]

    items = []
    for movie_id, score in ranked:
        movie_row = movies.filter(pl.col("movieId") == movie_id)
        if movie_row.height == 0:
            continue
        items.append(RecommendationItem(
            movie_id=movie_id,
            title=movie_row["title"][0],
            genres=movie_row["genres"][0],
            score=score,
        ))

    return RecommendationResponse(user_id=req.user_id, recommendations=items)


@app.get("/similar/{movie_id}", response_model=SimilarItemsResponse)
def similar_items(movie_id: int, top_n: int = 10):
    """Content-based "more like this", via the cold-start (title+genre)
    embedding index -- works even for items too new to have a meaningful
    two-tower embedding. 404s if run_cold_start.py hasn't been run."""
    retriever = _state.get("cold_start_retriever")
    if retriever is None:
        raise HTTPException(
            status_code=404,
            detail="no cold-start index loaded. Run scripts/run_cold_start.py and "
                   "scripts/build_serving_artifacts.py first.",
        )
    if movie_id not in retriever.idx_to_item_id.values():
        raise HTTPException(status_code=404, detail=f"movie {movie_id} not found in cold-start embeddings")

    item_idx_to_vec = {v: k for k, v in retriever.idx_to_item_id.items()}  # movieId -> faiss idx
    query_idx = item_idx_to_vec[movie_id]
    query_vec = retriever.index.reconstruct(query_idx)

    neighbors = retriever.query(query_vec, top_n=top_n + 1)  # +1: query item matches itself
    movies = _state["movies"]

    items = []
    for candidate_id, score in neighbors:
        if candidate_id == movie_id:
            continue
        movie_row = movies.filter(pl.col("movieId") == candidate_id)
        if movie_row.height == 0:
            continue
        items.append(RecommendationItem(
            movie_id=candidate_id, title=movie_row["title"][0], genres=movie_row["genres"][0], score=score,
        ))
        if len(items) == top_n:
            break

    return SimilarItemsResponse(movie_id=movie_id, similar=items)
