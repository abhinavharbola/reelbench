import logging
import os
from contextlib import asynccontextmanager

import polars as pl
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from src.config import DATA_DIR
from src.data.split import build_user_seen_items, load_cutoff
from src.ranking.features import build_feature_context, build_item_genre_map
from src.ranking.pipeline import EMBEDDING_MODELS, NORMALIZE_BY_MODEL, EmbeddingRankerPipeline
from src.retrieval.faiss_index import FaissRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recsys.serving")

DEFAULT_SERVING_MODEL = "sasrec"
MAX_TOP_N = 100
MAX_POOL_SIZE = 2000

_state: dict = {}


def load_artifacts() -> None:
    serving_model = os.environ.get("RECSYS_SERVING_MODEL", DEFAULT_SERVING_MODEL)
    if serving_model not in EMBEDDING_MODELS:
        raise ValueError(f"RECSYS_SERVING_MODEL must be one of {EMBEDDING_MODELS}, got {serving_model!r}")
    logger.info("loading cached artifacts for %s", serving_model)

    train = pl.read_parquet(DATA_DIR / "train.parquet")
    movies = pl.read_parquet(DATA_DIR / "movies.parquet")

    item_genres = build_item_genre_map(movies)
    feature_context = build_feature_context(
        train, item_genres, reference_timestamp=load_cutoff(DATA_DIR, train)
    )
    pipeline = EmbeddingRankerPipeline.from_artifacts(
        DATA_DIR, serving_model, NORMALIZE_BY_MODEL[serving_model],
        feature_context, build_user_seen_items(train), train,
    )

    cold_start_index_path = DATA_DIR / "faiss_index" / "cold_start_items.index"
    cold_start_retriever = None
    if cold_start_index_path.exists():
        cold_start_retriever = FaissRetriever()
        cold_start_retriever.load(cold_start_index_path)

    movie_lookup = {
        row["movieId"]: (row["title"], row["genres"]) for row in movies.iter_rows(named=True)
    }

    _state.update({
        "serving_model": serving_model,
        "pipeline": pipeline,
        "movie_lookup": movie_lookup,
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
    top_n: int = Field(10, ge=1, le=MAX_TOP_N)
    retrieval_pool_size: int | None = Field(None, ge=1, le=MAX_POOL_SIZE)

    @model_validator(mode="after")
    def pool_covers_top_n(self):
        if self.retrieval_pool_size is not None and self.retrieval_pool_size < self.top_n:
            raise ValueError("retrieval_pool_size must be at least top_n")
        return self


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


def _require_loaded() -> None:
    if not _state:
        raise HTTPException(status_code=503, detail="artifacts are not loaded")


def _to_item(movie_id: int, score: float) -> RecommendationItem | None:
    entry = _state["movie_lookup"].get(movie_id)
    if entry is None:
        return None
    return RecommendationItem(movie_id=movie_id, title=entry[0], genres=entry[1], score=score)


@app.get("/health")
def health():
    return {
        "status": "ok" if _state else "loading",
        "artifacts_loaded": len(_state) > 0,
        "serving_model": _state.get("serving_model"),
    }


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: RecommendationRequest):
    _require_loaded()
    pipeline = _state["pipeline"]
    if not pipeline.has_user(req.user_id):
        raise HTTPException(status_code=404, detail=f"user {req.user_id} not found in cached embeddings")

    ranked = pipeline.recommend_scored(req.user_id, req.top_n, pool_size=req.retrieval_pool_size)
    items = [item for item in (_to_item(m, s) for m, s in ranked) if item is not None]
    return RecommendationResponse(user_id=req.user_id, recommendations=items)


@app.get("/similar/{movie_id}", response_model=SimilarItemsResponse)
def similar_items(movie_id: int, top_n: int = Query(10, ge=1, le=MAX_TOP_N)):
    _require_loaded()
    retriever = _state.get("cold_start_retriever")
    if retriever is None:
        raise HTTPException(
            status_code=404,
            detail="no cold-start index loaded. Run scripts/run_cold_start.py and "
                   "scripts/build_cold_start_index.py first.",
        )
    if not retriever.has_item(movie_id):
        raise HTTPException(status_code=404, detail=f"movie {movie_id} not found in cold-start embeddings")

    neighbors = retriever.query(retriever.vector_for(movie_id), top_n=top_n, exclude={movie_id})
    items = [item for item in (_to_item(m, s) for m, s in neighbors) if item is not None]
    return SimilarItemsResponse(movie_id=movie_id, similar=items)
