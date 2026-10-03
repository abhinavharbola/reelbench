from pathlib import Path

import numpy as np
import polars as pl

from src.data.split import RANKER_SPLIT_SUFFIX
from src.ranking.features import build_features_for_candidates
from src.ranking.ranker import load_model, rank_candidates
from src.retrieval.faiss_index import FaissRetriever

NORMALIZE_BY_MODEL = {"two_tower": True, "sasrec": False}
EMBEDDING_MODELS = tuple(NORMALIZE_BY_MODEL)
DEFAULT_POOL_SIZE = 100
POOL_MULTIPLIER = 5


def pool_size_for(k: int) -> int:
    return max(k * POOL_MULTIPLIER, DEFAULT_POOL_SIZE)


def artifact_paths(data_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "item_embeddings": data_dir / f"{prefix}_item_embeddings.parquet",
        "user_embeddings": data_dir / f"{prefix}_user_embeddings.parquet",
        "index": data_dir / "faiss_index" / f"{prefix}_items.index",
        "ranker": data_dir / "ranker" / f"{prefix}_ranker.txt",
    }


def load_user_embeddings(path: Path) -> dict[int, np.ndarray]:
    frame = pl.read_parquet(path)
    return {
        uid: np.asarray(emb, dtype=np.float32)
        for uid, emb in zip(frame["userId"].to_list(), frame["embedding"].to_list())
    }


def check_alignment(item_ids: set, user_ids: set, train: pl.DataFrame, label: str) -> None:
    train_items = set(train["movieId"].unique().to_list())
    train_users = set(train["userId"].unique().to_list())
    stray_items = item_ids - train_items
    if stray_items:
        raise ValueError(
            f"{label}: {len(stray_items)} item ids in the embeddings are absent from the train data; "
            "artifacts were built from a different split"
        )
    stray_users = user_ids - train_users
    if stray_users:
        raise ValueError(
            f"{label}: {len(stray_users)} user ids in the embeddings are absent from the train data; "
            "artifacts were built from a different split"
        )


def generate_candidate_features(
    retriever: FaissRetriever,
    user_emb_lookup: dict[int, np.ndarray],
    feature_context: dict,
    seen_by_user: dict[int, set],
    pool_size: int,
) -> dict[int, pl.DataFrame]:
    per_user = {}
    for uid, embedding in user_emb_lookup.items():
        seen = seen_by_user.get(uid, set())
        candidates = retriever.query(embedding, top_n=pool_size, exclude=seen)
        per_user[uid] = build_features_for_candidates(uid, candidates, **feature_context, seen_items=seen)
    return per_user


class EmbeddingRankerPipeline:
    def __init__(self, retriever, ranker, user_emb_lookup, feature_context, seen_by_user):
        self.retriever = retriever
        self.ranker = ranker
        self.user_emb_lookup = user_emb_lookup
        self.feature_context = feature_context
        self.seen_by_user = seen_by_user

    @classmethod
    def from_artifacts(cls, data_dir: Path, prefix: str, normalize: bool, feature_context: dict,
                       seen_by_user: dict[int, set], train: pl.DataFrame):
        paths = artifact_paths(data_dir, prefix)
        missing = [name for name in ("index", "ranker", "user_embeddings") if not paths[name].exists()]
        if missing:
            raise FileNotFoundError(f"{prefix}: missing artifacts {missing}")

        retriever = FaissRetriever(normalize=normalize)
        retriever.load(paths["index"])
        if retriever.normalize != normalize:
            raise ValueError(f"{prefix}: index normalization does not match the expected setting")
        user_emb_lookup = load_user_embeddings(paths["user_embeddings"])
        check_alignment(retriever.item_ids(), set(user_emb_lookup), train, prefix)
        return cls(retriever, load_model(paths["ranker"]), user_emb_lookup, feature_context, seen_by_user)

    def has_user(self, user_id: int) -> bool:
        return user_id in self.user_emb_lookup

    def recommend_scored(self, user_id: int, k: int, pool_size: int | None = None) -> list[tuple[int, float]]:
        embedding = self.user_emb_lookup.get(user_id)
        if embedding is None:
            return []
        seen = self.seen_by_user.get(user_id, set())
        candidates = self.retriever.query(embedding, top_n=pool_size or pool_size_for(k), exclude=seen)
        features = build_features_for_candidates(user_id, candidates, **self.feature_context, seen_items=seen)
        return rank_candidates(self.ranker, features, top_k=k)

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        return [item_id for item_id, _ in self.recommend_scored(user_id, k)]


__all__ = [
    "EMBEDDING_MODELS",
    "NORMALIZE_BY_MODEL",
    "RANKER_SPLIT_SUFFIX",
    "EmbeddingRankerPipeline",
    "artifact_paths",
    "check_alignment",
    "generate_candidate_features",
    "load_user_embeddings",
    "pool_size_for",
]
