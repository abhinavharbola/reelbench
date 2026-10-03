from pathlib import Path

import polars as pl

from src.data.split import RANKER_SPLIT_SUFFIX
from src.ranking.pipeline import (
    DEFAULT_POOL_SIZE,
    NORMALIZE_BY_MODEL,
    artifact_paths,
    check_alignment,
    generate_candidate_features,
    load_user_embeddings,
)
from src.ranking.ranker import build_training_table, save_model, train_ranker
from src.retrieval.faiss_index import FaissRetriever


def build_model_artifacts(
    data_dir: Path,
    prefix: str,
    ranker_split,
    feature_context: dict,
    ranker_seen_by_user: dict,
    ranker_positives: dict,
    pool_size: int = DEFAULT_POOL_SIZE,
    num_boost_round: int = 200,
) -> bool:
    normalize = NORMALIZE_BY_MODEL[prefix]
    full_paths = artifact_paths(data_dir, prefix)
    rs_paths = artifact_paths(data_dir, f"{prefix}{RANKER_SPLIT_SUFFIX}")

    required = [full_paths["item_embeddings"], full_paths["user_embeddings"],
                rs_paths["item_embeddings"], rs_paths["user_embeddings"]]
    missing = [p for p in required if not p.exists()]
    if missing:
        print(f"skipping {prefix}: missing embeddings {[p.name for p in missing]}. "
              f"Both the full-train and the ranker-split embeddings are required so the ranker "
              f"is never trained on candidates produced by embeddings that saw its labels.")
        return False

    rs_items = pl.read_parquet(rs_paths["item_embeddings"])
    rs_users = load_user_embeddings(rs_paths["user_embeddings"])
    rs_retriever = FaissRetriever(normalize=normalize)
    rs_retriever.build(rs_items)
    check_alignment(rs_retriever.item_ids(), set(rs_users), ranker_split.train, f"{prefix} ranker-split")

    per_user_features = generate_candidate_features(
        rs_retriever, rs_users, feature_context, ranker_seen_by_user, pool_size
    )
    training_table = build_training_table(per_user_features, ranker_positives)
    ranker = train_ranker(training_table, num_boost_round=num_boost_round)
    save_model(ranker, full_paths["ranker"])
    print(f"{prefix}: ranker trained on {training_table.height:,} rows from ranker-split embeddings, saved")

    full_items = pl.read_parquet(full_paths["item_embeddings"])
    retriever = FaissRetriever(normalize=normalize)
    retriever.build(full_items)
    retriever.save(full_paths["index"])
    print(f"{prefix}: serving FAISS index built from full-train embeddings, {retriever.index.ntotal} items")
    return True
