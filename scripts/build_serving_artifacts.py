"""
Builds the production serving artifacts that src/serving/app.py depends on
at startup: a FAISS index over the two-tower item embeddings, and a
LightGBM ranker trained on top of retrieval candidates. Also builds a
separate content-similarity index over cold-start (Gemini) item embeddings,
if that cache exists -- see the "cold start" note below.

The two-tower model is the designated production path (FastAPI serves one
approach; the Streamlit UI separately compares all 5 for demo purposes).
Run this after scripts/train_two_tower.py has produced
data/processed/two_tower_{item,user}_embeddings.parquet.

Ranker training labels come from an in-train supervision split
(src.data.split.carve_ranker_supervision_split), never from
data/processed/test.parquet. test.parquet is the harness's blind held-out
set (see src/eval/metrics.py, scripts/run_phase1.py) -- using it here would
mean the ranker's own training data already contained the labels the
harness later scores it against. Candidates that a user has already seen
(their full train history) are also excluded before the ranker ever sees
them, matching how the other four approaches already behave; see
src.ranking.features.build_features_for_candidates.

The reference-point features (item popularity, item recency, user
activity stats, genre profiles) are built from ranker_split.train, not
the full outer train, and as of ranker_split.cutoff_timestamp, not
train's own max timestamp (see src.ranking.features.build_feature_context).
This fixes a real leak: ranker_split.test (the positive labels) is a
subset of the full train, so computing these features from the full
train counted each label interaction into the very features describing
the candidate it labels -- e.g. item_popularity for a candidate could
include the exact watch event being predicted, and item_recency could
reflect that the item was watched at the label timestamp. Restricting
the reference dataframe to ranker_split.train (strictly before the
label window) keeps these features honest about what's actually known
before each label, the same discipline temporal_split already applies
at the outer train/test boundary.

Cold start: run_cold_start.py caches local text embeddings (Qwen3-Embedding-0.6B
via sentence-transformers) for every movie's title+genres, dimensionally
incompatible with the two-tower's
learned embedding space (no shared training signal ties the two spaces
together), so they can't be merged into the main item FAISS index. Instead
this builds a second, standalone FAISS index purely over the cold-start
embeddings, enabling content-based "more like this" lookups (see
src/serving/app.py's /similar endpoint) for items with too little
interaction history to have a meaningful two-tower embedding. Skipped
silently if data/processed/cold_start_embeddings.parquet doesn't exist.

Usage:
    python scripts/build_serving_artifacts.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src.data.split import assert_no_leakage, build_user_seen_items, carve_ranker_supervision_split
from src.ranking.features import (
    build_feature_context,
    build_features_for_candidates,
    build_item_genre_map,
)
from src.ranking.ranker import build_training_table, save_model, train_ranker
from src.retrieval.faiss_index import FaissRetriever, build_and_save_index

DATA_DIR = Path("data/processed")


def main():
    train_path = DATA_DIR / "train.parquet"
    movies_path = DATA_DIR / "movies.parquet"
    item_emb_path = DATA_DIR / "two_tower_item_embeddings.parquet"
    user_emb_path = DATA_DIR / "two_tower_user_embeddings.parquet"

    for p in [train_path, movies_path, item_emb_path, user_emb_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run scripts/run_phase1.py and "
                "scripts/train_two_tower.py first."
            )

    train = pl.read_parquet(train_path)
    movies = pl.read_parquet(movies_path)
    item_embeddings = pl.read_parquet(item_emb_path)
    user_embeddings = pl.read_parquet(user_emb_path)

    (DATA_DIR / "faiss_index").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "ranker").mkdir(parents=True, exist_ok=True)

    retriever = FaissRetriever()
    retriever.build(item_embeddings)
    retriever.save(DATA_DIR / "faiss_index" / "items.index")
    print(f"FAISS index built: {retriever.index.ntotal} items")

    cold_start_path = DATA_DIR / "cold_start_embeddings.parquet"
    if cold_start_path.exists():
        build_and_save_index(cold_start_path, DATA_DIR / "faiss_index" / "cold_start_items.index")
    else:
        print(f"no cold-start cache at {cold_start_path}, skipping content-similarity index "
              "(run scripts/run_cold_start.py to build it)")

    item_genres = build_item_genre_map(movies)

    # ranker supervision: entirely in-train, test.parquet is never read here
    ranker_split = carve_ranker_supervision_split(train)
    assert_no_leakage(ranker_split)  # hard stop if the inner split itself is broken
    ranker_seen_by_user = build_user_seen_items(ranker_split.train)
    ranker_positives = build_user_seen_items(ranker_split.test)
    print(f"ranker supervision split: {ranker_split.train.height:,} seen rows, "
          f"{ranker_split.test.height:,} label rows, cutoff {ranker_split.cutoff_timestamp}")

    # built from ranker_split.train (not the full outer train) -- see the
    # module docstring above and build_feature_context's own docstring
    feature_context = build_feature_context(ranker_split.train, item_genres)

    user_emb_lookup = dict(zip(user_embeddings["userId"].to_list(), user_embeddings["embedding"].to_list()))

    per_user_features = {}
    for user_id, embedding in user_emb_lookup.items():
        candidates = retriever.query(embedding, top_n=100)
        per_user_features[user_id] = build_features_for_candidates(
            user_id, candidates, **feature_context,
            seen_items=ranker_seen_by_user.get(user_id, set()),
        )

    training_table = build_training_table(per_user_features, ranker_positives)
    print(f"ranker training table: {training_table.height:,} rows")

    ranker = train_ranker(training_table)
    save_model(ranker, DATA_DIR / "ranker" / "lightgbm_ranker.txt")
    print(f"ranker saved to {DATA_DIR / 'ranker' / 'lightgbm_ranker.txt'}")


if __name__ == "__main__":
    main()
