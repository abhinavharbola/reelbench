"""
Builds the per-model FAISS indices and LightGBM rankers that the Streamlit
UI's model registry (ui/data_access.py) needs for the 5-way comparison
screen: one FAISS index + ranker per embedding-based model (two-tower,
SASRec), built from real trained embeddings.

This is distinct from scripts/build_serving_artifacts.py, which builds a
single production index/ranker (fixed filenames items.index,
lightgbm_ranker.txt) for FastAPI's one-model serving path. This script
builds per-model-named artifacts (two_tower_items.index,
two_tower_ranker.txt, sasrec_items.index, sasrec_ranker.txt) so the UI can
compare all embedding-based models side by side.

Run this any time you replace the embedding parquet files in
data/processed/ (e.g. after downloading a new Kaggle-trained run) --
stale FAISS indices built from old embeddings will raise a FAISS
dimension-mismatch assertion if queried with vectors of a different size
than what the index was built with.

Reference-point features (popularity, recency, user stats, genre
profiles) for ranker training are built from ranker_split.train, not the
full outer train -- see src.ranking.features.build_feature_context's
docstring for why: the full train contains ranker_split.test (the
positive labels), so computing these features from it would leak each
label interaction into the features describing the candidate it labels.

Usage:
    python scripts/build_ui_artifacts.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src.data.split import assert_no_leakage, build_user_seen_items, carve_ranker_supervision_split
from src.ranking.features import build_feature_context, build_features_for_candidates, build_item_genre_map
from src.ranking.ranker import build_training_table, save_model, train_ranker
from src.retrieval.faiss_index import FaissRetriever

DATA_DIR = Path("data/processed")
MODEL_PREFIXES = ["two_tower", "sasrec"]


def build_for_model(
    prefix: str, feature_context: dict, ranker_seen_by_user: dict, ranker_positives: dict,
) -> bool:
    item_emb_path = DATA_DIR / f"{prefix}_item_embeddings.parquet"
    user_emb_path = DATA_DIR / f"{prefix}_user_embeddings.parquet"

    if not item_emb_path.exists() or not user_emb_path.exists():
        print(f"skipping {prefix}: embeddings not found at {item_emb_path}")
        return False

    item_embeddings = pl.read_parquet(item_emb_path)
    user_embeddings = pl.read_parquet(user_emb_path)

    retriever = FaissRetriever()
    retriever.build(item_embeddings)
    (DATA_DIR / "faiss_index").mkdir(parents=True, exist_ok=True)
    retriever.save(DATA_DIR / "faiss_index" / f"{prefix}_items.index")
    embedding_dim = len(item_embeddings["embedding"][0])
    print(f"{prefix}: FAISS index built, {retriever.index.ntotal} items, dim {embedding_dim}")

    user_emb_lookup = dict(zip(user_embeddings["userId"].to_list(), user_embeddings["embedding"].to_list()))

    per_user_features = {}
    for user_id, embedding in user_emb_lookup.items():
        candidates = retriever.query(embedding, top_n=100)
        per_user_features[user_id] = build_features_for_candidates(
            user_id, candidates, **feature_context,
            seen_items=ranker_seen_by_user.get(user_id, set()),
        )

    training_table = build_training_table(per_user_features, ranker_positives)
    ranker = train_ranker(training_table)

    (DATA_DIR / "ranker").mkdir(parents=True, exist_ok=True)
    save_model(ranker, DATA_DIR / "ranker" / f"{prefix}_ranker.txt")
    print(f"{prefix}: ranker trained on {training_table.height:,} rows, saved")
    return True


def main():
    train_path = DATA_DIR / "train.parquet"
    movies_path = DATA_DIR / "movies.parquet"

    for p in [train_path, movies_path]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run scripts/run_phase1.py first.")

    train = pl.read_parquet(train_path)
    movies = pl.read_parquet(movies_path)

    item_genres = build_item_genre_map(movies)

    # ranker supervision: entirely in-train, mirrors build_serving_artifacts.py.
    # test.parquet is never read by this script.
    ranker_split = carve_ranker_supervision_split(train)
    assert_no_leakage(ranker_split)  # hard stop if the inner split itself is broken
    ranker_seen_by_user = build_user_seen_items(ranker_split.train)
    ranker_positives = build_user_seen_items(ranker_split.test)
    print(f"ranker supervision split: {ranker_split.train.height:,} seen rows, "
          f"{ranker_split.test.height:,} label rows, cutoff {ranker_split.cutoff_timestamp}")

    # built from ranker_split.train (not the full outer train), see module docstring
    feature_context = build_feature_context(ranker_split.train, item_genres)

    built_any = False
    for prefix in MODEL_PREFIXES:
        if build_for_model(prefix, feature_context, ranker_seen_by_user, ranker_positives):
            built_any = True

    if not built_any:
        print("no embedding files found for any model -- nothing built. "
              "Copy two_tower_*.parquet and/or sasrec_*.parquet into data/processed/ first.")
    else:
        print("\ndone. restart the Streamlit app to pick up the rebuilt artifacts.")


if __name__ == "__main__":
    main()
