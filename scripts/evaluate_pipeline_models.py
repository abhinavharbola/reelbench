"""
Extends the evaluation harness to score the two-tower and SASRec
retrieval+ranking pipelines, through the *same* harness in src/eval/metrics.py
that scores the three CPU baselines in run_phase1.py -- closing the gap the
README's "Known limitations" section used to describe: run_phase1.py wrote
the first 3 rows of results/comparison_table.csv; this script appends the
remaining 2, once trained embeddings and built artifacts exist for them.

Each embedding-based model is wrapped behind the same recommend(user_id, k)
interface run_phase1.py's evaluate_model() already expects, so no harness
code is duplicated or special-cased per model:
  1. FAISS retrieval over the model's item embeddings
  2. exclude the user's train history from the retrieved candidates (see
     src.ranking.features.build_features_for_candidates -- without this,
     the two neural models would be scored on a mix of already-seen and
     genuinely-new items while the 3 baselines are scored on new-only
     items, which isn't "the identical protocol" the README promises)
  3. LightGBM re-ranks the survivors down to top_n

This reads data/processed/test.parquet purely as ground truth to score
against -- never as ranker training signal (that's carve_ranker_supervision_
split's job in build_serving_artifacts.py / build_ui_artifacts.py, entirely
upstream of and separate from this script).

Requires scripts/build_ui_artifacts.py to have already produced
{prefix}_items.index and {prefix}_ranker.txt for each model you want scored;
models without those artifacts are skipped, not treated as an error, so this
can run against however many of the 2 models you've actually trained.

Usage:
    python scripts/evaluate_pipeline_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src.data.split import build_user_seen_items
from src.eval.metrics import evaluate_all
from src.eval.results_table import upsert_results
from src.eval.tracking import log_model_run
from src.ranking.features import build_feature_context, build_features_for_candidates, build_item_genre_map
from src.ranking.ranker import load_model, rank_candidates
from src.retrieval.faiss_index import FaissRetriever

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
K_VALUES = (10, 20)
TOP_K_FOR_RECS = max(K_VALUES)
MODEL_PREFIXES = ["two_tower", "sasrec"]


class PipelineRecommender:
    """FAISS retrieval -> seen-item filter -> LightGBM re-rank, behind the
    same recommend(user_id, k) interface as the baseline models, so
    evaluate_model() below (copied from run_phase1.py's function of the
    same name) treats every approach identically."""

    def __init__(self, retriever: FaissRetriever, ranker, feature_context: dict, seen_by_user: dict, user_emb_lookup: dict):
        self.retriever = retriever
        self.ranker = ranker
        self.feature_context = feature_context
        self.seen_by_user = seen_by_user
        self.user_emb_lookup = user_emb_lookup

    def recommend(self, user_id: int, k: int) -> list[int]:
        if user_id not in self.user_emb_lookup:
            return []  # cold user for this model's embedding table
        import numpy as np
        user_vec = np.array(self.user_emb_lookup[user_id], dtype=np.float32)
        candidates = self.retriever.query(user_vec, top_n=max(k * 5, 100))
        features = build_features_for_candidates(
            user_id, candidates, **self.feature_context,
            seen_items=self.seen_by_user.get(user_id, set()),
        )
        # recommend() must return list[item_id] per the harness contract
        # (src/eval/metrics.py) -- rank_candidates() now returns
        # [(item_id, score), ...], so unwrap to just the ids here.
        ranked = rank_candidates(self.ranker, features, top_k=k)
        return [item_id for item_id, _ in ranked]


def evaluate_model(name: str, model, test: pl.DataFrame, catalog_size: int, item_genres: dict) -> dict:
    test_by_user = test.group_by("userId").agg(pl.col("movieId")).to_dict(as_series=False)
    user_ids = test_by_user["userId"]
    relevant_lists = [set(items) for items in test_by_user["movieId"]]

    all_recommended = [model.recommend(uid, k=TOP_K_FOR_RECS) for uid in user_ids]

    metrics = evaluate_all(all_recommended, relevant_lists, catalog_size, item_genres, ks=K_VALUES)
    metrics["model"] = name
    return metrics


def load_pipeline_model(prefix: str, feature_context: dict, seen_by_user: dict) -> PipelineRecommender | None:
    index_path = DATA_DIR / "faiss_index" / f"{prefix}_items.index"
    ranker_path = DATA_DIR / "ranker" / f"{prefix}_ranker.txt"
    user_emb_path = DATA_DIR / f"{prefix}_user_embeddings.parquet"

    if not all(p.exists() for p in [index_path, ranker_path, user_emb_path]):
        print(f"skipping {prefix}: artifacts missing. Run scripts/build_ui_artifacts.py first.")
        return None

    retriever = FaissRetriever()
    retriever.load(index_path)
    ranker = load_model(ranker_path)
    user_embeddings = pl.read_parquet(user_emb_path)
    user_emb_lookup = dict(zip(user_embeddings["userId"].to_list(), user_embeddings["embedding"].to_list()))

    return PipelineRecommender(retriever, ranker, feature_context, seen_by_user, user_emb_lookup)


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
    feature_context = build_feature_context(train, item_genres)
    seen_by_user = build_user_seen_items(train)

    new_rows = []
    for prefix in MODEL_PREFIXES:
        model = load_pipeline_model(prefix, feature_context, seen_by_user)
        if model is None:
            continue
        metrics = evaluate_model(prefix, model, test, catalog_size, item_genres)
        new_rows.append(metrics)
        with log_model_run(prefix, params={"stage": "retrieval+ranking"}, metrics=metrics):
            pass
        print(f"{prefix} done: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "model"))

    if not new_rows:
        print("no pipeline models scored -- nothing to append.")
        return

    combined = upsert_results(table_path, new_rows, MODEL_PREFIXES)
    print(combined)
    print(f"\nwritten to {table_path}")


if __name__ == "__main__":
    main()
