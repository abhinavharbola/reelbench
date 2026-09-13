"""
Generates a small synthetic MovieLens-like dataset and trains/caches all
artifacts the Streamlit UI needs, so `streamlit run ui/app.py` works
immediately for demoing and screenshotting -- before running the real
pipeline against the full 25M dataset.

IMPORTANT: the two-tower and SASRec "embeddings" produced here are NOT
trained neural models -- this sandbox can't install torch (see Phase 2/3
notes). They're derived from ALS factors with a small random perturbation,
which is enough to produce plausible-looking, non-random demo
recommendations. Once you actually run scripts/train_two_tower.py and
scripts/train_sasrec.py on Colab/Kaggle per the real pipeline, re-run this
script's real counterpart (run_phase1.py + the training scripts) instead of
this one -- this script is a demo/UI-development aid only.
"""

import json
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from src.data.personas import curate_personas
from src.data.split import assert_no_leakage, build_user_seen_items, carve_ranker_supervision_split, temporal_split
from src.models.baseline import ItemItemCF, PopularityModel
from src.models.mf import MatrixFactorizationModel
from src.ranking.features import build_feature_context, build_features_for_candidates, build_item_genre_map
from src.ranking.ranker import build_training_table, save_model, train_ranker
from src.retrieval.faiss_index import FaissRetriever
from src.eval.metrics import evaluate_all

random.seed(42)
np.random.seed(42)

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
MODELS_DIR = DATA_DIR / "models"

ADJECTIVES = ["Silent", "Midnight", "Broken", "Last", "Hidden", "Distant", "Golden", "Crooked", "Endless", "Quiet"]
NOUNS = ["Harbor", "Signal", "Garden", "Machine", "River", "Letter", "Orchard", "Station", "Mirror", "Horizon"]
GENRES_POOL = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Horror", "Thriller", "Documentary", "Animation"]

N_USERS = 400
N_ITEMS = 350


def generate_synthetic_data():
    movies_rows = []
    for item_id in range(1, N_ITEMS + 1):
        title = f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)} ({1970 + item_id % 55})"
        genres = "|".join(random.sample(GENRES_POOL, random.randint(1, 3)))
        movies_rows.append({"movieId": item_id, "title": title, "genres": genres})
    movies = pl.DataFrame(movies_rows).cast({"movieId": pl.Int32})

    # give items a popularity skew so the popularity baseline is meaningful
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


def build_mock_neural_embeddings(als_model: MatrixFactorizationModel, item_ids: list[int], user_ids: list[int], dim: int):
    """Demo-only stand-in for two-tower/SASRec embeddings (see module
    docstring): ALS factors plus noise, distinct per "model" so the UI
    shows different (but not nonsensical) rankings across approaches."""
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


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "faiss_index").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "ranker").mkdir(parents=True, exist_ok=True)

    interactions, movies = generate_synthetic_data()
    interactions.write_parquet(DATA_DIR / "interactions.parquet")
    movies.write_parquet(DATA_DIR / "movies.parquet")

    split = temporal_split(interactions, test_quantile=0.85, min_train_interactions=5)
    split.train.write_parquet(DATA_DIR / "train.parquet")
    print(f"train: {split.train.height}, test: {split.test.height}")

    catalog_size = interactions["movieId"].n_unique()
    item_genres = build_item_genre_map(movies)

    results = []

    def evaluate(name, model):
        test_by_user = split.test.group_by("userId").agg(pl.col("movieId")).to_dict(as_series=False)
        uids = test_by_user["userId"]
        relevant = [set(x) for x in test_by_user["movieId"]]
        recs = [model.recommend(u, k=20) for u in uids]
        metrics = evaluate_all(recs, relevant, catalog_size, item_genres, ks=(10, 20))
        metrics["model"] = name
        results.append(metrics)

    pop = PopularityModel()
    pop.fit(split.train)
    evaluate("popularity", pop)
    with open(MODELS_DIR / "popularity.pkl", "wb") as f:
        pickle.dump(pop, f)

    cf = ItemItemCF(top_k=30)
    cf.fit(split.train)
    evaluate("item_item_cf", cf)
    with open(MODELS_DIR / "item_item_cf.pkl", "wb") as f:
        pickle.dump(cf, f)

    als = MatrixFactorizationModel(method="als", factors=32, iterations=15)
    als.fit(split.train)
    evaluate("als", als)
    with open(MODELS_DIR / "als.pkl", "wb") as f:
        pickle.dump(als, f)

    print("baselines done, building demo embedding-based models")

    item_ids = split.train["movieId"].unique().sort().to_list()
    user_ids = split.train["userId"].unique().sort().to_list()

    # eval-time features: full split.train is genuinely "known" as of the
    # real split cutoff, so this is the correct reference dataframe for
    # scoring MockRecommender below.
    eval_feature_context = build_feature_context(split.train, item_genres)

    # ranker supervision: carved out of train only, mirrors
    # build_serving_artifacts.py / build_ui_artifacts.py, so this demo
    # script exercises the same leakage-safe path the real pipeline uses
    # instead of a demo-only shortcut that would drift from it.
    ranker_split = carve_ranker_supervision_split(split.train, val_quantile=0.8)
    assert_no_leakage(ranker_split)
    ranker_seen_by_user = build_user_seen_items(ranker_split.train)
    ranker_positives = build_user_seen_items(ranker_split.test)
    full_seen_by_user = build_user_seen_items(split.train)

    # ranker-training-time features: ranker_split.train only, strictly
    # before ranker_split.test (the labels). ranker_split.test is a
    # subset of split.train, so using split.train here would count each
    # label interaction into the features describing the candidate it
    # labels -- see src.ranking.features.build_feature_context's
    # docstring for the full explanation of this leak.
    ranker_feature_context = build_feature_context(ranker_split.train, item_genres)

    for prefix in ["two_tower", "sasrec"]:
        item_df, user_df = build_mock_neural_embeddings(als, item_ids, user_ids, dim=32)
        item_df.write_parquet(DATA_DIR / f"{prefix}_item_embeddings.parquet")
        user_df.write_parquet(DATA_DIR / f"{prefix}_user_embeddings.parquet")

        retriever = FaissRetriever()
        retriever.build(item_df)
        retriever.save(DATA_DIR / "faiss_index" / f"{prefix}_items.index")

        per_user_features = {}
        for uid in user_df["userId"].to_list():
            uvec = np.array(user_df.filter(pl.col("userId") == uid)["embedding"][0])
            candidates = retriever.query(uvec, top_n=30)
            per_user_features[uid] = build_features_for_candidates(
                uid, candidates, **ranker_feature_context,
                seen_items=ranker_seen_by_user.get(uid, set()),
            )

        training_table = build_training_table(per_user_features, ranker_positives)
        ranker = train_ranker(training_table, num_boost_round=50)
        save_model(ranker, DATA_DIR / "ranker" / f"{prefix}_ranker.txt")

        class MockRecommender:
            def __init__(self, retriever, ranker):
                self.retriever, self.ranker = retriever, ranker

            def recommend(self, user_id, k):
                from src.ranking.ranker import rank_candidates
                uid_series = user_df.filter(pl.col("userId") == user_id)["embedding"]
                if uid_series.len() == 0:
                    return []
                uvec = np.array(uid_series[0])
                candidates = self.retriever.query(uvec, top_n=max(k * 3, 30))
                feats = build_features_for_candidates(
                    user_id, candidates, **eval_feature_context,
                    seen_items=full_seen_by_user.get(user_id, set()),
                )
                # recommend() must return list[item_id] per the harness
                # contract (src/eval/metrics.py) -- rank_candidates() now
                # returns [(item_id, score), ...], unwrap to just the ids.
                ranked = rank_candidates(self.ranker, feats, top_k=k)
                return [item_id for item_id, _ in ranked]

        evaluate(prefix, MockRecommender(retriever, ranker))
        print(f"{prefix} demo model done")

    table = pl.DataFrame(results)
    ordered_cols = ["model"] + [c for c in table.columns if c != "model"]
    table = table.select(ordered_cols)
    table.write_csv(RESULTS_DIR / "comparison_table.csv")
    print(table)

    personas = curate_personas(split.train, movies)
    (DATA_DIR / "personas.json").write_text(json.dumps(personas, indent=2))
    print(f"{len(personas)} personas cached")

    print("\ndemo artifacts ready. run: streamlit run ui/app.py")


if __name__ == "__main__":
    main()
