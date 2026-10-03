import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src.config import DATA_DIR
from src.data.split import assert_no_leakage, build_user_seen_items, carve_ranker_supervision_split
from src.ranking.build import build_model_artifacts
from src.ranking.features import build_feature_context, build_item_genre_map
from src.ranking.pipeline import EMBEDDING_MODELS


def main():
    train_path = DATA_DIR / "train.parquet"
    movies_path = DATA_DIR / "movies.parquet"

    for p in [train_path, movies_path]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run scripts/run_phase1.py first.")

    train = pl.read_parquet(train_path)
    movies = pl.read_parquet(movies_path)
    item_genres = build_item_genre_map(movies)

    ranker_split = carve_ranker_supervision_split(train)
    assert_no_leakage(ranker_split)
    ranker_seen_by_user = build_user_seen_items(ranker_split.train)
    ranker_positives = build_user_seen_items(ranker_split.test)
    print(f"ranker supervision split: {ranker_split.train.height:,} seen rows, "
          f"{ranker_split.test.height:,} label rows, cutoff {ranker_split.cutoff_timestamp}")

    feature_context = build_feature_context(
        ranker_split.train, item_genres, reference_timestamp=ranker_split.cutoff_timestamp
    )

    built_any = False
    for prefix in EMBEDDING_MODELS:
        if build_model_artifacts(
            DATA_DIR, prefix, ranker_split, feature_context, ranker_seen_by_user, ranker_positives
        ):
            built_any = True

    if not built_any:
        print("nothing built. Place <model>_*.parquet and <model>_rs_*.parquet in the data directory first.")
    else:
        print("\ndone. restart the Streamlit app and the serving API to pick up the rebuilt artifacts.")


if __name__ == "__main__":
    main()
