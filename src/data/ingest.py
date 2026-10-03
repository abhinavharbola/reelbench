from pathlib import Path

import polars as pl

from src.config import DATA_DIR

POSITIVE_THRESHOLD = 4.0

RATINGS_SCHEMA = {
    "userId": pl.Int32,
    "movieId": pl.Int32,
    "rating": pl.Float32,
    "timestamp": pl.Int64,
}

MOVIES_SCHEMA = {
    "movieId": pl.Int32,
    "title": pl.Utf8,
    "genres": pl.Utf8,
}


def load_ratings(raw_dir: Path) -> pl.DataFrame:
    path = raw_dir / "ml-25m" / "ratings.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download ml-25m.zip from "
            "https://files.grouplens.org/datasets/movielens/ml-25m.zip "
            "and unzip into data/raw/."
        )
    return pl.read_csv(path, schema_overrides=RATINGS_SCHEMA)


def load_movies(raw_dir: Path) -> pl.DataFrame:
    path = raw_dir / "ml-25m" / "movies.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See load_ratings() for download info.")
    return pl.read_csv(path, schema_overrides=MOVIES_SCHEMA)


def to_implicit_feedback(ratings: pl.DataFrame, threshold: float = POSITIVE_THRESHOLD) -> pl.DataFrame:
    return (
        ratings.filter(pl.col("rating") >= threshold)
        .select(["userId", "movieId", "timestamp"])
        .sort(["userId", "timestamp"])
    )


def build_processed_dataset(raw_dir: Path, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)

    ratings = load_ratings(raw_dir)
    movies = load_movies(raw_dir)
    interactions = to_implicit_feedback(ratings)

    interactions.write_parquet(processed_dir / "interactions.parquet")
    movies.write_parquet(processed_dir / "movies.parquet")

    print(f"interactions: {interactions.height:,} rows")
    print(f"unique users: {interactions['userId'].n_unique():,}")
    print(f"unique items: {interactions['movieId'].n_unique():,}")


if __name__ == "__main__":
    build_processed_dataset(Path("data/raw"), DATA_DIR)
