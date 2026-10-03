import math
from collections import Counter

import polars as pl


def compute_item_popularity(train: pl.DataFrame) -> dict[int, int]:
    counts = train.group_by("movieId").agg(pl.len().alias("count"))
    return dict(zip(counts["movieId"].to_list(), counts["count"].to_list()))


def compute_item_recency(train: pl.DataFrame, reference_timestamp: int) -> dict[int, float]:
    last_ts = train.group_by("movieId").agg(pl.col("timestamp").max().alias("last_ts"))
    out = {}
    for movie_id, ts in zip(last_ts["movieId"].to_list(), last_ts["last_ts"].to_list()):
        out[movie_id] = max(reference_timestamp - ts, 0) / 86400.0
    return out


def compute_user_stats(train: pl.DataFrame, reference_timestamp: int) -> dict[int, dict]:
    agg = train.group_by("userId").agg(
        pl.len().alias("n_interactions"),
        pl.col("timestamp").max().alias("last_ts"),
    )
    out = {}
    for uid, n, last_ts in zip(agg["userId"].to_list(), agg["n_interactions"].to_list(), agg["last_ts"].to_list()):
        out[uid] = {
            "n_interactions": n,
            "days_since_last_interaction": max(reference_timestamp - last_ts, 0) / 86400.0,
        }
    return out


NO_GENRES_SENTINEL = "(no genres listed)"


def build_item_genre_map(movies: pl.DataFrame) -> dict[int, set]:
    out = {}
    for row in movies.iter_rows(named=True):
        genres = row["genres"]
        if not genres or genres == NO_GENRES_SENTINEL:
            out[row["movieId"]] = set()
        else:
            out[row["movieId"]] = set(genres.split("|"))
    return out


def build_user_genre_profiles(train: pl.DataFrame, item_genres: dict[int, set]) -> dict[int, Counter]:
    by_user = train.group_by("userId").agg(pl.col("movieId")).to_dict(as_series=False)
    profiles = {}
    for uid, items in zip(by_user["userId"], by_user["movieId"]):
        counter = Counter()
        for item in items:
            for genre in item_genres.get(item, set()):
                counter[genre] += 1
        total = sum(counter.values())
        if total > 0:
            for genre in counter:
                counter[genre] /= total
        profiles[uid] = counter
    return profiles


def genre_match_score(user_profile: Counter, item_genre_set: set) -> float:
    if not item_genre_set:
        return 0.0
    return sum(user_profile.get(g, 0.0) for g in item_genre_set) / len(item_genre_set)


FEATURE_COLUMNS = [
    "embedding_similarity",
    "item_popularity",
    "item_recency_days",
    "user_n_interactions",
    "user_days_since_last_interaction",
    "genre_match",
]


def build_feature_context(reference_df: pl.DataFrame, item_genres: dict[int, set], reference_timestamp: int | None = None) -> dict:
    if reference_timestamp is None:
        reference_timestamp = reference_df.select(pl.col("timestamp").max()).item()
    return {
        "item_popularity": compute_item_popularity(reference_df),
        "item_recency": compute_item_recency(reference_df, reference_timestamp),
        "user_stats": compute_user_stats(reference_df, reference_timestamp),
        "user_genre_profiles": build_user_genre_profiles(reference_df, item_genres),
        "item_genres": item_genres,
    }


def build_features_for_candidates(
    user_id: int,
    candidates: list[tuple[int, float]],
    item_popularity: dict[int, int],
    item_recency: dict[int, float],
    user_stats: dict[int, dict],
    user_genre_profiles: dict[int, Counter],
    item_genres: dict[int, set],
    seen_items: set[int] | None = None,
) -> pl.DataFrame:
    stats = user_stats.get(user_id, {"n_interactions": 0, "days_since_last_interaction": 0.0})
    profile = user_genre_profiles.get(user_id, Counter())
    seen_items = seen_items or set()

    rows = []
    for movie_id, sim in candidates:
        if movie_id in seen_items:
            continue
        rows.append({
            "userId": user_id,
            "movieId": movie_id,
            "embedding_similarity": sim,
            "item_popularity": math.log1p(item_popularity.get(movie_id, 0)),
            "item_recency_days": math.log1p(item_recency.get(movie_id, 0.0)),
            "user_n_interactions": math.log1p(stats["n_interactions"]),
            "user_days_since_last_interaction": math.log1p(stats["days_since_last_interaction"]),
            "genre_match": genre_match_score(profile, item_genres.get(movie_id, set())),
        })
    return pl.DataFrame(rows)
