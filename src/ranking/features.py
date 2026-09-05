"""
Feature engineering for the LightGBM re-ranker. Features, per the project
spec: embedding similarity, recency, popularity, user stats, genre match.

All "reference point" features (recency, popularity, user stats) are
computed from train only, as of the split cutoff -- never from test, since
the ranker itself is evaluated through the same harness as every other
approach and must not see future information either.
"""

import math
from collections import Counter

import polars as pl


def compute_item_popularity(train: pl.DataFrame) -> dict[int, int]:
    counts = train.group_by("movieId").agg(pl.len().alias("count"))
    return dict(zip(counts["movieId"].to_list(), counts["count"].to_list()))


def compute_item_recency(train: pl.DataFrame, reference_timestamp: int) -> dict[int, float]:
    """days since each item's most recent train interaction, relative to
    reference_timestamp (the split cutoff) -- smaller = more currently
    trending, larger = older/evergreen."""
    last_ts = train.group_by("movieId").agg(pl.col("timestamp").max().alias("last_ts"))
    out = {}
    for movie_id, ts in zip(last_ts["movieId"].to_list(), last_ts["last_ts"].to_list()):
        out[movie_id] = max(reference_timestamp - ts, 0) / 86400.0  # seconds -> days
    return out


def compute_user_stats(train: pl.DataFrame, reference_timestamp: int) -> dict[int, dict]:
    """Per-user activity features: interaction count and days since last
    train interaction (how recently active the user was)."""
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
    """movieId -> set of genre strings.

    MovieLens marks items with no genre data using the literal string
    "(no genres listed)", not an empty string. Without special-casing it,
    every such item gets treated as belonging to a fake genre category
    called "(no genres listed)", which quietly skews genre_match_score and
    intra_list_diversity for those items. Both an empty string and the
    sentinel map to an empty genre set.
    """
    out = {}
    for row in movies.iter_rows(named=True):
        genres = row["genres"]
        if not genres or genres == NO_GENRES_SENTINEL:
            out[row["movieId"]] = set()
        else:
            out[row["movieId"]] = set(genres.split("|"))
    return out


def build_user_genre_profiles(train: pl.DataFrame, item_genres: dict[int, set]) -> dict[int, Counter]:
    """Normalized genre-preference distribution per user, from their train
    interaction history."""
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
    """Sum of the user's preference weight across the candidate item's
    genres, normalized by genre count -- higher = item's genres line up
    with what the user has historically watched."""
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


def build_features_for_candidates(
    user_id: int,
    candidates: list[tuple[int, float]],  # [(movieId, embedding_similarity), ...] from FAISS
    item_popularity: dict[int, int],
    item_recency: dict[int, float],
    user_stats: dict[int, dict],
    user_genre_profiles: dict[int, Counter],
    item_genres: dict[int, set],
    seen_items: set[int] | None = None,
) -> pl.DataFrame:
    """One row per candidate item for this user, columns = FEATURE_COLUMNS
    (plus userId/movieId for bookkeeping). log1p on popularity/recency to
    tame long-tailed distributions.

    seen_items: movieIds this user has already interacted with (typically
    their train history). FAISS returns nearest neighbors purely by
    embedding distance with no notion of what a user has already watched,
    so candidates already in seen_items are dropped here before feature
    rows are built -- this is the single place every retrieval-based
    caller (serving, UI, ranker training) funnels through, so "never
    recommend something already seen" is enforced once, not per call site.
    Popularity/CF/ALS already do their own seen-item filtering internally;
    this brings the embedding-based path to the same standard.
    """
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


