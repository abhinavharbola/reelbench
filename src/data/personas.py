import polars as pl

from src.ranking.features import build_item_genre_map, build_user_genre_profiles

PERSONA_DEFINITIONS = [
    ("The Action Fan", "Action"),
    ("The Drama Buff", "Drama"),
    ("Comedy Regular", "Comedy"),
    ("Sci-Fi Devotee", "Sci-Fi"),
]


def curate_personas(
    train: pl.DataFrame,
    movies: pl.DataFrame,
    min_interactions: int = 20,
    persona_definitions: list[tuple[str, str]] = PERSONA_DEFINITIONS,
) -> list[dict]:
    item_genres = build_item_genre_map(movies)
    profiles = build_user_genre_profiles(train, item_genres)

    activity = train.group_by("userId").agg(pl.len().alias("n")).filter(pl.col("n") >= min_interactions)
    active_users = sorted(activity["userId"].to_list())

    personas = []
    used_users: set[int] = set()
    for name, target_genre in persona_definitions:
        best_user, best_score = None, -1.0
        for uid in active_users:
            if uid in used_users:
                continue
            score = profiles.get(uid, {}).get(target_genre, 0.0)
            if score > best_score:
                best_user, best_score = uid, score
        if best_user is not None:
            used_users.add(best_user)
            top_genres = sorted(profiles[best_user].items(), key=lambda x: -x[1])[:3]
            personas.append({
                "name": name,
                "user_id": int(best_user),
                "description": f"Leans heavily {target_genre.lower()}, based on train interaction history.",
                "top_genres": [g for g, _ in top_genres],
            })
    return personas
