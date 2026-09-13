"""
Evaluation harness. Built and tested before any model (per project spec
section 4) — every approach is scored through these exact functions, same
protocol, no per-model exceptions.

All ranking metrics take:
  recommended: list[item_id]  -- top-K, in ranked order, for one user
  relevant:    set[item_id]   -- ground-truth positives for that user (test set)
"""

import math


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = len(set(top_k) & relevant)
    return hits / len(relevant)


def _dcg_at_k(recommended: list, relevant: set, k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 2)  # rank i is 0-indexed -> position i+1
    return dcg


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    dcg = _dcg_at_k(recommended, relevant, k)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = 0
    precision_sum = 0.0
    for i, item in enumerate(top_k):
        if item in relevant:
            hits += 1
            precision_sum += hits / (i + 1)
    denom = min(len(relevant), k)
    if denom == 0:
        return 0.0
    return precision_sum / denom


def map_at_k(all_recommended: list[list], all_relevant: list[set], k: int) -> float:
    """Mean of average_precision_at_k across all users."""
    if not all_recommended:
        return 0.0
    scores = [
        average_precision_at_k(rec, rel, k)
        for rec, rel in zip(all_recommended, all_relevant)
    ]
    return sum(scores) / len(scores)


def mean_recall_at_k(all_recommended: list[list], all_relevant: list[set], k: int) -> float:
    scores = [recall_at_k(rec, rel, k) for rec, rel in zip(all_recommended, all_relevant)]
    return sum(scores) / len(scores) if scores else 0.0


def mean_ndcg_at_k(all_recommended: list[list], all_relevant: list[set], k: int) -> float:
    scores = [ndcg_at_k(rec, rel, k) for rec, rel in zip(all_recommended, all_relevant)]
    return sum(scores) / len(scores) if scores else 0.0


def catalog_coverage(all_recommended: list[list], catalog_size: int) -> float:
    """Fraction of the full item catalog that appears at least once across
    all users' recommendation lists."""
    if catalog_size == 0:
        return 0.0
    recommended_items = set()
    for rec in all_recommended:
        recommended_items.update(rec)
    return len(recommended_items) / catalog_size


def intra_list_diversity(recommended: list, item_genres: dict) -> float:
    """
    1 - average pairwise genre-overlap (Jaccard) similarity within one
    user's recommendation list. item_genres maps item_id -> set[genre].
    Higher = more diverse. Returns 0.0 for lists of length < 2.
    """
    n = len(recommended)
    if n < 2:
        return 0.0

    pair_count = 0
    similarity_sum = 0.0
    for i in range(n):
        genres_i = item_genres.get(recommended[i], set())
        for j in range(i + 1, n):
            genres_j = item_genres.get(recommended[j], set())
            union = genres_i | genres_j
            if union:
                sim = len(genres_i & genres_j) / len(union)
            else:
                sim = 0.0
            similarity_sum += sim
            pair_count += 1

    if pair_count == 0:
        return 0.0
    avg_similarity = similarity_sum / pair_count
    return 1.0 - avg_similarity


def mean_intra_list_diversity(all_recommended: list[list], item_genres: dict) -> float:
    scores = [intra_list_diversity(rec, item_genres) for rec in all_recommended]
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_all(
    all_recommended: list[list],
    all_relevant: list[set],
    catalog_size: int,
    item_genres: dict,
    ks: tuple[int, ...] = (10, 20),
) -> dict:
    """Single entry point every model's evaluation run calls, so all 5
    approaches are scored identically."""
    results = {}
    for k in ks:
        results[f"recall@{k}"] = mean_recall_at_k(all_recommended, all_relevant, k)
        results[f"ndcg@{k}"] = mean_ndcg_at_k(all_recommended, all_relevant, k)
        results[f"map@{k}"] = map_at_k(all_recommended, all_relevant, k)

    top_k_for_coverage = max(ks)
    truncated = [rec[:top_k_for_coverage] for rec in all_recommended]
    results["coverage"] = catalog_coverage(truncated, catalog_size)
    results["diversity"] = mean_intra_list_diversity(truncated, item_genres)

    return results
