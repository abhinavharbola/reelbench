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
            dcg += 1.0 / math.log2(i + 2)
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
    if catalog_size == 0:
        return 0.0
    recommended_items = set()
    for rec in all_recommended:
        recommended_items.update(rec)
    return len(recommended_items) / catalog_size


def _pairwise_diversity(recommended: list, item_genres: dict) -> float | None:
    n = len(recommended)
    if n < 2:
        return None
    pair_count = 0
    similarity_sum = 0.0
    for i in range(n):
        genres_i = item_genres.get(recommended[i], set())
        for j in range(i + 1, n):
            genres_j = item_genres.get(recommended[j], set())
            union = genres_i | genres_j
            if not union:
                continue
            similarity_sum += len(genres_i & genres_j) / len(union)
            pair_count += 1
    if pair_count == 0:
        return None
    return 1.0 - similarity_sum / pair_count


def intra_list_diversity(recommended: list, item_genres: dict) -> float:
    value = _pairwise_diversity(recommended, item_genres)
    return 0.0 if value is None else value


def mean_intra_list_diversity(all_recommended: list[list], item_genres: dict) -> float:
    scores = [_pairwise_diversity(rec, item_genres) for rec in all_recommended]
    scores = [s for s in scores if s is not None]
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_all(
    all_recommended: list[list],
    all_relevant: list[set],
    catalog_size: int,
    item_genres: dict,
    ks: tuple[int, ...] = (10, 20),
) -> dict:
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
