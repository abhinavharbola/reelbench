import math

from src.eval.metrics import (
    average_precision_at_k,
    catalog_coverage,
    intra_list_diversity,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_at_k_hand_computed():
    # 2 of 4 relevant items appear in top 5 -> recall = 2/4 = 0.5
    recommended = ["a", "x", "b", "y", "z"]
    relevant = {"a", "b", "c", "d"}
    assert recall_at_k(recommended, relevant, k=5) == 0.5


def test_recall_at_k_all_hit():
    recommended = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert recall_at_k(recommended, relevant, k=3) == 1.0


def test_recall_at_k_no_relevant_items_returns_zero():
    assert recall_at_k(["a", "b"], set(), k=2) == 0.0


def test_ndcg_at_k_hand_computed():
    # relevant = {a, c}. recommended = [a, b, c] -> hits at rank 1 and 3
    # DCG = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
    # IDCG (2 relevant, k=3 -> ideal ranks 1,2) = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309...
    recommended = ["a", "b", "c"]
    relevant = {"a", "c"}
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    expected = dcg / idcg
    assert math.isclose(ndcg_at_k(recommended, relevant, k=3), expected, rel_tol=1e-9)


def test_ndcg_at_k_perfect_ranking_is_one():
    recommended = ["a", "b", "c", "d"]
    relevant = {"a", "b"}
    assert math.isclose(ndcg_at_k(recommended, relevant, k=4), 1.0, rel_tol=1e-9)


def test_average_precision_at_k_hand_computed():
    # relevant = {a, c}, recommended = [a, x, c]
    # hit at rank1 (prec=1/1=1.0), hit at rank3 (prec=2/3)
    # AP = (1.0 + 2/3) / min(2, 3) = (1.6667)/2 = 0.8333...
    recommended = ["a", "x", "c"]
    relevant = {"a", "c"}
    expected = (1.0 + 2 / 3) / 2
    assert math.isclose(average_precision_at_k(recommended, relevant, k=3), expected, rel_tol=1e-9)


def test_average_precision_at_k_no_hits_is_zero():
    assert average_precision_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_catalog_coverage_hand_computed():
    # catalog of 10 items, only 3 distinct items ever recommended
    all_recommended = [["a", "b"], ["b", "c"], ["a"]]
    assert catalog_coverage(all_recommended, catalog_size=10) == 0.3


def test_catalog_coverage_full_coverage():
    all_recommended = [["a", "b"], ["c", "d"]]
    assert catalog_coverage(all_recommended, catalog_size=4) == 1.0


def test_intra_list_diversity_identical_genres_is_zero():
    # both items share identical genre sets -> similarity 1.0 -> diversity 0.0
    recommended = ["m1", "m2"]
    item_genres = {"m1": {"Action", "Sci-Fi"}, "m2": {"Action", "Sci-Fi"}}
    assert intra_list_diversity(recommended, item_genres) == 0.0


def test_intra_list_diversity_disjoint_genres_is_one():
    recommended = ["m1", "m2"]
    item_genres = {"m1": {"Action"}, "m2": {"Romance"}}
    assert intra_list_diversity(recommended, item_genres) == 1.0


def test_intra_list_diversity_single_item_is_zero():
    assert intra_list_diversity(["m1"], {"m1": {"Action"}}) == 0.0
