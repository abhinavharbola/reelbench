import polars as pl

from src.models.baseline import ItemItemCF, PopularityModel


def make_train() -> pl.DataFrame:
    rows = [
        (1, 10), (1, 11), (1, 12),
        (2, 10), (2, 11), (2, 13),
        (3, 10), (3, 11), (3, 12),
        (4, 10), (4, 14),
    ]
    return pl.DataFrame(
        [{"userId": u, "movieId": m, "timestamp": i} for i, (u, m) in enumerate(rows)]
    ).cast({"userId": pl.Int32, "movieId": pl.Int32, "timestamp": pl.Int64})


def test_popularity_excludes_seen_and_orders_by_count():
    model = PopularityModel()
    model.fit(make_train())
    recs = model.recommend(4, k=3)
    assert 10 not in recs and 14 not in recs
    assert recs[0] == 11


def test_item_item_cf_recommends_unseen_co_occurring_items():
    model = ItemItemCF(top_k=5, block_size=2)
    model.fit(make_train())
    recs = model.recommend(2, k=3)
    assert 10 not in recs and 11 not in recs and 13 not in recs
    assert recs[0] == 12


def test_item_item_cf_unknown_user_returns_empty():
    model = ItemItemCF(top_k=5)
    model.fit(make_train())
    assert model.recommend(999, k=5) == []


def test_item_item_cf_neighbors_exclude_self():
    model = ItemItemCF(top_k=5)
    model.fit(make_train())
    for item, neighbors in model.neighbors.items():
        assert all(neighbor != item for neighbor, _ in neighbors)
