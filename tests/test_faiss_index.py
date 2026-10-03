import numpy as np
import pytest

pytest.importorskip("faiss")
import polars as pl

from src.retrieval.faiss_index import FaissRetriever


def make_items(n: int = 50, dim: int = 8) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    return pl.DataFrame({"movieId": list(range(1000, 1000 + n)), "embedding": vectors.tolist()})


def test_exclude_still_returns_full_top_n():
    retriever = FaissRetriever()
    items = make_items()
    retriever.build(items)
    query = np.array(items["embedding"][0], dtype=np.float32)

    unfiltered = retriever.query(query, top_n=10)
    exclude = {item_id for item_id, _ in unfiltered[:8]}
    filtered = retriever.query(query, top_n=10, exclude=exclude)

    assert len(filtered) == 10
    assert not exclude & {item_id for item_id, _ in filtered}


def test_exclude_larger_than_catalog_returns_what_is_left():
    retriever = FaissRetriever()
    retriever.build(make_items(n=20))
    exclude = set(range(1000, 1015))
    results = retriever.query(np.ones(8, dtype=np.float32), top_n=10, exclude=exclude)
    assert len(results) == 5


def test_nan_query_returns_empty():
    retriever = FaissRetriever()
    retriever.build(make_items())
    assert retriever.query(np.full(8, np.nan, dtype=np.float32)) == []


def test_nan_items_are_rejected_at_build_time():
    items = make_items()
    vectors = np.array(items["embedding"].to_list(), dtype=np.float32)
    vectors[3, 2] = np.nan
    bad = pl.DataFrame({"movieId": items["movieId"], "embedding": vectors.tolist()})
    with pytest.raises(ValueError):
        FaissRetriever().build(bad)


def test_wrong_query_dimension_raises():
    retriever = FaissRetriever()
    retriever.build(make_items())
    with pytest.raises(ValueError):
        retriever.query(np.ones(5, dtype=np.float32))


def test_raw_inner_product_preserves_norm_information():
    items = pl.DataFrame({"movieId": [1, 2], "embedding": [[1.0, 0.0], [10.0, 0.0]]})
    query = np.array([1.0, 0.0], dtype=np.float32)

    cosine = FaissRetriever(normalize=True)
    cosine.build(items)
    raw = FaissRetriever(normalize=False)
    raw.build(items)

    assert abs(cosine.query(query, top_n=2)[0][1] - cosine.query(query, top_n=2)[1][1]) < 1e-6
    assert raw.query(query, top_n=2)[0][0] == 2


def test_save_load_round_trip_keeps_normalization_setting(tmp_path):
    retriever = FaissRetriever(normalize=False)
    retriever.build(make_items())
    path = tmp_path / "idx" / "items.index"
    retriever.save(path)

    loaded = FaissRetriever()
    loaded.load(path)
    assert loaded.normalize is False
    assert loaded.item_ids() == retriever.item_ids()
    assert loaded.has_item(1000) and not loaded.has_item(1)
