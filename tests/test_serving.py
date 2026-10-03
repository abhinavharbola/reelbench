import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException
from pydantic import ValidationError

from src.serving import app as serving


class FakePipeline:
    def __init__(self):
        self.calls = []

    def has_user(self, user_id):
        return user_id == 1

    def recommend_scored(self, user_id, k, pool_size=None):
        self.calls.append((user_id, k, pool_size))
        return [(10, 0.9), (11, 0.8), (999, 0.1)][:k]


@pytest.fixture
def loaded_state():
    serving._state.clear()
    pipeline = FakePipeline()
    serving._state.update({
        "serving_model": "sasrec",
        "pipeline": pipeline,
        "movie_lookup": {10: ("Alpha", "Drama"), 11: ("Beta", "Comedy")},
        "cold_start_retriever": None,
    })
    yield pipeline
    serving._state.clear()


def test_recommend_returns_known_movies_only(loaded_state):
    response = serving.recommend(serving.RecommendationRequest(user_id=1, top_n=3))
    assert [item.movie_id for item in response.recommendations] == [10, 11]
    assert loaded_state.calls == [(1, 3, None)]


def test_recommend_unknown_user_is_404(loaded_state):
    with pytest.raises(HTTPException) as exc:
        serving.recommend(serving.RecommendationRequest(user_id=2))
    assert exc.value.status_code == 404


def test_recommend_without_artifacts_is_503():
    serving._state.clear()
    with pytest.raises(HTTPException) as exc:
        serving.recommend(serving.RecommendationRequest(user_id=1))
    assert exc.value.status_code == 503


@pytest.mark.parametrize("payload", [
    {"user_id": 1, "top_n": 0},
    {"user_id": 1, "top_n": -5},
    {"user_id": 1, "top_n": 101},
    {"user_id": 1, "retrieval_pool_size": 0},
    {"user_id": 1, "retrieval_pool_size": 5000},
    {"user_id": 1, "top_n": 20, "retrieval_pool_size": 10},
])
def test_request_validation_rejects_bad_sizes(payload):
    with pytest.raises(ValidationError):
        serving.RecommendationRequest(**payload)


def test_similar_without_cold_start_index_is_404(loaded_state):
    with pytest.raises(HTTPException) as exc:
        serving.similar_items(10, top_n=5)
    assert exc.value.status_code == 404


def test_health_reports_loading_and_ready(loaded_state):
    assert serving.health()["status"] == "ok"
    serving._state.clear()
    assert serving.health()["status"] == "loading"
