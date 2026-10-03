from src.eval.metrics import intra_list_diversity, mean_intra_list_diversity


def test_pairs_with_no_genres_are_skipped_not_counted_as_maximally_diverse():
    genres = {1: set(), 2: set(), 3: {"Drama"}, 4: {"Drama"}}
    assert intra_list_diversity([1, 2], genres) == 0.0
    assert intra_list_diversity([3, 4], genres) == 0.0
    assert abs(intra_list_diversity([1, 2, 3, 4], genres) - (1 - 1 / 5)) < 1e-9


def test_mean_excludes_lists_too_short_to_have_pairs():
    genres = {1: {"Drama"}, 2: {"Comedy"}}
    lists = [[1, 2], [1], []]
    assert mean_intra_list_diversity(lists, genres) == 1.0


def test_mean_of_no_scorable_lists_is_zero():
    assert mean_intra_list_diversity([[], [1]], {1: {"Drama"}}) == 0.0
