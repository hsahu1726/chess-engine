import pytest

from chess_engine_2.value_ranking import (
    average_ranks,
    pairwise_ordering_accuracy,
    pearson_correlation,
    spearman_correlation,
    summarize,
)


def test_average_ranks_handle_ties() -> None:
    assert average_ranks([9, 5, 5, 1]) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_detects_matching_and_reversed_rankings() -> None:
    assert spearman_correlation([4, 3, 2, 1], [0.4, 0.3, 0.2, 0.1]) == pytest.approx(1.0)
    assert spearman_correlation([4, 3, 2, 1], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(-1.0)


def test_pairwise_accuracy_counts_prediction_ties_as_half() -> None:
    assert pairwise_ordering_accuracy([3, 2, 1], [3, 2, 1]) == 1.0
    assert pairwise_ordering_accuracy([3, 2, 1], [1, 2, 3]) == 0.0
    assert pairwise_ordering_accuracy([2, 1], [0, 0]) == 0.5


def test_pearson_returns_zero_for_constant_values() -> None:
    assert pearson_correlation([1, 1], [2, 3]) == 0.0


def test_summarize_handles_no_positions() -> None:
    assert summarize([])["positions"] == 0
