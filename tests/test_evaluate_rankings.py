import chess
import pytest
import torch

from chess_engine_2.evaluate_rankings import evaluate_group
from chess_engine_2.neural import PolicyValueNet


def test_evaluate_group_reports_perfect_ranking_for_matching_values(monkeypatch) -> None:
    board = chess.Board()
    children = []
    for move_uci, score in (("e2e4", 100), ("d2d4", 50), ("a2a3", -100)):
        child = board.copy()
        child.push_uci(move_uci)
        children.append(
            {
                "move_uci": move_uci,
                "child_fen": child.fen(),
                "search_score": score,
                "child_value_target": 0.0,
            }
        )

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, planes):
            return torch.zeros((3, 1)), torch.tensor([-0.9, -0.4, 0.5])

    result = evaluate_group(FakeModel(), torch.device("cpu"), {"root_fen": board.fen(), "children": children})

    assert result.spearman == pytest.approx(1.0)
    assert result.pairwise_accuracy == 1.0
    assert result.top1_correct
    assert result.neural_top1_regret_cp == 0
