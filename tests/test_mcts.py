import chess
import pytest
import torch

from chess_engine_2.encoding import POLICY_SIZE
from chess_engine_2.mcts import MCTSEngine, MCTSNode, NeuralPolicyValue, position_cache_key, terminal_value


def test_mcts_search_prefers_high_prior_move() -> None:
    def policy_value(board: chess.Board):
        moves = list(board.legal_moves)
        priors = {move: 0.01 for move in moves}
        priors[chess.Move.from_uci("e2e4")] = 1.0
        return priors, 0.0

    result = MCTSEngine(policy_value, simulations=8).search(chess.Board())

    assert result.move == chess.Move.from_uci("e2e4")
    assert result.simulations == 8
    assert result.root_visits == 8
    assert result.root_moves[0].move_uci == "e2e4"
    assert result.root_moves[0].prior > result.root_moves[-1].prior
    assert result.leaf_value_stddev == 0.0


def test_mcts_search_returns_none_when_no_legal_moves() -> None:
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")

    result = MCTSEngine(lambda board: ({}, 0.0), simulations=4).search(board)

    assert result.move is None
    assert result.simulations == 0
    assert result.root_value == -1.0


def test_terminal_value_is_from_side_to_move_perspective() -> None:
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")

    assert terminal_value(board) == -1.0


def test_select_child_uses_child_value_from_parent_perspective() -> None:
    root = MCTSNode(visits=10)
    good_for_parent = MCTSNode(prior=0.1, visits=1, value_sum=-1.0)
    bad_for_parent = MCTSNode(prior=0.1, visits=1, value_sum=1.0)
    root.children = {
        chess.Move.from_uci("e2e4"): good_for_parent,
        chess.Move.from_uci("d2d4"): bad_for_parent,
    }

    move, _ = MCTSEngine(lambda board: ({}, 0.0)).select_child(root)

    assert move == chess.Move.from_uci("e2e4")


def test_root_move_stats_expose_policy_value_and_exploration_terms() -> None:
    root = MCTSNode(visits=16)
    child = MCTSNode(prior=0.25, visits=3, value_sum=-0.6)
    root.children = {chess.Move.from_uci("e2e4"): child}
    engine = MCTSEngine(lambda board: ({}, 0.0), cpuct=2.0)

    stat = engine.root_move_stats(root)[0]

    assert stat.move_uci == "e2e4"
    assert stat.q_value == pytest.approx(0.2)
    assert stat.exploration == pytest.approx(0.5)
    assert stat.puct_score == pytest.approx(0.7)


def test_neural_policy_value_caches_repeated_position() -> None:
    class FakeModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, planes):
            self.calls += 1
            return torch.zeros((1, POLICY_SIZE)), torch.tensor([0.25])

    model = FakeModel()
    evaluator = NeuralPolicyValue(model, torch.device("cpu"), cache_size=8)
    board = chess.Board()

    first = evaluator(board)
    second = evaluator(board)

    assert first == second
    assert model.calls == 1
    assert evaluator.network_evaluations == 1
    assert evaluator.cache_hits == 1
    assert evaluator.cache_hit_percent == 50.0


def test_position_cache_key_ignores_move_clocks() -> None:
    first = chess.Board()
    second = chess.Board()
    second.halfmove_clock = 12
    second.fullmove_number = 9

    assert position_cache_key(first) == position_cache_key(second)
