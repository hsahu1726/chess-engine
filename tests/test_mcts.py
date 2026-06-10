import chess

from chess_engine_2.mcts import MCTSEngine, MCTSNode, terminal_value


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
