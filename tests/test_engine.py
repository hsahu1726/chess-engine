import chess
import pytest

from chess_engine_2.engine import (
    RandomEngine,
    SearchEngine,
    bishop_pair_score,
    color_king_safety_score,
    color_mobility_score,
    color_pawn_structure_score,
    color_rook_file_score,
    evaluate,
    game_phase,
    move_history_key,
    ordered_moves,
    perft,
    tactical_moves,
)


@pytest.mark.parametrize(
    ("depth", "nodes"),
    [
        (0, 1),
        (1, 20),
        (2, 400),
        (3, 8902),
        (4, 197281),
    ],
)
def test_starting_position_perft(depth: int, nodes: int) -> None:
    assert perft(chess.Board(), depth) == nodes


def test_random_engine_returns_legal_move() -> None:
    board = chess.Board()
    move = RandomEngine().choose_move(board)

    assert move in board.legal_moves


def test_random_engine_returns_none_when_game_has_no_legal_moves() -> None:
    board = chess.Board("7k/5Q2/7K/8/8/8/8/8 b - - 0 1")

    assert RandomEngine().choose_move(board) is None


def test_evaluate_rewards_material_for_side_to_move() -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1")

    assert evaluate(board) > 800


def test_bishop_pair_bonus_rewards_two_bishops() -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/2BBK3 w - - 0 1")

    assert bishop_pair_score(board) > 0


def test_mobility_rewards_more_available_piece_moves() -> None:
    open_board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    blocked_board = chess.Board("4k3/8/8/8/8/8/P7/R3K3 w - - 0 1")

    assert color_mobility_score(open_board, chess.WHITE) > color_mobility_score(blocked_board, chess.WHITE)


def test_pawn_structure_penalizes_doubled_isolated_pawns() -> None:
    healthy_board = chess.Board("4k3/8/8/8/8/8/3PP3/4K3 w - - 0 1")
    weak_board = chess.Board("4k3/8/8/8/8/3P4/3P4/4K3 w - - 0 1")

    assert color_pawn_structure_score(healthy_board, chess.WHITE) > color_pawn_structure_score(weak_board, chess.WHITE)


def test_pawn_structure_rewards_passed_pawn_progress() -> None:
    early_passer = chess.Board("4k3/8/8/8/8/4P3/8/4K3 w - - 0 1")
    advanced_passer = chess.Board("4k3/4P3/8/8/8/8/8/4K3 w - - 0 1")

    assert color_pawn_structure_score(advanced_passer, chess.WHITE) > color_pawn_structure_score(early_passer, chess.WHITE)


def test_rook_file_score_rewards_open_files() -> None:
    open_file = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    blocked_file = chess.Board("4k3/8/8/8/8/8/P7/R3K3 w - - 0 1")

    assert color_rook_file_score(open_file, chess.WHITE) > color_rook_file_score(blocked_file, chess.WHITE)


def test_game_phase_drops_in_endgame() -> None:
    opening = chess.Board()
    endgame = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")

    assert game_phase(opening) > game_phase(endgame)


def test_king_safety_rewards_pawn_shield() -> None:
    shielded = chess.Board("4k3/8/8/8/8/8/5PPP/6K1 w - - 0 1")
    exposed = chess.Board("4k3/8/8/8/8/8/8/6K1 w - - 0 1")

    assert color_king_safety_score(shielded, chess.WHITE) > color_king_safety_score(exposed, chess.WHITE)


def test_search_finds_mate_in_one() -> None:
    board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
    result = SearchEngine(max_depth=1).search(board)

    assert result.move == chess.Move.from_uci("g6g7")
    assert result.score > 90_000


def test_tactical_moves_include_captures_but_not_quiet_moves() -> None:
    board = chess.Board("4k3/8/8/8/4p3/3P4/8/4K3 w - - 0 1")
    moves = tactical_moves(board)

    assert chess.Move.from_uci("d3e4") in moves
    assert chess.Move.from_uci("e1e2") not in moves


def test_quiescence_sees_capture_after_depth_runs_out() -> None:
    board = chess.Board("4k3/8/8/3q4/3R4/8/8/4K3 w - - 0 1")
    result = SearchEngine(max_depth=1).search(board)

    assert result.move == chess.Move.from_uci("d4d5")
    assert result.score > 400


def test_transposition_table_reuses_previous_search() -> None:
    board = chess.Board()
    engine = SearchEngine(max_depth=3)

    first_result = engine.search(board)
    cached_entries = len(engine.transposition_table)
    second_result = engine.search(board)

    assert cached_entries > 0
    assert second_result.move == first_result.move
    assert second_result.score == first_result.score
    assert second_result.nodes < first_result.nodes


def test_iterative_search_returns_each_completed_depth() -> None:
    board = chess.Board()
    results = SearchEngine(max_depth=3).iterative_search(board)

    assert [result.depth for result in results] == [1, 2, 3]
    assert all(result.move in board.legal_moves for result in results)
    assert results[-1].nodes >= results[0].nodes
    assert results[-1].pv
    assert results[-1].pv[0] == results[-1].move


def test_iterative_search_with_movetime_returns_legal_move() -> None:
    board = chess.Board()
    results = SearchEngine(max_depth=10).iterative_search(board, movetime_ms=1)

    assert results
    assert results[-1].move in board.legal_moves


def test_killer_move_is_ordered_before_other_quiet_moves() -> None:
    board = chess.Board()
    killer = chess.Move.from_uci("g1f3")
    moves = ordered_moves(board, killer_moves=[killer])

    assert moves[0] == killer


def test_history_score_improves_quiet_move_ordering() -> None:
    board = chess.Board()
    move = chess.Move.from_uci("b1c3")
    moves = ordered_moves(board, history_scores={move_history_key(move): 5_000})

    assert moves[0] == move


def test_search_records_move_ordering_heuristics() -> None:
    board = chess.Board()
    engine = SearchEngine(max_depth=3)

    engine.search(board)

    assert engine.killer_moves or engine.history_scores
