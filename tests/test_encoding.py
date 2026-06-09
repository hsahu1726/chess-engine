import chess
import pytest

from chess_engine_2.encoding import POLICY_SIZE, move_to_policy_index, policy_index_to_move


@pytest.mark.parametrize(
    ("fen", "move"),
    [
        (chess.STARTING_FEN, chess.Move.from_uci("e2e4")),
        (chess.STARTING_FEN, chess.Move.from_uci("g1f3")),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", chess.Move.from_uci("a7a8q")),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", chess.Move.from_uci("a7a8n")),
        ("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1", chess.Move.from_uci("b7a8r")),
    ],
)
def test_policy_encoding_round_trips_white_moves(fen: str, move: chess.Move) -> None:
    board = chess.Board(fen)
    encoded = move_to_policy_index(move, board)

    assert 0 <= encoded < POLICY_SIZE
    assert policy_index_to_move(encoded, board) == move


def test_policy_encoding_round_trips_black_move() -> None:
    board = chess.Board()
    board.turn = chess.BLACK
    move = chess.Move.from_uci("e7e5")
    encoded = move_to_policy_index(move, board)

    assert policy_index_to_move(encoded, board) == move
