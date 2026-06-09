import json

import chess

from chess_engine_2.data.dataset import (
    BLACK_KINGSIDE_PLANE,
    INPUT_PLANES,
    SIDE_TO_MOVE_PLANE,
    WHITE_KINGSIDE_PLANE,
    board_to_planes,
    validate_jsonl,
)
from chess_engine_2.encoding import move_to_policy_index


def test_board_to_planes_encodes_start_position() -> None:
    board = chess.Board()
    planes = board_to_planes(board)

    assert len(planes) == INPUT_PLANES
    assert len(planes[0]) == 8
    assert len(planes[0][0]) == 8
    assert planes[0][6][4] == 1
    assert planes[6][1][4] == 1
    assert planes[SIDE_TO_MOVE_PLANE][0][0] == 1
    assert planes[WHITE_KINGSIDE_PLANE][0][0] == 1
    assert planes[BLACK_KINGSIDE_PLANE][0][0] == 1


def test_validate_jsonl_checks_policy_and_legal_move(tmp_path) -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    path = tmp_path / "samples.jsonl"
    path.write_text(
        json.dumps(
            {
                "fen": board.fen(),
                "move_uci": move.uci(),
                "policy_index": move_to_policy_index(move, board),
                "value": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = validate_jsonl(path)

    assert summary.rows == 1
    assert summary.valid_rows == 1
    assert summary.invalid_rows == 0
