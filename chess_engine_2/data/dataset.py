from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess

from chess_engine_2.encoding import POLICY_SIZE, move_to_policy_index


PIECE_PLANES = {
    (chess.WHITE, chess.PAWN): 0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4,
    (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10,
    (chess.BLACK, chess.KING): 11,
}

SIDE_TO_MOVE_PLANE = 12
WHITE_KINGSIDE_PLANE = 13
WHITE_QUEENSIDE_PLANE = 14
BLACK_KINGSIDE_PLANE = 15
BLACK_QUEENSIDE_PLANE = 16
EN_PASSANT_PLANE = 17
INPUT_PLANES = 18
BOARD_SIZE = 8


@dataclass(frozen=True)
class JsonlSample:
    fen: str
    move_uci: str
    policy_index: int
    value: float


@dataclass(frozen=True)
class ValidationSummary:
    rows: int
    valid_rows: int
    invalid_rows: int


def sample_from_dict(row: dict[str, Any]) -> JsonlSample:
    return JsonlSample(
        fen=str(row["fen"]),
        move_uci=str(row["move_uci"]),
        policy_index=int(row["policy_index"]),
        value=float(row["value"]),
    )


def iter_jsonl_samples(path: Path, max_samples: int | None = None):
    with path.open("r", encoding="utf-8") as stream:
        for row_number, line in enumerate(stream, start=1):
            if max_samples is not None and row_number > max_samples:
                break
            if not line.strip():
                continue
            yield sample_from_dict(json.loads(line))


def board_to_planes(board: chess.Board) -> list[list[list[int]]]:
    planes = [[[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)] for _ in range(INPUT_PLANES)]

    for square, piece in board.piece_map().items():
        plane = PIECE_PLANES[(piece.color, piece.piece_type)]
        rank = 7 - chess.square_rank(square)
        file = chess.square_file(square)
        planes[plane][rank][file] = 1

    if board.turn == chess.WHITE:
        fill_plane(planes[SIDE_TO_MOVE_PLANE], 1)
    if board.has_kingside_castling_rights(chess.WHITE):
        fill_plane(planes[WHITE_KINGSIDE_PLANE], 1)
    if board.has_queenside_castling_rights(chess.WHITE):
        fill_plane(planes[WHITE_QUEENSIDE_PLANE], 1)
    if board.has_kingside_castling_rights(chess.BLACK):
        fill_plane(planes[BLACK_KINGSIDE_PLANE], 1)
    if board.has_queenside_castling_rights(chess.BLACK):
        fill_plane(planes[BLACK_QUEENSIDE_PLANE], 1)

    if board.ep_square is not None:
        rank = 7 - chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        planes[EN_PASSANT_PLANE][rank][file] = 1

    return planes


def fill_plane(plane: list[list[int]], value: int) -> None:
    for rank in range(BOARD_SIZE):
        for file in range(BOARD_SIZE):
            plane[rank][file] = value


def validate_sample(sample: JsonlSample) -> None:
    if sample.policy_index < 0 or sample.policy_index >= POLICY_SIZE:
        raise ValueError(f"policy index out of range: {sample.policy_index}")
    if sample.value < -1.0 or sample.value > 1.0:
        raise ValueError(f"value out of range: {sample.value}")

    board = chess.Board(sample.fen)
    move = chess.Move.from_uci(sample.move_uci)
    if move not in board.legal_moves:
        raise ValueError(f"move is not legal in FEN: {sample.move_uci}")

    expected_policy_index = move_to_policy_index(move, board)
    if sample.policy_index != expected_policy_index:
        raise ValueError(f"policy index mismatch: expected {expected_policy_index}, got {sample.policy_index}")

    board_to_planes(board)


def validate_jsonl(path: Path, max_samples: int | None = None) -> ValidationSummary:
    rows = 0
    invalid_rows = 0

    for sample in iter_jsonl_samples(path, max_samples):
        rows += 1
        try:
            validate_sample(sample)
        except Exception:
            invalid_rows += 1

    return ValidationSummary(rows=rows, valid_rows=rows - invalid_rows, invalid_rows=invalid_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate JSONL chess training samples.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    summary = validate_jsonl(args.path, args.max_samples)
    print(f"rows: {summary.rows}")
    print(f"valid rows: {summary.valid_rows}")
    print(f"invalid rows: {summary.invalid_rows}")


if __name__ == "__main__":
    main()
