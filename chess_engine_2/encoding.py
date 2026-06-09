from __future__ import annotations

import chess

POLICY_SIZE = 8 * 8 * 73

QUEEN_DIRECTIONS = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
]

KNIGHT_DIRECTIONS = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
]

UNDERPROMOTION_PIECES = [
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
]

UNDERPROMOTION_FILE_DELTAS = [-1, 0, 1]


def move_to_policy_index(move: chess.Move, board: chess.Board) -> int:
    from_square = orient_square(move.from_square, board.turn)
    to_square = orient_square(move.to_square, board.turn)
    plane = move_to_plane(from_square, to_square, move.promotion)
    return from_square * 73 + plane


def policy_index_to_move(index: int, board: chess.Board) -> chess.Move:
    if index < 0 or index >= POLICY_SIZE:
        raise ValueError(f"policy index must be in [0, {POLICY_SIZE})")

    from_square = index // 73
    plane = index % 73
    to_square, promotion = plane_to_move_target(from_square, plane)

    actual_from_square = orient_square(from_square, board.turn)
    actual_to_square = orient_square(to_square, board.turn)
    actual_promotion = promotion or infer_queen_promotion(board, actual_from_square, actual_to_square)

    return chess.Move(actual_from_square, actual_to_square, promotion=actual_promotion)


def orient_square(square: chess.Square, turn: chess.Color) -> chess.Square:
    return square if turn == chess.WHITE else chess.square_mirror(square)


def move_to_plane(from_square: chess.Square, to_square: chess.Square, promotion: chess.PieceType | None) -> int:
    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)
    to_file = chess.square_file(to_square)
    to_rank = chess.square_rank(to_square)
    file_delta = to_file - from_file
    rank_delta = to_rank - from_rank

    if promotion in UNDERPROMOTION_PIECES:
        return underpromotion_plane(file_delta, promotion)

    knight_delta = (file_delta, rank_delta)
    if knight_delta in KNIGHT_DIRECTIONS:
        return 56 + KNIGHT_DIRECTIONS.index(knight_delta)

    distance = max(abs(file_delta), abs(rank_delta))
    if distance < 1 or distance > 7:
        raise ValueError(f"cannot encode move delta {(file_delta, rank_delta)}")

    direction = normalize_direction(file_delta, rank_delta)
    if direction not in QUEEN_DIRECTIONS:
        raise ValueError(f"cannot encode non-queen-like move delta {(file_delta, rank_delta)}")

    return QUEEN_DIRECTIONS.index(direction) * 7 + distance - 1


def plane_to_move_target(from_square: chess.Square, plane: int) -> tuple[chess.Square, chess.PieceType | None]:
    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)

    if plane < 56:
        direction = QUEEN_DIRECTIONS[plane // 7]
        distance = plane % 7 + 1
        to_file = from_file + direction[0] * distance
        to_rank = from_rank + direction[1] * distance
        return square_from_file_rank(to_file, to_rank), None

    if plane < 64:
        direction = KNIGHT_DIRECTIONS[plane - 56]
        to_file = from_file + direction[0]
        to_rank = from_rank + direction[1]
        return square_from_file_rank(to_file, to_rank), None

    promotion_index = (plane - 64) // 3
    file_delta_index = (plane - 64) % 3
    to_file = from_file + UNDERPROMOTION_FILE_DELTAS[file_delta_index]
    to_rank = from_rank + 1
    return square_from_file_rank(to_file, to_rank), UNDERPROMOTION_PIECES[promotion_index]


def normalize_direction(file_delta: int, rank_delta: int) -> tuple[int, int]:
    return (
        sign(file_delta),
        sign(rank_delta),
    )


def underpromotion_plane(file_delta: int, promotion: chess.PieceType) -> int:
    if file_delta not in UNDERPROMOTION_FILE_DELTAS:
        raise ValueError(f"invalid underpromotion file delta {file_delta}")

    promotion_index = UNDERPROMOTION_PIECES.index(promotion)
    file_delta_index = UNDERPROMOTION_FILE_DELTAS.index(file_delta)
    return 64 + promotion_index * 3 + file_delta_index


def square_from_file_rank(file: int, rank: int) -> chess.Square:
    if file < 0 or file > 7 or rank < 0 or rank > 7:
        raise ValueError(f"square is off board: file={file}, rank={rank}")
    return chess.square(file, rank)


def infer_queen_promotion(
    board: chess.Board,
    from_square: chess.Square,
    to_square: chess.Square,
) -> chess.PieceType | None:
    piece = board.piece_at(from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return None

    to_rank = chess.square_rank(to_square)
    if (piece.color == chess.WHITE and to_rank == 7) or (piece.color == chess.BLACK and to_rank == 0):
        return chess.QUEEN

    return None


def sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0
