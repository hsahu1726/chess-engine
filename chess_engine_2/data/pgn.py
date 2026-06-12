from __future__ import annotations

import argparse
import io
import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn
import zstandard

from chess_engine_2.encoding import move_to_policy_index
from chess_engine_2.engine import PIECE_VALUES


RESULT_TO_WHITE_VALUE = {
    "1-0": 1.0,
    "1/2-1/2": 0.0,
    "0-1": -1.0,
}


@dataclass(frozen=True)
class TrainingSample:
    fen: str
    move_uci: str
    policy_index: int
    value: float
    material_value: float
    discounted_value: float
    ply: int
    game_plies: int


@dataclass(frozen=True)
class PgnParseSummary:
    games: int
    samples: int
    skipped_games: int
    accepted_games: int = 0
    filtered_games: int = 0
    skipped_output_games: int = 0


@dataclass(frozen=True)
class GameFilter:
    min_elo: int | None = None
    max_elo: int | None = None
    require_both_ratings: bool = True

    def accepts(self, game: chess.pgn.Game) -> bool:
        ratings = player_ratings(game)
        if ratings is None:
            return not self.require_both_ratings and self.min_elo is None and self.max_elo is None
        white_elo, black_elo = ratings
        if self.min_elo is not None and (white_elo < self.min_elo or black_elo < self.min_elo):
            return False
        if self.max_elo is not None and (white_elo > self.max_elo or black_elo > self.max_elo):
            return False
        return True


def player_ratings(game: chess.pgn.Game) -> tuple[int, int] | None:
    try:
        return int(game.headers["WhiteElo"]), int(game.headers["BlackElo"])
    except (KeyError, TypeError, ValueError):
        return None


def samples_from_game(game: chess.pgn.Game) -> list[TrainingSample]:
    result = game.headers.get("Result", "*")
    if result not in RESULT_TO_WHITE_VALUE:
        return []

    board = game.board()
    white_value = RESULT_TO_WHITE_VALUE[result]
    moves = list(game.mainline_moves())
    game_plies = len(moves)
    samples = []

    for ply, move in enumerate(moves, start=1):
        value = white_value if board.turn == chess.WHITE else -white_value
        samples.append(
            TrainingSample(
                fen=board.fen(),
                move_uci=move.uci(),
                policy_index=move_to_policy_index(move, board),
                value=value,
                material_value=material_value(board),
                discounted_value=value * math.sqrt(ply / game_plies),
                ply=ply,
                game_plies=game_plies,
            )
        )
        board.push(move)

    return samples


def material_value(board: chess.Board) -> float:
    white_score = 0
    black_score = 0
    for piece in board.piece_map().values():
        if piece.piece_type == chess.KING:
            continue
        if piece.color == chess.WHITE:
            white_score += PIECE_VALUES[piece.piece_type]
        else:
            black_score += PIECE_VALUES[piece.piece_type]
    side_to_move_score = white_score - black_score
    if board.turn == chess.BLACK:
        side_to_move_score = -side_to_move_score
    return max(-1.0, min(1.0, side_to_move_score / 3900.0))


def iter_training_samples(
    stream: TextIO,
    max_games: int | None = None,
    game_filter: GameFilter | None = None,
    max_output_games: int | None = None,
):
    games_seen = 0
    accepted_games = 0
    while (max_games is None or games_seen < max_games) and (
        max_output_games is None or accepted_games < max_output_games
    ):
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        games_seen += 1
        if game_filter is not None and not game_filter.accepts(game):
            continue
        accepted_games += 1
        yield from samples_from_game(game)


@contextmanager
def open_pgn_text(path: Path):
    if path.suffix == ".zst":
        with path.open("rb") as compressed:
            reader = zstandard.ZstdDecompressor().stream_reader(compressed, read_across_frames=True)
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            try:
                yield text_stream
            finally:
                text_stream.close()
    else:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            yield stream


def parse_pgn_file(
    path: Path,
    max_games: int | None = None,
    game_filter: GameFilter | None = None,
    max_output_games: int | None = None,
    skip_games: int = 0,
) -> tuple[list[TrainingSample], PgnParseSummary]:
    samples = []
    games = 0
    skipped_games = 0
    accepted_games = 0
    filtered_games = 0

    with open_pgn_text(path) as stream:
        for _ in range(max(0, skip_games)):
            if chess.pgn.read_game(stream) is None:
                return samples, PgnParseSummary(games, 0, 0, 0, 0)
        while (max_games is None or games < max_games) and (
            max_output_games is None or accepted_games < max_output_games
        ):
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            games += 1
            if game_filter is not None and not game_filter.accepts(game):
                filtered_games += 1
                continue
            game_samples = samples_from_game(game)
            if game_samples:
                accepted_games += 1
                samples.extend(game_samples)
            else:
                skipped_games += 1

    return samples, PgnParseSummary(games, len(samples), skipped_games, accepted_games, filtered_games)


def write_training_jsonl(
    input_path: Path,
    output_path: Path,
    max_games: int | None = None,
    game_filter: GameFilter | None = None,
    max_output_games: int | None = None,
    skip_games: int = 0,
) -> PgnParseSummary:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open_pgn_text(input_path) as stream:
        return write_training_jsonl_stream(
            stream,
            output_path,
            max_games,
            game_filter,
            max_output_games,
            skip_games,
        )


def write_training_jsonl_stream(
    stream: TextIO,
    output_path: Path,
    max_games: int | None = None,
    game_filter: GameFilter | None = None,
    max_output_games: int | None = None,
    skip_games: int = 0,
    skip_output_games: int = 0,
) -> PgnParseSummary:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    games = 0
    samples = 0
    skipped_games = 0
    accepted_games = 0
    filtered_games = 0
    skipped_accepted_games = 0

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for _ in range(max(0, skip_games)):
            if chess.pgn.read_game(stream) is None:
                return PgnParseSummary(games, samples, skipped_games, accepted_games, filtered_games)
        while (max_games is None or games < max_games) and (
            max_output_games is None or accepted_games < max_output_games
        ):
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            games += 1
            if game_filter is not None and not game_filter.accepts(game):
                filtered_games += 1
                continue

            game_samples = samples_from_game(game)
            if not game_samples:
                skipped_games += 1
                continue

            if skipped_accepted_games < max(0, skip_output_games):
                skipped_accepted_games += 1
                continue

            accepted_games += 1
            for sample in game_samples:
                output.write(json.dumps(asdict(sample), separators=(",", ":")))
                output.write("\n")
                samples += 1

    return PgnParseSummary(
        games,
        samples,
        skipped_games,
        accepted_games,
        filtered_games,
        skipped_accepted_games,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a PGN file into supervised training samples.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--max-games", type=int, default=10)
    parser.add_argument("--max-output-games", type=int)
    parser.add_argument("--skip-games", type=int, default=0)
    parser.add_argument("--min-elo", type=int)
    parser.add_argument("--max-elo", type=int)
    parser.add_argument("--output", type=Path, help="Write samples to a JSONL file without keeping them in memory.")
    args = parser.parse_args()
    game_filter = (
        GameFilter(min_elo=args.min_elo, max_elo=args.max_elo)
        if args.min_elo is not None or args.max_elo is not None
        else None
    )

    if args.output:
        summary = write_training_jsonl(
            args.path,
            args.output,
            args.max_games,
            game_filter,
            args.max_output_games,
            args.skip_games,
        )
        print(f"wrote: {args.output}")
        print(f"games scanned: {summary.games}")
        print(f"games accepted: {summary.accepted_games}")
        print(f"games filtered: {summary.filtered_games}")
        print(f"samples: {summary.samples}")
        print(f"skipped games: {summary.skipped_games}")
        return

    samples, summary = parse_pgn_file(
        args.path,
        args.max_games,
        game_filter,
        args.max_output_games,
        args.skip_games,
    )
    print(f"games scanned: {summary.games}")
    print(f"games accepted: {summary.accepted_games}")
    print(f"games filtered: {summary.filtered_games}")
    print(f"samples: {summary.samples}")
    print(f"skipped games: {summary.skipped_games}")
    if samples:
        first = samples[0]
        print(f"first sample: {first.fen} -> {first.move_uci} policy={first.policy_index} value={first.value}")


if __name__ == "__main__":
    main()
