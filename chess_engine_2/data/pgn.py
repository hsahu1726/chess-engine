from __future__ import annotations

import argparse
import io
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn
import zstandard

from chess_engine_2.encoding import move_to_policy_index


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


@dataclass(frozen=True)
class PgnParseSummary:
    games: int
    samples: int
    skipped_games: int


def samples_from_game(game: chess.pgn.Game) -> list[TrainingSample]:
    result = game.headers.get("Result", "*")
    if result not in RESULT_TO_WHITE_VALUE:
        return []

    board = game.board()
    white_value = RESULT_TO_WHITE_VALUE[result]
    samples = []

    for move in game.mainline_moves():
        value = white_value if board.turn == chess.WHITE else -white_value
        samples.append(
            TrainingSample(
                fen=board.fen(),
                move_uci=move.uci(),
                policy_index=move_to_policy_index(move, board),
                value=value,
            )
        )
        board.push(move)

    return samples


def iter_training_samples(stream: TextIO, max_games: int | None = None):
    games_seen = 0
    while max_games is None or games_seen < max_games:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        games_seen += 1
        yield from samples_from_game(game)


@contextmanager
def open_pgn_text(path: Path):
    if path.suffix == ".zst":
        with path.open("rb") as compressed:
            reader = zstandard.ZstdDecompressor().stream_reader(compressed)
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            try:
                yield text_stream
            finally:
                text_stream.close()
    else:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            yield stream


def parse_pgn_file(path: Path, max_games: int | None = None) -> tuple[list[TrainingSample], PgnParseSummary]:
    samples = []
    games = 0
    skipped_games = 0

    with open_pgn_text(path) as stream:
        while max_games is None or games < max_games:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            games += 1
            game_samples = samples_from_game(game)
            if game_samples:
                samples.extend(game_samples)
            else:
                skipped_games += 1

    return samples, PgnParseSummary(games, len(samples), skipped_games)


def write_training_jsonl(
    input_path: Path,
    output_path: Path,
    max_games: int | None = None,
) -> PgnParseSummary:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    games = 0
    samples = 0
    skipped_games = 0

    with open_pgn_text(input_path) as stream:
        with output_path.open("w", encoding="utf-8", newline="\n") as output:
            while max_games is None or games < max_games:
                game = chess.pgn.read_game(stream)
                if game is None:
                    break
                games += 1

                game_samples = samples_from_game(game)
                if not game_samples:
                    skipped_games += 1
                    continue

                for sample in game_samples:
                    output.write(json.dumps(asdict(sample), separators=(",", ":")))
                    output.write("\n")
                    samples += 1

    return PgnParseSummary(games, samples, skipped_games)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a PGN file into supervised training samples.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--max-games", type=int, default=10)
    parser.add_argument("--output", type=Path, help="Write samples to a JSONL file without keeping them in memory.")
    args = parser.parse_args()

    if args.output:
        summary = write_training_jsonl(args.path, args.output, args.max_games)
        print(f"wrote: {args.output}")
        print(f"games: {summary.games}")
        print(f"samples: {summary.samples}")
        print(f"skipped games: {summary.skipped_games}")
        return

    samples, summary = parse_pgn_file(args.path, args.max_games)
    print(f"games: {summary.games}")
    print(f"samples: {summary.samples}")
    print(f"skipped games: {summary.skipped_games}")
    if samples:
        first = samples[0]
        print(f"first sample: {first.fen} -> {first.move_uci} policy={first.policy_index} value={first.value}")


if __name__ == "__main__":
    main()
