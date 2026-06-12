from io import StringIO
import json
import math

import chess
import chess.pgn
import zstandard

from chess_engine_2.data.pgn import (
    GameFilter,
    iter_training_samples,
    material_value,
    parse_pgn_file,
    player_ratings,
    samples_from_game,
    write_training_jsonl,
)
from chess_engine_2.encoding import move_to_policy_index


TINY_PGN = """
[Event "Tiny"]
[Site "?"]
[Date "2026.01.01"]
[Round "-"]
[White "White"]
[Black "Black"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0
"""

RATED_PGN = """
[Event "Rated"]
[Site "?"]
[Date "2026.01.01"]
[Round "-"]
[White "Strong White"]
[Black "Strong Black"]
[WhiteElo "2200"]
[BlackElo "2100"]
[Result "1-0"]

1. e4 e5 1-0
"""

LOW_RATED_PGN = RATED_PGN.replace('[BlackElo "2100"]', '[BlackElo "1800"]')


def test_samples_from_game_extracts_policy_and_value_targets() -> None:
    game = chess.pgn.read_game(StringIO(TINY_PGN))
    samples = samples_from_game(game)

    assert len(samples) == 4
    assert samples[0].move_uci == "e2e4"
    assert samples[0].value == 1.0
    assert samples[1].move_uci == "e7e5"
    assert samples[1].value == -1.0
    assert samples[0].material_value == 0.0
    assert samples[0].ply == 1
    assert samples[0].game_plies == 4
    assert samples[0].discounted_value == 0.5
    assert samples[1].discounted_value == -math.sqrt(0.5)
    assert samples[-1].discounted_value == -1.0

    board = chess.Board()
    assert samples[0].policy_index == move_to_policy_index(chess.Move.from_uci("e2e4"), board)


def test_material_value_is_relative_to_side_to_move() -> None:
    white_to_move = chess.Board("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1")
    black_to_move = chess.Board("4k3/8/8/8/8/8/8/4KQ2 b - - 0 1")

    assert material_value(white_to_move) > 0
    assert material_value(black_to_move) < 0


def test_iter_training_samples_limits_games() -> None:
    stream = StringIO(TINY_PGN + "\n" + TINY_PGN)
    samples = list(iter_training_samples(stream, max_games=1))

    assert len(samples) == 4


def test_parse_pgn_file_reads_zst(tmp_path) -> None:
    path = tmp_path / "tiny.pgn.zst"
    compressed = zstandard.ZstdCompressor().compress(TINY_PGN.encode("utf-8"))
    path.write_bytes(compressed)

    samples, summary = parse_pgn_file(path, max_games=1)

    assert summary.games == 1
    assert summary.samples == 4
    assert samples[0].move_uci == "e2e4"


def test_write_training_jsonl_streams_samples(tmp_path) -> None:
    input_path = tmp_path / "tiny.pgn"
    output_path = tmp_path / "samples.jsonl"
    input_path.write_text(TINY_PGN, encoding="utf-8")

    summary = write_training_jsonl(input_path, output_path, max_games=1)
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert summary.games == 1
    assert summary.samples == 4
    assert len(rows) == 4
    assert rows[0]["move_uci"] == "e2e4"
    assert rows[0]["policy_index"] == 877
    assert rows[0]["value"] == 1.0
    assert rows[0]["discounted_value"] == 0.5
    assert rows[0]["ply"] == 1
    assert rows[0]["game_plies"] == 4


def test_game_filter_requires_both_players_to_meet_minimum_elo() -> None:
    strong_game = chess.pgn.read_game(StringIO(RATED_PGN))
    mixed_game = chess.pgn.read_game(StringIO(LOW_RATED_PGN))
    game_filter = GameFilter(min_elo=2000)

    assert player_ratings(strong_game) == (2200, 2100)
    assert game_filter.accepts(strong_game)
    assert not game_filter.accepts(mixed_game)
    assert not game_filter.accepts(chess.pgn.read_game(StringIO(TINY_PGN)))


def test_write_training_jsonl_limits_accepted_games_after_filtering(tmp_path) -> None:
    input_path = tmp_path / "rated.pgn"
    output_path = tmp_path / "samples.jsonl"
    input_path.write_text(LOW_RATED_PGN + "\n" + RATED_PGN + "\n" + RATED_PGN, encoding="utf-8")

    summary = write_training_jsonl(
        input_path,
        output_path,
        game_filter=GameFilter(min_elo=2000),
        max_output_games=1,
    )

    assert summary.games == 2
    assert summary.accepted_games == 1
    assert summary.filtered_games == 1
    assert summary.samples == 2


def test_write_training_jsonl_can_skip_training_prefix(tmp_path) -> None:
    input_path = tmp_path / "games.pgn"
    output_path = tmp_path / "samples.jsonl"
    input_path.write_text(TINY_PGN + "\n" + RATED_PGN, encoding="utf-8")

    summary = write_training_jsonl(
        input_path,
        output_path,
        max_output_games=1,
        skip_games=1,
    )
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert summary.games == 1
    assert summary.accepted_games == 1
    assert len(rows) == 2
