from io import StringIO
import json

import chess
import chess.pgn
import zstandard

from chess_engine_2.data.pgn import iter_training_samples, parse_pgn_file, samples_from_game, write_training_jsonl
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


def test_samples_from_game_extracts_policy_and_value_targets() -> None:
    game = chess.pgn.read_game(StringIO(TINY_PGN))
    samples = samples_from_game(game)

    assert len(samples) == 4
    assert samples[0].move_uci == "e2e4"
    assert samples[0].value == 1.0
    assert samples[1].move_uci == "e7e5"
    assert samples[1].value == -1.0

    board = chess.Board()
    assert samples[0].policy_index == move_to_policy_index(chess.Move.from_uci("e2e4"), board)


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
