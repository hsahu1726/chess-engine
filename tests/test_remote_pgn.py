from io import BytesIO

import zstandard

from chess_engine_2.data import remote_pgn
from chess_engine_2.data.pgn import GameFilter, write_training_jsonl_stream


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


def test_remote_stream_can_filter_and_stop_early(tmp_path, monkeypatch) -> None:
    compressed = zstandard.ZstdCompressor().compress((RATED_PGN + "\n" + RATED_PGN).encode("utf-8"))

    def fake_urlopen(url: str, timeout: float):
        assert url.endswith(".pgn.zst")
        assert timeout == 30.0
        return BytesIO(compressed)

    monkeypatch.setattr(remote_pgn.urllib.request, "urlopen", fake_urlopen)
    output_path = tmp_path / "samples.jsonl"

    with remote_pgn.open_remote_pgn_text("https://example.test/games.pgn.zst", 30.0) as stream:
        summary = write_training_jsonl_stream(
            stream,
            output_path,
            game_filter=GameFilter(min_elo=2000),
            max_output_games=1,
        )

    assert summary.games == 1
    assert summary.accepted_games == 1
    assert summary.samples == 2


def test_remote_stream_can_skip_qualifying_validation_games(tmp_path, monkeypatch) -> None:
    compressed = zstandard.ZstdCompressor().compress((RATED_PGN + "\n" + RATED_PGN).encode("utf-8"))
    monkeypatch.setattr(
        remote_pgn.urllib.request,
        "urlopen",
        lambda url, timeout: BytesIO(compressed),
    )
    output_path = tmp_path / "samples.jsonl"

    with remote_pgn.open_remote_pgn_text("https://example.test/games.pgn.zst") as stream:
        summary = write_training_jsonl_stream(
            stream,
            output_path,
            game_filter=GameFilter(min_elo=2000),
            max_output_games=1,
            skip_output_games=1,
        )

    assert summary.games == 2
    assert summary.skipped_output_games == 1
    assert summary.accepted_games == 1
