from __future__ import annotations

import argparse
import io
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import zstandard

from chess_engine_2.data.download import lichess_month_url
from chess_engine_2.data.pgn import GameFilter, write_training_jsonl_stream


@contextmanager
def open_remote_pgn_text(url: str, timeout_seconds: float = 60.0):
    response = urllib.request.urlopen(url, timeout=timeout_seconds)
    reader = zstandard.ZstdDecompressor().stream_reader(response, read_across_frames=True)
    text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    try:
        yield text_stream
    finally:
        text_stream.close()


def export_remote_lichess_month(
    month: str,
    output_path: Path,
    min_elo: int,
    max_output_games: int,
    max_games: int | None = None,
    timeout_seconds: float = 60.0,
    skip_output_games: int = 0,
):
    url = lichess_month_url(month)
    with open_remote_pgn_text(url, timeout_seconds) as stream:
        return write_training_jsonl_stream(
            stream,
            output_path,
            max_games=max_games,
            game_filter=GameFilter(min_elo=min_elo),
            max_output_games=max_output_games,
            skip_output_games=skip_output_games,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream filtered Lichess games without downloading a full archive.")
    parser.add_argument("--month", required=True, help="Month in YYYY-MM format.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-elo", type=int, default=2000)
    parser.add_argument("--max-output-games", type=int, default=1000)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--skip-output-games", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    summary = export_remote_lichess_month(
        args.month,
        args.output,
        max(0, args.min_elo),
        max(1, args.max_output_games),
        args.max_games,
        args.timeout,
        max(0, args.skip_output_games),
    )
    print(f"source: {lichess_month_url(args.month)}")
    print(f"wrote: {args.output}")
    print(f"games scanned: {summary.games}")
    print(f"games accepted: {summary.accepted_games}")
    print(f"games filtered: {summary.filtered_games}")
    print(f"qualifying games skipped: {summary.skipped_output_games}")
    print(f"samples: {summary.samples}")
    print(f"skipped games: {summary.skipped_games}")


if __name__ == "__main__":
    main()
