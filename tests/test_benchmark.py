from chess_engine_2.benchmark import (
    BenchmarkRow,
    append_row_csv,
    benchmark_depth_streaming,
    format_rows,
    parse_int_list,
    row_from_prefix,
    save_rows_csv,
    search_player_name,
)
from chess_engine_2.match import SearchPlayer
from chess_engine_2.match import GameResult, MatchResult


def test_parse_int_list_accepts_spaces_and_commas() -> None:
    assert parse_int_list(["1", "2,3"]) == [1, 2, 3]


def test_search_player_name_includes_movetime() -> None:
    assert search_player_name(4, 100) == "search-depth-4-100ms"
    assert search_player_name(4, 100, use_mobility=False) == "search-depth-4-100ms-no-mobility"


def test_row_from_prefix_summarizes_match() -> None:
    match = MatchResult(
        [
            GameResult("a", "b", "1-0", 20, "checkmate", ""),
            GameResult("b", "a", "1/2-1/2", 40, "move limit", ""),
        ],
        "a",
        "b",
    )

    row = row_from_prefix(depth=1, match=match, elapsed_seconds=2.0)

    assert row.wins == 1
    assert row.losses == 0
    assert row.draws == 1
    assert row.score == 1.5
    assert row.score_percent == 75.0
    assert row.average_plies == 30.0


def test_format_rows_includes_header_and_values() -> None:
    output = format_rows(
        [
            BenchmarkRow(
                depth=1,
                games=10,
                wins=9,
                losses=0,
                draws=1,
                score=9.5,
                score_percent=95.0,
                average_plies=50.0,
                elapsed_seconds=1.0,
                a_average_depth=1.0,
                a_average_nodes=100.0,
                a_nps=1000.0,
            )
        ]
    )

    assert "depth | games" in output
    assert "95.0%" in output
    assert "a dep" in output


def test_save_rows_csv(tmp_path) -> None:
    path = tmp_path / "benchmark.csv"
    save_rows_csv(
        [
            BenchmarkRow(
                depth=1,
                games=10,
                wins=9,
                losses=0,
                draws=1,
                score=9.5,
                score_percent=95.0,
                average_plies=50.0,
                elapsed_seconds=1.0,
            )
        ],
        path,
    )

    assert "score_percent" in path.read_text(encoding="utf-8")


def test_append_row_csv(tmp_path) -> None:
    path = tmp_path / "benchmark.csv"
    row = BenchmarkRow(1, 10, 9, 0, 1, 9.5, 95.0, 50.0, 1.0)

    append_row_csv(row, path)
    append_row_csv(row, path)

    assert path.read_text(encoding="utf-8").count("score_percent") == 1


def test_benchmark_depth_streaming_returns_checkpoints() -> None:
    rows = benchmark_depth_streaming(depth=1, game_counts=[1, 2], max_plies=1)

    assert [row.games for row in rows] == [1, 2]


def test_benchmark_depth_streaming_accepts_search_opponent() -> None:
    rows = benchmark_depth_streaming(
        depth=1,
        game_counts=[1],
        max_plies=1,
        opponent=SearchPlayer("search-depth-1", depth=1),
    )

    assert rows[0].games == 1


def test_benchmark_depth_streaming_accepts_opening_plies() -> None:
    rows = benchmark_depth_streaming(depth=1, game_counts=[1], max_plies=4, opening_plies=2)

    assert rows[0].games == 1
