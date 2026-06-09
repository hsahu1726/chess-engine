import chess

from chess_engine_2.match import (
    GameResult,
    RandomPlayer,
    SearchPlayer,
    game_points_for,
    play_game,
    play_match,
    save_match_pgn,
)


def test_play_game_returns_result() -> None:
    result = play_game(RandomPlayer(), RandomPlayer(), max_plies=2)

    assert result.result in {"1-0", "0-1", "1/2-1/2"}
    assert result.plies <= 2


def test_play_game_can_start_with_random_opening() -> None:
    result = play_game(RandomPlayer(), RandomPlayer(), max_plies=4, opening_plies=2)

    assert result.plies >= 2


def test_play_match_alternates_colors() -> None:
    player_a = RandomPlayer("a")
    player_b = RandomPlayer("b")
    result = play_match(player_a, player_b, games=2, max_plies=1)

    assert result.games[0].white == "a"
    assert result.games[0].black == "b"
    assert result.games[1].white == "b"
    assert result.games[1].black == "a"


def test_game_points_for_player() -> None:
    white_win = GameResult("a", "b", "1-0", 10, "checkmate", "")
    black_win = GameResult("a", "b", "0-1", 10, "checkmate", "")
    draw = GameResult("a", "b", "1/2-1/2", 10, "move limit", "")

    assert game_points_for("a", white_win) == 1.0
    assert game_points_for("a", black_win) == 0.0
    assert game_points_for("a", draw) == 0.5


def test_match_summary_includes_score() -> None:
    result = play_match(RandomPlayer("a"), RandomPlayer("b"), games=1, max_plies=1)

    assert "games: 1" in result.summary()
    assert "a score:" in result.summary()


def test_search_player_returns_legal_move() -> None:
    board = chess.Board()
    move = SearchPlayer("search-depth-1", depth=1).choose_move(board)

    assert move in board.legal_moves


def test_timed_search_player_returns_legal_move() -> None:
    board = chess.Board()
    move = SearchPlayer("search-depth-4-1ms", depth=4, movetime_ms=1).choose_move(board)

    assert move in board.legal_moves


def test_save_match_pgn_writes_games(tmp_path) -> None:
    result = play_match(RandomPlayer("a"), RandomPlayer("b"), games=1, max_plies=1)
    path = tmp_path / "match.pgn"

    save_match_pgn(result, path)

    assert '[White "' in path.read_text(encoding="utf-8")
