import chess

from chess_engine_2.match import (
    AdjudicationConfig,
    AdjudicationState,
    GameResult,
    NeuralPolicyPlayer,
    MCTSPlayer,
    RandomPlayer,
    SearchPlayer,
    adjudicate_game,
    build_player,
    game_points_for,
    material_score,
    play_game,
    play_match,
    save_match_pgn,
    white_relative_eval,
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


def test_white_relative_eval_is_positive_for_white_advantage() -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1")

    assert white_relative_eval(board) > 800
    board.turn = chess.BLACK
    assert white_relative_eval(board) > 800


def test_material_score_is_white_relative() -> None:
    assert material_score(chess.Board("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1")) == 900
    assert material_score(chess.Board("4kq2/8/8/8/8/8/8/4K3 w - - 0 1")) == -900


def test_adjudicate_game_declares_eval_win_after_consecutive_plies() -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/4KQ2 w - - 0 10")
    state = AdjudicationState()
    config = AdjudicationConfig(enabled=True, eval_threshold=500, eval_plies=2, min_plies=0)

    assert adjudicate_game(board, config, state) is None
    assert adjudicate_game(board, config, state) == ("1-0", "adjudicated eval")


def test_adjudicate_game_declares_material_win_after_consecutive_plies() -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/4KQ2 w - - 0 10")
    state = AdjudicationState()
    config = AdjudicationConfig(
        enabled=True,
        eval_threshold=10_000,
        material_threshold=500,
        material_plies=2,
        min_plies=0,
    )

    assert adjudicate_game(board, config, state) is None
    assert adjudicate_game(board, config, state) == ("1-0", "adjudicated material")


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


def test_search_player_accepts_policy_scorer() -> None:
    class FakePolicyScorer:
        calls = 0

        def score_moves(self, board: chess.Board):
            self.calls += 1
            return {chess.Move.from_uci("a2a3"): 100.0}

    scorer = FakePolicyScorer()
    player = SearchPlayer("search-policy", depth=1)
    player.policy_scorer = scorer
    player.reset_for_new_game()
    board = chess.Board()
    move = player.choose_move(board)

    assert move in board.legal_moves
    assert scorer.calls > 0


def test_neural_policy_player_chooses_highest_scored_legal_move() -> None:
    class FakePolicyScorer:
        def score_moves(self, board: chess.Board):
            return {
                chess.Move.from_uci("a2a3"): 1.0,
                chess.Move.from_uci("e2e4"): 5.0,
            }

    player = NeuralPolicyPlayer.__new__(NeuralPolicyPlayer)
    player.name = "neural-policy"
    player.checkpoint = None
    player.channels = 32
    player.stats = RandomPlayer().stats
    player.policy_scorer = FakePolicyScorer()

    assert player.choose_move(chess.Board()) == chess.Move.from_uci("e2e4")


def test_mcts_player_returns_legal_move_with_fake_policy_value() -> None:
    def policy_value(board: chess.Board):
        moves = list(board.legal_moves)
        return {move: 1.0 / len(moves) for move in moves}, 0.0

    player = MCTSPlayer.__new__(MCTSPlayer)
    player.name = "mcts-test"
    player.checkpoint = None
    player.channels = 32
    player.simulations = 4
    player.cpuct = 1.5
    player.stats = RandomPlayer().stats
    player.policy_value = policy_value

    move = player.choose_move(chess.Board())

    assert move in chess.Board().legal_moves
    assert player.stats.total_nodes == 4


def test_build_player_rejects_neural_without_checkpoint() -> None:
    try:
        build_player("neural", depth=1)
    except ValueError as exc:
        assert "checkpoint" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_player_rejects_mcts_without_checkpoint() -> None:
    try:
        build_player("mcts", depth=1)
    except ValueError as exc:
        assert "checkpoint" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_save_match_pgn_writes_games(tmp_path) -> None:
    result = play_match(RandomPlayer("a"), RandomPlayer("b"), games=1, max_plies=1)
    path = tmp_path / "match.pgn"

    save_match_pgn(result, path)

    assert '[White "' in path.read_text(encoding="utf-8")
