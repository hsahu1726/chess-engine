from io import StringIO

import chess

from chess_engine_2.uci import (
    UciSession,
    allocate_movetime,
    build_engine,
    choose_movetime_from_clock,
    parse_go_depth,
    parse_go_int,
    parse_go_movetime,
)


def run_session(commands: str) -> str:
    output = StringIO()
    UciSession(input_stream=StringIO(commands), output_stream=output).run()
    return output.getvalue()


def test_uci_handshake() -> None:
    output = run_session("uci\nquit\n")

    assert "id name Chess Engine 2" in output
    assert "uciok" in output


def test_isready() -> None:
    assert run_session("isready\nquit\n") == "readyok\n"


def test_position_startpos_moves_and_go_returns_bestmove() -> None:
    output = run_session("position startpos moves e2e4 e7e5\ngo depth 2\nquit\n")

    assert "info depth 1" in output
    assert "info depth 2" in output
    assert "bestmove " in output
    assert "bestmove 0000" not in output


def test_parse_go_depth() -> None:
    assert parse_go_depth("depth 3") == 3
    assert parse_go_depth("movetime 100") is None
    assert parse_go_depth("depth nope") is None


def test_parse_go_movetime() -> None:
    assert parse_go_movetime("movetime 250") == 250
    assert parse_go_movetime("depth 3") is None
    assert parse_go_movetime("movetime nope") is None


def test_parse_go_int() -> None:
    assert parse_go_int("wtime 60000 btime 30000", "wtime") == 60000
    assert parse_go_int("wtime nope", "wtime") is None
    assert parse_go_int("wtime", "wtime") is None


def test_choose_movetime_from_clock_uses_side_to_move() -> None:
    args = "wtime 60000 btime 30000 winc 1000 binc 0"

    assert choose_movetime_from_clock(args, chess.WHITE) > choose_movetime_from_clock(args, chess.BLACK)


def test_allocate_movetime_is_capped() -> None:
    allocated = allocate_movetime(remaining_ms=30000, increment_ms=1000)

    assert 1 <= allocated <= 6000


def test_go_movetime_returns_bestmove() -> None:
    output = run_session("position startpos\ngo movetime 1\nquit\n")

    assert "bestmove " in output
    assert "bestmove 0000" not in output


def test_go_clock_returns_bestmove() -> None:
    output = run_session("position startpos\ngo wtime 1000 btime 1000 winc 0 binc 0\nquit\n")

    assert "bestmove " in output
    assert "bestmove 0000" not in output


def test_build_engine_configures_neural_ordering_without_checkpoint() -> None:
    engine = build_engine(neural_ordering="depth", neural_min_depth=3)

    assert engine.policy_ordering_mode == "depth"
    assert engine.policy_ordering_min_depth == 3
    assert engine.policy_scorer is None


def test_build_engine_configures_value_evaluation_mode_without_checkpoint() -> None:
    engine = build_engine(evaluation_mode="blend", neural_value_weight=0.3)

    assert engine.evaluation_mode == "blend"
    assert engine.neural_value_weight == 0.3
    assert engine.value_evaluator is None
