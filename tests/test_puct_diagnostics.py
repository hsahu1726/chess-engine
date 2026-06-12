import chess

from chess_engine_2.puct_diagnostics import collect_diagnostics, parse_numbers


def test_parse_numbers_accepts_spaces_and_commas() -> None:
    assert parse_numbers(["0.25,0.5", "1.0"], float) == [0.25, 0.5, 1.0]


def test_collect_diagnostics_reports_root_terms(monkeypatch, tmp_path) -> None:
    class FakeEvaluator:
        def __call__(self, board):
            moves = list(board.legal_moves)
            return {move: 1.0 / len(moves) for move in moves}, 0.25

    monkeypatch.setattr(
        "chess_engine_2.puct_diagnostics.NeuralPolicyValue.from_checkpoint",
        lambda *args, **kwargs: FakeEvaluator(),
    )

    rows = collect_diagnostics(
        tmp_path / "unused.pt",
        [0.5],
        [4],
        fens=(chess.STARTING_FEN,),
        top_n=3,
    )

    assert len(rows) == 1
    assert rows[0].cpuct == 0.5
    assert rows[0].leaf_value_mean == 0.25
    assert len(rows[0].top_moves) == 3
    assert {"prior", "visits", "q_value", "exploration", "puct_score"} <= rows[0].top_moves[0].keys()
