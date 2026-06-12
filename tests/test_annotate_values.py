import json

import chess
import pytest

from chess_engine_2.annotate_values import annotate_jsonl, normalize_classical_score
from chess_engine_2.encoding import move_to_policy_index


def test_normalize_classical_score_is_bounded_and_symmetric() -> None:
    assert normalize_classical_score(0) == 0.0
    assert normalize_classical_score(600) == pytest.approx(-normalize_classical_score(-600))
    assert 0.0 < normalize_classical_score(600) < 1.0


def test_annotate_jsonl_preserves_fields_and_adds_dense_targets(tmp_path) -> None:
    board = chess.Board()
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "fen": board.fen(),
                "move_uci": "e2e4",
                "policy_index": move_to_policy_index(chess.Move.from_uci("e2e4"), board),
                "value": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = annotate_jsonl(input_path, output_path, lambda position: 300)
    row = json.loads(output_path.read_text(encoding="utf-8"))

    assert summary.rows == 1
    assert row["move_uci"] == "e2e4"
    assert row["material_value"] == 0.0
    assert row["classical_value"] == pytest.approx(normalize_classical_score(300))


def test_annotate_jsonl_can_add_material_only(tmp_path) -> None:
    board = chess.Board()
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "fen": board.fen(),
                "move_uci": "e2e4",
                "policy_index": move_to_policy_index(chess.Move.from_uci("e2e4"), board),
                "value": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    annotate_jsonl(input_path, output_path)
    row = json.loads(output_path.read_text(encoding="utf-8"))

    assert row["material_value"] == 0.0
    assert "classical_value" not in row


def test_annotate_jsonl_can_backfill_discounted_game_progress(tmp_path) -> None:
    board = chess.Board()
    rows = []
    for move_uci in ("e2e4", "e7e5", "g1f3", "b8c6"):
        move = chess.Move.from_uci(move_uci)
        rows.append(
            {
                "fen": board.fen(),
                "move_uci": move_uci,
                "policy_index": move_to_policy_index(move, board),
                "value": 1.0 if board.turn == chess.WHITE else -1.0,
            }
        )
        board.push(move)

    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    annotate_jsonl(input_path, output_path, backfill_progress=True)
    output_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert output_rows[0]["ply"] == 1
    assert output_rows[0]["game_plies"] == 4
    assert output_rows[0]["discounted_value"] == 0.5
    assert output_rows[-1]["discounted_value"] == -1.0
