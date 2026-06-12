import math

import json

from chess_engine_2.distill_rankings import load_completed_groups, normalized_child_target


def test_normalized_child_target_uses_child_perspective() -> None:
    assert normalized_child_target(0) == 0.0
    assert normalized_child_target(600) == -math.tanh(1.0)
    assert normalized_child_target(-600) == math.tanh(1.0)


def test_load_completed_groups_indexes_by_root_fen(tmp_path) -> None:
    path = tmp_path / "groups.jsonl"
    group = {"root_fen": "example", "child_depth": 3, "children": []}
    path.write_text(json.dumps(group) + "\n", encoding="utf-8")

    assert load_completed_groups(path) == {"example": group}
