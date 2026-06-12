import csv
import json

from chess_engine_2.match import GameResult, MatchResult, PlayerStats
from chess_engine_2.mcts_benchmark import (
    MCTSScalingRow,
    parse_simulations,
    row_from_match,
    save_csv,
    save_json,
)


def test_parse_simulations_accepts_spaces_and_commas() -> None:
    assert parse_simulations(["64,128", "256", "512"]) == [64, 128, 256, 512]


def test_row_from_match_collects_scaling_metrics() -> None:
    mcts_stats = PlayerStats()
    mcts_stats.record(depth=0, nodes=64, seconds=1.0)
    opponent_stats = PlayerStats()
    opponent_stats.record(depth=1, nodes=100, seconds=0.5)
    match = MatchResult(
        games=[
            GameResult("mcts-64", "search-depth-1", "1-0", 40, "checkmate", ""),
            GameResult("search-depth-1", "mcts-64", "1/2-1/2", 60, "threefold_repetition", ""),
        ],
        player_a="mcts-64",
        player_b="search-depth-1",
        player_a_stats=mcts_stats,
        player_b_stats=opponent_stats,
    )

    row = row_from_match(64, match, elapsed_seconds=3.0)

    assert row.wins == 1
    assert row.losses == 0
    assert row.draws == 1
    assert row.score_percent == 75.0
    assert row.average_plies == 50.0
    assert row.average_simulations_per_move == 64.0
    assert row.simulations_per_second == 64.0
    assert row.average_network_evaluations_per_move == 0.0
    assert row.average_cache_hits_per_move == 0.0
    assert row.cache_hit_percent == 0.0
    assert row.terminations == {"checkmate": 1, "threefold_repetition": 1}


def test_scaling_results_save_as_csv_and_json(tmp_path) -> None:
    row = MCTSScalingRow(
        simulations=64,
        games=2,
        wins=1,
        losses=1,
        draws=0,
        score=1.0,
        score_percent=50.0,
        average_plies=42.0,
        elapsed_seconds=10.0,
        average_simulations_per_move=64.0,
        simulations_per_second=70.0,
        average_network_evaluations_per_move=50.0,
        average_cache_hits_per_move=15.0,
        cache_hit_percent=23.1,
        opponent_average_nodes=80.0,
        opponent_nodes_per_second=1500.0,
        terminations={"checkmate": 2},
    )
    csv_path = tmp_path / "results.csv"
    json_path = tmp_path / "results.json"

    save_csv([row], csv_path)
    save_json([row], json_path, {"games_per_rung": 2})

    with csv_path.open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    json_data = json.loads(json_path.read_text(encoding="utf-8"))

    assert csv_rows[0]["simulations"] == "64"
    assert json_data["experiment"] == "MCTS inference scaling study"
    assert json_data["results"][0]["score_percent"] == 50.0
