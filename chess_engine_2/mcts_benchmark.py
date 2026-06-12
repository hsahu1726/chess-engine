from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from chess_engine_2.match import AdjudicationConfig, MCTSPlayer, MatchResult, SearchPlayer, play_match


@dataclass(frozen=True)
class MCTSScalingRow:
    simulations: int
    games: int
    wins: int
    losses: int
    draws: int
    score: float
    score_percent: float
    average_plies: float
    elapsed_seconds: float
    average_simulations_per_move: float
    simulations_per_second: float
    average_network_evaluations_per_move: float
    average_cache_hits_per_move: float
    cache_hit_percent: float
    opponent_average_nodes: float
    opponent_nodes_per_second: float
    terminations: dict[str, int]


def parse_simulations(values: list[str]) -> list[int]:
    simulations = []
    for value in values:
        for part in value.split(","):
            if part.strip():
                simulations.append(max(1, int(part)))
    return simulations


def row_from_match(simulations: int, match: MatchResult, elapsed_seconds: float) -> MCTSScalingRow:
    games = len(match.games)
    cache_requests = (
        match.player_a_stats.total_neural_value_evaluations
        + match.player_a_stats.total_neural_value_cache_hits
    )
    return MCTSScalingRow(
        simulations=simulations,
        games=games,
        wins=match.a_wins,
        losses=match.b_wins,
        draws=match.draws,
        score=match.a_score,
        score_percent=(match.a_score / games * 100) if games else 0.0,
        average_plies=match.average_plies,
        elapsed_seconds=elapsed_seconds,
        average_simulations_per_move=match.player_a_stats.average_nodes,
        simulations_per_second=match.player_a_stats.nodes_per_second,
        average_network_evaluations_per_move=match.player_a_stats.average_neural_value_evaluations,
        average_cache_hits_per_move=match.player_a_stats.average_neural_value_cache_hits,
        cache_hit_percent=(
            match.player_a_stats.total_neural_value_cache_hits / cache_requests * 100
            if cache_requests
            else 0.0
        ),
        opponent_average_nodes=match.player_b_stats.average_nodes,
        opponent_nodes_per_second=match.player_b_stats.nodes_per_second,
        terminations=match.termination_counts,
    )


def run_scaling_study(
    checkpoint: Path,
    simulation_counts: list[int],
    games: int = 2,
    opponent_depth: int = 1,
    channels: int = 32,
    cpuct: float = 1.5,
    max_plies: int = 200,
    opening_plies: int = 4,
    quiescence_depth: int = 2,
    seed: int = 1,
    cache_size: int = 100_000,
    adjudication: AdjudicationConfig | None = None,
) -> list[MCTSScalingRow]:
    rows = []
    for simulations in simulation_counts:
        random.seed(seed)
        mcts_player = MCTSPlayer(
            name=f"mcts-{simulations}",
            checkpoint=checkpoint,
            channels=channels,
            simulations=simulations,
            cpuct=cpuct,
            cache_size=cache_size,
        )
        opponent = SearchPlayer(
            name=f"search-depth-{opponent_depth}",
            depth=opponent_depth,
            quiescence_depth=quiescence_depth,
        )

        start = time.perf_counter()
        match = play_match(
            mcts_player,
            opponent,
            games=games,
            max_plies=max_plies,
            record_pgn=False,
            opening_plies=opening_plies,
            adjudication=adjudication,
        )
        rows.append(row_from_match(simulations, match, time.perf_counter() - start))
    return rows


def format_rows(rows: list[MCTSScalingRow]) -> str:
    header = (
        " simulations | games | wins | losses | draws | score% | avg plies | elapsed | "
        "sims/move | sims/sec | eval/move | hits/move | hit% | opp nodes | opp nps | terminations"
    )
    divider = (
        "-------------|-------|------|--------|-------|--------|-----------|---------|"
        "-----------|----------|-----------|-----------|------|-----------|---------|-------------"
    )
    lines = [header, divider]
    for row in rows:
        terminations = ", ".join(f"{name}={count}" for name, count in sorted(row.terminations.items()))
        lines.append(
            f"{row.simulations:>12} | {row.games:>5} | {row.wins:>4} | {row.losses:>6} | "
            f"{row.draws:>5} | {row.score_percent:>6.1f}% | {row.average_plies:>9.1f} | "
            f"{row.elapsed_seconds:>7.1f}s | {row.average_simulations_per_move:>9.0f} | "
            f"{row.simulations_per_second:>8.1f} | "
            f"{row.average_network_evaluations_per_move:>9.1f} | "
            f"{row.average_cache_hits_per_move:>9.1f} | {row.cache_hit_percent:>4.1f}% | "
            f"{row.opponent_average_nodes:>9.0f} | "
            f"{row.opponent_nodes_per_second:>7.0f} | {terminations}"
        )
    return "\n".join(lines)


def save_csv(rows: list[MCTSScalingRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "simulations",
        "games",
        "wins",
        "losses",
        "draws",
        "score",
        "score_percent",
        "average_plies",
        "elapsed_seconds",
        "average_simulations_per_move",
        "simulations_per_second",
        "average_network_evaluations_per_move",
        "average_cache_hits_per_move",
        "cache_hit_percent",
        "opponent_average_nodes",
        "opponent_nodes_per_second",
        "terminations",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            data["terminations"] = json.dumps(data["terminations"], sort_keys=True)
            writer.writerow(data)


def save_json(rows: list[MCTSScalingRow], path: Path, configuration: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment": "MCTS inference scaling study",
                "configuration": configuration,
                "results": [asdict(row) for row in rows],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure MCTS strength as simulations increase.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--simulations", nargs="+", default=["64", "128", "256", "512"])
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--opponent-depth", type=int, default=1)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--cpuct", type=float, default=1.5)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--qdepth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cache-size", type=int, default=100_000)
    parser.add_argument("--adjudicate", action="store_true")
    parser.add_argument("--adjudicate-eval", type=int, default=500)
    parser.add_argument("--adjudicate-eval-plies", type=int, default=8)
    parser.add_argument("--adjudicate-material", type=int, default=900)
    parser.add_argument("--adjudicate-material-plies", type=int, default=8)
    parser.add_argument("--adjudicate-min-plies", type=int, default=40)
    parser.add_argument("--csv", type=Path, default=Path("benchmark_phase14_mcts_scaling.csv"))
    parser.add_argument("--json", type=Path, default=Path("benchmark_phase14_mcts_scaling.json"))
    args = parser.parse_args()

    simulation_counts = parse_simulations(args.simulations)
    adjudication = AdjudicationConfig(
        enabled=args.adjudicate,
        eval_threshold=max(1, args.adjudicate_eval),
        eval_plies=max(1, args.adjudicate_eval_plies),
        material_threshold=max(1, args.adjudicate_material),
        material_plies=max(1, args.adjudicate_material_plies),
        min_plies=max(0, args.adjudicate_min_plies),
    )
    configuration = {
        "checkpoint": str(args.checkpoint),
        "simulations": simulation_counts,
        "games_per_rung": max(1, args.games),
        "opponent_depth": max(1, args.opponent_depth),
        "channels": max(1, args.channels),
        "cpuct": max(0.01, args.cpuct),
        "max_plies": max(1, args.max_plies),
        "opening_plies": max(0, args.opening_plies),
        "quiescence_depth": max(0, args.qdepth),
        "seed": args.seed,
        "cache_size": max(0, args.cache_size),
        "adjudication": asdict(adjudication),
    }
    rows = run_scaling_study(
        checkpoint=args.checkpoint,
        simulation_counts=simulation_counts,
        games=max(1, args.games),
        opponent_depth=max(1, args.opponent_depth),
        channels=max(1, args.channels),
        cpuct=max(0.01, args.cpuct),
        max_plies=max(1, args.max_plies),
        opening_plies=max(0, args.opening_plies),
        quiescence_depth=max(0, args.qdepth),
        seed=args.seed,
        cache_size=max(0, args.cache_size),
        adjudication=adjudication,
    )
    save_csv(rows, args.csv)
    save_json(rows, args.json, configuration)
    print(format_rows(rows))
    print(f"csv: {args.csv}")
    print(f"json: {args.json}")


if __name__ == "__main__":
    main()
