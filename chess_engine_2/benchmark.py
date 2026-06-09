from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

from chess_engine_2.match import (
    AdjudicationConfig,
    MatchResult,
    Player,
    RandomPlayer,
    SearchPlayer,
    build_player,
    play_game,
    play_match,
)


DEFAULT_GAME_COUNTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
CSV_FIELDNAMES = [
    "depth",
    "games",
    "wins",
    "losses",
    "draws",
    "score",
    "score_percent",
    "average_plies",
    "elapsed_seconds",
    "a_average_depth",
    "a_average_nodes",
    "a_average_main_nodes",
    "a_average_quiescence_nodes",
    "a_nps",
    "a_average_evaluations",
    "a_average_mobility_evaluations",
    "a_average_neural_value_evaluations",
    "a_average_neural_value_cache_hits",
    "b_average_depth",
    "b_average_nodes",
    "b_average_main_nodes",
    "b_average_quiescence_nodes",
    "b_nps",
    "b_average_evaluations",
    "b_average_mobility_evaluations",
    "b_average_neural_value_evaluations",
    "b_average_neural_value_cache_hits",
]


@dataclass
class BenchmarkRow:
    depth: int
    games: int
    wins: int
    losses: int
    draws: int
    score: float
    score_percent: float
    average_plies: float
    elapsed_seconds: float
    a_average_depth: float = 0.0
    a_average_nodes: float = 0.0
    a_average_main_nodes: float = 0.0
    a_average_quiescence_nodes: float = 0.0
    a_nps: float = 0.0
    a_average_evaluations: float = 0.0
    a_average_mobility_evaluations: float = 0.0
    a_average_neural_value_evaluations: float = 0.0
    a_average_neural_value_cache_hits: float = 0.0
    b_average_depth: float = 0.0
    b_average_nodes: float = 0.0
    b_average_main_nodes: float = 0.0
    b_average_quiescence_nodes: float = 0.0
    b_nps: float = 0.0
    b_average_evaluations: float = 0.0
    b_average_mobility_evaluations: float = 0.0
    b_average_neural_value_evaluations: float = 0.0
    b_average_neural_value_cache_hits: float = 0.0

    def format(self) -> str:
        return (
            f"{self.depth:>5} | {self.games:>5} | {self.wins:>4} | {self.losses:>6} | "
            f"{self.draws:>5} | {self.score:>6.1f} | {self.score_percent:>6.1f}% | "
            f"{self.average_plies:>8.1f} | {self.elapsed_seconds:>7.2f}s | "
            f"{self.a_average_depth:>5.2f} | {self.a_average_nodes:>8.0f} | {self.a_nps:>7.0f} | "
            f"{self.a_average_main_nodes:>6.0f} | {self.a_average_quiescence_nodes:>5.0f} | "
            f"{self.a_average_evaluations:>7.0f} | {self.a_average_mobility_evaluations:>6.0f} | "
            f"{self.a_average_neural_value_evaluations:>6.0f} | {self.a_average_neural_value_cache_hits:>6.0f} | "
            f"{self.b_average_depth:>5.2f} | {self.b_average_nodes:>8.0f} | {self.b_nps:>7.0f} | "
            f"{self.b_average_main_nodes:>6.0f} | {self.b_average_quiescence_nodes:>5.0f} | "
            f"{self.b_average_evaluations:>7.0f} | {self.b_average_mobility_evaluations:>6.0f} | "
            f"{self.b_average_neural_value_evaluations:>6.0f} | {self.b_average_neural_value_cache_hits:>6.0f}"
        )


def benchmark_depth(
    depth: int,
    game_counts: list[int],
    max_plies: int,
    opponent: Player | None = None,
    opening_plies: int = 0,
    movetime_ms: int | None = None,
    use_mobility: bool = True,
    quiescence_depth: int = 6,
    time_check_interval: int = 1024,
    neural_checkpoint: Path | None = None,
    neural_channels: int = 32,
    neural_ordering_mode: str = "root",
    neural_min_depth: int = 2,
    value_checkpoint: Path | None = None,
    evaluation_mode: str = "classical",
    neural_value_weight: float = 0.2,
    neural_value_scale: int = 1000,
    adjudication: AdjudicationConfig | None = None,
) -> list[BenchmarkRow]:
    max_games = max(game_counts)
    player_a = SearchPlayer(
        search_player_name(depth, movetime_ms, use_mobility, neural_checkpoint is not None),
        depth,
        movetime_ms,
        use_mobility,
        quiescence_depth,
        time_check_interval,
        neural_checkpoint,
        neural_channels,
        neural_ordering_mode,
        neural_min_depth,
        value_checkpoint,
        evaluation_mode,
        neural_value_weight,
        neural_value_scale,
    )
    player_b = opponent or RandomPlayer()
    start = time.perf_counter()
    match = play_match(
        player_a,
        player_b,
        games=max_games,
        max_plies=max_plies,
        record_pgn=False,
        opening_plies=opening_plies,
        adjudication=adjudication,
    )
    elapsed = time.perf_counter() - start
    return [
        row_from_prefix(depth, prefix_match(match, games), elapsed * games / max_games)
        for games in game_counts
    ]


def benchmark_depth_streaming(
    depth: int,
    game_counts: list[int],
    max_plies: int,
    opponent: Player | None = None,
    opening_plies: int = 0,
    movetime_ms: int | None = None,
    use_mobility: bool = True,
    quiescence_depth: int = 6,
    time_check_interval: int = 1024,
    neural_checkpoint: Path | None = None,
    neural_channels: int = 32,
    neural_ordering_mode: str = "root",
    neural_min_depth: int = 2,
    value_checkpoint: Path | None = None,
    evaluation_mode: str = "classical",
    neural_value_weight: float = 0.2,
    neural_value_scale: int = 1000,
    adjudication: AdjudicationConfig | None = None,
) -> list[BenchmarkRow]:
    checkpoints = sorted(set(game_counts))
    max_games = max(checkpoints)
    player_a = SearchPlayer(
        search_player_name(depth, movetime_ms, use_mobility, neural_checkpoint is not None),
        depth,
        movetime_ms,
        use_mobility,
        quiescence_depth,
        time_check_interval,
        neural_checkpoint,
        neural_channels,
        neural_ordering_mode,
        neural_min_depth,
        value_checkpoint,
        evaluation_mode,
        neural_value_weight,
        neural_value_scale,
    )
    player_b = opponent or RandomPlayer()
    games = []
    rows = []
    start = time.perf_counter()

    for game_index in range(max_games):
        if game_index % 2 == 0:
            white, black = player_a, player_b
        else:
            white, black = player_b, player_a
        games.append(
            play_game(
                white,
                black,
                max_plies,
                record_pgn=False,
                opening_plies=opening_plies,
                adjudication=adjudication,
            )
        )

        completed_games = game_index + 1
        if completed_games in checkpoints:
            match = MatchResult(games.copy(), player_a.name, player_b.name, player_a.stats, player_b.stats)
            rows.append(row_from_prefix(depth, match, time.perf_counter() - start))

    return rows


def prefix_match(match: MatchResult, games: int) -> MatchResult:
    return MatchResult(
        match.games[:games],
        match.player_a,
        match.player_b,
        match.player_a_stats,
        match.player_b_stats,
    )


def row_from_prefix(depth: int, match: MatchResult, elapsed_seconds: float) -> BenchmarkRow:
    games = len(match.games)
    score_percent = (match.a_score / games * 100) if games else 0.0
    return BenchmarkRow(
        depth=depth,
        games=games,
        wins=match.a_wins,
        losses=match.b_wins,
        draws=match.draws,
        score=match.a_score,
        score_percent=score_percent,
        average_plies=match.average_plies,
        elapsed_seconds=elapsed_seconds,
        a_average_depth=match.player_a_stats.average_depth,
        a_average_nodes=match.player_a_stats.average_nodes,
        a_average_main_nodes=match.player_a_stats.average_main_nodes,
        a_average_quiescence_nodes=match.player_a_stats.average_quiescence_nodes,
        a_nps=match.player_a_stats.nodes_per_second,
        a_average_evaluations=match.player_a_stats.average_evaluations,
        a_average_mobility_evaluations=match.player_a_stats.average_mobility_evaluations,
        a_average_neural_value_evaluations=match.player_a_stats.average_neural_value_evaluations,
        a_average_neural_value_cache_hits=match.player_a_stats.average_neural_value_cache_hits,
        b_average_depth=match.player_b_stats.average_depth,
        b_average_nodes=match.player_b_stats.average_nodes,
        b_average_main_nodes=match.player_b_stats.average_main_nodes,
        b_average_quiescence_nodes=match.player_b_stats.average_quiescence_nodes,
        b_nps=match.player_b_stats.nodes_per_second,
        b_average_evaluations=match.player_b_stats.average_evaluations,
        b_average_mobility_evaluations=match.player_b_stats.average_mobility_evaluations,
        b_average_neural_value_evaluations=match.player_b_stats.average_neural_value_evaluations,
        b_average_neural_value_cache_hits=match.player_b_stats.average_neural_value_cache_hits,
    )


def run_benchmark(
    depths: list[int],
    game_counts: list[int],
    max_plies: int,
    streaming: bool = False,
    opponent: Player | None = None,
    opening_plies: int = 0,
    movetime_ms: int | None = None,
    use_mobility: bool = True,
    quiescence_depth: int = 6,
    time_check_interval: int = 1024,
    neural_checkpoint: Path | None = None,
    neural_channels: int = 32,
    neural_ordering_mode: str = "root",
    neural_min_depth: int = 2,
    value_checkpoint: Path | None = None,
    evaluation_mode: str = "classical",
    neural_value_weight: float = 0.2,
    neural_value_scale: int = 1000,
    adjudication: AdjudicationConfig | None = None,
) -> list[BenchmarkRow]:
    rows = []
    for depth in depths:
        if streaming:
            rows.extend(
                benchmark_depth_streaming(
                    depth,
                    sorted(game_counts),
                    max_plies,
                    opponent,
                    opening_plies,
                    movetime_ms,
                    use_mobility,
                    quiescence_depth,
                    time_check_interval,
                    neural_checkpoint,
                    neural_channels,
                    neural_ordering_mode,
                    neural_min_depth,
                    value_checkpoint,
                    evaluation_mode,
                    neural_value_weight,
                    neural_value_scale,
                    adjudication,
                )
            )
        else:
            rows.extend(
                benchmark_depth(
                    depth,
                    sorted(game_counts),
                    max_plies,
                    opponent,
                    opening_plies,
                    movetime_ms,
                    use_mobility,
                    quiescence_depth,
                    time_check_interval,
                    neural_checkpoint,
                    neural_channels,
                    neural_ordering_mode,
                    neural_min_depth,
                    value_checkpoint,
                    evaluation_mode,
                    neural_value_weight,
                    neural_value_scale,
                    adjudication,
                )
            )
    return rows


def format_rows(rows: list[BenchmarkRow]) -> str:
    header = (
        "depth | games | wins | losses | draws |  score | score% | avg plies | elapsed | "
        "a dep |  a nodes |   a nps | a main |   a q | a eval | a mob | "
        "a nval | a vhit | b dep |  b nodes |   b nps | b main |   b q | b eval | b mob | b nval | b vhit"
    )
    divider = (
        "------|-------|------|--------|-------|--------|--------|-----------|---------|"
        "-------|----------|---------|--------|-------|--------|-------|"
        "--------|--------|-------|----------|---------|--------|-------|--------|-------|--------|------"
    )
    return "\n".join([header, divider, *(row.format() for row in rows)])


def save_rows_csv(rows: list[BenchmarkRow], path: str | Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row_to_dict(row))


def append_row_csv(row: BenchmarkRow, path: str | Path) -> None:
    csv_path = Path(path)
    should_write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        if should_write_header:
            writer.writeheader()
        writer.writerow(row_to_dict(row))


def row_to_dict(row: BenchmarkRow) -> dict[str, float | int]:
    return {fieldname: getattr(row, fieldname) for fieldname in CSV_FIELDNAMES}


def run_benchmark_streaming(
    depths: list[int],
    game_counts: list[int],
    max_plies: int,
    opponent: Player | None = None,
    opening_plies: int = 0,
    movetime_ms: int | None = None,
    use_mobility: bool = True,
    quiescence_depth: int = 6,
    time_check_interval: int = 1024,
    csv_path: Path | None = None,
    neural_checkpoint: Path | None = None,
    neural_channels: int = 32,
    neural_ordering_mode: str = "root",
    neural_min_depth: int = 2,
    value_checkpoint: Path | None = None,
    evaluation_mode: str = "classical",
    neural_value_weight: float = 0.2,
    neural_value_scale: int = 1000,
    adjudication: AdjudicationConfig | None = None,
) -> list[BenchmarkRow]:
    rows = []
    for depth in depths:
        for row in benchmark_depth_streaming(
            depth,
            sorted(game_counts),
            max_plies,
            opponent,
            opening_plies,
            movetime_ms,
            use_mobility,
            quiescence_depth,
            time_check_interval,
            neural_checkpoint,
            neural_channels,
            neural_ordering_mode,
            neural_min_depth,
            value_checkpoint,
            evaluation_mode,
            neural_value_weight,
            neural_value_scale,
            adjudication,
        ):
            rows.append(row)
            if csv_path is not None:
                append_row_csv(row, csv_path)
            print(row.format(), flush=True)
    return rows


def parse_int_list(values: list[str]) -> list[int]:
    result = []
    for value in values:
        for part in value.split(","):
            if part.strip():
                result.append(max(1, int(part)))
    return result


def search_player_name(
    depth: int,
    movetime_ms: int | None = None,
    use_mobility: bool = True,
    neural_ordering: bool = False,
) -> str:
    suffix = f"-{movetime_ms}ms" if movetime_ms is not None else ""
    mobility_suffix = "" if use_mobility else "-no-mobility"
    neural_suffix = "-neural-order" if neural_ordering else ""
    return f"search-depth-{depth}{suffix}{mobility_suffix}{neural_suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cumulative engine benchmarks.")
    parser.add_argument("--depths", nargs="+", default=["1"], help="Depths, e.g. 1 2 3 or 1,2,3.")
    parser.add_argument(
        "--games-list",
        nargs="+",
        default=[",".join(str(count) for count in DEFAULT_GAME_COUNTS)],
        help="Cumulative game checkpoints.",
    )
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--opponent", choices=["search", "random", "neural"], default="random")
    parser.add_argument("--opponent-depth", type=int, default=1)
    parser.add_argument("--movetime", type=int, help="Per-move time limit for benchmarked search player.")
    parser.add_argument("--opponent-movetime", type=int, help="Per-move time limit for search opponent.")
    parser.add_argument("--no-mobility", action="store_true")
    parser.add_argument("--opponent-no-mobility", action="store_true")
    parser.add_argument("--qdepth", type=int, default=6)
    parser.add_argument("--opponent-qdepth", type=int, default=6)
    parser.add_argument("--time-check-interval", type=int, default=1024)
    parser.add_argument("--neural-checkpoint", type=Path)
    parser.add_argument("--opponent-neural-checkpoint", type=Path)
    parser.add_argument("--neural-channels", type=int, default=32)
    parser.add_argument("--neural-ordering", choices=["root", "depth", "all"], default="root")
    parser.add_argument("--neural-min-depth", type=int, default=2)
    parser.add_argument("--value-checkpoint", type=Path)
    parser.add_argument("--opponent-value-checkpoint", type=Path)
    parser.add_argument("--evaluation-mode", choices=["classical", "neural", "blend"], default="classical")
    parser.add_argument("--neural-value-weight", type=float, default=0.2)
    parser.add_argument("--neural-value-scale", type=int, default=1000)
    parser.add_argument("--opening-plies", type=int, default=0)
    parser.add_argument("--adjudicate", action="store_true")
    parser.add_argument("--adjudicate-eval", type=int, default=500)
    parser.add_argument("--adjudicate-eval-plies", type=int, default=8)
    parser.add_argument("--adjudicate-material", type=int, default=900)
    parser.add_argument("--adjudicate-material-plies", type=int, default=8)
    parser.add_argument("--adjudicate-min-plies", type=int, default=20)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--stream", action="store_true", help="Print and save each checkpoint as it completes.")
    args = parser.parse_args()

    depths = parse_int_list(args.depths)
    game_counts = parse_int_list(args.games_list)
    use_mobility = not args.no_mobility
    opponent = build_player(
        args.opponent,
        max(1, args.opponent_depth),
        args.opponent_movetime,
        not args.opponent_no_mobility,
        max(0, args.opponent_qdepth),
        max(1, args.time_check_interval),
        args.opponent_neural_checkpoint,
        max(1, args.neural_channels),
        args.neural_ordering,
        max(1, args.neural_min_depth),
        args.opponent_value_checkpoint,
        args.evaluation_mode,
        args.neural_value_weight,
        max(1, args.neural_value_scale),
    )
    adjudication = AdjudicationConfig(
        enabled=args.adjudicate,
        eval_threshold=max(1, args.adjudicate_eval),
        eval_plies=max(1, args.adjudicate_eval_plies),
        material_threshold=max(1, args.adjudicate_material),
        material_plies=max(1, args.adjudicate_material_plies),
        min_plies=max(0, args.adjudicate_min_plies),
    )
    if args.stream:
        print(format_rows([]).splitlines()[0])
        print(format_rows([]).splitlines()[1])
        rows = run_benchmark_streaming(
            depths,
            game_counts,
            max(1, args.max_plies),
            opponent,
            max(0, args.opening_plies),
            args.movetime,
            use_mobility,
            max(0, args.qdepth),
            max(1, args.time_check_interval),
            args.csv,
            args.neural_checkpoint,
            max(1, args.neural_channels),
            args.neural_ordering,
            max(1, args.neural_min_depth),
            args.value_checkpoint,
            args.evaluation_mode,
            args.neural_value_weight,
            max(1, args.neural_value_scale),
            adjudication,
        )
    else:
        rows = run_benchmark(
            depths,
            game_counts,
            max(1, args.max_plies),
            opponent=opponent,
            opening_plies=max(0, args.opening_plies),
            movetime_ms=args.movetime,
            use_mobility=use_mobility,
            quiescence_depth=max(0, args.qdepth),
            time_check_interval=max(1, args.time_check_interval),
            neural_checkpoint=args.neural_checkpoint,
            neural_channels=max(1, args.neural_channels),
            neural_ordering_mode=args.neural_ordering,
            neural_min_depth=max(1, args.neural_min_depth),
            value_checkpoint=args.value_checkpoint,
            evaluation_mode=args.evaluation_mode,
            neural_value_weight=args.neural_value_weight,
            neural_value_scale=max(1, args.neural_value_scale),
            adjudication=adjudication,
        )
        if args.csv is not None:
            save_rows_csv(rows, args.csv)
        print(format_rows(rows))


if __name__ == "__main__":
    main()
