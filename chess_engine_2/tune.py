from __future__ import annotations

import argparse
from dataclasses import dataclass

from chess_engine_2.benchmark import BenchmarkRow, benchmark_depth_streaming, format_rows, parse_int_list
from chess_engine_2.match import SearchPlayer


@dataclass
class QDepthTuningResult:
    qdepth: int
    row: BenchmarkRow

    @property
    def sort_key(self) -> tuple[float, float, float]:
        return (
            self.row.score_percent,
            self.row.a_average_depth,
            -self.row.a_average_quiescence_nodes,
        )


def tune_qdepth(
    qdepths: list[int],
    depth: int,
    opponent_depth: int,
    games: int,
    movetime_ms: int,
    max_plies: int,
    opening_plies: int,
) -> list[QDepthTuningResult]:
    results = []
    for qdepth in qdepths:
        opponent = SearchPlayer(
            f"search-depth-{opponent_depth}-{movetime_ms}ms-q{qdepth}",
            opponent_depth,
            movetime_ms,
            True,
            qdepth,
        )
        rows = benchmark_depth_streaming(
            depth=depth,
            game_counts=[games],
            max_plies=max_plies,
            opponent=opponent,
            opening_plies=opening_plies,
            movetime_ms=movetime_ms,
            use_mobility=True,
            quiescence_depth=qdepth,
        )
        results.append(QDepthTuningResult(qdepth, rows[-1]))

    return results


def best_qdepth(results: list[QDepthTuningResult]) -> QDepthTuningResult:
    return max(results, key=lambda result: result.sort_key)


def format_qdepth_results(results: list[QDepthTuningResult]) -> str:
    rows = [result.row for result in results]
    best = best_qdepth(results)
    return "\n".join(
        [
            format_rows(rows),
            "",
            (
                f"recommended qdepth: {best.qdepth} "
                f"(score {best.row.score_percent:.1f}%, "
                f"avg depth {best.row.a_average_depth:.2f}, "
                f"avg q nodes {best.row.a_average_quiescence_nodes:.0f})"
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune engine parameters with short benchmark sweeps.")
    parser.add_argument("--qdepths", nargs="+", default=["2,4,6"])
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--opponent-depth", type=int, default=2)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--movetime", type=int, default=500)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--opening-plies", type=int, default=4)
    args = parser.parse_args()

    results = tune_qdepth(
        qdepths=parse_int_list(args.qdepths),
        depth=max(1, args.depth),
        opponent_depth=max(1, args.opponent_depth),
        games=max(1, args.games),
        movetime_ms=max(1, args.movetime),
        max_plies=max(1, args.max_plies),
        opening_plies=max(0, args.opening_plies),
    )
    print(format_qdepth_results(results))


if __name__ == "__main__":
    main()

