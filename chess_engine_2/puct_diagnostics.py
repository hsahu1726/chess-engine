from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import chess

from chess_engine_2.match import AdjudicationConfig
from chess_engine_2.mcts import MCTSEngine, NeuralPolicyValue
from chess_engine_2.mcts_benchmark import run_scaling_study


DEFAULT_FENS = (
    chess.STARTING_FEN,
    "r1bq1rk1/ppp2ppp/2np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 8",
    "2r2rk1/1bqnbppp/p2ppn2/1p6/3NP3/1BN1B3/PPPQ1PPP/2RR2K1 w - - 2 14",
)


@dataclass(frozen=True)
class DiagnosticRow:
    cpuct: float
    simulations: int
    fen: str
    selected_move: str | None
    root_value: float
    leaf_value_mean: float
    leaf_value_stddev: float
    leaf_value_min: float
    leaf_value_max: float
    top_moves: list[dict[str, float | int | str]]


def parse_numbers(values: list[str], converter):
    numbers = []
    for value in values:
        for part in value.split(","):
            if part.strip():
                numbers.append(converter(part))
    return numbers


def collect_diagnostics(
    checkpoint: Path,
    cpuct_values: list[float],
    simulation_counts: list[int],
    channels: int = 32,
    cache_size: int = 100_000,
    fens: tuple[str, ...] = DEFAULT_FENS,
    top_n: int = 5,
) -> list[DiagnosticRow]:
    evaluator = NeuralPolicyValue.from_checkpoint(checkpoint, channels=channels, cache_size=cache_size)
    rows = []
    for cpuct in cpuct_values:
        for simulations in simulation_counts:
            engine = MCTSEngine(evaluator, simulations=simulations, cpuct=cpuct)
            for fen in fens:
                result = engine.search(chess.Board(fen))
                rows.append(
                    DiagnosticRow(
                        cpuct=cpuct,
                        simulations=simulations,
                        fen=fen,
                        selected_move=result.move.uci() if result.move is not None else None,
                        root_value=result.root_value,
                        leaf_value_mean=result.leaf_value_mean,
                        leaf_value_stddev=result.leaf_value_stddev,
                        leaf_value_min=result.leaf_value_min,
                        leaf_value_max=result.leaf_value_max,
                        top_moves=[asdict(stat) for stat in result.root_moves[:top_n]],
                    )
                )
    return rows


def run_sweep(
    checkpoint: Path,
    cpuct_values: list[float],
    simulation_counts: list[int],
    games: int,
    opponent_depth: int,
    channels: int,
    max_plies: int,
    opening_plies: int,
    qdepth: int,
    seed: int,
    cache_size: int,
    adjudication: AdjudicationConfig,
):
    rows = []
    for cpuct in cpuct_values:
        started = time.perf_counter()
        matches = run_scaling_study(
            checkpoint=checkpoint,
            simulation_counts=simulation_counts,
            games=games,
            opponent_depth=opponent_depth,
            channels=channels,
            cpuct=cpuct,
            max_plies=max_plies,
            opening_plies=opening_plies,
            quiescence_depth=qdepth,
            seed=seed,
            cache_size=cache_size,
            adjudication=adjudication,
        )
        for match in matches:
            row = asdict(match)
            row["cpuct"] = cpuct
            rows.append(row)
        print(f"cpuct {cpuct:g}: completed in {time.perf_counter() - started:.1f}s", flush=True)
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["terminations"] = json.dumps(output["terminations"], sort_keys=True)
            writer.writerow(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure how PUCT parameters use policy and value signals.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--cpuct", nargs="+", default=["0.25", "0.5", "1.0", "2.0", "4.0"])
    parser.add_argument("--simulations", nargs="+", default=["64", "256"])
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--opponent-depth", type=int, default=1)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--qdepth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cache-size", type=int, default=100_000)
    parser.add_argument("--top-moves", type=int, default=5)
    parser.add_argument("--csv", type=Path, default=Path("benchmark_puct_sensitivity.csv"))
    parser.add_argument("--json", type=Path, default=Path("benchmark_puct_sensitivity.json"))
    args = parser.parse_args()

    cpuct_values = [max(0.001, value) for value in parse_numbers(args.cpuct, float)]
    simulation_counts = [max(1, value) for value in parse_numbers(args.simulations, int)]
    adjudication = AdjudicationConfig(enabled=True)
    diagnostics = collect_diagnostics(
        args.checkpoint,
        cpuct_values,
        simulation_counts,
        max(1, args.channels),
        max(0, args.cache_size),
        top_n=max(1, args.top_moves),
    )
    matches = run_sweep(
        args.checkpoint,
        cpuct_values,
        simulation_counts,
        max(1, args.games),
        max(1, args.opponent_depth),
        max(1, args.channels),
        max(1, args.max_plies),
        max(0, args.opening_plies),
        max(0, args.qdepth),
        args.seed,
        max(0, args.cache_size),
        adjudication,
    )

    save_csv(matches, args.csv)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(
            {
                "experiment": "PUCT sensitivity and root diagnostics",
                "checkpoint": str(args.checkpoint),
                "cpuct_values": cpuct_values,
                "simulation_counts": simulation_counts,
                "games_per_configuration": max(1, args.games),
                "matches": matches,
                "diagnostics": [asdict(row) for row in diagnostics],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for row in matches:
        print(
            f"cpuct={row['cpuct']:g} simulations={row['simulations']} "
            f"score={row['score_percent']:.1f}% "
            f"({row['wins']}W {row['losses']}L {row['draws']}D)"
        )
    print(f"csv: {args.csv}")
    print(f"json: {args.json}")


if __name__ == "__main__":
    main()
