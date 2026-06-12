from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import chess

from chess_engine_2.data.pgn import material_value
from chess_engine_2.engine import SearchEngine


@dataclass(frozen=True)
class AnnotationSummary:
    rows: int
    elapsed_seconds: float
    positions_per_second: float


def normalize_classical_score(score_centipawns: int, scale: float = 600.0) -> float:
    return math.tanh(score_centipawns / max(1.0, scale))


def annotate_jsonl(
    input_path: Path,
    output_path: Path,
    evaluator: Callable[[chess.Board], int] | None = None,
    max_samples: int | None = None,
    classical_scale: float = 600.0,
    backfill_progress: bool = False,
) -> AnnotationSummary:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    start = time.perf_counter()

    with input_path.open("r", encoding="utf-8") as source:
        with output_path.open("w", encoding="utf-8", newline="\n") as output:
            samples = (json.loads(line) for line in source if line.strip())
            if backfill_progress:
                samples = iter_samples_with_progress(samples)
            for sample in samples:
                board = chess.Board(sample["fen"])
                sample["material_value"] = material_value(board)
                if evaluator is not None:
                    sample["classical_value"] = normalize_classical_score(evaluator(board), classical_scale)
                output.write(json.dumps(sample, separators=(",", ":")))
                output.write("\n")
                rows += 1
                if rows % 1000 == 0:
                    elapsed = time.perf_counter() - start
                    print(f"annotated: {rows} ({rows / elapsed:.1f} positions/s)", flush=True)
                if max_samples is not None and rows >= max_samples:
                    break

    elapsed = time.perf_counter() - start
    return AnnotationSummary(rows, elapsed, rows / elapsed if elapsed > 0 else 0.0)


def iter_samples_with_progress(samples):
    game = []
    for sample in samples:
        board = chess.Board(sample["fen"])
        if board.ply() == 0 and game:
            yield from add_progress_to_game(game)
            game = []
        game.append(sample)
    if game:
        yield from add_progress_to_game(game)


def add_progress_to_game(samples: list[dict]):
    game_plies = len(samples)
    for ply, sample in enumerate(samples, start=1):
        sample["ply"] = ply
        sample["game_plies"] = game_plies
        sample["discounted_value"] = float(sample["value"]) * math.sqrt(ply / game_plies)
        yield sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Add dense material and classical value targets to JSONL.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--classical-scale", type=float, default=600.0)
    parser.add_argument("--no-mobility", action="store_true")
    parser.add_argument("--qdepth", type=int, default=2)
    parser.add_argument("--material-only", action="store_true")
    parser.add_argument("--backfill-progress", action="store_true")
    args = parser.parse_args()

    evaluate_position = None
    if not args.material_only:
        engine = SearchEngine(
            max_depth=max(1, args.depth),
            max_quiescence_depth=max(0, args.qdepth),
            use_mobility=not args.no_mobility,
        )

        def evaluate_position(board: chess.Board) -> int:
            return engine.search(board, max(1, args.depth)).score

    summary = annotate_jsonl(
        args.input,
        args.output,
        evaluate_position,
        args.max_samples,
        args.classical_scale,
        args.backfill_progress,
    )
    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"rows: {summary.rows}")
    print(f"elapsed: {summary.elapsed_seconds:.2f}s")
    print(f"positions/s: {summary.positions_per_second:.2f}")


if __name__ == "__main__":
    main()
