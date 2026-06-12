from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import time
from pathlib import Path

import chess

from chess_engine_2.engine import SearchEngine
from chess_engine_2.value_ranking import sample_positions, search_child_scores


def normalized_child_target(parent_score_cp: int, scale: float = 600.0) -> float:
    return -math.tanh(parent_score_cp / max(1.0, scale))


def distill_board(task: tuple[int, str, int, int, float, bool]) -> dict:
    group_id, fen, child_depth, qdepth, classical_scale, use_mobility = task
    board = chess.Board(fen)
    engine = SearchEngine(
        max_depth=child_depth,
        max_quiescence_depth=qdepth,
        use_mobility=use_mobility,
    )
    moves = list(board.legal_moves)
    scores = search_child_scores(engine, board, moves, child_depth)
    children = []
    for move, score in zip(moves, scores):
        child = board.copy(stack=False)
        child.push(move)
        children.append(
            {
                "move_uci": move.uci(),
                "child_fen": child.fen(),
                "search_score": score,
                "child_value_target": normalized_child_target(score, classical_scale),
            }
        )
    return {
        "group_id": group_id,
        "root_fen": fen,
        "child_depth": child_depth,
        "children": children,
    }


def load_completed_groups(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    completed = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                group = json.loads(line)
                completed[group["root_fen"]] = group
    return completed


def generate_ranking_groups(
    source: Path,
    output: Path,
    positions: int,
    child_depth: int,
    qdepth: int,
    seed: int,
    min_ply: int,
    max_ply: int,
    classical_scale: float = 600.0,
    use_mobility: bool = True,
    workers: int = 1,
    resume: bool = False,
    seed_from: Path | None = None,
) -> dict[str, float | int]:
    boards = sample_positions(source, positions, seed, min_ply, max_ply)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_groups(output) if resume else {}
    if seed_from is not None:
        completed.update(load_completed_groups(seed_from))
    started = time.perf_counter()
    ordered_groups = {}
    tasks = []
    for group_id, board in enumerate(boards):
        cached = completed.get(board.fen())
        if cached is not None and cached.get("child_depth") == child_depth:
            cached["group_id"] = group_id
            ordered_groups[group_id] = cached
        else:
            tasks.append((group_id, board.fen(), child_depth, qdepth, classical_scale, use_mobility))

    mode = "a" if resume and output.exists() else "w"
    with output.open(mode, encoding="utf-8", newline="\n") as stream:
        if mode == "w":
            for group_id in sorted(ordered_groups):
                stream.write(json.dumps(ordered_groups[group_id], separators=(",", ":")) + "\n")
        executor = (
            concurrent.futures.ProcessPoolExecutor(max_workers=workers)
            if workers > 1
            else None
        )
        try:
            results = executor.map(distill_board, tasks) if executor is not None else map(distill_board, tasks)
            for group in results:
                ordered_groups[group["group_id"]] = group
                stream.write(json.dumps(group, separators=(",", ":")) + "\n")
                stream.flush()
                child_count = sum(len(item["children"]) for item in ordered_groups.values())
                elapsed = time.perf_counter() - started
                print(
                    f"{len(ordered_groups)}/{len(boards)} groups, {child_count} children "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )
        finally:
            if executor is not None:
                executor.shutdown()

    # Parallel completion order can differ, so normalize the final file order.
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for group_id in sorted(ordered_groups):
            stream.write(json.dumps(ordered_groups[group_id], separators=(",", ":")) + "\n")

    child_count = sum(len(group["children"]) for group in ordered_groups.values())
    elapsed = time.perf_counter() - started
    return {
        "groups": len(boards),
        "children": child_count,
        "elapsed_seconds": elapsed,
        "children_per_second": child_count / elapsed if elapsed else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sibling move rankings from alpha-beta search.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--positions", type=int, default=25)
    parser.add_argument("--child-depth", type=int, default=3)
    parser.add_argument("--qdepth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--min-ply", type=int, default=12)
    parser.add_argument("--max-ply", type=int, default=60)
    parser.add_argument("--classical-scale", type=float, default=600.0)
    parser.add_argument("--no-mobility", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed-from", type=Path)
    args = parser.parse_args()

    summary = generate_ranking_groups(
        args.source,
        args.output,
        max(1, args.positions),
        max(1, args.child_depth),
        max(0, args.qdepth),
        args.seed,
        max(0, args.min_ply),
        max(args.min_ply, args.max_ply),
        max(1.0, args.classical_scale),
        not args.no_mobility,
        max(1, args.workers),
        args.resume,
        args.seed_from,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
