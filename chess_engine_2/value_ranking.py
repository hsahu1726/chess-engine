from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import torch

from chess_engine_2.data.dataset import board_to_planes
from chess_engine_2.engine import SearchEngine
from chess_engine_2.neural import PolicyValueNet, load_checkpoint


@dataclass(frozen=True)
class MoveRanking:
    move_uci: str
    search_score: int
    neural_value: float
    search_rank: float
    neural_rank: float


@dataclass(frozen=True)
class PositionRanking:
    fen: str
    legal_moves: int
    spearman: float
    pairwise_accuracy: float
    top1_correct: bool
    top3_correct: bool
    neural_top1_regret_cp: int
    neural_top3_regret_cp: int
    search_score_spread: int
    neural_value_spread: float
    moves: list[MoveRanking]


def average_ranks(values: list[float], descending: bool = True) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=descending)
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = average_rank
        position = end
    return ranks


def pearson_correlation(first: list[float], second: list[float]) -> float:
    if len(first) < 2:
        return 0.0
    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    numerator = sum((a - first_mean) * (b - second_mean) for a, b in zip(first, second))
    first_scale = math.sqrt(sum((value - first_mean) ** 2 for value in first))
    second_scale = math.sqrt(sum((value - second_mean) ** 2 for value in second))
    denominator = first_scale * second_scale
    return numerator / denominator if denominator else 0.0


def spearman_correlation(first: list[float], second: list[float]) -> float:
    return pearson_correlation(average_ranks(first), average_ranks(second))


def pairwise_ordering_accuracy(truth: list[float], predictions: list[float]) -> float:
    correct = 0.0
    comparisons = 0
    for first in range(len(truth)):
        for second in range(first + 1, len(truth)):
            truth_difference = truth[first] - truth[second]
            if truth_difference == 0:
                continue
            prediction_difference = predictions[first] - predictions[second]
            comparisons += 1
            if prediction_difference == 0:
                correct += 0.5
            elif truth_difference * prediction_difference > 0:
                correct += 1.0
    return correct / comparisons if comparisons else 0.0


def neural_child_values(
    model: PolicyValueNet,
    device: torch.device,
    board: chess.Board,
    moves: list[chess.Move],
) -> list[float]:
    child_boards = []
    for move in moves:
        child = board.copy(stack=False)
        child.push(move)
        child_boards.append(child)
    planes = torch.tensor(
        [board_to_planes(child) for child in child_boards],
        dtype=torch.float32,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        _, child_values = model(planes)
    return [-float(value) for value in child_values.detach().cpu().tolist()]


def search_child_scores(
    engine: SearchEngine,
    board: chess.Board,
    moves: list[chess.Move],
    child_depth: int,
) -> list[int]:
    scores = []
    for move in moves:
        child = board.copy(stack=False)
        child.push(move)
        scores.append(-engine.search(child, child_depth).score)
    return scores


def rank_position(
    board: chess.Board,
    model: PolicyValueNet,
    device: torch.device,
    engine: SearchEngine,
    child_depth: int,
) -> PositionRanking:
    moves = list(board.legal_moves)
    search_scores = search_child_scores(engine, board, moves, child_depth)
    neural_values = neural_child_values(model, device, board, moves)
    search_ranks = average_ranks(search_scores)
    neural_ranks = average_ranks(neural_values)
    best_search_score = max(search_scores)
    best_search_indices = {index for index, score in enumerate(search_scores) if score == best_search_score}
    neural_order = sorted(range(len(moves)), key=neural_values.__getitem__, reverse=True)
    move_rows = [
        MoveRanking(move.uci(), search_scores[index], neural_values[index], search_ranks[index], neural_ranks[index])
        for index, move in enumerate(moves)
    ]
    move_rows.sort(key=lambda row: row.search_score, reverse=True)
    return PositionRanking(
        fen=board.fen(),
        legal_moves=len(moves),
        spearman=spearman_correlation(search_scores, neural_values),
        pairwise_accuracy=pairwise_ordering_accuracy(search_scores, neural_values),
        top1_correct=neural_order[0] in best_search_indices,
        top3_correct=bool(best_search_indices.intersection(neural_order[:3])),
        neural_top1_regret_cp=best_search_score - search_scores[neural_order[0]],
        neural_top3_regret_cp=best_search_score - max(search_scores[index] for index in neural_order[:3]),
        search_score_spread=max(search_scores) - min(search_scores),
        neural_value_spread=max(neural_values) - min(neural_values),
        moves=move_rows,
    )


def sample_positions(
    jsonl: Path,
    count: int,
    seed: int,
    min_ply: int = 8,
    max_ply: int = 80,
) -> list[chess.Board]:
    candidates = []
    with jsonl.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            sample = json.loads(line)
            board = chess.Board(sample["fen"])
            if min_ply <= board.ply() <= max_ply and not board.is_game_over(claim_draw=True):
                candidates.append(board)
    random.Random(seed).shuffle(candidates)
    unique = []
    seen = set()
    for board in candidates:
        key = board._transposition_key()
        if key in seen:
            continue
        seen.add(key)
        unique.append(board)
        if len(unique) >= count:
            break
    return unique


def summarize(rows: list[PositionRanking]) -> dict[str, float | int]:
    count = len(rows)
    return {
        "positions": count,
        "moves": sum(row.legal_moves for row in rows),
        "mean_spearman": sum(row.spearman for row in rows) / count if count else 0.0,
        "mean_pairwise_accuracy": sum(row.pairwise_accuracy for row in rows) / count if count else 0.0,
        "top1_accuracy": sum(row.top1_correct for row in rows) / count if count else 0.0,
        "top3_accuracy": sum(row.top3_correct for row in rows) / count if count else 0.0,
        "mean_top1_regret_cp": sum(row.neural_top1_regret_cp for row in rows) / count if count else 0.0,
        "mean_top3_regret_cp": sum(row.neural_top3_regret_cp for row in rows) / count if count else 0.0,
        "mean_search_score_spread": sum(row.search_score_spread for row in rows) / count if count else 0.0,
        "mean_neural_value_spread": sum(row.neural_value_spread for row in rows) / count if count else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare neural move rankings with deeper alpha-beta rankings.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--positions", type=int, default=25)
    parser.add_argument("--child-depth", type=int, default=4)
    parser.add_argument("--qdepth", type=int, default=2)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--min-ply", type=int, default=8)
    parser.add_argument("--max-ply", type=int, default=80)
    parser.add_argument("--no-mobility", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmark_value_ranking.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyValueNet(channels=max(1, args.channels)).to(device)
    load_checkpoint(args.checkpoint, model, device)
    engine = SearchEngine(
        max_depth=max(1, args.child_depth),
        max_quiescence_depth=max(0, args.qdepth),
        use_mobility=not args.no_mobility,
    )
    boards = sample_positions(
        args.jsonl,
        max(1, args.positions),
        args.seed,
        max(0, args.min_ply),
        max(args.min_ply, args.max_ply),
    )
    started = time.perf_counter()
    rows = []
    for index, board in enumerate(boards, start=1):
        row = rank_position(board, model, device, engine, max(1, args.child_depth))
        rows.append(row)
        print(
            f"{index}/{len(boards)} moves={row.legal_moves} spearman={row.spearman:.3f} "
            f"pairwise={row.pairwise_accuracy:.3f} top1={int(row.top1_correct)} top3={int(row.top3_correct)}",
            flush=True,
        )

    summary = summarize(rows)
    result = {
        "experiment": "Neural value move-ranking study",
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.jsonl),
        "device": str(device),
        "child_depth": max(1, args.child_depth),
        "quiescence_depth": max(0, args.qdepth),
        "seed": args.seed,
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary,
        "positions": [asdict(row) for row in rows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
